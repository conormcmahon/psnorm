"""Parquet output for scripts/sample_tree_points.py (--format parquet).

Samples are taken a scene at a time (every point on the scene at once), but
the dataset is organized by point -- partition folders such as
genus=Acer/treeid=5468701/ (--partition-by) -- so it is written in two
steps:

1. Staging (sample_to_staging; one process per shard). Each scene's samples
   go into in-memory buffers, one per *bucket* of points (points ordered by
   the partition fields and split into --buckets groups). A full buffer
   set (--flush-rows rows) is written as one small file per bucket,
   <staging>/points/bucket=NNNN/<flush id>.parquet, holding only the
   per-point values (point index, scene, band, reflectances, UDM2 flag,
   pixel position). Fields that are the same for every point of a scene
   and band (registration and normalization fields) are written once per
   scene and band, to <staging>/scenes/<flush id>.parquet. Only after all
   of a flush's files are written is the flush recorded in
   <staging>/done/<shard>.txt, with its scenes; a rerun skips the recorded
   scenes and the finalize step ignores files of unrecorded flushes, so an
   interrupted run never yields duplicate rows.

2. Finalizing (finalize; --workers processes, one bucket each at a time).
   A bucket's staged rows are joined with the point fields and the scene and
   band fields, sorted by the partition fields, acquisition time and band,
   and written as one file per partition, <output>/<field>=<value>/...
   /part-0.parquet (the partition fields are in the folder names only, the
   standard hive layout: reading the dataset, or any part of it, with
   pandas or pyarrow adds them back as columns). Finished buckets
   are recorded in <staging>/finalized/, so a rerun continues where it left
   off.

The columns and their values are those of the CSV output, with typed
columns (dates, integers, floats; empty CSV values are nulls) and year and
month columns added after date.

--layout slim keeps the fields that are the same for every point of a scene
and band (registration and normalization fields) out of the point files:
these keep only the per-point values (TREE_COLUMNS: time, scene, band, the
reflectances, the UDM2 flag, the pixel position and selected_reflectance),
and the scene-and-band fields are written once, to
<output>/_scene_bands.parquet (one row per scene_id and band; the leading
underscore keeps it out of a dataset read of the point folders). Join the
two on scene_id and band to get the full rows. The full and slim layouts can
be finalized from the same staging, to different --output folders.
"""
from __future__ import annotations

import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import sample_tree_points as stp

COMPRESSION = "zstd"

# Column types (columns not listed are float64; point fields other than the
# id are strings).
STRING_COLUMNS = {"scene_id", "product", "band", "selected_reflectance", "status_affine", "status_tps",
                  "norm_composite", "norm_fallback_reason"}
INT_COLUMNS = {"band_index": pa.int8(), "udm2_clear": pa.int8(), "pixel_col": pa.int32(), "pixel_row": pa.int32(),
               "ransac_inliers_affine": pa.int32(), "ransac_inliers_tps": pa.int32(),
               "norm_consensus_pixels_unmasked": pa.int32(), "norm_consensus_pixels_fit": pa.int32(),
               "norm_ransac_inliers": pa.int32(), "norm_identity_fallback": pa.int8(),
               "year": pa.int16(), "month": pa.int8()}
REFLECTANCE_COLUMNS = ("reflectance_original", "reflectance_affine", "reflectance_tps")
SCENE_COLUMNS = ("date", "acquired", "year", "month", "scene_id", "product")
# The per-point columns a --layout slim point file keeps (besides the point
# fields that are not partition folders).
TREE_COLUMNS = ["date", "year", "month", "acquired", "scene_id", "band", "band_index",
                "reflectance_original", "reflectance_affine", "reflectance_tps", "selected_reflectance",
                "udm2_clear", "pixel_col", "pixel_row"]
SCENE_TABLE_NAME = "_scene_bands.parquet"


def staging_dir(args) -> str:
    return args.staging_dir or os.path.join(args.output, "_staging")


def partition_fields(args) -> list[str]:
    fields = [f.strip() for f in (args.partition_by or args.id_field).split(",") if f.strip()]
    return fields


def point_order(points, args, extra_fields) -> np.ndarray:
    """Rank of every point when ordered by its partition fields (the point
    fields among --partition-by) and then its id: buckets are contiguous
    runs of this order, so every partition falls in one bucket."""
    keys = [f for f in partition_fields(args) if f == args.id_field or f in extra_fields]
    if args.id_field not in keys:
        keys.append(args.id_field)

    def key(k):
        pid, _, _, extra = points[k]
        out = []
        for f in keys:
            v = pid if f == args.id_field else extra.get(f)
            v = "" if v is None else v
            # numeric ids sort numerically
            out.append((0, float(v), "") if _is_number(v) else (1, 0.0, str(v)))
        return out
    order = sorted(range(len(points)), key=key)
    rank = np.empty(len(points), dtype=np.int64)
    rank[np.asarray(order, dtype=np.int64)] = np.arange(len(points))
    return rank


def _is_number(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _typed(name: str, values: list) -> pa.Array:
    """A column of CSV-style values ("" = missing) as its parquet type."""
    if name in STRING_COLUMNS:
        return pa.array([None if v in ("", None) else str(v) for v in values], type=pa.string())
    if name in INT_COLUMNS:
        return pa.array([None if v in ("", None) else int(float(v)) for v in values], type=INT_COLUMNS[name])
    return pa.array([None if v in ("", None) else float(v) for v in values], type=pa.float64())


def scene_band_table(sample) -> pa.Table:
    """One row per band of the scene with every field that is the same for
    all of its points."""
    n = len(sample.band_names)
    sf = sample.scene_fields
    columns = {
        "date": pa.array(np.full(n, np.datetime64(sf["date"], "D")), type=pa.date32()),
        "acquired": pa.array(np.full(n, np.datetime64(sf["acquired"], "s")), type=pa.timestamp("s")),
        "year": pa.array([int(sf["date"][:4])] * n, type=pa.int16()),
        "month": pa.array([int(sf["date"][5:7])] * n, type=pa.int8()),
    }
    for name in ("scene_id", "product"):
        columns[name] = _typed(name, [sf[name]] * n)
    columns["band"] = _typed("band", sample.band_names)
    columns["band_index"] = pa.array(np.arange(1, n + 1), type=pa.int8())
    columns["selected_reflectance"] = _typed("selected_reflectance", [sf["selected_reflectance"]] * n)
    for name in stp.COLUMNS:
        if name in sf and name not in columns:
            columns[name] = _typed(name, [sf[name]] * n)
        elif name in sample.band_metrics[0] and name not in columns:
            columns[name] = _typed(name, [m[name] for m in sample.band_metrics])
    for name in stp.NORM_COLUMNS:
        columns[name] = _typed(name, [sample.norm.get(b, {}).get(name, "") for b in sample.band_names])
    return pa.table(columns)


def point_rows_table(sample) -> pa.Table:
    """The per-point values: one row per (point, band)."""
    n, nb = len(sample), len(sample.band_names)
    columns = {
        "point": pa.array(np.repeat(sample.point_index, nb).astype(np.int32)),
        "scene_id": pa.array([sample.scene_fields["scene_id"]] * (n * nb), type=pa.string()),
        "band_index": pa.array(np.tile(np.arange(1, nb + 1), n), type=pa.int8()),
    }
    for col, version in zip(REFLECTANCE_COLUMNS, ("original", "affine", "tps")):
        arr = sample.values[version]
        if arr is None:
            columns[col] = pa.nulls(n * nb, type=pa.float32())
        else:
            flat = (arr[:, :nb] / stp.REFLECTANCE_SCALE).astype(np.float32).ravel()
            columns[col] = pa.array(flat, mask=np.isnan(flat), type=pa.float32())
    if sample.clear is None:
        columns["udm2_clear"] = pa.nulls(n * nb, type=pa.int8())
    else:
        clear = np.repeat(sample.clear, nb)
        columns["udm2_clear"] = pa.array(np.nan_to_num(clear).astype(np.int8), mask=np.isnan(clear), type=pa.int8())
    columns["pixel_col"] = pa.array(np.repeat(sample.cols, nb).astype(np.int32))
    columns["pixel_row"] = pa.array(np.repeat(sample.rows, nb).astype(np.int32))
    return pa.table(columns)


def _done_scenes(staging: str) -> tuple[set, set]:
    """(scene ids, flush ids) recorded as completely staged."""
    scenes, flushes = set(), set()
    for path in glob.glob(os.path.join(staging, "done", "*.txt")):
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    flushes.add(rec["flush"])
                    scenes.update(rec["scenes"])
    return scenes, flushes


def sample_to_staging(ctx, tasks, args, extra_fields) -> None:
    """Sample `tasks` (one shard's scenes) into the staging area; see the
    module docstring."""
    staging = staging_dir(args)
    for sub in ("points", "scenes", "done"):
        os.makedirs(os.path.join(staging, sub), exist_ok=True)
    rank = point_order(ctx.points, args, extra_fields)
    bucket_of = (rank * args.buckets // max(1, len(ctx.points))).astype(np.int32)
    with open(os.path.join(staging, "settings.json"), "w") as f:
        json.dump({"points": os.path.abspath(args.points), "n_points": len(ctx.points), "buckets": args.buckets,
                   "id_field": args.id_field, "extra_fields": extra_fields,
                   "partition_by": partition_fields(args)}, f)
    done, _ = _done_scenes(staging)
    todo = [t for t in tasks if t["scene_id"] not in done]
    shard = (args.shard or "1/1").replace("/", "of")
    print(f"staging: {len(tasks) - len(todo)} of {len(tasks)} scenes already staged", flush=True)

    buffers: dict[int, list] = {}
    scene_tables, scene_ids = [], []
    n_buffered = n_rows = n_flush = 0
    start = time.time()
    timings = {"sample": 0.0, "arrange": 0.0, "write": 0.0}

    def flush():
        nonlocal n_buffered, n_flush, buffers, scene_tables, scene_ids
        if not scene_ids:
            return
        t0 = time.time()
        flush_id = f"{shard}-{os.getpid()}-{int(time.time() * 1000)}-{n_flush}"
        for b, parts in buffers.items():
            bucket_dir = os.path.join(staging, "points", f"bucket={b:05d}")
            os.makedirs(bucket_dir, exist_ok=True)
            pq.write_table(pa.concat_tables(parts), os.path.join(bucket_dir, f"{flush_id}.parquet"),
                           compression=COMPRESSION)
        pq.write_table(pa.concat_tables(scene_tables, promote_options="permissive"),
                       os.path.join(staging, "scenes", f"{flush_id}.parquet"), compression=COMPRESSION)
        with open(os.path.join(staging, "done", f"{shard}.txt"), "a") as f:
            f.write(json.dumps({"flush": flush_id, "scenes": scene_ids}) + "\n")
        timings["write"] += time.time() - t0
        buffers, scene_tables, scene_ids = {}, [], []
        n_buffered = 0
        n_flush += 1

    for i, task in enumerate(todo, start=1):
        t0 = time.time()
        sample = ctx.sample_scene(task)
        t1 = time.time()
        timings["sample"] += t1 - t0
        if sample is not None:
            table = point_rows_table(sample)
            buckets = bucket_of[table.column("point").to_numpy()]
            order = np.argsort(buckets, kind="stable")
            table = table.take(pa.array(order))
            sorted_buckets = buckets[order]
            edges = np.flatnonzero(np.diff(sorted_buckets)) + 1
            starts = np.concatenate([[0], edges])
            ends = np.concatenate([edges, [len(order)]])
            for s, e in zip(starts, ends):
                buffers.setdefault(int(sorted_buckets[s]), []).append(table.slice(s, e - s))
            scene_tables.append(scene_band_table(sample))
            n_buffered += table.num_rows
            n_rows += table.num_rows
        scene_ids.append(task["scene_id"])
        timings["arrange"] += time.time() - t1
        if n_buffered >= args.flush_rows:
            flush()
        if i % 20 == 0 or i == len(todo):
            print(f"[progress] {i}/{len(todo)} scenes, {n_rows} rows, {time.time() - start:.0f} s "
                  f"(sampling {timings['sample']:.0f} s, arranging {timings['arrange']:.0f} s, "
                  f"writing {timings['write']:.0f} s)", flush=True)
    flush()
    print(f"DONE: {n_rows} rows staged in {staging} ({len(todo)} scenes, {n_flush} flushes; "
          f"sampling {timings['sample']:.0f} s, arranging {timings['arrange']:.0f} s, "
          f"writing {timings['write']:.0f} s)", flush=True)


def _point_table(points, args, extra_fields) -> pa.Table:
    ids = [p[0] for p in points]
    # Integer ids are stored as int32 where they fit: the type a hive-style
    # partition folder (treeid=5468701) is read back as, so a dataset read
    # of the folders and the column kept in each file agree.
    numeric = all(_is_number(v) and float(v).is_integer() for v in ids)
    if numeric:
        values = [int(float(v)) for v in ids]
        id_type = pa.int32() if max(map(abs, values), default=0) < 2**31 else pa.int64()
        columns = {args.id_field: pa.array(values, type=id_type)}
    else:
        columns = {args.id_field: pa.array(ids, type=pa.string())}
    for f in extra_fields:
        columns[f] = pa.array([None if p[3].get(f) in ("", None) else str(p[3].get(f)) for p in points],
                              type=pa.string())
    return pa.table(columns)


def _finalize_bucket(job) -> tuple[int, int, int, float]:
    """Arrange one bucket's staged rows into its partition files; returns
    (bucket, rows, files, seconds).

    Memory stays at about the bucket's narrow staged rows: those are sorted
    (with the point fields and acquisition time attached), and the scene
    and band fields are only added partition by partition, by row number
    into the scene table."""
    bucket, files, scenes_path, points_path, args_d = job
    t0 = time.time()
    staging, output = args_d["staging"], args_d["output"]
    partition_by = args_d["partition_by"]
    if not files:
        return bucket, 0, 0, 0.0
    rows = pa.concat_tables([pq.read_table(p) for p in files])
    points = pq.read_table(points_path)
    scenes = pq.read_table(scenes_path)

    # each row's scene-and-band row number in the scene table
    keys = pa.table({"scene_id": scenes.column("scene_id"), "band_index": scenes.column("band_index"),
                     "_scene_row": pa.array(np.arange(scenes.num_rows, dtype=np.int32))})
    rows = rows.join(keys, keys=["scene_id", "band_index"], join_type="left outer")
    if rows.column("_scene_row").null_count:
        raise ValueError(f"bucket {bucket}: {rows.column('_scene_row').null_count} staged rows have no scene fields")
    scene_row = rows.column("_scene_row").combine_chunks()

    # sort keys: the partition fields (point fields, or scene fields such as
    # year), then acquisition time and band
    point_index = rows.column("point").combine_chunks()
    sort_cols = {}
    for f in partition_by:
        if f in points.column_names:
            sort_cols[f] = points.column(f).take(point_index)
        else:
            sort_cols[f] = scenes.column(f).take(scene_row)
    sort_cols["_acquired"] = scenes.column("acquired").take(scene_row)
    sort_cols["_band_index"] = rows.column("band_index")
    sort_cols["_point"] = point_index
    indices = pc.sort_indices(pa.table(sort_cols), sort_keys=[(c, "ascending") for c in sort_cols])
    rows = rows.take(indices)
    part_keys = pa.table({f: sort_cols[f].take(indices) for f in partition_by}) if partition_by else None
    del sort_cols

    n = rows.num_rows
    if partition_by:
        change = np.zeros(n, dtype=bool)
        change[0] = True
        for f in partition_by:
            col = part_keys.column(f).combine_chunks()
            if n > 1:
                diff = pc.not_equal(col.slice(1), col.slice(0, n - 1))
                diff = pc.fill_null(pc.or_(diff, pc.xor(pc.is_null(col.slice(1)), pc.is_null(col.slice(0, n - 1)))),
                                    True)
                change[1:] |= diff.to_numpy(zero_copy_only=False)
        starts = np.flatnonzero(change)
    else:
        starts = np.array([0])
    ends = np.append(starts[1:], n)

    point_rows = rows.select([c for c in rows.column_names if c not in ("_scene_row",)])
    columns = args_d["columns"]
    n_files = 0
    for s, e in zip(starts, ends):
        chunk = point_rows.slice(s, e - s)
        pidx = chunk.column("point")
        srow = rows.column("_scene_row").slice(s, e - s)
        parts = {}
        for name in points.column_names:
            parts[name] = points.column(name).take(pidx)
        for name in scenes.column_names:
            if name not in ("scene_id", "band_index"):
                parts[name] = scenes.column(name).take(srow)
        for name in chunk.column_names:
            if name != "point":
                parts[name] = chunk.column(name)
        table = pa.table({c: parts[c] for c in columns if c in parts and c not in partition_by})
        if partition_by:
            values = [part_keys.column(f)[int(s)].as_py() for f in partition_by]
            directory = os.path.join(output, *[f"{f}={_partition_value(v)}" for f, v in zip(partition_by, values)])
            os.makedirs(directory, exist_ok=True)
            # The partition fields are in the folder names only (standard
            # hive layout: pandas/pyarrow add them back as columns).
            pq.write_table(table, os.path.join(directory, "part-0.parquet"), compression=COMPRESSION)
        else:
            pq.write_table(table, os.path.join(output, f"part-{bucket:05d}.parquet"), compression=COMPRESSION)
        n_files += 1
    with open(os.path.join(staging, args_d["finalized_dir"], f"bucket={bucket:05d}.done"), "w") as f:
        f.write(f"{n} rows, {n_files} files\n")
    return bucket, n, n_files, time.time() - t0


def _partition_value(value) -> str:
    if value is None:
        return "__NULL__"
    text = str(value)
    if hasattr(value, "isoformat"):
        text = value.isoformat()
    return text.replace("/", "_")


def output_columns(args, extra_fields) -> list[str]:
    """The columns of a point file (before the partition fields are left
    out)."""
    cols = stp.output_columns(args.id_field, extra_fields, True)
    i = cols.index("date") + 1
    cols = cols[:i] + ["year", "month"] + cols[i:]
    if getattr(args, "layout", "full") == "slim":
        cols = [args.id_field, *extra_fields, *TREE_COLUMNS]
    return cols


def scene_table_columns(args, extra_fields) -> list[str]:
    """The columns of the --layout slim scene-and-band table."""
    full = output_columns(type("A", (), {"id_field": args.id_field, "layout": "full"})(), extra_fields)
    point_level = {args.id_field, *extra_fields, "reflectance_original", "reflectance_affine", "reflectance_tps",
                   "udm2_clear", "pixel_col", "pixel_row"}
    cols = [c for c in full if c not in point_level]
    first = ["scene_id", "band", "band_index"]
    return first + [c for c in cols if c not in first]


def finalized_dir(args) -> str:
    """Where finished buckets are recorded, per layout (so the full and the
    slim layout can be finalized from one staging)."""
    return "finalized" if getattr(args, "layout", "full") == "full" else "finalized_slim"


def finalize(args, extra_fields, workers: int = 1, buckets: list[int] | None = None) -> None:
    """Arrange the staged samples into the final partitions; see the module
    docstring."""
    staging = staging_dir(args)
    os.makedirs(os.path.join(staging, finalized_dir(args)), exist_ok=True)
    _, flushes = _done_scenes(staging)

    # the scene-and-band fields of every committed flush, one table
    scene_files = [p for p in glob.glob(os.path.join(staging, "scenes", "*.parquet"))
                   if os.path.basename(p)[:-8] in flushes]
    scenes = pa.concat_tables([pq.read_table(p) for p in scene_files], promote_options="permissive")
    scenes_path = os.path.join(staging, "scenes_all.parquet")
    pq.write_table(scenes, scenes_path)
    points = stp.read_points(args.points, args.id_field, extra_fields)
    points_path = os.path.join(staging, "points_all.parquet")
    pq.write_table(_point_table(points, args, extra_fields), points_path)

    if getattr(args, "layout", "full") == "slim":
        os.makedirs(args.output, exist_ok=True)
        cols = [c for c in scene_table_columns(args, extra_fields) if c in scenes.column_names]
        table = scenes.select(cols).sort_by([("acquired", "ascending"), ("scene_id", "ascending"),
                                             ("band_index", "ascending")])
        pq.write_table(table, os.path.join(args.output, SCENE_TABLE_NAME), compression=COMPRESSION)
        print(f"wrote {table.num_rows} scene bands to {os.path.join(args.output, SCENE_TABLE_NAME)}", flush=True)
    done = {int(os.path.basename(p)[7:12])
            for p in glob.glob(os.path.join(staging, finalized_dir(args), "*.done"))}
    jobs = []
    for bucket_dir in sorted(glob.glob(os.path.join(staging, "points", "bucket=*"))):
        b = int(os.path.basename(bucket_dir)[7:])
        if b in done or (buckets is not None and b not in buckets):
            continue
        files = [p for p in sorted(glob.glob(os.path.join(bucket_dir, "*.parquet")))
                 if os.path.basename(p)[:-8] in flushes]
        args_d = {"staging": staging, "output": args.output, "id_field": args.id_field,
                  "finalized_dir": finalized_dir(args),
                  "partition_by": partition_fields(args), "columns": output_columns(args, extra_fields)}
        jobs.append((b, files, scenes_path, points_path, args_d))
    print(f"finalizing {len(jobs)} buckets ({len(done)} already done), {len(scenes)} scene bands", flush=True)
    start = time.time()
    n_rows = n_files = 0
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = pool.map(_finalize_bucket, jobs)
            for i, (b, r, nf, dt) in enumerate(results, start=1):
                n_rows += r
                n_files += nf
                print(f"[finalize] {i}/{len(jobs)} buckets ({b}: {r} rows, {nf} files, {dt:.0f} s), "
                      f"{time.time() - start:.0f} s", flush=True)
    else:
        for i, job in enumerate(jobs, start=1):
            b, r, nf, dt = _finalize_bucket(job)
            n_rows += r
            n_files += nf
            print(f"[finalize] {i}/{len(jobs)} buckets ({b}: {r} rows, {nf} files, {dt:.0f} s), "
                  f"{time.time() - start:.0f} s", flush=True)
    print(f"DONE: {n_rows} rows in {n_files} files under {args.output}", flush=True)
