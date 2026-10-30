import csv
import os
import subprocess
import sys

import numpy as np
from osgeo import gdal

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS)

import remove_duplicate_scenes as dedup  # noqa: E402
from psnorm import io, sensors  # noqa: E402


def _raster(path, bands):
    ds = gdal.GetDriverByName("GTiff").Create(str(path), 4, 4, len(bands), gdal.GDT_UInt16)
    for i, name in enumerate(bands, start=1):
        ds.GetRasterBand(i).SetDescription(name)
        ds.GetRasterBand(i).WriteArray(np.ones((4, 4), dtype=np.uint16))
    ds = None


def _scene(folder, scene_id, n_bands):
    folder.mkdir(parents=True, exist_ok=True)
    suffix = io.EIGHT_BAND_SUFFIX if n_bands == 8 else io.FOUR_BAND_SUFFIX
    path = folder / f"{scene_id}{suffix}"
    _raster(path, sensors.BAND_PROFILES[n_bands])
    for sidecar in ("_3B_udm2_clip.tif", "_metadata.json", "_3B_AnalyticMS_metadata_clip.xml"):
        (folder / f"{scene_id}{sidecar}").write_text("x")
    return str(path)


def test_the_4band_copy_of_a_scene_also_in_8band_is_removed_with_its_outputs(tmp_path):
    raw = tmp_path / "raw"
    dup = "20200502_150055_00_2271"
    four = _scene(raw / "p4b" / "p4b_2020_122_152", dup, 4)
    eight = _scene(raw / "p8b" / "p8b_2020_122_152", dup, 8)
    other = _scene(raw / "p4b" / "p4b_2020_122_152", "20200502_150055_00_22710", 4)  # ID a prefix-extension of dup's
    same = [_scene(raw / f"p4b_{i}", "20200503_000000_00_aaaa", 4) for i in range(2)]

    registered = tmp_path / "affine"
    (registered / "alignment_metrics").mkdir(parents=True)
    out_four = registered / os.path.basename(four)
    out_four.write_text("registered")
    with open(registered / "registration_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "analytic_path", "output_path"])
        w.writerow([dup, four, str(out_four)])
        w.writerow(["20200502_150055_00_22710", other, ""])
    with open(registered / "alignment_metrics" / "registration_metrics_4band.csv", "w", newline="") as f:
        csv.writer(f).writerows([["moving_filename", "x"], [os.path.basename(four), 1], [os.path.basename(other), 2]])

    removals, unresolved = dedup.plan_removals(dedup.find_analytic_files(str(raw)))
    assert removals == [(dup, four, eight)]
    assert [u[0] for u in unresolved] == ["20200503_000000_00_aaaa"]

    subprocess.run([sys.executable, os.path.join(SCRIPTS, "remove_duplicate_scenes.py"), str(raw),
                    "--registered-dir", str(registered), "--apply"], check=True)

    assert sorted(os.listdir(raw / "p4b" / "p4b_2020_122_152")) == sorted(
        os.path.basename(p) for p in dedup.scene_files("20200502_150055_00_22710", other))
    assert len(os.listdir(raw / "p8b" / "p8b_2020_122_152")) == 4 and all(os.path.exists(p) for p in same)
    assert not out_four.exists()
    with open(registered / "registration_manifest.csv", newline="") as f:
        assert [r["scene_id"] for r in csv.DictReader(f)] == ["20200502_150055_00_22710"]
    with open(registered / "alignment_metrics" / "registration_metrics_4band.csv", newline="") as f:
        assert [r["moving_filename"] for r in csv.DictReader(f)] == [os.path.basename(other)]
    with open(raw / "removed_duplicate_scenes.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == 5  # 4 raw files + 1 registered output
