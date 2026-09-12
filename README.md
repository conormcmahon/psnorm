# psnorm

Relative radiometric normalization for PlanetScope imagery. Combines elements
from two existing libraries:

[ArrNorm](https://github.com/SMByC/ArrNorm)
- provides IR-MAD approach for detecting invariant targets

[spectralmatch](https://github.com/spectralmatch/spectralmatch)
- provides framework for designing pipeline / scaling over large datasets
- implements 

Also provides a few new tools relative to those approaches:
- Automated selection of consensus invariant targets across large sets of points,
  using a combination of thresholded voting and RANSAC outlier rejection when
  fitting a calibration model (orthogonal least squares).

To do:
- Build a comparison tool to get external validation of reflectance from the EMIT
  imaging spectrometer.
- Continue incremental improvements to scaling design for analyzing large data volumes
- Optional topographic / surface orientation correction
- Georegistration (to incorporate [existing experimental library](https://github.com/conormcmahon/planet_georeg_opencv))

## Motivation for Radiometric Normalization

Comparing PlanetScope scenes over time (e.g. to track phenology, land cover change,
or plant mortality) requires correcting for radiometric drift and errors in surface
reflectance retrieval — resulting from changing sun angle, mis-estimated atmospheric 
composition, or degrading sensor performance — without erasing genuine 
ground change. IR-MAD finds pixels that are *jointly* invariant across all bands 
via canonical correlation + iterative chi-square reweighting, which is more robust 
to real land-cover change leaking into the fit than simpler per-band PCA/threshold
methods.

Invariant-target selection is a three-phase process computed over time series
rather than a single pairwise fit between two images — see "How invariant 
targets are selected" below. Every artifact along the way (cloud/water/vegetation
masks, per-pair candidates, the final consensus set) is written to disk, 
alongside the final gain/offset.

## How invariant targets are selected

1. **Phase A — per-target candidate detection.** Each target scene is
   compared to the reference *only* (never target-vs-target) via IR-MAD,
   restricted to pixels that pass that pair's search mask (nodata, water,
   vegetation, UDM2, and OmniCloudMask excluded — see Masking below).
   Pixels whose no-change probability exceeds `ncp_threshold` (default
   0.70 — user-facing parameter) are flagged as invariant *candidates* for
   that pair and saved as a boolean raster
   (`candidates/{target_id}_candidate.tif`); nothing is regressed yet.
   `ncp_threshold` is deliberately more lenient than the "obvious" choice
   of a strict cutoff (e.g. 0.95): per-pair invariant-pixel selection is a
   threshold on a noisy per-pixel statistic, and a strict per-pair
   threshold keeps only a small, nearly-random "lucky" subset of the true
   invariant population each time (near-zero overlap with the *next*
   pair's lucky subset) — empirically this made Phase B's consensus set
   collapse to almost nothing. A more lenient per-pair threshold feeds
   Phase B a much larger pool to actually vote on, which is what makes
   cross-pair consensus work in practice. See `spectral_coverage_report.csv`
   below for a per-scene, per-band diagnostic of the resulting targets.
   IR-MAD itself fits against **log(DN)**, not raw DN, by default
   (`use_log_transform`, default True — see "Log-transform" below): sensor
   noise and real BRDF/illumination sensitivity both scale roughly with
   signal level, so a raw-DN chi-square test systematically favors dark
   surfaces over equally-stable bright ones purely because bright surfaces
   show larger *absolute* swings for the same *relative* stability.
2. **Phase B — cross-scene consensus.** For every reference pixel, two
   counts are accumulated across all target comparisons: how many times it
   was *evaluated* (passed the search mask in both scenes) and how many of
   those times it was flagged invariant. A pixel only makes the final
   consensus set if `times_invariant / times_evaluated` exceeds
   `invariance_frequency_threshold` (default 0.5, i.e. >50% — user-facing
   parameter) **and** it was evaluated at least `min_observations` times (a
   floor so a pixel seen once can't reach 100% by chance). Masked
   comparisons never enter either count, so cloud/water exclusion in one
   scene doesn't count against a pixel's invariance percentage. Persisted
   as `consensus/{times_evaluated,times_invariant,frequency,consensus_mask}.tif`,
   all on the reference scene's own grid (every candidate is, by
   construction, inside the reference's footprint, so no separate mosaic
   canvas is needed).
3. **Phase C — final fit + apply.** Each target's gain/offset is re-fit
   (per-band orthogonal regression) using *only* the consensus set, then
   applied to the *entire* target scene — masked/excluded pixels included.
   Masking only ever gates which pixels are searched for invariant targets;
   it never gates what gets corrected in the output. Before the final fit,
   a RANSAC pass (`outlier_relative_threshold`, default 0.05 — see
   "Outlier handling" below) excludes consensus pixels the winning model
   can't explain within a relative tolerance, so a handful of contaminated
   points (e.g. cloud/haze leak-through) can't skew an otherwise-good band.
   The resulting fit is then sanity-checked (`slope_bounds`,
   `min_fit_pixels` — see "Plausibility guard" below) and rejected in favor
   of the identity transform if it's implausible or poorly supported.

## Log-transform (IR-MAD invariant-target detection)

`use_log_transform` (default `True`) fits IR-MAD (Phase A) against
log(DN) rather than raw DN. Sensor noise and real BRDF/illumination-driven
fluctuation both scale roughly with signal level — a bright rooftop can be
just as *relatively* stable as a dark road while still showing a larger
*absolute* DN swing between acquisitions purely from that brightness
scaling. IR-MAD's chi-square test operates on absolute differences in
whatever space it's given, so on raw DN it systematically favors dark,
low-DN pixels over equally-stable bright ones. Log-DN differences are
proportional (percentage) differences, which puts bright and dark surfaces
on a comparable footing — see `test_irmad.py`'s
`test_log_transform_balances_dark_and_bright_invariant_detection` for a
synthetic demonstration (raw DN: an equally-stable dark group gets flagged
invariant 5-280x more often than a bright group with the same relative
noise; log DN: roughly even). Set `use_log_transform=False` to restore the
original raw-DN behavior. This only affects Phase A candidate detection —
Phase C's regression (below) still fits and reports in raw DN space, since
that's the space `apply.apply_model`'s linear correction actually operates
in.

## Outlier handling in the Phase C regression

Passing IR-MAD's own statistical no-change test (Phases A/B) doesn't
guarantee a pixel's value in *this specific* target scene is clean — a
one-off sensor defect, a UDM2/OmniCloudMask false-negative near a cloud
edge, or sub-pixel misregistration bleed can still leave one or a few
consensus pixels far off the true relationship for that scene.

`fit_regression_from_mask` handles this with RANSAC (`_ransac_orthogonal_
regression`), not a single fit-then-clip pass: repeatedly fit a candidate
orthogonal-regression line from a small random subset of the consensus
points, score each candidate by how many of *all* the points it explains
within `outlier_relative_threshold`, and keep whichever candidate has the
most support; the winning inlier set is then refit and re-scored once more
(a standard RANSAC "polish" step) before the caller's own final fit. The
model form stays orthogonal regression throughout — neither the reference
nor the target scene is ground truth, both carry error, so a total-least-
squares fit is kept rather than switching to OLS (which would implicitly
treat the reference as error-free).

**`slope_bounds` is enforced during the search itself**, not just as a
post-hoc check (see "Plausibility guard" below): a candidate whose own
slope falls outside `slope_bounds` is never scored or eligible to win,
regardless of how many points it would otherwise explain. This matters
specifically when contamination (cloud shadow, vegetation, misregistration
bleed) outnumbers the genuine invariant pixels within a scene's consensus
set — without this, RANSAC's "most-supported" logic has no way to prefer a
smaller-but-plausible cluster over a larger-but-implausible one, and can
converge on and report an implausible answer that the post-hoc plausibility
check then has to reject, wasting the whole search on an answer that was
never going to be used. With plausibility folded into the search, RANSAC
keeps looking until it finds a plausible, well-supported model or
exhausts its iteration budget — in which case it falls back to the
untrimmed consensus set, same as the too-few-points case, and the caller's
post-hoc check (unchanged) still catches it. A winning candidate must also
be supported by at least as many inlier points as the minimal sample used
to fit it (`sample_size`, default 12) — a 2-point "winner" trivially fits a
line with r²=1 no matter how nonsensical the resulting slope is, which
becomes more likely to surface once plausibility-gating narrows the
eligible-candidate pool.

**`ransac_inliers/{scene_id}_{band}.tif`** (one boolean raster per band,
on the reference grid, same convention as `consensus/consensus_mask.tif`)
records exactly which consensus pixels survived as RANSAC's final inlier
set for that band — written for every band regardless of whether it was
ultimately accepted or fell back to identity, so a rejected band's raster
is still useful for seeing *what* RANSAC found (a small, spatially
clustered inlier set is a different failure mode than a large-but-
implausible one, and both look different from "found nothing plausible at
all"). Cross-reference against `identity_fallback`/`fallback_reason` in the
corresponding `models/{scene_id}_model.json` to see whether that inlier set
was actually applied.

The outlier criterion is **relative to the model's own prediction**, not a
fixed DN difference: a point is an outlier if `|observed - predicted| /
|predicted| > outlier_relative_threshold` (predicted = intercept +
slope×target; the denominator is floored at 1 DN so a near-zero prediction
can't blow up the ratio). A fixed *absolute* DN threshold would itself
reproduce the dark/bright imbalance this whole feature is meant to guard
against — a 50 DN deviation is huge for a dark road and trivial for a
bright roof; a percentage threshold is equally strict for both. Excluded
points are dropped entirely (weight 0), not down-weighted, since
contamination sources like cloud/haze/shadow leak-through tend to bias
residuals in one direction — a soft (IRLS/Huber-style) weighting would
still let that directional bias pull the fit. Pass
`outlier_relative_threshold=None` to disable this and use every consensus
pixel as-is.

## Plausibility guard: rejecting implausible fits

Even a well-conditioned RANSAC fit can land on a physically nonsensical
answer — most often when a scene has too few consensus pixels for RANSAC's
"most-supported cluster" logic to reliably tell the true relationship apart
from a coincidentally-tight small subset (empirically, this project's own
test dataset shows a sharp quality split around ~5000 consensus pixels: at
or above it, fits are consistently sane; well below it, a meaningful
fraction land on wildly-scaled or even negative slopes, regardless of
`outlier_relative_threshold`). PlanetScope radiometric drift is expected to
be *subtle* — after RANSAC, each band's fit is checked against that prior,
and rejected (not down-weighted, not partially applied) if it fails:

- **Too little support**: fewer than `min_fit_pixels` (default 500) points
  survived RANSAC for this band.
- **Implausible slope**: the fitted slope falls outside `slope_bounds`
  (default `(0.8, 1.2)`; pass `None` to disable this check).

A rejected band falls back to the **identity transform** (`slope=1.0,
intercept=0.0`) — i.e. that band is left uncorrected in the output rather
than having an untrustworthy correction applied. This is a per-band
decision: one band in a scene can be corrected normally while another in
the same scene falls back, if only that band's fit is untrustworthy.
`summary.md` reports a scene-level count of fallback bands, and each
`BandModel` records the detail:

| field | meaning |
|---|---|
| `identity_fallback` | `True` if this band's slope/intercept were replaced with the identity transform |
| `fallback_reason` | why, e.g. `"only 350 consensus pixels survived RANSAC (< min_fit_pixels=500)"` or `"slope 0.427 outside plausible bounds (0.8, 1.2)"` — `None` if no fallback |
| `raw_slope`, `raw_intercept` | the rejected RANSAC fit, kept for inspection — `None` when there was no fallback (in which case they'd just duplicate `slope`/`intercept`) |

Every `BandModel` in a saved `models/{scene_id}_model.json` reports, over
the *final* (post-RANSAC) point set for that band:

| field | meaning |
|---|---|
| `n_invariant_pixels` | consensus pixels actually used in the final fit for this band (post-outlier-removal) |
| `n_outliers_excluded` | how many consensus pixels this band's RANSAC pass dropped |
| `r2` | fit quality of the underlying *fitted* model (raw_slope/raw_intercept when a fallback occurred) — a diagnostic of the fit itself, not of what was actually applied |
| `rmse_before` | target vs. reference RMSE at the final point set, *before* applying the fitted slope/intercept |
| `rmse_after` | same points, *after* applying the fitted slope/intercept — isolates what the correction itself does, holding the point set fixed. Like `r2`, this describes the underlying fit, not necessarily what was applied — check `identity_fallback` |

**Read `r2` alongside `rmse_before`/`rmse_after`, not in isolation.** Across
this project's own test dataset, per-band `r2` at the consensus targets is
usually well below the whole-scene "before" `r2` in
`adjacent_pair_report.csv` (median ~0.10 vs. >0.9) — but `rmse_after` is
lower than `rmse_before` for essentially every band (100% in that same
test run, often by 2-4x), meaning the fitted correction is doing real,
substantial work despite the low `r2`. This isn't a contradiction: `r2` is
the *Pearson correlation* between raw target/reference values at those
pixels, which is sensitive to how much of the whole scene's dynamic range
the targets happen to span (compare against `variance_ratio` in
`spectral_coverage_report.csv` — a narrower target-value range mechanically
suppresses `r2` even when absolute agreement, i.e. RMSE, is fine) as well
as to genuine leftover per-pixel scatter (no sub-pixel registration step
is implemented yet — see `registration.py` — and `ncp_threshold`/
`invariance_frequency_threshold` were deliberately loosened from stricter
defaults to fix cross-scene repeatability, which trades some per-target
tightness for a larger, more reliably-repeated consensus set). A low `r2`
with a large `rmse_after` improvement means "this band's correction is
doing real work, but treat the exact `slope`/`intercept` with somewhat
less confidence than a high-`r2` band would warrant" — it's a QA signal
on the fit, not evidence the correction is failing.

Note `n_invariant_pixels` can differ per band even within one scene (each
band's outlier pass is independent), while `n_consensus_pixels` on the
model as a whole is the pre-outlier-removal count shared by every band —
see "model.json field reference" below for the full list of conditions
that can make these numbers diverge.

## model.json field reference

`slope`/`intercept` are the only two numbers `apply.apply_model` actually
uses — everything else on `NormalizationModel`/`BandModel` is provenance:
what the fit was built from, not part of the transform itself.

`n_invariant_pixels` (per band) and `n_consensus_pixels` (once, on the
model) both describe "how many pixels went into this fit" and are equal
for every band under normal fitting — they can diverge in two ways:

- **Per-band outlier removal** (see above): each band's own RANSAC pass can
  drop a different number of points, so `n_invariant_pixels` varies band to
  band while `n_consensus_pixels` stays fixed at the pre-RANSAC count.
- **Bands the target has but the reference doesn't** (e.g. a SuperDove
  8-band target normalized against a 4-band harmonized reference): those
  bands never enter the regression at all and get a synthetic identity
  `BandModel` (`slope=1.0, intercept=0.0, r2=NaN, n_invariant_pixels=0`) —
  `n_consensus_pixels` still reports the count from whichever bands *were*
  fit.

## Masking

Combines up to eight sources into one bitmask raster per scene
(`masks/{scene_id}_flags.tif`, saved as the "final cloud/water mask"
output, each source its own bit so they stay individually recoverable):

| bit | source | notes |
|---|---|---|
| `EXCLUDE_NODATA` (1) | scene nodata value | |
| `EXCLUDE_WATER` (2) | NDWI = (green-nir)/(green+nir) `> ndwi_threshold` | McFeeters convention; default threshold 0.0 |
| `EXCLUDE_UDM2` (4) | Planet's delivered UDM2 `clear` band | |
| `EXCLUDE_OMNICLOUD` (8) | [OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask), optional | UDM2 alone is known to miss thin cloud/haze |
| `EXCLUDE_VEGETATION` (16) | NDVI = (nir-red)/(nir+red) `> ndvi_threshold` | default threshold 0.2; same convention as spectralmatch's PIF vegetation filter — canopy reflectance drifts with phenology/moisture on timescales far shorter than a useful invariant-target baseline |
| `EXCLUDE_LIDAR_SLOPE` (32) | DSM surface-normal angle `> max_slope_deg`, optional | see "LiDAR/DSM masks" below |
| `EXCLUDE_LIDAR_ROUGH` (64) | DSM local surface-orientation variability `> roughness_max_deg`, optional | ditto |
| `EXCLUDE_LIDAR_SHADOW` (128) | DSM shadow ray-trace at this scene's sun position, optional | ditto |

Computed once per scene (not once per pair), so OmniCloudMask/NDWI/NDVI/LiDAR
masks never rerun redundantly across the many pairs a scene participates in.
Every source is independently toggleable (`use_water_mask`,
`use_vegetation_mask`, `use_omnicloudmask`, `dsm_path`).

### Search mask (erosion/dilation)

`masks/{scene_id}_search_mask.tif` is a second bitmask, derived from
`_flags.tif` by morphologically opening (eroding then dilating,
`mask_erode_px`/`mask_dilate_px`, default 1px each) each exclusion reason
independently — this is what invariant-target search and consensus
aggregation actually read, not the raw `_flags.tif`. The point is avoiding
edge effects at exclusion-region boundaries (a mixed pixel straddling a
cloud edge, a misregistered building edge) without disturbing what reference
selection, clear-percent reporting, and the adjacent-pair metrics report see
— those all still read the raw, unmodified `_flags.tif`. Set
`mask_erode_px=0, mask_dilate_px=0` to make the search mask identical to the
raw flags.

## LiDAR/DSM masks (optional)

Pass `dsm_path` (a single GeoTIFF, or a directory of tiles — mosaicked via a
GDAL VRT, no separate merge step needed, and the DSM's own tiling scheme
doesn't need to match the PlanetScope imagery's) to add three more
purely-geometric, spectrum-independent exclusion sources. `dsm_path=None`
(the default) disables all three entirely — nothing DSM-related runs.

```python
result = run_pipeline(
    ...,
    dsm_path="path/to/dsm.tif",       # or a directory of tiles; None disables everything below
    dsm_height_units="m",             # "m" or "ft" -- vertical unit of the DSM's own pixel values
    max_slope_deg=5.0,                # horizontality: exclude surface-normal angle > this
    roughness_window_radius_px=2,     # roughness: local-variability window radius, in DSM pixels
    roughness_max_deg=10.0,           # roughness: exclude local orientation variability > this
    use_shadow_mask=False,            # separate opt-in -- ray-tracing is far more expensive than slope/roughness
    max_building_height_m=150.0,      # shadow: farthest an obstruction this tall could still cast a shadow
    mask_erode_px=1, mask_dilate_px=1,  # search-mask morphology, see above -- applies regardless of dsm_path
)
```

- **Horizontality** (`EXCLUDE_LIDAR_SLOPE`): a target sitting on sloped
  ground has real BRDF/illumination-geometry effects a single pair of images
  can't distinguish from genuine radiometric change, so a pixel's IR-MAD
  "invariance" there is coincidental rather than physically stable. Slope is
  the DSM's surface-normal angle from vertical (a central-difference
  gradient).
- **Roughness** (`EXCLUDE_LIDAR_ROUGH`): the Vector Ruggedness Measure
  (Sappington et al. 2007) — the angular spread of surface-normal *direction*
  within a local window, as opposed to slope's *magnitude*. This
  deliberately does NOT flag a smooth-but-steep slope (every normal points
  the same way there, so VRM ~ 0) — it flags places where normals point in
  divergent directions within a small area, the signature of vegetation
  canopy and building/tree edges rather than merely sloped-but-stable
  ground.
- **Shadow** (`EXCLUDE_LIDAR_SHADOW`, opt-in via `use_shadow_mask=True`): a
  vectorized ray-march (whole-array shift-and-compare per sample distance,
  not a per-pixel loop) against each scene's own sun position — from
  metadata.json's `sun_azimuth`/`sun_elevation` if present, else computed
  from acquisition time (always UTC) + scene centroid lat/lon via the same
  NOAA solar-position formula `pipeline.sun_zenith_angle_for_scene` already
  used for reference selection. Cost scales with
  `max_building_height_m / tan(elevation) / ray_step_m` samples per pixel —
  `shadow_ray_step_m` (coarsen the march) trades accuracy for speed.
  Processed in row-blocks (`lidar_masks.compute_and_cache_shadow`), each
  padded by a halo covering the full search radius, so peak memory is
  bounded regardless of how large the shared window is — a genuinely
  necessary fix, not just an optimization: a single whole-array shadow
  computation over a multi-scene union window caused a real OOM in
  production (see git history). Unlike slope/roughness,
  `shadow_downsample_factor` is not supported by this blocked path (would
  reintroduce the cross-block grid-phase-alignment hazard already solved
  for `io.read_block_flat` elsewhere in this codebase) — shadow always runs
  at native DSM resolution for now. The principled fix if this needs to
  scale past a modest number of scenes/sun-angle buckets is a different
  algorithm: precompute, once per pixel and independent of any scene's
  date/time, the horizon angle in a fixed set of azimuth bins (see GRASS
  GIS `r.horizon`) — a lookup instead of a ray-march, shared across every
  scene and every year for free. Not implemented here to keep this first
  pass's scope bounded.

**Cost-sharing across scenes.** Slope/roughness depend only on terrain, so
they're computed once per pipeline run — over the union of every scene's
own extent that geometrically overlaps the reference (not just the
reference's own footprint; a target scene from a different overpass/strip
commonly extends beyond it, and would otherwise get zero LiDAR coverage in
the part outside the reference even where the DSM has real data there),
plus a buffer for the roughness window and for the shadow search radius if
enabled — and reused for every target scene via a cheap warp-crop onto that
scene's grid, not recomputed per scene. Shadow masks depend on per-scene sun
position, but scenes captured close together in time share near-identical
sun angles, so they're cached to disk under `masks/_lidar_cache/` keyed on a
rounded `(azimuth, elevation)` bucket
(`shadow_angle_bucket_deg`, default 1°) and reused by any later scene (or
worker process, or resumed run) whose sun position rounds to the same
bucket — a practical first version of a sun-angle lookup table; the
per-pixel horizon-angle datacube above is the more general version of the
same idea.

Every mask is computed against the DSM's own native resolution/CRS, not
resampled down to the coarser PlanetScope grid first (which would blur out
exactly the small building/canopy edges these masks exist to catch), then
warped onto each scene's grid with max-resampling — a single flagged DSM
pixel conservatively excludes whichever (coarser) PlanetScope pixel it falls
in. Wherever the DSM has no coverage at all, none of the three masks exclude
anything there (fail-open, the same convention as a missing UDM2 file
elsewhere in masking.py) rather than assuming the worst.

Diagnostic rasters — useful for judging whether `max_slope_deg`/
`roughness_max_deg` need retuning against a real scene, since neither has a
principled "correct" default the way, say, NDVI's 0.2 threshold does —
are saved per scene alongside the bitmask:
`masks/{scene_id}_horizontality_flag.tif`, `masks/{scene_id}_roughness_flag.tif`,
`masks/{scene_id}_unshadowed_flag.tif` (only when `use_shadow_mask=True`).

## Install

```bash
python3 -m venv --system-site-packages .venv   # --system-site-packages picks up
                                                 # apt-installed GDAL/numpy/scipy
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install omnicloudmask             # optional, see below
```

GDAL's Python bindings must match your installed `libgdal` — installing it
via your system package manager (`apt install gdal-bin python3-gdal` on
Debian/Ubuntu) and creating the venv with `--system-site-packages` is the
path of least resistance; a plain `pip install GDAL` often fails to build.

OmniCloudMask specifically pulls in `torch` and downloads pretrained
weights from huggingface.co on first use — set `use_omnicloudmask=False` to
skip it entirely (e.g. to avoid the dependency, or over imagery it isn't
well validated on).

## Usage

```python
from psnorm.pipeline import run_pipeline

result = run_pipeline(
    input_folder="test_images/output_20241001_4b/files",
    output_folder="test_images/output_20241001_4b/psnorm_output",
    use_water_mask=True, ndwi_threshold=0.0,
    use_vegetation_mask=True, ndvi_threshold=0.2,
    use_omnicloudmask=True,           # False to skip the OmniCloudMask pass
    use_log_transform=True,               # fit IR-MAD against log(DN); see "Log-transform" below
    ncp_threshold=0.70,                   # per-pair "how invariant must a pixel be" cutoff
    invariance_frequency_threshold=0.5,  # keep pixels invariant in >50% of evaluations
    min_observations=2,                   # floor so 1-2 lucky observations can't qualify
    downsample_targets=False,      # True to search for targets/fit corrections at reduced resolution
    downsample_resolution_m=15.0,  # only used when downsample_targets=True
    outlier_relative_threshold=0.05,  # RANSAC pass, relative-to-prediction threshold; None disables
    slope_bounds=(0.8, 1.2),   # reject (fall back to identity) fits outside this slope range; None disables
    min_fit_pixels=500,        # reject (fall back to identity) fits from fewer post-RANSAC points than this
    workers="cpu",
    device="auto",   # "auto"/"cpu"/"gpu" -- see "GPU support" below
)
```

This discovers `*_AnalyticMS_SR_harmonized_clip.tif` scenes (paired with
their `*_udm2_clip.tif` and `*_metadata.json`), auto-selects a reference
scene (lowest 10th percentile of sun zenith angle, then highest clear
fraction — pass `reference_path=...` to override), runs the three-phase
invariant-target selection above, fits+applies a normalization model per
target scene, and reports R²/RMSE agreement between chronologically
adjacent scene pairs before vs. after normalization
(`adjacent_pair_report.csv` + `summary.md` in `output_folder`).

Output layout:
```
output_folder/
  masks/{scene_id}_flags.tif                  per-scene exclusion bitmask (raw, see "Masking")
  masks/{scene_id}_search_mask.tif            eroded/dilated search mask (see "Search mask")
  masks/{scene_id}_horizontality_flag.tif     LiDAR: not flat/level (only if dsm_path set)
  masks/{scene_id}_roughness_flag.tif         LiDAR: locally rough surface (only if dsm_path set)
  masks/{scene_id}_unshadowed_flag.tif        LiDAR: likely shadowed (only if use_shadow_mask=True)
  masks/_lidar_cache/                         cached DSM-derived rasters shared across scenes (see "LiDAR/DSM masks")
  candidates/{target_id}_candidate.tif  per-target Phase A invariant candidates
  candidates/{target_id}_stats.json     IR-MAD diagnostics for that candidate fit
  consensus/{times_evaluated,times_invariant,frequency,consensus_mask}.tif
  models/{scene_id}_model.json          final per-band gain/offset
  ransac_inliers/{scene_id}_{band}.tif  final RANSAC inlier pixels for that band's fit (see "Outlier handling")
  normalized/{scene_id}_normalized.tif  corrected scene (every pixel, unmasked, always native resolution)
  adjacent_pair_report.csv, summary.md
  spectral_coverage_report.csv          per-scene, per-band invariant-target spectral coverage (see below)
```

## Spectral coverage report

`spectral_coverage_report.csv` (one row per scene per band, for the
reference and every "fitted" target — scenes reused via `resume` or that
never got a fit have no fresh data and are omitted) is a QA check on
whether a scene's invariant targets actually span its real reflectance
range, rather than being clustered in a narrow slice that the regression
then extrapolates outside of. Columns:

| column | meaning |
|---|---|
| `n_target_pixels` | invariant-target pixel count contributing to this band |
| `target_variance` | variance of the band's values among invariant targets |
| `whole_image_variance` | variance of the band's values across the whole scene |
| `variance_ratio` | `target_variance / whole_image_variance` — near 0 means targets are far less variable than the scene itself (not necessarily bad on its own, but worth reading alongside `coverage_fraction`) |
| `whole_image_p10`, `whole_image_p90` | the scene's own 10th/90th percentile for this band |
| `target_min`, `target_max` | the invariant targets' own value range |
| `coverage_fraction` | how much of `[p10, p90]` the targets' `[min, max]` actually spans — near 0 means the targets sample only a thin slice of the scene's real dynamic range; near 1 means they span (close to) the full interquantile range |

This is always computed from full native-resolution pixel values,
independent of `downsample_targets` (see below) — it's meant to honestly
describe the real reflectance diversity present, which coarsened values
would understate.

## Downsampling targets/correction factors (optional)

`downsample_targets=True` (default `False`) makes IR-MAD fitting, invariant
candidate detection, and the final per-band regression all analyze
spatially-coarsened pixel values — each `downsample_resolution_m /
native_pixel_size` block of native pixels is downsampled (aggregated via
GDAL averaging) to one value, reducing sensitivity to per-pixel sensor/
registration noise (the
finer the resolution, the noisier per-pixel change detection tends to be).
`downsample_resolution_m` (default 15.0) sets the target resolution.

This only affects *how targets are found and how correction factors are
fit* — window/mask shapes, the consensus mask, and all saved rasters stay
at native resolution and shape throughout (each native pixel within one
coarsened cell just carries the same smoothed value during fitting). Final
`apply.py` output is completely unaffected: a fitted model is just a
handful of per-band scalars (slope/intercept), applied to the *original*
full-resolution scene regardless of how those scalars were derived.

## GPU support

`device` (default `"auto"`) selects the array backend (`numpy` or `cupy`)
for Phase A's IR-MAD fit and invariant-pixel classification — the only part
of the pipeline where GPU acceleration is worthwhile (see below for why):

- `"auto"` uses a GPU if `cupy` is importable and reports a usable CUDA
  device, else falls back to plain CPU/NumPy — silently, since that's the
  point of "auto".
- `"gpu"` requires a usable GPU and **raises** if none is found, rather than
  silently downgrading to CPU — an explicit request that can't be honored
  should be visible, not quietly slower.
- `"cpu"` always runs the original NumPy path, regardless of what's
  available — this is the fallback every other value degrades to, so
  results (and performance characteristics) on a machine with no GPU are
  unchanged from before GPU support existed.

Install `cupy` matching your CUDA toolkit (e.g. `pip install cupy-cuda12x`)
to enable it; nothing else needs to change, and no code imports `cupy`
unless `device` actually resolves to `"gpu"`.

**What stays on CPU regardless of `device`**: the canonical-correlation
eigensolve (`irmad._solve_canonical_correlation`) operates on a tiny
`n_bands x n_bands` matrix (4-8 for PlanetScope) — a GPU solve there would
be dominated by kernel-launch latency, not compute — and
`scipy.stats.chi2.sf` (the no-change-probability/reweighting step) has no
reliably-available GPU equivalent. Both run on small, cheaply-transferred
arrays each iteration; only the large `(n_pixels, n_bands)` per-pixel work
(log-transform, the MAD chi-square statistic, weighted-covariance
accumulation) actually runs on `device`.

**Why only Phase A**: profiling the pipeline's own cost structure (see
`irmad.py`'s per-iteration streaming and `normalize.fit_regression_from_mask`)
shows Phase A's IR-MAD fit is the dominant cost — O(scenes × pixels ×
iterations), with up to 30 iterations per scene — while Phase C's
regression only touches the (deliberately small) consensus-pixel set and is
dominated by disk I/O, not per-pixel math. Moving Phase C to GPU wouldn't
meaningfully change its runtime, so it stays CPU-only.

**Parallelism note**: `workers` (CPU process count) and `device` are
somewhat in tension — many CPU processes each independently grabbing the
one GPU will contend for it rather than getting N-way speedup the way
CPU-only workers do. `run_pipeline` logs a note when `device` resolves to
`"gpu"` and `workers` isn't 1; consider a small worker count in that case.

### Window caching (why re-running IR-MAD is fast even on CPU)

`irmad.fit_irmad` reads its reference/target window from disk **exactly
once**, in row-block chunks, and keeps every block resident in memory (or
on-device, under `device="gpu"`) for however many iterations it takes to
converge — the iteration loop itself never touches disk again. Without this,
IR-MAD's own reweighting scheme (re-scoring the same pixels against an
updated model every iteration) would otherwise re-read — and, for
compressed source rasters, re-decompress — the same pixels from disk up to
`max_iter` (default 30) times per scene.

`normalize.detect_invariant_candidates` reuses that same cache (via
`fit_irmad(..., return_cache=True)`) for its own classification pass
instead of reading the window a second time, so a full Phase A candidate
detection touches each pixel's source file exactly once per scene,
regardless of how many IR-MAD iterations it took to converge.

### Saved IR-MAD fits: cheap `ncp_threshold` sweeps

Every target scene's converged `IrMadFit` (canonical-correlation weights,
means, sigma — a handful of scalars and small matrices, not pixel data) is
saved to `candidates/{target_id}_irmad_fit.json` (`model_io.save_irmad_fit`)
alongside the existing candidate mask and stats — `resume="yes"/"validate"`
now requires this file to exist too before treating a scene as already
detected.

This is what makes `ncp_threshold` cheap to re-tune after a run:
`normalize.reclassify_invariant_pixels` reloads a saved fit and reclassifies
against a *new* threshold in one O(pixels) pass, skipping IR-MAD's
expensive O(pixels × iterations) iterative refit entirely (which doesn't
depend on `ncp_threshold` at all — only the final classification step
does). `invariance_frequency_threshold`/`min_observations` (Phase B) were
already cheap to re-sweep this way, since `consensus.aggregate_consensus`
separates count accumulation from threshold application.

`scripts/sweep_thresholds.py` uses this to sweep `ncp_threshold` x
`invariance_frequency_threshold` against an already-completed run — see its
module docstring for usage and the exact metrics it reports.

## Module map

| module | responsibility |
|---|---|
| `sensors.py` | band-name detection (Dove-C/R, SuperDove 4b/8b, ...) |
| `io.py` | scene discovery, raster metadata, grid alignment/overlap, chunked reads/writes |
| `backend.py` | GPU/CPU array-backend selection (NumPy/CuPy) — see "GPU support" above |
| `masking.py` | per-scene exclusion bitmask: nodata + NDWI water + UDM2 + OmniCloudMask |
| `registration.py` | grid-alignment check (implemented) + co-registration (**stub**, see below) |
| `irmad.py` | IR-MAD: weighted covariance, canonical correlation, chi-square weighting |
| `normalize.py` | Phase A candidate detection + Phase C regression-from-consensus-mask |
| `consensus.py` | Phase B: cross-scene frequency aggregation → final consensus mask |
| `model_io.py` | save/load fitted models, IR-MAD fits, + candidate stats (JSON) — enables resumable runs and cheap threshold re-sweeps |
| `apply.py` | chunked application of a fitted model to a full scene |
| `metrics.py` | R²/RMSE agreement between two rasters over a shared mask |
| `pipeline.py` | orchestration: discover → reference selection → phases A/B/C → adjacency report |

## Registration is not implemented yet

`registration.register_to_reference()` is a stub (raises `NotImplementedError`).
`pipeline.py` only calls it when two scenes are not already pixel-grid-aligned
(`io.grids_aligned` / `registration.is_grid_aligned`); such scenes are
skipped with a recorded reason rather than crashing the run. See the
docstring in `registration.py` for the intended contract for whoever
implements it next.

## Multi-sensor / multi-band support

Nothing is hardcoded to 4 bands. `sensors.detect_band_names` reads each
scene's own GDAL band descriptions (falling back to a band-count-keyed
default order with a warning). Reference and target scenes are matched by
*logical* band name, not position — if a target has bands the reference
doesn't (e.g. SuperDove's coastal blue/green I/yellow/red edge vs a 4-band
harmonized reference), those bands pass through unnormalized rather than
being dropped, and bands with no match on either side raise a clear error.
