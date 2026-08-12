# psnorm

Relative radiometric normalization for PlanetScope imagery: IR-MAD-based
automatic invariant-pixel selection (in the spirit of ArrNorm), with a
chunked/resumable/multi-sensor pipeline (in the spirit of spectralmatch).

## Why

Comparing PlanetScope scenes over time (e.g. to track phenology or tree
mortality) requires correcting for per-scene radiometric drift — atmosphere,
sun angle, sensor calibration — without erasing genuine ground change. IR-MAD
finds pixels that are *jointly* invariant across all bands via canonical
correlation + iterative chi-square reweighting, which is more robust to real
land-cover change leaking into the fit than simpler per-band PCA/threshold
methods.

Invariant-target selection is a three-phase, cross-scene-checked process
rather than a single pairwise fit — see "How invariant targets are
selected" below — and every artifact along the way (per-scene exclusion
masks, per-pair candidates, the final consensus set) is written to disk, not
just the final gain/offset.

## How invariant targets are selected

1. **Phase A — per-target candidate detection.** Each target scene is
   compared to the reference *only* (never target-vs-target) via IR-MAD,
   restricted to pixels that pass that pair's search mask (nodata, water,
   vegetation, UDM2, and OmniCloudMask all excluded — see Masking below).
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

Every `BandModel` in a saved `models/{scene_id}_model.json` reports, over
the *final* (post-RANSAC) point set for that band:

| field | meaning |
|---|---|
| `n_invariant_pixels` | consensus pixels actually used in the final fit for this band (post-outlier-removal) |
| `n_outliers_excluded` | how many consensus pixels this band's RANSAC pass dropped |
| `r2` | fit quality over the final point set |
| `rmse_before` | target vs. reference RMSE at the final point set, *before* applying slope/intercept |
| `rmse_after` | same points, *after* applying slope/intercept — isolates what the correction itself does, holding the point set fixed |

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

Combines five sources into one bitmask raster per scene
(`masks/{scene_id}_flags.tif`, saved as the "final cloud/water mask"
output, each source its own bit so they stay individually recoverable):

| bit | source | notes |
|---|---|---|
| `EXCLUDE_NODATA` (1) | scene nodata value | |
| `EXCLUDE_WATER` (2) | NDWI = (green-nir)/(green+nir) `> ndwi_threshold` | McFeeters convention; default threshold 0.0 |
| `EXCLUDE_UDM2` (4) | Planet's delivered UDM2 `clear` band | |
| `EXCLUDE_OMNICLOUD` (8) | [OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask), optional | UDM2 alone is known to miss thin cloud/haze |
| `EXCLUDE_VEGETATION` (16) | NDVI = (nir-red)/(nir+red) `> ndvi_threshold` | default threshold 0.2; same convention as spectralmatch's PIF vegetation filter — canopy reflectance drifts with phenology/moisture on timescales far shorter than a useful invariant-target baseline |

Computed once per scene (not once per pair), so OmniCloudMask/NDWI/NDVI never
rerun redundantly across the many pairs a scene participates in. Every
source is independently toggleable (`use_water_mask`, `use_vegetation_mask`,
`use_omnicloudmask`).

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
    workers="cpu",
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
  masks/{scene_id}_flags.tif            per-scene exclusion bitmask
  candidates/{target_id}_candidate.tif  per-target Phase A invariant candidates
  candidates/{target_id}_stats.json     IR-MAD diagnostics for that candidate fit
  consensus/{times_evaluated,times_invariant,frequency,consensus_mask}.tif
  models/{scene_id}_model.json          final per-band gain/offset
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

## Module map

| module | responsibility |
|---|---|
| `sensors.py` | band-name detection (Dove-C/R, SuperDove 4b/8b, ...) |
| `io.py` | scene discovery, raster metadata, grid alignment/overlap, chunked reads/writes |
| `masking.py` | per-scene exclusion bitmask: nodata + NDWI water + UDM2 + OmniCloudMask |
| `registration.py` | grid-alignment check (implemented) + co-registration (**stub**, see below) |
| `irmad.py` | IR-MAD: weighted covariance, canonical correlation, chi-square weighting |
| `normalize.py` | Phase A candidate detection + Phase C regression-from-consensus-mask |
| `consensus.py` | Phase B: cross-scene frequency aggregation → final consensus mask |
| `model_io.py` | save/load fitted models + candidate stats (JSON) — enables resumable runs |
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
