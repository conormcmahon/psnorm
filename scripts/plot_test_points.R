#!/usr/bin/env Rscript
# Plot daily variation at each test point, one figure per (band, class)
# combination, one line per point in that class -- original DN (dashed)
# and psnorm-corrected DN (solid) overlaid in matching color per point, so
# the effect of the correction is visible directly on each point's own
# trajectory rather than needing a separate before/after figure.
#
# Cloud-masked observations (flags_bitmask bit 4 = UDM2 cloud/shadow/snow/
# haze, bit 8 = OMNICLOUD) are still shown as points -- so a masked outlier
# is visible -- but excluded from the connecting lines, so the line reflects
# only the observations psnorm actually trusts.
#
# Usage:
#   Rscript scripts/plot_test_points.R <input_csv> <output_dir>

suppressMessages({
  library(ggplot2)
  has_dt <- requireNamespace("data.table", quietly = TRUE)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript plot_test_points.R <input_csv> <output_dir>")
}
input_csv <- args[1]
output_dir <- args[2]

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# The consensus-sample run produces a CSV in the millions of rows (2000
# points x ~850 scenes x 4 bands); data.table::fread is both much faster and
# much lighter on memory than base read.csv at that size, so prefer it when
# available. Converted straight back to a plain data.frame so the rest of
# this script (data.frame-style `[` subsetting throughout) behaves exactly
# the same regardless of which reader ran.
if (has_dt) {
  data <- as.data.frame(data.table::fread(input_csv, showProgress = FALSE))
} else {
  data <- read.csv(input_csv, stringsAsFactors = FALSE)
}
data$date <- as.Date(data$date)

# Long format: one row per (point, scene, band, value_type), so both
# original and corrected values can share one "value" column and be
# distinguished by linetype in the same plot.
is_cloud_masked <- bitwAnd(data$flags_bitmask, 4L) != 0 | bitwAnd(data$flags_bitmask, 8L) != 0
is_cloud_masked[is.na(is_cloud_masked)] <- FALSE

long <- rbind(
  data.frame(
    point_id = data$point_id, class = data$class, date = data$date, band = data$band,
    value_type = "original", value = data$original_dn, cloud_masked = is_cloud_masked
  ),
  data.frame(
    point_id = data$point_id, class = data$class, date = data$date, band = data$band,
    value_type = "corrected", value = data$corrected_dn, cloud_masked = is_cloud_masked
  )
)
long <- long[!is.na(long$value), ]
long$point_id <- factor(long$point_id)

bands <- sort(unique(data$band))
classes <- sort(unique(data$class))

n_written <- 0
for (band_name in bands) {
  for (class_name in classes) {
    subset_data <- long[long$band == band_name & long$class == class_name, ]
    if (nrow(subset_data) == 0) {
      next
    }
    n_points <- length(unique(subset_data$point_id))
    line_data <- subset_data[!subset_data$cloud_masked, ]

    p <- ggplot(subset_data, aes(x = date, y = value, color = point_id, linetype = value_type, group = interaction(point_id, value_type))) +
      geom_line(data = line_data, alpha = 0.5, linewidth = 0.3) +
      geom_point(aes(shape = cloud_masked), size = 0.6, alpha = 0.4) +
      scale_linetype_manual(values = c(original = "dashed", corrected = "solid")) +
      scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 4)) +
      labs(
        title = sprintf("%s band -- class '%s' (%d points)", band_name, class_name, n_points),
        subtitle = "dashed = original DN, solid = psnorm-corrected DN; lines exclude cloud-masked points (shown as 'x')",
        x = "date", y = "DN", color = "point_id", linetype = "value type", shape = "cloud-masked"
      ) +
      theme_minimal() +
      theme(legend.position = "right")

    # A per-point color legend is only useful with a handful of points --
    # past ~15 it's an unreadable wall of swatches that also balloons the
    # PNG, so drop it once there are too many points to show colors for the
    # trajectories themselves (still distinguishable by eye via color hue)
    # without pretending the legend maps them individually.
    if (n_points > 15) {
      p <- p + guides(color = "none")
    }

    out_path <- file.path(output_dir, sprintf("%s_%s.png", band_name, class_name))
    ggsave(out_path, plot = p, width = 10, height = 6, dpi = 150)
    n_written <- n_written + 1
    cat(sprintf("wrote %s (%d rows, %d points)\n", out_path, nrow(subset_data), n_points))
  }
}

cat(sprintf("Done: %d plots written to %s\n", n_written, output_dir))
