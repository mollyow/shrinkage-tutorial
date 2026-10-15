# Output schemas (comparison-v1.yml `reporting.machine_readable_required_outputs`).
#
# Every table a simulation job emits is validated against these column specs
# before it is written. endpoint_manifest.yml belongs to Gate C2.

CMP_KEY_COLUMNS <- c(phase = "character", geometry = "character", cell_id = "integer",
                     cell_name = "character", multiplier = "double", replication = "integer")

CMP_SCHEMAS <- list(
  job_ledger = c(CMP_KEY_COLUMNS, method = "character", outer_fold = "integer",
                 status = "character", elapsed_seconds = "double", peak_memory_mb = "double",
                 n_train_rows = "integer", n_heldout_rows = "integer",
                 n_heldout_respondents = "integer", selected = "character",
                 n_heldout_projected = "integer", n_integration_projected = "double",
                 n_integration_predictions = "double", n_warnings = "integer",
                 warnings = "character", error = "character", retries = "integer",
                 fallbacks = "integer"),
  fold_predictions = c(CMP_KEY_COLUMNS, id = "character", task = "integer", side = "integer",
                       outer_fold = "integer", method = "character", prediction_raw = "double",
                       prediction = "double", y = "double", true_mean = "double"),
  respondent_scores = c(CMP_KEY_COLUMNS, id = "character", outer_fold = "integer",
                        method = "character", coordinate = "character", score = "double"),
  tuning_history = c(CMP_KEY_COLUMNS, method = "character", outer_fold = "integer",
                     candidate = "integer", parameters = "character", mean_inner_mse = "double",
                     inner_losses = "character", selected = "logical", error = "character"),
  failures = c(CMP_KEY_COLUMNS, method = "character", outer_fold = "integer",
               stage = "character", message = "character"),
  effect_estimates = c(CMP_KEY_COLUMNS, family = "character", coordinate = "character",
                       key = "character", truth = "double", estimate = "double",
                       std_error = "double", error = "double", lower_95 = "double",
                       upper_95 = "double", simultaneous_lower_95 = "double",
                       simultaneous_upper_95 = "double", covered_95 = "logical",
                       method = "character"),
  family_metrics = c(CMP_KEY_COLUMNS, family = "character", dimension = "integer",
                     covariance_rank = "integer", squared_error = "double",
                     normalized_risk = "double", variance_trace = "double",
                     normalized_variance = "double", wald_statistic = "double",
                     wald_df = "integer", wald_p_value = "double",
                     truth_wald_statistic = "double", truth_wald_covered_95 = "logical",
                     simultaneous_critical = "double", simultaneous_covered_95 = "logical",
                     null_family = "logical", method = "character", prediction_mse = "double",
                     prediction_log_loss = "double")
)

CMP_SESSION_MANIFEST_KEYS <- c("created_utc", "git_head", "worktree_dirty", "r_version",
                               "platform", "blas", "package_versions")

cmp_empty_table <- function(name) {
  spec <- CMP_SCHEMAS[[name]]
  cols <- lapply(spec, function(type) vector(if (type == "double") "numeric" else type, 0L))
  as.data.frame(cols, stringsAsFactors = FALSE)
}

# Coerce a table to its schema; fail on missing or unexpected columns.
cmp_conform <- function(df, name) {
  spec <- CMP_SCHEMAS[[name]]
  cmp_assert(!is.null(spec), "Unknown output table `%s`.", name)
  if (is.null(df) || !nrow(df)) return(cmp_empty_table(name))
  missing <- setdiff(names(spec), names(df))
  extra <- setdiff(names(df), names(spec))
  cmp_assert(!length(missing), "Table %s is missing columns: %s", name, paste(missing, collapse = ", "))
  cmp_assert(!length(extra), "Table %s has unexpected columns: %s", name, paste(extra, collapse = ", "))
  out <- df[names(spec)]
  for (col in names(spec)) {
    out[[col]] <- switch(spec[[col]],
                         character = as.character(out[[col]]),
                         integer = as.integer(out[[col]]),
                         double = as.numeric(out[[col]]),
                         logical = as.logical(out[[col]]))
  }
  rownames(out) <- NULL
  out
}

cmp_conform_dataset <- function(result) {
  for (name in names(CMP_SCHEMAS)) result[[name]] <- cmp_conform(result[[name]], name)
  result
}

# A dataset is complete when every method x fold job and every method x
# coordinate estimate is present (incomplete_expected_result_keys is a hard fail).
cmp_check_complete <- function(result, geom, methods, outer_folds) {
  l <- result$job_ledger
  expected <- expand.grid(method = methods, outer_fold = seq_len(outer_folds),
                          stringsAsFactors = FALSE)
  got <- paste(l$method, l$outer_fold)
  cmp_assert(setequal(paste(expected$method, expected$outer_fold), got) && !anyDuplicated(got),
             "Incomplete or duplicated job keys.")
  e <- result$effect_estimates
  want <- paste(rep(methods, each = nrow(geom$coordinates)),
                rep(geom$coordinates$key, times = length(methods)))
  have <- paste(e$method, e$key)
  cmp_assert(setequal(want, have) && !anyDuplicated(have), "Incomplete or duplicated estimate keys.")
  f <- result$family_metrics
  want_f <- paste(rep(methods, each = length(geom$families)),
                  rep(names(geom$families), times = length(methods)))
  cmp_assert(setequal(want_f, paste(f$method, f$family)), "Incomplete family metric keys.")
  n_score <- length(methods) * nrow(geom$coordinates) * result$meta$respondents
  cmp_assert(nrow(result$respondent_scores) == n_score, "Incomplete respondent scores.")
  invisible(TRUE)
}

cmp_write_parquet <- function(df, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp-", Sys.getpid())
  # nanoparquet 0.4.3 needs an explicit row group for zero-row tables.
  if (nrow(df)) nanoparquet::write_parquet(df, tmp) else nanoparquet::write_parquet(df, tmp, row_groups = 1L)
  cmp_assert(file.rename(tmp, path), "Could not move %s into place.", path)
  invisible(path)
}
