.narrow_ml_file <- function(...) {
  root <- normalizePath(file.path(testthat::test_path(), "..", ".."), mustWork = TRUE)
  file.path(root, ...)
}

.load_narrow_ml_functions <- function() {
  environment <- new.env(parent = globalenv())
  sys.source(
    .narrow_ml_file("analysis", "narrow_ml", "benchmark_functions.R"),
    envir = environment
  )
  environment
}

testthat::test_that("narrow-ML production thresholds and method set are locked", {
  testthat::skip_if_not_installed("yaml")
  benchmark <- .load_narrow_ml_functions()
  config <- benchmark$read_benchmark_config(
    .narrow_ml_file("analysis", "narrow_ml", "benchmark-production.yml")
  )
  testthat::expect_identical(config$status, "locked_before_benchmark")
  testthat::expect_identical(config$respondents, 600L)
  testthat::expect_identical(config$tasks_per_respondent, 8L)
  testthat::expect_identical(config$replications, 200L)
  testthat::expect_identical(
    config$profile_sampling, "without_replacement_within_respondent"
  )
  testthat::expect_identical(config$models, c(
    "unadjusted", "additive_lm", "interacted_lm", "elastic_net", "ranger",
    "plugin_ranger", "oracle"
  ))
  testthat::expect_equal(config$thresholds$oracle_headroom_mse_ratio, 0.75)
  testthat::expect_equal(config$thresholds$nonlinear_mse_ratio_to_additive, 0.85)
  testthat::expect_equal(config$thresholds$application_score_variance_ratio, 0.95)
})

testthat::test_that("candidate-geometry DGP preserves the locked null and assignment", {
  testthat::skip_if_not_installed("yaml")
  testthat::skip_if_not_installed("readr")
  raw <- .narrow_ml_file(
    "Candidate recruitment", "replication", "PSRM_replication", "data",
    "CleanedData.csv"
  )
  testthat::skip_if_not(file.exists(raw), "Local candidate replication data are absent.")
  benchmark <- .load_narrow_ml_functions()
  config <- benchmark$read_benchmark_config(
    .narrow_ml_file("analysis", "narrow_ml", "benchmark-fast.yml")
  )
  design <- benchmark$load_candidate_benchmark_design(
    normalizePath(.narrow_ml_file(), mustWork = TRUE)
  )
  design <- benchmark$prepare_scenario_calibration(design, config)

  testthat::expect_true(all(design$calibration$minimum_conditional_mean > 0))
  testthat::expect_true(all(design$calibration$maximum_conditional_mean < 1))
  uniform <- benchmark$scenario_contrast(design, "additive_null")
  shifted <- benchmark$scenario_contrast(design, "additive_target_shift")
  testthat::expect_equal(uniform$max_target_assignment_ratio, 1)
  testthat::expect_equal(shifted$max_target_assignment_ratio, 1.5)

  null_plugin <- benchmark$true_context_plugins(
    design, "additive_null", uniform$contrast
  )
  null_truth <- as.numeric(crossprod(
    design$contexts$population_probability, null_plugin
  ))
  testthat::expect_lt(max(abs(null_truth)), 1e-10)

  config$respondents <- 40L
  first <- benchmark$simulate_candidate_geometry(
    design, "nonlinear_heterogeneity", 1L, config
  )$data
  shifted_data <- benchmark$simulate_candidate_geometry(
    design, "nonlinear_target_shift", 1L, config
  )$data
  testthat::expect_identical(first$profile_index, shifted_data$profile_index)
  testthat::expect_equal(first$y, shifted_data$y)
  testthat::expect_true(all(first$y * 9 == round(first$y * 9)))
  testthat::expect_true(all(first$y >= 0 & first$y <= 1))
  testthat::expect_true(all(vapply(split(first$profile_index, first$id), function(x) {
    !anyDuplicated(x)
  }, logical(1))))
  testthat::expect_true(all(vapply(split(first$fold, first$id), function(x) {
    length(unique(x)) == 1L
  }, logical(1))))
  testthat::expect_true(all(vapply(split(first$weight, first$id), function(x) {
    length(unique(x)) == 1L
  }, logical(1))))
})

testthat::test_that("elastic-net factorial encoding is symmetric to level order", {
  testthat::skip_if_not_installed("yaml")
  testthat::skip_if_not_installed("readr")
  raw <- .narrow_ml_file(
    "Candidate recruitment", "replication", "PSRM_replication", "data",
    "CleanedData.csv"
  )
  testthat::skip_if_not(file.exists(raw), "Local candidate replication data are absent.")
  benchmark <- .load_narrow_ml_functions()
  config <- benchmark$read_benchmark_config(
    .narrow_ml_file("analysis", "narrow_ml", "benchmark-fast.yml")
  )
  design <- benchmark$prepare_scenario_calibration(
    benchmark$load_candidate_benchmark_design(
      normalizePath(.narrow_ml_file(), mustWork = TRUE)
    ),
    config
  )
  config$respondents <- 20L
  data <- benchmark$simulate_candidate_geometry(
    design, "additive_null", 2L, config
  )$data
  original <- benchmark$invariant_elastic_matrix(data, design)
  permuted_data <- data
  permuted_design <- design
  attribute <- design$factor_names[[1L]]
  reversed <- rev(levels(data[[attribute]]))
  permuted_data[[attribute]] <- factor(
    as.character(data[[attribute]]), levels = reversed
  )
  permuted_design$target[[attribute]] <- factor(
    as.character(design$target[[attribute]]), levels = reversed
  )
  permuted <- benchmark$invariant_elastic_matrix(permuted_data, permuted_design)
  testthat::expect_setequal(colnames(original), colnames(permuted))
  columns <- sort(colnames(original))
  testthat::expect_equal(original[, columns], permuted[, columns])
})

