repo_root <- Sys.getenv("CANDIDATE_REPO_ROOT")
stopifnot(nzchar(repo_root))
candidate_dir <- file.path(repo_root, "analysis", "candidate")
source(file.path(candidate_dir, "contract_helpers.R"), local = TRUE)

testthat::test_that("the YAML lock is internally coherent and executable", {
  testthat::skip_if_not_installed("yaml")
  cfg <- yaml::read_yaml(file.path(candidate_dir, "estimands.yml"))
  resolved <- validate_candidate_contract(cfg)

  testthat::expect_identical(resolved$profile_count, 1440)
  testthat::expect_identical(resolved$max_order, 7L)
  testthat::expect_identical(resolved$outer_folds, 5L)
  testthat::expect_identical(resolved$inner_folds, 5L)
  testthat::expect_identical(length(resolved$models), 6L)
  testthat::expect_equal(resolved$assignment_probability, 1 / 1440,
                         tolerance = 1e-15)
  testthat::expect_identical(
    cfg$comparison_role,
    "developmental_application_not_confirmatory_evidence"
  )
  testthat::expect_identical(
    cfg$assignment$across_tasks,
    "unknown_non_iid_no_repeated_complete_profiles"
  )

  changed <- cfg
  changed$assignment$assumption_acknowledged <- FALSE
  testthat::expect_error(validate_candidate_contract(changed),
                         "not acknowledged")
  changed <- cfg
  changed$analysis$seed <- NULL
  testthat::expect_error(validate_candidate_contract(changed), "analysis.seed")
  changed <- cfg
  changed$target$max_interaction_order <- 8
  testthat::expect_error(validate_candidate_contract(changed),
                         "max_interaction_order")
  changed <- cfg
  changed$models <- rev(changed$models)
  testthat::expect_error(validate_candidate_contract(changed),
                         "six-model comparison")
})

testthat::test_that("respondent weights must be invariant within cluster", {
  good <- assert_respondent_invariant_weights(
    id = rep(c("a", "b"), each = 3),
    weight = rep(c(0.7, 1.4), each = 3),
    tolerance = 1e-12
  )
  testthat::expect_true(good$invariant)
  testthat::expect_error(
    assert_respondent_invariant_weights(
      id = rep(c("a", "b"), each = 3),
      weight = c(1, 1, 1.01, 2, 2, 2),
      tolerance = 1e-12
    ),
    "vary within respondent"
  )
})

testthat::test_that("profile evaluation retains the full target and labels common support", {
  aggregated <- data.frame(
    model = c("a", "b", "a", "b"),
    fold = rep("1", 4),
    profile_id = c("p1", "p1", "p2", "p2"),
    observed = c(0.1, 0.1, 0.2, 0.2),
    prediction = c(0.1, 0.15, 0.2, NA_real_),
    weight = 1,
    rows = 1L,
    stringsAsFactors = FALSE
  )
  complete <- complete_profile_predictions(
    aggregated, models = c("a", "b"), folds = "1",
    profile_ids = c("p1", "p2", "p3"), target_weights = rep(1 / 3, 3)
  )
  metrics <- profile_metric_table_complete(complete)

  testthat::expect_equal(nrow(complete), 6L)
  testthat::expect_equal(sum(complete$rows == 0), 2L)
  testthat::expect_true(all(complete$common_support[complete$profile_id == "p1"]))
  testthat::expect_false(any(complete$common_support[complete$profile_id != "p1"]))
  testthat::expect_true(all(is.na(metrics$rmse)))
  testthat::expect_equal(metrics$target_profile_coverage, rep(1 / 3, 2))
  testthat::expect_true(all(is.finite(metrics$common_support_rmse)))
})

testthat::test_that("real-data preparation and API contract are fast-checkable when data exist", {
  testthat::skip_if_not_installed("yaml")
  testthat::skip_if_not_installed("readr")
  testthat::skip_if_not_installed("tibble")
  cleaned <- file.path(
    repo_root, "Candidate recruitment", "replication", "PSRM_replication",
    "data", "CleanedData.csv"
  )
  testthat::skip_if_not(file.exists(cleaned),
                        "Public replication data are not installed in this checkout")

  output_dir <- tempfile("candidate-contract-prep-")
  dir.create(output_dir)
  rscript <- file.path(R.home("bin"), "Rscript")
  prep_log <- system2(
    rscript,
    c(file.path(candidate_dir, "prepare_candidate_data.R"),
      "--output-dir", output_dir),
    stdout = TRUE, stderr = TRUE
  )
  testthat::expect_null(attr(prep_log, "status"), info = paste(prep_log, collapse = "\n"))
  prepared <- readRDS(file.path(output_dir, "candidate_prepared.rds"))
  testthat::expect_equal(nrow(prepared$data), 24000L)
  testthat::expect_equal(length(unique(prepared$data$id)), 3000L)
  testthat::expect_equal(nrow(prepared$target_grid), 1440L)
  testthat::expect_equal(ncol(prepared$contrast_matrix), 22L)
  testthat::expect_true(all(vapply(split(prepared$data$weight, prepared$data$id),
                                   function(z) diff(range(z)) <= 1e-12,
                                   logical(1))))
  testthat::expect_equal(
    max(abs(crossprod(prepared$contrast_matrix) - diag(22))), 0,
    tolerance = 1e-10
  )

  for (family_id in c("authority_x_majority", "authority_x_staff")) {
    family <- prepared$contrast_families[[family_id]]
    a <- family$attributes[[1L]]
    b <- family$attributes[[2L]]
    la <- levels(prepared$target_grid[[a]])
    lb <- levels(prepared$target_grid[[b]])
    n_other <- nrow(prepared$target_grid) / 4
    expected_did <- ifelse(
      (as.character(prepared$target_grid[[a]]) == la[[1L]] &
         as.character(prepared$target_grid[[b]]) == lb[[1L]]) |
        (as.character(prepared$target_grid[[a]]) == la[[2L]] &
           as.character(prepared$target_grid[[b]]) == lb[[2L]]),
      1 / n_other, -1 / n_other
    )
    testthat::expect_equal(unname(family$classic_did), expected_did,
                           tolerance = 1e-12)
  }

  testthat::skip_if_not_installed("pkgload")
  api_log <- system2(
    rscript,
    c(file.path(candidate_dir, "run_candidate_analysis.R"),
      "--prepared", file.path(output_dir, "candidate_prepared.rds"),
      "--api-check-only"),
    stdout = TRUE, stderr = TRUE
  )
  testthat::expect_null(attr(api_log, "status"), info = paste(api_log, collapse = "\n"))
  unlink(output_dir, recursive = TRUE)
})
