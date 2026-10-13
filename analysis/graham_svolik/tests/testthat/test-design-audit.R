testthat::test_that("Graham-Svolik narrow Gate-B audit passes without loading an endpoint", {
  repo_root <- Sys.getenv("GS_AUDIT_REPO_ROOT")
  testthat::expect_true(nzchar(repo_root))
  audit_script <- file.path(repo_root, "analysis", "graham_svolik", "audit_design.R")
  output_dir <- tempfile("graham-svolik-design-audit-")
  dir.create(output_dir)
  on.exit(unlink(output_dir, recursive = TRUE, force = TRUE), add = TRUE)

  result <- suppressWarnings(system2(
    "Rscript",
    c(audit_script, "--output-dir", output_dir),
    env = c("OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "VECLIB_MAXIMUM_THREADS=1"),
    stdout = TRUE,
    stderr = TRUE
  ))
  testthat::expect_null(attr(result, "status"), info = paste(result, collapse = "\n"))
  testthat::expect_match(paste(result, collapse = "\n"),
                         "Graham-Svolik narrow Gate-B audit passed")

  read_output <- function(name) {
    utils::read.csv(file.path(output_dir, name), stringsAsFactors = FALSE,
                    check.names = FALSE)
  }
  archive <- read_output("archive_checksums.csv")
  validation <- read_output("design_validation.csv")
  loaded <- read_output("loaded_columns.csv")
  sample <- read_output("sample_summary.csv")
  target <- read_output("target_weight_diagnostics.csv")
  cells <- read_output("target_cells.csv")
  families <- read_output("family_identification.csv")
  folds <- read_output("respondent_folds.csv")
  features <- read_output("covariate_feature_lock.csv")
  manifest <- yaml::read_yaml(file.path(output_dir, "design_audit_manifest.yml"))
  contract <- yaml::read_yaml(file.path(
    repo_root, "config", "datasets", "graham-svolik.yml"
  ))

  testthat::expect_true(all(archive$pass))
  testthat::expect_equal(nrow(validation), 52L)
  testthat::expect_true(all(validation$pass))
  testthat::expect_identical(manifest$status, "passed")
  testthat::expect_identical(
    manifest$gate_result, "green_for_narrow_coarsened_democracy_families"
  )
  testthat::expect_identical(
    manifest$complete_profile_result, "failed_assignment_law_not_certified"
  )
  testthat::expect_true(manifest$outcome_blind_audit_run)
  testthat::expect_false(manifest$new_outcome_values_loaded_by_audit)
  testthat::expect_false(manifest$outcome_models_fit)
  testthat::expect_true(manifest$pre_audit_protocol_deviation$occurred)

  forbidden <- c(
    unlist(contract$read_barrier$forbidden_experiment_outcomes, use.names = FALSE),
    unlist(contract$read_barrier$forbidden_derived_or_post_choice, use.names = FALSE)
  )
  experiment_allowed <- unlist(
    contract$read_barrier$experiment_allowed_columns, use.names = FALSE
  )
  testthat::expect_false(any(experiment_allowed %in% forbidden))
  testthat::expect_false(any(loaded$column %in% forbidden))
  testthat::expect_false(any(loaded$outcome_loaded))

  testthat::expect_equal(sample$released_respondents, 1682L)
  testthat::expect_equal(sample$released_tasks, 21866L)
  testthat::expect_equal(sample$released_experiment_rows, 43732L)
  testthat::expect_equal(sample$primary_respondents, 1682L)
  testthat::expect_equal(sample$primary_tasks, 18502L)
  testthat::expect_equal(sample$primary_tasks_per_respondent_min, 11L)
  testthat::expect_equal(sample$primary_tasks_per_respondent_max, 11L)

  testthat::expect_equal(nrow(cells), 64L)
  testthat::expect_gt(min(cells$observed_count), 0L)
  testthat::expect_equal(target$supported_target_cells, 64L)
  testthat::expect_equal(target$target_cell_probability, 1 / 64, tolerance = 1e-15)
  testthat::expect_lt(target$maximum_mu_over_q, 2)

  testthat::expect_equal(nrow(families), 3L)
  testthat::expect_true(all(families$identified_and_invariant))
  testthat::expect_equal(families$observed_interaction_rank, rep(7L, 3L))
  testthat::expect_equal(families$minimum_cell_count, c(830L, 750L, 606L))
  invariance_columns <- grep("projector_error$", names(families), value = TRUE)
  testthat::expect_lt(max(as.matrix(families[invariance_columns])), 1e-12)

  testthat::expect_equal(length(unique(folds$id)), 1682L)
  testthat::expect_equal(sort(as.integer(table(folds$outer_fold))),
                         c(336L, 336L, 336L, 337L, 337L))
  testthat::expect_equal(length(unique(features$feature)), 28L)
  testthat::expect_identical(
    contract$protocol$prospective_endpoint_rule$primary, "c_win"
  )
  testthat::expect_identical(
    contract$protocol$prospective_endpoint_rule$held_endpoint, "c_vote"
  )
})
