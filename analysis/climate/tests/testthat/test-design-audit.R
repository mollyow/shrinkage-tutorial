testthat::test_that("climate audit locks the outcome-blind design and split Gate-B decision", {
  output_dir <- tempfile("climate-design-audit-")
  dir.create(output_dir)
  on.exit(unlink(output_dir, recursive = TRUE, force = TRUE), add = TRUE)
  repo_root <- Sys.getenv("CLIMATE_AUDIT_REPO_ROOT")
  testthat::expect_true(nzchar(repo_root))
  audit_script <- file.path(repo_root, "analysis", "climate", "audit_design.R")

  result <- suppressWarnings(system2(
    "Rscript",
    c(audit_script, "--output-dir", output_dir),
    stdout = TRUE,
    stderr = TRUE
  ))
  status <- attr(result, "status")
  if (is.null(status)) status <- 0L
  testthat::expect_equal(status, 0L, info = paste(result, collapse = "\n"))
  if (status != 0L) return(invisible(NULL))

  read_output <- function(name) utils::read.csv(file.path(output_dir, name), check.names = FALSE)
  archive <- read_output("archive_checksums.csv")
  validation <- read_output("design_validation.csv")
  sample <- read_output("sample_summary.csv")
  countries <- read_output("country_counts.csv")
  levels <- read_output("level_lock.csv")
  pair_support <- read_output("all_pair_support_summary.csv")
  assignment <- read_output("assignment_diagnostics.csv")
  opponent <- read_output("opponent_law_diagnostics.csv")
  profile_uniformity <- read_output("profile_uniformity_diagnostics.csv")
  strata <- read_output("profile_probability_strata.csv")
  families <- read_output("family_identification.csv")
  coordinates <- read_output("contrast_coordinates.csv")
  projectors <- read_output("family_projectors.csv")
  loaded <- read_output("loaded_columns.csv")
  folds <- read_output("respondent_folds.csv")
  covariates <- read_output("covariate_audit.csv")
  feature_lock <- read_output("covariate_feature_lock.csv")
  target_weights <- read_output("target_weight_diagnostics.csv")
  manifest <- yaml::read_yaml(file.path(output_dir, "design_audit_manifest.yml"))
  contract <- yaml::read_yaml(file.path(repo_root, "config", "datasets", "climate-agreements.yml"))

  testthat::expect_true(all(archive$pass))
  testthat::expect_true(all(validation$pass))
  testthat::expect_identical(manifest$gate_result, "split_rating_green_paired_choice_amber")
  testthat::expect_identical(manifest$portfolio_eligible_scope,
                             "rating_only_under_mean_no_opponent_and_no_carryover_restrictions")
  testthat::expect_true(manifest$outcome_blind)
  testthat::expect_false(manifest$outcomes_loaded)
  testthat::expect_false(manifest$outcome_models_fit)
  testthat::expect_false(manifest$pair_law$q_times_q_certified)
  testthat::expect_false(any(loaded$outcome_loaded))
  testthat::expect_false(any(loaded$column %in% c("choice_cj", "rating_cj", "incon")))
  testthat::expect_setequal(unlist(contract$read_barrier$forbidden_outcomes, use.names = FALSE),
                            c("choice_cj", "rating_cj", "incon"))
  testthat::expect_false(any(unlist(contract$read_barrier$allowed_columns, use.names = FALSE) %in%
                               c("choice_cj", "rating_cj", "incon")))

  testthat::expect_equal(sample$respondents, 8500L)
  testthat::expect_equal(sample$tasks, 34000L)
  testthat::expect_equal(sample$profile_rows, 68000L)
  testthat::expect_equal(sample$tasks_per_respondent_min, 4L)
  testthat::expect_equal(sample$tasks_per_respondent_max, 4L)
  testthat::expect_equal(sample$profiles_per_task_min, 2L)
  testthat::expect_equal(sample$profiles_per_task_max, 2L)
  testthat::expect_true(sample$all_design_columns_complete)
  testthat::expect_equal(countries$respondents, c(2000L, 2000L, 2000L, 2500L))
  testthat::expect_equal(countries$profile_rows, c(16000L, 16000L, 16000L, 20000L))
  testthat::expect_equal(nrow(levels), 23L)
  testthat::expect_equal(nrow(pair_support), 75L)
  testthat::expect_true(all(pair_support$complete_cartesian))
  testthat::expect_lt(max(abs(assignment$pearson_residual)), 4)

  pair_row <- opponent[opponent$diagnostic == "identical_complete_profile_pairs", ]
  repeat_row <- opponent[
    opponent$diagnostic == "repeated_profile_rows_across_tasks_within_respondent", ]
  testthat::expect_equal(pair_row$observed, 0)
  testthat::expect_equal(pair_row$expected_under_iid, 34000 / 2880, tolerance = 1e-12)
  testthat::expect_equal(pair_row$probability_or_candidate_value,
                         (1 - 1 / 2880)^34000, tolerance = 1e-16)
  testthat::expect_lt(pair_row$probability_or_candidate_value, 1e-5)
  cluster_bound <- opponent[
    opponent$diagnostic == "upper_probability_zero_with_arbitrary_four_task_dependence", ]
  testthat::expect_equal(cluster_bound$probability_or_candidate_value,
                         (1 - 1 / 2880)^8500, tolerance = 1e-15)
  testthat::expect_gt(cluster_bound$probability_or_candidate_value, 0.05)
  testthat::expect_equal(repeat_row$observed, 67)
  testthat::expect_equal(profile_uniformity$profile_support_size, 2880L)
  testthat::expect_equal(profile_uniformity$minimum_profile_count, 9L)
  testthat::expect_equal(profile_uniformity$maximum_profile_count, 44L)
  testthat::expect_gt(profile_uniformity$iid_reference_p_value, 0.01)
  testthat::expect_identical(strata$law,
    c("row_marginal_uniform_product", "candidate_uniform_ordered_distinct_pair"))
  testthat::expect_equal(strata$q_profile, rep(1 / 2880, 2), tolerance = 1e-15)
  testthat::expect_true(is.na(strata$ordered_pair_probability[[1]]))
  testthat::expect_equal(strata$ordered_pair_support_size[[2]], 2880 * 2879)

  testthat::expect_equal(
    setNames(families$observed_interaction_rank, families$family),
    c(
      cost_x_burden_distribution = 12L,
      sanctions_x_monitoring = 9L,
      participation_x_emissions_coverage = 4L
    )
  )
  testthat::expect_true(all(families$identified))
  testthat::expect_gt(min(families$minimum_country_cell_count), 0)
  testthat::expect_gt(min(families$minimum_task_cell_count), 0)
  testthat::expect_lt(max(families$orthonormality_error), 1e-10)
  testthat::expect_lt(max(families$main_effect_orthogonality_error), 1e-10)
  testthat::expect_lt(max(families$projector_idempotence_error), 1e-10)

  # The locked interaction operator is a property of the target-weighted
  # function space. It cannot depend on a baseline, a level order, or a
  # rotation of the coordinate basis.
  family_keys <- unique(coordinates$family)
  set.seed(20260901)
  for (family in family_keys) {
    cc <- coordinates[coordinates$family == family, ]
    pp <- projectors[projectors$family == family, ]
    basis <- xtabs(basis_value ~ cell_index + coordinate, data = cc)
    stored_projector <- xtabs(value ~ row_cell_index + column_cell_index, data = pp)
    n_cells <- nrow(basis)
    reconstructed <- basis %*% t(basis) / n_cells
    testthat::expect_lt(max(abs(reconstructed - stored_projector)), 1e-12)

    rotation <- qr.Q(qr(matrix(stats::rnorm(ncol(basis)^2), ncol(basis), ncol(basis))))
    rotated_projector <- basis %*% rotation %*% t(basis %*% rotation) / n_cells
    testthat::expect_lt(max(abs(rotated_projector - stored_projector)), 1e-12)

    nonsingular_map <- diag(ncol(basis))
    nonsingular_map[row(nonsingular_map) < col(nonsingular_map)] <- 0.15
    reparameterized <- basis %*% nonsingular_map
    reparameterized_gram <- crossprod(reparameterized) / n_cells
    reparameterized_projector <- reparameterized %*%
      solve(reparameterized_gram, t(reparameterized) / n_cells)
    testthat::expect_lt(max(abs(reparameterized_projector - stored_projector)), 1e-12)

    cell_map <- unique(cc[c("cell_index", "a_code", "b_code")])
    cell_map <- cell_map[order(cell_map$cell_index), ]
    reference_main <- stats::model.matrix(~ factor(a_code) + factor(b_code), data = cell_map)
    testthat::expect_lt(max(abs(stored_projector %*% reference_main)), 1e-10)

    permutation <- sample(seq_len(n_cells))
    permuted_basis <- basis[permutation, , drop = FALSE]
    testthat::expect_lt(max(abs(
      permuted_basis %*% t(permuted_basis) / n_cells -
        stored_projector[permutation, permutation]
    )), 1e-12)
  }

  testthat::expect_equal(nrow(folds), 8500L)
  testthat::expect_equal(anyDuplicated(folds$ID), 0L)
  testthat::expect_equal(sort(unique(folds$outer_fold)), 1:5)
  testthat::expect_equal(sort(unique(folds$inner_fold_label)), 1:5)
  fold_strata <- folds$country
  outer_balance <- vapply(split(folds$outer_fold, fold_strata),
                          function(x) diff(range(tabulate(x, nbins = 5))) <= 1L, logical(1))
  testthat::expect_true(all(outer_balance))
  for (held_out in 1:5) {
    coverage <- vapply(split(folds, fold_strata), function(x) {
      length(unique(x$inner_fold_label[x$outer_fold != held_out])) == 5L
    }, logical(1))
    testthat::expect_true(all(coverage))
  }

  testthat::expect_equal(sort(feature_lock$variable), sort(c(
    "country", "age", "female", "education_category",
    "income_category", "support_iec_high_group"
  )))
  primary <- covariates[covariates$classification == "primary_feature", ]
  testthat::expect_equal(nrow(primary), 6L)
  missing_by_feature <- setNames(primary$missing_n, primary$variable)
  testthat::expect_equal(missing_by_feature[c(
    "age", "female", "education_category", "income_category", "support_iec_high_group"
  )], c(age = 0L, female = 0L, education_category = 1L,
         income_category = 301L, support_iec_high_group = 1L))
  testthat::expect_equal(target_weights$target_weight_sum, rep(0.25, 4), tolerance = 1e-12)
  testthat::expect_equal(sum(target_weights$target_weight_sum), 1, tolerance = 1e-12)
})

testthat::test_that("unexpected stale climate audit files fail closed", {
  output_dir <- tempfile("climate-design-audit-stale-")
  dir.create(output_dir)
  on.exit(unlink(output_dir, recursive = TRUE, force = TRUE), add = TRUE)
  file.create(file.path(output_dir, "stale_result.csv"))

  repo_root <- Sys.getenv("CLIMATE_AUDIT_REPO_ROOT")
  audit_script <- file.path(repo_root, "analysis", "climate", "audit_design.R")
  result <- suppressWarnings(system2(
    "Rscript",
    c(audit_script, "--output-dir", output_dir),
    stdout = TRUE,
    stderr = TRUE
  ))
  testthat::expect_false(is.null(attr(result, "status")))
  testthat::expect_match(paste(result, collapse = "\n"), "unexpected generated files")
})
