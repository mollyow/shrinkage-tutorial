testthat::test_that("income-tax Gate B audit passes without loading an outcome", {
  repo_root <- Sys.getenv("TAX_AUDIT_REPO_ROOT")
  testthat::expect_true(nzchar(repo_root))
  audit_script <- file.path(repo_root, "analysis", "tax", "audit_design.R")
  output_dir <- tempfile("tax-design-audit-")
  dir.create(output_dir)
  on.exit(unlink(output_dir, recursive = TRUE, force = TRUE), add = TRUE)

  result <- suppressWarnings(system2(
    "Rscript",
    c(audit_script, "--output-dir", output_dir),
    stdout = TRUE,
    stderr = TRUE
  ))
  testthat::expect_null(attr(result, "status"), info = paste(result, collapse = "\n"))
  testthat::expect_match(paste(result, collapse = "\n"),
                         "Tax design audit passed outcome-blind")

  read_output <- function(name) {
    utils::read.csv(file.path(output_dir, name), stringsAsFactors = FALSE,
                    check.names = FALSE)
  }
  archive <- read_output("archive_checksums.csv")
  validation <- read_output("design_validation.csv")
  loaded <- read_output("loaded_columns.csv")
  sample <- read_output("sample_summary.csv")
  levels <- read_output("level_lock.csv")
  pair_support <- read_output("all_pair_support_summary.csv")
  assignment <- read_output("assignment_diagnostics.csv")
  profile_uniformity <- read_output("profile_uniformity_diagnostics.csv")
  opponent <- read_output("opponent_law_diagnostics.csv")
  probabilities <- read_output("profile_probability_strata.csv")
  revenue <- read_output("revenue_descendant_diagnostics.csv")
  families <- read_output("family_identification.csv")
  coordinates <- read_output("contrast_coordinates.csv")
  projectors <- read_output("family_projectors.csv")
  folds <- read_output("respondent_folds.csv")
  covariates <- read_output("covariate_audit.csv")
  feature_lock <- read_output("covariate_feature_lock.csv")
  target_weights <- read_output("target_weight_diagnostics.csv")
  manifest <- yaml::read_yaml(file.path(output_dir, "design_audit_manifest.yml"))
  contract <- yaml::read_yaml(file.path(
    repo_root, "config", "datasets", "income-tax-plans.yml"
  ))

  testthat::expect_true(all(archive$pass))
  testthat::expect_true(all(validation$pass))
  testthat::expect_identical(manifest$gate_result,
                             "green_for_narrow_total_schedule_effects")
  testthat::expect_identical(manifest$portfolio_eligible_scope,
                             "six_randomized_rate_total_effects_only")
  testthat::expect_true(manifest$outcome_blind)
  testthat::expect_false(manifest$outcomes_loaded)
  testthat::expect_false(manifest$outcome_models_fit)
  testthat::expect_true(manifest$assignment_law$q_times_q_locked)
  testthat::expect_true(manifest$assignment_law$source_supported_reconstruction)
  testthat::expect_false(manifest$assignment_law$original_generator_released)
  testthat::expect_identical(manifest$revenue$primary_nuisance_action,
                             "do_not_condition_on_taxrev")
  testthat::expect_false(any(loaded$outcome_loaded))

  forbidden <- c(
    unlist(contract$read_barrier$forbidden_outcomes, use.names = FALSE),
    unlist(contract$read_barrier$forbidden_post_choice_or_outcome_adjacent,
           use.names = FALSE)
  )
  allowed <- unlist(contract$read_barrier$allowed_columns, use.names = FALSE)
  canonical_choice <- c(paste0("pref_plan_", 1:8), "pref_plan", "chose_plan")
  testthat::expect_true(all(canonical_choice %in% forbidden))
  testthat::expect_false(any(allowed %in% forbidden))
  testthat::expect_setequal(loaded$column, allowed)
  testthat::expect_false(any(loaded$column %in% forbidden))

  testthat::expect_equal(sample$archive_respondents, 2250L)
  testthat::expect_equal(sample$archive_tasks, 18000L)
  testthat::expect_equal(sample$archive_profile_rows, 36000L)
  testthat::expect_equal(sample$primary_respondents, 2000L)
  testthat::expect_equal(sample$primary_tasks, 16000L)
  testthat::expect_equal(sample$primary_profile_rows, 32000L)
  testthat::expect_equal(sample$held_no_revenue_respondents, 250L)
  testthat::expect_equal(sample$held_no_revenue_tasks, 2000L)
  testthat::expect_equal(sample$held_no_revenue_profile_rows, 4000L)
  testthat::expect_equal(sample$tasks_per_primary_respondent_min, 8L)
  testthat::expect_equal(sample$tasks_per_primary_respondent_max, 8L)
  testthat::expect_equal(sample$profiles_per_task_min, 2L)
  testthat::expect_equal(sample$profiles_per_task_max, 2L)
  testthat::expect_true(sample$all_primary_design_columns_complete)

  randomized_levels <- levels[levels$component_type == "randomized_rate_factor", ]
  revenue_levels <- levels[levels$component_type == "randomized_downstream_mediator", ]
  testthat::expect_equal(nrow(randomized_levels), 27L)
  testthat::expect_equal(nrow(revenue_levels), 5L)
  testthat::expect_identical(randomized_levels$contract_label,
                             randomized_levels$source_label)
  testthat::expect_identical(revenue_levels$contract_label,
                             revenue_levels$source_label)
  testthat::expect_identical(
    loaded$role[loaded$column == "saw_revenue"],
    "author_defined_experiment_version"
  )
  testthat::expect_equal(
    as.integer(table(randomized_levels$component)),
    c(6L, 4L, 4L, 4L, 5L, 4L)
  )
  testthat::expect_equal(nrow(pair_support), 15L * 11L)
  testthat::expect_true(all(pair_support$complete_cartesian))
  testthat::expect_gt(min(pair_support$minimum_cell_count), 0)
  testthat::expect_equal(nrow(assignment), 27L * 11L)
  testthat::expect_lt(max(abs(assignment$pearson_residual)), 3)

  testthat::expect_equal(profile_uniformity$schedule_support_size, 7680L)
  testthat::expect_equal(profile_uniformity$observed_distinct_schedules, 7554L)
  testthat::expect_equal(profile_uniformity$unobserved_schedules, 126L)
  testthat::expect_equal(profile_uniformity$minimum_count_including_zeros, 0L)
  testthat::expect_equal(profile_uniformity$maximum_count, 13L)
  testthat::expect_equal(profile_uniformity$expected_occupied_under_iid_rows,
                         7680 * (1 - (1 - 1 / 7680)^32000), tolerance = 1e-10)
  occupancy_probability <- 1 - (1 - 1 / 7680)^32000
  both_occupied_probability <- 1 - 2 * (1 - 1 / 7680)^32000 +
    (1 - 2 / 7680)^32000
  occupancy_variance <- 7680 * occupancy_probability * (1 - occupancy_probability) +
    7680 * 7679 * (both_occupied_probability - occupancy_probability^2)
  testthat::expect_equal(
    profile_uniformity$occupied_standard_deviation_under_iid_rows,
    sqrt(occupancy_variance), tolerance = 1e-10
  )
  testthat::expect_equal(
    profile_uniformity$occupied_z_score,
    (7554 - 7680 * occupancy_probability) / sqrt(occupancy_variance),
    tolerance = 1e-10
  )
  # The high-dimensional chi-square is retained only as a descriptive number;
  # its 4.17 expected observations per cell make its asymptotic p-value an
  # unsuitable pass/fail gate.
  testthat::expect_true(is.finite(profile_uniformity$iid_row_reference_p_value))

  identical_row <- opponent[
    opponent$diagnostic == "identical_complete_schedule_pairs", , drop = FALSE
  ]
  repeat_row <- opponent[
    opponent$diagnostic == "repeated_schedule_rows_across_tasks_within_respondent",
    , drop = FALSE
  ]
  cross_side <- opponent[opponent$diagnostic == "cross_side_level_cell", , drop = FALSE]
  factor_match <- opponent[
    opponent$diagnostic == "same_factor_level_across_sides", , drop = FALSE
  ]
  testthat::expect_equal(nrow(identical_row), 1L)
  testthat::expect_equal(identical_row$observed, 0)
  testthat::expect_equal(identical_row$expected_under_q_times_q,
                         16000 / 7680, tolerance = 1e-12)
  testthat::expect_equal(identical_row$probability_or_candidate_value,
                         (1 - 1 / 7680)^16000, tolerance = 1e-12)
  testthat::expect_gt(identical_row$probability_or_candidate_value, 0.05)
  testthat::expect_equal(repeat_row$observed, 32)
  testthat::expect_equal(nrow(cross_side), sum(c(4, 4, 4, 4, 5, 6))^2)
  testthat::expect_lt(max(abs(cross_side$pearson_residual)), 3)
  testthat::expect_equal(nrow(factor_match), 6L)
  testthat::expect_lt(max(abs(factor_match$pearson_residual)), 1)
  testthat::expect_identical(
    probabilities$law,
    c("focal_uniform_product_q", "within_task_ordered_pair_q_times_q")
  )
  testthat::expect_lt(max(abs(probabilities$q_schedule - 1 / 7680)), 1e-15)
  testthat::expect_true(is.na(probabilities$ordered_pair_probability[[1]]))
  testthat::expect_lt(abs(probabilities$ordered_pair_probability[[2]] -
                           1 / 7680^2), 1e-20)
  testthat::expect_equal(probabilities$ordered_pair_support_size[[2]], 7680^2)
  testthat::expect_equal(probabilities$structural_zeros, c(0L, 0L))

  revenue_values <- setNames(revenue$observed, revenue$diagnostic)
  testthat::expect_equal(revenue_values[["observed_revenue_categories"]], 5)
  testthat::expect_equal(revenue_values[["revenue_missing_rows"]], 0)
  testthat::expect_equal(revenue_values[["observed_distinct_rate_schedules"]], 7554)
  testthat::expect_equal(revenue_values[["rate_schedules_observed_more_than_once"]], 7080)
  testthat::expect_equal(
    revenue_values[["repeated_rate_schedules_with_multiple_revenue_categories"]], 434
  )
  testthat::expect_equal(
    revenue_values[["rows_in_rate_schedules_with_multiple_revenue_categories"]], 2102
  )
  testthat::expect_equal(
    revenue_values[["maximum_revenue_categories_within_one_rate_schedule"]], 2
  )
  testthat::expect_true(all(revenue$pass))

  testthat::expect_equal(
    setNames(families$observed_interaction_rank, families$family),
    c(lowest_x_highest = 15L, upper_middle_x_highest = 15L,
      top_adjacent_brackets = 20L)
  )
  testthat::expect_true(all(families$identified))
  testthat::expect_gt(min(families$minimum_task_cell_count), 0)
  testthat::expect_gt(min(families$minimum_side_cell_count), 0)
  testthat::expect_lt(max(families$orthonormality_error), 1e-10)
  testthat::expect_lt(max(families$main_effect_orthogonality_error), 1e-10)
  testthat::expect_lt(max(families$projector_idempotence_error), 1e-10)

  # The locked whole-family interaction operator cannot depend on a baseline,
  # a level order, a rotated coordinate basis, or another nonsingular coding.
  set.seed(20260901)
  for (family in unique(coordinates$family)) {
    cc <- coordinates[coordinates$family == family, ]
    pp <- projectors[projectors$family == family, ]
    basis <- xtabs(basis_value ~ cell_index + coordinate, data = cc)
    stored_projector <- xtabs(value ~ row_cell_index + column_cell_index, data = pp)
    n_cells <- nrow(basis)
    reconstructed <- basis %*% t(basis) / n_cells
    testthat::expect_lt(max(abs(reconstructed - stored_projector)), 1e-12)

    rotation <- qr.Q(qr(matrix(stats::rnorm(ncol(basis)^2),
                               ncol(basis), ncol(basis))))
    rotated <- basis %*% rotation
    testthat::expect_lt(max(abs(
      rotated %*% t(rotated) / n_cells - stored_projector
    )), 1e-12)

    nonsingular_map <- diag(ncol(basis))
    nonsingular_map[row(nonsingular_map) < col(nonsingular_map)] <- 0.15
    reparameterized <- basis %*% nonsingular_map
    reparameterized_gram <- crossprod(reparameterized) / n_cells
    reparameterized_projector <- reparameterized %*%
      solve(reparameterized_gram, t(reparameterized) / n_cells)
    testthat::expect_lt(max(abs(
      reparameterized_projector - stored_projector
    )), 1e-12)

    cell_map <- unique(cc[c("cell_index", "a_code", "b_code")])
    cell_map <- cell_map[order(cell_map$cell_index), ]
    reference_main <- stats::model.matrix(
      ~ factor(a_code) + factor(b_code), data = cell_map
    )
    testthat::expect_lt(max(abs(stored_projector %*% reference_main)), 1e-10)

    permutation <- sample(seq_len(n_cells))
    permuted_basis <- basis[permutation, , drop = FALSE]
    testthat::expect_lt(max(abs(
      permuted_basis %*% t(permuted_basis) / n_cells -
        stored_projector[permutation, permutation]
    )), 1e-12)
  }

  testthat::expect_equal(nrow(folds), 2000L)
  testthat::expect_equal(anyDuplicated(folds$ID), 0L)
  testthat::expect_equal(sort(unique(folds$weight_decile)), 1:10)
  testthat::expect_equal(sort(unique(folds$outer_fold)), 1:5)
  testthat::expect_equal(sort(unique(folds$inner_fold_label)), 1:5)
  outer_balance <- vapply(split(folds$outer_fold, folds$weight_decile),
                          function(x) diff(range(tabulate(x, nbins = 5))) <= 1L,
                          logical(1))
  inner_balance <- vapply(split(folds$inner_fold_label, folds$weight_decile),
                          function(x) diff(range(tabulate(x, nbins = 5))) <= 1L,
                          logical(1))
  testthat::expect_true(all(outer_balance))
  testthat::expect_true(all(inner_balance))
  for (held_out in 1:5) {
    coverage <- vapply(split(folds, folds$weight_decile), function(x) {
      length(unique(x$inner_fold_label[x$outer_fold != held_out])) == 5L
    }, logical(1))
    testthat::expect_true(all(coverage))
  }

  testthat::expect_equal(sort(feature_lock$variable), sort(c(
    "age", "female", "race", "educ", "pid7", "conservative_ideology", "newsint"
  )))
  testthat::expect_equal(nrow(covariates), 7L)
  missing_by_feature <- setNames(covariates$missing_n, covariates$variable)
  testthat::expect_equal(missing_by_feature[c(
    "age", "female", "race", "educ", "pid7", "conservative_ideology", "newsint"
  )], c(age = 0L, female = 0L, race = 0L, educ = 0L, pid7 = 0L,
         conservative_ideology = 182L, newsint = 46L))
  testthat::expect_equal(covariates$minimum[covariates$variable == "age"], 19)
  testthat::expect_equal(covariates$maximum[covariates$variable == "age"], 91)
  testthat::expect_equal(sum(target_weights$respondents), 2000L)
  testthat::expect_equal(sum(target_weights$target_weight_sum), 1,
                         tolerance = 1e-12)
  testthat::expect_equal(sum(folds$target_weight), 1, tolerance = 1e-12)
})

testthat::test_that("unexpected stale tax audit files fail closed", {
  output_dir <- tempfile("tax-design-audit-stale-")
  dir.create(output_dir)
  on.exit(unlink(output_dir, recursive = TRUE, force = TRUE), add = TRUE)
  file.create(file.path(output_dir, "stale_result.csv"))

  repo_root <- Sys.getenv("TAX_AUDIT_REPO_ROOT")
  audit_script <- file.path(repo_root, "analysis", "tax", "audit_design.R")
  result <- suppressWarnings(system2(
    "Rscript",
    c(audit_script, "--output-dir", output_dir),
    stdout = TRUE,
    stderr = TRUE
  ))
  testthat::expect_false(is.null(attr(result, "status")))
  testthat::expect_match(paste(result, collapse = "\n"),
                         "unexpected generated files")
})
