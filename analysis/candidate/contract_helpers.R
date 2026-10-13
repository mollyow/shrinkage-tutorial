# Shared, fast validation helpers for the prespecified candidate application.
# This file deliberately has no package dependencies beyond base R.

candidate_assert <- function(ok, message) {
  if (!isTRUE(ok)) stop(message, call. = FALSE)
}

candidate_scalar_character <- function(x) {
  is.character(x) && length(x) == 1L && !is.na(x) && nzchar(x)
}

candidate_scalar_number <- function(x) {
  is.numeric(x) && length(x) == 1L && is.finite(x)
}

validate_candidate_contract <- function(cfg) {
  expected_models <- c(
    "additive_wls", "two_way_wls", "saturated_wls", "common_ridge",
    "unrestricted_order_ridge", "monotone_order_ridge"
  )
  expected_families <- c(
    "authority_x_majority", "authority_x_staff",
    "fundraising_x_winning", "fundraising_x_pay"
  )

  candidate_assert(is.list(cfg), "The application lockfile must parse to a YAML mapping.")
  candidate_assert(identical(as.integer(cfg$version), 4L),
                   "estimands.yml must use corrected application contract version 4.")
  candidate_assert(identical(
    cfg$status,
    "developmental_contract_corrected_rebuild_required_before_reuse"
  ), "The candidate application must remain labeled developmental and stale until rebuilt.")
  candidate_assert(candidate_scalar_character(cfg$locked_on),
                   "locked_on must be documented.")
  candidate_assert(candidate_scalar_character(cfg$terminology_corrected_on),
                   "terminology_corrected_on must be documented.")
  candidate_assert(identical(
    cfg$comparison_role,
    "developmental_application_not_confirmatory_evidence"
  ), "The candidate comparison role must remain developmental.")

  expected <- cfg$data$expected
  candidate_assert(candidate_scalar_character(cfg$data$allowed_input) &&
                     identical(cfg$data$allowed_input,
                               "Candidate recruitment/replication/PSRM_replication/data/CleanedData.csv"),
                   "The allowed cleaned-data path changed.")
  candidate_assert(candidate_scalar_character(cfg$data$md5) &&
                     grepl("^[0-9a-f]{32}$", cfg$data$md5),
                   "data.md5 must be a lowercase MD5 checksum.")
  candidate_assert(candidate_scalar_character(cfg$data$forbidden_input_pattern) &&
                     !grepl(cfg$data$forbidden_input_pattern, cfg$data$allowed_input,
                            fixed = TRUE),
                   "The allowed input collides with the forbidden raw-data path.")
  candidate_assert(as.integer(expected$rows) == 24000L,
                   "The locked application requires 24,000 rows.")
  candidate_assert(as.integer(expected$respondents) == 3000L,
                   "The locked application requires 3,000 respondents.")
  candidate_assert(as.integer(expected$tasks_per_respondent) == 8L,
                   "The locked application requires eight tasks per respondent.")
  candidate_assert(as.integer(expected$profiles) == 1440L,
                   "The locked application requires the complete 1,440-profile target.")
  candidate_assert(isTRUE(expected$complete_profiles) &&
                     isTRUE(expected$no_missing_core_values),
                   "Complete target support and nonmissing core values must remain required.")

  candidate_assert(identical(names(cfg$roles),
                             c("respondent", "task", "weight", "outcome")),
                   "The four source roles or their order changed.")
  candidate_assert(all(vapply(cfg$roles, function(x) {
    candidate_scalar_character(x$canonical) && is.character(x$aliases) &&
      length(x$aliases) >= 1L && !anyNA(x$aliases)
  }, logical(1))), "Every source role needs a canonical name and aliases.")

  candidate_assert(length(cfg$attributes) == 7L,
                   "Exactly seven randomized factors must be locked.")
  factor_names <- names(cfg$attributes)
  candidate_assert(length(unique(factor_names)) == 7L,
                   "Locked factor names must be unique.")
  factor_levels <- lapply(cfg$attributes, `[[`, "labels")
  candidate_assert(all(vapply(cfg$attributes, function(x) {
    values <- unlist(x$values, use.names = FALSE)
    is.character(x$aliases) && length(x$aliases) >= 1L && !anyNA(x$aliases) &&
      is.logical(x$ordered) && length(x$ordered) == 1L && !is.na(x$ordered) &&
      is.numeric(values) && all(is.finite(values)) &&
      length(values) == length(x$labels) && !anyDuplicated(values)
  }, logical(1))), "Every attribute needs valid aliases, ordering, values, and labels.")
  candidate_assert(all(vapply(factor_levels, function(x) {
    is.character(x) && length(x) >= 2L && !anyNA(x) && !anyDuplicated(x)
  }, logical(1))), "Each factor needs at least two unique, nonmissing locked levels.")
  profile_count <- prod(vapply(factor_levels, length, integer(1)))
  candidate_assert(profile_count == as.integer(expected$profiles),
                   "The Cartesian product of locked levels must equal 1,440 profiles.")

  prep <- cfg$preparation
  candidate_assert(candidate_scalar_number(prep$numeric_tolerance) &&
                     prep$numeric_tolerance > 0,
                   "preparation.numeric_tolerance must be positive.")
  candidate_assert(identical(prep$assignment, "respondent_level"),
                   "Outer folds must be assigned at respondent level.")
  candidate_assert(candidate_scalar_number(prep$fold_seed),
                   "preparation.fold_seed must be a finite scalar.")
  candidate_assert(as.integer(prep$outer_folds) >= 2L,
                   "preparation.outer_folds must be at least two.")
  candidate_assert(isTRUE(prep$require_respondent_invariant_weights),
                   "The lockfile must require respondent-invariant survey weights.")
  candidate_assert(candidate_scalar_number(prep$weight_invariance_tolerance) &&
                     prep$weight_invariance_tolerance >= 0,
                   "preparation.weight_invariance_tolerance must be nonnegative.")

  target <- cfg$target
  candidate_assert(identical(target$population, "complete_factorial_profile_grid"),
                   "The target population must be the uniform full factorial.")
  candidate_assert(identical(target$weighting, "uniform"),
                   "Target weights must be uniform over profiles.")
  candidate_assert(as.integer(target$profiles) == profile_count,
                   "target.profiles must equal the locked Cartesian product.")
  max_order <- as.integer(target$max_interaction_order)
  candidate_assert(length(max_order) == 1L && max_order >= 1L &&
                     max_order <= length(factor_names),
                   "target.max_interaction_order is outside the valid factor range.")

  candidate_assert(identical(cfg$evaluation$fold_unit, "respondent_cluster"),
                   "Evaluation must use respondent clusters.")
  candidate_assert(identical(cfg$evaluation$row_level, "survey_weighted"),
                   "Evaluation loss must be survey weighted.")
  candidate_assert(identical(cfg$evaluation$profile_level,
                             "uniform_over_complete_target_grid"),
                   "Profile evaluation must target the full uniform factorial.")
  candidate_assert(isTRUE(cfg$evaluation$require_complete_profile_target),
                   "Profile evaluation must require the complete target.")
  candidate_assert(identical(cfg$evaluation$incomplete_target_action,
                             "primary_metrics_na_report_common_support"),
                   "Incomplete profile targets must yield NA primary metrics and explicit common-support metrics.")
  candidate_assert(all(c("rmse", "mae", "r_squared", "calibration_intercept",
                         "calibration_slope", "prediction_coverage",
                         "prediction_weight_coverage") %in% cfg$evaluation$metrics),
                   "The locked evaluation metrics are incomplete.")

  models <- unlist(cfg$models, use.names = FALSE)
  candidate_assert(identical(models, expected_models),
                   "The six-model comparison set or its order changed.")

  tuning <- cfg$comparison_tuning
  candidate_assert(identical(tuning$assignment, "respondent_level"),
                   "Inner folds must be assigned at respondent level.")
  candidate_assert(isTRUE(tuning$shared_within_outer_fold),
                   "Ridge comparators must share the same inner fold_id.")
  candidate_assert(identical(tuning$selection_rule, "minimum_cv"),
                   "The tuning rule must be minimum CV error.")
  candidate_assert(identical(tuning$loss, "survey_weighted_mean_squared_error"),
                   "Tuning loss must be survey-weighted squared error.")
  candidate_assert(identical(tuning$task_position, "unpenalized"),
                   "Task-position adjustment must be unpenalized.")
  candidate_assert(as.integer(tuning$inner_folds) >= 2L,
                   "tuning.inner_folds must be at least two.")
  controls <- tuning$control
  for (nm in c("common_grid_size", "maxit", "reltol", "log_lambda_lower",
               "log_lambda_upper")) {
    candidate_assert(candidate_scalar_number(controls[[nm]]),
                     sprintf("tuning.production_controls.%s must be finite.", nm))
  }
  candidate_assert(controls$common_grid_size >= 2 && controls$maxit >= 1 &&
                     controls$reltol > 0 &&
                     controls$log_lambda_lower < controls$log_lambda_upper,
                   "The locked tuning control ranges are invalid.")

  candidate_assert(identical(cfg$saturated_wls$specification,
                             "profile_fixed_effects_plus_task_position_fixed_effects"),
                   "The saturated benchmark specification changed.")
  candidate_assert(identical(cfg$saturated_wls$unseen_outer_training_profile_prediction,
                             "NA"),
                   "Absent outer-training profiles must produce NA predictions.")

  candidate_assert(identical(cfg$ranking$model, "monotone_order_ridge"),
                   "Only monotone_order_ridge may produce profile rankings.")
  candidate_assert(isFALSE(cfg$ranking$other_model_rankings),
                   "Rankings for other models must remain disabled.")
  candidate_assert(identical(cfg$ranking$task_position_target,
                             "uniform_average_over_eight_tasks"),
                   "Ranking must average uniformly over observed task positions.")

  inf <- cfg$inference
  candidate_assert(candidate_scalar_number(inf$alpha) && inf$alpha > 0 && inf$alpha < 1,
                   "inference.alpha must lie strictly between zero and one.")
  candidate_assert(identical(inf$simultaneous_method, "gaussian_max_t"),
                   "The simultaneous-inference method must remain gaussian_max_t.")
  candidate_assert(as.integer(inf$simultaneous_draws) >= 1000L,
                   "inference.simultaneous_draws must be at least 1,000.")
  candidate_assert(candidate_scalar_number(inf$simultaneous_seed),
                   "inference.simultaneous_seed must be locked.")
  candidate_assert(candidate_scalar_number(inf$crossfit_seed),
                   "inference.crossfit_seed must be locked.")
  candidate_assert(candidate_scalar_number(inf$family_simulation_seed_offset) &&
                     inf$family_simulation_seed_offset >= 1,
                   "inference.family_simulation_seed_offset must be a positive scalar.")
  candidate_assert(candidate_scalar_number(inf$cell_simulation_seed_offset) &&
                     inf$cell_simulation_seed_offset >= 1 &&
                     inf$cell_simulation_seed_offset != inf$family_simulation_seed_offset,
                   "inference.cell_simulation_seed_offset must be positive and distinct.")
  candidate_assert(identical(inf$coordinate_basis, "full_rank_orthonormal"),
                   "The AMIE coordinate basis changed.")
  candidate_assert(identical(inf$assignment_probability,
                             "locked_row_marginal_profile_probability"),
                   "AMIE assignment probability must come from the acknowledged rowwise marginal lock.")
  candidate_assert(identical(inf$global_interval_scope, "all_prespecified_coordinates"),
                   "Global simultaneous intervals must cover all prespecified coordinates.")

  analysis_seed <- cfg$analysis$seed
  candidate_assert(candidate_scalar_number(analysis_seed),
                   "analysis.seed must be locked.")

  assignment <- cfg$assignment
  candidate_assert(isTRUE(assignment$assumption_acknowledged),
                   paste0("The rowwise uniform assignment assumption is not acknowledged. ",
                          "Review estimands.yml assignment.design_source and set ",
                          "assignment.assumption_acknowledged: true only after accepting the limitation."))
  candidate_assert(identical(assignment$mechanism,
                             "independent_uniform_within_profile_attributes_rowwise_uniform_profile_marginal_across_tasks_unknown"),
                   "The locked assignment mechanism changed.")
  candidate_assert(identical(assignment$verification_status,
                             "within_profile_documented_by_article_preregistration_and_replication_code_across_tasks_not_verified_and_non_iid_repeat_pattern_observed"),
                   "The assignment verification status must state its evidentiary limitation.")
  candidate_assert(identical(assignment$across_tasks,
                             "unknown_non_iid_no_repeated_complete_profiles"),
                   "The unknown across-task mechanism must remain explicit.")
  q <- assignment$row_marginal_profile_probability
  candidate_assert(candidate_scalar_number(q) && q > 0 && q <= 1,
                   "assignment.row_marginal_profile_probability must be in (0, 1].")
  candidate_assert(abs(q - 1 / profile_count) <= 1e-15,
                   "The locked rowwise marginal assignment probability must equal 1/1,440.")
  for (nm in c("study_citation", "local_randomization_evidence",
               "download_instructions", "license_status", "limitation")) {
    candidate_assert(candidate_scalar_character(assignment$design_source[[nm]]),
                     sprintf("assignment.design_source.%s must be documented.", nm))
  }

  families <- vapply(cfg$families, `[[`, character(1), "id")
  candidate_assert(identical(families, expected_families),
                   "The four prespecified AMIE families or their order changed.")
  candidate_assert(all(vapply(cfg$families, function(x) {
    length(x$attributes) == 2L && all(x$attributes %in% factor_names) &&
      identical(x$estimand, "weighted_functional_anova_two_way_interaction_surface") &&
      identical(x$coordinate_reduction, "orthonormal_full_rank")
  }, logical(1))), "A prespecified AMIE family contract changed.")
  candidate_assert(all(vapply(cfg$families, function(x) {
    candidate_scalar_character(x$label) && isTRUE(x$prespecified) &&
      is.logical(x$report_classic_did) && length(x$report_classic_did) == 1L
  }, logical(1))), "Every AMIE family must remain labeled and prespecified.")
  candidate_assert(all(vapply(cfg$families[1:2], function(x) {
    isTRUE(x$report_classic_did) && identical(
      x$classic_did_scale,
      "mu11 - mu10 - mu01 + mu00, averaged over all other attributes"
    )
  }, logical(1))), "The binary-by-binary classic DID contrast changed.")
  candidate_assert(all(!vapply(cfg$families[3:4], `[[`, logical(1),
                                "report_classic_did")),
                   "Classic DID may only be reported for the two binary families.")

  estimand <- cfg$estimand_definition
  candidate_assert(identical(estimand$name,
                             "uniform_target_functional_anova_interaction") &&
                     candidate_scalar_character(estimand$formula) &&
                     isTRUE(estimand$direct_cell_weights) &&
                     isTRUE(estimand$no_implicit_rescaling) &&
                     isTRUE(estimand$redundant_cells_retained_for_interpretation) &&
                     isTRUE(estimand$inference_on_full_rank_coordinates),
                   "The locked AMIE estimand definition changed.")
  candidate_assert(identical(
    cfg$selection_rule,
    "No interaction family, cell, coordinate, sign, or contrast may be added, removed, or selected after inspecting the outcome."
  ), "The no-post-outcome-selection rule changed.")

  coordinate_dimension <- sum(vapply(cfg$families, function(x) {
    pair <- x$attributes
    (length(factor_levels[[pair[[1L]]]]) - 1L) *
      (length(factor_levels[[pair[[2L]]]]) - 1L)
  }, integer(1)))
  candidate_assert(coordinate_dimension == 22L,
                   "The locked AMIE families must span exactly 22 coordinates.")

  execution <- cfg$execution
  candidate_assert(isTRUE(execution$resume_outer_folds),
                   "Outer-fold checkpoint/resume must remain enabled.")
  candidate_assert(candidate_scalar_character(execution$checkpoint_subdirectory),
                   "execution.checkpoint_subdirectory must be set.")
  candidate_assert(isTRUE(execution$require_local_package_source),
                   "Execution must require the local package source.")

  list(
    factor_names = factor_names,
    factor_levels = factor_levels,
    profile_count = profile_count,
    rows = as.integer(expected$rows),
    respondents = as.integer(expected$respondents),
    tasks_per_respondent = as.integer(expected$tasks_per_respondent),
    coordinate_dimension = coordinate_dimension,
    models = models,
    max_order = max_order,
    outer_folds = as.integer(prep$outer_folds),
    inner_folds = as.integer(tuning$inner_folds),
    analysis_seed = as.integer(analysis_seed),
    inference_crossfit_seed = as.integer(inf$crossfit_seed),
    simultaneous_seed = as.integer(inf$simultaneous_seed),
    family_seed_offset = as.integer(inf$family_simulation_seed_offset),
    cell_seed_offset = as.integer(inf$cell_simulation_seed_offset),
    assignment_probability = as.numeric(q),
    ranking_model = cfg$ranking$model,
    ranking_task_rule = cfg$ranking$task_position_target,
    weight_tolerance = as.numeric(prep$weight_invariance_tolerance),
    checkpoint_subdirectory = execution$checkpoint_subdirectory,
    resume_outer_folds = isTRUE(execution$resume_outer_folds)
  )
}

respondent_weight_diagnostics <- function(id, weight, tolerance = 1e-12) {
  candidate_assert(length(id) == length(weight), "id and weight lengths differ.")
  candidate_assert(!anyNA(id), "Respondent ids may not be missing.")
  candidate_assert(all(is.finite(weight) & weight > 0),
                   "Survey weights must be finite and strictly positive.")
  ranges <- vapply(split(as.numeric(weight), id), function(z) diff(range(z)), numeric(1))
  data.frame(
    respondents = length(ranges),
    maximum_within_respondent_range = max(ranges),
    tolerance = tolerance,
    invariant = all(ranges <= tolerance),
    stringsAsFactors = FALSE
  )
}

assert_respondent_invariant_weights <- function(id, weight, tolerance = 1e-12) {
  out <- respondent_weight_diagnostics(id, weight, tolerance)
  candidate_assert(out$invariant,
                   sprintf(paste0("Survey weights vary within respondent (maximum range %.6g; ",
                                  "tolerance %.6g). The respondent-cluster estimand requires one weight per respondent."),
                           out$maximum_within_respondent_range, tolerance))
  invisible(out)
}

candidate_code_manifest <- function(repo_root) {
  files <- c(file.path(repo_root, "DESCRIPTION"), file.path(repo_root, "NAMESPACE"),
             sort(list.files(file.path(repo_root, "R"), pattern = "\\.[Rr]$", full.names = TRUE)))
  candidate_assert(length(files) >= 3L && all(file.exists(files)),
                   "A complete local package source (DESCRIPTION, NAMESPACE, and R/*.R) is required.")
  package_name <- read.dcf(file.path(repo_root, "DESCRIPTION"), fields = "Package")[[1L]]
  candidate_assert(identical(package_name, "orderridge"),
                   "The local DESCRIPTION must declare Package: orderridge.")
  checksums <- unname(tools::md5sum(files))
  rel <- substring(normalizePath(files, winslash = "/"),
                   nchar(normalizePath(repo_root, winslash = "/")) + 2L)
  checksum_lines <- paste(rel, checksums, sep = "\t")
  checksum_file <- tempfile("candidate-code-checksum-", fileext = ".txt")
  on.exit(unlink(checksum_file), add = TRUE)
  writeLines(checksum_lines, checksum_file, useBytes = TRUE)
  list(
    package = package_name,
    aggregate_md5 = unname(tools::md5sum(checksum_file)),
    files = data.frame(file = rel, md5 = checksums, stringsAsFactors = FALSE)
  )
}

complete_profile_predictions <- function(aggregated, models, folds, profile_ids,
                                         target_weights = NULL) {
  required <- c("model", "fold", "profile_id", "observed", "prediction", "weight", "rows")
  candidate_assert(all(required %in% names(aggregated)),
                   "Aggregated profile predictions are missing required columns.")
  if (is.null(target_weights)) target_weights <- rep(1 / length(profile_ids), length(profile_ids))
  candidate_assert(length(target_weights) == length(profile_ids) &&
                     all(is.finite(target_weights) & target_weights >= 0) &&
                     abs(sum(target_weights) - 1) <= 1e-10,
                   "Target profile weights must be finite, nonnegative, and sum to one.")

  grid <- expand.grid(
    model = models,
    fold = as.character(folds),
    profile_id = as.character(profile_ids),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  incoming_key <- paste(aggregated$model, as.character(aggregated$fold),
                        as.character(aggregated$profile_id), sep = "\r")
  candidate_assert(!anyDuplicated(incoming_key),
                   "Aggregated predictions contain duplicate model/fold/profile rows.")
  grid_key <- paste(grid$model, grid$fold, grid$profile_id, sep = "\r")
  idx <- match(grid_key, incoming_key)
  for (nm in c("observed", "prediction", "weight", "rows")) grid[[nm]] <- aggregated[[nm]][idx]
  grid$rows[is.na(grid$rows)] <- 0L
  grid$weight[is.na(grid$weight)] <- 0
  grid$target_weight <- target_weights[match(grid$profile_id, as.character(profile_ids))]

  support_key <- paste(grid$fold, grid$profile_id, sep = "\r")
  support_ok <- is.finite(grid$observed) & is.finite(grid$prediction)
  common_by_key <- vapply(split(support_ok, support_key), all, logical(1))
  grid$common_support <- unname(common_by_key[support_key])
  grid
}

candidate_metric_values <- function(y, pred, weights) {
  weights <- weights / sum(weights)
  err <- y - pred
  ybar <- sum(weights * y)
  sst <- sum(weights * (y - ybar)^2)
  fit <- stats::lm.wfit(cbind(1, pred), y, weights)
  co <- fit$coefficients
  c(
    rmse = sqrt(sum(weights * err^2)),
    mae = sum(weights * abs(err)),
    weighted_r_squared = if (sst > 0) 1 - sum(weights * err^2) / sst else NA_real_,
    calibration_intercept = unname(co[1L]),
    calibration_slope = unname(co[2L])
  )
}

profile_metric_table_complete <- function(complete_predictions) {
  required <- c("model", "fold", "profile_id", "observed", "prediction",
                "target_weight", "common_support")
  candidate_assert(all(required %in% names(complete_predictions)),
                   "Complete profile predictions are missing required columns.")
  groups <- split(seq_len(nrow(complete_predictions)),
                  paste(complete_predictions$model, complete_predictions$fold, sep = "\r"))
  rows <- lapply(groups, function(ii) {
    z <- complete_predictions[ii, , drop = FALSE]
    support <- z$common_support & is.finite(z$observed) & is.finite(z$prediction)
    coverage <- sum(z$target_weight[support]) / sum(z$target_weight)
    vals <- if (any(support)) candidate_metric_values(
      z$observed[support], z$prediction[support], z$target_weight[support]
    ) else rep(NA_real_, 5L)
    complete <- isTRUE(all(support))
    data.frame(
      scope = if (identical(unique(z$fold), "pooled")) "profile_pooled" else "profile_fold",
      fold = unique(z$fold),
      model = unique(z$model),
      rmse = if (complete) vals[["rmse"]] else NA_real_,
      mae = if (complete) vals[["mae"]] else NA_real_,
      r_squared = if (complete) vals[["weighted_r_squared"]] else NA_real_,
      calibration_intercept = if (complete) vals[["calibration_intercept"]] else NA_real_,
      calibration_slope = if (complete) vals[["calibration_slope"]] else NA_real_,
      target_profile_coverage = coverage,
      prediction_coverage = coverage,
      prediction_weight_coverage = coverage,
      common_support_profiles = sum(support),
      target_profiles = nrow(z),
      n = sum(support),
      n_total = nrow(z),
      common_support_rmse = vals[["rmse"]],
      common_support_mae = vals[["mae"]],
      common_support_r_squared = vals[["weighted_r_squared"]],
      common_support_calibration_intercept = vals[["calibration_intercept"]],
      common_support_calibration_slope = vals[["calibration_slope"]],
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}
