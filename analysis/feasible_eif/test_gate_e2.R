#!/usr/bin/env Rscript

script_path <- function() {
  flag <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(flag)) {
    return(normalizePath("analysis/feasible_eif/test_gate_e2.R"))
  }
  normalizePath(sub("^--file=", "", flag[[1L]]), mustWork = TRUE)
}

script <- script_path()
analysis_dir <- dirname(script)
repo_root <- normalizePath(file.path(analysis_dir, "..", ".."), mustWork = TRUE)
result_dir <- file.path(repo_root, "results", "feasible-eif")
result_path <- file.path(result_dir, "gate_e2_correctness_tests.yml")
dir.create(result_dir, recursive = TRUE, showWarnings = FALSE)

source(file.path(repo_root, "analysis", "covariance_headroom",
                 "calibration_solver.R"), local = TRUE)
source(file.path(analysis_dir, "feasible_estimator.R"), local = TRUE)
source(file.path(analysis_dir, "simulation_functions.R"), local = TRUE)
source(file.path(analysis_dir, "sensitivity_functions.R"), local = TRUE)
source(file.path(analysis_dir, "e2_simulation_functions.R"), local = TRUE)

if (!requireNamespace("yaml", quietly = TRUE)) {
  stop("Package `yaml` is required for Gate-E2 correctness tests.", call. = FALSE)
}

config_path <- file.path(analysis_dir, "gate-e2-production.yml")
config <- read_gate_e2_config(config_path)
geometry <- gate_d_geometry(config)

state <- new.env(parent = emptyenv())
state$checks <- list()
state$maximum_error <- 0

record_check <- function(label, error = 0, tolerance = 0, details = NULL) {
  error <- as.numeric(error)
  tolerance <- as.numeric(tolerance)
  if (length(error) != 1L || !is.finite(error) ||
      length(tolerance) != 1L || !is.finite(tolerance) || error > tolerance) {
    stop(sprintf("%s failed: %.8g > %.8g", label, error, tolerance),
         call. = FALSE)
  }
  state$maximum_error <- max(state$maximum_error, error)
  state$checks[[length(state$checks) + 1L]] <- list(
    label = label, error = error, tolerance = tolerance,
    details = details
  )
  cat(sprintf("PASS %-62s %.3e\n", label, error))
  invisible(error)
}

check_close <- function(observed, expected, tolerance = 1e-10,
                        label = "check") {
  observed <- as.numeric(observed)
  expected <- as.numeric(expected)
  error <- if (!length(observed) && !length(expected)) 0 else
    max(abs(observed - expected))
  record_check(label, error, tolerance)
}

check_true <- function(condition, label = "check", details = NULL) {
  ok <- isTRUE(condition)
  record_check(label, if (ok) 0 else 1, 0, details)
}

scenario_by_id <- function(id) {
  ids <- vapply(config$simulation$scenarios, `[[`, character(1), "id")
  hit <- match(id, ids)
  if (is.na(hit)) stop("Unknown frozen Gate-E2 scenario: ", id, call. = FALSE)
  config$simulation$scenarios[[hit]]
}

enumerate_schedules <- function(profile_count, tasks,
                                mechanism = c("product",
                                              "uniform_without_replacement")) {
  mechanism <- match.arg(mechanism)
  schedules <- as.matrix(expand.grid(
    rep(list(seq_len(profile_count)), tasks),
    KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
  ))
  if (identical(mechanism, "uniform_without_replacement")) {
    keep <- apply(schedules, 1L, function(x) !anyDuplicated(x))
    schedules <- schedules[keep, , drop = FALSE]
  }
  schedules
}

enumerated_score_target <- function(
    schedules, operator, ordinary, departure, departure_scale,
    coordinate, true_coefficient, nuisance_coefficient, task_weights,
    riesz_lookup) {
  mixing <- sensitivity_mixing(operator)
  ordinary_mixing <- sensitivity_mixing(ordinary)
  coordinate_index <- match(coordinate, colnames(riesz_lookup))
  score <- target <- numeric(nrow(schedules))
  for (ss in seq_len(nrow(schedules))) {
    schedule <- schedules[ss, ]
    assigned_riesz <- riesz_lookup[schedule, coordinate_index]
    base_mean <- true_coefficient * assigned_riesz
    nuisance_mean <- nuisance_coefficient * assigned_riesz
    departure_feature <- departure$feature_lookup[schedule]
    departure_shift <- departure_scale * drop(
      matrix(departure_feature, nrow = 1L) %*% t(departure$temporal_map)
    )
    complete_mean <- base_mean + departure_shift
    plugin <- sum(task_weights * nuisance_coefficient)
    score[[ss]] <- plugin + sum(
      drop(matrix(complete_mean - nuisance_mean, nrow = 1L) %*% mixing) *
        assigned_riesz
    )
    target[[ss]] <- sum(
      drop(matrix(complete_mean, nrow = 1L) %*% ordinary_mixing) *
        assigned_riesz
    )
  }
  c(score = mean(score), target = mean(target), bias = mean(score - target))
}

run_tests <- function() {
  check_true(config$mode == "production" &&
               config$version == 3L &&
               config$replications_per_scenario == 3000L &&
               length(config$simulation$scenarios) == 14L,
             "frozen production configuration loaded")
  check_true(identical(
    config$reference_design$default_score_mean_learner,
    "task_specific_saturated_16_profile_cell_means"
  ), "correct-law saturated score nuisance is frozen")
  check_true(identical(
    config$reference_design$default_covariance_mean_learner,
    "task_specific_saturated_16_profile_cell_means"
  ), "correct-law saturated covariance nuisance is frozen")

  tasks <- 8L
  task_weights <- rep(1 / tasks, tasks)
  sigma_cs <- matrix(0.8, tasks, tasks)
  diag(sigma_cs) <- 1
  departure_store <- list()

  # Product and without-replacement geometry for every departure class.
  for (mechanism in c("product", "uniform_without_replacement")) {
    support <- if (identical(mechanism, "uniform_without_replacement")) {
      geometry$profile_count
    } else NULL
    assignment_R <- sensitivity_assignment_correlation(
      tasks, mechanism, support
    )
    ordinary <- ordinary_task_operator(
      tasks, task_weights, assignment_R, mechanism, support
    )
    oracle <- solve_task_mixing(
      sigma_cs, task_weights, mechanism, support
    )
    sensitivity <- mixing_sensitivity_K(
      oracle, ordinary, sigma_cs, assignment_R,
      geometry$riesz_lookup, geometry$target_probability
    )
    variance_gap <- mixing_quadratic(
      ordinary$mixing, sigma_cs, assignment_R
    ) - mixing_quadratic(oracle$mixing, sigma_cs, assignment_R)
    check_close(
      sensitivity$K / tasks,
      variance_gap * sensitivity$riesz_gram,
      label = paste("K/T equals variance gap under", mechanism)
    )

    for (type in c("none", "least_favorable", "lag", "blind")) {
      departure <- build_sensitivity_departure(
        type, geometry, tasks, sigma_cs, ordinary, oracle,
        mechanism, support, direction = "x1_x2",
        primary_feature = "x1_x2", blind_feature = "x3_x4"
      )
      departure_store[[paste(mechanism, type, sep = "::")]] <- departure
      expected_norm <- if (identical(type, "none")) 0 else 1
      check_close(
        departure$unit_epsilon, expected_norm,
        label = paste("unit departure norm", mechanism, type)
      )
      check_close(
        departure$diagnostics$orthogonality_residual, 0,
        label = paste("mean-space projection", mechanism, type)
      )
      if (identical(type, "least_favorable")) {
        check_close(
          departure$directional_bias_unit,
          departure$directional_support,
          label = paste("least-favourable support", mechanism)
        )
      }
      if (identical(type, "blind")) {
        check_close(
          departure$oracle_bias_unit,
          rep(0, ncol(geometry$riesz_lookup)),
          label = paste("blind vector bias", mechanism)
        )
        check_close(
          departure$cross_gram,
          rep(0, ncol(geometry$riesz_lookup)),
          label = paste("blind feature orthogonality", mechanism)
        )
      }
    }
  }

  # The triangular-array implementation must make sqrt(G) epsilon invariant.
  scaling_departure <- departure_store[["product::least_favorable"]]
  radius <- 1 / sqrt(tasks)
  profile_64 <- matrix(rep(seq_len(geometry$profile_count), length.out = 64 * tasks),
                       64, tasks)
  profile_256 <- matrix(rep(seq_len(geometry$profile_count),
                            length.out = 256 * tasks), 256, tasks)
  scaled_64 <- evaluate_sensitivity_departure(
    profile_64, scaling_departure, radius, 64L
  )
  scaled_256 <- evaluate_sensitivity_departure(
    profile_256, scaling_departure, radius, 256L
  )
  check_close(scaled_64$sqrt_G_epsilon, radius,
              label = "sqrtG epsilon scaling at G=64")
  check_close(scaled_256$sqrt_G_epsilon, radius,
              label = "sqrtG epsilon scaling at G=256")
  check_close(scaled_64$epsilon / scaled_256$epsilon, 2,
              label = "epsilon shrinks at inverse sqrt G")

  # Output-coordinate rotation must rotate K and preserve the profile Gram.
  set.seed(2026082741)
  rotation <- qr.Q(qr(matrix(rnorm(16), 4, 4)))
  product_R <- sensitivity_assignment_correlation(tasks, "product")
  product_ordinary <- ordinary_task_operator(
    tasks, task_weights, product_R, "product"
  )
  product_oracle <- solve_task_mixing(
    sigma_cs, task_weights, "product"
  )
  base_K <- mixing_sensitivity_K(
    product_oracle, product_ordinary, sigma_cs, product_R,
    geometry$riesz_lookup, geometry$target_probability
  )$K
  rotated_K <- mixing_sensitivity_K(
    product_oracle, product_ordinary, sigma_cs, product_R,
    geometry$riesz_lookup %*% rotation, geometry$target_probability
  )$K
  check_close(rotated_K, crossprod(rotation, base_K %*% rotation),
              label = "output rotation equivariance of K")

  # Exhaustive two-task expectations verify target and score-bias formulas.
  small_tasks <- 2L
  small_weights <- rep(1 / small_tasks, small_tasks)
  sigma_small <- matrix(0.45, small_tasks, small_tasks)
  diag(sigma_small) <- 1
  coordinate <- "x1_x2"
  true_coefficient <- 0.23
  nuisance_coefficient <- c(-0.31, 0.67)
  departure_scale <- 0.07
  for (mechanism in c("product", "uniform_without_replacement")) {
    support <- if (identical(mechanism, "uniform_without_replacement")) {
      geometry$profile_count
    } else NULL
    schedules <- enumerate_schedules(
      geometry$profile_count, small_tasks, mechanism
    )
    true_R <- sensitivity_assignment_correlation(
      small_tasks, mechanism, support
    )
    ordinary <- ordinary_task_operator(
      small_tasks, small_weights, true_R, mechanism, support
    )
    oracle <- solve_task_mixing(
      sigma_small, small_weights, mechanism, support
    )
    for (type in c("none", "least_favorable", "lag", "blind")) {
      departure <- build_sensitivity_departure(
        type, geometry, small_tasks, sigma_small, ordinary, oracle,
        mechanism, support, direction = coordinate,
        primary_feature = coordinate, blind_feature = "x3_x4"
      )
      scale <- if (identical(type, "none")) 0 else departure_scale
      enumerated <- enumerated_score_target(
        schedules, oracle, ordinary, departure, scale, coordinate,
        true_coefficient, nuisance_coefficient, small_weights,
        geometry$riesz_lookup
      )
      departure$pattern <- type
      departure$map <- departure$temporal_map
      departure$scale <- scale
      formula_bias <- gate_e2_exact_bias(
        oracle$mixing, ordinary$mixing, true_R, small_weights,
        true_coefficient, nuisance_coefficient, departure, coordinate
      )
      formula_target <- true_coefficient +
        scale * departure$truth_shift_unit[[coordinate]]
      check_close(
        enumerated[["target"]], formula_target,
        label = paste("enumerated target", mechanism, type)
      )
      check_close(
        enumerated[["bias"]], formula_bias,
        label = paste("enumerated exact bias", mechanism, type)
      )
    }
  }

  # Wrong-law nuisance and direct-carryover terms, in both directions.
  for (true_mechanism in c("product", "uniform_without_replacement")) {
    analysis_mechanism <- if (identical(true_mechanism, "product")) {
      "uniform_without_replacement"
    } else "product"
    true_support <- if (identical(true_mechanism,
                                  "uniform_without_replacement")) {
      geometry$profile_count
    } else NULL
    analysis_support <- if (identical(analysis_mechanism,
                                      "uniform_without_replacement")) {
      geometry$profile_count
    } else NULL
    schedules <- enumerate_schedules(
      geometry$profile_count, small_tasks, true_mechanism
    )
    true_R <- sensitivity_assignment_correlation(
      small_tasks, true_mechanism, true_support
    )
    ordinary_true <- ordinary_task_operator(
      small_tasks, small_weights, true_R, true_mechanism, true_support
    )
    oracle_true <- solve_task_mixing(
      sigma_small, small_weights, true_mechanism, true_support
    )
    operator_wrong <- solve_task_mixing(
      sigma_small, small_weights, analysis_mechanism, analysis_support
    )
    departure_none <- build_sensitivity_departure(
      "none", geometry, small_tasks, sigma_small,
      ordinary_true, oracle_true, true_mechanism, true_support,
      direction = coordinate
    )
    departure_none$pattern <- "none"
    departure_none$map <- departure_none$temporal_map
    departure_none$scale <- 0
    enumerated_wrong <- enumerated_score_target(
      schedules, operator_wrong, ordinary_true, departure_none, 0,
      coordinate, true_coefficient, nuisance_coefficient,
      small_weights, geometry$riesz_lookup
    )
    formula_wrong <- gate_e2_exact_bias(
      operator_wrong$mixing, ordinary_true$mixing, true_R, small_weights,
      true_coefficient, nuisance_coefficient, departure_none, coordinate
    )
    check_close(
      enumerated_wrong[["bias"]], formula_wrong,
      label = paste("wrong-law decomposition", true_mechanism, "as",
                    analysis_mechanism)
    )

    rescue_coefficient <- rep(true_coefficient, small_tasks)
    enumerated_rescue <- enumerated_score_target(
      schedules, operator_wrong, ordinary_true, departure_none, 0,
      coordinate, true_coefficient, rescue_coefficient,
      small_weights, geometry$riesz_lookup
    )
    check_close(
      enumerated_rescue[["bias"]], 0,
      label = paste("wrong-law correct-mean rescue", true_mechanism, "as",
                    analysis_mechanism)
    )

    departure_lag <- build_sensitivity_departure(
      "lag", geometry, small_tasks, sigma_small,
      ordinary_true, oracle_true, true_mechanism, true_support,
      direction = coordinate, primary_feature = coordinate
    )
    departure_lag$pattern <- "lag"
    departure_lag$map <- departure_lag$temporal_map
    departure_lag$scale <- departure_scale
    enumerated_combined <- enumerated_score_target(
      schedules, operator_wrong, ordinary_true, departure_lag,
      departure_scale, coordinate, true_coefficient,
      nuisance_coefficient, small_weights, geometry$riesz_lookup
    )
    formula_combined <- gate_e2_exact_bias(
      operator_wrong$mixing, ordinary_true$mixing, true_R, small_weights,
      true_coefficient, nuisance_coefficient, departure_lag, coordinate
    )
    check_close(
      enumerated_combined[["bias"]], formula_combined,
      label = paste("wrong-law plus carryover decomposition",
                    true_mechanism, "as", analysis_mechanism)
    )
  }

  # Saturated task-profile dimensions, support, and outcome-blind provenance.
  set.seed(2026082742)
  respondents <- 96L
  support_tasks <- 3L
  profile_index <- sapply(seq_len(support_tasks), function(tt) {
    ((seq_len(respondents) + 5L * tt - 2L) %% geometry$profile_count) + 1L
  })
  outcome <- matrix(rnorm(respondents * support_tasks),
                    respondents, support_tasks)
  folds <- balanced_supported_cluster_folds(
    profile_index, 4L, 2026082743L, geometry$profile_count, 1000L
  )
  check_true(length(folds) == respondents && length(unique(folds)) == 4L,
             "respondent fold provenance and completeness")
  for (fold in sort(unique(folds))) {
    training <- folds != fold
    check_true(all(vapply(seq_len(support_tasks), function(tt) {
      all(tabulate(profile_index[training, tt],
                   nbins = geometry$profile_count) > 0)
    }, logical(1))), paste("saturated training support fold", fold))
    fit <- fit_task_saturated(
      profile_index, outcome, training, geometry,
      rep(1 / support_tasks, support_tasks)
    )
    check_true(identical(dim(fit$target_coefficients),
                         c(support_tasks, ncol(geometry$riesz_lookup))),
               paste("saturated target coefficient dimensions fold", fold))
    changed_outcome <- outcome
    changed_outcome[!training, ] <- changed_outcome[!training, ] + 1e6
    fit_changed <- fit_task_saturated(
      profile_index, changed_outcome, training, geometry,
      rep(1 / support_tasks, support_tasks)
    )
    check_close(
      fit_changed$target_coefficients, fit$target_coefficients,
      label = paste("held-out outcomes cannot affect nuisance fold", fold)
    )
  }

  # The production outer split is assignment-independent.  Changing the
  # realized profiles cannot change fold membership.
  fixed_outer <- balanced_cluster_folds(respondents, 4L, 2026082743L)
  changed_profiles <- profile_index
  changed_profiles[,] <- geometry$profile_count
  fixed_outer_changed <- balanced_cluster_folds(
    nrow(changed_profiles), 4L, 2026082743L
  )
  check_true(identical(fixed_outer, fixed_outer_changed),
             "outer folds are assignment-independent")
  sparse_training <- seq_len(respondents) <= 8L
  sparse_fit <- fit_task_saturated(
    profile_index, outcome, sparse_training, geometry,
    rep(1 / support_tasks, support_tasks)
  )
  check_true(sparse_fit$empty_cell_count > 0L &&
               all(is.finite(sparse_fit$prediction)) &&
               all(is.finite(sparse_fit$target_coefficients)),
             "saturated empty-cell rule is finite and recorded")
  oof <- saturated_task_profile_oof(
    outcome, profile_index, folds, geometry$profile_count,
    riesz_lookup = geometry$riesz_lookup,
    profile_probability = geometry$target_probability,
    task_weights = rep(1 / support_tasks, support_tasks)
  )
  check_true(all(is.finite(oof$prediction)) &&
               all(vapply(oof$target_coefficients_by_fold, function(x) {
                 identical(dim(x), c(support_tasks,
                                      ncol(geometry$riesz_lookup)))
               }, logical(1))),
             "saturated OOF target dimensions and finite support")

  # The selector-fold outcomes cannot enter its covariance operator indirectly
  # through another fold's residual nuisance.
  inner_plan <- supported_inner_fold_plan(
    profile_index, 4L, 202608274L, geometry$profile_count
  )
  inner_fold <- integer(respondents)
  inner_fold[inner_plan$eligible] <- inner_plan$fold
  outer_train <- rep(TRUE, respondents)
  leave_two <- gate_e2_leave_two_covariance_residual(
    profile_index, outcome, outer_train, inner_fold, 1L, geometry,
    rep(1 / support_tasks, support_tasks)
  )
  perturbed <- outcome
  perturbed[inner_fold == 1L, ] <- perturbed[inner_fold == 1L, ] + 1e6
  leave_two_perturbed <- gate_e2_leave_two_covariance_residual(
    profile_index, perturbed, outer_train, inner_fold, 1L, geometry,
    rep(1 / support_tasks, support_tasks)
  )
  check_close(
    leave_two$residual[leave_two$covariance_train, , drop = FALSE],
    leave_two_perturbed$residual[
      leave_two_perturbed$covariance_train, , drop = FALSE
    ],
    label = "selector-fold outcomes cannot enter covariance residuals"
  )
  check_true(!any(inner_plan$anchor & inner_fold > 0L),
             "support anchors are excluded from inner validation")
  missing_support_profiles <- profile_index
  missing_support_profiles[missing_support_profiles ==
                             geometry$profile_count] <- 1L
  missing_plan <- supported_inner_fold_plan(
    missing_support_profiles, 4L, 202608274L, geometry$profile_count
  )
  check_true(length(missing_plan$fold) == sum(missing_plan$eligible) &&
               missing_plan$anchor_count > 0L,
             "inner anchors cover observed support when an outer cell is absent")

  # A saturated-mean rescue under a wrong assignment law is not orthogonal.
  # Its inference-only operator must include the first-order contribution from
  # estimating the task-profile cell means.
  rescue_tasks <- 8L
  rescue_weights <- rep(1 / rescue_tasks, rescue_tasks)
  rescue_sigma <- scenario_covariance(
    list(covariance = "compound_symmetry", rho = 0.45), rescue_tasks
  )
  wrong_rescue <- solve_task_mixing(
    rescue_sigma, rescue_weights, "uniform_without_replacement",
    geometry$profile_count
  )
  rescue_true_R <- gate_e2_assignment_correlation(
    rescue_tasks, "product", geometry$profile_count
  )
  corrected_rescue <- gate_e2_saturated_rescue_if_operator(
    wrong_rescue, rescue_true_R, rescue_weights, "product"
  )
  check_close(
    diag(corrected_rescue$mixing %*% rescue_true_R), rescue_weights,
    label = "saturated rescue first-order IF calibration"
  )
  check_true(max(abs(corrected_rescue$mixing - wrong_rescue$mixing)) > 1e-8,
             "saturated rescue IF includes the cell-mean contribution")

  # Generic E1.24b tau must equal direct scalar numerical minimization.
  set.seed(2026082744)
  n_tau <- 240L
  ordinary_score <- cbind(
    x1_x2 = rnorm(n_tau),
    x1_x3 = rnorm(n_tau, sd = 1.2)
  )
  difference_score <- cbind(
    -0.55 * ordinary_score[, 1] + rnorm(n_tau, sd = 0.45),
    -0.30 * ordinary_score[, 2] + rnorm(n_tau, sd = 0.70)
  )
  adaptive_score <- ordinary_score + difference_score
  K_tau <- diag(c(0.8, 1.3))
  dimnames(K_tau) <- list(colnames(ordinary_score), colnames(ordinary_score))
  local_radius <- 0.4
  tau_fit <- coordinatewise_generic_tau(
    ordinary_score, adaptive_score, K_tau, local_radius
  )
  for (jj in seq_len(ncol(ordinary_score))) {
    objective <- function(tau) {
      2 * tau * tau_fit$c_coordinate[[jj]] +
        tau^2 * (tau_fit$v_coordinate[[jj]] +
                   local_radius^2 * tau_fit$k_coordinate[[jj]])
    }
    numeric_optimum <- stats::optimize(objective, c(0, 1))$minimum
    check_close(
      tau_fit$tau[[jj]], numeric_optimum, tolerance = 1e-6,
      label = paste("generic tau numerical optimum coordinate", jj)
    )
  }
  zero_tau <- coordinatewise_generic_tau(
    ordinary_score, ordinary_score, matrix(0, 2, 2), local_radius
  )
  check_close(zero_tau$tau, c(0, 0),
              label = "generic tau handles D equals zero")

  # Fixed-length bias-aware critical values solve the exact coverage equation.
  bias_over_se <- c(0, 0.5, 2)
  critical <- gate_e2_bias_aware_critical(bias_over_se)
  coverage <- stats::pnorm(critical - bias_over_se) -
    stats::pnorm(-critical - bias_over_se)
  check_close(coverage, rep(0.95, length(coverage)), tolerance = 1e-9,
              label = "bias-aware critical coverage equation")
  additive <- bias_aware_critical_value(
    standard_error = 1, epsilon = 0.5, sensitivity = 2
  )
  additive_coverage <- stats::pnorm(
    additive$total_half_width - 1
  ) - stats::pnorm(-additive$total_half_width - 1)
  check_true(additive_coverage >= 0.95,
             "additive bias-aware interval is conservative at boundary")

  # One complete replication for each required implementation path.
  end_to_end_ids <- c(
    "null_cs_product",
    "lf_r0_5_cs_product",
    "lag_r1_cs_product",
    "blind_r2_cs_product",
    "qerr_product_as_wor_null",
    "qerr_wor_as_product_null"
  )
  end_to_end_rows <- list()
  for (id in end_to_end_ids) {
    replication <- run_gate_e2_replication(
      scenario_by_id(id), 1L, config, geometry
    )
    check_true(is.null(replication$error),
               paste("end-to-end replication has no failure", id))
    check_true(all(is.finite(replication$results$estimate)) &&
                 all(is.finite(replication$results$standard_error)),
               paste("end-to-end finite estimates", id))
    check_true(max(replication$results$calibration_residual) <= 1e-10,
               paste("end-to-end exact calibration", id))
    ordinary <- replication$results[
      replication$results$method == "ordinary_taskwise", , drop = FALSE
    ]
    check_true(nrow(ordinary) == ncol(geometry$riesz_lookup),
               paste("ordinary coordinate completeness", id))
    check_close(
      ordinary$predicted_bias, rep(0, nrow(ordinary)),
      label = paste("ordinary predicted bias is zero", id)
    )
    check_true(nrow(replication$selector) == config$estimator$outer_folds &&
                 !anyDuplicated(replication$selector$outer_fold),
               paste("outer respondent fold isolation", id))
    end_to_end_rows[[id]] <- list(
      result_rows = nrow(replication$results),
      selector_rows = nrow(replication$selector),
      maximum_calibration_residual =
        max(replication$results$calibration_residual),
      maximum_ordinary_predicted_bias = max(abs(ordinary$predicted_bias))
    )
  }
  for (failed_seed_replication in c(326L, 2288L)) {
    regression <- run_gate_e2_replication(
      scenario_by_id("lf_r1_cs_product_g250"),
      failed_seed_replication, config, geometry
    )
    check_true(is.null(regression$error) &&
                 all(is.finite(regression$results$estimate)),
               paste("formerly missing-support replication completes",
                     failed_seed_replication))
    check_true(any(regression$selector$outer_saturated_empty_cells > 0L),
               paste("missing-support fallback is recorded",
                     failed_seed_replication))
  }
  state$end_to_end <- end_to_end_rows
  invisible(TRUE)
}

failure <- NULL
tryCatch(
  run_tests(),
  error = function(condition) {
    failure <<- conditionMessage(condition)
  }
)

source_paths <- c(
  solver = file.path(repo_root, "analysis", "covariance_headroom",
                     "calibration_solver.R"),
  feasible = file.path(analysis_dir, "feasible_estimator.R"),
  simulation = file.path(analysis_dir, "simulation_functions.R"),
  sensitivity = file.path(analysis_dir, "sensitivity_functions.R"),
  e2 = file.path(analysis_dir, "e2_simulation_functions.R"),
  config = config_path,
  test = script
)
result <- list(
  gate = "E2_correctness_tests",
  passed = is.null(failure),
  failure = failure,
  checks_passed = length(state$checks),
  maximum_recorded_error = state$maximum_error,
  completed_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  production_scenario_count = length(config$simulation$scenarios),
  end_to_end_scenarios = if (exists("end_to_end", envir = state,
                                    inherits = FALSE)) {
    state$end_to_end
  } else NULL,
  checks = state$checks,
  source_md5 = as.list(unname(tools::md5sum(source_paths)))
)
names(result$source_md5) <- names(source_paths)
yaml::write_yaml(result, result_path)

if (!is.null(failure)) {
  stop("Gate-E2 correctness tests failed: ", failure, call. = FALSE)
}
cat(sprintf("All %d Gate-E2 correctness tests passed.\n",
            length(state$checks)))
