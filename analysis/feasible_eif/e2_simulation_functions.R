`%||%` <- function(x, y) if (is.null(x)) y else x

read_gate_e2_config <- function(path) {
  if (!requireNamespace("yaml", quietly = TRUE)) {
    stop("Package `yaml` is required.", call. = FALSE)
  }
  path <- normalizePath(path, mustWork = TRUE)
  config <- yaml::read_yaml(path)
  if (!is.null(config$inherits)) {
    parent <- read_gate_e2_config(file.path(dirname(path), config$inherits))
    config$inherits <- NULL
    config <- utils::modifyList(parent, config)
  }
  required <- c("version", "mode", "seed", "replications_per_scenario",
                "estimand", "estimator", "simulation")
  missing <- setdiff(required, names(config))
  if (length(missing)) {
    stop("Gate-E2 configuration is missing: ", paste(missing, collapse = ", "),
         call. = FALSE)
  }
  config$seed <- as.integer(config$seed)
  config$replications_per_scenario <- as.integer(config$replications_per_scenario)
  config$monte_carlo_bootstrap_draws <-
    as.integer(config$monte_carlo_bootstrap_draws)
  config$monte_carlo_bootstrap_seed <-
    as.integer(config$monte_carlo_bootstrap_seed)
  feasible_assert(config$version == 3L && config$replications_per_scenario >= 1L,
                  "The Gate-E2 configuration version or replication count is invalid.")
  feasible_assert(length(config$simulation$scenarios) >= 1L,
                  "At least one Gate-E2 scenario is required.")
  scenario_ids <- vapply(config$simulation$scenarios, `[[`, character(1), "id")
  feasible_assert(!anyDuplicated(scenario_ids),
                  "Gate-E2 scenario identifiers must be unique.")
  config$simulation$default_respondents <- as.integer(
    config$simulation$default_respondents %||%
      config$reference_design$default_respondents %||% 500L
  )
  config$simulation$outcome_noise_scale <- as.numeric(
    config$simulation$outcome_noise_scale %||%
      config$reference_design$outcome_noise_scale %||% 1
  )
  config$estimator$blend_grid <- config$estimator$blend_grid %||%
    config$estimator$gate_d_blend_grid
  config$estimator$mean_learner <- config$estimator$mean_learner %||%
    config$reference_design$default_score_mean_learner
  config$pass_rules <- config$pass_rules %||% list(
    hard_correctness = config$hard_correctness_pass_rules,
    practical_value = config$practical_value_pass_rules
  )
  config$simulation$scenarios <- lapply(config$simulation$scenarios, function(s) {
    s$tasks <- as.integer(s$tasks %||% config$simulation$default_tasks %||%
                            config$reference_design$tasks)
    s[["covariance"]] <- s[["covariance"]] %||% s$residual_covariance
    s[["rho"]] <- as.numeric(s[["rho"]] %||% s$residual_rho)
    s
  })
  config$.path <- path
  config
}

gate_e2_scenario_seed <- function(scenario, replication, config) {
  stream <- as.integer(scenario$random_stream %||%
                         match(scenario$id, vapply(
                           config$simulation$scenarios, `[[`, character(1), "id"
                         )))
  as.integer(config$seed + stream * 100000L + replication)
}

gate_e2_assignment_correlation <- function(tasks, mechanism, support_size) {
  sensitivity_assignment_correlation(
    tasks, mechanism,
    if (identical(mechanism, "uniform_without_replacement")) support_size else NULL
  )
}

gate_e2_mixing_variance <- function(mixing, covariance, assignment_R) {
  mixing_quadratic(mixing, covariance, assignment_R)
}

gate_e2_mixing_k <- function(difference_mixing, covariance, assignment_R) {
  nrow(difference_mixing) *
    gate_e2_mixing_variance(difference_mixing, covariance, assignment_R)
}

gate_e2_make_departure <- function(scenario, respondents, geometry, sigma_true,
                                   ordinary_true, oracle_true) {
  pattern <- switch(
    scenario$carryover,
    none = "none",
    least_favourable = "least_favorable",
    least_favorable = "least_favorable",
    lag_only = "lag",
    lag = "lag",
    blind = "blind",
    stop("Unknown Gate-E2 carryover pattern: ", scenario$carryover,
         call. = FALSE)
  )
  built <- build_sensitivity_departure(
    type = pattern,
    geometry = geometry,
    task_count = as.integer(scenario$tasks),
    true_residual_covariance = sigma_true,
    ordinary_operator = ordinary_true,
    oracle_operator = oracle_true,
    mechanism = scenario$true_assignment,
    support_size = if (identical(scenario$true_assignment,
                                 "uniform_without_replacement")) {
      geometry$profile_count
    } else NULL,
    direction = "x1_x2",
    primary_feature = "x1_x2",
    blind_feature = "x3_x4"
  )
  local_radius <- if (identical(pattern, "none")) {
    0
  } else if (identical(scenario$scale, "fixed")) {
    sqrt(respondents) * as.numeric(scenario$epsilon_M)
  } else {
    as.numeric(scenario$radius_multiplier_r) / sqrt(as.integer(scenario$tasks))
  }
  scale <- if (identical(pattern, "none")) 0 else
    local_radius / sqrt(respondents)
  built$pattern <- pattern
  built$feature <- if (identical(pattern, "blind")) "x3_x4" else "x1_x2"
  built$map <- built$temporal_map
  built$scale <- scale
  built$epsilon <- abs(scale) * built$unit_epsilon
  built$local_radius <- local_radius
  built$normalized_r <- sqrt(as.integer(scenario$tasks)) * local_radius
  built$unit_bias_oracle <- drop(crossprod(
    built$direction, built$oracle_bias_unit
  ))
  built$projection_residual <- max(
    built$diagnostics$orthogonality_residual,
    built$diagnostics$normalized_norm_error,
    if (is.finite(built$diagnostics$blind_bias_residual)) {
      built$diagnostics$blind_bias_residual
    } else 0
  )
  built
}

gate_e2_apply_departure <- function(data, departure, geometry = NULL) {
  apply_sensitivity_departure(data, departure, departure$local_radius)
}

gate_e2_departure_bias <- function(difference_mixing, departure,
                                   assignment_R, coordinate = NULL) {
  if (identical(departure$pattern, "none") || departure$scale == 0) {
    if (is.null(coordinate)) return(0)
    return(0)
  }
  multiplier <- sum(
    (t(as.matrix(difference_mixing)) %*% departure$map) * assignment_R
  )
  bias <- departure$scale * departure$cross_gram * multiplier
  if (is.null(coordinate)) {
    return(drop(crossprod(departure$direction, bias)))
  }
  if (!coordinate %in% names(bias)) return(0)
  unname(bias[[coordinate]])
}

balanced_supported_cluster_folds <- function(profile_index, folds, seed,
                                              profile_count,
                                              maximum_attempts = 1000L,
                                              minimum_training_count = 1L) {
  profile_index <- as.matrix(profile_index)
  n <- nrow(profile_index)
  folds <- as.integer(folds)
  for (attempt in seq_len(as.integer(maximum_attempts))) {
    candidate <- balanced_cluster_folds(
      n, folds, as.integer(seed + 7919L * (attempt - 1L))
    )
    supported <- all(vapply(seq_len(folds), function(fold) {
      training <- candidate != fold
      all(vapply(seq_len(ncol(profile_index)), function(tt) {
        all(tabulate(profile_index[training, tt], nbins = profile_count) >=
              as.integer(minimum_training_count))
      }, logical(1)))
    }, logical(1)))
    if (supported) return(candidate)
  }
  stop("Could not construct outcome-blind supported respondent folds.",
       call. = FALSE)
}

supported_inner_fold_plan <- function(profile_index, folds, seed,
                                      profile_count) {
  profile_index <- as.matrix(profile_index)
  n <- nrow(profile_index)
  tasks <- ncol(profile_index)
  cell_id <- sweep(profile_index, 2L,
                   (seq_len(tasks) - 1L) * profile_count, "+")
  # Anchor every task-profile cell that is actually represented in this outer
  # training sample.  A cell absent from the outer sample cannot be anchored;
  # all inner fits use the same prespecified task-mean fallback for it.
  uncovered <- sort(unique(as.vector(cell_id)))
  anchor <- logical(n)
  # Deterministic assignment-only set cover.  Anchors are used in every inner
  # training fit and never in inner validation, so they guarantee saturated
  # support without leaking an outcome into its own selector score.
  while (length(uncovered)) {
    gain <- vapply(seq_len(n), function(i) {
      if (anchor[[i]]) return(-1L)
      sum(unique(cell_id[i, ]) %in% uncovered)
    }, integer(1))
    chosen <- which.max(gain)
    feasible_assert(gain[[chosen]] > 0,
                    "Outer training support cannot cover all task-profile cells.")
    anchor[[chosen]] <- TRUE
    uncovered <- setdiff(uncovered, unique(cell_id[chosen, ]))
  }
  eligible <- !anchor
  feasible_assert(sum(eligible) >= 2L * as.integer(folds),
                  "Too few non-anchor respondents remain for inner cross-fitting.")
  list(
    fold = balanced_cluster_folds(sum(eligible), folds, seed),
    eligible = eligible,
    anchor = anchor,
    anchor_count = sum(anchor)
  )
}

gate_e2_leave_two_covariance_residual <- function(
    profile_index, outcome, outer_train, inner_fold, selector_fold,
    geometry, task_weights) {
  profile_index <- as.matrix(profile_index)
  outcome <- as.matrix(outcome)
  outer_train <- as.logical(outer_train)
  selector_pool <- outer_train & inner_fold > 0L
  tasks <- ncol(outcome)
  out <- matrix(NA_real_, nrow(outcome), tasks)
  inner_values <- sort(unique(inner_fold[selector_pool]))
  for (covariance_fold in setdiff(inner_values, selector_fold)) {
    covariance_valid <- selector_pool & inner_fold == covariance_fold
    covariance_mean_train <- outer_train &
      (inner_fold == 0L |
         (inner_fold != selector_fold & inner_fold != covariance_fold))
    covariance_fit <- fit_task_saturated(
      profile_index, outcome, covariance_mean_train, geometry, task_weights
    )
    out[covariance_valid, ] <-
      outcome[covariance_valid, , drop = FALSE] -
      covariance_fit$prediction[covariance_valid, , drop = FALSE]
  }
  covariance_train <- selector_pool & inner_fold != selector_fold
  feasible_assert(all(is.finite(out[covariance_train, , drop = FALSE])),
                  "Leave-two-fold covariance residuals are incomplete.")
  list(residual = out, covariance_train = covariance_train)
}

fit_task_saturated <- function(profile_index, outcome, train_respondent,
                               geometry, task_weights) {
  profile_index <- as.matrix(profile_index)
  outcome <- as.matrix(outcome)
  n <- nrow(outcome)
  tasks <- ncol(outcome)
  profile_count <- geometry$profile_count
  prediction <- matrix(NA_real_, n, tasks)
  target_prediction <- matrix(NA_real_, profile_count, tasks)
  rank <- integer(tasks)
  for (tt in seq_len(tasks)) {
    train_profile <- profile_index[train_respondent, tt]
    train_outcome <- outcome[train_respondent, tt]
    overall <- mean(train_outcome)
    sums <- rowsum(train_outcome, train_profile, reorder = FALSE)
    counts <- tabulate(train_profile, nbins = profile_count)
    # This makes the learner total on every assignment-independent split.
    # Observed cells remain their exact empirical means; an empty cell receives
    # the prespecified task-level training mean.  The fallback event vanishes
    # exponentially under overlap and is reported for audit.
    cell_mean <- rep(overall, profile_count)
    observed <- as.integer(rownames(sums))
    cell_mean[observed] <- drop(sums) / counts[observed]
    prediction[, tt] <- cell_mean[profile_index[, tt]]
    target_prediction[, tt] <- cell_mean
    rank[[tt]] <- sum(counts > 0)
  }
  target_coefficients <- matrix(
    0, tasks, ncol(geometry$riesz_lookup),
    dimnames = list(NULL, colnames(geometry$riesz_lookup))
  )
  for (tt in seq_len(tasks)) {
    target_coefficients[tt, ] <- drop(crossprod(
      geometry$target_probability,
      geometry$riesz_lookup * target_prediction[, tt]
    ))
  }
  list(
    prediction = prediction,
    plugin = drop(crossprod(task_weights, target_coefficients)),
    target_coefficients = target_coefficients,
    target_prediction = target_prediction,
    rank = sum(rank),
    empty_cell_count = tasks * profile_count - sum(rank)
  )
}

fit_gate_e2_mean <- function(learner, design, target_design, profile_index,
                             outcome, train_respondent, geometry,
                             task_weights) {
  if (learner %in% c("task_specific_saturated_cell_means",
                     "task_specific_saturated_16_profile_cell_means")) {
    return(fit_task_saturated(
      profile_index, outcome, train_respondent, geometry, task_weights
    ))
  }
  if (identical(learner, "task_specific_additive_ols")) {
    fit <- fit_task_additive(
      design, target_design, outcome, train_respondent, geometry, task_weights
    )
    fit$target_coefficients <- matrix(
      0, ncol(outcome), ncol(geometry$riesz_lookup),
      dimnames = list(NULL, colnames(geometry$riesz_lookup))
    )
    return(fit)
  }
  stop("Unknown Gate-E2 mean learner: ", learner, call. = FALSE)
}

gate_e2_cluster_covariance <- function(x, y = NULL, cluster_weight = NULL) {
  x <- as.matrix(x)
  if (is.null(y)) y <- x
  y <- as.matrix(y)
  feasible_assert(nrow(x) == nrow(y),
                  "Cluster covariance inputs must have the same rows.")
  if (is.null(cluster_weight)) cluster_weight <- rep(1, nrow(x))
  cluster_weight <- as.numeric(cluster_weight)
  normalized <- cluster_weight / sum(cluster_weight)
  x_center <- sweep(x, 2L, drop(crossprod(normalized, x)), "-")
  y_center <- sweep(y, 2L, drop(crossprod(normalized, y)), "-")
  denominator <- 1 - sum(normalized^2)
  crossprod(x_center * sqrt(normalized),
            y_center * sqrt(normalized)) / denominator
}

gate_e2_clip <- function(x, lower = 0, upper = 1) {
  pmin(upper, pmax(lower, x))
}

gate_e2_generic_tau <- function(ordinary_score, selected_score, k,
                                local_radius, cluster_weight = NULL) {
  ordinary_score <- as.matrix(ordinary_score)
  selected_score <- as.matrix(selected_score)
  difference <- selected_score - ordinary_score
  cross <- diag(gate_e2_cluster_covariance(
    ordinary_score, difference, cluster_weight
  ))
  difference_variance <- diag(gate_e2_cluster_covariance(
    difference, difference, cluster_weight
  ))
  k <- rep(as.numeric(k), length.out = ncol(ordinary_score))
  denominator <- difference_variance + as.numeric(local_radius)^2 * k
  tau <- numeric(length(denominator))
  positive <- denominator > 1e-14
  tau[positive] <- gate_e2_clip(-cross[positive] / denominator[positive])
  tau[!positive] <- 0
  names(tau) <- colnames(ordinary_score)
  list(
    tau = tau,
    ordinary_difference_cross_covariance = cross,
    difference_variance = difference_variance,
    k = k,
    denominator = denominator
  )
}

gate_e2_exact_bias <- function(mixing, ordinary_mixing, true_assignment_R,
                               task_weights, true_coefficient,
                               nuisance_coefficient, departure = NULL,
                               coordinate) {
  mixing <- as.matrix(mixing)
  ordinary_mixing <- as.matrix(ordinary_mixing)
  true_assignment_R <- as.matrix(true_assignment_R)
  nuisance_coefficient <- as.numeric(nuisance_coefficient)
  tasks <- nrow(mixing)
  feasible_assert(length(nuisance_coefficient) == tasks,
                  "The nuisance target coefficient needs one value per task.")
  own_residual <- as.numeric(true_coefficient) - nuisance_coefficient
  plugin <- sum(task_weights * nuisance_coefficient)
  own_score <- sum((mixing * true_assignment_R) * own_residual)
  base_bias <- plugin + own_score - as.numeric(true_coefficient)
  carry_bias <- 0
  if (!is.null(departure) && departure$scale != 0) {
    carry_bias <- gate_e2_departure_bias(
      mixing - ordinary_mixing, departure, true_assignment_R, coordinate
    )
  }
  base_bias + carry_bias
}

gate_e2_target_truth <- function(geometry, departure, ordinary_mixing,
                                 true_assignment_R, base_truth = NULL) {
  if (is.null(base_truth)) base_truth <- geometry$truth
  truth <- as.numeric(base_truth)
  names(truth) <- names(base_truth)
  if (!is.null(departure) && departure$scale != 0) {
    truth <- truth + departure$scale * departure$truth_shift_unit[names(truth)]
  }
  truth
}

gate_e2_operator_for_coordinate <- function(mixing, coordinate_count) {
  if (is.matrix(mixing)) return(rep(list(mixing), coordinate_count))
  feasible_assert(is.list(mixing) && length(mixing) == coordinate_count,
                  "Coordinate-specific mixing list is malformed.")
  mixing
}

gate_e2_bias_aware_critical <- function(bias_over_se, level = 0.95) {
  bias_over_se <- abs(as.numeric(bias_over_se))
  alpha <- 1 - level
  vapply(bias_over_se, function(b) {
    objective <- function(c) {
      stats::pnorm(c - b) - stats::pnorm(-c - b) - level
    }
    lower <- stats::qnorm(1 - alpha / 2)
    upper <- lower + b + 10
    stats::uniroot(objective, c(lower, upper), tol = 1e-10)$root
  }, numeric(1))
}

gate_e2_normal_power <- function(noncentrality, alpha = 0.05) {
  critical <- stats::qnorm(1 - alpha / 2)
  noncentrality <- as.numeric(noncentrality)
  stats::pnorm(-critical - noncentrality) +
    1 - stats::pnorm(critical - noncentrality)
}

gate_e2_reported_method <- function(method) {
  map <- c(
    ordinary = "ordinary_taskwise",
    ordinary_joint_oracle = "oracle_unrestricted_joint",
    selected = "selected_adaptive",
    oracle_restricted = "oracle_adaptive",
    selected_correct_law = "selected_adaptive_correct_assignment",
    selected_saturated_rescue = "wrong_assignment_saturated_mean",
    oracle_correct_law = "oracle_adaptive_correct_assignment",
    analysis_oracle_own_mean = "wrong_assignment_oracle_mean",
    oracle_tau_0_25 = "oracle_tau_0_25",
    oracle_tau_0_50 = "oracle_tau_0_50",
    oracle_tau_0_75 = "oracle_tau_0_75",
    oracle_unrestricted_sensitivity = "oracle_tau_mm",
    oracle_pattern_sensitivity = "oracle_tau_pattern",
    feasible_sensitivity = "feasible_tau_generic"
  )
  unname(map[[method]] %||% method)
}

gate_e2_saturated_rescue_if_operator <- function(operator, true_R,
                                                  task_weights,
                                                  true_mechanism,
                                                  true_support = NULL) {
  # With a misspecified assignment law, cross-fitting a saturated conditional
  # mean makes the point estimator consistent but does not make its score
  # orthogonal.  The first-order contribution from estimating those cell means
  # is diag(w - diag(H R_true)).  Adding it gives the influence-function
  # operator used for variance estimation while leaving the point estimator
  # itself unchanged.
  out <- operator
  out$mixing <- operator$mixing + diag(
    as.numeric(task_weights) - diag(operator$mixing %*% true_R),
    nrow = length(task_weights)
  )
  out$assignment_correlation <- true_R
  out$mechanism <- true_mechanism
  out$support_size <- true_support
  out$calibration <- diag(out$mixing %*% true_R)
  out$calibration_residual <- max(abs(out$calibration - task_weights))
  out$candidate <- paste0(operator$candidate, "_saturated_mean_if")
  out
}

run_gate_e2_replication <- function(scenario, replication, config, geometry) {
  seed <- gate_e2_scenario_seed(scenario, replication, config)
  set.seed(seed)
  scenario_for_draw <- scenario
  scenario_for_draw$assignment <- scenario$true_assignment
  data <- draw_gate_d_data(scenario_for_draw, config, geometry)
  n <- data$respondents
  tasks <- data$tasks
  coordinate_names <- colnames(geometry$riesz_lookup)
  coordinate_count <- length(coordinate_names)
  task_weights <- rep(1 / tasks, tasks)
  analysis_assignment <- scenario$analysis_assignment %||% scenario$true_assignment
  support_analysis <- if (identical(analysis_assignment,
                                    "uniform_without_replacement")) {
    geometry$profile_count
  } else NULL
  support_true <- if (identical(scenario$true_assignment,
                                "uniform_without_replacement")) {
    geometry$profile_count
  } else NULL
  sigma_true <- scenario_covariance(scenario_for_draw, tasks) *
    as.numeric(config$simulation$outcome_noise_scale)^2
  ordinary_true <- ordinary_task_operator(
    tasks, task_weights,
    gate_e2_assignment_correlation(tasks, scenario$true_assignment,
                                   geometry$profile_count),
    scenario$true_assignment, support_true
  )
  oracle_true <- solve_task_mixing(
    sigma_true, task_weights, scenario$true_assignment, support_true
  )
  oracle_analysis_reference <- solve_task_mixing(
    sigma_true, task_weights, analysis_assignment, support_analysis
  )
  base_truth <- geometry$truth
  if (isTRUE(scenario$aligned_nuisance_error)) {
    calibration_error <- sum(
      oracle_analysis_reference$mixing * ordinary_true$assignment_correlation
    ) - 1
    difference_variance <- gate_e2_mixing_variance(
      oracle_analysis_reference$mixing - ordinary_true$mixing,
      sigma_true, ordinary_true$assignment_correlation
    )
    desired_ncp <- as.numeric(
      config$assignment_error_construction$directional_noncentrality
    )
    feasible_assert(abs(calibration_error) > 1e-10 &&
                      difference_variance > 1e-12,
                    "The frozen assignment-error alignment has no headroom.")
    aligned_coefficient <- desired_ncp * sqrt(difference_variance / n) /
      calibration_error
    own_shift <- aligned_coefficient - base_truth[[1L]]
    primary_z <- matrix(
      geometry$riesz_lookup[as.vector(data$profile_index), 1L], n, tasks
    )
    data$mean <- data$mean + own_shift * primary_z
    data$outcome <- data$outcome + own_shift * primary_z
    base_truth[[1L]] <- aligned_coefficient
  }
  departure <- gate_e2_make_departure(
    scenario, n, geometry, sigma_true, ordinary_true, oracle_true
  )
  data$base_mean <- data$mean
  data <- gate_e2_apply_departure(data, departure, geometry)
  true_R <- ordinary_true$assignment_correlation
  truth <- gate_e2_target_truth(
    geometry, departure, ordinary_true$mixing, true_R, base_truth
  )

  design <- task_additive_design(data$profile_index, geometry)
  target_design <- target_additive_design(tasks, geometry)
  outer_folds <- balanced_cluster_folds(
    n, as.integer(config$estimator$outer_folds), seed + 1L
  )
  # The outer split is a prespecified function of respondent IDs and the seed.
  # It cannot depend on held-out assignments if foldwise unbiasedness is to be
  # literal rather than conditional on a support-selection rule.  The frozen
  # saturated learner is defined even on the rare empty-cell event.
  learner <- scenario$score_mean_learner %||% config$estimator$mean_learner
  method_names <- c(
    "ordinary", "ordinary_joint_oracle", "selected", "oracle_restricted",
    "selected_correct_law", "selected_saturated_rescue",
    "oracle_correct_law", "analysis_oracle_own_mean", "oracle_tau_0_25",
    "oracle_tau_0_50", "oracle_tau_0_75",
    "oracle_unrestricted_sensitivity", "oracle_pattern_sensitivity",
    "feasible_sensitivity"
  )
  score <- stats::setNames(lapply(method_names, function(x) {
    matrix(NA_real_, n, coordinate_count,
           dimnames = list(NULL, coordinate_names))
  }), method_names)
  inference_score <- stats::setNames(lapply(method_names, function(x) {
    matrix(NA_real_, n, coordinate_count,
           dimnames = list(NULL, coordinate_names))
  }), method_names)
  predicted_bias_fold <- stats::setNames(lapply(method_names, function(x) {
    matrix(NA_real_, length(unique(outer_folds)), coordinate_count,
           dimnames = list(NULL, coordinate_names))
  }), method_names)
  omega_fold <- predicted_bias_fold
  mixing_fold <- stats::setNames(lapply(method_names, function(x) {
    vector("list", length(unique(outer_folds)))
  }), method_names)
  sigma_hat_fold <- vector("list", length(unique(outer_folds)))
  fold_fraction <- numeric(length(unique(outer_folds)))
  tau_rows <- list()
  selector_rows <- list()
  max_calibration <- 0

  for (outer in sort(unique(outer_folds))) {
    outer_valid <- outer_folds == outer
    outer_train <- !outer_valid
    fold_fraction[[outer]] <- mean(outer_valid)
    train_ids <- which(outer_train)
    inner_plan <- supported_inner_fold_plan(
      data$profile_index[train_ids, , drop = FALSE],
      as.integer(config$estimator$inner_folds), seed + 100L * outer,
      geometry$profile_count
    )
    inner_local <- inner_plan$fold
    inner_fold <- integer(n)
    inner_fold[train_ids[inner_plan$eligible]] <- inner_local
    selector_pool <- outer_train & inner_fold > 0L
    residual_oof <- matrix(NA_real_, n, tasks)
    saturated_residual_oof <- matrix(NA_real_, n, tasks)
    inner_plugin <- vector("list", max(inner_local))
    for (inner in sort(unique(inner_local))) {
      inner_valid <- outer_train & inner_fold == inner
      inner_train <- outer_train & inner_fold != inner
      fit <- fit_gate_e2_mean(
        learner, design, target_design, data$profile_index, data$outcome,
        inner_train, geometry, task_weights
      )
      residual_oof[inner_valid, ] <-
        data$outcome[inner_valid, , drop = FALSE] -
        fit$prediction[inner_valid, , drop = FALSE]
      inner_plugin[[inner]] <- fit$plugin
      saturated_fit <- fit_task_saturated(
        data$profile_index, data$outcome, inner_train, geometry, task_weights
      )
      saturated_residual_oof[inner_valid, ] <-
        data$outcome[inner_valid, , drop = FALSE] -
        saturated_fit$prediction[inner_valid, , drop = FALSE]
    }
    feasible_assert(all(is.finite(residual_oof[selector_pool, , drop = FALSE])),
                    "Gate-E2 inner cross-fitted score residuals are incomplete.")
    feasible_assert(all(is.finite(
      saturated_residual_oof[selector_pool, , drop = FALSE]
    )), "Gate-E2 saturated covariance residuals are incomplete.")

    candidate_scores <- NULL
    candidate_metadata <- NULL
    for (inner in sort(unique(inner_local))) {
      selector_valid <- outer_train & inner_fold == inner
      covariance_train <- selector_pool & inner_fold != inner
      # A residual for covariance fold k is fitted after excluding both k and
      # the selector-validation fold `inner`.  Hence the operator used to score
      # fold `inner` is completely free of that fold's outcomes, including
      # through another fold's residual nuisance.
      leave_two <- gate_e2_leave_two_covariance_residual(
        data$profile_index, data$outcome, outer_train, inner_fold, inner,
        geometry, task_weights
      )
      feasible_assert(identical(leave_two$covariance_train, covariance_train),
                      "Leave-two-fold covariance provenance rows changed.")
      covariance_set <- estimate_working_covariances(
        leave_two$residual[covariance_train, , drop = FALSE],
        data$cluster_weight[covariance_train],
        as.numeric(config$estimator$covariance_eigenvalue_relative_floor)
      )
      library <- make_operator_library(
        covariance_set, task_weights, analysis_assignment, support_analysis,
        as.numeric(unlist(config$estimator$blend_grid))
      )
      if (is.null(candidate_scores)) {
        candidate_metadata <- attr(library, "metadata")
        candidate_scores <- stats::setNames(lapply(names(library), function(x) {
          matrix(NA_real_, sum(selector_pool), coordinate_count,
                 dimnames = list(NULL, coordinate_names))
        }), names(library))
        training_position <- integer(n)
        training_position[selector_pool] <- seq_len(sum(selector_pool))
      }
      local_position <- training_position[selector_valid]
      for (candidate in names(library)) {
        candidate_scores[[candidate]][local_position, ] <-
          score_factorial_clusters(
            residual_oof[selector_valid, , drop = FALSE],
            data$profile_index[selector_valid, , drop = FALSE],
            geometry$riesz_lookup, library[[candidate]],
            inner_plugin[[inner]]
          )
      }
    }
    feasible_assert(all(vapply(candidate_scores, function(x) all(is.finite(x)),
                                logical(1))),
                    "Gate-E2 selector scores are incomplete.")
    selection <- select_operator_one_se(
      candidate_scores, candidate_metadata,
      data$cluster_weight[selector_pool]
    )
    final_covariance_set <- estimate_working_covariances(
      saturated_residual_oof[selector_pool, , drop = FALSE],
      data$cluster_weight[selector_pool],
      as.numeric(config$estimator$covariance_eigenvalue_relative_floor)
    )
    sigma_hat_fold[[outer]] <- final_covariance_set$unstructured$covariance
    final_library <- make_operator_library(
      final_covariance_set, task_weights, analysis_assignment, support_analysis,
      as.numeric(unlist(config$estimator$blend_grid))
    )
    ordinary <- final_library$ordinary
    selected <- final_library[[selection$candidate]]
    oracle_analysis <- solve_task_mixing(
      sigma_true, task_weights, analysis_assignment, support_analysis
    )
    difference_mixing <- selected$mixing - ordinary$mixing
    k_hat_selected <- gate_e2_mixing_k(
      difference_mixing, final_covariance_set$unstructured$covariance,
      ordinary$assignment_correlation
    )
    generic <- gate_e2_generic_tau(
      candidate_scores$ordinary,
      candidate_scores[[selection$candidate]], k_hat_selected,
      departure$local_radius, data$cluster_weight[selector_pool]
    )
    tau_unrestricted <- 1 / (1 + departure$normalized_r^2)
    oracle_gap <- gate_e2_mixing_variance(
      ordinary$mixing, sigma_true, true_R
    ) - gate_e2_mixing_variance(
      oracle_true$mixing, sigma_true, true_R
    )
    oracle_pattern_bias <- gate_e2_departure_bias(
      oracle_true$mixing - ordinary_true$mixing, departure, true_R
    )
    pattern_noncentrality_sq <- if (oracle_gap > 1e-14) {
      n * oracle_pattern_bias^2 / oracle_gap
    } else 0
    tau_pattern <- 1 / (1 + pattern_noncentrality_sq)
    oracle_blend <- function(tau, label) {
      blend_task_operator(ordinary, oracle_analysis, tau, label)
    }
    common_operators <- list(
      ordinary = ordinary,
      selected = selected,
      oracle_restricted = oracle_analysis,
      oracle_correct_law = oracle_true,
      analysis_oracle_own_mean = oracle_analysis,
      oracle_tau_0_25 = oracle_blend(0.25, "oracle"),
      oracle_tau_0_50 = oracle_blend(0.50, "oracle"),
      oracle_tau_0_75 = oracle_blend(0.75, "oracle"),
      oracle_unrestricted_sensitivity = oracle_blend(
        tau_unrestricted, "oracle_sensitivity"
      ),
      oracle_pattern_sensitivity = oracle_blend(
        tau_pattern, "oracle_pattern"
      )
    )
    selected_correct_law <- if (identical(selected$covariance_class,
                                           "ordinary")) {
      ordinary_true
    } else {
      adaptive_correct_law <- solve_task_mixing(
        final_covariance_set[[selected$covariance_class]]$covariance,
        task_weights, scenario$true_assignment, support_true
      )
      blend_task_operator(
        ordinary_true, adaptive_correct_law, selected$blend,
        selected$covariance_class
      )
    }
    common_operators$selected_correct_law <- selected_correct_law
    common_operators$selected_saturated_rescue <- selected
    feasible_mixing <- lapply(seq_len(coordinate_count), function(j) {
      ordinary$mixing + generic$tau[[j]] * difference_mixing
    })

    final_fit <- fit_gate_e2_mean(
      learner, design, target_design, data$profile_index, data$outcome,
      outer_train, geometry, task_weights
    )
    valid_residual <- data$outcome[outer_valid, , drop = FALSE] -
      final_fit$prediction[outer_valid, , drop = FALSE]
    saturated_final_fit <- if (learner %in% c(
      "task_specific_saturated_cell_means",
      "task_specific_saturated_16_profile_cell_means"
    )) final_fit else fit_task_saturated(
      data$profile_index, data$outcome, outer_train, geometry, task_weights
    )
    for (method in setdiff(names(common_operators),
                           c("analysis_oracle_own_mean",
                             "selected_saturated_rescue"))) {
      score[[method]][outer_valid, ] <- score_factorial_clusters(
        valid_residual,
        data$profile_index[outer_valid, , drop = FALSE],
        geometry$riesz_lookup, common_operators[[method]], final_fit$plugin
      )
    }
    score$selected_saturated_rescue[outer_valid, ] <-
      score_factorial_clusters(
        data$outcome[outer_valid, , drop = FALSE] -
          saturated_final_fit$prediction[outer_valid, , drop = FALSE],
        data$profile_index[outer_valid, , drop = FALSE],
        geometry$riesz_lookup, selected, saturated_final_fit$plugin
      )
    rescue_if_operator <- gate_e2_saturated_rescue_if_operator(
      selected, true_R, task_weights, scenario$true_assignment, support_true
    )
    inference_score$selected_saturated_rescue[outer_valid, ] <-
      score_factorial_clusters(
        data$outcome[outer_valid, , drop = FALSE] -
          saturated_final_fit$prediction[outer_valid, , drop = FALSE],
        data$profile_index[outer_valid, , drop = FALSE],
        geometry$riesz_lookup, rescue_if_operator,
        saturated_final_fit$plugin
      )
    score$analysis_oracle_own_mean[outer_valid, ] <-
      score_factorial_clusters(
        data$outcome[outer_valid, , drop = FALSE] -
          data$base_mean[outer_valid, , drop = FALSE],
        data$profile_index[outer_valid, , drop = FALSE],
        geometry$riesz_lookup, oracle_analysis, base_truth
      )
    score$ordinary_joint_oracle[outer_valid, ] <-
      score_factorial_clusters(
        data$outcome[outer_valid, , drop = FALSE] -
          data$mean[outer_valid, , drop = FALSE],
        data$profile_index[outer_valid, , drop = FALSE],
        geometry$riesz_lookup, ordinary, truth
      )
    selected_valid_score <- score$selected[outer_valid, , drop = FALSE]
    ordinary_valid_score <- score$ordinary[outer_valid, , drop = FALSE]
    score$feasible_sensitivity[outer_valid, ] <- ordinary_valid_score +
      sweep(selected_valid_score - ordinary_valid_score, 2L, generic$tau, "*")
    for (method in setdiff(method_names, "selected_saturated_rescue")) {
      inference_score[[method]][outer_valid, ] <-
        score[[method]][outer_valid, , drop = FALSE]
    }

    method_mixing <- lapply(common_operators, function(x) {
      gate_e2_operator_for_coordinate(x$mixing, coordinate_count)
    })
    method_mixing$ordinary_joint_oracle <-
      gate_e2_operator_for_coordinate(ordinary$mixing, coordinate_count)
    method_mixing$feasible_sensitivity <- feasible_mixing
    for (method in method_names) {
      if (identical(method, "ordinary_joint_oracle")) {
        predicted_bias_fold[[method]][outer, ] <- 0
      } else {
        eta <- if (identical(method, "analysis_oracle_own_mean")) {
          matrix(rep(as.numeric(base_truth), each = tasks),
                 tasks, coordinate_count,
                 dimnames = list(NULL, coordinate_names))
        } else if (identical(method, "selected_saturated_rescue")) {
          saturated_final_fit$target_coefficients
        } else final_fit$target_coefficients
        predicted_bias_fold[[method]][outer, ] <- vapply(
          seq_len(coordinate_count), function(j) {
            gate_e2_exact_bias(
              method_mixing[[method]][[j]], ordinary_true$mixing, true_R,
              task_weights, base_truth[[j]], eta[, j], departure,
              coordinate_names[[j]]
            )
          }, numeric(1)
        )
      }
      omega_fold[[method]][outer, ] <- vapply(
        seq_len(coordinate_count), function(j) {
          gate_e2_mixing_variance(
            method_mixing[[method]][[j]], sigma_true, true_R
          )
        }, numeric(1)
      )
      mixing_fold[[method]][[outer]] <- method_mixing[[method]]
    }
    max_calibration <- max(
      max_calibration,
      vapply(common_operators, `[[`, numeric(1), "calibration_residual")
    )
    tau_rows[[outer]] <- data.frame(
      scenario = scenario$id, replication = replication, outer_fold = outer,
      coordinate = coordinate_names,
      feasible_tau = generic$tau,
      oracle_unrestricted_tau = tau_unrestricted,
      oracle_pattern_tau = tau_pattern,
      local_radius = departure$local_radius,
      normalized_r = departure$normalized_r,
      selected_candidate = selection$candidate,
      selected_blend = selected$blend,
      k_hat_selected = k_hat_selected,
      stringsAsFactors = FALSE
    )
    selector_rows[[outer]] <- data.frame(
      scenario = scenario$id, replication = replication, outer_fold = outer,
      selected_candidate = selection$candidate,
      minimum_candidate = selection$minimum_candidate,
      selected_blend = selected$blend,
      selected_covariance_class = selected$covariance_class,
      inner_anchor_count = inner_plan$anchor_count,
      outer_saturated_empty_cells = saturated_final_fit$empty_cell_count,
      calibration_residual = selected$calibration_residual,
      stringsAsFactors = FALSE
    )
  }

  # Fold-specific estimators average linearly.  The sharp sensitivity operator
  # uses the effective representer, not the mean of fold-specific K values.
  feasible_assert(all(vapply(inference_score, function(x) all(is.finite(x)),
                              logical(1))),
                  "Gate-E2 inference scores are incomplete.")
  fold_fraction <- fold_fraction / sum(fold_fraction)
  sigma_hat_effective <- Reduce(`+`, Map(
    function(s, p) p * s, sigma_hat_fold, fold_fraction
  ))
  results <- list()
  diagnostics <- list()
  for (method in method_names) {
    point_summary <- weighted_score_summary(score[[method]], data$cluster_weight)
    inference_summary <- weighted_score_summary(
      inference_score[[method]], data$cluster_weight
    )
    summary <- point_summary
    summary$covariance <- inference_summary$covariance
    summary$standard_error <- inference_summary$standard_error
    summary$influence <- inference_summary$influence
    effective_mixing <- lapply(seq_len(coordinate_count), function(j) {
      Reduce(`+`, lapply(seq_along(fold_fraction), function(f) {
        fold_fraction[[f]] * mixing_fold[[method]][[f]][[j]]
      }))
    })
    predicted_bias <- drop(crossprod(
      fold_fraction, predicted_bias_fold[[method]]
    ))
    omega_reference <- drop(crossprod(fold_fraction, omega_fold[[method]]))
    k_true <- vapply(seq_len(coordinate_count), function(j) {
      gate_e2_mixing_k(
        effective_mixing[[j]] - ordinary_true$mixing, sigma_true, true_R
      )
    }, numeric(1))
    k_hat <- vapply(seq_len(coordinate_count), function(j) {
      gate_e2_mixing_k(
        effective_mixing[[j]] - ordinary_true$mixing,
        sigma_hat_effective, true_R
      )
    }, numeric(1))
    margin_true <- departure$epsilon * sqrt(pmax(k_true, 0))
    margin_hat <- departure$epsilon * sqrt(pmax(k_hat, 0))
    valid_sensitivity <- identical(analysis_assignment, scenario$true_assignment)
    if (!valid_sensitivity) {
      margin_true[] <- NA_real_
      margin_hat[] <- NA_real_
    }
    z <- stats::qnorm(0.975)
    half_add <- z * summary$standard_error + margin_hat
    half_add_true <- z * summary$standard_error + margin_true
    critical <- rep(NA_real_, coordinate_count)
    critical_true <- rep(NA_real_, coordinate_count)
    ok_flc <- is.finite(margin_hat) & summary$standard_error > 0
    ok_flc_true <- is.finite(margin_true) & summary$standard_error > 0
    critical[ok_flc] <- gate_e2_bias_aware_critical(
      margin_hat[ok_flc] / summary$standard_error[ok_flc]
    )
    critical_true[ok_flc_true] <- gate_e2_bias_aware_critical(
      margin_true[ok_flc_true] / summary$standard_error[ok_flc_true]
    )
    half_flc <- critical * summary$standard_error
    half_flc_true <- critical_true * summary$standard_error
    structured_margin <- rep(NA_real_, coordinate_count)
    if (valid_sensitivity && departure$pattern %in% c("lag", "blind")) {
      structured_margin <- vapply(seq_len(coordinate_count), function(j) {
        abs(gate_e2_departure_bias(
          effective_mixing[[j]] - ordinary_true$mixing,
          departure, true_R,
          coordinate_names[[j]]
        ))
      }, numeric(1))
    }
    half_structured <- z * summary$standard_error + structured_margin
    family_critical <- stats::qnorm(1 - 0.05 / (2 * coordinate_count))
    family_half_naive <- family_critical * summary$standard_error
    family_half_bias <- family_half_naive + margin_hat
    family_half_bias_true <- family_half_naive + margin_true
    family_covered_naive <- all(
      abs(summary$estimate - truth) <= family_half_naive
    )
    family_covered_bias <- if (all(is.finite(family_half_bias))) all(
      abs(summary$estimate - truth) <= family_half_bias
    ) else NA
    family_covered_bias_true <- if (all(is.finite(family_half_bias_true))) all(
      abs(summary$estimate - truth) <= family_half_bias_true
    ) else NA
    required_row_values <- list(
      reported_method = gate_e2_reported_method(method), truth = truth,
      estimate = summary$estimate, standard_error = summary$standard_error,
      predicted_bias = predicted_bias, omega_reference = omega_reference,
      k_true = k_true, k_hat = k_hat, half_add = half_add,
      half_flc = half_flc, structured_margin = structured_margin,
      margin_true = margin_true, margin_hat = margin_hat,
      scenario = scenario$id, scenario_class = scenario$class,
      true_assignment = scenario$true_assignment,
      analysis_assignment = analysis_assignment,
      covariance = scenario$covariance,
      residual_rho = scenario$residual_rho %||% scenario$rho,
      departure_pattern = departure$pattern,
      departure_feature = departure$feature,
      epsilon = departure$epsilon,
      local_radius = departure$local_radius,
      normalized_r = departure$normalized_r,
      departure_unit_epsilon = departure$unit_epsilon,
      departure_unit_bias_oracle = departure$unit_bias_oracle
    )
    empty_row_values <- names(required_row_values)[
      lengths(required_row_values) == 0L
    ]
    feasible_assert(!length(empty_row_values), paste0(
      "Empty Gate-E2 result fields: ", paste(empty_row_values, collapse = ", ")
    ))
    results[[method]] <- data.frame(
      scenario = scenario$id,
      scenario_class = scenario$class,
      replication = replication,
      seed = seed,
      method = gate_e2_reported_method(method),
      method_internal = method,
      inference_score = if (identical(method,
                                      "selected_saturated_rescue")) {
        "saturated_mean_first_order_corrected"
      } else "estimator_score",
      coordinate = coordinate_names,
      truth = truth,
      estimate = summary$estimate,
      error = summary$estimate - truth,
      standard_error = summary$standard_error,
      covered_naive = abs(summary$estimate - truth) <=
        z * summary$standard_error,
      predicted_bias = predicted_bias,
      omega_reference = omega_reference,
      theoretical_scaled_mse = omega_reference + n * predicted_bias^2,
      k_true = k_true,
      k_hat = k_hat,
      sensitivity_margin_true = margin_true,
      sensitivity_margin_hat = margin_hat,
      covered_bias_additive = ifelse(
        is.finite(half_add), abs(summary$estimate - truth) <= half_add, NA
      ),
      covered_bias_additive_oracle_k = ifelse(
        is.finite(half_add_true),
        abs(summary$estimate - truth) <= half_add_true, NA
      ),
      covered_bias_flc = ifelse(
        is.finite(half_flc), abs(summary$estimate - truth) <= half_flc, NA
      ),
      covered_bias_flc_oracle_k = ifelse(
        is.finite(half_flc_true),
        abs(summary$estimate - truth) <= half_flc_true, NA
      ),
      width_naive = 2 * z * summary$standard_error,
      width_bias_additive = 2 * half_add,
      width_bias_additive_oracle_k = 2 * half_add_true,
      width_bias_flc = 2 * half_flc,
      width_bias_flc_oracle_k = 2 * half_flc_true,
      family_covered_naive = family_covered_naive,
      family_covered_bias_additive = family_covered_bias,
      family_covered_bias_additive_oracle_k = family_covered_bias_true,
      structured_margin = structured_margin,
      covered_structured_additive = ifelse(
        is.finite(half_structured),
        abs(summary$estimate - truth) <= half_structured, NA
      ),
      width_structured_additive = 2 * half_structured,
      respondents = n,
      tasks = tasks,
      true_assignment = scenario$true_assignment,
      analysis_assignment = analysis_assignment,
      covariance = scenario$covariance,
      residual_rho = as.numeric(scenario$residual_rho %||% scenario$rho),
      departure_pattern = departure$pattern,
      departure_feature = departure$feature,
      epsilon = departure$epsilon,
      local_radius = departure$local_radius,
      normalized_r = departure$normalized_r,
      departure_unit_epsilon = departure$unit_epsilon,
      departure_unit_bias_oracle = departure$unit_bias_oracle,
      calibration_residual = max_calibration,
      stringsAsFactors = FALSE
    )
  }

  for (adaptive_method in c("selected", "oracle_restricted")) {
    difference <- score[[adaptive_method]] - score$ordinary
    difference_summary <- weighted_score_summary(
      difference, data$cluster_weight
    )
    estimate_difference <- difference_summary$estimate
    se_difference <- difference_summary$standard_error
    predicted_difference <-
      results[[adaptive_method]]$predicted_bias - results$ordinary$predicted_bias
    noncentrality <- predicted_difference / se_difference
    diagnostics[[adaptive_method]] <- data.frame(
      scenario = scenario$id,
      scenario_class = scenario$class,
      replication = replication,
      diagnostic = gate_e2_reported_method(adaptive_method),
      coordinate = coordinate_names,
      estimate_difference = estimate_difference,
      standard_error_difference = se_difference,
      predicted_difference = predicted_difference,
      noncentrality = noncentrality,
      predicted_power = gate_e2_normal_power(noncentrality),
      rejected_0_05 = abs(estimate_difference / se_difference) >
        stats::qnorm(0.975),
      stable_rank = se_difference > 1e-10,
      stringsAsFactors = FALSE
    )
  }

  list(
    results = do.call(rbind, results),
    diagnostics = do.call(rbind, diagnostics),
    selector = do.call(rbind, selector_rows),
    tau = do.call(rbind, tau_rows),
    departure = data.frame(
      scenario = scenario$id, replication = replication,
      pattern = departure$pattern, feature = departure$feature,
      epsilon = departure$epsilon, local_radius = departure$local_radius,
      normalized_r = departure$normalized_r,
      unit_epsilon = departure$unit_epsilon,
      unit_bias_oracle = departure$unit_bias_oracle,
      target_shift = paste(format(truth - base_truth, digits = 16),
                           collapse = ";"),
      projection_residual = departure$projection_residual,
      stringsAsFactors = FALSE
    ),
    error = NULL
  )
}
