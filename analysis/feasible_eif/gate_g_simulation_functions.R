# Fresh Gate G simulation functions.  Frozen Gate E2 functions are sourced as
# dependencies but are not modified.

read_gate_g_config <- function(path) {
  if (!requireNamespace("yaml", quietly = TRUE)) {
    stop("Package `yaml` is required.", call. = FALSE)
  }
  path <- normalizePath(path, mustWork = TRUE)
  child <- yaml::read_yaml(path)
  feasible_assert(!is.null(child$inherits),
                  "Gate G requires an explicitly inherited frozen config.")
  parent <- read_gate_e2_config(file.path(dirname(path), child$inherits))
  child$inherits <- NULL
  child_scenarios <- child$simulation$scenarios
  child$simulation$scenarios <- NULL
  config <- utils::modifyList(parent, child)
  # utils::modifyList recursively merges unnamed lists by position.  Scenario
  # registries are declarations, not partial list edits, so replace them.
  config$simulation$scenarios <- child_scenarios
  config$seed <- as.integer(config$seed)
  config$replications_per_scenario <- as.integer(
    config$replications_per_scenario
  )
  config$monte_carlo_bootstrap_draws <- as.integer(
    config$monte_carlo_bootstrap_draws
  )
  config$monte_carlo_bootstrap_seed <- as.integer(
    config$monte_carlo_bootstrap_seed
  )
  config$simulation$default_respondents <- as.integer(
    config$simulation$default_respondents
  )
  config$simulation$outcome_noise_scale <- as.numeric(
    config$simulation$outcome_noise_scale
  )
  config$simulation$scenarios <- lapply(
    config$simulation$scenarios,
    function(scenario) {
      scenario$tasks <- as.integer(
        scenario$tasks %||% config$simulation$default_tasks
      )
      scenario$respondents <- as.integer(
        scenario$respondents %||% config$simulation$default_respondents
      )
      scenario$covariance <- scenario$covariance %||%
        scenario$residual_covariance
      scenario$rho <- as.numeric(scenario$rho %||% scenario$residual_rho)
      scenario
    }
  )
  ids <- vapply(config$simulation$scenarios, `[[`, character(1), "id")
  feasible_assert(length(ids) >= 1L && !anyDuplicated(ids),
                  "Gate-G scenario identifiers must be nonempty and unique.")
  config$.path <- path
  config
}

gate_g_inner_covariance_fit <- function(
    profile_index, outcome, geometry, task_weights, inner_folds,
    seed, relative_floor = 1e-8) {
  inner_fold <- tryCatch(
    balanced_supported_cluster_folds(
      profile_index, inner_folds, seed, geometry$profile_count
    ),
    error = function(error) NULL
  )
  if (is.null(inner_fold)) {
    # A cell observed only once cannot appear in every inner training sample.
    # The exact projection-df correction uses the complete outer training
    # sample instead and remains outcome-blind with respect to the outer fold.
    fallback <- gate_g_saturated_residual_df_covariances(
      profile_index, outcome, geometry, task_weights, relative_floor
    )
    correction_factor <- fallback$degrees_of_freedom / nrow(outcome)
    return(list(
      raw = fallback$raw,
      corrected = c(
        fallback$corrected,
        list(
          inflation = correction_factor,
          raw_second_moment = fallback$raw_second_moment,
          corrected_second_moment = fallback$corrected_second_moment
        )
      ),
      inner_fold = rep(NA_integer_, nrow(outcome)),
      residual = fallback$residual,
      strategy = "full_residual_df_fallback",
      empty_cell_count = fallback$empty_cell_count
    ))
  }
  oof <- gate_g_saturated_oof_residual(
    profile_index, outcome, inner_fold, geometry, task_weights
  )
  feasible_assert(oof$empty_cell_count == 0L,
                  "Supported Gate-G inner folds produced an empty mean cell.")
  raw <- estimate_working_covariances(
    oof$residual, relative_floor = relative_floor
  )
  corrected <- gate_g_count_corrected_covariances(
    oof$residual, profile_index, inner_fold, geometry$profile_count,
    relative_floor
  )
  list(
    raw = raw,
    corrected = corrected,
    inner_fold = inner_fold,
    residual = oof$residual,
    strategy = "oof_count_correction",
    empty_cell_count = oof$empty_cell_count
  )
}

gate_g_method_result <- function(
    method, estimate, score, linear_load, truth, true_mean,
    sigma_true, first_order_scaled_variance, respondents,
    scenario, replication, seed, exact_for_realized_operator,
    estimated_covariance = NULL, calibration_residual = 0) {
  coordinate_names <- names(truth)
  coordinate_count <- length(coordinate_names)
  propagated_covariance <- gate_g_conditional_covariance(
    linear_load, sigma_true
  )
  frozen_operator_mean <- vapply(seq_len(coordinate_count), function(coordinate) {
    sum(linear_load[, , coordinate] * true_mean)
  }, numeric(1))
  names(frozen_operator_mean) <- coordinate_names
  frozen_operator_bias <- frozen_operator_mean - truth
  if (is.null(score)) {
    sandwich_variance <- rep(NA_real_, coordinate_count)
  } else {
    sandwich_variance <- diag(weighted_score_summary(score)$covariance)
  }
  if (is.null(estimated_covariance)) {
    plugin_covariance <- propagated_covariance
  } else {
    plugin_covariance <- gate_g_conditional_covariance(
      linear_load, estimated_covariance
    )
  }
  z <- stats::qnorm(0.975)
  propagated_se <- sqrt(pmax(diag(propagated_covariance), 0))
  plugin_se <- sqrt(pmax(diag(plugin_covariance), 0))
  sandwich_se <- sqrt(pmax(sandwich_variance, 0))
  data.frame(
    scenario = scenario$id,
    scenario_class = scenario$class,
    replication = replication,
    seed = seed,
    respondents = respondents,
    method = method,
    coordinate = coordinate_names,
    truth = as.numeric(truth),
    estimate = as.numeric(estimate),
    error = as.numeric(estimate - truth),
    frozen_operator_mean = as.numeric(frozen_operator_mean),
    frozen_operator_bias = as.numeric(frozen_operator_bias),
    propagated_variance = diag(propagated_covariance),
    plugin_propagated_variance = diag(plugin_covariance),
    sandwich_variance = sandwich_variance,
    first_order_scaled_variance = rep(
      as.numeric(first_order_scaled_variance),
      length.out = coordinate_count
    ),
    predicted_mse_propagated =
      diag(propagated_covariance) + frozen_operator_bias^2,
    covered_propagated = abs(estimate - truth) <= z * propagated_se,
    covered_plugin = abs(estimate - truth) <= z * plugin_se,
    covered_sandwich = ifelse(
      is.finite(sandwich_se),
      abs(estimate - truth) <= z * sandwich_se,
      NA
    ),
    propagated_variance_is_exact = exact_for_realized_operator,
    calibration_residual = calibration_residual,
    stringsAsFactors = FALSE
  )
}

gate_g_crossfit_method <- function(
    method, operators, data, outer_fold, geometry, task_weights,
    truth, sigma_true, scenario, replication, seed,
    first_order_scaled_variance, exact_for_realized_operator,
    estimated_covariance = NULL) {
  linearized <- gate_g_crossfit_linearization(
    data$profile_index, data$outcome, outer_fold, operators,
    geometry, task_weights
  )
  calibration_residual <- max(vapply(operators, function(operator) {
    if (is.null(operator$calibration_residual)) return(0)
    operator$calibration_residual
  }, numeric(1)))
  list(
    result = gate_g_method_result(
      method = method,
      estimate = linearized$estimate,
      score = linearized$score,
      linear_load = linearized$linear_load,
      truth = truth,
      true_mean = data$mean,
      sigma_true = sigma_true,
      first_order_scaled_variance = first_order_scaled_variance,
      respondents = data$respondents,
      scenario = scenario,
      replication = replication,
      seed = seed,
      exact_for_realized_operator = exact_for_realized_operator,
      estimated_covariance = estimated_covariance,
      calibration_residual = calibration_residual
    ),
    linearized = linearized
  )
}

gate_g_gls_result <- function(
    method, gls, working_covariance, data, truth, sigma_true,
    first_order_scaled_variance, scenario, replication, seed,
    exact_for_realized_operator) {
  gate_g_method_result(
    method = method,
    estimate = gls$estimate,
    score = NULL,
    linear_load = gls$linear_load,
    truth = truth,
    true_mean = data$mean,
    sigma_true = sigma_true,
    first_order_scaled_variance = first_order_scaled_variance,
    respondents = data$respondents,
    scenario = scenario,
    replication = replication,
    seed = seed,
    exact_for_realized_operator = exact_for_realized_operator,
    estimated_covariance = working_covariance,
    calibration_residual = gls$estimability_residual
  )
}

gate_g_split_gls <- function(
    profile_index, outcome, split_fold, geometry, task_weights,
    covariance_kind = c("raw", "df_corrected"), relative_floor = 1e-8) {
  covariance_kind <- match.arg(covariance_kind)
  profile_index <- as.matrix(profile_index)
  outcome <- as.matrix(outcome)
  split_fold <- as.integer(split_fold)
  split_values <- sort(unique(split_fold))
  feasible_assert(length(split_values) == 2L,
                  "Honest split GLS requires exactly two supported halves.")
  n <- nrow(outcome)
  tasks <- ncol(outcome)
  coordinate_count <- ncol(geometry$riesz_lookup)
  estimate <- numeric(coordinate_count)
  linear_load <- array(0, c(n, tasks, coordinate_count))
  diagnostics <- vector("list", 2L)

  for (position in seq_along(split_values)) {
    estimation_fold <- split_values[[position]]
    estimate_rows <- split_fold == estimation_fold
    covariance_rows <- !estimate_rows
    covariance_fit <- gate_g_saturated_residual_df_covariances(
      profile_index[covariance_rows, , drop = FALSE],
      outcome[covariance_rows, , drop = FALSE],
      geometry, task_weights, relative_floor
    )
    working_covariance <- if (identical(covariance_kind, "raw")) {
      covariance_fit$raw$compound_symmetry$covariance
    } else {
      covariance_fit$corrected$compound_symmetry$covariance
    }
    fit <- gate_g_full_oracle_gls(
      profile_index[estimate_rows, , drop = FALSE],
      outcome[estimate_rows, , drop = FALSE],
      geometry, task_weights, working_covariance
    )
    estimate <- estimate + fit$estimate / 2
    linear_load[estimate_rows, , ] <- fit$linear_load / 2
    diagnostics[[position]] <- data.frame(
      estimation_fold = estimation_fold,
      estimation_count = sum(estimate_rows),
      covariance_training_count = sum(covariance_rows),
      covariance_kind = covariance_kind,
      covariance_diagonal = mean(diag(working_covariance)),
      covariance_off_diagonal = mean(working_covariance[
        row(working_covariance) != col(working_covariance)
      ]),
      minimum_covariance_df = min(covariance_fit$degrees_of_freedom),
      information_rank = fit$information_rank,
      estimability_residual = fit$estimability_residual,
      strategy = "honest_supported_split",
      stringsAsFactors = FALSE
    )
  }
  representation <- vapply(seq_len(coordinate_count), function(coordinate) {
    sum(linear_load[, , coordinate] * outcome)
  }, numeric(1))
  feasible_assert(max(abs(representation - estimate)) <= 1e-10,
                  "Honest split-GLS loading representation failed.")
  list(
    estimate = estimate,
    linear_load = linear_load,
    diagnostics = do.call(rbind, diagnostics),
    covariance_kind = covariance_kind
  )
}

run_gate_g_replication <- function(scenario, replication, config, geometry) {
  seed <- gate_e2_scenario_seed(scenario, replication, config)
  set.seed(seed)
  scenario_for_draw <- scenario
  scenario_for_draw$assignment <- scenario$true_assignment
  data <- draw_gate_d_data(scenario_for_draw, config, geometry)
  n <- data$respondents
  tasks <- data$tasks
  coordinate_count <- ncol(geometry$riesz_lookup)
  task_weights <- rep(1 / tasks, tasks)
  sigma_true <- scenario_covariance(scenario_for_draw, tasks) *
    as.numeric(config$simulation$outcome_noise_scale)^2
  ordinary <- ordinary_task_operator(
    tasks, task_weights, diag(tasks), "product"
  )
  oracle <- solve_task_mixing(
    sigma_true, task_weights, "product"
  )
  base_truth <- geometry$truth
  departure <- gate_e2_make_departure(
    scenario, n, geometry, sigma_true, ordinary, oracle
  )
  data$base_mean <- data$mean
  data <- gate_e2_apply_departure(data, departure, geometry)
  truth <- gate_e2_target_truth(
    geometry, departure, ordinary$mixing, diag(tasks), base_truth
  )

  # Outer folds are assignment-independent and respondent-balanced. The
  # frozen saturated learner's explicit task-mean fallback handles a rare
  # empty training cell, and its exact linear influence is propagated below.
  outer_fold <- balanced_cluster_folds(
    n, as.integer(config$estimator$outer_folds), seed + 1L
  )
  fold_values <- sort(unique(outer_fold))
  ordinary_operators <- rep(list(ordinary), length(fold_values))
  oracle_operators <- rep(list(oracle), length(fold_values))
  blend_values <- c(0.25, 0.50, 0.75)
  blend_operators <- lapply(blend_values, function(blend) {
    rep(list(blend_task_operator(
      ordinary, oracle, blend, "oracle"
    )), length(fold_values))
  })
  names(blend_operators) <- sprintf("oracle_tau_%0.2f_crossfit", blend_values)

  raw_operators <- vector("list", length(fold_values))
  count_operators <- vector("list", length(fold_values))
  finite_g_operators <- vector("list", length(fold_values))
  feasible_finite_g_operators <- vector("list", length(fold_values))
  covariance_rows <- vector("list", length(fold_values))
  raw_covariance_average <- matrix(0, tasks, tasks)
  count_covariance_average <- matrix(0, tasks, tasks)
  fold_fraction <- numeric(length(fold_values))
  relative_floor <- as.numeric(
    config$estimator$covariance_eigenvalue_relative_floor
  )

  for (fold_position in seq_along(fold_values)) {
    fold <- fold_values[[fold_position]]
    outer_train <- outer_fold != fold
    fold_fraction[[fold_position]] <- mean(!outer_train)
    covariance_fit <- gate_g_inner_covariance_fit(
      data$profile_index[outer_train, , drop = FALSE],
      data$outcome[outer_train, , drop = FALSE],
      geometry, task_weights,
      as.integer(config$estimator$inner_folds),
      seed + 1000L * fold,
      relative_floor
    )
    raw_covariance <-
      covariance_fit$raw$compound_symmetry$covariance
    count_covariance <-
      covariance_fit$corrected$compound_symmetry$covariance
    raw_operators[[fold_position]] <- solve_task_mixing(
      raw_covariance, task_weights, "product"
    )
    count_operators[[fold_position]] <- solve_task_mixing(
      count_covariance, task_weights, "product"
    )
    outer_training_count <- sum(outer_train)
    finite_g_operators[[fold_position]] <- gate_g_count_aware_operator(
      sigma_true, outer_training_count, geometry$profile_count,
      task_weights
    )
    feasible_finite_g_operators[[fold_position]] <-
      gate_g_count_aware_operator(
        count_covariance, outer_training_count, geometry$profile_count,
        task_weights
      )
    raw_covariance_average <- raw_covariance_average +
      fold_fraction[[fold_position]] * raw_covariance
    count_covariance_average <- count_covariance_average +
      fold_fraction[[fold_position]] * count_covariance
    covariance_rows[[fold_position]] <- data.frame(
      scenario = scenario$id,
      replication = replication,
      seed = seed,
      outer_fold = fold,
      respondents = n,
      raw_frobenius_squared = sum((raw_covariance - sigma_true)^2),
      count_frobenius_squared = sum((count_covariance - sigma_true)^2),
      raw_diagonal = mean(diag(raw_covariance)),
      count_diagonal = mean(diag(count_covariance)),
      true_diagonal = mean(diag(sigma_true)),
      raw_off_diagonal = mean(raw_covariance[
        row(raw_covariance) != col(raw_covariance)
      ]),
      count_off_diagonal = mean(count_covariance[
        row(count_covariance) != col(count_covariance)
      ]),
      true_off_diagonal = mean(sigma_true[
        row(sigma_true) != col(sigma_true)
      ]),
      maximum_inflation = max(covariance_fit$corrected$inflation),
      minimum_inflation = min(covariance_fit$corrected$inflation),
      covariance_correction_strategy = covariance_fit$strategy,
      stringsAsFactors = FALSE
    )
  }

  # Normalize defensively if fold sizes differ by one respondent.
  fold_fraction <- fold_fraction / sum(fold_fraction)
  raw_covariance_average <- raw_covariance_average / sum(fold_fraction)
  count_covariance_average <- count_covariance_average / sum(fold_fraction)

  oracle_first_order <- gate_e2_mixing_variance(
    oracle$mixing, sigma_true, diag(tasks)
  )
  ordinary_first_order <- gate_e2_mixing_variance(
    ordinary$mixing, sigma_true, diag(tasks)
  )
  average_first_order <- function(operators) {
    sum(vapply(seq_along(operators), function(index) {
      fold_fraction[[index]] * gate_e2_mixing_variance(
        gate_g_operator_mixing(operators[[index]]), sigma_true, diag(tasks)
      )
    }, numeric(1)))
  }

  output <- list()
  ordinary_fit <- gate_g_crossfit_method(
    "ordinary_crossfit", ordinary_operators, data, outer_fold,
    geometry, task_weights, truth, sigma_true, scenario, replication, seed,
    ordinary_first_order, TRUE
  )
  output[["ordinary_crossfit"]] <- ordinary_fit$result
  output[["oracle_hstar_crossfit"]] <- gate_g_crossfit_method(
    "oracle_hstar_crossfit", oracle_operators, data, outer_fold,
    geometry, task_weights, truth, sigma_true, scenario, replication, seed,
    oracle_first_order, TRUE
  )$result
  output[["oracle_finite_g_crossfit"]] <- gate_g_crossfit_method(
    "oracle_finite_g_crossfit", finite_g_operators, data, outer_fold,
    geometry, task_weights, truth, sigma_true, scenario, replication, seed,
    average_first_order(finite_g_operators), TRUE
  )$result
  for (name in names(blend_operators)) {
    output[[name]] <- gate_g_crossfit_method(
      name, blend_operators[[name]], data, outer_fold,
      geometry, task_weights, truth, sigma_true, scenario, replication, seed,
      average_first_order(blend_operators[[name]]), TRUE
    )$result
  }
  output[["feasible_raw_hstar_crossfit"]] <- gate_g_crossfit_method(
    "feasible_raw_hstar_crossfit", raw_operators, data, outer_fold,
    geometry, task_weights, truth, sigma_true, scenario, replication, seed,
    average_first_order(raw_operators), FALSE,
    raw_covariance_average
  )$result
  output[["feasible_count_hstar_crossfit"]] <- gate_g_crossfit_method(
    "feasible_count_hstar_crossfit", count_operators, data, outer_fold,
    geometry, task_weights, truth, sigma_true, scenario, replication, seed,
    average_first_order(count_operators), FALSE,
    count_covariance_average
  )$result
  output[["feasible_count_finite_g_crossfit"]] <- gate_g_crossfit_method(
    "feasible_count_finite_g_crossfit", feasible_finite_g_operators,
    data, outer_fold, geometry, task_weights, truth, sigma_true,
    scenario, replication, seed,
    average_first_order(feasible_finite_g_operators), FALSE,
    count_covariance_average
  )$result

  # Known-mean oracle: there is no shared training-mean contribution.
  known_score <- matrix(NA_real_, n, coordinate_count)
  known_load <- array(0, c(n, tasks, coordinate_count))
  for (respondent in seq_len(n)) {
    assigned_riesz <- geometry$riesz_lookup[
      data$profile_index[respondent, ], , drop = FALSE
    ]
    known_load[respondent, , ] <- oracle$mixing %*% assigned_riesz / n
  }
  known_score[,] <- score_factorial_clusters(
    data$outcome - data$base_mean,
    data$profile_index,
    geometry$riesz_lookup,
    oracle,
    base_truth
  )
  known_result <- gate_g_method_result(
    "known_mean_oracle", colMeans(known_score), known_score, known_load,
    truth, data$mean - data$base_mean, sigma_true, oracle_first_order,
    n, scenario, replication, seed, TRUE
  )
  # gate_g_method_result treats the supplied true mean as the entire linear
  # mean. Add the fixed plug-in target back to its frozen-operator diagnostics.
  known_result$frozen_operator_mean <-
    known_result$frozen_operator_mean + base_truth
  known_result$frozen_operator_bias <-
    known_result$frozen_operator_mean - truth
  known_result$predicted_mse_propagated <-
    known_result$propagated_variance +
    known_result$frozen_operator_bias^2
  output[["known_mean_oracle"]] <- known_result

  # A global OOF covariance estimate supports the finite-support feasible GLS
  # comparison. This is a benchmark for the restricted mean model, not a
  # replacement for general covariate-adjusted cross-fitting.
  global_covariance_fit <- gate_g_inner_covariance_fit(
    data$profile_index, data$outcome, geometry, task_weights,
    as.integer(config$estimator$inner_folds), seed + 90000L,
    relative_floor
  )
  global_raw_covariance <-
    global_covariance_fit$raw$compound_symmetry$covariance
  global_count_covariance <-
    global_covariance_fit$corrected$compound_symmetry$covariance
  oracle_gls <- gate_g_full_oracle_gls(
    data$profile_index, data$outcome, geometry, task_weights, sigma_true
  )
  raw_gls <- gate_g_full_oracle_gls(
    data$profile_index, data$outcome, geometry, task_weights,
    global_raw_covariance
  )
  count_gls <- gate_g_full_oracle_gls(
    data$profile_index, data$outcome, geometry, task_weights,
    global_count_covariance
  )
  output[["oracle_full_gls"]] <- gate_g_gls_result(
    "oracle_full_gls", oracle_gls, sigma_true, data, truth, sigma_true,
    oracle_first_order, scenario, replication, seed, TRUE
  )
  output[["feasible_raw_full_gls"]] <- gate_g_gls_result(
    "feasible_raw_full_gls", raw_gls, global_raw_covariance,
    data, truth, sigma_true, oracle_first_order,
    scenario, replication, seed, FALSE
  )
  output[["feasible_count_full_gls"]] <- gate_g_gls_result(
    "feasible_count_full_gls", count_gls, global_count_covariance,
    data, truth, sigma_true, oracle_first_order,
    scenario, replication, seed, FALSE
  )

  # Honest feasible GLS: each half's covariance is learned only from the
  # opposite half, so the weights are independent of the outcomes they weight.
  # Its point estimator is exactly centered under the restricted saturated
  # mean model. The frozen-weight propagated variance still omits randomness
  # from covariance learning and is therefore not marked exact.
  split_fold <- tryCatch(
    balanced_supported_cluster_folds(
      data$profile_index, 2L, seed + 120000L, geometry$profile_count
    ),
    error = function(error) NULL
  )
  if (is.null(split_fold)) {
    fallback_diagnostic <- function(kind) data.frame(
      estimation_fold = NA_integer_,
      estimation_count = n,
      covariance_training_count = 0L,
      covariance_kind = kind,
      covariance_diagonal = NA_real_,
      covariance_off_diagonal = NA_real_,
      minimum_covariance_df = NA_real_,
      information_rank = NA_integer_,
      estimability_residual = 0,
      strategy = "ordinary_support_fallback",
      stringsAsFactors = FALSE
    )
    split_raw <- list(
      estimate = ordinary_fit$linearized$estimate,
      linear_load = ordinary_fit$linearized$linear_load,
      diagnostics = fallback_diagnostic("raw")
    )
    split_df <- list(
      estimate = ordinary_fit$linearized$estimate,
      linear_load = ordinary_fit$linearized$linear_load,
      diagnostics = fallback_diagnostic("df_corrected")
    )
  } else {
    split_raw <- gate_g_split_gls(
      data$profile_index, data$outcome, split_fold, geometry, task_weights,
      "raw", relative_floor
    )
    split_df <- gate_g_split_gls(
      data$profile_index, data$outcome, split_fold, geometry, task_weights,
      "df_corrected", relative_floor
    )
  }
  output[["feasible_split_raw_gls"]] <- gate_g_method_result(
    "feasible_split_raw_gls", split_raw$estimate, NULL,
    split_raw$linear_load, truth, data$mean, sigma_true,
    oracle_first_order, n, scenario, replication, seed, FALSE
  )
  output[["feasible_split_df_gls"]] <- gate_g_method_result(
    "feasible_split_df_gls", split_df$estimate, NULL,
    split_df$linear_load, truth, data$mean, sigma_true,
    oracle_first_order, n, scenario, replication, seed, FALSE
  )

  list(
    results = do.call(rbind, output),
    covariance = do.call(rbind, covariance_rows),
    folds = data.frame(
      scenario = scenario$id,
      replication = replication,
      seed = seed,
      respondent = seq_len(n),
      outer_fold = outer_fold,
      stringsAsFactors = FALSE
    ),
    support_diagnostics = transform(
      ordinary_fit$linearized$fold_diagnostics,
      scenario = scenario$id,
      replication = replication,
      seed = seed
    ),
    split_diagnostics = rbind(
      transform(split_raw$diagnostics, method = "feasible_split_raw_gls"),
      transform(split_df$diagnostics, method = "feasible_split_df_gls")
    ),
    error = NULL
  )
}
