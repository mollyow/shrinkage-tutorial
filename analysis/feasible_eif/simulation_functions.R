`%||%` <- function(x, y) if (is.null(x)) y else x

read_gate_d_config <- function(path) {
  if (!requireNamespace("yaml", quietly = TRUE)) {
    stop("Package `yaml` is required.", call. = FALSE)
  }
  path <- normalizePath(path, mustWork = TRUE)
  config <- yaml::read_yaml(path)
  if (!is.null(config$inherits)) {
    parent <- read_gate_d_config(file.path(dirname(path), config$inherits))
    config$inherits <- NULL
    config <- utils::modifyList(parent, config)
  }
  required <- c("version", "mode", "seed", "replications_per_scenario",
                "estimand", "estimator", "simulation", "pass_rules")
  missing <- setdiff(required, names(config))
  if (length(missing)) {
    stop("Gate-D configuration is missing: ", paste(missing, collapse = ", "),
         call. = FALSE)
  }
  config$seed <- as.integer(config$seed)
  config$replications_per_scenario <- as.integer(config$replications_per_scenario)
  config$monte_carlo_bootstrap_draws <-
    as.integer(config$monte_carlo_bootstrap_draws)
  config$monte_carlo_bootstrap_seed <-
    as.integer(config$monte_carlo_bootstrap_seed)
  feasible_assert(config$version == 1L && config$replications_per_scenario >= 1L,
                  "The Gate-D configuration version or replication count is invalid.")
  feasible_assert(length(config$simulation$scenarios) >= 1L,
                  "At least one Gate-D scenario is required.")
  config$.path <- path
  config
}

gate_d_geometry <- function(config) {
  factor_count <- as.integer(config$estimand$factors)
  feasible_assert(factor_count == 4L &&
                    as.integer(config$estimand$levels_per_factor) == 2L,
                  "Gate-D version 1 uses the frozen four-factor binary geometry.")
  profiles <- expand.grid(
    rep(list(c(-1, 1)), factor_count),
    KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
  )
  names(profiles) <- paste0("x", seq_len(factor_count))
  with(profiles, {
    contrast <- cbind(
      x1_x2 = x1 * x2,
      x1_x3 = x1 * x3,
      x2_x4 = x2 * x4,
      x1_x2_x3 = x1 * x2 * x3
    )
    requested <- unlist(config$estimand$coordinates, use.names = FALSE)
    contrast <- contrast[, requested, drop = FALSE]
    invisible(validate_centered_riesz_lookup(contrast))
    coefficients <- as.numeric(unlist(config$estimand$true_coefficients))
    feasible_assert(length(coefficients) == ncol(contrast),
                    "The frozen truth needs one coefficient per target coordinate.")
    names(coefficients) <- colnames(contrast)
    # All additional terms are distinct orthogonal factorial directions.
    mean_surface <- drop(contrast %*% coefficients) +
      0.16 * x1 - 0.12 * x2 + 0.09 * x3 + 0.06 * x4 +
      0.07 * x1 * x4 - 0.05 * x2 * x3 + 0.04 * x1 * x2 * x4
    list(
      profiles = profiles,
      profile_scores = as.matrix(profiles),
      riesz_lookup = contrast,
      truth = coefficients,
      mean_surface = mean_surface,
      profile_count = nrow(profiles),
      target_probability = rep(1 / nrow(profiles), nrow(profiles))
    )
  })
}

balanced_cluster_folds <- function(n, folds, seed) {
  set.seed(as.integer(seed))
  order <- sample.int(n)
  out <- integer(n)
  out[order] <- rep(seq_len(as.integer(folds)), length.out = n)
  out
}

draw_profile_matrix <- function(n, tasks, profile_count, assignment) {
  if (identical(assignment, "product")) {
    return(matrix(sample.int(profile_count, n * tasks, replace = TRUE), n, tasks))
  }
  if (identical(assignment, "uniform_without_replacement")) {
    feasible_assert(tasks <= profile_count,
                    "Without-replacement tasks exceed the profile support.")
    return(t(replicate(n, sample.int(profile_count, tasks, replace = FALSE))))
  }
  stop("Unknown assignment: ", assignment, call. = FALSE)
}

scenario_covariance <- function(scenario, tasks) {
  covariance_name <- scenario$covariance
  rho <- as.numeric(scenario$rho)
  if (identical(covariance_name, "compound_symmetry") ||
      identical(covariance_name, "profile_scaled_compound_symmetry")) {
    covariance <- matrix(rho, tasks, tasks)
    diag(covariance) <- 1
  } else if (identical(covariance_name, "ar1")) {
    covariance <- outer(seq_len(tasks), seq_len(tasks), function(a, b) {
      rho^abs(a - b)
    })
  } else {
    stop("Unknown covariance structure: ", covariance_name, call. = FALSE)
  }
  eigenvalues <- eigen(covariance, symmetric = TRUE, only.values = TRUE)$values
  feasible_assert(min(eigenvalues) > 1e-10,
                  paste("The scenario covariance is not positive definite:",
                        scenario$id))
  covariance
}

draw_gate_d_data <- function(scenario, config, geometry) {
  n <- as.integer(scenario$respondents %||% config$simulation$default_respondents)
  tasks <- as.integer(scenario$tasks)
  assignment <- scenario$assignment
  profile_index <- draw_profile_matrix(
    n, tasks, geometry$profile_count, assignment
  )
  task_drift <- seq(-0.08, 0.08, length.out = tasks)
  mean <- matrix(geometry$mean_surface[profile_index], n, tasks) +
    matrix(task_drift, n, tasks, byrow = TRUE)
  covariance <- scenario_covariance(scenario, tasks)
  innovation <- matrix(stats::rnorm(n * tasks), n, tasks) %*% chol(covariance)
  if (identical(scenario$covariance, "profile_scaled_compound_symmetry")) {
    coefficient <- as.numeric(scenario$log_sd_coefficient)
    x1 <- matrix(
      geometry$profile_scores[as.vector(profile_index), "x1"], n, tasks
    )
    innovation <- innovation * exp(coefficient * x1)
  }
  outcome <- mean + as.numeric(config$simulation$outcome_noise_scale) * innovation
  list(
    outcome = outcome,
    mean = mean,
    profile_index = profile_index,
    covariance = covariance,
    respondents = n,
    tasks = tasks,
    assignment = assignment,
    cluster_weight = rep(1, n)
  )
}

task_additive_design <- function(profile_index, geometry) {
  profile_index <- as.matrix(profile_index)
  n <- nrow(profile_index)
  tasks <- ncol(profile_index)
  block_width <- 1L + ncol(geometry$profile_scores)
  design <- matrix(0, n * tasks, tasks * block_width)
  for (tt in seq_len(tasks)) {
    rows <- seq.int((tt - 1L) * n + 1L, tt * n)
    columns <- seq.int((tt - 1L) * block_width + 1L, tt * block_width)
    design[rows, columns] <- cbind(
      1, geometry$profile_scores[profile_index[, tt], , drop = FALSE]
    )
  }
  design
}

target_additive_design <- function(tasks, geometry) {
  profile_count <- geometry$profile_count
  block_width <- 1L + ncol(geometry$profile_scores)
  design <- matrix(0, profile_count * tasks, tasks * block_width)
  for (tt in seq_len(tasks)) {
    rows <- seq.int((tt - 1L) * profile_count + 1L, tt * profile_count)
    columns <- seq.int((tt - 1L) * block_width + 1L, tt * block_width)
    design[rows, columns] <- cbind(1, geometry$profile_scores)
  }
  design
}

fit_task_additive <- function(design, target_design, outcome, train_respondent,
                              geometry, task_weights) {
  outcome <- as.matrix(outcome)
  n <- nrow(outcome)
  tasks <- ncol(outcome)
  train_rows <- rep(as.logical(train_respondent), tasks)
  fit <- stats::lm.fit(design[train_rows, , drop = FALSE],
                       as.vector(outcome)[train_rows])
  coefficient <- fit$coefficients
  coefficient[is.na(coefficient)] <- 0
  feasible_assert(all(is.finite(coefficient)),
                  "The task-specific additive fit produced nonfinite coefficients.")
  prediction <- matrix(drop(design %*% coefficient), n, tasks)
  target_prediction <- matrix(
    drop(target_design %*% coefficient), geometry$profile_count, tasks
  )
  plugin <- numeric(ncol(geometry$riesz_lookup))
  for (tt in seq_len(tasks)) {
    plugin <- plugin + task_weights[[tt]] * drop(crossprod(
      geometry$target_probability,
      geometry$riesz_lookup * target_prediction[, tt]
    ))
  }
  list(
    prediction = prediction,
    plugin = plugin,
    rank = fit$rank,
    coefficients = coefficient
  )
}

constant_oracle_covariance <- function(data, scenario, geometry) {
  if (!identical(scenario$covariance, "profile_scaled_compound_symmetry")) {
    return(data$covariance)
  }
  coefficient <- as.numeric(scenario$log_sd_coefficient)
  sd_profile <- exp(coefficient * geometry$profile_scores[, "x1"])
  diagonal <- mean(sd_profile^2)
  off_scale <- mean(sd_profile)^2
  out <- data$covariance * off_scale
  diag(out) <- diagonal
  out
}

run_gate_d_replication <- function(scenario, replication, config, geometry) {
  scenario_index <- match(
    scenario$id,
    vapply(config$simulation$scenarios, `[[`, character(1), "id")
  )
  seed <- as.integer(config$seed + scenario_index * 100000L + replication)
  set.seed(seed)
  data <- draw_gate_d_data(scenario, config, geometry)
  n <- data$respondents
  tasks <- data$tasks
  task_weights <- rep(1 / tasks, tasks)
  design <- task_additive_design(data$profile_index, geometry)
  target_design <- target_additive_design(tasks, geometry)
  outer_folds <- balanced_cluster_folds(
    n, as.integer(config$estimator$outer_folds), seed + 1L
  )
  method_names <- c(
    "ordinary", "selected", "safe_half_cs", "learned_cs_full",
    "learned_unstructured_full", "oracle"
  )
  score <- stats::setNames(lapply(method_names, function(x) {
    matrix(NA_real_, n, ncol(geometry$riesz_lookup),
           dimnames = list(NULL, colnames(geometry$riesz_lookup)))
  }), method_names)
  selector_rows <- list()
  max_calibration <- 0

  for (outer in sort(unique(outer_folds))) {
    outer_valid <- outer_folds == outer
    outer_train <- !outer_valid
    train_ids <- which(outer_train)
    inner_local <- balanced_cluster_folds(
      length(train_ids), as.integer(config$estimator$inner_folds),
      seed + 100L * outer
    )
    inner_fold <- integer(n)
    inner_fold[train_ids] <- inner_local
    residual_oof <- matrix(NA_real_, n, tasks)
    inner_plugin <- vector("list", max(inner_local))
    for (inner in sort(unique(inner_local))) {
      inner_valid <- outer_train & inner_fold == inner
      inner_train <- outer_train & inner_fold != inner
      fit <- fit_task_additive(
        design, target_design, data$outcome, inner_train,
        geometry, task_weights
      )
      residual_oof[inner_valid, ] <-
        data$outcome[inner_valid, , drop = FALSE] -
        fit$prediction[inner_valid, , drop = FALSE]
      inner_plugin[[inner]] <- fit$plugin
    }
    feasible_assert(all(is.finite(residual_oof[outer_train, , drop = FALSE])),
                    "Inner cross-fitted residuals are incomplete.")

    candidate_scores <- NULL
    candidate_metadata <- NULL
    for (inner in sort(unique(inner_local))) {
      selector_valid <- outer_train & inner_fold == inner
      covariance_train <- outer_train & inner_fold != inner
      covariance_set <- estimate_working_covariances(
        residual_oof[covariance_train, , drop = FALSE],
        data$cluster_weight[covariance_train],
        as.numeric(config$estimator$covariance_eigenvalue_relative_floor)
      )
      library <- make_operator_library(
        covariance_set, task_weights, data$assignment,
        if (identical(data$assignment, "uniform_without_replacement")) {
          geometry$profile_count
        } else NULL,
        as.numeric(unlist(config$estimator$blend_grid))
      )
      if (is.null(candidate_scores)) {
        candidate_metadata <- attr(library, "metadata")
        candidate_scores <- stats::setNames(lapply(names(library), function(x) {
          matrix(NA_real_, sum(outer_train), ncol(geometry$riesz_lookup))
        }), names(library))
        training_position <- integer(n)
        training_position[outer_train] <- seq_len(sum(outer_train))
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
                    "Selector scores are incomplete.")
    selection <- select_operator_one_se(
      candidate_scores, candidate_metadata,
      data$cluster_weight[outer_train]
    )

    final_covariance_set <- estimate_working_covariances(
      residual_oof[outer_train, , drop = FALSE],
      data$cluster_weight[outer_train],
      as.numeric(config$estimator$covariance_eigenvalue_relative_floor)
    )
    final_library <- make_operator_library(
      final_covariance_set, task_weights, data$assignment,
      if (identical(data$assignment, "uniform_without_replacement")) {
        geometry$profile_count
      } else NULL,
      as.numeric(unlist(config$estimator$blend_grid))
    )
    selected_operator <- final_library[[selection$candidate]]
    final_fit <- fit_task_additive(
      design, target_design, data$outcome, outer_train,
      geometry, task_weights
    )
    valid_residual <- data$outcome[outer_valid, , drop = FALSE] -
      final_fit$prediction[outer_valid, , drop = FALSE]

    oracle_covariance <- constant_oracle_covariance(data, scenario, geometry)
    oracle_operator <- solve_task_mixing(
      oracle_covariance, task_weights, data$assignment,
      if (identical(data$assignment, "uniform_without_replacement")) {
        geometry$profile_count
      } else NULL
    )
    operator_by_method <- list(
      ordinary = final_library$ordinary,
      selected = selected_operator,
      safe_half_cs = final_library$compound_symmetry_tau_0.5,
      learned_cs_full = final_library$compound_symmetry_tau_1,
      learned_unstructured_full = final_library$unstructured_tau_1,
      oracle = oracle_operator
    )
    for (method in names(operator_by_method)) {
      operator <- operator_by_method[[method]]
      score[[method]][outer_valid, ] <- score_factorial_clusters(
        valid_residual,
        data$profile_index[outer_valid, , drop = FALSE],
        geometry$riesz_lookup, operator, final_fit$plugin
      )
      max_calibration <- max(max_calibration, operator$calibration_residual)
    }
    selector_rows[[length(selector_rows) + 1L]] <- data.frame(
      scenario = scenario$id,
      replication = replication,
      outer_fold = outer,
      selected_candidate = selection$candidate,
      minimum_candidate = selection$minimum_candidate,
      selected_blend = selected_operator$blend,
      selected_covariance_class = selected_operator$covariance_class,
      cs_projected = final_covariance_set$compound_symmetry$projected,
      unstructured_projected = final_covariance_set$unstructured$projected,
      calibration_residual = selected_operator$calibration_residual,
      stringsAsFactors = FALSE
    )
  }

  results <- do.call(rbind, lapply(method_names, function(method) {
    summary <- weighted_score_summary(score[[method]], data$cluster_weight)
    data.frame(
      scenario = scenario$id,
      scenario_class = scenario$class,
      replication = replication,
      seed = seed,
      method = method,
      coordinate = names(geometry$truth),
      truth = as.numeric(geometry$truth),
      estimate = summary$estimate,
      standard_error = summary$standard_error,
      covered = abs(summary$estimate - geometry$truth) <=
        stats::qnorm(0.975) * summary$standard_error,
      calibration_residual = max_calibration,
      respondents = n,
      tasks = tasks,
      assignment = data$assignment,
      covariance = scenario$covariance,
      rho = as.numeric(scenario$rho),
      stringsAsFactors = FALSE
    )
  }))
  list(results = results, selector = do.call(rbind, selector_rows), error = NULL)
}
