# Analysis-only helpers for Gate E2 carryover sensitivity.  These functions
# assume `calibration_solver.R` and `feasible_estimator.R` have already been
# sourced.  In particular, they reuse `feasible_assert()` and
# `estimate_working_covariances()` without changing the frozen Gate D code.

sensitivity_symmetric <- function(x) (x + t(x)) / 2

sensitivity_pseudoinverse <- function(x, tolerance = 1e-10,
                                      allow_zero = FALSE) {
  x <- sensitivity_symmetric(as.matrix(x))
  feasible_assert(nrow(x) == ncol(x) && all(is.finite(x)),
                  "A finite square matrix is required.")
  eig <- eigen(x, symmetric = TRUE)
  scale <- max(abs(eig$values))
  if (!is.finite(scale) || scale == 0) {
    feasible_assert(allow_zero, "A required matrix is numerically zero.")
    return(list(
      inverse = matrix(0, nrow(x), ncol(x)),
      rank = 0L,
      eigenvalues = eig$values,
      threshold = 0
    ))
  }
  feasible_assert(
    min(eig$values) >= -tolerance * scale,
    sprintf("A required matrix is not positive semidefinite (minimum eigenvalue %.6g).",
            min(eig$values))
  )
  keep <- eig$values > tolerance * scale
  inverse_value <- numeric(length(eig$values))
  inverse_value[keep] <- 1 / eig$values[keep]
  list(
    inverse = sensitivity_symmetric(
      eig$vectors %*% (inverse_value * t(eig$vectors))
    ),
    rank = sum(keep),
    eigenvalues = eig$values,
    threshold = tolerance * scale
  )
}

sensitivity_spd_inverse <- function(x, tolerance = 1e-10) {
  x <- sensitivity_symmetric(as.matrix(x))
  feasible_assert(nrow(x) == ncol(x) && all(is.finite(x)),
                  "A finite square positive-definite matrix is required.")
  eig <- eigen(x, symmetric = TRUE, only.values = TRUE)$values
  scale <- max(abs(eig))
  feasible_assert(is.finite(scale) && scale > 0 &&
                    min(eig) > tolerance * scale,
                  "The residual covariance or task metric is not positive definite.")
  chol2inv(chol(x))
}

sensitivity_assignment_correlation <- function(
    task_count,
    mechanism = c("product", "uniform_without_replacement"),
    support_size = NULL) {
  mechanism <- match.arg(mechanism)
  feasible_assert(length(task_count) == 1L && is.finite(task_count) &&
                    task_count == as.integer(task_count) && task_count >= 1L,
                  "`task_count` must be a positive integer.")
  task_count <- as.integer(task_count)
  out <- diag(task_count)
  if (identical(mechanism, "uniform_without_replacement")) {
    feasible_assert(length(support_size) == 1L && is.finite(support_size) &&
                      support_size == as.integer(support_size),
                    "Without-replacement support size must be an integer.")
    support_size <- as.integer(support_size)
    feasible_assert(support_size >= task_count && support_size >= 2L,
                    "Without-replacement assignment requires support size M >= T.")
    out[] <- -1 / (support_size - 1)
    diag(out) <- 1
  }
  out
}

true_residual_covariance <- function(base_covariance, outcome_noise_scale = 1) {
  base_covariance <- sensitivity_symmetric(as.matrix(base_covariance))
  outcome_noise_scale <- as.numeric(outcome_noise_scale)
  feasible_assert(length(outcome_noise_scale) == 1L &&
                    is.finite(outcome_noise_scale) && outcome_noise_scale > 0,
                  "`outcome_noise_scale` must be finite and positive.")
  out <- outcome_noise_scale^2 * base_covariance
  invisible(sensitivity_spd_inverse(out))
  out
}

full_binary_factorial_scores <- function(profiles) {
  profiles <- as.matrix(profiles)
  feasible_assert(ncol(profiles) == 4L && nrow(profiles) >= 1L &&
                    all(is.finite(profiles)) &&
                    all(profiles %in% c(-1, 1)),
                  "Gate E2 requires a finite four-factor {-1,+1} profile matrix.")
  factor_names <- colnames(profiles)
  if (is.null(factor_names) || any(!nzchar(factor_names))) {
    factor_names <- paste0("x", seq_len(ncol(profiles)))
  }
  columns <- list()
  column_names <- character()
  for (order in seq_len(ncol(profiles))) {
    subsets <- utils::combn(seq_len(ncol(profiles)), order, simplify = FALSE)
    for (subset in subsets) {
      columns[[length(columns) + 1L]] <- apply(
        profiles[, subset, drop = FALSE], 1L, prod
      )
      column_names[[length(column_names) + 1L]] <-
        paste(factor_names[subset], collapse = "_")
    }
  }
  out <- do.call(cbind, columns)
  colnames(out) <- column_names
  rownames(out) <- rownames(profiles)
  feasible_assert(ncol(out) == 15L && all(is.finite(out)),
                  "The complete four-factor score matrix must have 15 columns.")
  out
}

sensitivity_profile_probability <- function(profile_count, probability = NULL) {
  if (is.null(probability)) probability <- rep(1 / profile_count, profile_count)
  probability <- as.numeric(probability)
  feasible_assert(length(probability) == profile_count &&
                    all(is.finite(probability)) && all(probability > 0) &&
                    abs(sum(probability) - 1) <= 1e-10,
                  "Profile probabilities must be positive and sum to one.")
  probability
}

sensitivity_mixing <- function(operator) {
  mixing <- if (is.matrix(operator)) operator else operator$mixing
  mixing <- as.matrix(mixing)
  feasible_assert(nrow(mixing) == ncol(mixing) && all(is.finite(mixing)),
                  "An operator must supply a finite square task-mixing matrix.")
  mixing
}

# E[f' A f] / E(f^2) for a centered profile feature whose across-task
# correlation matrix is `assignment_correlation`.
mixing_quadratic <- function(left, covariance, assignment_correlation,
                             right = left) {
  left <- as.matrix(left)
  right <- as.matrix(right)
  covariance <- as.matrix(covariance)
  assignment_correlation <- as.matrix(assignment_correlation)
  task_count <- nrow(covariance)
  feasible_assert(
    all(dim(covariance) == c(task_count, task_count)) &&
      all(dim(assignment_correlation) == c(task_count, task_count)) &&
      all(dim(left) == c(task_count, task_count)) &&
      all(dim(right) == c(task_count, task_count)) &&
      all(is.finite(covariance)) && all(is.finite(assignment_correlation)) &&
      all(is.finite(left)) && all(is.finite(right)),
    "Mixing quadratic matrices must be finite and have the same task dimension."
  )
  sum((t(left) %*% covariance %*% right) * assignment_correlation)
}

mixing_sensitivity_K <- function(
    operator, ordinary, true_residual_covariance, assignment_correlation,
    riesz_lookup, profile_probability = NULL, tolerance = 1e-10) {
  operator_mixing <- sensitivity_mixing(operator)
  ordinary_mixing <- sensitivity_mixing(ordinary)
  feasible_assert(identical(dim(operator_mixing), dim(ordinary_mixing)),
                  "The adaptive and ordinary operators must have equal dimensions.")
  riesz_lookup <- as.matrix(riesz_lookup)
  probability <- sensitivity_profile_probability(
    nrow(riesz_lookup), profile_probability
  )
  feasible_assert(all(is.finite(riesz_lookup)),
                  "The Riesz lookup must be finite.")
  gram <- crossprod(riesz_lookup * sqrt(probability))
  difference <- operator_mixing - ordinary_mixing
  multiplier <- mixing_quadratic(
    difference, true_residual_covariance, assignment_correlation
  )
  feasible_assert(multiplier >= -tolerance * max(1, abs(multiplier)),
                  "The sensitivity multiplier is unexpectedly negative.")
  multiplier <- max(multiplier, 0)
  task_count <- nrow(difference)
  K <- sensitivity_symmetric(task_count * multiplier * gram)
  list(
    K = K,
    multiplier = multiplier,
    riesz_gram = sensitivity_symmetric(gram),
    mixing_difference = difference,
    task_count = task_count
  )
}

# Project A f(A_1,...,A_T) off all task-specific own-profile versions of the
# same centered feature in the M metric.  The factor `feature_variance` makes
# the returned normalized map have per-task epsilon exactly one.
metric_project_temporal_map <- function(
    temporal_map, assignment_correlation, metric,
    feature_variance = 1, tolerance = 1e-10) {
  temporal_map <- as.matrix(temporal_map)
  assignment_correlation <- sensitivity_symmetric(
    as.matrix(assignment_correlation)
  )
  metric <- sensitivity_symmetric(as.matrix(metric))
  task_count <- nrow(temporal_map)
  feasible_assert(
    task_count >= 1L && ncol(temporal_map) == task_count &&
      all(dim(assignment_correlation) == c(task_count, task_count)) &&
      all(dim(metric) == c(task_count, task_count)) &&
      all(is.finite(temporal_map)) && all(is.finite(assignment_correlation)) &&
      all(is.finite(metric)),
    "The temporal map, assignment correlation, and metric must be finite T by T matrices."
  )
  feature_variance <- as.numeric(feature_variance)
  feasible_assert(length(feature_variance) == 1L &&
                    is.finite(feature_variance) && feature_variance > 0,
                  "The profile feature must have positive finite variance.")
  invisible(sensitivity_spd_inverse(metric, tolerance))

  normal_matrix <- sensitivity_symmetric(metric * assignment_correlation)
  normal_inverse <- sensitivity_pseudoinverse(normal_matrix, tolerance)
  rhs <- diag(metric %*% temporal_map %*% assignment_correlation)
  range_residual <- max(abs(
    rhs - normal_matrix %*% normal_inverse$inverse %*% rhs
  ))
  feasible_assert(range_residual <= 100 * tolerance * max(1, max(abs(rhs))),
                  "The temporal-map projection normal equations are not estimable.")
  projection_coefficient <- drop(normal_inverse$inverse %*% rhs)
  projected <- temporal_map - diag(projection_coefficient, task_count)
  orthogonality <- diag(metric %*% projected %*% assignment_correlation)
  norm_squared <- feature_variance * mixing_quadratic(
    projected, metric, assignment_correlation
  ) / task_count
  feasible_assert(is.finite(norm_squared) && norm_squared > tolerance,
                  "The temporal departure is zero modulo own-task means.")
  normalized <- projected / sqrt(norm_squared)
  normalized_norm_squared <- feature_variance * mixing_quadratic(
    normalized, metric, assignment_correlation
  ) / task_count
  normalized_orthogonality <- diag(
    metric %*% normalized %*% assignment_correlation
  )
  list(
    raw_map = temporal_map,
    projected_map = projected,
    normalized_map = normalized,
    projection_coefficient = projection_coefficient,
    norm_squared_before_normalizing = norm_squared,
    normalized_norm_squared = normalized_norm_squared,
    orthogonality_residual = max(abs(normalized_orthogonality)),
    normal_equation_range_residual = range_residual,
    normal_matrix_rank = normal_inverse$rank,
    feature_variance = feature_variance
  )
}

operator_departure_moment <- function(
    operator, temporal_map, feature_lookup, riesz_lookup,
    assignment_correlation, profile_probability = NULL) {
  mixing <- sensitivity_mixing(operator)
  temporal_map <- as.matrix(temporal_map)
  feature_lookup <- as.numeric(feature_lookup)
  riesz_lookup <- as.matrix(riesz_lookup)
  probability <- sensitivity_profile_probability(
    nrow(riesz_lookup), profile_probability
  )
  feasible_assert(length(feature_lookup) == nrow(riesz_lookup) &&
                    all(is.finite(feature_lookup)) &&
                    all(dim(temporal_map) == dim(mixing)),
                  "The departure feature/map does not align with the operator and Riesz lookup.")
  cross_gram <- drop(crossprod(riesz_lookup * probability, feature_lookup))
  temporal_multiplier <- sum(
    (t(mixing) %*% temporal_map) * assignment_correlation
  )
  out <- cross_gram * temporal_multiplier
  names(out) <- colnames(riesz_lookup)
  out
}

sensitivity_direction <- function(direction, riesz_lookup,
                                  default_coordinate = "x1_x2") {
  riesz_lookup <- as.matrix(riesz_lookup)
  dimension <- ncol(riesz_lookup)
  if (is.null(direction)) direction <- default_coordinate
  if (is.character(direction)) {
    feasible_assert(length(direction) == 1L &&
                      !is.null(colnames(riesz_lookup)) &&
                      direction %in% colnames(riesz_lookup),
                    "The named least-favourable direction is not in the Riesz lookup.")
    out <- numeric(dimension)
    out[match(direction, colnames(riesz_lookup))] <- 1
  } else {
    out <- as.numeric(direction)
    feasible_assert(length(out) == dimension && all(is.finite(out)) &&
                      sum(out^2) > 0,
                    "The least-favourable direction must be a nonzero finite vector.")
  }
  names(out) <- colnames(riesz_lookup)
  out
}

build_sensitivity_departure <- function(
    type = c("none", "least_favorable", "lag", "blind"),
    geometry, task_count, true_residual_covariance,
    ordinary_operator, oracle_operator,
    mechanism = c("product", "uniform_without_replacement"),
    support_size = NULL, direction = NULL,
    primary_feature = "x1_x2", blind_feature = "x3_x4",
    tolerance = 1e-10) {
  type <- match.arg(type)
  mechanism <- match.arg(mechanism)
  task_count <- as.integer(task_count)
  feasible_assert(is.list(geometry) && !is.null(geometry$profiles) &&
                    !is.null(geometry$riesz_lookup),
                  "`geometry` must supply profiles and a Riesz lookup.")
  profiles <- as.matrix(geometry$profiles)
  riesz_lookup <- as.matrix(geometry$riesz_lookup)
  probability <- sensitivity_profile_probability(
    nrow(profiles), geometry$target_probability
  )
  feasible_assert(nrow(riesz_lookup) == nrow(profiles) &&
                    all(is.finite(riesz_lookup)),
                  "The profile table and Riesz lookup do not align.")
  complete_scores <- full_binary_factorial_scores(profiles)
  residual_covariance <- sensitivity_symmetric(
    as.matrix(true_residual_covariance)
  )
  feasible_assert(all(dim(residual_covariance) == c(task_count, task_count)),
                  "The true residual covariance must be T by T.")
  metric <- sensitivity_spd_inverse(residual_covariance, tolerance)
  assignment_correlation <- sensitivity_assignment_correlation(
    task_count, mechanism, support_size
  )
  ordinary_mixing <- sensitivity_mixing(ordinary_operator)
  oracle_mixing <- sensitivity_mixing(oracle_operator)
  feasible_assert(all(dim(ordinary_mixing) == c(task_count, task_count)) &&
                    all(dim(oracle_mixing) == c(task_count, task_count)),
                  "The supplied score operators have the wrong task dimension.")
  oracle_geometry <- mixing_sensitivity_K(
    oracle_operator, ordinary_operator, residual_covariance,
    assignment_correlation, riesz_lookup, probability, tolerance
  )
  K <- oracle_geometry$K
  a <- sensitivity_direction(direction, riesz_lookup, primary_feature)

  if (identical(type, "none")) {
    zero_map <- matrix(0, task_count, task_count)
    zero_feature <- rep(0, nrow(profiles))
    zero <- rep(0, ncol(riesz_lookup))
    names(zero) <- colnames(riesz_lookup)
    return(structure(list(
      type = type,
      feature_name = "none",
      feature_lookup = zero_feature,
      temporal_map = zero_map,
      raw_temporal_map = zero_map,
      unit_epsilon = 0,
      truth_shift_unit = zero,
      oracle_bias_unit = zero,
      oracle_K = K,
      pattern_K = matrix(0, nrow(K), ncol(K),
                         dimnames = dimnames(K)),
      direction = a,
      directional_bias_unit = 0,
      directional_support = sqrt(max(drop(crossprod(a, K %*% a)), 0)),
      assignment_correlation = assignment_correlation,
      true_residual_covariance = residual_covariance,
      metric = metric,
      profile_probability = probability,
      riesz_lookup = riesz_lookup,
      projection = NULL,
      cross_gram = zero,
      diagnostics = list(
        orthogonality_residual = 0,
        normalized_norm_error = 0,
        blind_bias_residual = 0
      )
    ), class = "factorial_sensitivity_departure"))
  }

  lag_map <- matrix(0, task_count, task_count)
  feasible_assert(task_count >= 2L,
                  "Lag and carryover departures require at least two tasks.")
  lag_map[cbind(2:task_count, seq_len(task_count - 1L))] <- 1

  if (identical(type, "least_favorable")) {
    difference <- oracle_mixing - ordinary_mixing
    direction_support_squared <- drop(crossprod(a, K %*% a))
    feasible_assert(direction_support_squared > tolerance,
                    "The requested direction has no oracle sensitivity headroom.")
    feature_lookup <- drop(riesz_lookup %*% a)
    feature_name <- paste0("least_favorable(",
                           paste(names(a)[a != 0], collapse = "+"), ")")
    raw_map <- task_count * residual_covariance %*% difference /
      sqrt(direction_support_squared)
  } else {
    feature_name <- if (identical(type, "lag")) primary_feature else blind_feature
    feasible_assert(feature_name %in% colnames(complete_scores),
                    paste0("Unknown complete-factorial feature: ", feature_name))
    feature_lookup <- complete_scores[, feature_name]
    raw_map <- lag_map
  }

  feature_center <- sum(probability * feature_lookup)
  feature_lookup <- feature_lookup - feature_center
  feature_variance <- sum(probability * feature_lookup^2)
  projection <- metric_project_temporal_map(
    raw_map, assignment_correlation, metric,
    feature_variance, tolerance
  )
  temporal_map <- projection$normalized_map
  cross_gram <- drop(crossprod(riesz_lookup * probability, feature_lookup))
  names(cross_gram) <- colnames(riesz_lookup)
  truth_shift <- operator_departure_moment(
    ordinary_operator, temporal_map, feature_lookup, riesz_lookup,
    assignment_correlation, probability
  )
  oracle_moment <- operator_departure_moment(
    oracle_operator, temporal_map, feature_lookup, riesz_lookup,
    assignment_correlation, probability
  )
  oracle_bias <- oracle_moment - truth_shift
  pattern_K <- tcrossprod(oracle_bias)
  dimnames(pattern_K) <- list(colnames(riesz_lookup), colnames(riesz_lookup))
  directional_bias <- drop(crossprod(a, oracle_bias))
  directional_support <- sqrt(max(drop(crossprod(a, K %*% a)), 0))

  if (identical(type, "least_favorable")) {
    feasible_assert(
      abs(directional_bias - directional_support) <=
        500 * tolerance * max(1, directional_support),
      "The least-favourable departure does not attain the oracle support bound."
    )
  }
  blind_bias_residual <- if (identical(type, "blind")) {
    max(abs(oracle_bias))
  } else {
    NA_real_
  }
  if (identical(type, "blind")) {
    feasible_assert(max(abs(cross_gram)) <=
                      100 * tolerance * max(1, max(abs(riesz_lookup))),
                    "The requested blind feature is not orthogonal to the reported family.")
    feasible_assert(blind_bias_residual <= 500 * tolerance,
                    "The nominally blind departure changes an oracle target score.")
  }

  structure(list(
    type = type,
    feature_name = feature_name,
    feature_lookup = feature_lookup,
    temporal_map = temporal_map,
    raw_temporal_map = raw_map,
    unit_epsilon = sqrt(projection$normalized_norm_squared),
    truth_shift_unit = truth_shift,
    oracle_bias_unit = oracle_bias,
    oracle_K = K,
    pattern_K = pattern_K,
    direction = a,
    directional_bias_unit = directional_bias,
    directional_support = directional_support,
    assignment_correlation = assignment_correlation,
    true_residual_covariance = residual_covariance,
    metric = metric,
    profile_probability = probability,
    riesz_lookup = riesz_lookup,
    projection = projection,
    cross_gram = cross_gram,
    diagnostics = list(
      feature_center_before_centering = feature_center,
      feature_variance = feature_variance,
      orthogonality_residual = projection$orthogonality_residual,
      normalized_norm_error = abs(projection$normalized_norm_squared - 1),
      support_attainment_error = if (identical(type, "least_favorable")) {
        abs(directional_bias - directional_support)
      } else NA_real_,
      blind_bias_residual = blind_bias_residual
    )
  ), class = "factorial_sensitivity_departure")
}

evaluate_sensitivity_departure <- function(
    profile_index, departure, local_radius, respondent_count = nrow(profile_index)) {
  feasible_assert(inherits(departure, "factorial_sensitivity_departure"),
                  "`departure` must come from build_sensitivity_departure().")
  profile_index <- as.matrix(profile_index)
  respondent_count <- as.integer(respondent_count)
  local_radius <- as.numeric(local_radius)
  feasible_assert(nrow(profile_index) == respondent_count &&
                    respondent_count >= 1L &&
                    length(local_radius) == 1L && is.finite(local_radius) &&
                    local_radius >= 0,
                  "A nonnegative local radius and one profile row per respondent are required.")
  feasible_assert(ncol(profile_index) == nrow(departure$temporal_map) &&
                    all(profile_index >= 1L &
                          profile_index <= length(departure$feature_lookup)),
                  "Profile indices do not align with the departure support/tasks.")
  if (identical(departure$type, "none")) {
    feasible_assert(local_radius == 0,
                    "The `none` departure must have local radius zero.")
  }
  unit_feature <- matrix(
    departure$feature_lookup[as.vector(profile_index)],
    nrow(profile_index), ncol(profile_index)
  )
  unit_shift <- unit_feature %*% t(departure$temporal_map)
  scale <- if (identical(departure$type, "none")) 0 else
    local_radius / sqrt(respondent_count)
  shift <- scale * unit_shift
  list(
    shift = shift,
    unit_shift = unit_shift,
    scale = scale,
    epsilon = abs(scale) * departure$unit_epsilon,
    sqrt_G_epsilon = sqrt(respondent_count) * abs(scale) *
      departure$unit_epsilon,
    truth_shift = scale * departure$truth_shift_unit,
    oracle_bias = scale * departure$oracle_bias_unit
  )
}

apply_sensitivity_departure <- function(data, departure, local_radius) {
  feasible_assert(is.list(data) && !is.null(data$profile_index),
                  "The data object must contain `profile_index`.")
  evaluated <- evaluate_sensitivity_departure(
    data$profile_index, departure, local_radius, nrow(data$profile_index)
  )
  if (!is.null(data$mean)) {
    feasible_assert(identical(dim(as.matrix(data$mean)), dim(evaluated$shift)),
                    "The outcome mean and departure shift do not align.")
    data$mean <- as.matrix(data$mean) + evaluated$shift
  }
  if (!is.null(data$outcome)) {
    feasible_assert(identical(dim(as.matrix(data$outcome)), dim(evaluated$shift)),
                    "The outcome and departure shift do not align.")
    data$outcome <- as.matrix(data$outcome) + evaluated$shift
  }
  data$departure <- c(
    list(type = departure$type, local_radius = local_radius), evaluated
  )
  data
}

fit_saturated_task_profiles <- function(
    outcome, profile_index, train_respondent,
    profile_count = max(profile_index), cluster_weight = NULL,
    riesz_lookup = NULL, profile_probability = NULL,
    task_weights = NULL) {
  outcome <- as.matrix(outcome)
  profile_index <- as.matrix(profile_index)
  train_respondent <- as.logical(train_respondent)
  feasible_assert(identical(dim(outcome), dim(profile_index)) &&
                    length(train_respondent) == nrow(outcome) &&
                    sum(train_respondent) >= 2L &&
                    all(is.finite(outcome)) &&
                    all(profile_index >= 1L & profile_index <= profile_count),
                  "Saturated task-profile fitting requires aligned finite data and a valid training set.")
  if (is.null(cluster_weight)) cluster_weight <- rep(1, nrow(outcome))
  cluster_weight <- as.numeric(cluster_weight)
  feasible_assert(length(cluster_weight) == nrow(outcome) &&
                    all(is.finite(cluster_weight)) && all(cluster_weight > 0),
                  "Cluster weights must be finite, positive, and respondent aligned.")
  tasks <- ncol(outcome)
  target_prediction <- matrix(NA_real_, profile_count, tasks)
  cell_weight <- matrix(0, profile_count, tasks)
  cell_count <- matrix(0L, profile_count, tasks)
  for (tt in seq_len(tasks)) {
    index <- profile_index[train_respondent, tt]
    weight <- cluster_weight[train_respondent]
    value <- outcome[train_respondent, tt]
    total_weight <- rowsum(weight, index, reorder = FALSE)
    total_outcome <- rowsum(weight * value, index, reorder = FALSE)
    observed <- as.integer(rownames(total_weight))
    cell_weight[observed, tt] <- drop(total_weight)
    cell_count[, tt] <- tabulate(index, nbins = profile_count)
    target_prediction[observed, tt] <- drop(total_outcome / total_weight)
  }
  missing <- which(!is.finite(target_prediction), arr.ind = TRUE)
  feasible_assert(
    !nrow(missing),
    if (nrow(missing)) sprintf(
      "Saturated task-profile training has %d empty task-profile cells.",
      nrow(missing)
    ) else ""
  )
  prediction <- matrix(NA_real_, nrow(outcome), tasks)
  for (tt in seq_len(tasks)) {
    prediction[, tt] <- target_prediction[profile_index[, tt], tt]
  }

  target_coefficients <- plugin <- NULL
  if (!is.null(riesz_lookup)) {
    riesz_lookup <- as.matrix(riesz_lookup)
    feasible_assert(nrow(riesz_lookup) == profile_count &&
                      all(is.finite(riesz_lookup)),
                    "The Riesz lookup must have one finite row per profile.")
    probability <- sensitivity_profile_probability(
      profile_count, profile_probability
    )
    target_coefficients <- matrix(
      0, tasks, ncol(riesz_lookup),
      dimnames = list(paste0("task_", seq_len(tasks)), colnames(riesz_lookup))
    )
    for (tt in seq_len(tasks)) {
      target_coefficients[tt, ] <- drop(crossprod(
        probability, riesz_lookup * target_prediction[, tt]
      ))
    }
    if (is.null(task_weights)) task_weights <- rep(1 / tasks, tasks)
    task_weights <- as.numeric(task_weights)
    feasible_assert(length(task_weights) == tasks && all(is.finite(task_weights)),
                    "Task weights must align with saturated task coefficients.")
    plugin <- drop(crossprod(task_weights, target_coefficients))
    names(plugin) <- colnames(riesz_lookup)
  }
  list(
    prediction = prediction,
    target_prediction = target_prediction,
    target_coefficients = target_coefficients,
    plugin = plugin,
    cell_weight = cell_weight,
    cell_count = cell_count,
    minimum_cell_count = min(cell_count),
    training_respondents = sum(train_respondent)
  )
}

saturated_task_profile_oof <- function(
    outcome, profile_index, fold,
    profile_count = max(profile_index), cluster_weight = NULL,
    riesz_lookup = NULL, profile_probability = NULL,
    task_weights = NULL, covariance_eigenvalue_relative_floor = 1e-8) {
  outcome <- as.matrix(outcome)
  profile_index <- as.matrix(profile_index)
  fold <- as.character(fold)
  feasible_assert(identical(dim(outcome), dim(profile_index)) &&
                    length(fold) == nrow(outcome) &&
                    !anyNA(fold) && length(unique(fold)) >= 2L,
                  "OOF saturated fitting requires aligned outcomes, profiles, and at least two folds.")
  if (is.null(cluster_weight)) cluster_weight <- rep(1, nrow(outcome))
  prediction <- matrix(NA_real_, nrow(outcome), ncol(outcome))
  fit_by_fold <- list()
  plugin_by_respondent <- if (is.null(riesz_lookup)) NULL else
    matrix(NA_real_, nrow(outcome), ncol(as.matrix(riesz_lookup)),
           dimnames = list(NULL, colnames(as.matrix(riesz_lookup))))
  for (value in sort(unique(fold))) {
    valid <- fold == value
    fit <- fit_saturated_task_profiles(
      outcome, profile_index, !valid, profile_count, cluster_weight,
      riesz_lookup, profile_probability, task_weights
    )
    prediction[valid, ] <- fit$prediction[valid, , drop = FALSE]
    if (!is.null(plugin_by_respondent)) {
      plugin_by_respondent[valid, ] <- matrix(
        fit$plugin, sum(valid), length(fit$plugin), byrow = TRUE
      )
    }
    fit_by_fold[[value]] <- fit
  }
  feasible_assert(all(is.finite(prediction)) &&
                    (is.null(plugin_by_respondent) ||
                       all(is.finite(plugin_by_respondent))),
                  "Saturated OOF predictions or plug-ins are incomplete.")
  residual <- outcome - prediction
  covariance_set <- estimate_working_covariances(
    residual, cluster_weight,
    as.numeric(covariance_eigenvalue_relative_floor), centered = FALSE
  )
  list(
    prediction = prediction,
    residual = residual,
    covariance_set = covariance_set,
    plugin_by_respondent = plugin_by_respondent,
    target_coefficients_by_fold = lapply(
      fit_by_fold, `[[`, "target_coefficients"
    ),
    plugin_by_fold = lapply(fit_by_fold, `[[`, "plugin"),
    fit_by_fold = fit_by_fold,
    fold = fold
  )
}

exact_operator_bias <- function(
    operator, ordinary_operator,
    assignment_correlation_true,
    assignment_correlation_analysis = NULL,
    task_weights = NULL,
    true_task_coefficients = NULL,
    nuisance_task_coefficients = NULL,
    departure = NULL, departure_scale = 1,
    riesz_lookup = NULL, profile_probability = NULL) {
  mixing <- sensitivity_mixing(operator)
  ordinary_mixing <- sensitivity_mixing(ordinary_operator)
  task_count <- nrow(mixing)
  feasible_assert(all(dim(ordinary_mixing) == c(task_count, task_count)),
                  "The ordinary and analysis operators must have equal task dimensions.")
  assignment_correlation_true <- as.matrix(assignment_correlation_true)
  if (is.null(assignment_correlation_analysis)) {
    assignment_correlation_analysis <- if (is.matrix(operator)) {
      assignment_correlation_true
    } else operator$assignment_correlation
  }
  assignment_correlation_analysis <- as.matrix(assignment_correlation_analysis)
  feasible_assert(all(dim(assignment_correlation_true) == c(task_count, task_count)) &&
                    all(dim(assignment_correlation_analysis) == c(task_count, task_count)) &&
                    all(is.finite(assignment_correlation_true)) &&
                    all(is.finite(assignment_correlation_analysis)),
                  "True and analysis assignment laws must have the operator's task dimension.")
  if (is.null(task_weights)) {
    task_weights <- if (!is.matrix(ordinary_operator) &&
                          !is.null(ordinary_operator$task_weights)) {
      ordinary_operator$task_weights
    } else diag(ordinary_mixing)
  }
  task_weights <- as.numeric(task_weights)
  feasible_assert(length(task_weights) == task_count && all(is.finite(task_weights)),
                  "Task weights must align with the operator.")

  calibration_true <- diag(mixing %*% assignment_correlation_true)
  calibration_analysis <- diag(mixing %*% assignment_correlation_analysis)
  nuisance_bias <- NULL
  if (is.null(true_task_coefficients) && is.null(nuisance_task_coefficients)) {
    dimension <- if (!is.null(riesz_lookup)) ncol(as.matrix(riesz_lookup)) else
      if (!is.null(departure)) length(departure$truth_shift_unit) else 1L
    nuisance_bias <- numeric(dimension)
  } else {
    true_task_coefficients <- as.matrix(true_task_coefficients)
    nuisance_task_coefficients <- as.matrix(nuisance_task_coefficients)
    feasible_assert(identical(dim(true_task_coefficients),
                              dim(nuisance_task_coefficients)) &&
                      nrow(true_task_coefficients) == task_count &&
                      all(is.finite(true_task_coefficients)) &&
                      all(is.finite(nuisance_task_coefficients)),
                    "True and nuisance task coefficients must be finite T by d matrices.")
    nuisance_bias <- drop(crossprod(
      true_task_coefficients - nuisance_task_coefficients,
      calibration_true - task_weights
    ))
  }

  departure_bias <- numeric(length(nuisance_bias))
  if (!is.null(departure) && !identical(departure$type, "none")) {
    feasible_assert(inherits(departure, "factorial_sensitivity_departure"),
                    "The departure must come from build_sensitivity_departure().")
    if (is.null(riesz_lookup)) riesz_lookup <- departure$riesz_lookup
    if (is.null(profile_probability)) {
      profile_probability <- departure$profile_probability
    }
    operator_moment <- operator_departure_moment(
      operator, departure$temporal_map, departure$feature_lookup,
      riesz_lookup, assignment_correlation_true, profile_probability
    )
    ordinary_moment <- operator_departure_moment(
      ordinary_operator, departure$temporal_map, departure$feature_lookup,
      riesz_lookup, assignment_correlation_true, profile_probability
    )
    departure_scale <- as.numeric(departure_scale)
    feasible_assert(length(departure_scale) == 1L &&
                      is.finite(departure_scale),
                    "The departure scale must be one finite number.")
    departure_bias <- departure_scale *
      (operator_moment - ordinary_moment)
  }
  feasible_assert(length(departure_bias) == length(nuisance_bias),
                  "Nuisance and departure bias coordinates do not align.")
  total_bias <- nuisance_bias + departure_bias
  names(total_bias) <- if (!is.null(riesz_lookup))
    colnames(as.matrix(riesz_lookup)) else names(departure_bias)
  list(
    total_bias = total_bias,
    nuisance_calibration_bias = nuisance_bias,
    departure_bias = departure_bias,
    calibration_true = calibration_true,
    calibration_analysis = calibration_analysis,
    true_law_calibration_residual = max(abs(calibration_true - task_weights)),
    analysis_law_calibration_residual =
      max(abs(calibration_analysis - task_weights))
  )
}

cluster_scale_cross_covariance <- function(
    left_score, right_score = left_score, cluster_weight = NULL) {
  left_score <- as.matrix(left_score)
  right_score <- as.matrix(right_score)
  feasible_assert(nrow(left_score) == nrow(right_score) &&
                    nrow(left_score) >= 2L &&
                    all(is.finite(left_score)) && all(is.finite(right_score)),
                  "Cluster score matrices must be finite and respondent aligned.")
  if (is.null(cluster_weight)) cluster_weight <- rep(1, nrow(left_score))
  cluster_weight <- as.numeric(cluster_weight)
  feasible_assert(length(cluster_weight) == nrow(left_score) &&
                    all(is.finite(cluster_weight)) && all(cluster_weight > 0),
                  "Cluster weights must be finite, positive, and respondent aligned.")
  normalized <- cluster_weight / sum(cluster_weight)
  left_center <- drop(crossprod(normalized, left_score))
  right_center <- drop(crossprod(normalized, right_score))
  left_influence <- sweep(left_score, 2L, left_center, "-") *
    (cluster_weight / mean(cluster_weight))
  right_influence <- sweep(right_score, 2L, right_center, "-") *
    (cluster_weight / mean(cluster_weight))
  cross <- crossprod(left_influence, right_influence) /
    (nrow(left_score) - 1L)
  list(
    covariance = cross,
    left_influence = left_influence,
    right_influence = right_influence,
    left_center = left_center,
    right_center = right_center
  )
}

coordinatewise_generic_tau <- function(
    ordinary_score, adaptive_score, K, local_radius,
    cluster_weight = NULL, tolerance = 1e-12) {
  ordinary_score <- as.matrix(ordinary_score)
  adaptive_score <- as.matrix(adaptive_score)
  feasible_assert(identical(dim(ordinary_score), dim(adaptive_score)),
                  "Ordinary and adaptive score matrices must align.")
  K <- sensitivity_symmetric(as.matrix(K))
  feasible_assert(all(dim(K) == c(ncol(ordinary_score), ncol(ordinary_score))) &&
                    all(is.finite(K)),
                  "K must align with the score coordinates.")
  K_eigen <- eigen(K, symmetric = TRUE, only.values = TRUE)$values
  feasible_assert(min(K_eigen) >= -tolerance * max(1, max(abs(K_eigen))),
                  "K must be positive semidefinite.")
  local_radius <- as.numeric(local_radius)
  feasible_assert(length(local_radius) == 1L && is.finite(local_radius) &&
                    local_radius >= 0,
                  "The local sensitivity radius must be nonnegative.")
  difference_score <- adaptive_score - ordinary_score
  cross <- cluster_scale_cross_covariance(
    ordinary_score, difference_score, cluster_weight
  )
  difference <- cluster_scale_cross_covariance(
    difference_score, difference_score, cluster_weight
  )
  C_0D <- cross$covariance
  V_D <- sensitivity_symmetric(difference$covariance)
  c_coordinate <- diag(C_0D)
  v_coordinate <- pmax(diag(V_D), 0)
  k_coordinate <- pmax(diag(K), 0)
  denominator <- v_coordinate + local_radius^2 * k_coordinate
  tau <- numeric(length(denominator))
  informative <- denominator > tolerance * pmax(
    1, abs(c_coordinate), v_coordinate, local_radius^2 * k_coordinate
  )
  tau[informative] <- -c_coordinate[informative] / denominator[informative]
  tau <- pmin(1, pmax(0, tau))
  names(tau) <- colnames(ordinary_score)
  list(
    tau = tau,
    C_0D = C_0D,
    V_D = V_D,
    K = K,
    c_coordinate = c_coordinate,
    v_coordinate = v_coordinate,
    k_coordinate = k_coordinate,
    denominator = denominator,
    informative = informative,
    local_radius = local_radius,
    blended_score = ordinary_score +
      sweep(difference_score, 2L, tau, "*")
  )
}

bias_aware_critical_value <- function(
    standard_error, epsilon, sensitivity = NULL,
    K = NULL, direction = NULL, confidence_level = 0.95) {
  standard_error <- as.numeric(standard_error)
  epsilon <- as.numeric(epsilon)
  confidence_level <- as.numeric(confidence_level)
  feasible_assert(all(is.finite(standard_error)) && all(standard_error >= 0) &&
                    length(epsilon) == 1L && is.finite(epsilon) && epsilon >= 0 &&
                    length(confidence_level) == 1L &&
                    confidence_level > 0 && confidence_level < 1,
                  "Standard errors, epsilon, and confidence level are invalid.")
  if (is.null(sensitivity)) {
    feasible_assert(!is.null(K),
                    "Supply either a sensitivity value or K and a direction.")
    K <- sensitivity_symmetric(as.matrix(K))
    if (is.null(direction)) {
      feasible_assert(length(standard_error) == nrow(K),
                      "Coordinatewise standard errors do not align with K.")
      sensitivity <- sqrt(pmax(diag(K), 0))
    } else {
      direction <- as.numeric(direction)
      feasible_assert(length(direction) == nrow(K),
                      "The contrast direction does not align with K.")
      sensitivity <- sqrt(max(drop(crossprod(direction, K %*% direction)), 0))
    }
  }
  sensitivity <- as.numeric(sensitivity)
  feasible_assert(all(is.finite(sensitivity)) && all(sensitivity >= 0) &&
                    length(sensitivity) %in% c(1L, length(standard_error)),
                  "Sensitivity values must be finite, nonnegative, and aligned with standard errors.")
  z <- stats::qnorm(1 - (1 - confidence_level) / 2)
  sampling_half_width <- z * standard_error
  bias_bound <- epsilon * sensitivity
  total_half_width <- sampling_half_width + bias_bound
  effective_z <- ifelse(standard_error > 0,
                        total_half_width / standard_error, Inf)
  list(
    normal_critical_value = z,
    sampling_half_width = sampling_half_width,
    bias_bound = bias_bound,
    total_half_width = total_half_width,
    effective_standard_error_critical_value = effective_z,
    confidence_level = confidence_level
  )
}

normal_two_sided_power <- function(noncentrality, alpha = 0.05) {
  noncentrality <- as.numeric(noncentrality)
  alpha <- as.numeric(alpha)
  feasible_assert(all(is.finite(noncentrality)) && length(alpha) == 1L &&
                    is.finite(alpha) && alpha > 0 && alpha < 1,
                  "Noncentrality must be finite and alpha must lie in (0,1).")
  critical <- stats::qnorm(1 - alpha / 2)
  stats::pnorm(-critical, mean = noncentrality) +
    stats::pnorm(-critical, mean = -noncentrality)
}

departure_diagnostics <- function(departure, local_radius, respondent_count) {
  feasible_assert(inherits(departure, "factorial_sensitivity_departure"),
                  "`departure` must come from build_sensitivity_departure().")
  local_radius <- as.numeric(local_radius)
  respondent_count <- as.integer(respondent_count)
  feasible_assert(length(local_radius) == 1L && is.finite(local_radius) &&
                    local_radius >= 0 && respondent_count >= 1L,
                  "A nonnegative local radius and positive respondent count are required.")
  if (identical(departure$type, "none")) {
    feasible_assert(local_radius == 0,
                    "The `none` departure must have local radius zero.")
  }
  scale <- if (identical(departure$type, "none")) 0 else
    local_radius / sqrt(respondent_count)
  support <- departure$directional_support
  directional_bias <- scale * departure$directional_bias_unit
  list(
    type = departure$type,
    respondent_count = respondent_count,
    requested_local_radius = local_radius,
    unit_epsilon = departure$unit_epsilon,
    epsilon = abs(scale) * departure$unit_epsilon,
    sqrt_G_epsilon = sqrt(respondent_count) * abs(scale) *
      departure$unit_epsilon,
    truth_shift = scale * departure$truth_shift_unit,
    oracle_bias = scale * departure$oracle_bias_unit,
    directional_bias = directional_bias,
    directional_bias_local_scale = sqrt(respondent_count) * directional_bias,
    directional_support = support,
    support_ratio = if (local_radius > 0 && support > 0) {
      abs(directional_bias) /
        (abs(scale) * departure$unit_epsilon * support)
    } else 0,
    orthogonality_residual = departure$diagnostics$orthogonality_residual,
    normalized_norm_error = departure$diagnostics$normalized_norm_error
  )
}
