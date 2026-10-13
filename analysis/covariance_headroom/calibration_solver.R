# Exact calibration solvers for the repeated-factorial EIF feasibility gate.
#
# These functions are deliberately kept in the analysis directory for now.  The
# gate decides whether they merit promotion into the package API.

assert_calibration <- function(ok, message) {
  if (!isTRUE(ok)) stop(message, call. = FALSE)
}

symmetric_part <- function(x) {
  (x + t(x)) / 2
}

symmetric_pseudoinverse <- function(x, tolerance = 1e-10) {
  x <- symmetric_part(as.matrix(x))
  eig <- eigen(x, symmetric = TRUE)
  scale <- max(abs(eig$values))
  assert_calibration(is.finite(scale) && scale > 0,
                     "A required information matrix is numerically zero.")
  assert_calibration(
    min(eig$values) >= -tolerance * scale,
    sprintf("A required information matrix is not positive semidefinite (minimum eigenvalue %.6g).",
            min(eig$values))
  )
  keep <- eig$values > tolerance * scale
  inverse_values <- numeric(length(eig$values))
  inverse_values[keep] <- 1 / eig$values[keep]
  inverse <- eig$vectors %*% (inverse_values * t(eig$vectors))
  list(
    inverse = symmetric_part(inverse),
    rank = sum(keep),
    eigenvalues = eig$values,
    threshold = tolerance * scale
  )
}

solve_spd <- function(x, rhs = NULL, tolerance = 1e-10) {
  x <- symmetric_part(as.matrix(x))
  eig <- eigen(x, symmetric = TRUE, only.values = TRUE)$values
  scale <- max(abs(eig))
  assert_calibration(is.finite(scale) && scale > 0,
                     "A required matrix is numerically zero.")
  assert_calibration(
    min(eig) > tolerance * scale,
    sprintf(
      "A required matrix is not positive definite (minimum eigenvalue %.6g).",
      min(eig)
    )
  )
  if (is.null(rhs)) rhs <- diag(nrow(x))
  out <- backsolve(chol(x), forwardsolve(t(chol(x)), rhs))
  if (is.null(dim(rhs))) drop(out) else out
}

# Full finite-support solution.  State s supplies a T x p mean-design matrix
# B[[s]], a T x T residual covariance covariance[[s]], and probability q[s].
# L is d x p and maps the mean parameter to the target contrasts.
solve_finite_support_calibration <- function(
    B, covariance, probability, L, tolerance = 1e-10) {
  assert_calibration(is.list(B) && length(B) >= 1L,
                     "`B` must be a nonempty list of state-specific matrices.")
  state_count <- length(B)
  probability <- as.numeric(probability)
  assert_calibration(
    length(probability) == state_count && all(is.finite(probability)) &&
      all(probability > 0) && abs(sum(probability) - 1) <= 100 * tolerance,
    "State probabilities must be positive and sum to one."
  )
  if (is.matrix(covariance)) {
    covariance <- rep(list(covariance), state_count)
  }
  assert_calibration(is.list(covariance) && length(covariance) == state_count,
                     "`covariance` must be one matrix or one matrix per state.")

  B <- lapply(B, as.matrix)
  covariance <- lapply(covariance, as.matrix)
  task_count <- nrow(B[[1L]])
  parameter_count <- ncol(B[[1L]])
  assert_calibration(
    all(vapply(B, function(x) {
      nrow(x) == task_count && ncol(x) == parameter_count && all(is.finite(x))
    }, logical(1))),
    "Every mean-design matrix must have the same finite dimensions."
  )
  assert_calibration(
    all(vapply(covariance, function(x) {
      nrow(x) == task_count && ncol(x) == task_count && all(is.finite(x))
    }, logical(1))),
    "Every covariance matrix must be finite and T by T."
  )
  L <- as.matrix(L)
  assert_calibration(ncol(L) == parameter_count && all(is.finite(L)),
                     "`L` must be a finite d by p target matrix.")

  precision <- lapply(covariance, solve_spd, tolerance = tolerance)
  information <- matrix(0, parameter_count, parameter_count)
  for (s in seq_len(state_count)) {
    information <- information + probability[[s]] *
      crossprod(B[[s]], precision[[s]] %*% B[[s]])
  }
  information <- symmetric_part(information)
  information_inverse <- symmetric_pseudoinverse(information, tolerance)
  row_projection <- information_inverse$inverse %*% information
  estimability_residual <- max(abs(L - L %*% row_projection))
  assert_calibration(
    estimability_residual <= 100 * tolerance * max(1, max(abs(L))),
    sprintf(
      paste0("The requested target is not identified on the supplied support ",
             "(estimability residual %.6g)."),
      estimability_residual
    )
  )

  middle <- information_inverse$inverse %*% t(L)
  H <- lapply(seq_len(state_count), function(s) {
    precision[[s]] %*% B[[s]] %*% middle
  })
  calibration <- matrix(0, parameter_count, nrow(L))
  bound <- matrix(0, nrow(L), nrow(L))
  for (s in seq_len(state_count)) {
    calibration <- calibration + probability[[s]] * crossprod(B[[s]], H[[s]])
    bound <- bound + probability[[s]] *
      crossprod(H[[s]], covariance[[s]] %*% H[[s]])
  }
  calibration_residual <- max(abs(calibration - t(L)))
  bound_formula <- L %*% information_inverse$inverse %*% t(L)
  bound_residual <- max(abs(bound - bound_formula))
  assert_calibration(
    calibration_residual <= 250 * tolerance * max(1, max(abs(L))),
    sprintf("Calibration failed numerically (residual %.6g).", calibration_residual)
  )
  assert_calibration(
    bound_residual <= 500 * tolerance * max(1, max(abs(bound_formula))),
    sprintf("The direct and information-form bounds disagree (residual %.6g).",
            bound_residual)
  )

  structure(list(
    weights = H,
    information = information,
    information_inverse = information_inverse$inverse,
    information_rank = information_inverse$rank,
    bound = symmetric_part(bound),
    calibration = calibration,
    calibration_residual = calibration_residual,
    estimability_residual = estimability_residual,
    bound_residual = bound_residual,
    state_probability = probability,
    task_count = task_count,
    parameter_count = parameter_count,
    target_dimension = nrow(L)
  ), class = "finite_factorial_calibration")
}

# For identical centered profile-score functions at each task, their assignment
# covariance is R times a common contrast Gram matrix.  Product assignment has
# R=I.  Ordered uniform sampling without replacement has off-diagonals
# -1/(M-1).  The optimal T x T mixing matrix is obtained without enumerating
# M^T schedules.
solve_task_mixing <- function(
    covariance, task_weights = NULL,
    mechanism = c("product", "uniform_without_replacement"),
    support_size = NULL, tolerance = 1e-10) {
  mechanism <- match.arg(mechanism)
  covariance <- symmetric_part(as.matrix(covariance))
  task_count <- nrow(covariance)
  assert_calibration(
    task_count >= 1L && ncol(covariance) == task_count && all(is.finite(covariance)),
    "`covariance` must be a finite square matrix."
  )
  if (is.null(task_weights)) task_weights <- rep(1 / task_count, task_count)
  task_weights <- as.numeric(task_weights)
  assert_calibration(
    length(task_weights) == task_count && all(is.finite(task_weights)),
    "`task_weights` must contain one finite value per task."
  )

  assignment_correlation <- diag(task_count)
  if (identical(mechanism, "uniform_without_replacement")) {
    assert_calibration(
      length(support_size) == 1L && is.finite(support_size) &&
        support_size == as.integer(support_size) && support_size >= task_count,
      "Uniform sampling without replacement requires integer M >= T."
    )
    assignment_correlation[] <- -1 / (support_size - 1)
    diag(assignment_correlation) <- 1
  }

  precision <- solve_spd(covariance, tolerance = tolerance)
  calibration_system <- precision * assignment_correlation
  coefficient <- solve_spd(calibration_system, task_weights,
                           tolerance = tolerance)
  mixing <- precision %*% diag(coefficient, nrow = task_count)
  calibration <- diag(mixing %*% assignment_correlation)
  calibration_residual <- max(abs(calibration - task_weights))
  optimal_multiplier <- sum(task_weights * coefficient)
  ordinary_multiplier <- drop(crossprod(
    task_weights,
    (covariance * assignment_correlation) %*% task_weights
  ))
  ratio <- optimal_multiplier / ordinary_multiplier

  assert_calibration(
    calibration_residual <= 250 * tolerance * max(1, max(abs(task_weights))),
    sprintf("Task-mixing calibration failed (residual %.6g).", calibration_residual)
  )
  assert_calibration(
    is.finite(ratio) && ratio > 0 && ratio <= 1 + 1000 * tolerance,
    sprintf("The computed optimal/ordinary variance ratio is invalid: %.6g.", ratio)
  )

  structure(list(
    mixing = mixing,
    coefficient = coefficient,
    covariance = covariance,
    precision = precision,
    assignment_correlation = assignment_correlation,
    task_weights = task_weights,
    mechanism = mechanism,
    support_size = support_size,
    optimal_multiplier = optimal_multiplier,
    ordinary_multiplier = ordinary_multiplier,
    variance_ratio = ratio,
    calibration = calibration,
    calibration_residual = calibration_residual
  ), class = "factorial_task_mixing")
}

apply_task_mixing <- function(object, riesz_by_task) {
  assert_calibration(inherits(object, "factorial_task_mixing"),
                     "`object` must come from solve_task_mixing().")
  riesz_by_task <- as.matrix(riesz_by_task)
  assert_calibration(nrow(riesz_by_task) == nrow(object$mixing),
                     "The Riesz-score matrix must have one row per task.")
  assert_calibration(all(is.finite(riesz_by_task)),
                     "The Riesz-score matrix must be finite.")
  object$mixing %*% riesz_by_task
}

validate_centered_riesz_lookup <- function(riesz_lookup, probability = NULL,
                                            tolerance = 1e-10) {
  riesz_lookup <- as.matrix(riesz_lookup)
  assert_calibration(nrow(riesz_lookup) >= 2L && ncol(riesz_lookup) >= 1L &&
                       all(is.finite(riesz_lookup)),
                     "The Riesz lookup must be a finite M by d matrix.")
  if (is.null(probability)) probability <- rep(1 / nrow(riesz_lookup),
                                               nrow(riesz_lookup))
  probability <- as.numeric(probability)
  assert_calibration(length(probability) == nrow(riesz_lookup) &&
                       all(is.finite(probability)) && all(probability > 0) &&
                       abs(sum(probability) - 1) <= 100 * tolerance,
                     "Riesz lookup probabilities must be positive and sum to one.")
  center <- drop(crossprod(probability, riesz_lookup))
  scale <- max(1, max(abs(riesz_lookup)))
  assert_calibration(
    max(abs(center)) <= 100 * tolerance * scale,
    paste0("The scalable task-mixing formula requires target-centered Riesz ",
           "functions; use the finite/pair-transition solver otherwise.")
  )
  gram <- crossprod(riesz_lookup * sqrt(probability))
  list(
    lookup = riesz_lookup,
    probability = probability,
    center = center,
    gram = symmetric_part(gram)
  )
}

compound_symmetry <- function(covariance, tolerance = 1e-10) {
  covariance <- symmetric_part(as.matrix(covariance))
  task_count <- nrow(covariance)
  assert_calibration(task_count >= 1L && ncol(covariance) == task_count,
                     "`covariance` must be square.")
  diagonal <- mean(diag(covariance))
  off_diagonal <- if (task_count == 1L) 0 else
    mean(covariance[row(covariance) != col(covariance)])
  projected <- matrix(off_diagonal, task_count, task_count)
  diag(projected) <- diagonal
  eig <- eigen(projected, symmetric = TRUE, only.values = TRUE)$values
  assert_calibration(
    min(eig) > tolerance * max(1, max(abs(eig))),
    "The compound-symmetry moment projection is not positive definite."
  )
  list(
    covariance = projected,
    diagonal = diagonal,
    off_diagonal = off_diagonal,
    residual_icc = off_diagonal / diagonal,
    eigenvalues = eig
  )
}
