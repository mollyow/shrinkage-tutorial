# Reproduce the two algebra checks in docs/orthogonal-ml-feasibility.md.
# This is a diagnostic, not an estimator implementation.

tol <- 1e-12

# 1. The cross-respondent U-statistic equals the squared sample mean minus
#    the trace of the sample covariance divided by the sample size.
set.seed(20260826)
z <- matrix(rnorm(11L * 7L), nrow = 11L, ncol = 7L)
g <- nrow(z)

u_pairs <- sum(z %*% t(z)) - sum(z * z)
u_pairs <- u_pairs / (g * (g - 1L))

u_diagonal_correction <- sum(colMeans(z)^2) -
  sum(diag(stats::cov(z))) / g

stopifnot(abs(u_pairs - u_diagonal_correction) < tol)

# 2. Cross-fold pairs need not be unbiased when each fold's nuisance learner
#    was trained on observations from the other validation fold.
#
# A_i is uniform on {-1, 1}, Y_i = A_i, and the target coefficient is
# E[A m(A)] = 1. The nuisance learner predicts the mean A in its training
# fold. Folds are {1, 2} and {3, 4}. For this constant learner, the augmented
# score is Z_i = 1 - A_i * mean(A_training).
assignments <- as.matrix(expand.grid(rep(list(c(-1, 1)), 4L)))

statistics <- t(apply(assignments, 1L, function(a) {
  a <- unname(a)
  training_mean_for_fold_1 <- mean(a[3:4])
  training_mean_for_fold_2 <- mean(a[1:2])

  score <- c(
    1 - a[1] * training_mean_for_fold_1,
    1 - a[2] * training_mean_for_fold_1,
    1 - a[3] * training_mean_for_fold_2,
    1 - a[4] * training_mean_for_fold_2
  )

  within_fold <- (score[1] * score[2] + score[3] * score[4]) / 2
  all_pairs <- sum(outer(score, score)[upper.tri(diag(4L))]) / choose(4L, 2L)
  representative_cross_pair <- score[1] * score[3]

  c(
    within_fold = within_fold,
    all_pairs = all_pairs,
    representative_cross_pair = representative_cross_pair
  )
}))

expected <- c(
  within_fold = 1,
  all_pairs = 7 / 6,
  representative_cross_pair = 5 / 4
)
exact_expectations <- stats::setNames(unname(colMeans(statistics)), names(expected))

stopifnot(max(abs(exact_expectations - expected)) < tol)

print(c(
  identity_absolute_difference = abs(u_pairs - u_diagonal_correction),
  exact_expectations
))
