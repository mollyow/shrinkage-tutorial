# Exact algebra checks for docs/factorial-eif-theory-feasibility.md.
# This verifies identities in finite examples; it is not an estimator.

tol <- 1e-11

# 1. One-task vector target: direct covariance gap equals the conditional
#    assignment-variance formula.
px <- c(`0` = 0.4, `1` = 0.6)
q <- rbind(
  `0` = c(0.2, 0.3, 0.5),
  `1` = c(0.5, 0.25, 0.25)
)
mu <- c(0.25, 0.50, 0.25)
phi <- cbind(
  linear = c(-sqrt(2), 0, sqrt(2)),
  curved = c(1, -1, 1)
)
# The second column is centered and unit norm under mu; the first is too.
stopifnot(max(abs(crossprod(mu, phi))) < tol)
stopifnot(max(abs(crossprod(phi, phi * mu) - diag(2))) < tol)

m <- rbind(
  `0` = c(0.1, 0.4, 0.8),
  `1` = c(0.3, 0.2, 0.9)
)
eta <- rbind(
  `0` = c(0.2, 0.5, 0.6),
  `1` = c(0.4, 0.1, 0.7)
)
sigma2 <- rbind(
  `0` = c(0.7, 0.5, 1.0),
  `1` = c(0.6, 0.9, 0.8)
)
ell <- phi * mu

g_m <- m %*% ell
g_eta <- eta %*% ell
theta <- as.vector(crossprod(px, g_m))

omega_eff <- matrix(0, 2, 2)
omega_eta <- matrix(0, 2, 2)
gap_formula <- matrix(0, 2, 2)

for (xx in seq_len(2)) {
  x_weight <- px[xx]
  centered_g <- g_m[xx, ] - theta
  omega_eff <- omega_eff + x_weight * tcrossprod(centered_g)

  d_by_a <- matrix(0, 3, 2)
  for (aa in seq_len(3)) {
    r <- ell[aa, ] / q[xx, aa]
    e <- eta[xx, aa] - m[xx, aa]
    conditional_mean_eta_score <-
      g_eta[xx, ] - theta - r * e

    omega_eff <- omega_eff +
      x_weight * q[xx, aa] * sigma2[xx, aa] * tcrossprod(r)
    omega_eta <- omega_eta + x_weight * q[xx, aa] *
      (tcrossprod(conditional_mean_eta_score) +
         sigma2[xx, aa] * tcrossprod(r))

    d_by_a[aa, ] <- (g_eta[xx, ] - g_m[xx, ]) - r * e
  }
  d_mean <- as.vector(crossprod(q[xx, ], d_by_a))
  stopifnot(max(abs(d_mean)) < tol)
  gap_formula <- gap_formula + x_weight *
    crossprod(d_by_a * sqrt(q[xx, ]))
}

stopifnot(max(abs((omega_eta - omega_eff) - gap_formula)) < tol)
stopifnot(min(eigen(gap_formula, symmetric = TRUE)$values) > -tol)

# 2. Two-task no-carryover model. The covariance-aware weight satisfies the
#    same calibration as ordinary taskwise AIPW and reaches the analytic bound.
assignments <- as.matrix(expand.grid(a1 = c(-1, 1), a2 = c(-1, 1)))
q_joint <- rep(1 / nrow(assignments), nrow(assignments))
rho <- 0.8
sigma_sq <- 1.7
Sigma <- sigma_sq * matrix(c(1, rho, rho, 1), 2, 2)
Sigma_inv <- solve(Sigma)

# Parameters are (alpha1, tau1, alpha2, tau2); target is (tau1 + tau2) / 2.
L <- matrix(c(0, 0.5, 0, 0.5), nrow = 1)
B_list <- lapply(seq_len(nrow(assignments)), function(j) {
  a <- assignments[j, ]
  rbind(c(1, a[1], 0, 0), c(0, 0, 1, a[2]))
})
J <- Reduce(`+`, Map(function(B, prob) {
  prob * crossprod(B, Sigma_inv %*% B)
}, B_list, q_joint))
H_star <- lapply(B_list, function(B) {
  Sigma_inv %*% B %*% solve(J, t(L))
})

calibration_star <- Reduce(`+`, Map(function(B, H, prob) {
  prob * crossprod(B, H)
}, B_list, H_star, q_joint))
stopifnot(max(abs(calibration_star - t(L))) < tol)

H_naive <- lapply(seq_len(nrow(assignments)), function(j) {
  matrix(assignments[j, ] / 2, ncol = 1)
})
calibration_naive <- Reduce(`+`, Map(function(B, H, prob) {
  prob * crossprod(B, H)
}, B_list, H_naive, q_joint))
stopifnot(max(abs(calibration_naive - t(L))) < tol)

variance_of_weights <- function(weights) {
  sum(vapply(seq_along(weights), function(j) {
    q_joint[j] * drop(crossprod(weights[[j]], Sigma %*% weights[[j]]))
  }, numeric(1)))
}

v_star <- variance_of_weights(H_star)
v_naive <- variance_of_weights(H_naive)
v_star_formula <- drop(L %*% solve(J, t(L)))

stopifnot(abs(v_star - v_star_formula) < tol)
stopifnot(abs(v_naive - sigma_sq / 2) < tol)
stopifnot(abs(v_star / v_naive - (1 - rho^2)) < tol)

# 3. Rotation equivariance of a vector bound.
set.seed(20260827)
raw_rotation <- qr.Q(qr(matrix(rnorm(4), 2, 2)))
L_vec <- rbind(L, c(0, 0.5, 0, -0.5))
bound <- L_vec %*% solve(J, t(L_vec))
L_rotated <- t(raw_rotation) %*% L_vec
bound_rotated <- L_rotated %*% solve(J, t(L_rotated))
stopifnot(max(abs(bound_rotated - t(raw_rotation) %*% bound %*% raw_rotation)) < tol)

print(c(
  unrestricted_gap_identity_error =
    max(abs((omega_eta - omega_eff) - gap_formula)),
  no_carryover_calibration_error =
    max(abs(calibration_star - t(L))),
  two_task_variance_ratio = v_star / v_naive,
  expected_two_task_variance_ratio = 1 - rho^2,
  rotation_equivariance_error =
    max(abs(bound_rotated - t(raw_rotation) %*% bound %*% raw_rotation))
))
