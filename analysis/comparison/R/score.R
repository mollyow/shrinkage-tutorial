# Cross-fitted known-randomization score and inference
# (comparison-v1.yml `cross_fitted_score` and `uncertainty`).
#
# Respondent score for coordinate c:
#   phi_ic = plugin_ic + mean over i's task/side rows of psi_c(cell) r (y - m_hat),
# where plugin_ic integrates i's held-out prediction over the target and
# r = mu / q is the known target-to-assignment ratio of the observed profile.

# Row-level Riesz weights (rows x coordinates).
cmp_riesz_rows <- function(geom, rows) {
  out <- do.call(cbind, lapply(geom$families, function(f) {
    cmp_family_psi_rows(f, rows[[f$a]], rows[[f$b]])
  }))
  colnames(out) <- geom$coordinates$key
  out * rows$target_ratio
}

# Average row contributions within respondent, in the order of `ids`.
cmp_respondent_mean <- function(values, row_ids, ids) {
  f <- factor(row_ids, levels = ids)
  counts <- tabulate(f, nbins = length(ids))
  cmp_assert(all(counts > 0L), "A respondent has no rows.")
  sums <- rowsum(values, f, reorder = TRUE)
  sums[match(ids, rownames(sums)), , drop = FALSE] / counts
}

cmp_aggregate <- function(scores, weights) {
  cmp_assert(all(is.finite(scores)), "Respondent scores must be finite.")
  g <- nrow(scores)
  w <- weights / sum(weights)
  estimate <- as.numeric(crossprod(w, scores))
  centered <- sweep(scores, 2L, estimate, "-")
  covariance <- (g / (g - 1)) * crossprod(centered * w)
  dimnames(covariance) <- list(colnames(scores), colnames(scores))
  list(estimate = stats::setNames(estimate, colnames(scores)), covariance = covariance)
}

cmp_simultaneous_critical <- function(covariance, draws, seed, level = 0.95) {
  sd <- sqrt(pmax(diag(covariance), 0))
  cmp_assert(all(sd > 0), "Zero variance in a simultaneous family.")
  corr <- covariance / tcrossprod(sd)
  e <- eigen(corr, symmetric = TRUE)
  root <- e$vectors %*% diag(sqrt(pmax(e$values, 0)), length(e$values))
  z <- cmp_with_seed(seed, matrix(stats::rnorm(draws * ncol(root)), draws))
  stat <- apply(abs(z %*% t(root)), 1L, max)
  as.numeric(stats::quantile(stat, level, type = 8, names = FALSE))
}

cmp_family_inference <- function(geom, fit, truth, draws, seed) {
  est <- fit$estimate
  se <- sqrt(pmax(diag(fit$covariance), 0))
  coord <- geom$coordinates
  rows <- lapply(geom$families, function(f) {
    idx <- which(coord$family == f$id)
    cov_f <- fit$covariance[idx, idx, drop = FALSE]
    e <- eigen(cov_f, symmetric = TRUE)
    keep <- e$values > max(e$values[[1]], 0) * 1e-10
    rank <- sum(keep)
    inv <- e$vectors[, keep, drop = FALSE] %*% diag(1 / e$values[keep], rank) %*%
      t(e$vectors[, keep, drop = FALSE])
    wald <- as.numeric(crossprod(est[idx], inv %*% est[idx]))
    err <- est[idx] - truth[idx]
    truth_wald <- as.numeric(crossprod(err, inv %*% err))
    crit <- cmp_simultaneous_critical(cov_f, draws, cmp_hashed_seed(seed, "simultaneous", f$id))
    list(
      family = data.frame(
        family = f$id, dimension = f$dimension, covariance_rank = rank,
        squared_error = sum(err^2), normalized_risk = sum(err^2) / f$dimension,
        variance_trace = sum(diag(cov_f)), normalized_variance = sum(diag(cov_f)) / f$dimension,
        wald_statistic = wald, wald_df = rank,
        wald_p_value = stats::pchisq(wald, df = rank, lower.tail = FALSE),
        truth_wald_statistic = truth_wald,
        truth_wald_covered_95 = truth_wald <= stats::qchisq(0.95, df = rank),
        simultaneous_critical = crit,
        simultaneous_covered_95 = all(abs(err) <= crit * se[idx]),
        null_family = sum(truth[idx]^2) < 1e-14,
        stringsAsFactors = FALSE
      ),
      critical = stats::setNames(rep(crit, length(idx)), coord$key[idx])
    )
  })
  family <- do.call(rbind, lapply(rows, `[[`, "family"))
  crit <- unlist(lapply(rows, `[[`, "critical"))[coord$key]
  effects <- data.frame(
    family = coord$family, coordinate = coord$coordinate, key = coord$key,
    truth = as.numeric(truth[coord$key]), estimate = as.numeric(est[coord$key]),
    std_error = as.numeric(se), error = as.numeric(est[coord$key] - truth[coord$key]),
    lower_95 = as.numeric(est - stats::qnorm(0.975) * se),
    upper_95 = as.numeric(est + stats::qnorm(0.975) * se),
    simultaneous_lower_95 = as.numeric(est - crit * se),
    simultaneous_upper_95 = as.numeric(est + crit * se),
    stringsAsFactors = FALSE
  )
  effects$covered_95 <- effects$truth >= effects$lower_95 & effects$truth <= effects$upper_95
  rownames(effects) <- NULL
  list(effects = effects, family = family)
}
