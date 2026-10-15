# Simulation metrics across replications (comparison-v1.yml `primary_metrics`,
# comparison-simulations-v1.yml `metrics`).
#
# Family-balanced risk: R = (1 / F) sum_f mean_r ||theta_hat_f - theta_f||^2 / d_f.
# Ratios use additive adjustment as the denominator. Intervals come from a paired
# replication bootstrap on the log ratio.

# Replication x method matrix of family-balanced risk (one row per replication).
cmp_risk_matrix <- function(family_metrics, methods, value = "normalized_risk") {
  fm <- family_metrics[family_metrics$method %in% methods, , drop = FALSE]
  per <- stats::aggregate(fm[[value]], list(replication = fm$replication, method = fm$method),
                          mean)
  reps <- sort(unique(per$replication))
  out <- matrix(NA_real_, length(reps), length(methods), dimnames = list(reps, methods))
  out[cbind(match(per$replication, reps), match(per$method, methods))] <- per$x
  out
}

cmp_bootstrap_log_ratio <- function(risk, numerator, denominator, draws, seed) {
  r <- nrow(risk)
  idx <- cmp_with_seed(seed, matrix(sample.int(r, r * draws, replace = TRUE), r))
  num <- colMeans(matrix(risk[idx, numerator], r))
  den <- colMeans(matrix(risk[idx, denominator], r))
  stats::quantile(log(num / den), c(0.025, 0.975), names = FALSE, type = 8)
}

cmp_classify_ratio <- function(estimate, lower, upper, rules) {
  if (!is.finite(estimate)) return("unresolved")
  if (estimate <= rules$material_benefit$estimate_at_most &&
      upper < rules$material_benefit$upper_95_below) return("material_benefit")
  if (lower > rules$material_harm$lower_95_above &&
      estimate >= rules$material_harm$estimate_at_least) return("material_harm")
  eq <- unlist(rules$practical_equivalence$full_95_interval_inside)
  if (lower >= eq[[1]] && upper <= eq[[2]]) return("practical_equivalence")
  sb <- unlist(rules$small_benefit$estimate_open_interval)
  if (estimate > sb[[1]] && estimate < sb[[2]]) return("small_benefit")
  "unresolved"
}

# Summaries for one cell (one multiplier). `results` is a list of dataset results.
cmp_cell_summary <- function(results, contracts, seed) {
  fm <- do.call(rbind, lapply(results, `[[`, "family_metrics"))
  ee <- do.call(rbind, lapply(results, `[[`, "effect_estimates"))
  jl <- do.call(rbind, lapply(results, `[[`, "job_ledger"))
  methods <- unique(fm$method)
  cmp_assert("additive" %in% methods, "The additive denominator is missing.")
  risk <- cmp_risk_matrix(fm, methods)
  ok <- stats::complete.cases(risk)
  draws <- as.integer(contracts$v1$uncertainty$simulation_comparison$draws)
  rules <- contracts$v1$interpretation
  base <- colMeans(risk[ok, , drop = FALSE])
  ratios <- do.call(rbind, lapply(methods, function(m) {
    est <- base[[m]] / base[["additive"]]
    ci <- if (identical(m, "additive")) c(0, 0) else
      cmp_bootstrap_log_ratio(risk[ok, , drop = FALSE], m, "additive", draws,
                              cmp_hashed_seed(seed, "ratio", m))
    data.frame(method = m, family_balanced_risk = base[[m]],
               additive_risk = base[["additive"]], ratio = est,
               ratio_lower_95 = exp(ci[[1]]), ratio_upper_95 = exp(ci[[2]]),
               classification = if (identical(m, "additive")) "denominator" else
                 cmp_classify_ratio(est, exp(ci[[1]]), exp(ci[[2]]), rules),
               stringsAsFactors = FALSE)
  }))
  headroom <- if ("oracle" %in% methods) base[["oracle"]] / base[["additive"]] else NA_real_
  gain_floor <- contracts$simulation$metrics$compute_gain_recovered_only_if_oracle_gain_at_least
  ratios$gain_recovered <- if (is.finite(headroom) && (1 - headroom) >= gain_floor) {
    (base[["additive"]] - base[ratios$method]) / (base[["additive"]] - base[["oracle"]])
  } else NA_real_

  coverage <- do.call(rbind, lapply(methods, function(m) {
    e <- ee[ee$method == m & is.finite(ee$estimate), , drop = FALSE]
    f <- fm[fm$method == m & is.finite(fm$normalized_risk), , drop = FALSE]
    bias <- stats::aggregate(e$error, list(key = e$key), mean)
    sdv <- stats::aggregate(e$estimate, list(key = e$key), stats::sd)
    std_bias <- bias$x / sdv$x[match(bias$key, sdv$key)]
    data.frame(
      method = m, replications = length(unique(e$replication)),
      pointwise_coverage = mean(e$covered_95),
      simultaneous_coverage = mean(f$simultaneous_covered_95),
      wald_coverage = mean(f$truth_wald_covered_95),
      max_abs_standardized_bias = max(abs(std_bias)),
      mean_interval_width = mean(e$upper_95 - e$lower_95),
      null_family_rejection = if (any(f$null_family)) mean(f$wald_p_value[f$null_family] < 0.05) else NA_real_,
      nonnull_family_rejection = if (any(!f$null_family)) mean(f$wald_p_value[!f$null_family] < 0.05) else NA_real_,
      prediction_mse = mean(f$prediction_mse),
      stringsAsFactors = FALSE)
  }))
  failures <- stats::aggregate(jl$status != "ok", list(method = jl$method), mean)
  names(failures)[2] <- "job_failure_rate"
  runtime <- stats::aggregate(jl$elapsed_seconds, list(method = jl$method), sum)
  names(runtime)[2] <- "total_job_seconds"
  out <- Reduce(function(a, b) merge(a, b, by = "method", all = TRUE),
                list(ratios, coverage, failures, runtime))
  out <- out[match(methods, out$method), ]
  rownames(out) <- NULL
  list(summary = out, headroom = headroom, replications_used = sum(ok),
       replications_total = nrow(risk))
}
