make_profile_grid <- function(number_factors, number_levels) {
  values <- replicate(number_factors, as.character(seq_len(number_levels) - 1L), simplify = FALSE)
  names(values) <- paste0("A", seq_len(number_factors))
  grid <- expand.grid(values, KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  grid[] <- lapply(grid, factor, levels = as.character(seq_len(number_levels) - 1L))
  grid
}

read_simulation_config <- function(path) {
  if (!requireNamespace("yaml", quietly = TRUE)) stop("Install the suggested package `yaml`.")
  cfg <- yaml::read_yaml(path)
  required <- c("seed", "replications", "folds", "designs", "n_over_p", "snr", "order_scales")
  missing <- setdiff(required, names(cfg))
  if (length(missing)) stop("Simulation configuration is missing: ", paste(missing, collapse = ", "))
  cfg
}

expand_simulation_scenarios <- function(config) {
  rows <- list()
  counter <- 0L
  within <- config$within_order %||% "dense"
  for (design in config$designs) {
    for (ratio in unlist(config$n_over_p)) {
      for (snr in unlist(config$snr)) {
        for (scale_name in names(config$order_scales)) {
          for (within_name in unlist(within)) {
            counter <- counter + 1L
            rows[[counter]] <- data.frame(
              scenario_id = counter, design = design$name,
              number_factors = as.integer(design$factors),
              number_levels = as.integer(design$levels),
              max_order = as.integer(design$max_order),
              n_over_p = as.numeric(ratio), snr = as.numeric(snr),
              scale = scale_name, within_order = within_name,
              stringsAsFactors = FALSE
            )
            rows[[counter]]$order_scales <- list(as.numeric(unlist(config$order_scales[[scale_name]])))
          }
        }
      }
    }
  }
  do.call(rbind, rows)
}

scenario_replications <- function(scenario, config) {
  base <- as.integer(config$replications)
  headline_n <- as.integer(config$headline_replications %||% base)
  rule <- config$headline
  if (is.null(rule) || headline_n <= base) return(base)
  matches <- TRUE
  fields <- intersect(names(rule), names(scenario))
  for (field in fields) {
    allowed <- as.character(unlist(rule[[field]], use.names = FALSE))
    matches <- matches && as.character(scenario[[field]][1L]) %in% allowed
  }
  if (matches) headline_n else base
}

draw_surface <- function(basis, order_scales, snr, within_order = "dense") {
  order <- basis$order
  raw <- stats::rnorm(length(order), sd = order_scales[order])
  if (identical(within_order, "heterogeneous")) {
    raw <- raw * exp(stats::rnorm(length(raw), sd = 0.7) - 0.7^2 / 2)
  } else if (!identical(within_order, "dense")) {
    stop("Unknown within-order structure: ", within_order)
  }
  energy <- sum(raw^2)
  if (!is.finite(energy) || energy <= 0) stop("Degenerate simulated surface.")
  raw * sqrt(snr / energy)
}

fit_or_status <- function(...) {
  tryCatch(
    list(status = "ok", fit = fit_order_ridge(...), error = NA_character_),
    error = function(e) list(status = "error", fit = NULL, error = conditionMessage(e))
  )
}

oracle_order_fit <- function(x, y, basis, truth, start_lambda, maxit = 50L) {
  z <- factorial_design(basis, x, include_intercept = FALSE)
  z_target <- factorial_design(basis, basis$target, include_intercept = FALSE)
  w <- rep(1 / nrow(x), nrow(x))
  u <- matrix(1, nrow(x), 1L)
  system <- orderridge:::.prepare_ridge_system(z, y, u, w)
  order <- basis$order
  k <- max(order)
  objective <- function(log_lambda) {
    lambda <- exp(pmin(12, pmax(-16, log_lambda)))
    co <- orderridge:::.solve_ridge_system(system, lambda[order])
    prediction <- as.vector(co$alpha[1L] + z_target %*% co$beta)
    sum(basis$target_weights * (prediction - truth)^2)
  }
  start_lambda <- as.matrix(start_lambda)
  if (ncol(start_lambda) != k) start_lambda <- matrix(as.numeric(start_lambda), nrow = 1L)
  start_logs <- log(pmax(start_lambda, exp(-16)))
  start_values <- apply(start_logs, 1L, objective)
  start <- start_logs[which.min(start_values), ]
  opt <- stats::optim(start, objective, method = "Nelder-Mead",
                      control = list(maxit = as.integer(maxit), reltol = 1e-7))
  candidates <- rbind(start_logs, opt$par)
  values <- apply(candidates, 1L, objective)
  chosen <- candidates[which.min(values), ]
  lambda <- exp(pmin(12, pmax(-16, chosen)))
  fit <- fit_order_ridge(x, y, basis, lambda = lambda,
                         penalty_structure = "unrestricted")
  list(fit = fit, oracle_risk = min(values), convergence = opt$convergence)
}

evaluate_simulation_fit <- function(fit_result, method, truth, beta, basis,
                                    scenario, replication, seed, epsilon = 0.1) {
  base <- data.frame(
    scenario_id = scenario$scenario_id, replication = replication, seed = seed,
    design = scenario$design, n_over_p = scenario$n_over_p, snr = scenario$snr,
    scale = scenario$scale, within_order = scenario$within_order,
    method = method, stringsAsFactors = FALSE
  )
  if (is.null(fit_result$fit)) {
    base$status <- fit_result$status
    base$error <- fit_result$error
    base$risk <- base$regret <- base$epsilon_optimal <- NA_real_
    base$lambda_1 <- base$lambda_2 <- base$lambda_3 <- NA_real_
    base$constraint_binding <- base$unrestricted_violation <- NA
    for (k in seq_len(basis$max_order)) {
      base[[paste0("order_", k, "_coefficient_risk")]] <- NA_real_
    }
    return(base)
  }
  fit <- fit_result$fit
  pred <- predict(fit, basis$target)
  risk <- sum(basis$target_weights * (pred - truth)^2)
  chosen <- which.max(pred)
  best <- max(truth)
  regret <- best - truth[chosen]
  lambda <- rep(NA_real_, 3L)
  lambda[seq_along(fit$lambda)] <- fit$lambda
  binding <- if (length(fit$lambda) > 1L) any(abs(diff(log(fit$lambda))) < 0.05) else FALSE
  violation <- if (length(fit$lambda) > 1L) any(diff(fit$lambda) < 0) else FALSE
  base$status <- "ok"
  base$error <- NA_character_
  base$risk <- risk
  base$regret <- regret
  base$epsilon_optimal <- as.numeric(regret <= epsilon * stats::sd(truth))
  base$lambda_1 <- lambda[1L]
  base$lambda_2 <- lambda[2L]
  base$lambda_3 <- lambda[3L]
  base$constraint_binding <- binding
  base$unrestricted_violation <- violation
  for (k in sort(unique(basis$order))) {
    idx <- basis$order == k
    base[[paste0("order_", k, "_coefficient_risk")]] <- mean((fit$coefficients[idx] - beta[idx])^2)
  }
  base
}

run_simulation_replication <- function(scenario, replication, config) {
  seed <- as.integer(config$seed + scenario$scenario_id * 100000L + replication)
  set.seed(seed)
  grid <- make_profile_grid(scenario$number_factors, scenario$number_levels)
  basis <- factorial_basis(grid, factors = names(grid), target = grid,
                           target_weights = rep(1 / nrow(grid), nrow(grid)),
                           max_order = scenario$max_order)
  p <- length(basis$order)
  n <- max(20L, as.integer(round(scenario$n_over_p * p)))
  beta <- draw_surface(basis, unlist(scenario$order_scales[[1L]]),
                       scenario$snr, scenario$within_order)
  z_target <- factorial_design(basis, grid, include_intercept = FALSE)
  truth <- as.vector(z_target %*% beta)
  sampled <- sample.int(nrow(grid), n, replace = TRUE)
  x <- grid[sampled, , drop = FALSE]
  y <- truth[sampled] + stats::rnorm(n)
  folds <- cluster_folds(seq_len(n), nfolds = config$folds, seed = seed + 1L)
  tuning_control <- config$tuning %||% list()
  k <- max(basis$order)
  methods <- list()
  methods$additive_ols <- fit_or_status(
    x, y, basis, lambda = rep(0, k), active_orders = 1L,
    penalty_structure = "unrestricted"
  )
  methods$truncated_ols <- fit_or_status(
    x, y, basis, lambda = rep(0, k), active_orders = seq_len(min(2L, k)),
    penalty_structure = "unrestricted"
  )
  if (n > p + 1L) {
    methods$saturated_ols <- fit_or_status(
      x, y, basis, lambda = rep(0, k), penalty_structure = "unrestricted"
    )
  } else {
    methods$saturated_ols <- list(status = "not_estimable", fit = NULL,
                                  error = "n is not greater than saturated model rank")
  }
  methods$ordinary_ridge <- fit_or_status(
    x, y, basis, penalty_structure = "common", fold_id = folds,
    control = tuning_control
  )
  methods$unrestricted_order_ridge <- fit_or_status(
    x, y, basis, penalty_structure = "unrestricted", fold_id = folds,
    control = tuning_control
  )
  methods$monotone_order_ridge <- fit_or_status(
    x, y, basis, penalty_structure = "monotone", fold_id = folds,
    control = tuning_control
  )
  order_energy <- vapply(seq_len(k), function(j) mean(beta[basis$order == j]^2), numeric(1))
  oracle_start <- matrix(1 / pmax(n * order_energy, exp(-16)), nrow = 1L)
  feasible_starts <- lapply(
    methods[c("ordinary_ridge", "unrestricted_order_ridge", "monotone_order_ridge")],
    function(result) if (!is.null(result$fit)) unname(result$fit$lambda) else NULL
  )
  feasible_starts <- Filter(Negate(is.null), feasible_starts)
  if (length(feasible_starts)) oracle_start <- rbind(oracle_start, do.call(rbind, feasible_starts))
  oracle <- tryCatch(
    oracle_order_fit(x, y, basis, truth, oracle_start,
                     maxit = config$oracle_maxit %||% 50L),
    error = function(e) list(fit = NULL, error = conditionMessage(e))
  )
  methods$oracle_order_ridge <- if (is.null(oracle$fit)) {
    list(status = "error", fit = NULL, error = oracle$error)
  } else list(status = "ok", fit = oracle$fit, error = NA_character_)

  results <- Map(function(obj, nm) {
    evaluate_simulation_fit(obj, nm, truth, beta, basis, scenario,
                            replication, seed)
  }, methods, names(methods))
  out <- do.call(rbind, results)
  out$n <- n
  out$p <- p
  rownames(out) <- NULL
  out
}

`%||%` <- function(x, y) if (is.null(x)) y else x
