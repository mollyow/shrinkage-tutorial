#!/usr/bin/env Rscript

# Fit and evaluate the candidate-recruitment application after the estimand and
# data contracts have been locked by prepare_candidate_data.R.

required_packages <- c("yaml", "readr", "tibble", "dplyr", "tidyr", "pkgload")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace,
                                               quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing_packages)) {
  stop("Install required analysis packages before fitting: ",
       paste(missing_packages, collapse = ", "), call. = FALSE)
}

stopf <- function(fmt, ...) stop(sprintf(fmt, ...), call. = FALSE)
assert_that <- function(ok, fmt, ...) {
  if (!isTRUE(ok)) stopf(fmt, ...)
  invisible(TRUE)
}

script_file <- function() {
  flag <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(flag)) stop("Run this file with Rscript.", call. = FALSE)
  normalizePath(sub("^--file=", "", flag[[1]]), mustWork = TRUE)
}

source(file.path(dirname(script_file()), "contract_helpers.R"), local = TRUE)

parse_args <- function(args) {
  out <- list(prepared = NULL, output_dir = NULL, seed = NULL,
              api_check_only = FALSE)
  i <- 1L
  while (i <= length(args)) {
    arg <- args[[i]]
    if (grepl("^--prepared=", arg)) {
      out$prepared <- sub("^--prepared=", "", arg)
    } else if (identical(arg, "--prepared")) {
      i <- i + 1L
      if (i > length(args)) stop("--prepared requires a value.", call. = FALSE)
      out$prepared <- args[[i]]
    } else if (grepl("^--output-dir=", arg)) {
      out$output_dir <- sub("^--output-dir=", "", arg)
    } else if (identical(arg, "--output-dir")) {
      i <- i + 1L
      if (i > length(args)) stop("--output-dir requires a value.", call. = FALSE)
      out$output_dir <- args[[i]]
    } else if (grepl("^--seed=", arg)) {
      out$seed <- as.integer(sub("^--seed=", "", arg))
    } else if (identical(arg, "--seed")) {
      i <- i + 1L
      if (i > length(args)) stop("--seed requires a value.", call. = FALSE)
      out$seed <- as.integer(args[[i]])
    } else if (identical(arg, "--api-check-only")) {
      out$api_check_only <- TRUE
    } else {
      stop("Unknown argument: ", arg, call. = FALSE)
    }
    i <- i + 1L
  }
  if (!is.null(out$seed) && (length(out$seed) != 1L || is.na(out$seed))) {
    stop("--seed must be one integer.", call. = FALSE)
  }
  out
}

load_analysis_api <- function(repo_root) {
  required <- c("factorial_basis", "fit_order_ridge", "estimate_amie", "rank_profiles")
  code_manifest <- candidate_code_manifest(repo_root)
  candidate_assert(requireNamespace("pkgload", quietly = TRUE),
                   "pkgload is required so the application can load the checked local package source.")
  pkgload::load_all(repo_root, quiet = TRUE, helpers = FALSE)
  namespace <- asNamespace(code_manifest$package)
  namespace_path <- normalizePath(getNamespaceInfo(namespace, "path"), mustWork = TRUE)
  candidate_assert(identical(namespace_path, normalizePath(repo_root, mustWork = TRUE)),
                   "pkgload did not bind orderridge to this repository's local source.")
  code_manifest$namespace_path <- namespace_path
  missing <- required[!vapply(required, exists, envir = namespace, mode = "function",
                             inherits = FALSE, FUN.VALUE = logical(1))]
  if (length(missing)) {
    stop(
      "Required API functions are absent from the checked local orderridge source: ",
      paste(missing, collapse = ", "), ".\n",
      "Update the local package source before fitting.",
      call. = FALSE
    )
  }
  # Bind the exact functions loaded from this checkout. This prevents objects in
  # a user's global environment or an older installed build from taking precedence.
  for (fun in required) assign(fun, get(fun, envir = namespace), envir = .GlobalEnv)
  required_formals <- list(
    factorial_basis = c("factors", "target", "target_weights", "max_order"),
    fit_order_ridge = c("x", "y", "basis", "weights", "cluster", "unpenalized",
                        "penalty_structure", "nfolds", "fold_id", "seed", "control"),
    estimate_amie = c("x", "y", "cluster", "basis", "contrasts",
                      "assignment_prob", "weights", "outer_folds", "fit_args"),
    rank_profiles = c("object", "profiles")
  )
  formal_gaps <- lapply(names(required_formals), function(fun) {
    setdiff(required_formals[[fun]], names(formals(get(fun, mode = "function", inherits = TRUE))))
  })
  names(formal_gaps) <- names(required_formals)
  incompatible <- names(Filter(length, formal_gaps))
  if (length(incompatible)) {
    stop("The loaded package does not implement the planned named-argument API for: ",
         paste(incompatible, collapse = ", "), ". Update/reinstall the package before fitting.",
         call. = FALSE)
  }
  invisible(code_manifest)
}

basis_matrix <- function(basis, newdata) {
  if (exists("factorial_design", mode = "function", inherits = TRUE)) {
    return(factorial_design(basis, newdata, include_intercept = FALSE))
  }
  predict(basis, newdata = newdata, include_intercept = FALSE)
}

task_matrix <- function(data) {
  stats::model.matrix(~ factor(task) - 1, data)
}

cluster_fold_id <- function(cluster, nfolds = 5L, seed = 1L) {
  ids <- sort(unique(as.character(cluster)))
  set.seed(as.integer(seed))
  ids <- sample(ids)
  assignment <- rep(seq_len(as.integer(nfolds)), length.out = length(ids))
  assignment[match(as.character(cluster), ids)]
}

fit_wls_baseline <- function(y, weights, task, factorial, order, maximum_order) {
  keep <- order <= maximum_order
  design <- cbind(task, factorial[, keep, drop = FALSE])
  fit <- stats::lm.wfit(design, y, w = weights)
  coefficient <- fit$coefficients
  coefficient[is.na(coefficient)] <- 0
  list(coefficient = coefficient, keep = keep, maximum_order = maximum_order)
}

predict_wls_baseline <- function(object, task, factorial) {
  design <- cbind(task, factorial[, object$keep, drop = FALSE])
  drop(design %*% object$coefficient)
}

weighted_tabulate <- function(index, weights, nbins) {
  out <- numeric(nbins)
  summed <- rowsum(as.numeric(weights), as.integer(index), reorder = FALSE,
                   na.rm = TRUE)
  out[as.integer(rownames(summed))] <- drop(summed)
  out
}

fit_saturated_wls <- function(y, weights, profile_id, task,
                              profile_levels, task_levels) {
  profile_index <- match(as.character(profile_id), as.character(profile_levels))
  task_index <- match(as.character(task), as.character(task_levels))
  if (anyNA(profile_index) || anyNA(task_index)) {
    stop("Saturated WLS received a profile or task outside the locked support.",
         call. = FALSE)
  }
  n_profile <- length(profile_levels)
  n_task <- length(task_levels)
  cell_index <- profile_index + (task_index - 1L) * n_profile
  cell_weight <- matrix(
    weighted_tabulate(cell_index, weights, n_profile * n_task),
    nrow = n_profile, ncol = n_task
  )
  profile_weight <- rowSums(cell_weight)
  task_weight <- colSums(cell_weight)
  profile_outcome <- weighted_tabulate(profile_index, weights * y, n_profile)
  task_outcome <- weighted_tabulate(task_index, weights * y, n_task)
  profile_rows <- tabulate(profile_index, nbins = n_profile)
  estimable <- profile_weight > 0

  # Eliminate the profile indicators from the normal equations, leaving only
  # an 8 x 8 task-position system. The first task effect is the reference; this
  # avoids materializing a 24,000 x 1,448 dummy matrix.
  inverse_profile_weight <- numeric(n_profile)
  inverse_profile_weight[estimable] <- 1 / profile_weight[estimable]
  weighted_profile_task <- sweep(cell_weight, 1L, inverse_profile_weight, `*`)
  schur <- diag(task_weight, nrow = n_task) - crossprod(cell_weight, weighted_profile_task)
  rhs <- task_outcome - drop(crossprod(cell_weight,
                                       profile_outcome * inverse_profile_weight))
  task_effect <- numeric(n_task)
  if (n_task > 1L) {
    task_effect[-1L] <- tryCatch(
      qr.solve(schur[-1L, -1L, drop = FALSE], rhs[-1L], tol = 1e-10),
      error = function(e) stop("Saturated profile-plus-task WLS is not estimable: ",
                               conditionMessage(e), call. = FALSE)
    )
  }
  profile_effect <- rep(NA_real_, n_profile)
  profile_effect[estimable] <-
    (profile_outcome[estimable] -
       drop(cell_weight[estimable, , drop = FALSE] %*% task_effect)) /
    profile_weight[estimable]

  names(profile_effect) <- as.character(profile_levels)
  names(task_effect) <- as.character(task_levels)
  list(
    profile_effect = profile_effect,
    task_effect = task_effect,
    profile_rows = stats::setNames(profile_rows, as.character(profile_levels)),
    profile_weight = stats::setNames(profile_weight, as.character(profile_levels)),
    profile_levels = as.character(profile_levels),
    task_levels = as.character(task_levels)
  )
}

predict_saturated_wls <- function(object, profile_id, task) {
  profile_effect <- unname(object$profile_effect[match(
    as.character(profile_id), names(object$profile_effect)
  )])
  task_effect <- unname(object$task_effect[match(
    as.character(task), names(object$task_effect)
  )])
  profile_effect + task_effect
}

predict_saturated_profiles <- function(object, profile_id, task_weights = NULL) {
  if (is.null(task_weights)) task_weights <- rep(1 / length(object$task_effect),
                                                 length(object$task_effect))
  assert_that(length(task_weights) == length(object$task_effect) &&
                all(is.finite(task_weights)) && abs(sum(task_weights) - 1) < 1e-12,
              "Saturated target task weights must be finite and sum to one.")
  unname(object$profile_effect[match(
    as.character(profile_id), names(object$profile_effect)
  )]) + sum(task_weights * object$task_effect)
}

fit_ridge_set <- function(x, y, basis, weights, cluster, unpenalized,
                          fold_id, nfolds, seed, control) {
  structure_by_model <- c(
    common_ridge = "common",
    unrestricted_order_ridge = "unrestricted",
    monotone_order_ridge = "monotone"
  )
  fits <- lapply(structure_by_model, function(structure) {
    fit_order_ridge(
      x = x,
      y = y,
      basis = basis,
      weights = weights,
      cluster = cluster,
      unpenalized = unpenalized,
      penalty_structure = structure,
      nfolds = nfolds,
      fold_id = fold_id,
      seed = seed,
      control = control
    )
  })
  for (model in names(fits)) {
    fit <- fits[[model]]
    assert_that(identical(as.integer(fit$fold_id), as.integer(fold_id)),
                "%s did not retain the shared respondent-level inner folds.", model)
    scores <- c(fit$tuning$common_cv_error, fit$tuning$evaluations$score)
    minimum_seen <- min(scores[is.finite(scores)])
    tolerance <- 1e-8 * max(1, abs(minimum_seen))
    assert_that(is.finite(fit$tuning$cv_error) &&
                  fit$tuning$cv_error <= minimum_seen + tolerance,
                "%s did not select the minimum-CV solution.", model)
  }
  fits
}

ridge_diagnostics <- function(fits, fit_scope, outer_fold, inner_seed) {
  dplyr::bind_rows(lapply(names(fits), function(model) {
    object <- fits[[model]]
    tibble::tibble(
      fit = fit_scope,
      fold = outer_fold,
      model = model,
      penalty_structure = object$penalty_structure,
      order = seq_along(object$lambda),
      penalty = unname(object$lambda),
      cv_error = object$tuning$cv_error,
      common_cv_error = object$tuning$common_cv_error,
      convergence = object$tuning$convergence,
      selection_rule = "minimum_cv",
      inner_fold_seed = inner_seed
    )
  }))
}

ridge_tuning_evaluations <- function(fits, fit_scope, outer_fold) {
  dplyr::bind_rows(lapply(names(fits), function(model) {
    evaluation <- tibble::as_tibble(fits[[model]]$tuning$evaluations)
    dplyr::mutate(evaluation, fit = fit_scope, fold = outer_fold,
                  model = model, evaluation = dplyr::row_number(), .before = 1)
  }))
}

predict_order_fit <- function(fit, newdata, nuisance) {
  # The first call is the planned public API. Current package versions that
  # require explicit nuisance matrices for numeric-matrix fits are supported by
  # the narrowly scoped fallback.
  tryCatch(
    predict(fit, newdata),
    error = function(e) {
      if (!grepl("new_unpenalized", conditionMessage(e), fixed = TRUE)) stop(e)
      predict(fit, newdata, new_unpenalized = nuisance)
    }
  )
}

weighted_metrics <- function(y, prediction, weight) {
  eligible <- is.finite(y) & is.finite(weight) & weight > 0
  total_n <- sum(eligible)
  total_weight <- sum(weight[eligible])
  ok <- is.finite(y) & is.finite(prediction) & is.finite(weight) & weight > 0
  covered_n <- sum(ok)
  covered_weight <- sum(weight[ok])
  y <- y[ok]
  prediction <- prediction[ok]
  weight <- weight[ok]
  raw_weight_sum <- sum(weight)
  if (!length(y)) {
    return(c(
      rmse = NA_real_, mae = NA_real_, r_squared = NA_real_,
      calibration_intercept = NA_real_, calibration_slope = NA_real_, n = 0,
      n_total = total_n, prediction_coverage = 0,
      weight_sum_before_normalization = 0,
      weight_total = total_weight, prediction_weight_coverage = 0
    ))
  }
  weight <- weight / sum(weight)
  residual <- y - prediction
  center <- sum(weight * y)
  denominator <- sum(weight * (y - center)^2)
  calibration <- if (length(unique(prediction)) > 1L) {
    stats::lm.wfit(cbind(1, prediction), y, w = weight)$coefficients
  } else c(NA_real_, NA_real_)
  c(
    rmse = sqrt(sum(weight * residual^2)),
    mae = sum(weight * abs(residual)),
    r_squared = if (denominator > 0) 1 - sum(weight * residual^2) / denominator else NA_real_,
    calibration_intercept = unname(calibration[[1]]),
    calibration_slope = unname(calibration[[2]]),
    n = length(y),
    n_total = total_n,
    prediction_coverage = covered_n / total_n,
    weight_sum_before_normalization = raw_weight_sum,
    weight_total = total_weight,
    prediction_weight_coverage = covered_weight / total_weight
  )
}

metric_table <- function(predictions, profile_level = FALSE) {
  fold_split <- split(predictions, interaction(predictions$model, predictions$fold,
                                               drop = TRUE, lex.order = TRUE))
  fold_rows <- lapply(fold_split, function(x) {
    metric <- weighted_metrics(x$observed, x$prediction,
                               if (profile_level) rep(1, nrow(x)) else x$weight)
    data.frame(scope = "fold", fold = as.character(x$fold[[1]]),
               model = x$model[[1]], t(metric), row.names = NULL,
               check.names = FALSE)
  })
  pooled_split <- split(predictions, predictions$model)
  pooled_rows <- lapply(pooled_split, function(x) {
    metric <- weighted_metrics(x$observed, x$prediction,
                               if (profile_level) rep(1, nrow(x)) else x$weight)
    data.frame(scope = "pooled", fold = NA_character_, model = x$model[[1]],
               t(metric), row.names = NULL, check.names = FALSE)
  })
  tibble::as_tibble(do.call(rbind, c(fold_rows, pooled_rows)))
}

extract_amie <- function(object, expected_names) {
  table <- if (is.data.frame(object)) object else object$estimates
  if (is.null(table) || !is.data.frame(table)) {
    stop("estimate_amie() did not return a tidy estimates table in `$estimates`.",
         call. = FALSE)
  }
  name_column <- intersect(c("estimand", "contrast", "term", "name"), names(table))[1]
  estimate_column <- intersect(c("estimate", "coefficient", "value"), names(table))[1]
  se_column <- intersect(c("std_error", "std.error", "se"), names(table))[1]
  if (anyNA(c(name_column, estimate_column, se_column))) {
    stop("Could not identify estimand, estimate, and standard-error columns from estimate_amie().",
         call. = FALSE)
  }
  idx <- match(expected_names, as.character(table[[name_column]]))
  if (anyNA(idx)) {
    stop("estimate_amie() omitted locked coordinates: ",
         paste(expected_names[is.na(idx)], collapse = ", "), call. = FALSE)
  }
  covariance <- object$covariance
  if (is.null(covariance)) covariance <- object$vcov
  if (is.null(covariance)) {
    covariance <- tryCatch(stats::vcov(object), error = function(e) NULL)
  }
  if (is.null(covariance)) {
    stop("estimate_amie() did not expose a covariance matrix. Full family tests and ",
         "simultaneous inference require `$covariance`, `$vcov`, or vcov().",
         call. = FALSE)
  }
  covariance <- as.matrix(covariance)
  if (is.null(rownames(covariance))) {
    assert_that(nrow(covariance) == length(expected_names),
                "Unnamed AMIE covariance has the wrong dimension.")
    dimnames(covariance) <- list(expected_names, expected_names)
  }
  covariance <- covariance[expected_names, expected_names, drop = FALSE]
  ordered <- table[idx, , drop = FALSE]
  list(
    table = ordered,
    estimate = stats::setNames(as.numeric(ordered[[estimate_column]]), expected_names),
    std_error = stats::setNames(as.numeric(ordered[[se_column]]), expected_names),
    covariance = covariance
  )
}

max_t_critical <- function(covariance, alpha, draws, seed) {
  covariance <- (covariance + t(covariance)) / 2
  se <- sqrt(pmax(diag(covariance), 0))
  keep <- se > sqrt(.Machine$double.eps)
  if (sum(keep) <= 1L) return(stats::qnorm(1 - alpha / 2))
  correlation <- covariance[keep, keep, drop = FALSE] / outer(se[keep], se[keep])
  correlation <- (correlation + t(correlation)) / 2
  eig <- eigen(correlation, symmetric = TRUE)
  eig$values[eig$values < 0] <- 0
  root <- eig$vectors %*% diag(sqrt(eig$values), nrow = length(eig$values))
  set.seed(as.integer(seed))
  simulation <- root %*% matrix(stats::rnorm(nrow(root) * as.integer(draws)),
                                nrow = nrow(root))
  unname(stats::quantile(apply(abs(simulation), 2L, max), 1 - alpha,
                         names = FALSE, type = 8))
}

wald_test <- function(estimate, covariance) {
  covariance <- (covariance + t(covariance)) / 2
  eig <- eigen(covariance, symmetric = TRUE)
  tolerance <- max(eig$values, 0) * max(length(estimate), 1L) * .Machine$double.eps^0.75
  keep <- eig$values > tolerance
  if (!any(keep)) return(c(statistic = NA_real_, df = 0, p_value = NA_real_))
  projection <- crossprod(eig$vectors[, keep, drop = FALSE], estimate)
  statistic <- sum(projection^2 / eig$values[keep])
  c(statistic = statistic, df = sum(keep),
    p_value = stats::pchisq(statistic, df = sum(keep), lower.tail = FALSE))
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
script <- script_file()
candidate_dir <- dirname(script)
repo_root <- normalizePath(file.path(candidate_dir, "..", ".."), mustWork = TRUE)
lock_path <- file.path(candidate_dir, "estimands.yml")
current_lock_md5 <- unname(tools::md5sum(lock_path))
cfg <- yaml::read_yaml(lock_path)
contract <- validate_candidate_contract(cfg)
application_code_files <- c(
  run_candidate_analysis = script,
  prepare_candidate_data = file.path(candidate_dir, "prepare_candidate_data.R"),
  contract_helpers = file.path(candidate_dir, "contract_helpers.R")
)
assert_that(all(file.exists(application_code_files)),
            "Application source files required for provenance are missing.")
application_code_md5 <- unname(tools::md5sum(application_code_files))
names(application_code_md5) <- names(application_code_files)
if (!is.null(args$seed)) {
  assert_that(identical(as.integer(args$seed), contract$analysis_seed),
              "--seed may not override the locked analysis.seed (%d).",
              contract$analysis_seed)
}
analysis_seed <- contract$analysis_seed

prepared_path <- args$prepared
if (is.null(prepared_path)) prepared_path <- file.path(candidate_dir, "derived", "candidate_prepared.rds")
assert_that(file.exists(prepared_path),
            "Prepared data not found at %s. Run prepare_candidate_data.R first.", prepared_path)
prepared_path <- normalizePath(prepared_path, mustWork = TRUE)
bundle <- readRDS(prepared_path)

assert_that(identical(as.character(bundle$provenance$estimands_md5), current_lock_md5),
            "estimands.yml changed after preparation. Re-run prepare_candidate_data.R before fitting.")
locked_input <- file.path(repo_root, as.character(cfg$data$allowed_input))
assert_that(file.exists(locked_input), "Locked cleaned replication CSV is missing.")
current_input_md5 <- unname(tools::md5sum(locked_input))
assert_that(identical(as.character(bundle$provenance$input_md5), current_input_md5) &&
              identical(current_input_md5, as.character(cfg$data$md5)),
            "The cleaned replication CSV changed after preparation. Refusing to fit.")
assert_that(all(c("data", "target_grid", "contrast_matrix", "contrast_families") %in% names(bundle)),
            "Prepared bundle lacks required analysis components. Re-run preparation.")

dat <- bundle$data
target_grid <- bundle$target_grid
attributes <- as.character(bundle$attributes)
assert_that(nrow(dat) == contract$rows &&
              length(unique(dat$id)) == contract$respondents &&
              nrow(target_grid) == contract$profile_count &&
              !anyNA(dat[c("id", "task", "weight", "interest", attributes)]),
            "Prepared bundle no longer passes the locked row/cluster/profile contract.")
assert_that(length(unique(dat$profile_id)) == contract$profile_count,
            "Prepared data no longer contain every locked target profile.")
assert_that(ncol(bundle$contrast_matrix) == contract$coordinate_dimension,
            "Locked AMIE coordinate matrix has the wrong full-rank dimension.")
assert_respondent_invariant_weights(dat$id, dat$weight,
                                    tolerance = contract$weight_tolerance)
assert_that(identical(attributes, contract$factor_names),
            "Prepared factor order differs from the current lockfile contract.")
assert_that(identical(sort(unique(as.integer(dat$fold))),
                      seq_len(contract$outer_folds)),
            "Prepared outer folds differ from preparation.outer_folds.")

code_manifest <- load_analysis_api(repo_root)
if (isTRUE(args$api_check_only)) {
  message("Prepared-data lock, application contract, survey-weight, local-source, and API checks passed.")
  quit(save = "no", status = 0L)
}

output_dir <- args$output_dir
if (is.null(output_dir)) output_dir <- file.path(candidate_dir, "results")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_dir <- normalizePath(output_dir, mustWork = TRUE)

target_weights <- rep(1 / nrow(target_grid), nrow(target_grid))
b <- factorial_basis(
  dat,
  factors = attributes,
  target = target_grid,
  target_weights = target_weights,
  max_order = contract$max_order
)
z_all <- if (!is.null(b$design)) b$design else basis_matrix(b, dat)
z_target <- if (!is.null(b$target_design)) b$target_design else basis_matrix(b, target_grid)
factorial_order <- as.integer(attr(z_all, "order"))
if (!length(factorial_order) && !is.null(b$order)) factorial_order <- as.integer(b$order)
assert_that(length(factorial_order) == ncol(z_all),
            "The factorial basis did not expose one order label per design column.")

models <- contract$models
tuning_cfg <- cfg$comparison_tuning
inner_nfolds <- contract$inner_folds
tuning_control <- list(
  common_grid_size = as.integer(tuning_cfg$control$common_grid_size),
  maxit = as.integer(tuning_cfg$control$maxit),
  reltol = as.numeric(tuning_cfg$control$reltol),
  log_lambda_lower = as.numeric(tuning_cfg$control$log_lambda_lower),
  log_lambda_upper = as.numeric(tuning_cfg$control$log_lambda_upper)
)
profile_levels <- as.character(target_grid$profile_id)
task_levels <- sort(unique(dat$task))
outer_folds <- sort(unique(dat$fold))
assert_that(length(outer_folds) == contract$outer_folds,
            "Prepared outer fold count differs from the lockfile.")
prediction_parts <- list()
diagnostic_parts <- list()
tuning_evaluation_parts <- list()
inner_fold_parts <- list()
saturated_support_parts <- list()
checkpoint_dir <- file.path(output_dir, contract$checkpoint_subdirectory)
dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)
checkpoint_identity <- list(
  version = 1L,
  estimands_md5 = current_lock_md5,
  input_md5 = current_input_md5,
  package_source_md5 = code_manifest$aggregate_md5,
  application_code_md5 = application_code_md5,
  analysis_seed = analysis_seed,
  models = models
)

for (fold in outer_folds) {
  checkpoint_path <- file.path(checkpoint_dir,
                               sprintf("outer_fold_%02d.rds", as.integer(fold)))
  if (isTRUE(contract$resume_outer_folds) && file.exists(checkpoint_path)) {
    checkpoint <- readRDS(checkpoint_path)
    assert_that(identical(checkpoint$identity, checkpoint_identity) &&
                  identical(as.integer(checkpoint$fold), as.integer(fold)),
                paste0("Outer-fold checkpoint %s is stale or incompatible. ",
                       "Move it aside and rerun rather than mixing analyses."),
                checkpoint_path)
    prediction_parts[[length(prediction_parts) + 1L]] <- checkpoint$predictions
    diagnostic_parts[[length(diagnostic_parts) + 1L]] <- checkpoint$diagnostics
    tuning_evaluation_parts[[length(tuning_evaluation_parts) + 1L]] <-
      checkpoint$tuning_evaluations
    inner_fold_parts[[length(inner_fold_parts) + 1L]] <- checkpoint$inner_folds
    saturated_support_parts[[length(saturated_support_parts) + 1L]] <-
      checkpoint$saturated_support
    message("Restored outer fold ", fold, " from a checksum-matched checkpoint.")
    next
  }
  message("Outer fold ", fold, " of ", length(outer_folds), "...")
  first_prediction_part <- length(prediction_parts) + 1L
  train <- dat$fold != fold
  test <- !train
  train_task <- task_matrix(dat[train, , drop = FALSE])
  test_task <- task_matrix(dat[test, , drop = FALSE])
  task_columns <- union(colnames(train_task), colnames(test_task))
  align_task <- function(x) {
    missing <- setdiff(task_columns, colnames(x))
    if (length(missing)) x <- cbind(x, matrix(0, nrow(x), length(missing),
                                             dimnames = list(NULL, missing)))
    x[, task_columns, drop = FALSE]
  }
  train_task <- align_task(train_task)
  test_task <- align_task(test_task)

  additive <- fit_wls_baseline(dat$interest[train], dat$weight[train], train_task,
                               z_all[train, , drop = FALSE], factorial_order, 1L)
  two_way <- fit_wls_baseline(dat$interest[train], dat$weight[train], train_task,
                              z_all[train, , drop = FALSE], factorial_order, 2L)
  saturated <- fit_saturated_wls(
    dat$interest[train], dat$weight[train], dat$profile_id[train], dat$task[train],
    profile_levels = profile_levels, task_levels = task_levels
  )
  inner_seed <- analysis_seed + 100L * as.integer(fold)
  inner_fold <- cluster_fold_id(dat$id[train], nfolds = inner_nfolds,
                                seed = inner_seed)
  train_unpenalized <- model.matrix(~ factor(task) - 1, dat[train, , drop = FALSE])
  test_unpenalized <- model.matrix(~ factor(task) - 1, dat[test, , drop = FALSE])
  ridge_fits <- fit_ridge_set(
    x = dat[train, , drop = FALSE],
    y = dat$interest[train],
    basis = b,
    weights = dat$weight[train],
    cluster = dat$id[train],
    unpenalized = train_unpenalized,
    fold_id = inner_fold,
    nfolds = inner_nfolds,
    seed = inner_seed,
    control = tuning_control
  )

  fold_predictions <- list(
    additive_wls = predict_wls_baseline(additive, test_task, z_all[test, , drop = FALSE]),
    two_way_wls = predict_wls_baseline(two_way, test_task, z_all[test, , drop = FALSE]),
    saturated_wls = predict_saturated_wls(saturated, dat$profile_id[test], dat$task[test]),
    common_ridge = predict_order_fit(ridge_fits$common_ridge,
                                     dat[test, , drop = FALSE], test_unpenalized),
    unrestricted_order_ridge = predict_order_fit(
      ridge_fits$unrestricted_order_ridge, dat[test, , drop = FALSE], test_unpenalized
    ),
    monotone_order_ridge = predict_order_fit(
      ridge_fits$monotone_order_ridge, dat[test, , drop = FALSE], test_unpenalized
    )
  )
  for (model in models) {
    prediction_parts[[length(prediction_parts) + 1L]] <- tibble::tibble(
      row_id = which(test),
      id = dat$id[test],
      task = dat$task[test],
      fold = as.integer(fold),
      profile_id = dat$profile_id[test],
      weight = dat$weight[test],
      observed = dat$interest[test],
      model = model,
      prediction = as.numeric(fold_predictions[[model]])
    )
  }
  fold_diagnostics <- ridge_diagnostics(
    ridge_fits, "outer_fold", as.integer(fold), inner_seed
  )
  diagnostic_parts[[length(diagnostic_parts) + 1L]] <- fold_diagnostics
  fold_tuning_evaluations <-
    ridge_tuning_evaluations(ridge_fits, "outer_fold", as.integer(fold))
  tuning_evaluation_parts[[length(tuning_evaluation_parts) + 1L]] <-
    fold_tuning_evaluations
  training_ids <- unique(as.character(dat$id[train]))
  fold_inner_folds <- tibble::tibble(
    fit = "outer_fold",
    fold = as.integer(fold),
    id = training_ids,
    inner_fold = inner_fold[match(training_ids, as.character(dat$id[train]))],
    used_by = "common_ridge; unrestricted_order_ridge; monotone_order_ridge"
  )
  inner_fold_parts[[length(inner_fold_parts) + 1L]] <- fold_inner_folds
  fold_saturated_support <- tibble::tibble(
    fit = "outer_fold",
    fold = as.integer(fold),
    profile_id = profile_levels,
    training_rows = unname(saturated$profile_rows),
    training_weight = unname(saturated$profile_weight),
    estimable = unname(saturated$profile_rows) > 0,
    target_prediction = predict_saturated_profiles(saturated, profile_levels),
    prediction_if_absent = "NA"
  )
  saturated_support_parts[[length(saturated_support_parts) + 1L]] <-
    fold_saturated_support
  saveRDS(
    list(
      identity = checkpoint_identity,
      fold = as.integer(fold),
      predictions = dplyr::bind_rows(
        prediction_parts[first_prediction_part:length(prediction_parts)]
      ),
      diagnostics = fold_diagnostics,
      tuning_evaluations = fold_tuning_evaluations,
      inner_folds = fold_inner_folds,
      saturated_support = fold_saturated_support
    ),
    checkpoint_path,
    version = 3
  )
}

heldout_predictions <- dplyr::bind_rows(prediction_parts)
assert_that(nrow(heldout_predictions) == nrow(dat) * length(models),
            "Held-out prediction table is incomplete.")
row_metrics <- metric_table(heldout_predictions, profile_level = FALSE)

profile_fold_predictions <- heldout_predictions |>
  dplyr::group_by(model, fold, profile_id) |>
  dplyr::summarise(
    observed = stats::weighted.mean(observed, weight),
    prediction = stats::weighted.mean(prediction, weight),
    weight = sum(weight),
    rows = dplyr::n(),
    .groups = "drop"
  )
profile_fold_predictions <- complete_profile_predictions(
  profile_fold_predictions,
  models = models,
  folds = seq_len(contract$outer_folds),
  profile_ids = target_grid$profile_id,
  target_weights = target_grid$target_weight
)
profile_pooled_predictions <- heldout_predictions |>
  dplyr::group_by(model, profile_id) |>
  dplyr::summarise(
    observed = stats::weighted.mean(observed, weight),
    prediction = stats::weighted.mean(prediction, weight),
    weight = sum(weight),
    rows = dplyr::n(),
    .groups = "drop"
  ) |>
  dplyr::mutate(fold = "pooled")
profile_pooled_predictions <- complete_profile_predictions(
  profile_pooled_predictions,
  models = models,
  folds = "pooled",
  profile_ids = target_grid$profile_id,
  target_weights = target_grid$target_weight
)
profile_metrics <- tibble::as_tibble(profile_metric_table_complete(
  dplyr::bind_rows(profile_fold_predictions, profile_pooled_predictions)
))

message("Fitting final comparison models...")
all_task <- task_matrix(dat)
additive_final <- fit_wls_baseline(dat$interest, dat$weight, all_task,
                                   z_all, factorial_order, 1L)
two_way_final <- fit_wls_baseline(dat$interest, dat$weight, all_task,
                                  z_all, factorial_order, 2L)
saturated_final <- fit_saturated_wls(
  dat$interest, dat$weight, dat$profile_id, dat$task,
  profile_levels = profile_levels, task_levels = task_levels
)
final_inner_seed <- analysis_seed + 9000L
final_inner_fold <- cluster_fold_id(dat$id, nfolds = inner_nfolds,
                                    seed = final_inner_seed)
final_ridge_fits <- fit_ridge_set(
  x = dat,
  y = dat$interest,
  basis = b,
  weights = dat$weight,
  cluster = dat$id,
  unpenalized = model.matrix(~ factor(task) - 1, dat),
  fold_id = final_inner_fold,
  nfolds = inner_nfolds,
  seed = final_inner_seed,
  control = tuning_control
)

target_task_average <- matrix(
  1 / ncol(all_task), nrow = nrow(target_grid), ncol = ncol(all_task),
  dimnames = list(NULL, colnames(all_task))
)
final_prediction_list <- list(
  additive_wls = predict_wls_baseline(additive_final, target_task_average, z_target),
  two_way_wls = predict_wls_baseline(two_way_final, target_task_average, z_target),
  saturated_wls = predict_saturated_profiles(saturated_final, target_grid$profile_id),
  common_ridge = predict_order_fit(final_ridge_fits$common_ridge,
                                   target_grid, target_task_average),
  unrestricted_order_ridge = predict_order_fit(
    final_ridge_fits$unrestricted_order_ridge, target_grid, target_task_average
  ),
  monotone_order_ridge = predict_order_fit(
    final_ridge_fits$monotone_order_ridge, target_grid, target_task_average
  )
)
final_profile_predictions <- dplyr::bind_rows(lapply(models, function(model) {
  tibble::tibble(profile_id = target_grid$profile_id, model = model,
                 prediction = as.numeric(final_prediction_list[[model]]))
}))

ranking <- tryCatch(
  rank_profiles(final_ridge_fits[[contract$ranking_model]], profiles = target_grid),
  error = function(e) {
    if (!grepl("new_unpenalized", conditionMessage(e), fixed = TRUE)) stop(e)
    rank_profiles(final_ridge_fits[[contract$ranking_model]], profiles = target_grid,
                  new_unpenalized = target_task_average)
  }
)
if (!is.data.frame(ranking)) {
  ranking <- ranking$ranking
}
assert_that(is.data.frame(ranking),
            "rank_profiles() did not return a data frame or `$ranking` data frame.")
ranking <- tibble::as_tibble(ranking)

diagnostic_parts[[length(diagnostic_parts) + 1L]] <- ridge_diagnostics(
  final_ridge_fits, "final", NA_integer_, final_inner_seed
)
tuning_evaluation_parts[[length(tuning_evaluation_parts) + 1L]] <-
  ridge_tuning_evaluations(final_ridge_fits, "final", NA_integer_)
final_ids <- unique(as.character(dat$id))
inner_fold_parts[[length(inner_fold_parts) + 1L]] <- tibble::tibble(
  fit = "final",
  fold = NA_integer_,
  id = final_ids,
  inner_fold = final_inner_fold[match(final_ids, as.character(dat$id))],
  used_by = "common_ridge; unrestricted_order_ridge; monotone_order_ridge"
)
saturated_support_parts[[length(saturated_support_parts) + 1L]] <- tibble::tibble(
  fit = "final",
  fold = NA_integer_,
  profile_id = profile_levels,
  training_rows = unname(saturated_final$profile_rows),
  training_weight = unname(saturated_final$profile_weight),
  estimable = unname(saturated_final$profile_rows) > 0,
  target_prediction = predict_saturated_profiles(saturated_final, profile_levels),
  prediction_if_absent = "NA"
)
fit_diagnostics <- dplyr::bind_rows(diagnostic_parts)
ridge_tuning_history <- dplyr::bind_rows(tuning_evaluation_parts)
ridge_inner_folds <- dplyr::bind_rows(inner_fold_parts)
saturated_training_support <- dplyr::bind_rows(saturated_support_parts)

message("Estimating the four locked AMIE families...")
amie <- estimate_amie(
  x = dat,
  y = dat$interest,
  cluster = dat$id,
  basis = b,
  contrasts = bundle$contrast_matrix,
  assignment_prob = rep(contract$assignment_probability, nrow(target_grid)),
  weights = dat$weight,
  outer_folds = contract$outer_folds,
  fold_id = dat$fold,
  fit_args = list(
    unpenalized = model.matrix(~ factor(task) - 1, dat),
    penalty_structure = "monotone",
    nfolds = inner_nfolds,
    control = tuning_control
  ),
  level = 1 - as.numeric(cfg$inference$alpha),
  # The package call returns pointwise intervals only. All simultaneous
  # intervals below are computed once with the explicitly locked simulation seed.
  simultaneous = FALSE,
  simultaneous_draws = as.integer(cfg$inference$simultaneous_draws),
  seed = contract$inference_crossfit_seed
)

coordinate_names <- colnames(bundle$contrast_matrix)
amie_extracted <- extract_amie(amie, coordinate_names)
coordinate_estimate <- amie_extracted$estimate
coordinate_covariance <- amie_extracted$covariance
alpha <- as.numeric(cfg$inference$alpha)
draws <- as.integer(cfg$inference$simultaneous_draws)
sim_seed <- contract$simultaneous_seed
z_critical <- stats::qnorm(1 - alpha / 2)
global_critical <- max_t_critical(coordinate_covariance, alpha, draws, sim_seed)

coordinate_rows <- list()
cell_rows <- list()
did_rows <- list()
test_rows <- list()
for (i in seq_along(bundle$contrast_families)) {
  family <- bundle$contrast_families[[i]]
  coordinate <- colnames(family$coordinates)
  idx <- match(coordinate, names(coordinate_estimate))
  estimate <- unname(coordinate_estimate[idx])
  covariance <- coordinate_covariance[idx, idx, drop = FALSE]
  se <- sqrt(pmax(diag(covariance), 0))
  critical <- max_t_critical(
    covariance, alpha, draws, sim_seed + contract$family_seed_offset + i
  )
  meta <- family$coordinate_meta
  coordinate_rows[[i]] <- cbind(
    meta,
    estimate = estimate,
    std_error = se,
    conf_low = estimate - z_critical * se,
    conf_high = estimate + z_critical * se,
    simultaneous_conf_low = estimate - critical * se,
    simultaneous_conf_high = estimate + critical * se,
    simultaneous_critical = critical,
    all_coordinates_simultaneous_conf_low = estimate - global_critical * se,
    all_coordinates_simultaneous_conf_high = estimate + global_critical * se,
    all_coordinates_simultaneous_critical = global_critical
  )

  mapping <- family$cell_mapping
  cell_estimate <- drop(crossprod(mapping, estimate))
  cell_covariance <- crossprod(mapping, covariance %*% mapping)
  cell_se <- sqrt(pmax(diag(cell_covariance), 0))
  cell_critical <- max_t_critical(
    cell_covariance, alpha, draws, sim_seed + contract$cell_seed_offset + i
  )
  cell_rows[[i]] <- cbind(
    family$cells,
    estimate = cell_estimate,
    std_error = cell_se,
    conf_low = cell_estimate - z_critical * cell_se,
    conf_high = cell_estimate + z_critical * cell_se,
    simultaneous_conf_low = cell_estimate - cell_critical * cell_se,
    simultaneous_conf_high = cell_estimate + cell_critical * cell_se,
    simultaneous_critical = cell_critical
  )

  if (!is.null(family$classic_did)) {
    did_mapping <- unname(family$classic_did_mapping[coordinate])
    did_estimate <- sum(did_mapping * estimate)
    did_variance <- drop(crossprod(did_mapping, covariance %*% did_mapping))
    did_se <- sqrt(max(did_variance, 0))
    did_rows[[length(did_rows) + 1L]] <- tibble::tibble(
      family = family$id,
      family_label = family$label,
      contrast = family$classic_did_name,
      scale = "mu11 - mu10 - mu01 + mu00, averaged over all other attributes",
      estimate = did_estimate,
      std_error = did_se,
      conf_low = did_estimate - z_critical * did_se,
      conf_high = did_estimate + z_critical * did_se
    )
  }

  test <- wald_test(estimate, covariance)
  test_rows[[i]] <- tibble::tibble(
    family = family$id,
    family_label = family$label,
    coordinates = length(coordinate),
    statistic = unname(test[["statistic"]]),
    df = unname(test[["df"]]),
    p_value = unname(test[["p_value"]]),
    null_hypothesis = "all full-rank interaction coordinates equal zero"
  )
}

amie_coordinates <- dplyr::bind_rows(coordinate_rows)
amie_cells <- dplyr::bind_rows(cell_rows)
amie_did <- dplyr::bind_rows(did_rows)
amie_family_tests <- dplyr::bind_rows(test_rows)

assert_that(!is.null(amie$outer_fold_id) && length(amie$outer_fold_id) == nrow(dat),
            "estimate_amie() did not retain its outer cross-fitting folds.")
assert_that(identical(as.integer(amie$outer_fold_id), as.integer(dat$fold)),
            "estimate_amie() did not use the locked respondent-level outer folds.")
amie_outer_folds <- tibble::tibble(
  id = unique(as.character(dat$id)),
  outer_fold = as.integer(amie$outer_fold_id[
    match(unique(as.character(dat$id)), as.character(dat$id))
  ]),
  source = "locked_candidate_outer_fold"
)
assert_that(all(vapply(split(amie$outer_fold_id, dat$id), function(x) {
  length(unique(x)) == 1L
}, logical(1))), "AMIE outer folds split at least one respondent.")

assert_that(is.list(amie$fits) && length(amie$fits) == contract$outer_folds,
            "estimate_amie() did not retain one nuisance fit per outer fold.")
amie_inner_parts <- list()
amie_diagnostic_parts <- list()
amie_tuning_parts <- list()
for (position in seq_along(outer_folds)) {
  fold <- outer_folds[[position]]
  fit <- amie$fits[[as.character(fold)]]
  assert_that(!is.null(fit) && length(fit$fold_id) == sum(dat$fold != fold),
              "AMIE nuisance fit for outer fold %s lacks its inner fold_id.", fold)
  train <- dat$fold != fold
  train_ids <- unique(as.character(dat$id[train]))
  inner_for_ids <- as.integer(fit$fold_id[
    match(train_ids, as.character(dat$id[train]))
  ])
  assert_that(all(vapply(split(fit$fold_id, as.character(dat$id[train])), function(x) {
    length(unique(x)) == 1L
  }, logical(1))), "AMIE inner folds split a respondent in outer fit %s.", fold)
  amie_inner_parts[[position]] <- tibble::tibble(
    outer_holdout_fold = as.integer(fold),
    id = train_ids,
    inner_fold = inner_for_ids,
    seed = contract$inference_crossfit_seed + position,
    used_by = "AMIE monotone nuisance fit; separate from six-model comparison tuning"
  )
  nuisance_list <- list(monotone_order_ridge = fit)
  amie_diagnostic_parts[[position]] <- ridge_diagnostics(
    nuisance_list, "amie_nuisance", as.integer(fold),
    contract$inference_crossfit_seed + position
  )
  amie_tuning_parts[[position]] <- ridge_tuning_evaluations(
    nuisance_list, "amie_nuisance", as.integer(fold)
  )
}
amie_inner_folds <- dplyr::bind_rows(amie_inner_parts)
amie_fit_diagnostics <- dplyr::bind_rows(amie_diagnostic_parts)
amie_tuning_evaluations <- dplyr::bind_rows(amie_tuning_parts)

cluster_scores <- NULL
if (!is.null(amie$scores)) {
  cluster_scores <- tibble::as_tibble(as.data.frame(amie$scores), rownames = "id") |>
    tidyr::pivot_longer(-id, names_to = "coordinate", values_to = "score")
}

readr::write_csv(heldout_predictions, file.path(output_dir, "heldout_predictions.csv"))
readr::write_csv(row_metrics, file.path(output_dir, "row_metrics.csv"))
readr::write_csv(profile_fold_predictions,
                 file.path(output_dir, "heldout_profile_predictions_by_fold.csv"))
readr::write_csv(profile_pooled_predictions,
                 file.path(output_dir, "heldout_profile_predictions_pooled.csv"))
readr::write_csv(profile_metrics, file.path(output_dir, "profile_metrics.csv"))
readr::write_csv(final_profile_predictions,
                 file.path(output_dir, "final_profile_predictions.csv"))
readr::write_csv(ranking,
                 file.path(output_dir, "monotone_order_ridge_profile_ranking.csv"))
readr::write_csv(fit_diagnostics, file.path(output_dir, "fit_diagnostics.csv"))
readr::write_csv(ridge_tuning_history,
                 file.path(output_dir, "ridge_tuning_evaluations.csv"))
readr::write_csv(ridge_inner_folds,
                 file.path(output_dir, "ridge_inner_folds.csv"))
readr::write_csv(saturated_training_support,
                 file.path(output_dir, "saturated_training_support.csv"))
readr::write_csv(code_manifest$files,
                 file.path(output_dir, "package_source_checksums.csv"))
readr::write_csv(amie_coordinates, file.path(output_dir, "amie_coordinates.csv"))
readr::write_csv(amie_cells, file.path(output_dir, "amie_cells.csv"))
readr::write_csv(amie_did, file.path(output_dir, "amie_classic_did.csv"))
readr::write_csv(amie_family_tests, file.path(output_dir, "amie_family_tests.csv"))
readr::write_csv(amie_outer_folds, file.path(output_dir, "amie_outer_folds.csv"))
readr::write_csv(amie_inner_folds, file.path(output_dir, "amie_inner_folds.csv"))
readr::write_csv(amie_fit_diagnostics,
                 file.path(output_dir, "amie_fit_diagnostics.csv"))
readr::write_csv(amie_tuning_evaluations,
                 file.path(output_dir, "amie_tuning_evaluations.csv"))
if (!is.null(cluster_scores)) {
  readr::write_csv(cluster_scores, file.path(output_dir, "amie_cluster_scores.csv"))
}

compact_basis <- b
compact_basis$design <- NULL
compact_basis$profiles <- NULL
compact_final_ridge_fits <- lapply(final_ridge_fits, function(fit) {
  fit$basis <- compact_basis
  fit
})
compact_amie <- amie
if (!is.null(compact_amie$fits)) {
  compact_amie$fits <- lapply(compact_amie$fits, function(fit) {
    fit$basis <- compact_basis
    fit
  })
}

saveRDS(
  list(
    basis = compact_basis,
    additive_wls = additive_final,
    two_way_wls = two_way_final,
    saturated_wls = saturated_final,
    common_ridge = compact_final_ridge_fits$common_ridge,
    unrestricted_order_ridge = compact_final_ridge_fits$unrestricted_order_ridge,
    monotone_order_ridge = compact_final_ridge_fits$monotone_order_ridge
  ),
  file.path(output_dir, "candidate_final_fits.rds"), version = 3
)
saveRDS(compact_amie, file.path(output_dir, "candidate_amie_fit.rds"), version = 3)

manifest <- list(
  prepared_file = prepared_path,
  estimands_md5 = current_lock_md5,
  input_md5 = current_input_md5,
  analysis_seed = analysis_seed,
  inference_crossfit_seed = contract$inference_crossfit_seed,
  simultaneous_seed = contract$simultaneous_seed,
  package_source = code_manifest$package,
  package_namespace_path = code_manifest$namespace_path,
  package_source_aggregate_md5 = code_manifest$aggregate_md5,
  package_source_files = split(code_manifest$files$md5, code_manifest$files$file),
  application_code_md5 = split(unname(application_code_md5), names(application_code_md5)),
  assignment_mechanism = cfg$assignment$mechanism,
  assignment_verification_status = cfg$assignment$verification_status,
  across_task_assignment_mechanism = cfg$assignment$across_tasks,
  row_marginal_profile_assignment_probability = contract$assignment_probability,
  replication_doi = cfg$assignment$design_source$replication_doi,
  replication_license = cfg$assignment$design_source$license_status,
  completed_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  rows = nrow(dat),
  respondents = length(unique(dat$id)),
  profiles = nrow(target_grid),
  models = models,
  inner_folds = inner_nfolds,
  outer_folds = contract$outer_folds,
  maximum_interaction_order = contract$max_order,
  ranking_model = contract$ranking_model,
  ranking_task_position_target = contract$ranking_task_rule,
  ridge_selection_rule = "minimum_cv",
  ridge_tuning_control = tuning_control,
  locked_amie_families = vapply(bundle$contrast_families, `[[`, character(1), "id")
)
yaml::write_yaml(manifest, file.path(output_dir, "analysis_manifest.yml"))
writeLines(capture.output(sessionInfo()), file.path(output_dir, "session_info.txt"))

message("Candidate analysis complete. Tidy outputs are in: ", output_dir)
