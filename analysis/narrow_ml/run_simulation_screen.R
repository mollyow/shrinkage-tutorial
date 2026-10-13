#!/usr/bin/env Rscript

# Small, deliberately non-production screen of precision and coverage for
# learner-adjusted factorial coefficient scores. The goal is to decide whether
# a full benchmark is warranted, not to supply paper evidence.

if (!requireNamespace("ranger", quietly = TRUE) ||
    !requireNamespace("readr", quietly = TRUE)) {
  stop("Packages `ranger` and `readr` are required.", call. = FALSE)
}

args <- commandArgs(trailingOnly = TRUE)
arg_integer <- function(prefix, default) {
  hit <- grep(paste0("^", prefix, "="), args, value = TRUE)
  if (!length(hit)) return(as.integer(default))
  as.integer(sub(paste0("^", prefix, "="), "", hit[[1L]]))
}
reps <- arg_integer("--reps", 30L)
workers <- arg_integer("--workers", 1L)
num_trees <- arg_integer("--trees", 100L)
if (anyNA(c(reps, workers, num_trees)) || reps < 2L || workers < 1L ||
    num_trees < 50L) {
  stop("Use --reps>=2, --workers>=1, and --trees>=50.", call. = FALSE)
}

g <- 600L
tasks <- 6L
nfolds <- 3L
a_names <- paste0("A", 1:5)
x_names <- paste0("X", 1:4)
pairs <- utils::combn(a_names, 2L, simplify = FALSE)
pair_names <- vapply(pairs, paste, collapse = ":", FUN.VALUE = character(1))
target <- expand.grid(
  stats::setNames(rep(list(c(-1, 1)), length(a_names)), a_names),
  KEEP.OUT.ATTRS = FALSE
)
phi_target <- vapply(pairs, function(pair) {
  target[[pair[[1L]]]] * target[[pair[[2L]]]]
}, numeric(nrow(target)))
colnames(phi_target) <- pair_names
truth <- stats::setNames(rep(0, length(pairs)), pair_names)
truth[["A1:A2"]] <- 0.07
truth[["A3:A4"]] <- 0.04

as_model_data <- function(data) {
  for (nm in a_names) data[[nm]] <- factor(data[[nm]], levels = c(-1, 1))
  data$task <- factor(data$task, levels = seq_len(tasks))
  data
}

pair_terms <- vapply(pairs, paste, collapse = ":", FUN.VALUE = character(1))
a_sum <- paste(a_names, collapse = " + ")
x_sum <- paste(x_names, collapse = " + ")
effect_terms <- paste(c(a_names, pair_terms), collapse = " + ")
rich_formula <- stats::as.formula(paste0(
  "y ~ (", a_sum, ")^2 + task + (", x_sum, ") * (", effect_terms, ")"
))

fit_learner <- function(kind, train, seed) {
  if (identical(kind, "task_lm")) {
    fit <- stats::lm(y ~ task, data = train, model = FALSE, x = FALSE, y = FALSE)
    return(function(newdata) as.numeric(stats::predict(fit, newdata = newdata)))
  }
  if (identical(kind, "rich_lm")) {
    fit <- stats::lm(rich_formula, data = train,
                     model = FALSE, x = FALSE, y = FALSE)
    return(function(newdata) as.numeric(stats::predict(fit, newdata = newdata)))
  }
  if (identical(kind, "ranger")) {
    predictors <- c(a_names, x_names, "task")
    fit <- ranger::ranger(
      dependent.variable.name = "y",
      data = train[c("y", predictors)],
      num.trees = num_trees,
      min.node.size = 15L,
      sample.fraction = 0.8,
      respect.unordered.factors = "partition",
      seed = as.integer(seed),
      num.threads = 1L
    )
    return(function(newdata) {
      as.numeric(predict(fit, data = newdata[predictors],
                         num.threads = 1L)$predictions)
    })
  }
  stop("Unknown learner.", call. = FALSE)
}

make_target_rows <- function(x_context) {
  blocks <- lapply(seq_len(nrow(x_context)), function(i) {
    out <- target[rep(seq_len(nrow(target)), times = tasks), , drop = FALSE]
    out$task <- rep(seq_len(tasks), each = nrow(target))
    for (nm in x_names) out[[nm]] <- x_context[[nm]][i]
    out$.context <- i
    out
  })
  as_model_data(do.call(rbind, blocks))
}

target_plugins <- function(predict_fun, x_context) {
  newdata <- make_target_rows(x_context)
  prediction <- predict_fun(newdata)
  block_size <- nrow(target) * tasks
  out <- matrix(NA_real_, nrow(x_context), ncol(phi_target),
                dimnames = list(NULL, pair_names))
  for (i in seq_len(nrow(x_context))) {
    index <- ((i - 1L) * block_size + 1L):(i * block_size)
    by_task <- matrix(prediction[index], nrow = nrow(target), ncol = tasks)
    profile_prediction <- rowMeans(by_task)
    out[i, ] <- colMeans(phi_target * profile_prediction)
  }
  out
}

aggregate_scores <- function(scores) {
  estimate <- colMeans(scores)
  covariance <- stats::cov(scores) / nrow(scores)
  list(estimate = estimate, std_error = sqrt(pmax(diag(covariance), 0)))
}

simulate_once <- function(scenario, replication) {
  seed <- 202608270L + 10000L * match(scenario, c("simple", "linear", "nonlinear")) +
    replication
  set.seed(seed)
  respondent <- data.frame(id = seq_len(g))
  for (nm in x_names) respondent[[nm]] <- stats::rnorm(g)
  respondent$u <- stats::rnorm(g, sd = 0.25)
  respondent$fold <- sample(rep(seq_len(nfolds), length.out = g))

  dat <- respondent[rep(seq_len(g), each = tasks), , drop = FALSE]
  dat$task <- rep(seq_len(tasks), times = g)
  for (nm in a_names) dat[[nm]] <- sample(c(-1, 1), nrow(dat), replace = TRUE)

  base <- 0.30 + 0.12 * dat$A1 - 0.08 * dat$A2 +
    truth[["A1:A2"]] * dat$A1 * dat$A2 +
    truth[["A3:A4"]] * dat$A3 * dat$A4 +
    0.02 * (dat$task - mean(seq_len(tasks)))
  adjustment <- rep(0, nrow(dat))
  if (identical(scenario, "linear")) {
    adjustment <- 0.35 * dat$X1 + 0.25 * dat$X2 +
      0.20 * dat$A1 * dat$X1 - 0.15 * dat$A2 * dat$X2 +
      0.12 * dat$A1 * dat$A2 * dat$X3
  }
  if (identical(scenario, "nonlinear")) {
    adjustment <- 0.50 * sin(2 * dat$X1) + 0.35 * (dat$X2^2 - 1) +
      0.25 * ((dat$X3 > 0) - 0.5) +
      0.25 * dat$A1 * ((dat$X1 > 0) - 0.5) +
      0.20 * dat$A2 * sin(dat$X2) +
      0.20 * dat$A1 * dat$A2 * (dat$X3^2 - 1)
  }
  dat$y <- base + adjustment + dat$u + stats::rnorm(nrow(dat), sd = 0.50)

  phi_observed <- vapply(pairs, function(pair) {
    dat[[pair[[1L]]]] * dat[[pair[[2L]]]]
  }, numeric(nrow(dat)))
  colnames(phi_observed) <- pair_names
  dat_model <- as_model_data(dat)

  models <- c("unadjusted", "task_lm", "rich_lm", "ranger")
  score_list <- stats::setNames(vector("list", length(models)), models)
  prediction_list <- stats::setNames(vector("list", length(models)), models)

  unadjusted_rows <- phi_observed * dat$y
  score_list$unadjusted <- do.call(rbind, lapply(seq_len(g), function(i) {
    colMeans(unadjusted_rows[dat$id == i, , drop = FALSE])
  }))
  prediction_list$unadjusted <- rep(0, nrow(dat))

  for (kind in setdiff(models, "unadjusted")) {
    scores <- matrix(NA_real_, g, ncol(phi_observed),
                     dimnames = list(NULL, pair_names))
    prediction <- rep(NA_real_, nrow(dat))
    for (fold in seq_len(nfolds)) {
      train <- dat$fold != fold
      valid <- !train
      predict_fun <- fit_learner(
        kind, dat_model[train, , drop = FALSE],
        seed = seed + 100L * fold + match(kind, models)
      )
      prediction[valid] <- predict_fun(dat_model[valid, , drop = FALSE])
      valid_ids <- respondent$id[respondent$fold == fold]
      plugin <- target_plugins(
        predict_fun,
        respondent[match(valid_ids, respondent$id), x_names, drop = FALSE]
      )
      correction <- phi_observed[valid, , drop = FALSE] *
        (dat$y[valid] - prediction[valid])
      valid_id_rows <- dat$id[valid]
      for (i in seq_along(valid_ids)) {
        scores[valid_ids[[i]], ] <- plugin[i, ] +
          colMeans(correction[valid_id_rows == valid_ids[[i]], , drop = FALSE])
      }
    }
    score_list[[kind]] <- scores
    prediction_list[[kind]] <- prediction
  }

  do.call(rbind, lapply(models, function(model) {
    fit <- aggregate_scores(score_list[[model]])
    data.frame(
      scenario = scenario,
      replication = replication,
      model = model,
      estimand = pair_names,
      truth = unname(truth),
      estimate = fit$estimate,
      std_error = fit$std_error,
      covered = abs(fit$estimate - unname(truth)) <= 1.96 * fit$std_error,
      prediction_rmse = sqrt(mean((dat$y - prediction_list[[model]])^2)),
      stringsAsFactors = FALSE
    )
  }))
}

jobs <- expand.grid(
  scenario = c("simple", "linear", "nonlinear"),
  replication = seq_len(reps),
  stringsAsFactors = FALSE
)
run_job <- function(i) {
  message("job ", i, "/", nrow(jobs), ": ", jobs$scenario[[i]],
          " replication ", jobs$replication[[i]])
  simulate_once(jobs$scenario[[i]], jobs$replication[[i]])
}
parts <- if (workers > 1L && .Platform$OS.type != "windows") {
  parallel::mclapply(seq_len(nrow(jobs)), run_job, mc.cores = workers,
                     mc.preschedule = FALSE, mc.set.seed = FALSE)
} else {
  lapply(seq_len(nrow(jobs)), run_job)
}
results <- do.call(rbind, parts)

baseline <- results[results$model == "rich_lm",
                    c("scenario", "replication", "estimand", "std_error")]
names(baseline)[[4L]] <- "rich_lm_std_error"
results <- merge(results, baseline,
                 by = c("scenario", "replication", "estimand"), all.x = TRUE)
results$se_ratio_to_rich_lm <- results$std_error / results$rich_lm_std_error

split_rows <- split(results, interaction(results$scenario, results$model, drop = TRUE))
summary_rows <- do.call(rbind, lapply(split_rows, function(x) {
  data.frame(
    scenario = x$scenario[[1L]],
    model = x$model[[1L]],
    replications = length(unique(x$replication)),
    coefficient_rmse = sqrt(mean((x$estimate - x$truth)^2)),
    mean_reported_se = mean(x$std_error),
    coverage = mean(x$covered),
    prediction_rmse = mean(x$prediction_rmse),
    geometric_mean_se_ratio_to_rich_lm = exp(mean(log(x$se_ratio_to_rich_lm))),
    stringsAsFactors = FALSE
  )
}))
summary_rows <- summary_rows[order(summary_rows$scenario, summary_rows$model), ]

output_dir <- file.path(getwd(), "results", "narrow-ml")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
readr::write_csv(results, file.path(output_dir, "simulation_screen_results.csv"))
readr::write_csv(summary_rows, file.path(output_dir, "simulation_screen_summary.csv"))
print(summary_rows, row.names = FALSE, digits = 4)
