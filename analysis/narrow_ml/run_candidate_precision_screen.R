#!/usr/bin/env Rscript

# Exploratory precision screen for design-aware orthogonal adjustment.
# This does not implement a new estimator claim and does not alter the locked
# candidate-recruitment application.

required <- c("readr", "ranger", "glmnet")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  stop("Install required packages: ", paste(missing, collapse = ", "), call. = FALSE)
}

repo_root <- normalizePath(getwd(), mustWork = TRUE)
bundle_path <- file.path(
  repo_root, "analysis", "candidate", "derived", "candidate_prepared.rds"
)
raw_path <- file.path(
  repo_root, "Candidate recruitment", "replication", "PSRM_replication",
  "data", "CleanedData.csv"
)
if (!file.exists(bundle_path) || !file.exists(raw_path)) {
  stop("The locked prepared bundle and cleaned replication data are required.",
       call. = FALSE)
}

args <- commandArgs(trailingOnly = TRUE)
trees_arg <- grep("^--trees=", args, value = TRUE)
num_trees <- if (length(trees_arg)) {
  as.integer(sub("^--trees=", "", trees_arg[[1L]]))
} else {
  300L
}
if (is.na(num_trees) || num_trees < 50L) {
  stop("`--trees` must be an integer of at least 50.", call. = FALSE)
}

bundle <- readRDS(bundle_path)
dat <- as.data.frame(bundle$data)
target <- as.data.frame(bundle$target_grid)
contrast <- as.matrix(bundle$contrast_matrix)

factor_names <- as.character(bundle$attributes)
demographic_names <- c("woman", "poc", "young", "dem", "rep", "workingclass")
raw_needed <- c("id", "iteration", demographic_names)
raw <- readr::read_csv(
  raw_path,
  col_select = tidyselect::all_of(raw_needed),
  col_types = readr::cols(
    id = readr::col_character(),
    iteration = readr::col_integer(),
    .default = readr::col_double()
  ),
  show_col_types = FALSE,
  progress = FALSE
)

key <- function(id, task) paste(as.character(id), as.integer(task), sep = "\u001f")
raw_key <- key(raw$id, raw$iteration)
dat_key <- key(dat$id, dat$task)
if (anyDuplicated(raw_key) || anyNA(match(dat_key, raw_key))) {
  stop("Could not align the cleaned demographics to the locked prepared rows.",
       call. = FALSE)
}
row_match <- match(dat_key, raw_key)
for (nm in demographic_names) dat[[nm]] <- raw[[nm]][row_match]

if (anyNA(dat[c("interest", "weight", "fold", demographic_names)])) {
  stop("The precision screen requires complete outcomes, weights, folds, and demographics.",
       call. = FALSE)
}

# Treat randomized attributes as nominal in this screen, matching the primary
# factorial analysis rather than the historical ordinal sensitivity coding.
for (nm in factor_names) {
  declared <- levels(bundle$data[[nm]])
  dat[[nm]] <- factor(as.character(dat[[nm]]), levels = declared)
  target[[nm]] <- factor(as.character(target[[nm]]), levels = declared)
}
dat$task <- factor(dat$task, levels = sort(unique(dat$task)))
for (nm in demographic_names) {
  dat[[nm]] <- factor(dat[[nm]], levels = sort(unique(dat[[nm]])))
}

respondent_check <- vapply(demographic_names, function(nm) {
  all(vapply(split(dat[[nm]], dat$id), function(x) length(unique(x)) == 1L,
             logical(1)))
}, logical(1))
if (!all(respondent_check)) {
  stop("A proposed demographic adjustment variable varies within respondent.",
       call. = FALSE)
}

profile_index <- match(as.character(dat$profile_id), as.character(target$profile_id))
if (anyNA(profile_index) || nrow(contrast) != nrow(target)) {
  stop("Observed profiles or contrast rows do not align with the locked target.",
       call. = FALSE)
}
q <- as.numeric(bundle$contract$assignment_probability)
if (!is.finite(q) || q <= 0 || abs(q * nrow(target) - 1) > 1e-10) {
  stop("This screen expects the locked uniform profile assignment.", call. = FALSE)
}
riesz_observed <- contrast[profile_index, , drop = FALSE] / q
contrast_names <- colnames(contrast)

ids <- unique(as.character(dat$id))
id_index <- match(as.character(dat$id), ids)
g <- length(ids)
fold_by_id <- vapply(split(as.integer(dat$fold), dat$id), function(x) unique(x)[1L],
                     integer(1))
fold_by_id <- fold_by_id[match(ids, names(fold_by_id))]
if (anyNA(fold_by_id) || any(vapply(split(dat$fold, dat$id), function(x) {
  length(unique(x)) != 1L
}, logical(1)))) {
  stop("Locked folds do not isolate complete respondents.", call. = FALSE)
}

cluster_weight <- vapply(seq_along(ids), function(i) {
  mean(dat$weight[id_index == i])
}, numeric(1))

context_key <- function(x) {
  do.call(paste, c(lapply(x[demographic_names], as.character), sep = "\u001e"))
}
row_context <- context_key(dat)
context_by_id <- vapply(seq_along(ids), function(i) {
  unique(row_context[id_index == i])[[1L]]
}, character(1))

profile_terms <- paste(factor_names, collapse = " + ")
demographic_terms <- paste(demographic_names, collapse = " + ")
formulas <- list(
  task_lm = interest ~ task,
  additive_lm = stats::as.formula(paste(
    "interest ~", profile_terms, "+ task +", demographic_terms
  )),
  interacted_lm = stats::as.formula(paste0(
    "interest ~ (", profile_terms, ")^2 + task + ", demographic_terms,
    " + (", profile_terms, "):((", demographic_terms, "))"
  ))
)
elastic_formula <- stats::as.formula(paste0(
  "~ 0 + task + (", profile_terms, ")^2 + ", demographic_terms,
  " + (", profile_terms, "):(", demographic_terms, ")"
))

predictor_names <- c(factor_names, "task", demographic_names)

fit_learner <- function(kind, train, seed) {
  if (kind %in% names(formulas)) {
    fit <- stats::lm(
      formulas[[kind]], data = train, weights = weight,
      model = FALSE, x = FALSE, y = FALSE, qr = TRUE
    )
    return(list(
      predict = function(newdata) as.numeric(stats::predict(fit, newdata = newdata)),
      detail = sprintf("weighted least squares; rank=%d", fit$rank)
    ))
  }
  if (identical(kind, "elastic_net")) {
    design <- stats::model.matrix(elastic_formula, data = train)
    design_columns <- colnames(design)
    inner_fold <- match(train$fold, sort(unique(train$fold)))
    fit <- glmnet::cv.glmnet(
      x = design,
      y = train$interest,
      weights = train$weight,
      foldid = inner_fold,
      family = "gaussian",
      alpha = 0.5,
      nlambda = 50L,
      standardize = TRUE,
      type.measure = "mse"
    )
    return(list(
      predict = function(newdata) {
        new_design <- stats::model.matrix(elastic_formula, data = newdata)
        if (!identical(colnames(new_design), design_columns)) {
          stop("Elastic-net prediction columns differ from training columns.",
               call. = FALSE)
        }
        as.numeric(stats::predict(fit, newx = new_design, s = "lambda.min"))
      },
      detail = sprintf("elastic net; alpha=0.5; lambda.min=%.6g", fit$lambda.min)
    ))
  }
  if (identical(kind, "ranger")) {
    forest_data <- train[c("interest", predictor_names)]
    fit <- ranger::ranger(
      dependent.variable.name = "interest",
      data = forest_data,
      case.weights = train$weight,
      num.trees = num_trees,
      min.node.size = 20L,
      sample.fraction = 0.8,
      respect.unordered.factors = "partition",
      seed = as.integer(seed),
      num.threads = 2L,
      write.forest = TRUE
    )
    return(list(
      predict = function(newdata) {
        as.numeric(predict(fit, data = newdata[predictor_names],
                           num.threads = 2L)$predictions)
      },
      detail = sprintf("ranger; trees=%d", num_trees)
    ))
  }
  stop("Unknown learner: ", kind, call. = FALSE)
}

target_plugin <- function(predict_fun, contexts) {
  contexts <- contexts[!duplicated(context_key(contexts)), , drop = FALSE]
  keys <- context_key(contexts)
  out <- matrix(NA_real_, nrow(contexts), ncol(contrast),
                dimnames = list(keys, contrast_names))
  task_levels <- levels(dat$task)
  for (i in seq_len(nrow(contexts))) {
    prediction_sum <- numeric(nrow(target))
    for (tt in task_levels) {
      newdata <- target[factor_names]
      newdata$task <- factor(tt, levels = task_levels)
      for (nm in demographic_names) {
        newdata[[nm]] <- factor(
          rep(as.character(contexts[[nm]][i]), nrow(target)),
          levels = levels(dat[[nm]])
        )
      }
      prediction_sum <- prediction_sum + predict_fun(newdata)
    }
    target_prediction <- prediction_sum / length(task_levels)
    out[i, ] <- as.numeric(crossprod(contrast, target_prediction))
  }
  out
}

cluster_scores_from_rows <- function(row_correction, plugin, valid_rows) {
  valid_ids <- unique(as.character(dat$id[valid_rows]))
  out <- matrix(NA_real_, length(valid_ids), ncol(contrast),
                dimnames = list(valid_ids, contrast_names))
  valid_id_values <- as.character(dat$id[valid_rows])
  for (i in seq_along(valid_ids)) {
    local <- which(valid_id_values == valid_ids[[i]])
    respondent_context <- context_by_id[match(valid_ids[[i]], ids)]
    out[i, ] <- plugin[respondent_context, ] +
      colMeans(row_correction[local, , drop = FALSE])
  }
  out
}

aggregate_scores <- function(scores) {
  if (anyNA(scores)) stop("At least one respondent score is missing.", call. = FALSE)
  estimate <- as.numeric(crossprod(cluster_weight / sum(cluster_weight), scores))
  centered <- sweep(scores, 2L, estimate, "-")
  covariance <- (g / (g - 1L)) * crossprod(centered * cluster_weight) /
    sum(cluster_weight)^2
  list(
    estimate = estimate,
    covariance = covariance,
    std_error = sqrt(pmax(diag(covariance), 0))
  )
}

models <- c(
  "unadjusted", "task_lm", "additive_lm", "interacted_lm",
  "elastic_net", "ranger"
)
score_list <- stats::setNames(vector("list", length(models)), models)
prediction_list <- stats::setNames(vector("list", length(models)), models)
detail_list <- stats::setNames(vector("list", length(models)), models)

# The unadjusted Horvitz--Thompson/randomization score is independent of folds.
unadjusted_rows <- riesz_observed * dat$interest
unadjusted_scores <- matrix(NA_real_, g, ncol(contrast),
                            dimnames = list(ids, contrast_names))
for (i in seq_along(ids)) {
  unadjusted_scores[i, ] <- colMeans(unadjusted_rows[id_index == i, , drop = FALSE])
}
score_list$unadjusted <- unadjusted_scores
prediction_list$unadjusted <- rep(0, nrow(dat))
detail_list$unadjusted <- "known-randomization score with zero outcome regression"

learner_names <- setdiff(models, "unadjusted")
for (kind in learner_names) {
  message("Cross-fitting ", kind, "...")
  scores <- matrix(NA_real_, g, ncol(contrast),
                   dimnames = list(ids, contrast_names))
  predictions <- rep(NA_real_, nrow(dat))
  fold_details <- character()
  for (fold in sort(unique(fold_by_id))) {
    train <- dat$fold != fold
    valid <- !train
    learner <- fit_learner(kind, dat[train, , drop = FALSE],
                           seed = 20260827L + 100L * fold + match(kind, learner_names))
    predictions[valid] <- learner$predict(dat[valid, , drop = FALSE])
    contexts <- dat[valid, demographic_names, drop = FALSE]
    plugin <- target_plugin(learner$predict, contexts)
    correction <- riesz_observed[valid, , drop = FALSE] *
      (dat$interest[valid] - predictions[valid])
    fold_scores <- cluster_scores_from_rows(correction, plugin, valid)
    scores[match(rownames(fold_scores), ids), ] <- fold_scores
    fold_details <- c(fold_details, sprintf("fold %d: %s", fold, learner$detail))
  }
  score_list[[kind]] <- scores
  prediction_list[[kind]] <- predictions
  detail_list[[kind]] <- paste(fold_details, collapse = "; ")
}

fits <- lapply(score_list, aggregate_scores)
unadjusted_se <- fits$unadjusted$std_error
coordinate_family <- bundle$coordinate_meta$family[
  match(contrast_names, bundle$coordinate_meta$coordinate)
]

coordinate_results <- do.call(rbind, lapply(models, function(model) {
  data.frame(
    model = model,
    estimand = contrast_names,
    family = coordinate_family,
    estimate = fits[[model]]$estimate,
    std_error = fits[[model]]$std_error,
    se_ratio_to_unadjusted = fits[[model]]$std_error / unadjusted_se,
    stringsAsFactors = FALSE
  )
}))

weighted_mse <- function(y, prediction, weight) {
  stats::weighted.mean((y - prediction)^2, weight)
}
prediction_metrics <- data.frame(
  model = models,
  weighted_mse = vapply(models, function(model) {
    weighted_mse(dat$interest, prediction_list[[model]], dat$weight)
  }, numeric(1)),
  stringsAsFactors = FALSE
)
prediction_metrics$rmse <- sqrt(prediction_metrics$weighted_mse)

overall_results <- do.call(rbind, lapply(models, function(model) {
  rows <- coordinate_results[coordinate_results$model == model, , drop = FALSE]
  data.frame(
    model = model,
    weighted_rmse = prediction_metrics$rmse[prediction_metrics$model == model],
    geometric_mean_se_ratio = exp(mean(log(rows$se_ratio_to_unadjusted))),
    median_se_ratio = stats::median(rows$se_ratio_to_unadjusted),
    fraction_coordinates_improved = mean(rows$se_ratio_to_unadjusted < 1),
    minimum_se_ratio = min(rows$se_ratio_to_unadjusted),
    maximum_se_ratio = max(rows$se_ratio_to_unadjusted),
    stringsAsFactors = FALSE
  )
}))

output_dir <- file.path(repo_root, "results", "narrow-ml")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
readr::write_csv(coordinate_results,
                 file.path(output_dir, "candidate_coordinate_precision.csv"))
readr::write_csv(overall_results,
                 file.path(output_dir, "candidate_precision_summary.csv"))
saveRDS(
  list(
    coordinate_results = coordinate_results,
    overall_results = overall_results,
    prediction_metrics = prediction_metrics,
    scores = score_list,
    details = detail_list,
    folds = stats::setNames(fold_by_id, ids),
    num_trees = num_trees
  ),
  file.path(output_dir, "candidate_precision_screen.rds"),
  version = 3
)

print(overall_results, row.names = FALSE, digits = 4)
