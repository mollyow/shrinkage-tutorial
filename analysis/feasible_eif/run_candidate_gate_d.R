#!/usr/bin/env Rscript

script_path <- function() {
  flag <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(flag)) {
    return(normalizePath("analysis/feasible_eif/run_candidate_gate_d.R"))
  }
  normalizePath(sub("^--file=", "", flag[[1L]]), mustWork = TRUE)
}
script <- script_path()
analysis_dir <- dirname(script)
repo_root <- normalizePath(file.path(analysis_dir, "..", ".."), mustWork = TRUE)
source(file.path(repo_root, "analysis", "covariance_headroom",
                 "calibration_solver.R"), local = TRUE)
source(file.path(analysis_dir, "feasible_estimator.R"), local = TRUE)
source(file.path(analysis_dir, "simulation_functions.R"), local = TRUE)

config <- read_gate_d_config(file.path(analysis_dir, "gate-d-production.yml"))
bundle_path <- file.path(repo_root, "analysis", "candidate", "derived",
                         "candidate_prepared.rds")
if (!file.exists(bundle_path)) stop("Prepared candidate bundle is missing.", call. = FALSE)
bundle <- readRDS(bundle_path)
dat <- as.data.frame(bundle$data)
target <- as.data.frame(bundle$target_grid)
contrast <- as.matrix(bundle$contrast_matrix)
factor_names <- as.character(bundle$attributes)

for (nm in factor_names) {
  locked_levels <- levels(bundle$data[[nm]])
  dat[[nm]] <- factor(as.character(dat[[nm]]), levels = locked_levels,
                      ordered = FALSE)
  target[[nm]] <- factor(as.character(target[[nm]]), levels = locked_levels,
                         ordered = FALSE)
}
task_levels <- sort(unique(dat$task))
dat$task_factor <- factor(dat$task, levels = task_levels)
formula <- stats::as.formula(paste0(
  "~ 0 + task_factor * (", paste(factor_names, collapse = " + "), ")"
))
design <- stats::model.matrix(formula, data = dat)
target_expanded <- target[rep(seq_len(nrow(target)), times = length(task_levels)),
                          factor_names, drop = FALSE]
target_expanded$task_factor <- factor(
  rep(task_levels, each = nrow(target)), levels = task_levels
)
target_design <- stats::model.matrix(formula, data = target_expanded)
if (!identical(colnames(design), colnames(target_design))) {
  stop("Observed and target mean-design columns differ.", call. = FALSE)
}

ids <- unique(as.character(dat$id))
respondent_count <- length(ids)
task_count <- length(task_levels)
id_index <- match(as.character(dat$id), ids)
task_index <- match(dat$task, task_levels)
row_lookup <- matrix(NA_integer_, respondent_count, task_count)
row_lookup[cbind(id_index, task_index)] <- seq_len(nrow(dat))
if (anyNA(row_lookup)) stop("Candidate respondents lack complete task vectors.", call. = FALSE)
outcome <- matrix(dat$interest[row_lookup], respondent_count, task_count)
profile_row <- match(as.character(dat$profile_id), as.character(target$profile_id))
if (anyNA(profile_row)) stop("Observed profile outside locked target.", call. = FALSE)
profile_index <- matrix(profile_row[row_lookup], respondent_count, task_count)
fold <- vapply(seq_len(respondent_count), function(i) {
  value <- unique(dat$fold[id_index == i])
  if (length(value) != 1L) stop("A respondent crosses folds.", call. = FALSE)
  as.integer(value)
}, integer(1))
cluster_weight <- vapply(seq_len(respondent_count), function(i) {
  value <- unique(dat$weight[id_index == i])
  if (length(value) != 1L) stop("A survey weight varies within respondent.", call. = FALSE)
  as.numeric(value)
}, numeric(1))
q <- as.numeric(bundle$contract$assignment_probability)
riesz_lookup <- contrast / q
invisible(validate_centered_riesz_lookup(riesz_lookup))
task_weights <- rep(1 / task_count, task_count)

fit_candidate_mean <- function(train_id) {
  train_row <- train_id[id_index]
  fit <- stats::lm.wfit(
    design[train_row, , drop = FALSE], dat$interest[train_row],
    w = dat$weight[train_row]
  )
  coefficient <- fit$coefficients
  coefficient[is.na(coefficient)] <- 0
  prediction <- drop(design %*% coefficient)
  target_prediction <- matrix(
    drop(target_design %*% coefficient), nrow(target), task_count
  )
  plugin <- numeric(ncol(contrast))
  for (tt in seq_len(task_count)) {
    plugin <- plugin + task_weights[[tt]] * drop(crossprod(
      contrast, target_prediction[, tt]
    ))
  }
  list(prediction = prediction, plugin = plugin, rank = fit$rank)
}

assignment_specs <- list(
  documented_product_uniform = list(mechanism = "product", support_size = NULL),
  ordered_uniform_without_replacement = list(
    mechanism = "uniform_without_replacement", support_size = nrow(target)
  )
)
method_names <- c(
  "ordinary", "selected", "safe_half_cs", "learned_cs_full",
  "learned_unstructured_full"
)
score_by_assignment <- list()
selector_rows <- list()
fold_rows <- list()

for (assignment_name in names(assignment_specs)) {
  message("Candidate feasible scores under ", assignment_name, "...")
  spec <- assignment_specs[[assignment_name]]
  score <- stats::setNames(lapply(method_names, function(x) {
    matrix(NA_real_, respondent_count, ncol(contrast),
           dimnames = list(ids, colnames(contrast)))
  }), method_names)
  for (outer in sort(unique(fold))) {
    outer_valid <- fold == outer
    outer_train <- !outer_valid
    inner_values <- sort(unique(fold[outer_train]))
    residual_oof <- matrix(NA_real_, respondent_count, task_count)
    inner_plugin <- list()
    for (inner in inner_values) {
      inner_valid <- outer_train & fold == inner
      inner_train <- outer_train & fold != inner
      fit <- fit_candidate_mean(inner_train)
      valid_rows <- row_lookup[inner_valid, , drop = FALSE]
      residual_oof[inner_valid, ] <- outcome[inner_valid, , drop = FALSE] -
        matrix(fit$prediction[valid_rows], sum(inner_valid), task_count)
      inner_plugin[[as.character(inner)]] <- fit$plugin
    }
    if (any(!is.finite(residual_oof[outer_train, , drop = FALSE]))) {
      stop("Candidate inner residuals are incomplete.", call. = FALSE)
    }
    candidate_scores <- NULL
    metadata <- NULL
    training_position <- integer(respondent_count)
    training_position[outer_train] <- seq_len(sum(outer_train))
    for (inner in inner_values) {
      selector_valid <- outer_train & fold == inner
      covariance_train <- outer_train & fold != inner
      covariance_set <- estimate_working_covariances(
        residual_oof[covariance_train, , drop = FALSE],
        cluster_weight[covariance_train],
        as.numeric(config$estimator$covariance_eigenvalue_relative_floor)
      )
      library <- make_operator_library(
        covariance_set, task_weights, spec$mechanism, spec$support_size,
        as.numeric(unlist(config$estimator$blend_grid))
      )
      if (is.null(candidate_scores)) {
        metadata <- attr(library, "metadata")
        candidate_scores <- stats::setNames(lapply(names(library), function(x) {
          matrix(NA_real_, sum(outer_train), ncol(contrast))
        }), names(library))
      }
      local <- training_position[selector_valid]
      for (candidate in names(library)) {
        candidate_scores[[candidate]][local, ] <- score_factorial_clusters(
          residual_oof[selector_valid, , drop = FALSE],
          profile_index[selector_valid, , drop = FALSE],
          riesz_lookup, library[[candidate]],
          inner_plugin[[as.character(inner)]]
        )
      }
    }
    selection <- select_operator_one_se(
      candidate_scores, metadata, cluster_weight[outer_train]
    )
    covariance_set <- estimate_working_covariances(
      residual_oof[outer_train, , drop = FALSE], cluster_weight[outer_train],
      as.numeric(config$estimator$covariance_eigenvalue_relative_floor)
    )
    library <- make_operator_library(
      covariance_set, task_weights, spec$mechanism, spec$support_size,
      as.numeric(unlist(config$estimator$blend_grid))
    )
    outer_fit <- fit_candidate_mean(outer_train)
    valid_rows <- row_lookup[outer_valid, , drop = FALSE]
    valid_residual <- outcome[outer_valid, , drop = FALSE] -
      matrix(outer_fit$prediction[valid_rows], sum(outer_valid), task_count)
    operators <- list(
      ordinary = library$ordinary,
      selected = library[[selection$candidate]],
      safe_half_cs = library$compound_symmetry_tau_0.5,
      learned_cs_full = library$compound_symmetry_tau_1,
      learned_unstructured_full = library$unstructured_tau_1
    )
    for (method in names(operators)) {
      score[[method]][outer_valid, ] <- score_factorial_clusters(
        valid_residual, profile_index[outer_valid, , drop = FALSE],
        riesz_lookup, operators[[method]], outer_fit$plugin
      )
    }
    selected_operator <- operators$selected
    selector_rows[[length(selector_rows) + 1L]] <- data.frame(
      assignment = assignment_name,
      outer_fold = outer,
      selected_candidate = selection$candidate,
      selected_blend = selected_operator$blend,
      selected_covariance_class = selected_operator$covariance_class,
      calibration_residual = selected_operator$calibration_residual,
      cs_projected = covariance_set$compound_symmetry$projected,
      unstructured_projected = covariance_set$unstructured$projected,
      stringsAsFactors = FALSE
    )
    fold_rows[[length(fold_rows) + 1L]] <- data.frame(
      assignment = assignment_name,
      outer_fold = outer,
      training_respondents = sum(outer_train),
      validation_respondents = sum(outer_valid),
      mean_rank = outer_fit$rank,
      maximum_absolute_plugin = max(abs(outer_fit$plugin)),
      stringsAsFactors = FALSE
    )
  }
  if (any(!vapply(score, function(x) all(is.finite(x)), logical(1)))) {
    stop("Candidate feasible scores are incomplete.", call. = FALSE)
  }
  score_by_assignment[[assignment_name]] <- score
}

coordinate_family <- as.character(bundle$coordinate_meta$family[
  match(colnames(contrast), bundle$coordinate_meta$coordinate)
])
families <- unique(coordinate_family)
family_metric <- function(covariance) {
  vapply(families, function(family) {
    mean(diag(covariance)[coordinate_family == family])
  }, numeric(1))
}

summary_rows <- list()
family_rows <- list()
bootstrap_rows <- list()
bootstrap_draws <- 5000L
set.seed(2026082713L)
for (assignment_name in names(score_by_assignment)) {
  score <- score_by_assignment[[assignment_name]]
  fits <- lapply(score, weighted_score_summary, cluster_weight = cluster_weight)
  ordinary_family <- family_metric(fits$ordinary$covariance)
  for (method in method_names) {
    method_family <- family_metric(fits[[method]]$covariance)
    family_ratio <- method_family / ordinary_family
    summary_rows[[length(summary_rows) + 1L]] <- data.frame(
      assignment = assignment_name,
      method = method,
      equal_family_variance_ratio = mean(method_family) / mean(ordinary_family),
      maximum_family_variance_ratio = max(family_ratio),
      minimum_family_variance_ratio = min(family_ratio),
      stringsAsFactors = FALSE
    )
    for (family in families) {
      family_rows[[length(family_rows) + 1L]] <- data.frame(
        assignment = assignment_name,
        method = method,
        family = family,
        dimension = sum(coordinate_family == family),
        ordinary_trace_per_dimension = ordinary_family[[family]],
        method_trace_per_dimension = method_family[[family]],
        variance_ratio = family_ratio[[family]],
        stringsAsFactors = FALSE
      )
    }
  }
  for (method in setdiff(method_names, "ordinary")) {
    values <- numeric(bootstrap_draws)
    for (bb in seq_len(bootstrap_draws)) {
      index <- sample.int(respondent_count, respondent_count, replace = TRUE)
      ordinary_cov <- weighted_score_summary(
        score$ordinary[index, , drop = FALSE], cluster_weight[index]
      )$covariance
      method_cov <- weighted_score_summary(
        score[[method]][index, , drop = FALSE], cluster_weight[index]
      )$covariance
      values[[bb]] <- mean(family_metric(method_cov)) /
        mean(family_metric(ordinary_cov))
    }
    bootstrap_rows[[length(bootstrap_rows) + 1L]] <- data.frame(
      assignment = assignment_name,
      method = method,
      bootstrap_mean = mean(values),
      lower_2_5 = unname(stats::quantile(values, 0.025, type = 8)),
      upper_97_5 = unname(stats::quantile(values, 0.975, type = 8)),
      stringsAsFactors = FALSE
    )
  }
}
summary_rows <- do.call(rbind, summary_rows)
family_rows <- do.call(rbind, family_rows)
bootstrap_rows <- do.call(rbind, bootstrap_rows)
selector_rows <- do.call(rbind, selector_rows)
fold_rows <- do.call(rbind, fold_rows)

oracle_path <- file.path(repo_root, "results", "covariance-headroom",
                         "oracle_point_results.csv")
oracle <- utils::read.csv(oracle_path, stringsAsFactors = FALSE)
oracle <- oracle[
  oracle$residualizer == "task_specific_additive_wls" &
    oracle$covariance_class == "unstructured_8_by_8",
  c("assignment", "efficient_to_ordinary_ratio"), drop = FALSE
]
oracle$assignment <- sub("documented_product_uniform",
                         "documented_product_uniform", oracle$assignment,
                         fixed = TRUE)
selected_summary <- merge(
  summary_rows[summary_rows$method == "selected", , drop = FALSE],
  bootstrap_rows[bootstrap_rows$method == "selected", , drop = FALSE],
  by = c("assignment", "method"), sort = FALSE
)
selected_summary <- merge(selected_summary, oracle, by = "assignment",
                          all.x = TRUE, sort = FALSE)
selected_summary$oracle_gain_recovered <-
  (1 - selected_summary$equal_family_variance_ratio) /
  (1 - selected_summary$efficient_to_ordinary_ratio)
selected_summary$supporting_target_pass <- with(
  selected_summary,
  equal_family_variance_ratio <= config$application$desired_point_variance_ratio_max &
    upper_97_5 < config$application$desired_upper_97_5_max &
    oracle_gain_recovered >= 0.5 & maximum_family_variance_ratio <= 1.05
)

output_dir <- file.path(repo_root, "results", "feasible-eif", "candidate")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
utils::write.csv(summary_rows, file.path(output_dir, "method_summary.csv"),
                 row.names = FALSE)
utils::write.csv(family_rows, file.path(output_dir, "family_summary.csv"),
                 row.names = FALSE)
utils::write.csv(bootstrap_rows, file.path(output_dir, "bootstrap_summary.csv"),
                 row.names = FALSE)
utils::write.csv(selected_summary,
                 file.path(output_dir, "selected_supporting_summary.csv"),
                 row.names = FALSE)
utils::write.csv(selector_rows, file.path(output_dir, "selector_by_fold.csv"),
                 row.names = FALSE)
utils::write.csv(fold_rows, file.path(output_dir, "fold_diagnostics.csv"),
                 row.names = FALSE)
decision <- list(
  role = "supporting_non_gating_application_evidence",
  all_supporting_targets_passed = all(selected_summary$supporting_target_pass),
  maximum_calibration_residual = max(selector_rows$calibration_residual),
  maximum_absolute_plugin = max(fold_rows$maximum_absolute_plugin),
  product_selected_ratio = selected_summary$equal_family_variance_ratio[
    selected_summary$assignment == "documented_product_uniform"
  ],
  without_replacement_selected_ratio =
    selected_summary$equal_family_variance_ratio[
      selected_summary$assignment == "ordered_uniform_without_replacement"
    ],
  assignment_limitation = paste(
    "The platform transition law is unavailable; product and uniform",
    "without-replacement scores are parallel sensitivities, not verified laws."
  ),
  generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
)
yaml::write_yaml(decision, file.path(output_dir, "candidate_decision.yml"))
saveRDS(list(
  decision = decision, scores = score_by_assignment,
  summary = summary_rows, family = family_rows, bootstrap = bootstrap_rows,
  selected_summary = selected_summary, selector = selector_rows,
  folds = fold_rows
), file.path(output_dir, "candidate_gate_d_results.rds"), version = 3)

print(selected_summary, row.names = FALSE, digits = 5)
cat("Candidate supporting targets: ",
    if (all(selected_summary$supporting_target_pass)) "PASS" else "FAIL",
    "\n", sep = "")
