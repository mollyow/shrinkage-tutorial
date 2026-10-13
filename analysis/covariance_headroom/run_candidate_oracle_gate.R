#!/usr/bin/env Rscript

required <- c("yaml")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  stop("Install required packages: ", paste(missing, collapse = ", "), call. = FALSE)
}

script_path <- function() {
  flag <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(flag)) {
    return(normalizePath(
      "analysis/covariance_headroom/run_candidate_oracle_gate.R",
      mustWork = TRUE
    ))
  }
  normalizePath(sub("^--file=", "", flag[[1L]]), mustWork = TRUE)
}

script <- script_path()
analysis_dir <- dirname(script)
repo_root <- normalizePath(file.path(analysis_dir, "..", ".."), mustWork = TRUE)
source(file.path(analysis_dir, "calibration_solver.R"), local = TRUE)

config_path <- file.path(analysis_dir, "oracle-gate.yml")
config <- yaml::read_yaml(config_path)
bundle_path <- file.path(repo_root, config$data$prepared_bundle)
if (!file.exists(bundle_path)) {
  stop("Prepared candidate bundle is missing: ", bundle_path, call. = FALSE)
}
bundle <- readRDS(bundle_path)
dat <- as.data.frame(bundle$data)
target <- as.data.frame(bundle$target_grid)
contrast <- as.matrix(bundle$contrast_matrix)
factor_names <- as.character(bundle$attributes)

stopifnot(
  nrow(dat) == config$data$respondents * config$data$tasks_per_respondent,
  nrow(target) == config$data$profile_support,
  nrow(contrast) == nrow(target),
  ncol(contrast) == bundle$contract$coordinate_dimension,
  all(c("id", "task", "interest", "weight", "fold", "profile_id") %in% names(dat))
)

# Lock all randomized attributes to nominal factor spaces for this screen.
for (nm in factor_names) {
  declared <- levels(bundle$data[[nm]])
  dat[[nm]] <- factor(as.character(dat[[nm]]), levels = declared, ordered = FALSE)
}
dat$task_factor <- factor(dat$task, levels = sort(unique(dat$task)))

ids <- unique(as.character(dat$id))
respondent_count <- length(ids)
task_levels <- as.integer(levels(dat$task_factor))
task_count <- length(task_levels)
id_index <- match(as.character(dat$id), ids)
task_index <- match(dat$task, task_levels)
if (respondent_count != config$data$respondents ||
    task_count != config$data$tasks_per_respondent ||
    anyNA(id_index) || anyNA(task_index)) {
  stop("Candidate respondent/task geometry differs from the frozen gate.", call. = FALSE)
}
row_lookup <- matrix(NA_integer_, respondent_count, task_count)
row_lookup[cbind(id_index, task_index)] <- seq_len(nrow(dat))
if (anyNA(row_lookup)) stop("At least one respondent lacks a complete task vector.", call. = FALSE)

respondent_weight <- vapply(seq_len(respondent_count), function(i) {
  values <- dat$weight[id_index == i]
  if (max(values) - min(values) > 1e-12) {
    stop("Survey weights vary within respondent.", call. = FALSE)
  }
  values[[1L]]
}, numeric(1))
fold_by_id <- vapply(seq_len(respondent_count), function(i) {
  values <- unique(dat$fold[id_index == i])
  if (length(values) != 1L) stop("A respondent crosses outer folds.", call. = FALSE)
  as.integer(values)
}, integer(1))

outcome_matrix <- matrix(dat$interest[row_lookup], respondent_count, task_count)
profile_lookup <- match(as.character(dat$profile_id), as.character(target$profile_id))
if (anyNA(profile_lookup)) stop("An observed profile is outside the target grid.", call. = FALSE)
profile_matrix <- matrix(profile_lookup[row_lookup], respondent_count, task_count)
duplicate_profile_respondents <- sum(apply(profile_matrix, 1L, anyDuplicated) > 0L)

rhs <- paste(factor_names, collapse = " + ")
model_formulas <- list(
  task_only_wls = stats::as.formula("~ 0 + task_factor"),
  task_specific_additive_wls = stats::as.formula(
    paste0("~ 0 + task_factor * (", rhs, ")")
  ),
  common_two_way_wls = stats::as.formula(
    paste0("~ 0 + task_factor + (", rhs, ")^2")
  )
)

crossfit_wls <- function(formula, label) {
  message("Cross-fitting ", label, "...")
  design <- stats::model.matrix(formula, data = dat)
  prediction <- rep(NA_real_, nrow(dat))
  fold_details <- vector("list", length(sort(unique(fold_by_id))))
  for (ff in sort(unique(fold_by_id))) {
    valid_id <- fold_by_id == ff
    valid <- valid_id[id_index]
    train <- !valid
    fit <- stats::lm.wfit(
      x = design[train, , drop = FALSE], y = dat$interest[train],
      w = dat$weight[train]
    )
    coefficient <- fit$coefficients
    coefficient[is.na(coefficient)] <- 0
    prediction[valid] <- drop(design[valid, , drop = FALSE] %*% coefficient)
    fold_details[[match(ff, sort(unique(fold_by_id)))]] <- data.frame(
      residualizer = label,
      fold = ff,
      training_rows = sum(train),
      validation_rows = sum(valid),
      columns = ncol(design),
      rank = fit$rank,
      stringsAsFactors = FALSE
    )
  }
  if (any(!is.finite(prediction))) {
    stop("Cross-fitted predictions are incomplete for ", label, call. = FALSE)
  }
  residual <- dat$interest - prediction
  list(
    prediction = prediction,
    residual_matrix = matrix(residual[row_lookup], respondent_count, task_count),
    details = do.call(rbind, fold_details)
  )
}

residual_fits <- lapply(names(model_formulas), function(label) {
  crossfit_wls(model_formulas[[label]], label)
})
names(residual_fits) <- names(model_formulas)
fit_details <- do.call(rbind, lapply(residual_fits, `[[`, "details"))

residual_second_moment <- function(residual, survey_weight, centered = FALSE) {
  residual <- as.matrix(residual)
  variance_weight <- as.numeric(survey_weight)^2
  variance_weight <- variance_weight / sum(variance_weight)
  if (centered) {
    center <- drop(crossprod(variance_weight, residual))
    residual <- sweep(residual, 2L, center, "-")
  }
  weighted_residual <- residual * sqrt(variance_weight)
  symmetric_part(crossprod(weighted_residual))
}

stabilize_covariance <- function(covariance, relative_floor) {
  covariance <- symmetric_part(covariance)
  eig <- eigen(covariance, symmetric = TRUE)
  floor_value <- relative_floor * max(max(eig$values), mean(diag(covariance)), 1e-12)
  adjusted_values <- pmax(eig$values, floor_value)
  adjusted <- eig$vectors %*% (adjusted_values * t(eig$vectors))
  list(
    covariance = symmetric_part(adjusted),
    raw_eigenvalues = eig$values,
    adjusted_eigenvalues = adjusted_values,
    floor = floor_value,
    changed = any(eig$values < floor_value)
  )
}

covariance_classes <- function(residual, survey_weight) {
  raw <- residual_second_moment(residual, survey_weight, centered = FALSE)
  unstructured <- stabilize_covariance(
    raw, config$covariance$eigenvalue_relative_floor
  )
  cs_moment <- matrix(
    mean(raw[row(raw) != col(raw)]), task_count, task_count
  )
  diag(cs_moment) <- mean(diag(raw))
  cs <- stabilize_covariance(cs_moment, config$covariance$eigenvalue_relative_floor)
  list(
    unstructured_8_by_8 = unstructured,
    compound_symmetry = cs
  )
}

assignment_specs <- list(
  documented_product_uniform = list(
    mechanism = "product", support_size = NULL
  ),
  ordered_uniform_without_replacement = list(
    mechanism = "uniform_without_replacement", support_size = nrow(target)
  )
)
task_weights <- rep(1 / task_count, task_count)

moment_diagnostics <- list()
point_results <- list()
operators <- list()
for (residualizer in names(residual_fits)) {
  covariance_set <- covariance_classes(
    residual_fits[[residualizer]]$residual_matrix, respondent_weight
  )
  for (covariance_class in names(covariance_set)) {
    covariance_fit <- covariance_set[[covariance_class]]
    covariance_value <- covariance_fit$covariance
    correlation <- stats::cov2cor(covariance_value)
    moment_diagnostics[[length(moment_diagnostics) + 1L]] <- data.frame(
      residualizer = residualizer,
      covariance_class = covariance_class,
      raw_minimum_eigenvalue = min(covariance_fit$raw_eigenvalues),
      adjusted_minimum_eigenvalue = min(covariance_fit$adjusted_eigenvalues),
      maximum_eigenvalue = max(covariance_fit$adjusted_eigenvalues),
      condition_number = max(covariance_fit$adjusted_eigenvalues) /
        min(covariance_fit$adjusted_eigenvalues),
      eigenvalue_floor = covariance_fit$floor,
      floor_applied = covariance_fit$changed,
      mean_diagonal_second_moment = mean(diag(covariance_value)),
      mean_off_diagonal_second_moment =
        mean(covariance_value[row(covariance_value) != col(covariance_value)]),
      mean_residual_correlation =
        mean(correlation[row(correlation) != col(correlation)]),
      stringsAsFactors = FALSE
    )
    for (assignment in names(assignment_specs)) {
      spec <- assignment_specs[[assignment]]
      operator <- solve_task_mixing(
        covariance = covariance_value,
        task_weights = task_weights,
        mechanism = spec$mechanism,
        support_size = spec$support_size
      )
      key <- paste(residualizer, covariance_class, assignment, sep = "::")
      operators[[key]] <- operator
      point_results[[length(point_results) + 1L]] <- data.frame(
        residualizer = residualizer,
        covariance_class = covariance_class,
        assignment = assignment,
        ordinary_multiplier = operator$ordinary_multiplier,
        efficient_multiplier = operator$optimal_multiplier,
        efficient_to_ordinary_ratio = operator$variance_ratio,
        covariance_headroom = 1 - operator$variance_ratio,
        calibration_residual = operator$calibration_residual,
        stringsAsFactors = FALSE
      )
    }
  }
}
point_results <- do.call(rbind, point_results)
moment_diagnostics <- do.call(rbind, moment_diagnostics)

# Conditional respondent bootstrap for the plug-in oracle functional.  The
# nuisance residuals remain fixed, which is why this is a Gate-C uncertainty
# calculation rather than evidence for a feasible estimator.
primary_residual <- residual_fits[[config$residualization$primary]]$residual_matrix
bootstrap_draws <- as.integer(config$bootstrap$draws)
set.seed(as.integer(config$bootstrap$seed))
bootstrap_keys <- as.vector(outer(
  c("unstructured_8_by_8", "compound_symmetry"),
  names(assignment_specs), paste, sep = "::"
))
bootstrap_ratio <- matrix(
  NA_real_, bootstrap_draws, length(bootstrap_keys),
  dimnames = list(NULL, bootstrap_keys)
)
message("Running ", bootstrap_draws, " respondent bootstrap draws...")
for (bb in seq_len(bootstrap_draws)) {
  sampled <- sample.int(respondent_count, respondent_count, replace = TRUE)
  covariance_set <- covariance_classes(
    primary_residual[sampled, , drop = FALSE], respondent_weight[sampled]
  )
  for (covariance_class in names(covariance_set)) {
    for (assignment in names(assignment_specs)) {
      spec <- assignment_specs[[assignment]]
      operator <- solve_task_mixing(
        covariance_set[[covariance_class]]$covariance,
        task_weights,
        spec$mechanism,
        spec$support_size
      )
      bootstrap_ratio[bb, paste(covariance_class, assignment, sep = "::")] <-
        operator$variance_ratio
    }
  }
}

bootstrap_summary <- do.call(rbind, lapply(seq_along(bootstrap_keys), function(j) {
  pieces <- strsplit(bootstrap_keys[[j]], "::", fixed = TRUE)[[1L]]
  values <- bootstrap_ratio[, j]
  data.frame(
    residualizer = config$residualization$primary,
    covariance_class = pieces[[1L]],
    assignment = pieces[[2L]],
    bootstrap_mean = mean(values),
    bootstrap_standard_error = stats::sd(values),
    lower_2_5 = unname(stats::quantile(values, 0.025, type = 8)),
    upper_97_5 = unname(stats::quantile(values, 0.975, type = 8)),
    stringsAsFactors = FALSE
  )
}))

# Non-gating, observed-geometry diagnostic.  It applies the full-sample oracle
# operator to each respondent's actual profile schedule.  Since the primary
# nuisance contains only task-specific main effects, the plug-in value of every
# locked two-way interaction is zero and the correction is the complete score.
q <- as.numeric(bundle$contract$assignment_probability)
if (abs(q * nrow(target) - 1) > 1e-12) {
  stop("The candidate gate requires the locked uniform profile marginal.", call. = FALSE)
}
riesz_lookup <- contrast / q
coordinate_names <- colnames(contrast)
coordinate_family <- as.character(bundle$coordinate_meta$family[
  match(coordinate_names, bundle$coordinate_meta$coordinate)
])
families <- unique(coordinate_family)

influence_covariance <- function(score) {
  estimate <- drop(crossprod(respondent_weight / sum(respondent_weight), score))
  centered <- sweep(score, 2L, estimate, "-")
  influence <- centered * (respondent_weight / mean(respondent_weight))
  crossprod(influence) / (nrow(influence) - 1L)
}

family_geometry <- list()
primary_e <- primary_residual
ordinary_score <- matrix(0, respondent_count, ncol(contrast))
for (i in seq_len(respondent_count)) {
  F <- riesz_lookup[profile_matrix[i, ], , drop = FALSE]
  ordinary_score[i, ] <- drop(crossprod(diag(task_weights) %*% F, primary_e[i, ]))
}
ordinary_covariance <- influence_covariance(ordinary_score)

for (covariance_class in c("unstructured_8_by_8", "compound_symmetry")) {
  for (assignment in names(assignment_specs)) {
    key <- paste(config$residualization$primary, covariance_class, assignment,
                 sep = "::")
    operator <- operators[[key]]
    efficient_score <- matrix(0, respondent_count, ncol(contrast))
    for (i in seq_len(respondent_count)) {
      F <- riesz_lookup[profile_matrix[i, ], , drop = FALSE]
      efficient_score[i, ] <- drop(crossprod(
        apply_task_mixing(operator, F), primary_e[i, ]
      ))
    }
    efficient_covariance <- influence_covariance(efficient_score)
    family_rows <- lapply(families, function(family) {
      keep <- coordinate_family == family
      ordinary_trace_per_dimension <- mean(diag(ordinary_covariance)[keep])
      efficient_trace_per_dimension <- mean(diag(efficient_covariance)[keep])
      data.frame(
        covariance_class = covariance_class,
        assignment = assignment,
        family = family,
        dimension = sum(keep),
        ordinary_trace_per_dimension = ordinary_trace_per_dimension,
        efficient_trace_per_dimension = efficient_trace_per_dimension,
        efficient_to_ordinary_ratio =
          efficient_trace_per_dimension / ordinary_trace_per_dimension,
        stringsAsFactors = FALSE
      )
    })
    family_rows <- do.call(rbind, family_rows)
    family_rows <- rbind(family_rows, data.frame(
      covariance_class = covariance_class,
      assignment = assignment,
      family = "equal_family_aggregate",
      dimension = ncol(contrast),
      ordinary_trace_per_dimension = mean(family_rows$ordinary_trace_per_dimension),
      efficient_trace_per_dimension = mean(family_rows$efficient_trace_per_dimension),
      efficient_to_ordinary_ratio =
        mean(family_rows$efficient_trace_per_dimension) /
        mean(family_rows$ordinary_trace_per_dimension),
      stringsAsFactors = FALSE
    ))
    family_geometry[[length(family_geometry) + 1L]] <- family_rows
  }
}
family_geometry <- do.call(rbind, family_geometry)

# Apply the frozen Gate-C rule only to the primary residualizer.  Both covariance
# classes and both assignment scenarios must clear the inherited thresholds.
gate_rows <- merge(
  point_results[
    point_results$residualizer == config$residualization$primary &
      point_results$covariance_class %in%
        c(config$covariance$primary, config$covariance$sensitivity),
    , drop = FALSE
  ],
  bootstrap_summary,
  by = c("residualizer", "covariance_class", "assignment"),
  all.x = TRUE,
  sort = FALSE
)
gate_rows$point_pass <- gate_rows$efficient_to_ordinary_ratio <=
  config$pass_rules$primary_point_ratio_max
gate_rows$upper_bound_pass <- gate_rows$upper_97_5 <
  config$pass_rules$primary_upper_bound_max
gate_rows$calibration_pass <- gate_rows$calibration_residual <=
  config$pass_rules$maximum_calibration_residual
gate_rows$passed <- with(
  gate_rows, point_pass & upper_bound_pass & calibration_pass
)
gate_passed <- all(gate_rows$passed)

output_dir <- file.path(repo_root, "results", "covariance-headroom")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
utils::write.csv(point_results, file.path(output_dir, "oracle_point_results.csv"),
                 row.names = FALSE)
utils::write.csv(bootstrap_summary,
                 file.path(output_dir, "oracle_bootstrap_summary.csv"),
                 row.names = FALSE)
utils::write.csv(gate_rows, file.path(output_dir, "oracle_gate_rows.csv"),
                 row.names = FALSE)
utils::write.csv(moment_diagnostics,
                 file.path(output_dir, "residual_moment_diagnostics.csv"),
                 row.names = FALSE)
utils::write.csv(family_geometry,
                 file.path(output_dir, "family_geometry_diagnostics.csv"),
                 row.names = FALSE)
utils::write.csv(fit_details, file.path(output_dir, "crossfit_fit_details.csv"),
                 row.names = FALSE)

decision <- list(
  gate = "C_oracle_covariance_headroom",
  passed = gate_passed,
  decision = if (gate_passed) "proceed_to_feasible_estimator_gate_D" else
    "stop_or_reframe_covariance_efficiency_route",
  primary_residualizer = config$residualization$primary,
  point_threshold = config$pass_rules$primary_point_ratio_max,
  upper_bound_threshold = config$pass_rules$primary_upper_bound_max,
  worst_required_point_ratio = max(gate_rows$efficient_to_ordinary_ratio),
  worst_required_upper_97_5 = max(gate_rows$upper_97_5),
  maximum_calibration_residual = max(gate_rows$calibration_residual),
  duplicate_profile_respondents = duplicate_profile_respondents,
  assignment_warning = config$assignment_scenarios$limitation,
  config_md5 = unname(tools::md5sum(config_path)),
  prepared_bundle_md5 = unname(tools::md5sum(bundle_path)),
  generated_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
)
yaml::write_yaml(decision, file.path(output_dir, "oracle_gate_decision.yml"))
saveRDS(list(
  config = config,
  decision = decision,
  point_results = point_results,
  bootstrap_summary = bootstrap_summary,
  gate_rows = gate_rows,
  moment_diagnostics = moment_diagnostics,
  family_geometry = family_geometry,
  fit_details = fit_details,
  residuals = lapply(residual_fits, `[[`, "residual_matrix"),
  bootstrap_ratio = bootstrap_ratio,
  operators = operators,
  profile_matrix = profile_matrix,
  respondent_weight = respondent_weight,
  fold_by_id = fold_by_id
), file.path(output_dir, "oracle_gate_results.rds"), version = 3)

cat("\nFrozen Gate-C rows:\n")
print(gate_rows[c(
  "covariance_class", "assignment", "efficient_to_ordinary_ratio",
  "lower_2_5", "upper_97_5", "calibration_residual", "passed"
)], row.names = FALSE, digits = 5)
cat(sprintf("\nGATE C: %s\n", if (gate_passed) "PASS" else "FAIL"))
cat("Decision: ", decision$decision, "\n", sep = "")
