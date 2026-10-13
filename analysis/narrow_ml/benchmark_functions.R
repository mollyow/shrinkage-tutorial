`%||%` <- function(x, y) if (is.null(x)) y else x

benchmark_assert <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call. = FALSE)
  invisible(TRUE)
}

read_benchmark_config <- function(path) {
  if (!requireNamespace("yaml", quietly = TRUE)) {
    stop("Package `yaml` is required.", call. = FALSE)
  }
  config <- yaml::read_yaml(path)
  required <- c(
    "version", "mode", "seed", "replications", "bootstrap_draws",
    "respondents", "tasks_per_respondent", "profile_sampling", "outcome",
    "outer_folds", "inner_folds", "models",
    "forest", "elastic_net", "scenarios", "confirmatory_scenarios",
    "thresholds"
  )
  missing <- setdiff(required, names(config))
  benchmark_assert(!length(missing), paste(
    "Benchmark configuration is missing:", paste(missing, collapse = ", ")
  ))
  allowed_scenarios <- c(
    "additive_null", "linear_heterogeneity", "nonlinear_heterogeneity",
    "nonlinear_prognostic", "nonlinear_target_shift", "additive_target_shift"
  )
  benchmark_assert(
    all(unlist(config$scenarios) %in% allowed_scenarios),
    "The benchmark configuration contains an unknown scenario."
  )
  integer_fields <- c(
    "seed", "replications", "bootstrap_draws", "respondents",
    "tasks_per_respondent", "outer_folds", "inner_folds"
  )
  for (field in integer_fields) {
    benchmark_assert(
      length(config[[field]]) == 1L && is.finite(config[[field]]) &&
        config[[field]] >= 1,
      paste0("`", field, "` must be a positive scalar.")
    )
    config[[field]] <- as.integer(config[[field]])
  }
  benchmark_assert(config$outer_folds >= 2L, "At least two outer folds are required.")
  benchmark_assert(config$inner_folds >= 2L, "At least two inner folds are required.")
  benchmark_assert(config$forest$trees >= 50L, "At least 50 forest trees are required.")
  benchmark_assert(
    identical(config$tasks_per_respondent, 8L),
    "The candidate-geometry benchmark requires eight tasks per respondent."
  )
  benchmark_assert(
    identical(config$profile_sampling, "without_replacement_within_respondent"),
    "The primary benchmark must match the observed no-repeat profile geometry."
  )
  benchmark_assert(
    all(c("target_mean", "signal_scale", "beta_precision",
          "respondent_copula_rho", "rating_intervals") %in% names(config$outcome)),
    "The outcome calibration is incomplete."
  )
  benchmark_assert(config$outcome$target_mean > 0 && config$outcome$target_mean < 1,
                   "The target outcome mean must be inside (0, 1).")
  benchmark_assert(config$outcome$signal_scale > 0 &&
                     config$outcome$signal_scale <= 1 &&
                     config$outcome$beta_precision > 0,
                   "The signal scale and beta precision must be positive.")
  benchmark_assert(config$outcome$respondent_copula_rho >= 0 &&
                     config$outcome$respondent_copula_rho < 1,
                   "The respondent copula correlation must be in [0, 1).")
  benchmark_assert(config$outcome$rating_intervals == 9L,
                   "The candidate rating outcome requires nine intervals.")
  config$scenarios <- as.character(unlist(config$scenarios, use.names = FALSE))
  config$models <- as.character(unlist(config$models, use.names = FALSE))
  allowed_models <- c(
    "unadjusted", "additive_lm", "interacted_lm", "elastic_net", "ranger",
    "plugin_ranger", "oracle"
  )
  benchmark_assert(length(config$models) > 0L &&
                     all(config$models %in% allowed_models) &&
                     !anyDuplicated(config$models),
                   "The configured benchmark model set is invalid.")
  benchmark_assert(!("plugin_ranger" %in% config$models) ||
                     "ranger" %in% config$models,
                   "The ranger plugin requires the corrected ranger fit.")
  config
}

context_key <- function(data, context_names) {
  do.call(paste, c(lapply(data[context_names], as.character), sep = "\u001e"))
}

load_candidate_benchmark_design <- function(repo_root) {
  if (!requireNamespace("readr", quietly = TRUE)) {
    stop("Package `readr` is required.", call. = FALSE)
  }
  bundle_path <- file.path(
    repo_root, "analysis", "candidate", "derived", "candidate_prepared.rds"
  )
  raw_path <- file.path(
    repo_root, "Candidate recruitment", "replication", "PSRM_replication",
    "data", "CleanedData.csv"
  )
  benchmark_assert(file.exists(bundle_path), "The prepared candidate bundle is missing.")
  benchmark_assert(file.exists(raw_path), "The cleaned candidate replication data are missing.")

  bundle <- readRDS(bundle_path)
  factor_names <- as.character(bundle$attributes)
  context_names <- c("woman", "poc", "young", "dem", "rep", "workingclass")
  raw <- readr::read_csv(
    raw_path,
    col_select = tidyselect::all_of(c("id", "iteration", "weight", context_names)),
    col_types = readr::cols(
      id = readr::col_character(),
      iteration = readr::col_integer(),
      .default = readr::col_double()
    ),
    show_col_types = FALSE,
    progress = FALSE
  )
  respondent <- raw[!duplicated(raw$id), c("id", "weight", context_names)]
  benchmark_assert(nrow(respondent) == 3000L, "Expected 3,000 candidate respondents.")
  invariance <- vapply(split(raw, raw$id), function(x) {
    all(vapply(c("weight", context_names), function(nm) {
      length(unique(x[[nm]])) == 1L
    }, logical(1)))
  }, logical(1))
  benchmark_assert(all(invariance), "A benchmark context or weight varies within respondent.")
  benchmark_assert(!anyNA(respondent[c("weight", context_names)]),
                   "Benchmark contexts and weights must be complete.")

  key <- context_key(respondent, context_names)
  split_index <- split(seq_len(nrow(respondent)), key)
  contexts <- do.call(rbind, lapply(split_index, function(index) {
    out <- respondent[index[[1L]], context_names, drop = FALSE]
    out$population_weight <- sum(respondent$weight[index])
    out
  }))
  rownames(contexts) <- NULL
  contexts$population_probability <- contexts$population_weight /
    sum(contexts$population_weight)
  contexts$.context_key <- context_key(contexts, context_names)

  target <- as.data.frame(bundle$target_grid)
  for (nm in factor_names) {
    locked_levels <- levels(bundle$data[[nm]])
    target[[nm]] <- factor(as.character(target[[nm]]),
                           levels = locked_levels, ordered = FALSE)
  }
  target$profile_id <- as.character(target$profile_id)
  contrast <- as.matrix(bundle$contrast_matrix)
  benchmark_assert(nrow(target) == 1440L, "Expected the complete 1,440-profile grid.")
  benchmark_assert(nrow(contrast) == nrow(target) && ncol(contrast) == 22L,
                   "Expected the locked 1,440 by 22 contrast matrix.")
  benchmark_assert(max(abs(crossprod(contrast) - diag(22L))) < 1e-10,
                   "The uniform-target contrast coordinates changed.")

  means <- vapply(context_names, function(nm) {
    sum(contexts$population_probability * as.numeric(contexts[[nm]]))
  }, numeric(1))
  nonlinear_raw <- nonlinear_context_features(contexts)
  nonlinear_means <- as.numeric(crossprod(
    contexts$population_probability, nonlinear_raw
  ))
  names(nonlinear_means) <- colnames(nonlinear_raw)

  list(
    target = target,
    uniform_contrast = contrast,
    coordinate_meta = as.data.frame(bundle$coordinate_meta),
    factor_names = factor_names,
    context_names = context_names,
    contexts = contexts,
    respondents = respondent,
    context_means = means,
    nonlinear_means = nonlinear_means,
    profile_count = nrow(target),
    assignment_probability = 1 / nrow(target),
    tasks = 8L
  )
}

nonlinear_context_features <- function(data) {
  value <- function(name) as.numeric(as.character(data[[name]]))
  woman <- value("woman")
  poc <- value("poc")
  young <- value("young")
  dem <- value("dem")
  rep <- value("rep")
  working <- value("workingclass")
  cbind(
    xor_woman_young = as.numeric(woman != young),
    poc_working = poc * working,
    dem_young = dem * young,
    nonrep_poc_young = (1 - rep) * poc * young
  )
}

factor_scores <- function(data, design, curvature = FALSE) {
  out <- matrix(NA_real_, nrow(data), length(design$factor_names),
                dimnames = list(NULL, design$factor_names))
  for (nm in design$factor_names) {
    locked_levels <- levels(design$target[[nm]])
    index <- match(as.character(data[[nm]]), locked_levels)
    benchmark_assert(!anyNA(index), paste("Unknown level for", nm))
    score <- seq(-1, 1, length.out = length(locked_levels))
    score <- score - mean(score)
    if (isTRUE(curvature)) {
      score <- score^2 - mean(score^2)
      if (sqrt(mean(score^2)) < 1e-12) score[] <- 0
    }
    scale <- sqrt(mean(score^2))
    if (scale > 0) score <- score / scale
    out[, nm] <- score[index]
  }
  out
}

candidate_latent_mean <- function(data, design, scenario, intercept = 0) {
  benchmark_assert(scenario %in% c(
    "additive_null", "linear_heterogeneity", "nonlinear_heterogeneity",
    "nonlinear_prognostic", "nonlinear_target_shift", "additive_target_shift"
  ), "Unknown response-surface scenario.")
  score <- factor_scores(data, design, curvature = FALSE)
  curve <- factor_scores(data, design, curvature = TRUE)
  main_coefficients <- c(
    fundraising = -0.035, winning = 0.055, pay = 0.035,
    media = 0.025, majority = -0.020, authority = 0.030, staff = 0.025
  )
  mu <- intercept + as.numeric(score %*% main_coefficients)
  task <- as.numeric(as.character(data$task))
  mu <- mu - 0.004 * (task - 4.5)

  w <- vapply(design$context_names, function(nm) {
    as.numeric(as.character(data[[nm]]))
  }, numeric(nrow(data)))
  centered <- sweep(w, 2L, design$context_means, "-")
  prognostic_coefficients <- c(
    woman = 0.070, poc = -0.055, young = 0.050,
    dem = 0.060, rep = -0.045, workingclass = 0.045
  )
  mu <- mu + as.numeric(centered %*% prognostic_coefficients)

  if (!scenario %in% c("additive_null", "additive_target_shift")) {
    mu <- mu +
      0.025 * score[, "authority"] * score[, "majority"] +
      0.020 * score[, "authority"] * score[, "staff"] +
      0.018 * score[, "fundraising"] * score[, "winning"] +
      0.015 * curve[, "fundraising"] * score[, "pay"]
  }

  if (identical(scenario, "linear_heterogeneity")) {
    mu <- mu +
      0.055 * score[, "fundraising"] * centered[, "woman"] +
      0.045 * score[, "winning"] * centered[, "poc"] -
      0.045 * score[, "authority"] * centered[, "dem"] +
      0.035 * score[, "staff"] * centered[, "workingclass"]
  }

  if (identical(scenario, "nonlinear_prognostic")) {
    nonlinear <- nonlinear_context_features(data)
    nonlinear <- sweep(nonlinear, 2L, design$nonlinear_means, "-")
    mu <- mu +
      0.120 * nonlinear[, "xor_woman_young"] -
      0.100 * nonlinear[, "poc_working"] +
      0.080 * nonlinear[, "dem_young"] +
      0.060 * nonlinear[, "nonrep_poc_young"]
  }
  if (scenario %in% c("nonlinear_heterogeneity", "nonlinear_target_shift")) {
    nonlinear <- nonlinear_context_features(data)
    nonlinear <- sweep(nonlinear, 2L, design$nonlinear_means, "-")
    mu <- mu +
      0.085 * score[, "fundraising"] * nonlinear[, "xor_woman_young"] +
      0.070 * curve[, "winning"] * nonlinear[, "poc_working"] +
      0.065 * score[, "authority"] * nonlinear[, "dem_young"] +
      0.050 * score[, "media"] * nonlinear[, "nonrep_poc_young"]
  }
  intercept + design$outcome$signal_scale * (mu - intercept)
}

calibrate_scenario_intercept <- function(design, scenario, config) {
  context_rows <- lapply(seq_len(nrow(design$contexts)), function(i) {
    newdata <- make_target_prediction_data(
      design$contexts[i, , drop = FALSE], design
    )
    latent_without_intercept <- candidate_latent_mean(
      newdata, design, scenario, intercept = 0
    )
    list(signal = latent_without_intercept,
         probability = design$contexts$population_probability[[i]] /
           length(latent_without_intercept))
  })
  signal <- unlist(lapply(context_rows, `[[`, "signal"), use.names = FALSE)
  probability <- unlist(lapply(context_rows, function(x) {
    rep(x$probability, length(x$signal))
  }), use.names = FALSE)
  weighted_signal_mean <- sum(probability * signal)
  config$outcome$target_mean - weighted_signal_mean
}

prepare_scenario_calibration <- function(design, config) {
  design$outcome <- config$outcome
  design$scenario_intercepts <- stats::setNames(
    vapply(config$scenarios, function(scenario) {
      calibrate_scenario_intercept(design, scenario, config)
    }, numeric(1)),
    config$scenarios
  )
  design$calibration <- do.call(rbind, lapply(config$scenarios, function(scenario) {
    values <- unlist(lapply(seq_len(nrow(design$contexts)), function(i) {
      newdata <- make_target_prediction_data(design$contexts[i, , drop = FALSE], design)
      candidate_latent_mean(
        newdata, design, scenario, design$scenario_intercepts[[scenario]]
      )
    }), use.names = FALSE)
    data.frame(
      scenario = scenario,
      minimum_conditional_mean = min(values),
      maximum_conditional_mean = max(values),
      stringsAsFactors = FALSE
    )
  }))
  benchmark_assert(
    all(design$calibration$minimum_conditional_mean > 0) &&
      all(design$calibration$maximum_conditional_mean < 1),
    "The response-scale surface leaves the open unit interval."
  )
  design$scenario_truth <- stats::setNames(lapply(config$scenarios, function(scenario) {
    contrast_info <- scenario_contrast(design, scenario)
    plugins <- true_context_plugins(design, scenario, contrast_info$contrast)
    truth <- as.numeric(crossprod(
      design$contexts$population_probability,
      plugins[design$contexts$.context_key, , drop = FALSE]
    ))
    names(truth) <- colnames(contrast_info$contrast)
    list(plugins = plugins, truth = truth)
  }), config$scenarios)
  design
}

candidate_conditional_mean <- function(data, design, scenario) {
  benchmark_assert(!is.null(design$scenario_intercepts[[scenario]]),
                   "The scenario intercept has not been calibrated.")
  candidate_latent_mean(
    data, design, scenario, design$scenario_intercepts[[scenario]]
  )
}

scenario_contrast <- function(design, scenario) {
  contrast <- design$uniform_contrast
  ratio <- rep(1, design$profile_count)
  if (grepl("target_shift$", scenario)) {
    media_index <- match(
      as.character(design$target$media), levels(design$target$media)
    )
    level_ratio <- c(0.5, 1.0, 1.5)
    benchmark_assert(length(level_ratio) == nlevels(design$target$media),
                     "The target-shift ratios do not match the media support.")
    ratio <- level_ratio[media_index]
    contrast <- contrast * ratio
  }
  list(
    contrast = contrast,
    target_assignment_ratio = ratio,
    max_target_assignment_ratio = max(ratio)
  )
}

as_benchmark_model_data <- function(data, design) {
  out <- as.data.frame(data)
  for (nm in design$factor_names) {
    out[[nm]] <- factor(as.character(out[[nm]]),
                        levels = levels(design$target[[nm]]), ordered = FALSE)
  }
  out$task <- factor(as.character(out$task), levels = as.character(seq_len(design$tasks)))
  for (nm in design$context_names) {
    out[[nm]] <- factor(as.character(out[[nm]]), levels = c("0", "1"))
  }
  out
}

make_balanced_folds <- function(cluster, nfolds, seed) {
  ids <- unique(as.character(cluster))
  set.seed(as.integer(seed))
  assignment <- rep(seq_len(nfolds), length.out = length(ids))
  assignment <- sample(assignment, length(assignment), replace = FALSE)
  assignment[match(as.character(cluster), ids)]
}

benchmark_dataset_seed <- function(scenario, replication, config) {
  group <- switch(
    scenario,
    additive_null = 1L,
    additive_target_shift = 1L,
    linear_heterogeneity = 2L,
    nonlinear_prognostic = 3L,
    nonlinear_heterogeneity = 4L,
    nonlinear_target_shift = 4L,
    stop("Unknown benchmark seed group.", call. = FALSE)
  )
  as.integer(config$seed + 100000L * group + replication)
}

simulate_candidate_geometry <- function(design, scenario, replication, config) {
  seed <- benchmark_dataset_seed(scenario, replication, config)
  set.seed(seed)
  g <- config$respondents
  tasks <- config$tasks_per_respondent
  sampled_respondent <- sample.int(nrow(design$respondents), g, replace = TRUE)
  respondent <- design$respondents[
    sampled_respondent, c("weight", design$context_names), drop = FALSE
  ]
  respondent$id <- seq_len(g)
  respondent$fold <- make_balanced_folds(
    respondent$id, config$outer_folds, seed + 11L
  )
  dat <- respondent[rep(seq_len(g), each = tasks), , drop = FALSE]
  dat$task <- rep(seq_len(tasks), times = g)
  profile_draws <- replicate(
    g, sample.int(design$profile_count, tasks, replace = FALSE)
  )
  dat$profile_index <- as.vector(profile_draws)
  for (nm in design$factor_names) {
    dat[[nm]] <- design$target[[nm]][dat$profile_index]
  }
  dat$conditional_mean <- candidate_conditional_mean(dat, design, scenario)
  shared_normal <- stats::rnorm(g)
  rho <- config$outcome$respondent_copula_rho
  task_normal <- sqrt(rho) * rep(shared_normal, each = tasks) +
    sqrt(1 - rho) * stats::rnorm(nrow(dat))
  beta_probability <- stats::qbeta(
    stats::pnorm(task_normal),
    shape1 = dat$conditional_mean * config$outcome$beta_precision,
    shape2 = (1 - dat$conditional_mean) * config$outcome$beta_precision
  )
  dat$y <- stats::rbinom(
    nrow(dat), size = config$outcome$rating_intervals,
    prob = beta_probability
  ) / config$outcome$rating_intervals
  dat$.context_key <- context_key(dat, design$context_names)
  dat <- as_benchmark_model_data(dat, design)
  list(data = dat, seed = seed)
}

benchmark_formulas <- function(design) {
  a_sum <- paste(design$factor_names, collapse = " + ")
  w_sum <- paste(design$context_names, collapse = " + ")
  list(
    additive_lm = stats::as.formula(paste0(
      "y ~ ", a_sum, " + task + ", w_sum
    )),
    interacted_lm = stats::as.formula(paste0(
      "y ~ (", a_sum, ")^2 + task + ", w_sum,
      " + (", a_sum, "):(", w_sum, ")"
    )),
    elastic_net = stats::as.formula(paste0(
      "~ 0 + task + (", a_sum, ")^2 + ", w_sum,
      " + (", a_sum, "):(", w_sum, ")"
    ))
  )
}

one_hot_factor <- function(x, prefix) {
  factor_x <- factor(x, levels = levels(x), ordered = FALSE)
  index <- as.integer(factor_x)
  benchmark_assert(!anyNA(index), paste("Missing factor level in", prefix))
  out <- matrix(0, nrow = length(index), ncol = nlevels(factor_x))
  out[cbind(seq_along(index), index)] <- 1
  colnames(out) <- paste0(prefix, "=", levels(factor_x))
  out
}

dummy_interaction <- function(left, right, separator = ":") {
  out <- left[, rep(seq_len(ncol(left)), each = ncol(right)), drop = FALSE] *
    right[, rep(seq_len(ncol(right)), times = ncol(left)), drop = FALSE]
  colnames(out) <- paste(
    rep(colnames(left), each = ncol(right)),
    rep(colnames(right), times = ncol(left)),
    sep = separator
  )
  out
}

invariant_elastic_matrix <- function(data, design) {
  variables <- c(design$factor_names, "task", design$context_names)
  dummies <- stats::setNames(lapply(variables, function(name) {
    one_hot_factor(data[[name]], name)
  }), variables)
  blocks <- c(
    list(dummies$task),
    dummies[design$factor_names],
    dummies[design$context_names]
  )
  factor_pairs <- utils::combn(design$factor_names, 2L, simplify = FALSE)
  blocks <- c(blocks, lapply(factor_pairs, function(pair) {
    dummy_interaction(dummies[[pair[[1L]]]], dummies[[pair[[2L]]]])
  }))
  for (attribute in design$factor_names) {
    for (context in design$context_names) {
      blocks[[length(blocks) + 1L]] <- dummy_interaction(
        dummies[[attribute]], dummies[[context]]
      )
    }
  }
  out <- do.call(cbind, blocks)
  benchmark_assert(!anyDuplicated(colnames(out)),
                   "The invariant elastic-net matrix has duplicate columns.")
  out
}

fit_benchmark_learner <- function(kind, train, design, config, seed) {
  formulas <- benchmark_formulas(design)
  if (kind %in% c("additive_lm", "interacted_lm")) {
    fit <- stats::lm(
      formulas[[kind]], data = train, weights = weight,
      model = FALSE, x = FALSE, y = FALSE, qr = TRUE
    )
    predict_fun <- function(newdata) {
      suppressWarnings(as.numeric(stats::predict(fit, newdata = newdata)))
    }
    return(list(predict = predict_fun, detail = paste0("lm rank=", fit$rank)))
  }
  if (identical(kind, "elastic_net")) {
    if (!requireNamespace("glmnet", quietly = TRUE)) {
      stop("Package `glmnet` is required.", call. = FALSE)
    }
    design_matrix <- invariant_elastic_matrix(train, design)
    columns <- colnames(design_matrix)
    inner_fold <- make_balanced_folds(
      train$id, config$inner_folds, as.integer(seed + 1701L)
    )
    fit <- glmnet::cv.glmnet(
      x = design_matrix,
      y = train$y,
      weights = train$weight,
      foldid = inner_fold,
      family = "gaussian",
      alpha = as.numeric(config$elastic_net$alpha),
      nlambda = as.integer(config$elastic_net$nlambda),
      standardize = TRUE,
      type.measure = "mse",
      parallel = FALSE
    )
    predict_fun <- function(newdata) {
      new_matrix <- invariant_elastic_matrix(newdata, design)
      benchmark_assert(identical(colnames(new_matrix), columns),
                       "Elastic-net model-matrix columns changed at prediction.")
      as.numeric(stats::predict(fit, newx = new_matrix, s = "lambda.min"))
    }
    return(list(
      predict = predict_fun,
      detail = sprintf("elastic net alpha=%.2f lambda=%.6g",
                       config$elastic_net$alpha, fit$lambda.min)
    ))
  }
  if (identical(kind, "ranger")) {
    if (!requireNamespace("ranger", quietly = TRUE)) {
      stop("Package `ranger` is required.", call. = FALSE)
    }
    predictors <- c(design$factor_names, "task", design$context_names)
    fit <- ranger::ranger(
      dependent.variable.name = "y",
      data = train[c("y", predictors)],
      case.weights = train$weight,
      num.trees = as.integer(config$forest$trees),
      min.node.size = as.integer(config$forest$min_node_size),
      sample.fraction = as.numeric(config$forest$sample_fraction),
      respect.unordered.factors = "partition",
      seed = as.integer(seed),
      num.threads = 1L,
      write.forest = TRUE
    )
    predict_fun <- function(newdata) {
      as.numeric(predict(fit, data = newdata[predictors],
                         num.threads = 1L)$predictions)
    }
    return(list(
      predict = predict_fun,
      detail = sprintf("ranger trees=%d min.node.size=%d",
                       config$forest$trees, config$forest$min_node_size)
    ))
  }
  stop("Unknown learner: ", kind, call. = FALSE)
}

make_target_prediction_data <- function(context, design) {
  blocks <- lapply(seq_len(design$tasks), function(tt) {
    out <- design$target[design$factor_names]
    out$task <- tt
    for (nm in design$context_names) out[[nm]] <- context[[nm]][[1L]]
    out
  })
  as_benchmark_model_data(do.call(rbind, blocks), design)
}

target_plugins <- function(predict_fun, contexts, design, contrast) {
  contexts <- contexts[!duplicated(context_key(contexts, design$context_names)), , drop = FALSE]
  keys <- context_key(contexts, design$context_names)
  out <- matrix(NA_real_, nrow(contexts), ncol(contrast),
                dimnames = list(keys, colnames(contrast)))
  for (i in seq_len(nrow(contexts))) {
    newdata <- make_target_prediction_data(contexts[i, , drop = FALSE], design)
    prediction <- predict_fun(newdata)
    benchmark_assert(length(prediction) == design$profile_count * design$tasks,
                     "A learner returned the wrong number of target predictions.")
    benchmark_assert(all(is.finite(prediction)),
                     "A learner returned non-finite target predictions.")
    profile_prediction <- rowMeans(matrix(
      prediction, nrow = design$profile_count, ncol = design$tasks
    ))
    out[i, ] <- as.numeric(crossprod(contrast, profile_prediction))
  }
  out
}

true_context_plugins <- function(design, scenario, contrast) {
  contexts <- design$contexts
  out <- matrix(NA_real_, nrow(contexts), ncol(contrast),
                dimnames = list(contexts$.context_key, colnames(contrast)))
  for (i in seq_len(nrow(contexts))) {
    newdata <- make_target_prediction_data(contexts[i, , drop = FALSE], design)
    prediction <- candidate_conditional_mean(newdata, design, scenario)
    profile_prediction <- rowMeans(matrix(
      prediction, nrow = design$profile_count, ncol = design$tasks
    ))
    out[i, ] <- as.numeric(crossprod(contrast, profile_prediction))
  }
  out
}

aggregate_cluster_scores <- function(scores, weights = NULL) {
  benchmark_assert(!anyNA(scores) && all(is.finite(scores)),
                   "Cluster scores must be finite and complete.")
  if (is.null(weights)) weights <- rep(1, nrow(scores))
  benchmark_assert(length(weights) == nrow(scores) && all(is.finite(weights)) &&
                     all(weights > 0),
                   "Cluster weights must be finite, positive, and complete.")
  estimate <- as.numeric(crossprod(weights / sum(weights), scores))
  centered <- sweep(scores, 2L, estimate, "-")
  covariance <- (nrow(scores) / (nrow(scores) - 1L)) *
    crossprod(centered * weights) / sum(weights)^2
  list(
    estimate = estimate,
    covariance = covariance,
    std_error = sqrt(pmax(diag(covariance), 0))
  )
}

cluster_scores <- function(data, plugin, correction, design) {
  ids <- sort(unique(as.integer(as.character(data$id))))
  out <- matrix(NA_real_, length(ids), ncol(correction),
                dimnames = list(as.character(ids), colnames(correction)))
  for (i in seq_along(ids)) {
    rows <- which(as.integer(as.character(data$id)) == ids[[i]])
    key <- unique(as.character(data$.context_key[rows]))
    benchmark_assert(length(key) == 1L && key %in% rownames(plugin),
                     "Could not match a respondent to a target plugin.")
    out[i, ] <- plugin[key, ] + colMeans(correction[rows, , drop = FALSE])
  }
  out
}

wald_family_rows <- function(fit, truth, coordinate_meta, scenario,
                             replication, model) {
  families <- unique(coordinate_meta$family)
  do.call(rbind, lapply(families, function(family) {
    index <- which(coordinate_meta$family == family)
    covariance <- fit$covariance[index, index, drop = FALSE]
    eigen_decomposition <- eigen(covariance, symmetric = TRUE)
    tolerance <- max(eigen_decomposition$values, 0) * 1e-10
    keep <- eigen_decomposition$values > max(tolerance, .Machine$double.eps)
    rank <- sum(keep)
    if (rank) {
      inverse <- eigen_decomposition$vectors[, keep, drop = FALSE] %*%
        diag(1 / eigen_decomposition$values[keep], rank) %*%
        t(eigen_decomposition$vectors[, keep, drop = FALSE])
      statistic <- as.numeric(crossprod(
        fit$estimate[index], inverse %*% fit$estimate[index]
      ))
      p_value <- stats::pchisq(statistic, df = rank, lower.tail = FALSE)
      estimation_error <- fit$estimate[index] - truth[index]
      truth_statistic <- as.numeric(crossprod(
        estimation_error, inverse %*% estimation_error
      ))
      truth_covered_95 <- truth_statistic <= stats::qchisq(0.95, df = rank)
    } else {
      statistic <- NA_real_
      p_value <- NA_real_
      truth_statistic <- NA_real_
      truth_covered_95 <- FALSE
    }
    data.frame(
      scenario = scenario,
      replication = replication,
      model = model,
      family = family,
      dimension = length(index),
      covariance_rank = rank,
      null_family = sum(truth[index]^2) < 1e-12,
      statistic = statistic,
      p_value = p_value,
      reject_05 = is.finite(p_value) && p_value < 0.05,
      truth_statistic = truth_statistic,
      truth_covered_95 = truth_covered_95,
      stringsAsFactors = FALSE
    )
  }))
}

run_benchmark_replication <- function(design, scenario, replication, config) {
  started <- proc.time()[["elapsed"]]
  simulated <- simulate_candidate_geometry(design, scenario, replication, config)
  dat <- simulated$data
  contrast_info <- scenario_contrast(design, scenario)
  contrast <- contrast_info$contrast
  q <- design$assignment_probability
  cached_truth <- design$scenario_truth[[scenario]]
  if (is.null(cached_truth)) {
    truth_plugins <- true_context_plugins(design, scenario, contrast)
    truth <- as.numeric(crossprod(
      design$contexts$population_probability,
      truth_plugins[design$contexts$.context_key, , drop = FALSE]
    ))
    names(truth) <- colnames(contrast)
  } else {
    truth_plugins <- cached_truth$plugins
    truth <- cached_truth$truth
  }
  observed_riesz <- contrast[dat$profile_index, , drop = FALSE] / q
  colnames(observed_riesz) <- colnames(contrast)
  ids <- sort(unique(as.integer(as.character(dat$id))))

  score_list <- list()
  prediction_list <- list()
  detail_list <- list()

  if ("unadjusted" %in% config$models) {
    unadjusted_scores <- matrix(NA_real_, length(ids), ncol(contrast),
                                dimnames = list(as.character(ids), colnames(contrast)))
    unadjusted_rows <- observed_riesz * dat$y
    for (i in seq_along(ids)) {
      rows <- which(as.integer(as.character(dat$id)) == ids[[i]])
      unadjusted_scores[i, ] <- colMeans(unadjusted_rows[rows, , drop = FALSE])
    }
    score_list$unadjusted <- unadjusted_scores
    prediction_list$unadjusted <- rep(0, nrow(dat))
    detail_list$unadjusted <- "known-randomization score"
  }

  learners <- intersect(
    c("additive_lm", "interacted_lm", "elastic_net", "ranger"),
    config$models
  )
  for (kind in learners) {
    scores <- matrix(NA_real_, length(ids), ncol(contrast),
                     dimnames = list(as.character(ids), colnames(contrast)))
    plugin_only <- if (identical(kind, "ranger")) scores else NULL
    prediction <- rep(NA_real_, nrow(dat))
    details <- character()
    for (fold in seq_len(config$outer_folds)) {
      train <- as.integer(as.character(dat$fold)) != fold
      valid <- !train
      learner <- fit_benchmark_learner(
        kind, dat[train, , drop = FALSE], design, config,
        simulated$seed + 100L * fold + match(kind, learners)
      )
      prediction[valid] <- learner$predict(dat[valid, , drop = FALSE])
      benchmark_assert(all(is.finite(prediction[valid])),
                       paste(kind, "returned non-finite held-out predictions."))
      contexts <- dat[valid, design$context_names, drop = FALSE]
      plugin <- target_plugins(learner$predict, contexts, design, contrast)
      correction <- observed_riesz[valid, , drop = FALSE] *
        (dat$y[valid] - prediction[valid])
      fold_scores <- cluster_scores(dat[valid, , drop = FALSE], plugin,
                                    correction, design)
      scores[match(rownames(fold_scores), rownames(scores)), ] <- fold_scores
      if (identical(kind, "ranger")) {
        zero_correction <- matrix(0, nrow(correction), ncol(correction),
                                  dimnames = dimnames(correction))
        fold_plugin_scores <- cluster_scores(
          dat[valid, , drop = FALSE], plugin, zero_correction, design
        )
        plugin_only[match(rownames(fold_plugin_scores), rownames(plugin_only)), ] <-
          fold_plugin_scores
      }
      details <- c(details, paste0("fold ", fold, ": ", learner$detail))
    }
    score_list[[kind]] <- scores
    prediction_list[[kind]] <- prediction
    detail_list[[kind]] <- paste(details, collapse = "; ")
    if (identical(kind, "ranger")) {
      if ("plugin_ranger" %in% config$models) {
        score_list$plugin_ranger <- plugin_only
        prediction_list$plugin_ranger <- prediction
        detail_list$plugin_ranger <- "cross-fitted ranger plugin without correction"
      }
    }
  }

  if ("oracle" %in% config$models) {
    oracle_plugin <- truth_plugins
    oracle_correction <- observed_riesz * (dat$y - dat$conditional_mean)
    score_list$oracle <- cluster_scores(dat, oracle_plugin, oracle_correction, design)
    prediction_list$oracle <- dat$conditional_mean
    detail_list$oracle <- "known conditional mean"
  }

  model_order <- config$models
  cluster_weight <- vapply(ids, function(id) {
    rows <- which(as.integer(as.character(dat$id)) == id)
    value <- unique(dat$weight[rows])
    benchmark_assert(length(value) == 1L, "Survey weight varies within respondent.")
    value[[1L]]
  }, numeric(1))
  fits <- lapply(
    score_list[model_order], aggregate_cluster_scores, weights = cluster_weight
  )
  estimate_rows <- do.call(rbind, lapply(model_order, function(model) {
    fit <- fits[[model]]
    data.frame(
      scenario = scenario,
      replication = replication,
      seed = simulated$seed,
      model = model,
      estimand = colnames(contrast),
      family = design$coordinate_meta$family[
        match(colnames(contrast), design$coordinate_meta$coordinate)
      ],
      truth = truth,
      estimate = fit$estimate,
      error = fit$estimate - truth,
      std_error = fit$std_error,
      covered_95 = abs(fit$estimate - truth) <= 1.96 * fit$std_error,
      prediction_mse = stats::weighted.mean(
        (dat$y - prediction_list[[model]])^2, dat$weight
      ),
      max_target_assignment_ratio = contrast_info$max_target_assignment_ratio,
      stringsAsFactors = FALSE
    )
  }))
  family_rows <- do.call(rbind, lapply(model_order, function(model) {
    wald_family_rows(
      fits[[model]], truth, design$coordinate_meta, scenario, replication, model
    )
  }))
  list(
    estimates = estimate_rows,
    families = family_rows,
    details = detail_list,
    runtime_seconds = proc.time()[["elapsed"]] - started,
    truth = truth,
    scenario = scenario,
    replication = replication,
    seed = simulated$seed
  )
}
