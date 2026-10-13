# Analysis-only helpers for Gate D.  Promotion into the package is conditional
# on the frozen gate passing.

feasible_assert <- function(ok, message) {
  if (!isTRUE(ok)) stop(message, call. = FALSE)
}

feasible_symmetric <- function(x) (x + t(x)) / 2

stabilize_working_covariance <- function(covariance, relative_floor = 1e-8) {
  covariance <- feasible_symmetric(as.matrix(covariance))
  eig <- eigen(covariance, symmetric = TRUE)
  scale <- max(abs(eig$values))
  feasible_assert(is.finite(scale) && scale > 0,
                  "The working residual moment is numerically zero.")
  floor_value <- relative_floor * scale
  adjusted_values <- pmax(eig$values, floor_value)
  adjusted <- eig$vectors %*% (adjusted_values * t(eig$vectors))
  list(
    covariance = feasible_symmetric(adjusted),
    raw_eigenvalues = eig$values,
    adjusted_eigenvalues = adjusted_values,
    floor = floor_value,
    projected = any(eig$values < floor_value)
  )
}

estimate_working_covariances <- function(
    residual, cluster_weight = NULL, relative_floor = 1e-8,
    centered = FALSE) {
  residual <- as.matrix(residual)
  feasible_assert(nrow(residual) >= 2L && ncol(residual) >= 1L &&
                    all(is.finite(residual)),
                  "Residuals must be a finite respondent-by-task matrix.")
  if (is.null(cluster_weight)) cluster_weight <- rep(1, nrow(residual))
  cluster_weight <- as.numeric(cluster_weight)
  feasible_assert(length(cluster_weight) == nrow(residual) &&
                    all(is.finite(cluster_weight)) && all(cluster_weight > 0),
                  "Cluster weights must be finite and positive.")
  variance_weight <- cluster_weight^2
  variance_weight <- variance_weight / sum(variance_weight)
  if (centered) {
    center <- drop(crossprod(variance_weight, residual))
    residual <- sweep(residual, 2L, center, "-")
  } else {
    center <- rep(0, ncol(residual))
  }
  weighted_residual <- residual * sqrt(variance_weight)
  raw <- feasible_symmetric(crossprod(weighted_residual))
  unstructured <- stabilize_working_covariance(raw, relative_floor)
  task_count <- ncol(residual)
  cs_raw <- matrix(
    if (task_count == 1L) 0 else mean(raw[row(raw) != col(raw)]),
    task_count, task_count
  )
  diag(cs_raw) <- mean(diag(raw))
  compound_symmetry <- stabilize_working_covariance(cs_raw, relative_floor)
  list(
    unstructured = unstructured,
    compound_symmetry = compound_symmetry,
    raw_second_moment = raw,
    center = center,
    variance_weight = variance_weight
  )
}

ordinary_task_operator <- function(task_count, task_weights,
                                   assignment_correlation, mechanism,
                                   support_size = NULL) {
  mixing <- diag(as.numeric(task_weights), nrow = task_count)
  calibration <- diag(mixing %*% assignment_correlation)
  structure(list(
    mixing = mixing,
    task_weights = as.numeric(task_weights),
    assignment_correlation = assignment_correlation,
    mechanism = mechanism,
    support_size = support_size,
    calibration = calibration,
    calibration_residual = max(abs(calibration - task_weights)),
    covariance_class = "ordinary",
    blend = 0,
    candidate = "ordinary"
  ), class = c("feasible_task_operator", "factorial_task_mixing"))
}

blend_task_operator <- function(ordinary, adaptive, blend,
                                covariance_class = NULL) {
  blend <- as.numeric(blend)
  feasible_assert(length(blend) == 1L && is.finite(blend) &&
                    blend >= 0 && blend <= 1,
                  "The calibration-preserving blend must lie in [0, 1].")
  feasible_assert(
    identical(dim(ordinary$mixing), dim(adaptive$mixing)) &&
      max(abs(ordinary$assignment_correlation -
                adaptive$assignment_correlation)) < 1e-12,
    "Only operators for the same task assignment can be blended."
  )
  mixing <- (1 - blend) * ordinary$mixing + blend * adaptive$mixing
  calibration <- diag(mixing %*% ordinary$assignment_correlation)
  candidate <- if (blend == 0) "ordinary" else sprintf(
    "%s_tau_%s", covariance_class,
    sub("\\.?0+$", "", formatC(blend, format = "f", digits = 2))
  )
  structure(list(
    mixing = mixing,
    task_weights = ordinary$task_weights,
    assignment_correlation = ordinary$assignment_correlation,
    mechanism = ordinary$mechanism,
    support_size = ordinary$support_size,
    calibration = calibration,
    calibration_residual = max(abs(calibration - ordinary$task_weights)),
    covariance_class = covariance_class,
    blend = blend,
    candidate = candidate
  ), class = c("feasible_task_operator", "factorial_task_mixing"))
}

make_operator_library <- function(
    covariance_set, task_weights,
    mechanism = c("product", "uniform_without_replacement"),
    support_size = NULL, blend_grid = c(0.25, 0.5, 0.75, 1),
    tolerance = 1e-10) {
  mechanism <- match.arg(mechanism)
  task_count <- length(task_weights)
  reference_covariance <- covariance_set$compound_symmetry$covariance
  reference <- solve_task_mixing(
    reference_covariance, task_weights, mechanism, support_size, tolerance
  )
  ordinary <- ordinary_task_operator(
    task_count, task_weights, reference$assignment_correlation,
    mechanism, support_size
  )
  out <- list(ordinary = ordinary)
  metadata <- data.frame(
    candidate = "ordinary", covariance_class = "ordinary", blend = 0,
    complexity = 0L, stringsAsFactors = FALSE
  )
  complexity <- 0L
  for (blend in sort(unique(as.numeric(blend_grid)))) {
    for (covariance_class in c("compound_symmetry", "unstructured")) {
      adaptive <- solve_task_mixing(
        covariance_set[[covariance_class]]$covariance,
        task_weights, mechanism, support_size, tolerance
      )
      operator <- blend_task_operator(
        ordinary, adaptive, blend, covariance_class
      )
      complexity <- complexity + 1L
      out[[operator$candidate]] <- operator
      metadata <- rbind(metadata, data.frame(
        candidate = operator$candidate,
        covariance_class = covariance_class,
        blend = blend,
        complexity = complexity,
        stringsAsFactors = FALSE
      ))
    }
  }
  attr(out, "metadata") <- metadata
  out
}

score_factorial_clusters <- function(
    residual, profile_index, riesz_lookup, operator, plugin = NULL) {
  residual <- as.matrix(residual)
  profile_index <- as.matrix(profile_index)
  riesz_lookup <- as.matrix(riesz_lookup)
  feasible_assert(identical(dim(residual), dim(profile_index)),
                  "Residual and profile-index matrices must align.")
  feasible_assert(ncol(residual) == nrow(operator$mixing),
                  "The score operator has the wrong task dimension.")
  feasible_assert(all(is.finite(residual)) &&
                    all(profile_index >= 1L & profile_index <= nrow(riesz_lookup)),
                  "Scores require finite residuals and supported profiles.")
  transformed_residual <- residual %*% operator$mixing
  score <- matrix(0, nrow(residual), ncol(riesz_lookup))
  for (j in seq_len(ncol(riesz_lookup))) {
    assigned_riesz <- matrix(
      riesz_lookup[as.vector(profile_index), j], nrow(residual), ncol(residual)
    )
    score[, j] <- rowSums(transformed_residual * assigned_riesz)
  }
  if (!is.null(plugin)) {
    if (is.null(dim(plugin))) plugin <- matrix(plugin, nrow = 1L)
    plugin <- as.matrix(plugin)
    if (nrow(plugin) == 1L && nrow(score) > 1L) {
      plugin <- plugin[rep(1L, nrow(score)), , drop = FALSE]
    }
    feasible_assert(identical(dim(plugin), dim(score)) && all(is.finite(plugin)),
                    "The plug-in matrix must align with respondent scores.")
    score <- score + plugin
  }
  colnames(score) <- colnames(riesz_lookup)
  score
}

weighted_score_summary <- function(score, cluster_weight = NULL) {
  score <- as.matrix(score)
  if (is.null(cluster_weight)) cluster_weight <- rep(1, nrow(score))
  cluster_weight <- as.numeric(cluster_weight)
  feasible_assert(length(cluster_weight) == nrow(score) &&
                    all(is.finite(cluster_weight)) && all(cluster_weight > 0),
                  "Cluster weights must align with scores.")
  normalized <- cluster_weight / sum(cluster_weight)
  estimate <- drop(crossprod(normalized, score))
  centered <- sweep(score, 2L, estimate, "-")
  covariance <- (nrow(score) / (nrow(score) - 1L)) *
    crossprod(centered * cluster_weight) / sum(cluster_weight)^2
  influence <- centered * (cluster_weight / mean(cluster_weight))
  loss_by_cluster <- rowMeans(influence^2)
  list(
    estimate = estimate,
    covariance = feasible_symmetric(covariance),
    standard_error = sqrt(pmax(diag(covariance), 0)),
    influence = influence,
    loss_by_cluster = loss_by_cluster,
    loss = mean(loss_by_cluster),
    loss_standard_error = stats::sd(loss_by_cluster) / sqrt(nrow(score))
  )
}

select_operator_one_se <- function(score_list, metadata, cluster_weight = NULL) {
  feasible_assert(is.list(score_list) && length(score_list) >= 1L,
                  "Candidate score matrices are required.")
  feasible_assert(all(metadata$candidate %in% names(score_list)),
                  "Candidate metadata and score matrices do not align.")
  summaries <- lapply(metadata$candidate, function(candidate) {
    weighted_score_summary(score_list[[candidate]], cluster_weight)
  })
  loss <- vapply(summaries, `[[`, numeric(1), "loss")
  loss_se <- vapply(summaries, `[[`, numeric(1), "loss_standard_error")
  minimum <- which.min(loss)
  threshold <- loss[[minimum]] + loss_se[[minimum]]
  eligible <- which(loss <= threshold)
  chosen_row <- eligible[which.min(metadata$complexity[eligible])]
  table <- metadata
  table$loss <- loss
  table$loss_standard_error <- loss_se
  table$one_se_threshold <- threshold
  table$eligible <- seq_len(nrow(table)) %in% eligible
  table$selected <- seq_len(nrow(table)) == chosen_row
  list(
    candidate = metadata$candidate[[chosen_row]],
    table = table,
    minimum_candidate = metadata$candidate[[minimum]],
    threshold = threshold
  )
}
