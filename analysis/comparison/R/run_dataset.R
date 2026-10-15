# One simulated dataset: simulate, cross-fit every requested method on the
# common respondent folds and target draws, score, and summarize.
#
# A job is one method x outer fold. A failed job never borrows another method's
# predictions: the method's results for the dataset become schema-complete NA
# rows and the failure is recorded (comparison-v1.yml `failures`).

cmp_engine <- function(root) {
  contracts <- cmp_load_contracts(root)
  geoms <- list(rating = cmp_rating_geometry(contracts), choice = cmp_choice_geometry(contracts))
  geoms$rating$ridge_basis <- cmp_ridge_basis(geoms$rating,
                                              contracts$v1$learners$definitions$unrestricted_order_ridge$focal_basis_max_order)
  geoms$choice$ridge_basis <- cmp_ridge_basis(geoms$choice,
                                              contracts$v1$learners$definitions$unrestricted_order_ridge$focal_basis_max_order)
  geoms$rating$grid_rows <- cmp_rating_grid_rows(geoms$rating)
  list(root = root, contracts = contracts, geoms = geoms, surfaces = new.env())
}

cmp_cell <- function(engine, cell_id) {
  cells <- engine$contracts$simulation$cells
  cell <- cells[[which(vapply(cells, `[[`, 0L, "id") == cell_id)]]
  cell$geometry_obj <- engine$geoms[[cell$geometry]]
  cell
}

cmp_surface <- function(engine, cell, multiplier = NULL) {
  key <- paste(cell$id, multiplier %||% "fixed")
  if (!is.null(engine$surfaces[[key]])) return(engine$surfaces[[key]])
  geom <- cell$geometry_obj
  s <- if (identical(geom$name, "rating")) {
    cmp_rating_surface(geom, cell, multiplier)
  } else {
    cmp_choice_surface(geom, cell, multiplier)
  }
  assign(key, s, envir = engine$surfaces)
  s
}

cmp_peak_mb <- function() sum(gc(verbose = FALSE)[, 6L])

cmp_json <- function(x) {
  if (is.null(x) || !length(x)) return("{}")
  as.character(jsonlite::toJSON(x, auto_unbox = TRUE, digits = NA, null = "null"))
}

cmp_tuning_long <- function(tuning, method, fold) {
  if (is.null(tuning) || !nrow(tuning)) return(NULL)
  loss_cols <- grep("^inner_(mse|rounds)_", names(tuning), value = TRUE)
  param_cols <- setdiff(names(tuning), c("candidate", "mean_inner_mse", "selected", "error",
                                         loss_cols))
  data.frame(
    method = method, outer_fold = fold, candidate = tuning$candidate,
    parameters = vapply(seq_len(nrow(tuning)), function(i) {
      cmp_json(as.list(tuning[i, param_cols, drop = FALSE]))
    }, character(1)),
    mean_inner_mse = tuning$mean_inner_mse %||% NA_real_,
    inner_losses = vapply(seq_len(nrow(tuning)), function(i) {
      cmp_json(as.list(tuning[i, loss_cols, drop = FALSE]))
    }, character(1)),
    selected = if (is.null(tuning$selected)) NA else tuning$selected,
    error = if (is.null(tuning$error)) "" else as.character(tuning$error),
    stringsAsFactors = FALSE
  )
}

# job: list(phase, cell_id, replication, multiplier (or NULL), methods, draws)
cmp_run_dataset <- function(engine, job) {
  started <- proc.time()[["elapsed"]]
  sim <- engine$contracts$simulation
  cell <- cmp_cell(engine, job$cell_id)
  geom <- cell$geometry_obj
  surface <- cmp_surface(engine, cell, job$multiplier)
  cmp_assert(isTRUE(surface$valid),
             "Cell %s multiplier %s leaves the open unit interval.", cell$name,
             format(job$multiplier %||% NA))
  seeds <- vapply(c("data", "folds", "target_integration", "learner", "bootstrap"), function(p) {
    cmp_seed(sim, job$phase, cell$id, job$replication, p)
  }, integer(1))

  sim_data <- if (identical(geom$name, "rating")) {
    cmp_rating_simulate(geom, surface, seeds[["data"]])
  } else {
    cmp_choice_simulate(geom, surface, seeds[["data"]])
  }
  rows <- sim_data$rows
  resp <- sim_data$respondents
  ids <- resp$id
  folds <- cmp_respondent_folds(ids, sim$common$outer_folds, sim$common$inner_folds,
                                seeds[["folds"]])
  rows$outer_fold <- folds$outer_fold[match(rows$id, folds$id)]
  rows$inner_fold <- folds$inner_fold[match(rows$id, folds$id)]
  riesz <- cmp_riesz_rows(geom, rows)
  fmap <- cmp_feature_map(geom)
  plan <- if (identical(geom$name, "choice")) {
    cmp_choice_plan(geom, ids, job$draws, seeds[["target_integration"]])
  } else NULL
  mu <- surface$target_ratio / nrow(geom$grid)
  k_coord <- nrow(geom$coordinates)

  ledger <- list()
  predictions <- list()
  scores <- list()
  tuning <- list()
  failures <- list()
  for (method in job$methods) {
    phi <- matrix(NA_real_, length(ids), k_coord, dimnames = list(ids, geom$coordinates$key))
    pred_all <- rep(NA_real_, nrow(rows))
    raw_all <- rep(NA_real_, nrow(rows))
    method_failed <- FALSE
    for (k in seq_len(sim$common$outer_folds)) {
      t0 <- proc.time()[["elapsed"]]
      invisible(gc(reset = TRUE, verbose = FALSE))
      train <- rows[rows$outer_fold != k, , drop = FALSE]
      held <- rows$outer_fold == k
      held_ids <- folds$id[folds$outer_fold == k]
      ctx <- cmp_learner_context(geom, fmap, train, rep(1, nrow(train)), train$inner_fold,
                                 cmp_hashed_seed(seeds[["learner"]], method, k),
                                 engine$contracts, basis = geom$ridge_basis, surface = surface)
      res <- cmp_capture({
        fit <- cmp_fit_method(method, ctx)
        raw <- fit$predict(rows[held, , drop = FALSE])
        cmp_assert(length(raw) == sum(held) && all(is.finite(raw)),
                   "Non-finite or misaligned held-out predictions.")
        integ <- if (identical(geom$name, "rating")) {
          hr <- resp[match(held_ids, resp$id), , drop = FALSE]
          ctx_u <- hr[!duplicated(hr$context_key), c("context_key", geom$context_names)]
          r <- cmp_rating_plugins(geom, mu, fit$predict, ctx_u, geom$grid_rows)
          r$plugins <- r$plugins[match(hr$context_key, ctx_u$context_key), , drop = FALSE]
          rownames(r$plugins) <- held_ids
          r
        } else {
          cmp_choice_plugins(geom, plan, fit$predict, resp, held_ids)
        }
        list(fit = fit, raw = raw, integ = integ)
      })
      status <- if (res$failed) "failed" else "ok"
      n_proj <- NA_integer_
      if (!res$failed) {
        v <- res$value
        pred <- cmp_project_unit(v$raw)
        raw_all[held] <- v$raw
        pred_all[held] <- pred
        n_proj <- sum(v$raw < 0 | v$raw > 1)
        corr <- riesz[held, , drop = FALSE] * (rows$y[held] - pred)
        phi[held_ids, ] <- v$integ$plugins[held_ids, , drop = FALSE] +
          cmp_respondent_mean(corr, rows$id[held], held_ids)
        tuning[[length(tuning) + 1L]] <- cmp_tuning_long(v$fit$tuning, method, k)
      } else {
        method_failed <- TRUE
        failures[[length(failures) + 1L]] <- data.frame(
          method = method, outer_fold = k, stage = "fit_predict_or_integrate",
          message = as.character(res$value), stringsAsFactors = FALSE)
      }
      ledger[[length(ledger) + 1L]] <- data.frame(
        method = method, outer_fold = k, status = status,
        elapsed_seconds = cmp_elapsed(t0), peak_memory_mb = cmp_peak_mb(),
        n_train_rows = nrow(train), n_heldout_rows = sum(held),
        n_heldout_respondents = length(held_ids),
        selected = if (res$failed) "{}" else cmp_json(res$value$fit$selected),
        n_heldout_projected = n_proj,
        n_integration_projected = if (res$failed) NA_real_ else res$value$integ$projected,
        n_integration_predictions = if (res$failed) NA_real_ else res$value$integ$total,
        n_warnings = length(res$warnings), warnings = paste(res$warnings, collapse = " | "),
        error = if (res$failed) as.character(res$value) else "",
        retries = 0L, fallbacks = 0L, stringsAsFactors = FALSE)
    }
    if (method_failed) phi[] <- NA_real_
    predictions[[method]] <- data.frame(
      id = rows$id, task = rows$task, side = rows$side, outer_fold = rows$outer_fold,
      method = method, prediction_raw = raw_all, prediction = pred_all,
      y = rows$y, true_mean = rows$true_mean, stringsAsFactors = FALSE)
    scores[[method]] <- phi
  }

  effects <- list()
  family <- list()
  for (method in job$methods) {
    phi <- scores[[method]]
    if (anyNA(phi)) {
      effects[[method]] <- cmp_na_effects(geom, surface$truth, method)
      family[[method]] <- cmp_na_family(geom, method)
      next
    }
    fit <- cmp_aggregate(phi, resp$weight)
    inf <- cmp_family_inference(geom, fit, surface$truth,
                                as.integer(engine$contracts$v1$uncertainty$effect_intervals$simultaneous_draws),
                                cmp_hashed_seed(seeds[["bootstrap"]], method))
    inf$effects$method <- method
    inf$family$method <- method
    p <- predictions[[method]]
    inf$family$prediction_mse <- cmp_weighted_mse(p$y, p$prediction, rep(1, nrow(p)))
    inf$family$prediction_log_loss <- if (identical(geom$name, "choice")) {
      e <- as.numeric(engine$contracts$v1$outcome_scale$binary_log_loss_epsilon)
      q <- pmin(1 - e, pmax(e, p$prediction))
      -mean(p$y * log(q) + (1 - p$y) * log(1 - q))
    } else NA_real_
    effects[[method]] <- inf$effects
    family[[method]] <- inf$family
  }

  meta <- list(
    phase = job$phase, geometry = geom$name, cell_id = cell$id, cell_name = cell$name,
    multiplier = job$multiplier %||% NA_real_, replication = job$replication,
    draws = job$draws %||% NA_integer_, methods = job$methods, seeds = as.list(seeds),
    respondents = length(ids), rows = nrow(rows), runtime_seconds = cmp_elapsed(started)
  )
  tag <- function(df) {
    if (is.null(df) || !nrow(df)) return(df)
    cbind(data.frame(phase = meta$phase, geometry = meta$geometry, cell_id = meta$cell_id,
                     cell_name = meta$cell_name, multiplier = meta$multiplier,
                     replication = meta$replication, stringsAsFactors = FALSE), df)
  }
  long_scores <- do.call(rbind, lapply(job$methods, function(method) {
    phi <- scores[[method]]
    data.frame(id = rep(ids, times = k_coord),
               outer_fold = rep(folds$outer_fold, times = k_coord),
               method = method,
               coordinate = rep(colnames(phi), each = length(ids)),
               score = as.numeric(phi), stringsAsFactors = FALSE)
  }))
  list(
    meta = meta,
    truth = surface$truth,
    job_ledger = tag(do.call(rbind, ledger)),
    fold_predictions = tag(do.call(rbind, predictions)),
    respondent_scores = tag(long_scores),
    tuning_history = tag(do.call(rbind, tuning)),
    failures = tag(do.call(rbind, failures)),
    effect_estimates = tag(do.call(rbind, effects)),
    family_metrics = tag(do.call(rbind, family))
  )
}

cmp_na_effects <- function(geom, truth, method) {
  coord <- geom$coordinates
  data.frame(family = coord$family, coordinate = coord$coordinate, key = coord$key,
             truth = as.numeric(truth[coord$key]), estimate = NA_real_, std_error = NA_real_,
             error = NA_real_, lower_95 = NA_real_, upper_95 = NA_real_,
             simultaneous_lower_95 = NA_real_, simultaneous_upper_95 = NA_real_,
             covered_95 = NA, method = method, stringsAsFactors = FALSE)
}

cmp_na_family <- function(geom, method) {
  data.frame(family = names(geom$families),
             dimension = vapply(geom$families, `[[`, 0L, "dimension"),
             covariance_rank = NA_integer_, squared_error = NA_real_, normalized_risk = NA_real_,
             variance_trace = NA_real_, normalized_variance = NA_real_, wald_statistic = NA_real_,
             wald_df = NA_integer_, wald_p_value = NA_real_, truth_wald_statistic = NA_real_,
             truth_wald_covered_95 = NA, simultaneous_critical = NA_real_,
             simultaneous_covered_95 = NA, null_family = NA, method = method,
             prediction_mse = NA_real_, prediction_log_loss = NA_real_,
             stringsAsFactors = FALSE, row.names = NULL)
}
