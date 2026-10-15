# Learner adapters for the frozen library (comparison-v1.yml `learners`).
#
# Every adapter receives only outer-training rows, tunes on the locked inner
# respondent labels, and returns a prediction function plus a tuning history.
# Predictions are returned unprojected; callers project to [0, 1] before loss,
# integration, and scoring. Tuning losses use projected predictions and
# mean-one validation weights, except order ridge, whose package optimizer uses
# its closed-form weighted inner-fold MSE.

CMP_PRIMARY_METHODS <- c("unadjusted", "task_only", "additive", "two_way",
                         "unrestricted_order_ridge", "elastic_net", "ranger", "xgboost")
CMP_ABLATION_METHODS <- c("common_order_ridge", "monotone_order_ridge")
CMP_SIMULATION_METHODS <- c(CMP_PRIMARY_METHODS, CMP_ABLATION_METHODS, "oracle")

cmp_learner_context <- function(geom, fmap, train, weights, inner, seed, contracts,
                                basis = NULL, surface = NULL) {
  cmp_assert(length(weights) == nrow(train) && length(inner) == nrow(train),
             "Training weights and inner labels must align with training rows.")
  list(geom = geom, fmap = fmap, train = train, weights = weights / mean(weights),
       inner = inner, seed = seed, defs = contracts$v1$learners$definitions,
       basis = basis, surface = surface)
}

cmp_capture <- function(expr) {
  warnings <- character()
  value <- withCallingHandlers(
    tryCatch(expr, error = function(e) structure(conditionMessage(e), class = "cmp_error")),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(value = value, warnings = unique(warnings), failed = inherits(value, "cmp_error"))
}

cmp_inner_loss <- function(y, prediction, weights) {
  cmp_weighted_mse(y, cmp_project_unit(prediction), weights)
}

# ---------------------------------------------------------------------------
# Minimum-norm weighted least squares with a row-space extrapolation guard.
# ---------------------------------------------------------------------------

cmp_minnorm_wls <- function(x, y, w, tol = 1e-10) {
  sw <- sqrt(w)
  s <- svd(x * sw, nu = 0L, nv = ncol(x))
  rank <- sum(s$d > tol * s$d[[1]])
  v_r <- s$v[, seq_len(rank), drop = FALSE]
  # beta = V_r D_r^{-1} U_r' (sqrt(w) y), with U_r' = D_r^{-1} V_r' X' sqrt(w)
  xty <- crossprod(x * sw, y * sw)
  beta <- v_r %*% (crossprod(v_r, xty) / s$d[seq_len(rank)]^2)
  null <- if (rank < ncol(x)) s$v[, (rank + 1L):ncol(x), drop = FALSE] else NULL
  list(beta = as.numeric(beta), rank = rank, null = null, columns = colnames(x))
}

cmp_minnorm_predict <- function(fit, x, tol = 1e-8) {
  cmp_assert(identical(colnames(x), fit$columns), "Least-squares columns changed at prediction.")
  if (!is.null(fit$null)) {
    leak <- sqrt(rowSums((x %*% fit$null)^2)) / pmax(1, sqrt(rowSums(x^2)))
    cmp_assert(all(leak < tol),
               "Least-squares prediction outside the training column space (max %.3g).", max(leak))
  }
  as.numeric(x %*% fit$beta)
}

cmp_fit_ls <- function(ctx, kind) {
  prep <- cmp_fit_preprocess(ctx$fmap, ctx$train)
  x <- cmp_model_matrix(ctx$geom, ctx$fmap, prep, ctx$train, kind)
  fit <- cmp_minnorm_wls(x, ctx$train$y, ctx$weights)
  list(
    predict = function(rows) {
      cmp_minnorm_predict(fit, cmp_model_matrix(ctx$geom, ctx$fmap, prep, rows, kind))
    },
    selected = list(rank = fit$rank, columns = length(fit$columns)),
    tuning = NULL
  )
}

# ---------------------------------------------------------------------------
# Order ridge (package orderridge)
# ---------------------------------------------------------------------------

cmp_fit_order_ridge <- function(ctx, structure) {
  def <- ctx$defs$unrestricted_order_ridge
  prep <- cmp_fit_preprocess(ctx$fmap, ctx$train)
  inp <- cmp_ridge_inputs(ctx$geom, ctx$fmap, prep, ctx$train)
  fit <- orderridge::fit_order_ridge(
    inp$x, ctx$train$y, ctx$basis, weights = ctx$weights, cluster = ctx$train$id,
    unpenalized = inp$u, penalty_structure = structure, fold_id = ctx$inner,
    control = list(
      common_grid_size = as.integer(def$common_grid_size),
      maxit = as.integer(def$optimizer_max_iterations),
      reltol = as.numeric(def$optimizer_relative_tolerance),
      log_lambda_lower = as.numeric(def$log_lambda_bounds[[1]]),
      log_lambda_upper = as.numeric(def$log_lambda_bounds[[2]])
    )
  )
  ev <- fit$tuning$evaluations
  tuning <- if (!is.null(ev)) data.frame(candidate = seq_len(nrow(ev)), ev,
                                         check.names = FALSE) else NULL
  # The penalized part depends only on the focal profile, so evaluate it once on
  # the complete profile grid; the result equals predict.order_ridge exactly.
  grid_x <- as.data.frame(lapply(ctx$geom$factors, function(f) {
    factor(ctx$geom$grid[, f], levels = seq_len(ctx$geom$levels[[f]]))
  }))
  names(grid_x) <- ctx$geom$factors
  profile_part <- as.numeric(
    orderridge::factorial_design(ctx$basis, grid_x, include_intercept = FALSE) %*% fit$coefficients
  )
  list(
    predict = function(rows) {
      new <- cmp_ridge_inputs(ctx$geom, ctx$fmap, prep, rows)
      idx <- cmp_profile_index(ctx$geom, rows, ctx$geom$factors)
      as.numeric(cbind(1, new$u) %*% fit$unpenalized_coefficients) + profile_part[idx]
    },
    predict_package = function(rows) {
      new <- cmp_ridge_inputs(ctx$geom, ctx$fmap, prep, rows)
      stats::predict(fit, newdata = new$x, new_unpenalized = new$u)
    },
    selected = c(as.list(stats::setNames(fit$lambda, paste0("lambda_order_", names(fit$lambda)))),
                 list(cv_error = fit$tuning$cv_error %||% NA_real_,
                      convergence = fit$tuning$convergence %||% NA_integer_,
                      initializer = fit$tuning$initializer %||% NA_character_)),
    tuning = tuning
  )
}

# ---------------------------------------------------------------------------
# Elastic net (glmnet)
# ---------------------------------------------------------------------------

cmp_sparse_two_way <- function(ctx, prep, rows) {
  x <- cmp_model_matrix(ctx$geom, ctx$fmap, prep, rows, "two_way")
  pen <- attr(x, "penalty")[-1L]
  x <- x[, -1L, drop = FALSE]
  out <- Matrix::Matrix(x, sparse = TRUE)
  attr(out, "penalty") <- pen
  out
}

cmp_fit_elastic_net <- function(ctx) {
  def <- ctx$defs$elastic_net
  alphas <- as.numeric(unlist(def$alpha_grid))
  folds <- sort(unique(ctx$inner))
  prep <- cmp_fit_preprocess(ctx$fmap, ctx$train)
  x_full <- cmp_sparse_two_way(ctx, prep, ctx$train)
  pf <- attr(x_full, "penalty")
  y <- ctx$train$y
  inner_mats <- lapply(folds, function(k) {
    tr <- ctx$inner != k
    p_k <- cmp_fit_preprocess(ctx$fmap, ctx$train[tr, , drop = FALSE])
    list(tr = tr,
         x_tr = cmp_sparse_two_way(ctx, p_k, ctx$train[tr, , drop = FALSE]),
         x_va = cmp_sparse_two_way(ctx, p_k, ctx$train[!tr, , drop = FALSE]))
  })
  fits <- list()
  history <- list()
  for (a in alphas) {
    path <- glmnet::glmnet(x_full, y, weights = ctx$weights, family = "gaussian", alpha = a,
                           nlambda = as.integer(def$nlambda),
                           lambda.min.ratio = as.numeric(def$lambda_min_ratio),
                           standardize = isTRUE(def$standardize), penalty.factor = pf)
    lam <- path$lambda
    losses <- vapply(inner_mats, function(m) {
      fk <- glmnet::glmnet(m$x_tr, y[m$tr], weights = ctx$weights[m$tr], family = "gaussian",
                           alpha = a, lambda = lam, standardize = isTRUE(def$standardize),
                           penalty.factor = pf)
      pred <- as.matrix(stats::predict(fk, newx = m$x_va, s = lam))
      apply(pred, 2L, function(p) cmp_inner_loss(y[!m$tr], p, ctx$weights[!m$tr]))
    }, numeric(length(lam)))
    losses <- matrix(losses, nrow = length(lam))
    fits[[as.character(a)]] <- path
    history[[as.character(a)]] <- data.frame(
      alpha = a, lambda = lam, mean_inner_mse = rowMeans(losses),
      setNames(as.data.frame(losses), paste0("inner_mse_", folds)), check.names = FALSE
    )
  }
  h <- do.call(rbind, history)
  # exact minimum, then larger lambda, then larger alpha
  best <- order(h$mean_inner_mse, -h$lambda, -h$alpha)[[1]]
  h$selected <- seq_len(nrow(h)) == best
  a_star <- h$alpha[[best]]
  l_star <- h$lambda[[best]]
  path <- fits[[as.character(a_star)]]
  list(
    predict = function(rows) {
      as.numeric(stats::predict(path, newx = cmp_sparse_two_way(ctx, prep, rows), s = l_star))
    },
    selected = list(alpha = a_star, lambda = l_star, mean_inner_mse = h$mean_inner_mse[[best]]),
    tuning = data.frame(candidate = seq_len(nrow(h)), h, check.names = FALSE)
  )
}

# ---------------------------------------------------------------------------
# Regression forest (ranger)
# ---------------------------------------------------------------------------

cmp_ranger_fit <- function(frame, y, w, trees, mtry, node, fraction, seed) {
  ranger::ranger(
    x = frame, y = y, case.weights = w, num.trees = trees, mtry = mtry,
    min.node.size = node, sample.fraction = fraction, replace = FALSE,
    splitrule = "variance", respect.unordered.factors = "partition",
    num.threads = 1L, seed = seed, write.forest = TRUE, verbose = FALSE
  )
}

cmp_fit_ranger <- function(ctx) {
  def <- ctx$defs$ranger
  frame <- cmp_ranger_frame(ctx$geom, ctx$fmap, ctx$train)
  p <- ncol(frame)
  grid <- expand.grid(mtry_fraction = as.numeric(unlist(def$mtry_fraction_grid)),
                      min_node_size = as.integer(unlist(def$min_node_size_grid)),
                      sample_fraction = as.numeric(unlist(def$sample_fraction_grid)),
                      KEEP.OUT.ATTRS = FALSE)
  grid$mtry <- pmax(1L, as.integer(floor(grid$mtry_fraction * p)))
  folds <- sort(unique(ctx$inner))
  y <- ctx$train$y
  losses <- matrix(NA_real_, nrow(grid), length(folds))
  errors <- character(nrow(grid))
  for (g in seq_len(nrow(grid))) {
    for (j in seq_along(folds)) {
      tr <- ctx$inner != folds[[j]]
      res <- cmp_capture({
        fit <- cmp_ranger_fit(frame[tr, , drop = FALSE], y[tr], ctx$weights[tr],
                              as.integer(def$tuning_trees), grid$mtry[[g]],
                              grid$min_node_size[[g]], grid$sample_fraction[[g]],
                              cmp_hashed_seed(ctx$seed, "ranger", "inner", folds[[j]], g))
        pred <- stats::predict(fit, data = frame[!tr, , drop = FALSE], num.threads = 1L)$predictions
        cmp_inner_loss(y[!tr], pred, ctx$weights[!tr])
      })
      if (res$failed) errors[[g]] <- as.character(res$value) else losses[g, j] <- res$value
    }
  }
  h <- data.frame(candidate = seq_len(nrow(grid)), grid,
                  mean_inner_mse = rowMeans(losses),
                  setNames(as.data.frame(losses), paste0("inner_mse_", folds)),
                  error = errors, check.names = FALSE)
  ok <- which(is.finite(h$mean_inner_mse))
  cmp_assert(length(ok) > 0L, "Every forest tuning candidate failed.")
  # exact minimum, then larger node, smaller mtry, smaller sample fraction
  best <- ok[order(h$mean_inner_mse[ok], -h$min_node_size[ok], h$mtry[ok],
                   h$sample_fraction[ok])[[1]]]
  h$selected <- seq_len(nrow(h)) == best
  fit <- cmp_ranger_fit(frame, y, ctx$weights, as.integer(def$final_trees), h$mtry[[best]],
                        h$min_node_size[[best]], h$sample_fraction[[best]],
                        cmp_hashed_seed(ctx$seed, "ranger", "final"))
  list(
    predict = function(rows) {
      stats::predict(fit, data = cmp_ranger_frame(ctx$geom, ctx$fmap, rows),
                     num.threads = 1L)$predictions
    },
    selected = as.list(h[best, c("mtry", "min_node_size", "sample_fraction", "mean_inner_mse")]),
    tuning = h
  )
}

# ---------------------------------------------------------------------------
# Boosted trees (xgboost)
# ---------------------------------------------------------------------------

cmp_xgb_params <- function(def, eta, depth, mcw, seed) {
  xgboost::xgb.params(
    objective = def$objective, tree_method = def$tree_method, eta = eta,
    max_depth = depth, min_child_weight = mcw, subsample = def$subsample,
    colsample_bytree = def$colsample_bytree, reg_alpha = def$reg_alpha,
    reg_lambda = def$reg_lambda, gamma = def$gamma, nthread = 1L, seed = seed
  )
}

cmp_fit_xgboost <- function(ctx) {
  def <- ctx$defs$xgboost
  grid <- expand.grid(eta = as.numeric(unlist(def$eta_grid)),
                      max_depth = as.integer(unlist(def$max_depth_grid)),
                      min_child_weight = as.numeric(unlist(def$min_child_weight_grid)),
                      KEEP.OUT.ATTRS = FALSE)
  folds <- sort(unique(ctx$inner))
  y <- ctx$train$y
  inner_mats <- lapply(folds, function(k) {
    tr <- ctx$inner != k
    p_k <- cmp_fit_preprocess(ctx$fmap, ctx$train[tr, , drop = FALSE])
    list(tr = tr,
         d_tr = xgboost::xgb.DMatrix(cmp_sparse_two_way(ctx, p_k, ctx$train[tr, , drop = FALSE]),
                                     label = y[tr], weight = ctx$weights[tr], nthread = 1L),
         d_va = xgboost::xgb.DMatrix(cmp_sparse_two_way(ctx, p_k, ctx$train[!tr, , drop = FALSE]),
                                     label = y[!tr], weight = ctx$weights[!tr], nthread = 1L))
  })
  losses <- rounds <- matrix(NA_real_, nrow(grid), length(folds))
  errors <- character(nrow(grid))
  for (g in seq_len(nrow(grid))) {
    for (j in seq_along(folds)) {
      m <- inner_mats[[j]]
      res <- cmp_capture({
        fit <- xgboost::xgb.train(
          params = cmp_xgb_params(def, grid$eta[[g]], grid$max_depth[[g]],
                                  grid$min_child_weight[[g]],
                                  cmp_hashed_seed(ctx$seed, "xgboost", "inner", folds[[j]], g)),
          data = m$d_tr, nrounds = as.integer(def$max_rounds), evals = list(valid = m$d_va),
          early_stopping_rounds = as.integer(def$early_stopping_rounds), verbose = 0L
        )
        n_best <- as.integer(xgboost::xgb.attr(fit, "best_iteration")) + 1L
        pred <- stats::predict(fit, m$d_va, iterationrange = c(1L, n_best))
        c(cmp_inner_loss(y[!m$tr], pred, ctx$weights[!m$tr]), n_best)
      })
      if (res$failed) {
        errors[[g]] <- as.character(res$value)
      } else {
        losses[g, j] <- res$value[[1]]
        rounds[g, j] <- res$value[[2]]
      }
    }
  }
  h <- data.frame(candidate = seq_len(nrow(grid)), grid,
                  mean_inner_mse = rowMeans(losses), mean_rounds = rowMeans(rounds),
                  setNames(as.data.frame(losses), paste0("inner_mse_", folds)),
                  setNames(as.data.frame(rounds), paste0("inner_rounds_", folds)),
                  error = errors, check.names = FALSE)
  ok <- which(is.finite(h$mean_inner_mse))
  cmp_assert(length(ok) > 0L, "Every boosting tuning candidate failed.")
  # minimum mean inner MSE, then fewer rounds, shallower trees, larger min_child_weight
  best <- ok[order(h$mean_inner_mse[ok], h$mean_rounds[ok], h$max_depth[ok],
                   -h$min_child_weight[ok])[[1]]]
  h$selected <- seq_len(nrow(h)) == best
  n_final <- max(1L, as.integer(round(h$mean_rounds[[best]])))
  prep <- cmp_fit_preprocess(ctx$fmap, ctx$train)
  d_full <- xgboost::xgb.DMatrix(cmp_sparse_two_way(ctx, prep, ctx$train), label = y,
                                 weight = ctx$weights, nthread = 1L)
  fit <- xgboost::xgb.train(
    params = cmp_xgb_params(def, h$eta[[best]], h$max_depth[[best]], h$min_child_weight[[best]],
                            cmp_hashed_seed(ctx$seed, "xgboost", "final")),
    data = d_full, nrounds = n_final, verbose = 0L
  )
  list(
    predict = function(rows) {
      stats::predict(fit, xgboost::xgb.DMatrix(cmp_sparse_two_way(ctx, prep, rows), nthread = 1L))
    },
    selected = c(as.list(h[best, c("eta", "max_depth", "min_child_weight", "mean_inner_mse")]),
                 list(rounds = n_final)),
    tuning = h
  )
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

cmp_true_mean <- function(geom, surface, rows) {
  if (identical(geom$name, "rating")) {
    cmp_rating_mean(geom, surface, as.matrix(rows[geom$factors]), rows$task,
                    as.matrix(rows[geom$context_names]))
  } else {
    focal <- cmp_profile_index(geom, rows, geom$factors)
    opp <- cmp_profile_index(geom, rows, geom$opponent_names)
    cmp_choice_mean(geom, surface, focal, opp, as.matrix(rows[geom$context_names]))
  }
}

# Profile index in the expand.grid ordering (first factor varies fastest).
cmp_profile_index <- function(geom, rows, columns) {
  idx <- rep(1L, nrow(rows))
  stride <- 1L
  for (j in seq_along(geom$factors)) {
    idx <- idx + (as.integer(rows[[columns[[j]]]]) - 1L) * stride
    stride <- stride * geom$levels[[j]]
  }
  idx
}

cmp_fit_method <- function(method, ctx) {
  switch(
    method,
    unadjusted = list(predict = function(rows) rep(0, nrow(rows)),
                      selected = list(), tuning = NULL),
    task_only = cmp_fit_ls(ctx, "task_only"),
    additive = cmp_fit_ls(ctx, "additive"),
    two_way = cmp_fit_ls(ctx, "two_way"),
    unrestricted_order_ridge = cmp_fit_order_ridge(ctx, "unrestricted"),
    common_order_ridge = cmp_fit_order_ridge(ctx, "common"),
    monotone_order_ridge = cmp_fit_order_ridge(ctx, "monotone"),
    elastic_net = cmp_fit_elastic_net(ctx),
    ranger = cmp_fit_ranger(ctx),
    xgboost = cmp_fit_xgboost(ctx),
    oracle = {
      cmp_assert(!is.null(ctx$surface), "The oracle is available only in simulations.")
      list(predict = function(rows) cmp_true_mean(ctx$geom, ctx$surface, rows),
           selected = list(), tuning = NULL)
    },
    stop("Unknown method `", method, "`.", call. = FALSE)
  )
}
