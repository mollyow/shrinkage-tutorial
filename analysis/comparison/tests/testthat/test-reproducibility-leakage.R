# Gate C1: fixed-coding reproducibility for representation-sensitive learners,
# and no outer or inner respondent leakage.

cmp_tiny_fit_data <- function(eng, cell_id, n = 60L) {
  cell <- cmp_cell(eng, cell_id)
  geom <- cell$geometry_obj
  geom$respondents <- n
  s <- cmp_surface(eng, cell)
  seed <- cmp_seed(eng$contracts$simulation, "test", cell_id, 2L, "data")
  d <- if (geom$name == "rating") cmp_rating_simulate(geom, s, seed) else cmp_choice_simulate(geom, s, seed)
  f <- cmp_respondent_folds(d$respondents$id, 5L, 5L, seed + 7L)
  d$rows$outer_fold <- f$outer_fold[match(d$rows$id, f$id)]
  d$rows$inner_fold <- f$inner_fold[match(d$rows$id, f$id)]
  list(geom = geom, rows = d$rows, folds = f, surface = s)
}

cmp_tiny_contracts <- function(eng) {
  # Shrink the forest and boosting budgets for fast tests only; production
  # budgets are exercised by the Gate C1 synthetic run.
  con <- eng$contracts
  con$v1$learners$definitions$ranger$tuning_trees <- 20L
  con$v1$learners$definitions$ranger$final_trees <- 30L
  con$v1$learners$definitions$ranger$mtry_fraction_grid <- list(0.5)
  con$v1$learners$definitions$ranger$min_node_size_grid <- list(5L, 20L)
  con$v1$learners$definitions$xgboost$max_rounds <- 40L
  con
}

testthat::test_that("representation-sensitive learners reproduce exactly under fixed seeds", {
  eng <- cmp_test_engine()
  con <- cmp_tiny_contracts(eng)
  for (cell_id in c(5L, 9L)) {
    d <- cmp_tiny_fit_data(eng, cell_id, n = if (cell_id == 5L) 100L else 60L)
    fmap <- cmp_feature_map(d$geom)
    train <- d$rows$outer_fold != 1
    for (method in c("elastic_net", "ranger", "xgboost")) {
      fits <- lapply(1:2, function(i) {
        ctx <- cmp_learner_context(d$geom, fmap, d$rows[train, ], rep(1, sum(train)),
                                   d$rows$inner_fold[train], 4242L, con)
        fit <- cmp_fit_method(method, ctx)
        list(pred = fit$predict(d$rows[!train, ]), tuning = fit$tuning, selected = fit$selected)
      })
      testthat::expect_identical(fits[[1]]$pred, fits[[2]]$pred)
      testthat::expect_identical(fits[[1]]$tuning, fits[[2]]$tuning)
      testthat::expect_identical(fits[[1]]$selected, fits[[2]]$selected)
    }
  }
})

testthat::test_that("nested respondent folds partition respondents and cover every inner label", {
  eng <- cmp_test_engine()
  for (cell_id in c(5L, 9L)) {
    d <- cmp_tiny_fit_data(eng, cell_id)
    # every row of a respondent shares one outer and one inner label
    by_id <- tapply(paste(d$rows$outer_fold, d$rows$inner_fold), d$rows$id,
                    function(x) length(unique(x)))
    testthat::expect_true(all(by_id == 1L))
    testthat::expect_silent(cmp_check_nested_folds(d$folds, 5L, 5L))
  }
})

testthat::test_that("held-out predictions do not depend on held-out outcomes", {
  eng <- cmp_test_engine()
  con <- cmp_tiny_contracts(eng)
  for (cell_id in c(5L, 9L)) {
    d <- cmp_tiny_fit_data(eng, cell_id, n = if (cell_id == 5L) 100L else 60L)
    fmap <- cmp_feature_map(d$geom)
    train <- d$rows$outer_fold != 1
    perturbed <- d$rows
    perturbed$y[!train] <- 1 - perturbed$y[!train]
    for (method in c("task_only", "additive", "unrestricted_order_ridge", "elastic_net",
                     "ranger", "xgboost")) {
      preds <- lapply(list(d$rows, perturbed), function(r) {
        ctx <- cmp_learner_context(d$geom, fmap, r[train, ], rep(1, sum(train)),
                                   r$inner_fold[train], 99L, con, basis = d$geom$ridge_basis)
        cmp_fit_method(method, ctx)$predict(r[!train, ])
      })
      testthat::expect_identical(preds[[1]], preds[[2]])
    }
  }
})

testthat::test_that("inner validation losses come from fits on inner-training rows only", {
  eng <- cmp_test_engine()
  con <- cmp_tiny_contracts(eng)
  d <- cmp_tiny_fit_data(eng, 5L, n = 100L)
  geom <- d$geom
  fmap <- cmp_feature_map(geom)
  tr_rows <- d$rows[d$rows$outer_fold != 1, ]
  w <- rep(1, nrow(tr_rows))
  ctx <- cmp_learner_context(geom, fmap, tr_rows, w, tr_rows$inner_fold, 77L, con)
  inner_tr <- tr_rows$inner_fold != 1

  # Elastic net: refit the alpha = 0.5 path on inner-training rows only.
  h <- cmp_fit_method("elastic_net", ctx)$tuning
  def <- con$v1$learners$definitions$elastic_net
  prep_all <- cmp_fit_preprocess(fmap, tr_rows)
  x_all <- cmp_sparse_two_way(ctx, prep_all, tr_rows)
  path <- glmnet::glmnet(x_all, tr_rows$y, weights = ctx$weights, alpha = 0.5,
                         nlambda = def$nlambda, lambda.min.ratio = def$lambda_min_ratio,
                         standardize = TRUE, penalty.factor = attr(x_all, "penalty"))
  p_k <- cmp_fit_preprocess(fmap, tr_rows[inner_tr, ])
  xk <- cmp_sparse_two_way(ctx, p_k, tr_rows[inner_tr, ])
  xv <- cmp_sparse_two_way(ctx, p_k, tr_rows[!inner_tr, ])
  fk <- glmnet::glmnet(xk, tr_rows$y[inner_tr], weights = ctx$weights[inner_tr], alpha = 0.5,
                       lambda = path$lambda, standardize = TRUE, penalty.factor = attr(xk, "penalty"))
  pred <- as.matrix(stats::predict(fk, newx = xv, s = path$lambda))
  loss <- apply(pred, 2L, function(p) cmp_inner_loss(tr_rows$y[!inner_tr], p, ctx$weights[!inner_tr]))
  testthat::expect_identical(h$inner_mse_1[h$alpha == 0.5], unname(loss))

  # Forest: refit candidate 1 on inner fold 1's training rows with the hashed seed.
  hr <- cmp_fit_method("ranger", ctx)$tuning
  frame <- cmp_ranger_frame(geom, fmap, tr_rows)
  g <- hr[1, ]
  fit <- cmp_ranger_fit(frame[inner_tr, ], tr_rows$y[inner_tr], ctx$weights[inner_tr],
                        as.integer(con$v1$learners$definitions$ranger$tuning_trees), g$mtry,
                        g$min_node_size, g$sample_fraction,
                        cmp_hashed_seed(ctx$seed, "ranger", "inner", 1L, 1L))
  pr <- stats::predict(fit, data = frame[!inner_tr, ], num.threads = 1L)$predictions
  testthat::expect_identical(hr$inner_mse_1[[1]],
                             cmp_inner_loss(tr_rows$y[!inner_tr], pr, ctx$weights[!inner_tr]))
})
