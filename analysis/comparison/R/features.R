# Feature maps, fold-internal preprocessing, and model matrices.
#
# Blocks follow comparison-v1.yml `model_matrix`: design context, focal main,
# eligible opponent main, and baseline covariates form the additive matrix; the
# two-way matrix adds cross-factor focal pairs, opponent pairs, focal-by-opponent,
# and profile-by-covariate products. Treatment factors use the target-orthonormal
# contrasts; categorical context and covariates use full one-hot coding.

cmp_feature_map <- function(geom) {
  if (identical(geom$name, "rating")) {
    list(
      design_context = c(task = geom$tasks),
      focal = geom$factors,
      opponent = character(),
      x_numeric = character(),
      x_categorical = geom$context_names
    )
  } else {
    list(
      design_context = c(task = geom$tasks, side = geom$sides),
      focal = geom$factors,
      opponent = geom$opponent_names,
      x_numeric = geom$x_numeric,
      x_categorical = geom$x_binary
    )
  }
}

# Preprocessing learned inside a training sample: medians and IQRs for numeric
# covariates. Categorical levels are fixed by the design (0/1 for covariates).
cmp_fit_preprocess <- function(fmap, train_rows) {
  numeric <- lapply(fmap$x_numeric, function(v) {
    x <- train_rows[[v]]
    cmp_assert(all(is.finite(x)), "Simulated numeric covariate %s has missing values.", v)
    iqr <- stats::IQR(x, type = 7)
    list(center = stats::median(x), scale = if (is.finite(iqr) && iqr > 0) iqr else 1)
  })
  names(numeric) <- fmap$x_numeric
  list(numeric = numeric)
}

cmp_one_hot <- function(codes, n_levels, prefix) {
  codes <- as.integer(codes)
  cmp_assert(!anyNA(codes) && all(codes >= 1L & codes <= n_levels),
             "Unseen or missing level in %s.", prefix)
  out <- matrix(0, length(codes), n_levels)
  out[cbind(seq_along(codes), codes)] <- 1
  colnames(out) <- paste0(prefix, "=", seq_len(n_levels))
  out
}

cmp_contrast_block <- function(geom, rows, columns, factor_names = columns) {
  do.call(cbind, Map(function(col, f) {
    h <- geom$bases[[f]][rows[[col]], , drop = FALSE]
    colnames(h) <- sprintf("%s[h%02d]", col, seq_len(ncol(h)))
    h
  }, columns, factor_names))
}

cmp_tensor <- function(left, right, sep = ":") {
  out <- left[, rep(seq_len(ncol(left)), each = ncol(right)), drop = FALSE] *
    right[, rep(seq_len(ncol(right)), times = ncol(left)), drop = FALSE]
  colnames(out) <- paste(rep(colnames(left), each = ncol(right)),
                         rep(colnames(right), times = ncol(left)), sep = sep)
  out
}

cmp_blocks <- function(geom, fmap, prep, rows) {
  n <- nrow(rows)
  context <- do.call(cbind, lapply(names(fmap$design_context), function(v) {
    cmp_one_hot(rows[[v]], fmap$design_context[[v]], v)
  }))
  focal <- cmp_contrast_block(geom, rows, fmap$focal)
  opponent <- if (length(fmap$opponent)) {
    cmp_contrast_block(geom, rows, fmap$opponent, fmap$focal)
  } else matrix(0, n, 0)
  x_num <- if (length(fmap$x_numeric)) {
    do.call(cbind, lapply(fmap$x_numeric, function(v) {
      out <- matrix((rows[[v]] - prep$numeric[[v]]$center) / prep$numeric[[v]]$scale, n, 1)
      colnames(out) <- v
      out
    }))
  } else matrix(0, n, 0)
  x_cat <- if (length(fmap$x_categorical)) {
    do.call(cbind, lapply(fmap$x_categorical, function(v) {
      cmp_one_hot(rows[[v]] + 1L, 2L, v)
    }))
  } else matrix(0, n, 0)
  list(context = context, focal = focal, opponent = opponent, x = cbind(x_num, x_cat))
}

cmp_focal_pairs <- function(geom, rows, columns, factor_names = columns) {
  idx <- utils::combn(seq_along(columns), 2L, simplify = FALSE)
  do.call(cbind, lapply(idx, function(ij) {
    cmp_tensor(cmp_contrast_block(geom, rows, columns[ij[[1]]], factor_names[ij[[1]]]),
               cmp_contrast_block(geom, rows, columns[ij[[2]]], factor_names[ij[[2]]]))
  }))
}

# kind: "task_only", "additive", or "two_way". Returns a dense matrix with an
# intercept column and a parallel `penalty` vector (0 = never penalized).
cmp_model_matrix <- function(geom, fmap, prep, rows, kind) {
  b <- cmp_blocks(geom, fmap, prep, rows)
  n <- nrow(rows)
  pieces <- list(`(Intercept)` = matrix(1, n, 1, dimnames = list(NULL, "(Intercept)")),
                 context = b$context)
  if (kind %in% c("additive", "two_way")) {
    pieces <- c(pieces, list(focal = b$focal, opponent = b$opponent, x = b$x))
  }
  if (identical(kind, "two_way")) {
    pieces$focal_pairs <- cmp_focal_pairs(geom, rows, fmap$focal)
    if (length(fmap$opponent)) {
      pieces$opponent_pairs <- cmp_focal_pairs(geom, rows, fmap$opponent, fmap$focal)
      pieces$focal_x_opponent <- cmp_tensor(b$focal, b$opponent)
    }
    pieces$focal_x_x <- cmp_tensor(b$focal, b$x)
    if (length(fmap$opponent)) pieces$opponent_x_x <- cmp_tensor(b$opponent, b$x)
  }
  pieces <- Filter(function(m) ncol(m) > 0L, pieces)
  x <- do.call(cbind, pieces)
  block <- rep(names(pieces), vapply(pieces, ncol, integer(1)))
  attr(x, "block") <- block
  attr(x, "penalty") <- ifelse(block %in% c("(Intercept)", "context"), 0, 1)
  cmp_assert(!anyDuplicated(colnames(x)), "Duplicate model-matrix columns.")
  x
}

# Raw predictor frame for the forest: locked factor levels, raw numeric values.
cmp_ranger_frame <- function(geom, fmap, rows) {
  out <- list()
  for (v in names(fmap$design_context)) {
    out[[v]] <- factor(rows[[v]], levels = seq_len(fmap$design_context[[v]]))
  }
  for (j in seq_along(fmap$focal)) {
    f <- fmap$focal[[j]]
    out[[f]] <- factor(rows[[f]], levels = seq_len(geom$levels[[f]]))
  }
  for (j in seq_along(fmap$opponent)) {
    out[[fmap$opponent[[j]]]] <- factor(rows[[fmap$opponent[[j]]]],
                                        levels = seq_len(geom$levels[[fmap$focal[[j]]]]))
  }
  for (v in fmap$x_numeric) out[[v]] <- as.numeric(rows[[v]])
  for (v in fmap$x_categorical) out[[v]] <- factor(rows[[v]], levels = c(0, 1))
  out <- as.data.frame(out, stringsAsFactors = FALSE, check.names = FALSE)
  cmp_assert(!anyNA(out), "Unseen or missing level in the forest input.")
  out
}

# Inputs for order ridge: focal factors (penalized factorial basis) and a
# full-rank unpenalized matrix (reference-coded context, opponent contrasts,
# covariates). The package adds the intercept.
cmp_ridge_inputs <- function(geom, fmap, prep, rows) {
  focal <- as.data.frame(lapply(fmap$focal, function(f) {
    factor(rows[[f]], levels = seq_len(geom$levels[[f]]))
  }))
  names(focal) <- fmap$focal
  b <- cmp_blocks(geom, fmap, prep, rows)
  keep_ref <- function(m) {
    prefixes <- sub("=.*$", "", colnames(m))
    first <- !duplicated(prefixes)
    m[, !first, drop = FALSE]
  }
  x_num <- b$x[, fmap$x_numeric, drop = FALSE]
  x_cat <- b$x[, setdiff(colnames(b$x), fmap$x_numeric), drop = FALSE]
  u <- cbind(keep_ref(b$context), b$opponent, x_num, keep_ref(x_cat))
  list(x = focal, u = u)
}

cmp_ridge_basis <- function(geom, max_order = 3L) {
  grid <- as.data.frame(lapply(geom$factors, function(f) {
    factor(geom$grid[, f], levels = seq_len(geom$levels[[f]]))
  }))
  names(grid) <- geom$factors
  orderridge::factorial_basis(grid, factors = geom$factors, target = grid,
                              max_order = max_order)
}
