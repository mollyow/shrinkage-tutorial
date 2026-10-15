# Paired-choice simulation geometry (comparison-simulations-v1.yml,
# choice_geometry): a synthetic four-factor subset of the audited asylum product
# design with independent uniform focal and opponent profiles.
#
# The frozen contract names the utility components and their scales but not the
# functional form of the nonlinear profile-by-covariate term. Gate C1 fixes it
# before any calibration (see README.md, "Implementation choices"):
#   NL(p, x) = sum_k g_k(p) psi_k(x),  g_k(p) = sum_j B[j, k] phi_j(p),
# where phi are the 15 orthonormal main-effect basis functions and psi_k are four
# standardized threshold features of the baseline covariates, mirroring the four
# Boolean nonlinear context features of the rating geometry. The thresholds make
# psi discrete, so exact truth is a 16-point enumeration.

CMP_CHOICE_X_NUMERIC <- c("z1", "z2", "z3", "z4")
CMP_CHOICE_X_BINARY <- c("b1", "b2", "b3", "b4")

cmp_choice_features_raw <- function(x) {
  cbind(
    xor_b1_b2 = as.numeric(x[, "b1"] != x[, "b2"]),
    z1_pos_and_z2_pos = as.numeric(x[, "z1"] > 0 & x[, "z2"] > 0),
    b3_and_z3_pos = as.numeric(x[, "b3"] == 1 & x[, "z3"] > 0),
    abs_z4_above_quartile = as.numeric(abs(x[, "z4"]) > stats::qnorm(0.75))
  )
}

# Exact means of the raw features under the covariate law.
CMP_CHOICE_FEATURE_MEAN <- c(0.5, 0.25, 0.25, 0.5)

cmp_choice_geometry <- function(contracts) {
  sim <- contracts$simulation$choice_geometry
  levels <- vapply(sim$factors, as.integer, integer(1))
  factors <- names(levels)
  grid <- do.call(expand.grid, c(lapply(levels, seq_len), KEEP.OUT.ATTRS = FALSE))
  names(grid) <- factors
  grid <- as.matrix(grid)
  storage.mode(grid) <- "integer"
  cmp_assert(nrow(grid) == sim$profiles, "Expected %d choice profiles.", sim$profiles)

  bases <- cmp_factor_bases(levels)
  fam_spec <- lapply(names(sim$target_families), function(id) {
    list(id = id, attributes = strsplit(id, "_x_", fixed = TRUE)[[1]])
  })
  families <- cmp_build_families(bases, fam_spec)
  for (f in families) {
    cmp_assert(f$dimension == sim$target_families[[f$id]],
               "Choice family %s has the wrong dimension.", f$id)
  }

  main <- do.call(cbind, lapply(factors, function(f) bases[[f]][grid[, f], , drop = FALSE]))
  pairs <- utils::combn(factors, 2L, simplify = FALSE)
  pair <- do.call(cbind, lapply(pairs, function(ab) {
    ha <- bases[[ab[[1]]]][grid[, ab[[1]]], , drop = FALSE]
    hb <- bases[[ab[[2]]]][grid[, ab[[2]]], , drop = FALSE]
    ha[, rep(seq_len(ncol(ha)), times = ncol(hb)), drop = FALSE] *
      hb[, rep(seq_len(ncol(hb)), each = ncol(ha)), drop = FALSE]
  }))

  cg <- sim$coefficient_generation
  coef <- cmp_with_seed(cg$seed, {
    b_main <- stats::rnorm(ncol(main))
    b_pair <- stats::rnorm(ncol(pair))
    b_nl <- matrix(stats::rnorm(ncol(main) * 4L), ncol(main), 4L)
    list(main = b_main, pair = b_pair, nl = b_nl)
  })
  coef$main <- coef$main * 0.35 / sqrt(sum(coef$main^2))
  coef$pair <- coef$pair * 0.15 / sqrt(sum(coef$pair^2))
  coef$nl <- coef$nl / sqrt(sum(coef$nl^2))

  # Verify the component RMS values against the frozen scales.
  p <- 1 / nrow(grid)
  rms <- function(v) sqrt(sum(p * v^2))
  cmp_assert(abs(rms(main %*% coef$main) - 0.35) < 1e-10, "Main utility RMS is wrong.")
  cmp_assert(abs(rms(pair %*% coef$pair) - 0.15) < 1e-10, "Pair utility RMS is wrong.")

  # The 16 joint values of the four binary raw features and their probabilities.
  combos <- as.matrix(expand.grid(f1 = 0:1, f2 = 0:1, f3 = 0:1, f4 = 0:1))
  combo_prob <- apply(combos, 1L, function(v) {
    prod(ifelse(v == 1, CMP_CHOICE_FEATURE_MEAN, 1 - CMP_CHOICE_FEATURE_MEAN))
  })
  feature_sd <- sqrt(CMP_CHOICE_FEATURE_MEAN * (1 - CMP_CHOICE_FEATURE_MEAN))

  list(
    name = "choice", factors = factors, levels = levels,
    labels = lapply(levels, function(n) as.character(seq_len(n))),
    grid = grid, bases = bases, families = families,
    coordinates = cmp_coordinate_table(families),
    tasks = as.integer(sim$tasks_per_respondent),
    respondents = as.integer(sim$respondents),
    has_opponent = TRUE, sides = 2L,
    opponent_names = paste0("opp_", factors),
    x_numeric = CMP_CHOICE_X_NUMERIC, x_binary = CMP_CHOICE_X_BINARY,
    context_names = c(CMP_CHOICE_X_NUMERIC, CMP_CHOICE_X_BINARY),
    assignment_probability = 1 / nrow(grid),
    main = main, pair = pair, coef = coef,
    combos = combos, combo_prob = combo_prob, feature_sd = feature_sd,
    latent_correlation = sim$within_respondent_outcome_generator$latent_correlation
  )
}

cmp_choice_cell_spec <- function(geom, cell, multiplier = NULL) {
  switch(
    cell$name,
    choice_additive = list(pair = 0, nl = 0),
    choice_low_headroom_nonlinear = list(pair = 1, nl = 0.25),
    choice_high_headroom_control = {
      cmp_assert(!is.null(multiplier), "Cell %s needs a heterogeneity multiplier.", cell$name)
      list(pair = 1, nl = multiplier)
    },
    stop("Unknown choice cell ", cell$name, call. = FALSE)
  )
}

cmp_choice_psi <- function(geom, raw_features) {
  sweep(sweep(raw_features, 2L, CMP_CHOICE_FEATURE_MEAN, "-"), 2L, geom$feature_sd, "/")
}

# Utility of every profile (rows) for each standardized feature vector (rows of psi).
cmp_choice_utility <- function(geom, spec, psi) {
  base <- as.numeric(geom$main %*% geom$coef$main) +
    spec$pair * as.numeric(geom$pair %*% geom$coef$pair)
  het <- geom$main %*% geom$coef$nl %*% t(psi)
  base + spec$nl * het
}

cmp_choice_surface <- function(geom, cell, multiplier = NULL) {
  spec <- cmp_choice_cell_spec(geom, cell, multiplier)
  psi <- cmp_choice_psi(geom, geom$combos)
  u <- cmp_choice_utility(geom, spec, psi)
  n_prof <- nrow(geom$grid)
  mu <- rep(1 / n_prof, n_prof)
  combo_theta <- matrix(NA_real_, nrow(psi), nrow(geom$coordinates),
                        dimnames = list(NULL, geom$coordinates$key))
  for (r in seq_len(nrow(psi))) {
    m <- stats::plogis(outer(u[, r], u[, r], "-"))
    focal_mean <- rowMeans(m)
    combo_theta[r, ] <- cmp_theta_from_profiles(geom, focal_mean, mu)
  }
  truth <- as.numeric(crossprod(geom$combo_prob, combo_theta))
  names(truth) <- geom$coordinates$key
  het_rms <- sqrt(sum(geom$combo_prob * colSums((geom$main %*% geom$coef$nl %*% t(psi))^2) / n_prof))
  list(cell = cell, multiplier = multiplier, spec = spec, valid = TRUE,
       truth = truth, combo_theta = combo_theta, het_rms = het_rms,
       target_ratio = rep(1, n_prof))
}

cmp_choice_mean <- function(geom, surface, focal_index, opponent_index, x) {
  psi <- cmp_choice_psi(geom, cmp_choice_features_raw(x))
  base <- as.numeric(geom$main %*% geom$coef$main) +
    surface$spec$pair * as.numeric(geom$pair %*% geom$coef$pair)
  gm <- geom$main %*% geom$coef$nl
  het_f <- rowSums(gm[focal_index, , drop = FALSE] * psi)
  het_o <- rowSums(gm[opponent_index, , drop = FALSE] * psi)
  stats::plogis(base[focal_index] - base[opponent_index] +
                  surface$spec$nl * (het_f - het_o))
}

cmp_choice_simulate <- function(geom, surface, seed_data) {
  g <- geom$respondents
  tt <- geom$tasks
  n_prof <- nrow(geom$grid)
  draws <- cmp_with_seed(seed_data, {
    z <- matrix(stats::rnorm(g * 4L), g, 4L)
    b <- matrix(stats::runif(g * 4L) < 0.5, g, 4L) * 1L
    p1 <- sample.int(n_prof, g * tt, replace = TRUE)
    p2 <- sample.int(n_prof, g * tt, replace = TRUE)
    shared <- stats::rnorm(g)
    idio <- stats::rnorm(g * tt)
    list(z = z, b = b, p1 = p1, p2 = p2, shared = shared, idio = idio)
  })
  x <- cbind(draws$z, draws$b)
  colnames(x) <- geom$context_names
  ids <- sprintf("r%04d", seq_len(g))
  x_task <- x[rep(seq_len(g), each = tt), , drop = FALSE]
  p_first <- cmp_choice_mean(geom, surface, draws$p1, draws$p2, x_task)
  rho <- geom$latent_correlation
  latent <- sqrt(rho) * rep(draws$shared, each = tt) + sqrt(1 - rho) * draws$idio
  y_first <- as.integer(stats::pnorm(latent) < p_first)

  # Each task contributes two oriented rows: side 1 has profile 1 as focal.
  task_rows <- data.frame(id = rep(ids, each = tt), task = rep(seq_len(tt), times = g),
                          stringsAsFactors = FALSE)
  orient <- function(side) {
    focal <- if (side == 1L) draws$p1 else draws$p2
    opp <- if (side == 1L) draws$p2 else draws$p1
    out <- task_rows
    out$side <- side
    fc <- geom$grid[focal, , drop = FALSE]
    oc <- geom$grid[opp, , drop = FALSE]
    colnames(oc) <- geom$opponent_names
    out <- cbind(out, as.data.frame(fc), as.data.frame(oc), as.data.frame(x_task))
    out$profile_index <- focal
    out$opponent_index <- opp
    out$y <- if (side == 1L) y_first else 1L - y_first
    out$true_mean <- if (side == 1L) p_first else 1 - p_first
    out
  }
  rows <- rbind(orient(1L), orient(2L))
  rows <- rows[order(rows$id, rows$task, rows$side), ]
  rownames(rows) <- NULL
  rows$y <- as.numeric(rows$y)
  rows$target_ratio <- 1
  respondents <- data.frame(id = ids, weight = 1, stringsAsFactors = FALSE)
  respondents <- cbind(respondents, as.data.frame(x))
  list(rows = rows, respondents = respondents)
}
