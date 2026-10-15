# Repeated-rating simulation geometry (comparison-simulations-v1.yml,
# rating_geometry): the developmental candidate design with 1,440 profiles,
# eight tasks per respondent, and the four locked candidate interaction families.

CMP_RATING_CONTEXT <- c("woman", "poc", "young", "dem", "rep", "workingclass")

cmp_rating_geometry <- function(contracts) {
  root <- contracts$root
  sim <- contracts$simulation$rating_geometry
  cand <- contracts$candidate
  factors <- as.character(unlist(sim$attributes))
  cmp_assert(identical(factors, names(cand$attributes)),
             "Rating factors differ from the candidate estimand contract.")
  levels <- vapply(cand$attributes[factors], function(a) length(a$values), integer(1))
  labels <- lapply(cand$attributes[factors], function(a) as.character(unlist(a$labels)))

  grid <- do.call(expand.grid, c(lapply(levels, seq_len), KEEP.OUT.ATTRS = FALSE))
  names(grid) <- factors
  grid <- as.matrix(grid)
  storage.mode(grid) <- "integer"
  cmp_assert(nrow(grid) == sim$profiles, "Expected %d profiles.", sim$profiles)

  bases <- cmp_factor_bases(levels)
  families <- cmp_build_families(bases, lapply(cand$families, function(f) {
    list(id = f$id, attributes = unlist(f$attributes))
  }))

  artifact_path <- file.path(root, sim$context_artifact$output)
  cmp_assert(file.exists(artifact_path),
             "The version-4 candidate context artifact has not been built.")
  population <- readRDS(artifact_path)
  cmp_assert(identical(names(population), c("id", "probability", CMP_RATING_CONTEXT)),
             "Unexpected context artifact columns.")
  cmp_assert(abs(sum(population$probability) - 1) < 1e-12, "Context probabilities must sum to one.")

  ctx_key <- do.call(paste0, population[CMP_RATING_CONTEXT])
  contexts <- population[!duplicated(ctx_key), CMP_RATING_CONTEXT]
  contexts$key <- ctx_key[!duplicated(ctx_key)]
  contexts$probability <- as.numeric(tapply(population$probability, ctx_key, sum)[contexts$key])
  contexts <- contexts[order(contexts$key), ]
  rownames(contexts) <- NULL

  geom <- list(
    name = "rating", factors = factors, levels = levels, labels = labels,
    grid = grid, bases = bases, families = families,
    coordinates = cmp_coordinate_table(families),
    tasks = as.integer(sim$tasks_per_respondent),
    respondents = as.integer(sim$respondents),
    population = population, contexts = contexts,
    context_names = CMP_RATING_CONTEXT,
    has_opponent = FALSE, sides = 1L,
    assignment_probability = 1 / nrow(grid),
    generator = sim$outcome_generator, components = sim$fixed_components,
    shift = sim$target_shift
  )
  geom$score <- cmp_rating_level_scores(levels, curvature = FALSE)
  geom$curve <- cmp_rating_level_scores(levels, curvature = TRUE)
  ctx <- as.matrix(contexts[CMP_RATING_CONTEXT])
  geom$context_mean <- colSums(ctx * contexts$probability)
  geom$nonlinear_mean <- colSums(cmp_rating_nonlinear(ctx) * contexts$probability)
  geom
}

cmp_rating_level_scores <- function(levels, curvature) {
  lapply(levels, function(n) {
    s <- seq(-1, 1, length.out = n)
    s <- s - mean(s)
    if (curvature) {
      s <- s^2 - mean(s^2)
      if (sqrt(mean(s^2)) < 1e-12) return(rep(0, n))
    }
    s / sqrt(mean(s^2))
  })
}

cmp_rating_nonlinear <- function(ctx) {
  cbind(
    xor_woman_young = as.numeric(ctx[, "woman"] != ctx[, "young"]),
    poc_times_workingclass = ctx[, "poc"] * ctx[, "workingclass"],
    dem_times_young = ctx[, "dem"] * ctx[, "young"],
    nonrep_times_poc_times_young = (1 - ctx[, "rep"]) * ctx[, "poc"] * ctx[, "young"]
  )
}

# Component matrix for rows given focal codes (n x 7), task (n), and context
# (n x 6, 0/1). Every component is centered under the uniform profile law,
# uniform tasks, and the population context law.
cmp_rating_components <- function(geom, codes, task, ctx) {
  fc <- geom$components
  s <- function(f) geom$score[[f]][codes[, f]]
  cv <- function(f) geom$curve[[f]][codes[, f]]
  w <- sweep(ctx[, geom$context_names, drop = FALSE], 2L, geom$context_mean, "-")
  nl <- sweep(cmp_rating_nonlinear(ctx), 2L, geom$nonlinear_mean, "-")
  main <- 0
  for (f in names(fc$main_coefficients)) main <- main + fc$main_coefficients[[f]] * s(f)
  lin <- as.numeric(w %*% unlist(fc$linear_context_coefficients)[geom$context_names])
  pi <- fc$fixed_profile_interactions
  pairs <- pi$authority_x_majority * s("authority") * s("majority") +
    pi$authority_x_staff * s("authority") * s("staff") +
    pi$fundraising_x_winning * s("fundraising") * s("winning") +
    pi$fundraising_curvature_x_pay * cv("fundraising") * s("pay")
  li <- fc$linear_profile_context_interactions
  lin_het <- li$fundraising_x_woman * s("fundraising") * w[, "woman"] +
    li$winning_x_poc * s("winning") * w[, "poc"] +
    li$authority_x_dem * s("authority") * w[, "dem"] +
    li$staff_x_workingclass * s("staff") * w[, "workingclass"]
  nl_prog <- as.numeric(nl %*% unlist(fc$nonlinear_context_coefficients))
  ni <- fc$nonlinear_profile_context_coefficients
  nl_het <- ni$fundraising_x_xor_woman_young * s("fundraising") * nl[, "xor_woman_young"] +
    ni$winning_curvature_x_poc_workingclass * cv("winning") * nl[, "poc_times_workingclass"] +
    ni$authority_x_dem_young * s("authority") * nl[, "dem_times_young"] +
    ni$media_x_nonrep_poc_young * s("media") * nl[, "nonrep_times_poc_times_young"]
  cbind(
    main = main,
    task = fc$task_slope_centered_at_4_5 * (task - 4.5),
    linear_prognostic = lin,
    fixed_pairs = pairs,
    linear_heterogeneity = lin_het,
    nonlinear_prognostic = nl_prog,
    nonlinear_heterogeneity = nl_het
  )
}

# Component weights for each rating cell. prognostic_x levels are cumulative
# (none < linear < nonlinear), matching the developmental generator.
cmp_rating_cell_weights <- function(cell, multiplier = NULL) {
  w <- c(main = 1, task = 1, linear_prognostic = 0, fixed_pairs = 0,
         linear_heterogeneity = 0, nonlinear_prognostic = 0, nonlinear_heterogeneity = 0)
  if (cell$profile_surface == "additive_plus_fixed_pairs") w[["fixed_pairs"]] <- 1
  if (cell$prognostic_x %in% c("linear", "nonlinear")) w[["linear_prognostic"]] <- 1
  if (cell$prognostic_x == "nonlinear") w[["nonlinear_prognostic"]] <- 1
  het <- cell$treatment_heterogeneity
  if (het == "linear") w[["linear_heterogeneity"]] <- 1
  if (het == "nonlinear") w[["nonlinear_heterogeneity"]] <- 1
  if (het == "nonlinear_calibrated_multiplier") {
    cmp_assert(!is.null(multiplier), "Cell %s needs a heterogeneity multiplier.", cell$name)
    w[["nonlinear_heterogeneity"]] <- multiplier
  }
  w
}

# Target-to-assignment ratio for every profile (1 unless the cell shifts the target).
cmp_rating_target_ratio <- function(geom, cell, codes = geom$grid) {
  if (!identical(cell$target, "shifted_max_ratio_1_5")) return(rep(1, nrow(codes)))
  ratio <- as.numeric(unlist(geom$shift$mu_over_q_by_locked_level_order))
  cmp_assert(length(ratio) == geom$levels[[geom$shift$attribute]],
             "Target-shift ratios do not match the shifted attribute.")
  ratio[codes[, geom$shift$attribute]]
}

# Build the response surface for one cell: intercept, validity, exact truth and
# exact per-context oracle plugins.
cmp_rating_surface <- function(geom, cell, multiplier = NULL) {
  weights <- cmp_rating_cell_weights(cell, multiplier)
  gen <- geom$generator
  n_prof <- nrow(geom$grid)
  tasks <- seq_len(geom$tasks)
  ratio <- cmp_rating_target_ratio(geom, cell)
  mu <- ratio / n_prof

  # Signal without intercept on the full (profile x task) grid for each context.
  signal_by_context <- lapply(seq_len(nrow(geom$contexts)), function(i) {
    ctx <- as.matrix(geom$contexts[rep(i, n_prof * geom$tasks), geom$context_names])
    codes <- geom$grid[rep(seq_len(n_prof), times = geom$tasks), , drop = FALSE]
    task <- rep(tasks, each = n_prof)
    comp <- cmp_rating_components(geom, codes, task, ctx)
    matrix(gen$signal_scale * as.numeric(comp %*% weights[colnames(comp)]),
           n_prof, geom$tasks)
  })
  # Intercept under the assignment law (uniform profiles and tasks, population
  # contexts) so the target-shift cell shares cell 5's response surface.
  mean_signal <- sum(vapply(seq_along(signal_by_context), function(i) {
    geom$contexts$probability[[i]] * mean(signal_by_context[[i]])
  }, numeric(1)))
  intercept <- gen$target_mean - mean_signal

  range_m <- range(unlist(signal_by_context)) + intercept
  valid <- range_m[[1]] > 0 && range_m[[2]] < 1

  plugins <- matrix(NA_real_, nrow(geom$contexts), nrow(geom$coordinates),
                    dimnames = list(geom$contexts$key, geom$coordinates$key))
  for (i in seq_along(signal_by_context)) {
    profile_mean <- rowMeans(signal_by_context[[i]]) + intercept
    plugins[i, ] <- cmp_theta_from_profiles(geom, profile_mean, mu)
  }
  truth <- as.numeric(crossprod(geom$contexts$probability, plugins))
  names(truth) <- geom$coordinates$key
  list(cell = cell, multiplier = multiplier, weights = weights, intercept = intercept,
       range = range_m, valid = valid, truth = truth, oracle_plugins = plugins,
       target_ratio = ratio)
}

# theta = sum_p mu(p) psi(a(p), b(p)) m(p) for every registered family, where
# m(p) is already averaged over tasks. mu must sum to one.
cmp_theta_from_profiles <- function(geom, profile_values, mu) {
  out <- lapply(geom$families, function(f) {
    psi <- cmp_family_psi_rows(f, geom$grid[, f$a], geom$grid[, f$b])
    as.numeric(crossprod(psi, mu * profile_values))
  })
  unlist(out, use.names = FALSE)
}

cmp_rating_mean <- function(geom, surface, codes, task, ctx) {
  comp <- cmp_rating_components(geom, codes, task, ctx)
  surface$intercept + geom$generator$signal_scale *
    as.numeric(comp %*% surface$weights[colnames(comp)])
}

cmp_rating_simulate <- function(geom, surface, seed_data) {
  gen <- geom$generator
  g <- geom$respondents
  tt <- geom$tasks
  n_prof <- nrow(geom$grid)
  draws <- cmp_with_seed(seed_data, {
    who <- sample.int(nrow(geom$population), g, replace = TRUE,
                      prob = geom$population$probability)
    profiles <- as.vector(vapply(seq_len(g), function(i) {
      sample.int(n_prof, tt, replace = FALSE)
    }, integer(tt)))
    shared <- stats::rnorm(g)
    idio <- stats::rnorm(g * tt)
    u_binom <- stats::runif(g * tt)
    list(who = who, profiles = profiles, shared = shared, idio = idio, u_binom = u_binom)
  })
  ids <- sprintf("r%04d", seq_len(g))
  ctx <- as.matrix(geom$population[draws$who, geom$context_names])
  rows <- data.frame(
    id = rep(ids, each = tt),
    task = rep(seq_len(tt), times = g),
    side = 1L,
    stringsAsFactors = FALSE
  )
  codes <- geom$grid[draws$profiles, , drop = FALSE]
  ctx_rows <- ctx[rep(seq_len(g), each = tt), , drop = FALSE]
  m <- cmp_rating_mean(geom, surface, codes, rows$task, ctx_rows)
  cmp_assert(all(m > 0 & m < 1), "A rating conditional mean left the open unit interval.")
  rho <- gen$respondent_gaussian_copula_rho
  z <- sqrt(rho) * rep(draws$shared, each = tt) + sqrt(1 - rho) * draws$idio
  prec <- gen$beta_precision
  p <- stats::qbeta(stats::pnorm(z), m * prec, (1 - m) * prec)
  y <- stats::qbinom(draws$u_binom, gen$intervals, p) / gen$intervals
  data <- cbind(rows, as.data.frame(codes), as.data.frame(ctx_rows))
  data$profile_index <- draws$profiles
  data$y <- y
  data$true_mean <- m
  data$target_ratio <- surface$target_ratio[draws$profiles]
  data$context_key <- do.call(paste0, as.data.frame(ctx_rows))
  respondents <- data.frame(id = ids, weight = 1, context_key = do.call(paste0, as.data.frame(ctx)),
                            stringsAsFactors = FALSE)
  respondents <- cbind(respondents, as.data.frame(ctx))
  list(rows = data, respondents = respondents)
}
