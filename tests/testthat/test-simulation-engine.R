.simulation_project_file <- function(...) {
  relative <- file.path(...)
  candidates <- c(relative, file.path("..", "..", relative))
  found <- candidates[file.exists(candidates)][1L]
  if (is.na(found)) {
    testthat::skip(paste0(
      "Simulation integration tests require the source checkout; missing: ",
      relative
    ))
  }
  normalizePath(found, mustWork = TRUE)
}

# Keep this file runnable both through devtools::test() and directly with
# testthat::test_file().
if (!exists("factorial_basis", mode = "function")) {
  source(.simulation_project_file("R", "factorial_basis.R"))
}
if (!exists("cluster_folds", mode = "function")) {
  source(.simulation_project_file("R", "cluster_folds.R"))
}
if (!exists("fit_order_ridge", mode = "function")) {
  source(.simulation_project_file("R", "order_ridge.R"))
}

.load_simulation_functions <- function() {
  env <- new.env(parent = parent.frame())
  sys.source(
    .simulation_project_file("simulation", "simulation_functions.R"),
    envir = env
  )
  env
}

.tiny_scenario <- function(scenario_id = 1L) {
  scenario <- data.frame(
    scenario_id = as.integer(scenario_id),
    design = "tiny_binary2_order2",
    number_factors = 2L,
    number_levels = 2L,
    max_order = 2L,
    n_over_p = 0.5,
    snr = 0.5,
    scale = "steep",
    within_order = "dense",
    stringsAsFactors = FALSE
  )
  scenario$order_scales <- list(c(1, 0.25))
  scenario
}

.tiny_config <- function() {
  list(
    seed = 17L,
    folds = 2L,
    tuning = list(
      common_grid_size = 3L,
      maxit = 2L,
      log_lambda_lower = -4,
      log_lambda_upper = 3
    ),
    oracle_maxit = 2L
  )
}

.gate_fixture <- function(replications = 2L, include_misspecification = TRUE,
                          scales = NULL) {
  if (is.null(scales)) {
    scales <- c("steep", "mild")
    if (include_misspecification) {
      scales <- c(scales, "flat", "mild_reversal")
    }
  }
  rows <- list()
  cursor <- 0L
  for (scenario_id in seq_along(scales)) {
    ratio <- if (scales[scenario_id] %in% c("steep", "mild")) 0.90 else 1.02
    for (replication in seq_len(replications)) {
      cursor <- cursor + 1L
      rows[[cursor]] <- data.frame(
        scenario_id = scenario_id,
        replication = replication,
        status = "ok",
        method = c("monotone_order_ridge", "unrestricted_order_ridge"),
        risk = c(ratio, 1),
        design = "tiny",
        n_over_p = 1,
        snr = 1,
        scale = scales[scenario_id],
        within_order = "dense",
        stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, rows)
}

.tiny_gate_config <- function(scales = c("steep", "mild", "flat", "mild_reversal")) {
  testthat::skip_if_not_installed("yaml")
  path <- tempfile(fileext = ".yml")
  order_scales <- stats::setNames(
    lapply(scales, function(scale) c(1, if (scale == "steep") 0.25 else 1)),
    scales
  )
  yaml::write_yaml(list(
    mode = "production", seed = 17L, replications = 2L, folds = 2L,
    designs = list(list(name = "tiny", factors = 2L, levels = 2L, max_order = 2L)),
    n_over_p = 1, snr = 1, order_scales = order_scales,
    within_order = "dense"
  ), path)
  path
}

.run_gate_summary <- function(data, config = NULL) {
  testthat::skip_if_not_installed("jsonlite")
  input <- tempfile(fileext = ".csv")
  output <- tempfile(fileext = ".json")
  on.exit(unlink(c(input, output)), add = TRUE)
  utils::write.csv(data, input, row.names = FALSE)
  script <- .simulation_project_file("simulation", "summarize_gates.R")
  if (is.null(config)) {
    config <- .simulation_project_file("config", "simulation-production.yml")
  }
  old_working_directory <- setwd(dirname(dirname(script)))
  on.exit(setwd(old_working_directory), add = TRUE)
  log <- suppressWarnings(system2(
    file.path(R.home("bin"), "Rscript"),
    c(script, input, output, config),
    stdout = TRUE,
    stderr = TRUE
  ))
  status <- attr(log, "status")
  if (is.null(status)) status <- 0L
  json <- if (file.exists(output)) {
    jsonlite::read_json(output, simplifyVector = FALSE)
  } else {
    NULL
  }
  list(status = as.integer(status), json = json, log = log)
}

.expect_gate_rejection <- function(result) {
  if (result$status != 0L) {
    testthat::succeed()
  } else {
    testthat::expect_false(is.null(result$json))
    testthat::expect_identical(result$json$full_methods_gate_passed, FALSE)
    criteria <- unlist(result$json$criteria, use.names = FALSE)
    testthat::expect_true(length(criteria) > 0L)
    testthat::expect_true(all(vapply(criteria, is.logical, logical(1))))
    testthat::expect_false(anyNA(criteria))
  }
}

testthat::test_that("configured factorial geometries have exactly 92 and 232 df", {
  sim <- .load_simulation_functions()

  binary <- sim$make_profile_grid(8L, 2L)
  binary_basis <- factorial_basis(
    binary, factors = names(binary), target = binary,
    target_weights = rep(1 / nrow(binary), nrow(binary)), max_order = 3L
  )
  testthat::expect_identical(unname(binary_basis$df_by_order), c(8L, 28L, 56L))
  testthat::expect_equal(length(binary_basis$order), 92L)

  ternary <- sim$make_profile_grid(6L, 3L)
  ternary_basis <- factorial_basis(
    ternary, factors = names(ternary), target = ternary,
    target_weights = rep(1 / nrow(ternary), nrow(ternary)), max_order = 3L
  )
  testthat::expect_identical(unname(ternary_basis$df_by_order), c(12L, 60L, 160L))
  testthat::expect_equal(length(ternary_basis$order), 232L)
})

testthat::test_that("fast and production configs expand deterministically", {
  testthat::skip_if_not_installed("yaml")
  sim <- .load_simulation_functions()
  fast <- sim$read_simulation_config(
    .simulation_project_file("config", "simulation-fast.yml")
  )
  production <- sim$read_simulation_config(
    .simulation_project_file("config", "simulation-production.yml")
  )
  fast_scenarios <- sim$expand_simulation_scenarios(fast)
  production_scenarios <- sim$expand_simulation_scenarios(production)

  testthat::expect_equal(nrow(fast_scenarios), 60L)
  testthat::expect_equal(nrow(production_scenarios), 120L)
  testthat::expect_identical(fast_scenarios, sim$expand_simulation_scenarios(fast))
  testthat::expect_false(anyDuplicated(fast_scenarios$scenario_id) > 0L)
  testthat::expect_equal(production$replications, 250L)
  testthat::expect_equal(production$headline_replications, 1000L)
  fast_counts <- vapply(seq_len(nrow(fast_scenarios)), function(i) {
    sim$scenario_replications(fast_scenarios[i, , drop = FALSE], fast)
  }, integer(1))
  production_counts <- vapply(seq_len(nrow(production_scenarios)), function(i) {
    sim$scenario_replications(production_scenarios[i, , drop = FALSE], production)
  }, integer(1))
  testthat::expect_true(all(fast_counts == 2L))
  testthat::expect_equal(sum(production_counts == 1000L), 8L)
  testthat::expect_equal(sum(production_counts == 250L), 112L)
})

testthat::test_that("target risk is the exact target-weighted finite-profile risk", {
  sim <- .load_simulation_functions()
  grid <- sim$make_profile_grid(2L, 2L)
  weights <- c(0.70, 0.10, 0.10, 0.10)
  basis <- factorial_basis(
    grid, factors = names(grid), target = grid,
    target_weights = weights, max_order = 2L
  )
  truth <- c(-1, 0, 1, 2)
  prediction <- truth + c(1, -1, 2, 0.5)
  beta <- seq_along(basis$order) / 10
  fit <- list(
    prediction = prediction,
    lambda = c(1, 2),
    coefficients = beta + 0.05
  )
  sim$predict <- function(object, newdata, ...) object$prediction
  scenario <- data.frame(
    scenario_id = 1L, design = "tiny", n_over_p = 1, snr = 1,
    scale = "steep", within_order = "dense", stringsAsFactors = FALSE
  )
  evaluated <- sim$evaluate_simulation_fit(
    list(status = "ok", fit = fit, error = NA_character_),
    "test", truth, beta, basis, scenario, replication = 1L, seed = 10L
  )

  expected <- sum(basis$target_weights * (prediction - truth)^2)
  testthat::expect_equal(evaluated$risk, expected, tolerance = 1e-14)
})

testthat::test_that("replications are deterministic, draw fresh truths, and reuse folds", {
  sim <- .load_simulation_functions()
  scenario <- .tiny_scenario(3L)
  config <- .tiny_config()

  original_draw <- sim$draw_surface
  captured_surfaces <- list()
  sim$draw_surface <- function(...) {
    value <- original_draw(...)
    captured_surfaces[[length(captured_surfaces) + 1L]] <<- value
    value
  }

  original_fit <- get("fit_order_ridge", envir = parent.env(sim), inherits = TRUE)
  tuned_calls <- list()
  sim$fit_order_ridge <- function(..., penalty_structure = NULL, lambda = NULL,
                                  fold_id = NULL) {
    if (is.null(lambda)) {
      tuned_calls[[length(tuned_calls) + 1L]] <<- list(
        penalty_structure = penalty_structure,
        fold_id = fold_id
      )
    }
    original_fit(
      ..., penalty_structure = penalty_structure, lambda = lambda,
      fold_id = fold_id
    )
  }

  first <- sim$run_simulation_replication(scenario, 4L, config)
  first_surface <- captured_surfaces[[1L]]
  first_calls <- tuned_calls
  testthat::expect_setequal(
    vapply(first_calls, `[[`, character(1), "penalty_structure"),
    c("common", "unrestricted", "monotone")
  )
  testthat::expect_equal(length(first_calls), 3L)
  reference_folds <- first_calls[[1L]]$fold_id
  testthat::expect_true(all(vapply(first_calls, function(x) {
    identical(x$fold_id, reference_folds)
  }, logical(1))))

  captured_surfaces <- list()
  tuned_calls <- list()
  stats::runif(10L)
  second <- sim$run_simulation_replication(scenario, 4L, config)
  second_surface <- captured_surfaces[[1L]]
  testthat::expect_identical(first, second)
  testthat::expect_identical(first_surface, second_surface)
  expected_seed <- as.integer(config$seed + scenario$scenario_id * 100000L + 4L)
  testthat::expect_identical(unique(first$seed), expected_seed)

  captured_surfaces <- list()
  tuned_calls <- list()
  third <- sim$run_simulation_replication(scenario, 5L, config)
  third_surface <- captured_surfaces[[1L]]
  testthat::expect_false(isTRUE(all.equal(first_surface, third_surface)))
  testthat::expect_false(identical(unique(first$seed), unique(third$seed)))
  testthat::expect_equal(sum(first_surface^2), scenario$snr, tolerance = 1e-12)
  testthat::expect_equal(sum(third_surface^2), scenario$snr, tolerance = 1e-12)
})

testthat::test_that("gate summary computes the preregistered ratios", {
  result <- .run_gate_summary(.gate_fixture(replications = 2L))
  testthat::expect_equal(result$status, 0L)
  testthat::expect_false(is.null(result$json))
  testthat::expect_equal(result$json$primary$geometric_mean_ratio, 0.90,
                         tolerance = 1e-12)
  testthat::expect_equal(result$json$robustness$geometric_mean_ratio, 1.02,
                         tolerance = 1e-12)
  testthat::expect_equal(result$json$fraction_primary_cells_won, 1)
  testthat::expect_equal(result$json$maximum_misspec_cell_ratio, 1.02,
                         tolerance = 1e-12)
  testthat::expect_true(all(unlist(result$json$criteria, use.names = FALSE)))
})

testthat::test_that("a complete small production configuration can pass", {
  config <- .tiny_gate_config()
  on.exit(unlink(config), add = TRUE)
  result <- .run_gate_summary(.gate_fixture(replications = 2L), config)

  testthat::expect_equal(result$status, 0L)
  testthat::expect_true(result$json$eligible_for_methods_verdict)
  testthat::expect_true(result$json$full_methods_gate_passed)
  testthat::expect_match(result$json$metric_definition, "exp\\(mean\\(log")
})

testthat::test_that("missing stress cells and metadata mismatches block eligibility", {
  scales <- c("steep", "mild", "flat", "mild_reversal", "interaction_dominant")
  config <- .tiny_gate_config(scales)
  on.exit(unlink(config), add = TRUE)

  missing_stress <- .run_gate_summary(
    .gate_fixture(replications = 2L), config
  )
  testthat::expect_false(missing_stress$json$eligible_for_methods_verdict)
  testthat::expect_false(missing_stress$json$full_methods_gate_passed)

  complete <- .gate_fixture(replications = 2L, scales = scales)
  complete$design[complete$scenario_id == 5L] <- "wrong_design"
  mismatch <- .run_gate_summary(complete, config)
  testthat::expect_false(mismatch$json$eligible_for_methods_verdict)
  testthat::expect_true(any(grepl("metadata mismatch", mismatch$json$issues)))
})

testthat::test_that("aggregate gate risk gives each scenario cell equal weight", {
  unequal <- .gate_fixture(replications = 4L)
  unequal$risk[
    unequal$scenario_id == 1L & unequal$method == "monotone_order_ridge"
  ] <- 0.50
  unequal$risk[
    unequal$scenario_id == 2L & unequal$method == "monotone_order_ridge"
  ] <- 1.00
  unequal <- unequal[!(
    unequal$scenario_id == 2L & unequal$replication > 1L
  ), , drop = FALSE]

  result <- .run_gate_summary(unequal)
  testthat::expect_equal(result$status, 0L)
  # Cell ratios are 0.50 and 1.00.  Replication counts must not change their
  # equal-cell geometric mean.
  testthat::expect_equal(
    result$json$primary$geometric_mean_ratio, sqrt(0.50), tolerance = 1e-4
  )
})

testthat::test_that("gate cannot pass with one replication per cell", {
  .expect_gate_rejection(.run_gate_summary(.gate_fixture(replications = 1L)))
})

testthat::test_that("gate rejects missing pairs, duplicates, and undefined ratios", {
  missing_pair <- .gate_fixture(replications = 2L)
  missing_pair <- missing_pair[-which(
    missing_pair$scenario_id == 1L & missing_pair$replication == 1L &
      missing_pair$method == "unrestricted_order_ridge"
  )[1L], , drop = FALSE]
  .expect_gate_rejection(.run_gate_summary(missing_pair))

  duplicated_pair <- .gate_fixture(replications = 2L)
  duplicated_pair <- rbind(duplicated_pair, duplicated_pair[1L, , drop = FALSE])
  .expect_gate_rejection(.run_gate_summary(duplicated_pair))

  zero_risk <- .gate_fixture(replications = 2L)
  zero_risk$risk <- 0
  .expect_gate_rejection(.run_gate_summary(zero_risk))

  missing_cells <- .gate_fixture(replications = 2L, include_misspecification = FALSE)
  .expect_gate_rejection(.run_gate_summary(missing_cells))
})
