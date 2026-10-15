# Gate C1: complete expected jobs and estimands, output schemas, failure
# handling, parquet round trips, and a synthetic report render.

cmp_tiny_engine <- function() {
  eng <- cmp_test_engine()
  eng$geoms$rating$respondents <- 60L
  eng$geoms$choice$respondents <- 40L
  eng
}

testthat::test_that("datasets emit schema-complete tables with every expected key", {
  eng <- cmp_tiny_engine()
  for (cell_id in c(2L, 8L)) {
    methods <- c("unadjusted", "task_only", "additive", "oracle")
    res <- cmp_run_dataset(eng, list(phase = "test", cell_id = cell_id, replication = 3L,
                                     multiplier = NULL, methods = methods, draws = 16L))
    res <- cmp_conform_dataset(res)
    geom <- eng$geoms[[cmp_cell(eng, cell_id)$geometry]]
    testthat::expect_silent(cmp_check_complete(res, geom, methods, 5L))
    for (name in names(CMP_SCHEMAS)) {
      testthat::expect_identical(names(res[[name]]), names(CMP_SCHEMAS[[name]]))
    }
    testthat::expect_true(all(res$job_ledger$status == "ok"))
    testthat::expect_equal(nrow(res$failures), 0L)
  }
})

testthat::test_that("a failed method yields NA results without borrowing another method", {
  eng <- cmp_tiny_engine()
  methods <- c("additive", "deliberately_unknown_method")
  res <- cmp_conform_dataset(cmp_run_dataset(eng, list(
    phase = "test", cell_id = 1L, replication = 4L, multiplier = NULL, methods = methods)))
  testthat::expect_silent(cmp_check_complete(res, eng$geoms$rating, methods, 5L))
  bad <- res$effect_estimates[res$effect_estimates$method == "deliberately_unknown_method", ]
  testthat::expect_true(all(is.na(bad$estimate)))
  testthat::expect_true(all(res$job_ledger$status[res$job_ledger$method != "additive"] == "failed"))
  testthat::expect_equal(nrow(res$failures), 5L)
  good <- res$effect_estimates[res$effect_estimates$method == "additive", ]
  testthat::expect_true(all(is.finite(good$estimate)))
  p <- res$fold_predictions[res$fold_predictions$method == "deliberately_unknown_method", ]
  testthat::expect_true(all(is.na(p$prediction)))
})

testthat::test_that("tables survive a parquet round trip and the report renders", {
  eng <- cmp_tiny_engine()
  res <- cmp_conform_dataset(cmp_run_dataset(eng, list(
    phase = "test", cell_id = 3L, replication = 1L, multiplier = NULL,
    methods = c("additive", "oracle", "unadjusted"))))
  dir <- withr::local_tempdir()
  for (name in c("fold_predictions", "respondent_scores", "tuning_history")) {
    path <- file.path(dir, paste0(name, ".parquet"))
    cmp_write_parquet(res[[name]], path)
    back <- as.data.frame(nanoparquet::read_parquet(path))
    testthat::expect_equal(nrow(back), nrow(res[[name]]))
    testthat::expect_identical(names(back), names(res[[name]]))
  }
  reps <- lapply(1:3, function(r) cmp_conform_dataset(cmp_run_dataset(eng, list(
    phase = "test", cell_id = 3L, replication = r, multiplier = NULL,
    methods = c("additive", "oracle", "unadjusted")))))
  summ <- cmp_cell_summary(reps, eng$contracts, 1L)
  testthat::expect_true(is.finite(summ$headroom))
  path <- file.path(dir, "report.md")
  cmp_render_report(path, "Synthetic report", "Test render.",
                    list(list(title = "Cell 3", summary = summ)))
  txt <- readLines(path)
  testthat::expect_true(any(grepl("^\\| method \\|", txt)))
})

testthat::test_that("phases run only after the gates before them pass", {
  cmp_test_engine()
  dir <- withr::local_tempdir()
  write_gate <- function(gate, value) {
    path <- file.path(dir, CMP_GATE_PATHS[[gate]])
    dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
    field <- if (gate == "gate_d") "gate_d_passed" else "all_passed"
    yaml::write_yaml(stats::setNames(list(value), field), path)
  }
  testthat::expect_false(cmp_phase_authorization(dir, "calibration"))
  write_gate("c1", FALSE)
  testthat::expect_false(cmp_phase_authorization(dir, "calibration"))
  write_gate("c1", TRUE)
  testthat::expect_true(cmp_phase_authorization(dir, "calibration"))
  testthat::expect_false(cmp_phase_authorization(dir, "smoke"))
  write_gate("gate_d", TRUE)
  testthat::expect_true(cmp_phase_authorization(dir, "smoke"))
  testthat::expect_false(cmp_phase_authorization(dir, "production"))
  write_gate("c2", TRUE)
  testthat::expect_true(cmp_phase_authorization(dir, "production"))
  testthat::expect_error(cmp_require_phase(file.path(dir, "none"), "calibration"), "not authorized")
})

testthat::test_that("the seed rule and registries match the frozen schedule", {
  eng <- cmp_test_engine()
  sim <- eng$contracts$simulation
  testthat::expect_identical(cmp_seed(sim, "production", 10L, 2000L, "bootstrap"),
                             as.integer(260903000 + 20000000 + 1000000 + 2000 + 50000))
  chk <- cmp_seed_collision_check(sim)
  testthat::expect_true(chk$passed)
  testthat::expect_true(chk$production_matches_contract)
  reg <- cmp_phase_registry(sim, "production", "additive", 16L, list(rating = 1, choice = 1))
  testthat::expect_length(reg, sim$replication_schedule$production$total_production_datasets)
  cal <- cmp_calibration_registry(sim, 1L, draws = 16L)
  testthat::expect_length(cal, 2L * 7L * 100L)
  testthat::expect_error(cmp_seed(sim, "production", 1L, 10000L, "data"), "outside")
})
