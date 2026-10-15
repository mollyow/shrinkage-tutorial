# Gate C1: phase execution, collection, and the Gate D rules.

cmp_fake_fm <- function(spec) {
  # spec: data frame geometry, multiplier, method, risk; two families, 4 reps.
  do.call(rbind, lapply(seq_len(nrow(spec)), function(i) {
    expand.grid(geometry = spec$geometry[[i]], multiplier = spec$multiplier[[i]],
                method = spec$method[[i]], family = c("f1", "f2"), replication = 1:4,
                stringsAsFactors = FALSE) |>
      transform(normalized_risk = spec$risk[[i]])
  }))
}

testthat::test_that("stage-1 selection follows the frozen rule", {
  pc <- cmp_test_engine()$contracts$simulation$positive_control_calibration
  spec <- data.frame(
    geometry = rep(c("rating", "choice"), each = 8),
    multiplier = rep(rep(c(0.5, 1, 1.5, 2), each = 2), 2),
    method = rep(c("additive", "oracle"), 8),
    risk = c(1, 0.95, 1, 0.80, 1, 0.66, 1, 0.74,   # rating: 0.66 and 0.74 tie at 0.04
             1, 0.99, 1, 0.97, 1, 0.90, 1, 0.80),  # choice: nothing <= 0.75
    stringsAsFactors = FALSE)
  sel <- cmp_gate_d_selection(cmp_fake_fm(spec), pc)
  testthat::expect_equal(sel$selected$rating, 1.5)   # tie broken toward the smaller multiplier
  testthat::expect_true(is.na(sel$selected$choice))
  testthat::expect_equal(unique(sel$table$replications), 4L)
})

testthat::test_that("stage-2 recovery requires both geometries and the same datasets", {
  pc <- cmp_test_engine()$contracts$simulation$positive_control_calibration
  s1 <- data.frame(geometry = rep(c("rating", "choice"), each = 2), multiplier = 2,
                   method = rep(c("additive", "oracle"), 2), risk = c(1, 0.7, 1, 0.72),
                   stringsAsFactors = FALSE)
  s2 <- data.frame(geometry = rep(c("rating", "choice"), each = 3), multiplier = 2,
                   method = rep(c("elastic_net", "ranger", "xgboost"), 2),
                   risk = c(0.99, 0.84, 0.95, 0.98, 0.97, 0.90), stringsAsFactors = FALSE)
  dec <- cmp_gate_d_recovery(cmp_fake_fm(s1), cmp_fake_fm(s2), list(rating = 2, choice = 2), pc)
  testthat::expect_equal(dec$geometries$rating$gain_recovered$ranger, (1 - 0.84) / 0.3)
  testthat::expect_true(dec$geometries$rating$passed)
  testthat::expect_false(dec$geometries$choice$passed)   # best recovery 0.10 / 0.28 < 0.5
  testthat::expect_false(dec$passed)
  dec_na <- cmp_gate_d_recovery(cmp_fake_fm(s1), cmp_fake_fm(s2), list(rating = 2, choice = NA), pc)
  testthat::expect_false(dec_na$passed)
  short <- cmp_fake_fm(s2)
  short <- short[short$replication != 4, ]
  testthat::expect_error(cmp_gate_d_recovery(cmp_fake_fm(s1), short, list(rating = 2, choice = 2), pc),
                         "stage-1 calibration datasets")
})

testthat::test_that("invalid generators are filtered and recorded", {
  eng <- cmp_test_engine()
  sim <- eng$contracts$simulation
  jobs <- cmp_calibration_registry(sim, 1L, draws = 16L)
  f <- cmp_filter_valid(eng, jobs)
  bad <- vapply(f$invalid, function(x) paste(x$geometry, x$multiplier), "")
  testthat::expect_setequal(bad, c("rating 3", "rating 4"))
  testthat::expect_length(f$jobs, length(jobs) - 2L * 100L)
})

testthat::test_that("jobs checkpoint, resume, and collect into the output schemas", {
  eng <- cmp_test_engine()
  eng$geoms$rating$respondents <- 40L
  dir <- withr::local_tempdir()
  jobs <- lapply(1:2, function(r) {
    j <- list(phase = "test", geometry = "rating", cell_id = 2L, multiplier = NULL,
              replication = r, methods = c("additive", "oracle"))
    j$path <- cmp_job_path(dir, "test", 0L, j)
    j
  })
  status <- vapply(jobs, function(j) cmp_run_job(eng, j), "")
  testthat::expect_false(any(grepl("^ERROR", status)))
  testthat::expect_true(all(file.exists(vapply(jobs, `[[`, "", "path"))))
  written <- cmp_collect(jobs, file.path(dir, "collected"))
  testthat::expect_length(written, length(CMP_SCHEMAS))
  fm <- utils::read.csv(file.path(dir, "collected", "family_metrics", "rating__2__fixed__part_01.csv"))
  testthat::expect_equal(sort(unique(fm$replication)), 1:2)
  bad <- cmp_run_job(eng, modifyList(jobs[[1]], list(methods = "additive", cell_id = 99L)))
  testthat::expect_match(bad, "^ERROR")
})
