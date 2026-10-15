# Run from the repository root:
#   Rscript analysis/comparison/tests/testthat.R
library(testthat)
testthat::test_dir(
  file.path("analysis", "comparison", "tests", "testthat"),
  reporter = testthat::ProgressReporter$new(show_praise = FALSE),
  stop_on_failure = TRUE
)
