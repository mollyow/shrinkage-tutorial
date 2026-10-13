#!/usr/bin/env Rscript

if (!requireNamespace("testthat", quietly = TRUE)) {
  stop("Install testthat to run the candidate application contract tests.",
       call. = FALSE)
}

script_flag <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_flag)) stop("Run this file with Rscript.", call. = FALSE)
script <- normalizePath(sub("^--file=", "", script_flag[[1L]]), mustWork = TRUE)
repo_root <- normalizePath(file.path(dirname(script), "..", "..", ".."),
                           mustWork = TRUE)
Sys.setenv(CANDIDATE_REPO_ROOT = repo_root)

testthat::test_dir(file.path(dirname(script), "testthat"),
                   reporter = testthat::ProgressReporter$new())
