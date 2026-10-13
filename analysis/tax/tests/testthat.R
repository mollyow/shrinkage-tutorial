#!/usr/bin/env Rscript

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_all, value = TRUE)
script <- if (length(file_arg)) {
  normalizePath(sub("^--file=", "", file_arg[[1]]), mustWork = TRUE)
} else {
  NA_character_
}
repo_root <- if (!is.na(script)) {
  normalizePath(file.path(dirname(script), "..", "..", ".."), mustWork = TRUE)
} else {
  normalizePath(getwd())
}
setwd(repo_root)
Sys.setenv(TAX_AUDIT_REPO_ROOT = repo_root)

if (!requireNamespace("testthat", quietly = TRUE)) {
  stop("Package 'testthat' is required", call. = FALSE)
}
testthat::test_dir("analysis/tax/tests/testthat", reporter = "summary")
