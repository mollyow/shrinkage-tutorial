# Load the comparison-v1 engine. Source from the repository root:
#   source("analysis/comparison/load.R")
# The orderridge package is loaded from the local source tree so the engine
# always runs the hashed package code.

local({
  root <- normalizePath(getwd(), mustWork = TRUE)
  if (!file.exists(file.path(root, "analysis", "comparison", "load.R"))) {
    stop("Source analysis/comparison/load.R from the repository root.", call. = FALSE)
  }
  for (pkg in c("yaml", "digest", "jsonlite", "Matrix", "glmnet", "ranger", "xgboost",
                "nanoparquet", "pkgload")) {
    if (!requireNamespace(pkg, quietly = TRUE)) stop("Package `", pkg, "` is required.", call. = FALSE)
  }
  suppressMessages(pkgload::load_all(root, quiet = TRUE, export_all = FALSE,
                                     helpers = FALSE, attach_testthat = FALSE))
  files <- sort(list.files(file.path(root, "analysis", "comparison", "R"),
                           pattern = "\\.R$", full.names = TRUE))
  for (f in files) sys.source(f, envir = globalenv())
})
