#!/usr/bin/env Rscript
# Build the outcome-free version-4 candidate context distribution used by the
# comparison-v1 rating simulations (config/comparison-simulations-v1.yml,
# rating_geometry.context_artifact).
#
# Reads exactly eight allowlisted columns from the cleaned replication file.
# The outcome column is never parsed. The output contains no task, profile, or
# response information and is deterministic given the locked source file.

suppressPackageStartupMessages({
  library(yaml)
  library(digest)
  library(data.table)
})

script_dir <- function() {
  arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(arg)) dirname(normalizePath(sub("^--file=", "", arg[[1]]))) else getwd()
}
root <- normalizePath(file.path(script_dir(), "..", ".."))
stopifnot(file.exists(file.path(root, "DESCRIPTION")))

fail <- function(...) stop(sprintf(...), call. = FALSE)

sim <- read_yaml(file.path(root, "config", "comparison-simulations-v1.yml"))
spec <- sim$rating_geometry$context_artifact
candidate <- read_yaml(file.path(root, "analysis", "candidate", "estimands.yml"))

source_path <- file.path(root, spec$allowed_source)
if (!file.exists(source_path)) fail("The allowed candidate source file is missing.")
if (!identical(digest(file = source_path, algo = "md5"), candidate$data$md5)) {
  fail("The candidate source checksum does not match analysis/candidate/estimands.yml.")
}

selected <- as.character(unlist(spec$selected_columns))
forbidden <- as.character(unlist(spec$forbidden_columns))
if (length(intersect(selected, forbidden))) fail("The allowlist contains a forbidden column.")

header <- names(fread(source_path, nrows = 0L))
if (!all(selected %in% header)) fail("An allowlisted column is absent from the source header.")

raw <- fread(source_path, select = selected, colClasses = list(character = "id"),
             showProgress = FALSE)
if (!identical(sort(names(raw)), sort(selected))) fail("The column barrier failed.")
if (any(forbidden %in% names(raw))) fail("A forbidden column was loaded.")

context <- c("woman", "poc", "young", "dem", "rep", "workingclass")
expected <- candidate$data$expected
if (nrow(raw) != expected$rows) fail("Expected %d rows, found %d.", expected$rows, nrow(raw))

invariant <- raw[, lapply(.SD, function(x) length(unique(x)) == 1L), by = id,
                 .SDcols = c("weight", context)]
if (!all(unlist(invariant[, -1L]))) fail("Weight or a context variable varies within respondent.")

resp <- unique(raw, by = "id")
setorder(resp, id)
if (nrow(resp) != expected$respondents) {
  fail("Expected %d respondents, found %d.", expected$respondents, nrow(resp))
}
if (anyNA(resp) || any(!is.finite(resp$weight)) || any(resp$weight <= 0)) {
  fail("Weights must be finite and positive and contexts complete.")
}
for (nm in context) {
  if (!all(resp[[nm]] %in% c(0, 1))) fail("Context variable %s is not binary.", nm)
  resp[[nm]] <- as.integer(resp[[nm]])
}

artifact <- data.frame(
  id = resp$id,
  probability = resp$weight / sum(resp$weight),
  resp[, context, with = FALSE],
  stringsAsFactors = FALSE
)
rownames(artifact) <- NULL

output <- file.path(root, spec$output)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
saveRDS(artifact, output, version = 3L, compress = "xz")

key <- do.call(paste, c(artifact[context], sep = ""))
manifest <- list(
  artifact = spec$output,
  artifact_version = 4L,
  sha256 = digest(file = output, algo = "sha256"),
  source = spec$allowed_source,
  source_md5 = candidate$data$md5,
  columns_read = selected,
  forbidden_columns_never_read = forbidden,
  respondents = nrow(artifact),
  distinct_contexts = length(unique(key)),
  weight_rule = "respondent weights normalized to sum to one",
  sampling_rule = "simulated respondents are drawn with probability equal to the normalized weight",
  stale_estimands_v3_artifacts_used = FALSE
)
writeLines(as.yaml(manifest, handlers = list(logical = verbatim_logical)),
           sub("\\.rds$", "-manifest.yml", output))
cat(sprintf("Wrote %s (%d respondents, %d contexts, sha256 %s)\n",
            spec$output, nrow(artifact), manifest$distinct_contexts, manifest$sha256))
