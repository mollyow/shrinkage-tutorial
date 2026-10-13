#!/usr/bin/env Rscript

# Reproduce the small exact-enumeration grid used for the theorem search.
# This wrapper deliberately stops at fold size four: size five has 810,000
# ordered supported assignment pairs and is unnecessary for the identity
# check. Gaussian outcomes are integrated analytically at every size.

arguments <- commandArgs(trailingOnly = FALSE)
file_argument <- grep("^--file=", arguments, value = TRUE)
script_path <- if (length(file_argument) == 1L) {
  normalizePath(sub("^--file=", "", file_argument), mustWork = TRUE)
} else {
  normalizePath(
    "analysis/theorem_gate/run_two_fold_size_grid.R", mustWork = TRUE
  )
}
script_directory <- dirname(script_path)
source(file.path(script_directory, "two_fold_exact_weight_search.R"))

trailing <- commandArgs(trailingOnly = TRUE)
output_argument <- grep("^--output=", trailing, value = TRUE)
output_path <- if (length(output_argument) == 1L) {
  sub("^--output=", "", output_argument)
} else {
  file.path(script_directory, "two_fold_exact_weight_size_grid.csv")
}
if (length(setdiff(trailing, output_argument)) > 0L) {
  stop("The grid wrapper accepts only `--output=...`.", call. = FALSE)
}

rho_grid <- c(0, 0.35, 0.70, 0.90)
fits <- lapply(2:4, function(fold_size) {
  run_search(list(
    fold_size = as.integer(fold_size),
    rho = rho_grid,
    output = NULL,
    quiet = TRUE
  ))
})
results <- do.call(rbind, lapply(fits, `[[`, "results"))
write.csv(results, output_path, row.names = FALSE)

identity_summary <- data.frame(
  fold_size = 2:4,
  supported_states_per_fold = vapply(
    fits, function(fit) fit$grams$state_count, numeric(1)
  ),
  supported_assignment_pairs = vapply(
    fits, function(fit) fit$grams$pair_count, numeric(1)
  ),
  maximum_complete_half_marginal_gram_residual = vapply(
    fits, `[[`, numeric(1), "complete_half_marginal_gram_residual"
  ),
  runtime_seconds = vapply(fits, `[[`, numeric(1), "runtime_seconds")
)
print(identity_summary, row.names = FALSE, digits = 8)
cat(sprintf("\nWrote %d method-risk rows to %s.\n",
            nrow(results), normalizePath(output_path, mustWork = TRUE)))

