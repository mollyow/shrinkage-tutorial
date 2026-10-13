#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
value_after <- function(flag, default = NULL) {
  where <- match(flag, args)
  if (is.na(where) || where == length(args)) return(default)
  args[where + 1L]
}
config_path <- value_after("--config", "config/simulation-fast.yml")
output_path <- value_after("--output", "results/simulations/fast/results.csv")
workers <- as.integer(value_after("--workers", "1"))

if (!requireNamespace("orderridge", quietly = TRUE)) {
  stop("Install the local package first, e.g. R CMD INSTALL .")
}
suppressPackageStartupMessages(library(orderridge))
source("simulation/simulation_functions.R")
config <- read_simulation_config(config_path)
scenarios <- expand_simulation_scenarios(config)
jobs <- do.call(rbind, lapply(seq_len(nrow(scenarios)), function(i) {
  repetitions <- scenario_replications(scenarios[i, , drop = FALSE], config)
  data.frame(scenario_row = i, replication = seq_len(repetitions))
}))
run_job <- function(j) {
  row <- jobs[j, ]
  run_simulation_replication(scenarios[row$scenario_row, , drop = FALSE],
                             row$replication, config)
}
if (workers > 1L && .Platform$OS.type != "windows") {
  pieces <- parallel::mclapply(seq_len(nrow(jobs)), run_job, mc.cores = workers,
                               mc.preschedule = FALSE, mc.set.seed = FALSE)
} else {
  pieces <- lapply(seq_len(nrow(jobs)), run_job)
}
results <- do.call(rbind, pieces)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
utils::write.csv(results, output_path, row.names = FALSE, na = "")
message("Wrote ", nrow(results), " method-replication rows to ", output_path)
