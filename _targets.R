library(targets)

mode <- Sys.getenv("ORDER_RIDGE_MODE", unset = "fast")
candidate_stage <- Sys.getenv("CANDIDATE_STAGE", unset = "off")
if (!candidate_stage %in% c("off", "check", "full")) {
  stop("CANDIDATE_STAGE must be one of: off, check, full.", call. = FALSE)
}
config_file <- if (identical(mode, "production")) {
  "config/simulation-production.yml"
} else {
  "config/simulation-fast.yml"
}
result_file <- file.path("results", "simulations", mode, "results.csv")
gate_file <- file.path("results", "simulations", mode, "gate-status.json")

tar_option_set(packages = c("yaml", "jsonlite"), seed = 240826)

simulation_targets <- list(
  tar_target(package_source_files, {
    c("DESCRIPTION", "NAMESPACE", sort(list.files("R", pattern = "\\.[Rr]$",
                                                   full.names = TRUE)))
  }, format = "file"),
  tar_target(simulation_source_files, {
    c(config_file, "simulation/simulation_functions.R",
      "simulation/run_simulations.R", "simulation/summarize_gates.R")
  }, format = "file"),
  tar_target(package_install, {
    package_source_files
    status <- system2(file.path(R.home("bin"), "R"), c("CMD", "INSTALL", "."))
    if (!identical(status, 0L)) stop("Local package installation failed.", call. = FALSE)
    stats::setNames(unname(tools::md5sum(package_source_files)), package_source_files)
  }, cue = tar_cue(mode = "always")),
  tar_target(simulation_results, {
    package_install
    simulation_source_files
    status <- system2(file.path(R.home("bin"), "Rscript"),
                      c("simulation/run_simulations.R", "--config", config_file,
                        "--output", result_file, "--workers", "1"))
    if (!identical(status, 0L)) stop("Simulation subprocess failed.", call. = FALSE)
    result_file
  }, format = "file"),
  tar_target(gate_status, {
    status <- system2(file.path(R.home("bin"), "Rscript"),
                      c("simulation/summarize_gates.R", simulation_results,
                        gate_file, config_file))
    if (!identical(status, 0L)) stop("Gate-summary subprocess failed.", call. = FALSE)
    gate_file
  }, format = "file")
)

candidate_targets <- list()
if (candidate_stage %in% c("check", "full")) {
  candidate_expected_preparation <- c(
    "candidate_prepared.rds", "candidate_column_map.csv", "candidate_level_lock.csv",
    "candidate_validation.csv", "candidate_folds.csv", "candidate_target_grid.csv",
    "candidate_profile_counts.csv", "candidate_coordinate_metadata.csv",
    "candidate_cell_metadata.csv", "candidate_contrast_coordinates.csv",
    "candidate_contrast_cells.csv", "candidate_contrast_mapping.csv",
    "candidate_classic_did_contrasts.csv", "candidate_preparation_manifest.yml",
    "candidate_design_and_levels_lock.yml"
  )
  candidate_expected_analysis <- c(
    "heldout_predictions.csv", "row_metrics.csv",
    "heldout_profile_predictions_by_fold.csv",
    "heldout_profile_predictions_pooled.csv", "profile_metrics.csv",
    "final_profile_predictions.csv", "monotone_order_ridge_profile_ranking.csv",
    "fit_diagnostics.csv", "ridge_tuning_evaluations.csv", "ridge_inner_folds.csv",
    "saturated_training_support.csv", "package_source_checksums.csv",
    "amie_coordinates.csv", "amie_cells.csv", "amie_classic_did.csv",
    "amie_family_tests.csv", "amie_outer_folds.csv", "amie_inner_folds.csv",
    "amie_fit_diagnostics.csv", "amie_tuning_evaluations.csv",
    "candidate_final_fits.rds", "candidate_amie_fit.rds",
    "analysis_manifest.yml", "session_info.txt"
  )

  candidate_targets <- list(
    tar_target(candidate_contract_sources, {
      c(
        "analysis/candidate/estimands.yml",
        "analysis/candidate/contract_helpers.R",
        "analysis/candidate/prepare_candidate_data.R",
        "analysis/candidate/run_candidate_analysis.R",
        "analysis/candidate/README.md",
        "analysis/candidate/tests/testthat.R",
        sort(list.files("analysis/candidate/tests/testthat", pattern = "\\.[Rr]$",
                        full.names = TRUE))
      )
    }, format = "file"),
    tar_target(candidate_cleaned_csv, {
      path <- file.path(
        "Candidate recruitment", "replication", "PSRM_replication",
        "data", "CleanedData.csv"
      )
      if (!file.exists(path)) {
        stop(
          "Candidate cleaned data are absent. Download Harvard Dataverse ",
          "doi:10.7910/DVN/BQAHGZ, verify PSRM.zip MD5 ",
          "0fb95eb8c2f57bb363c956a7eff8feb6, run the authors' Stata cleaning.do, ",
          "and place the generated CSV at: ", path,
          call. = FALSE
        )
      }
      path
    }, format = "file"),
    tar_target(candidate_contract_tests, {
      candidate_contract_sources
      package_source_files
      status <- system2(
        file.path(R.home("bin"), "Rscript"),
        "analysis/candidate/tests/testthat.R"
      )
      if (!identical(status, 0L)) {
        stop("Candidate application contract tests failed.", call. = FALSE)
      }
      TRUE
    }),
    tar_target(candidate_preparation_outputs, {
      candidate_contract_sources
      candidate_cleaned_csv
      out_dir <- file.path("analysis", "candidate", "derived")
      status <- system2(
        file.path(R.home("bin"), "Rscript"),
        c("analysis/candidate/prepare_candidate_data.R", "--output-dir", out_dir)
      )
      if (!identical(status, 0L)) stop("Candidate preparation failed.", call. = FALSE)
      outputs <- file.path(out_dir, candidate_expected_preparation)
      if (!all(file.exists(outputs))) {
        stop("Candidate preparation did not produce every locked output.", call. = FALSE)
      }
      outputs
    }, format = "file"),
    tar_target(candidate_application_checkpoint, {
      candidate_contract_tests
      candidate_preparation_outputs
      package_source_files
      prepared <- file.path("analysis", "candidate", "derived", "candidate_prepared.rds")
      status <- system2(
        file.path(R.home("bin"), "Rscript"),
        c("analysis/candidate/run_candidate_analysis.R", "--prepared", prepared,
          "--api-check-only")
      )
      if (!identical(status, 0L)) stop("Candidate API checkpoint failed.", call. = FALSE)
      checkpoint <- file.path("analysis", "candidate", "derived",
                              "application-checkpoint.yml")
      yaml::write_yaml(list(
        status = "passed",
        prepared_md5 = unname(tools::md5sum(prepared)),
        estimands_md5 = unname(tools::md5sum("analysis/candidate/estimands.yml")),
        package_source_md5 = split(unname(tools::md5sum(package_source_files)),
                                   package_source_files),
        checked_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE)
      ), checkpoint)
      checkpoint
    }, format = "file")
  )

  if (identical(candidate_stage, "full")) {
    candidate_targets <- c(candidate_targets, list(
      tar_target(candidate_analysis_outputs, {
        candidate_application_checkpoint
        gate <- jsonlite::read_json(gate_status, simplifyVector = TRUE)
        if (!isTRUE(gate$full_methods_gate_passed)) {
          stop(
            "The complete production methods gate has not passed; the locked ",
            "application fit is intentionally disabled.",
            call. = FALSE
          )
        }
        out_dir <- file.path("analysis", "candidate", "results")
        status <- system2(
          file.path(R.home("bin"), "Rscript"),
          c("analysis/candidate/run_candidate_analysis.R", "--prepared",
            file.path("analysis", "candidate", "derived", "candidate_prepared.rds"),
            "--output-dir", out_dir)
        )
        if (!identical(status, 0L)) stop("Candidate analysis failed.", call. = FALSE)
        outputs <- file.path(out_dir, candidate_expected_analysis)
        if (!all(file.exists(outputs))) {
          stop("Candidate analysis did not produce every locked output.", call. = FALSE)
        }
        outputs
      }, format = "file")
    ))
  }
}

c(simulation_targets, candidate_targets)
