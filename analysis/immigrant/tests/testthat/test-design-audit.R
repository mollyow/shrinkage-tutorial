testthat::test_that("outcome-blind design contract reproduces the locked audit", {
  output_dir <- tempfile("immigrant-design-audit-")
  dir.create(output_dir)
  on.exit(unlink(output_dir, recursive = TRUE, force = TRUE), add = TRUE)
  repo_root <- Sys.getenv("IMMIGRANT_AUDIT_REPO_ROOT")
  testthat::expect_true(nzchar(repo_root))
  audit_script <- file.path(repo_root, "analysis", "immigrant", "audit_design.R")

  result <- suppressWarnings(system2(
    "Rscript",
    c(audit_script, "--output-dir", output_dir),
    stdout = TRUE,
    stderr = TRUE
  ))
  status <- attr(result, "status")
  if (is.null(status)) status <- 0L
  testthat::expect_equal(status, 0L, info = paste(result, collapse = "\n"))
  if (status != 0L) return(invisible(NULL))

  validation <- utils::read.csv(file.path(output_dir, "design_validation.csv"))
  families <- utils::read.csv(file.path(output_dir, "family_identification.csv"))
  loaded <- utils::read.csv(file.path(output_dir, "loaded_columns.csv"))
  strata <- utils::read.csv(file.path(output_dir, "profile_probability_strata.csv"))
  covariates <- utils::read.csv(file.path(output_dir, "covariate_audit.csv"))
  feature_lock <- utils::read.csv(file.path(output_dir, "covariate_feature_lock.csv"))
  category_levels <- utils::read.csv(file.path(output_dir, "covariate_category_levels.csv"))
  projectors <- utils::read.csv(file.path(output_dir, "family_projectors.csv"))
  manifest <- yaml::read_yaml(file.path(output_dir, "design_audit_manifest.yml"))

  testthat::expect_true(all(validation$pass))
  testthat::expect_false(manifest$outcome_values_read)
  testthat::expect_false(manifest$models_fit)
  testthat::expect_equal(length(manifest$unresolved_blockers), 2L)
  testthat::expect_equal(
    validation$observed[validation$check == "design_complete_respondents"],
    "1407"
  )
  testthat::expect_equal(
    validation$observed[validation$check == "china_x_persecution_rows"],
    "500"
  )
  testthat::expect_false(any(loaded$outcome_loaded))
  testthat::expect_false(any(loaded$column %in% c(
    "Chosen_Immigrant", "Support_Admission", "Rating_Immigrant"
  )))
  testthat::expect_equal(
    setNames(families$identified_interaction_df, families$family),
    c(
      education_x_profession = 44L,
      origin_x_application_reason = 12L,
      language_x_education = 18L
    )
  )
  testthat::expect_lt(max(families$weighted_gram_error), 1e-10)
  testthat::expect_lt(max(families$additive_orthogonality_error), 1e-10)
  testthat::expect_lt(max(families$coding_projector_error), 1e-10)
  testthat::expect_lt(max(families$coding_basis_error), 1e-10)
  testthat::expect_lt(max(families$projector_idempotence_error), 1e-10)
  testthat::expect_lt(max(families$projector_additive_error), 1e-10)
  testthat::expect_lt(max(families$row_space_residual), 1e-10)
  testthat::expect_equal(nrow(projectors), 61^2 + 24^2 + 28^2)
  testthat::expect_equal(sum(strata$admissible_profiles), 936960)
  testthat::expect_equal(max(strata$uniform_mu_over_q), 1.57786885245902, tolerance = 1e-12)
  testthat::expect_equal(
    setNames(covariates$missing_n, covariates$variable)[c(
      "SM_index", "W1_Q4", "ethnocentrism", "ethnocentrism_strict"
    )],
    c(SM_index = 11L, W1_Q4 = 13L, ethnocentrism = 251L, ethnocentrism_strict = 271L)
  )
  testthat::expect_equal(nrow(feature_lock), 10L)
  testthat::expect_equal(
    sort(feature_lock$variable[feature_lock$semantic_type == "unordered_categorical"]),
    sort(c("ppethm", "ppgender", "ppstaten"))
  )
  testthat::expect_true(all(
    feature_lock$primary_encoding[grepl("categor", feature_lock$semantic_type)] ==
      "full_one_hot_all_locked_levels_no_reference_drop"
  ))
  testthat::expect_equal(nrow(category_levels), 91L)
  testthat::expect_equal(sum(category_levels$is_missing_level), 6L)
  testthat::expect_equal(
    category_levels$enrolled_count[
      category_levels$variable == "W1_Q4" & category_levels$is_missing_level
    ],
    13L
  )
})

testthat::test_that("unexpected stale audit files fail closed", {
  output_dir <- tempfile("immigrant-design-audit-stale-")
  dir.create(output_dir)
  on.exit(unlink(output_dir, recursive = TRUE, force = TRUE), add = TRUE)
  file.create(file.path(output_dir, "stale_result.csv"))

  repo_root <- Sys.getenv("IMMIGRANT_AUDIT_REPO_ROOT")
  audit_script <- file.path(repo_root, "analysis", "immigrant", "audit_design.R")
  result <- suppressWarnings(system2(
    "Rscript",
    c(audit_script, "--output-dir", output_dir),
    stdout = TRUE,
    stderr = TRUE
  ))
  testthat::expect_false(is.null(attr(result, "status")))
  testthat::expect_match(paste(result, collapse = "\n"), "unexpected generated files")
})
