testthat::test_that("outcome-blind asylum contract reproduces samples, support, folds, and invariant families", {
  output_dir <- tempfile("asylum-design-audit-")
  dir.create(output_dir)
  on.exit(unlink(output_dir, recursive = TRUE, force = TRUE), add = TRUE)
  repo_root <- Sys.getenv("ASYLUM_AUDIT_REPO_ROOT")
  testthat::expect_true(nzchar(repo_root))
  audit_script <- file.path(repo_root, "analysis", "asylum", "audit_design.R")

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

  read_output <- function(name) utils::read.csv(file.path(output_dir, name), check.names = FALSE)
  archive <- read_output("archive_checksums.csv")
  validation <- read_output("design_validation.csv")
  sample <- read_output("sample_summary.csv")
  countries <- read_output("country_counts.csv")
  levels <- read_output("level_lock.csv")
  pair_support <- read_output("all_pair_support_summary.csv")
  families <- read_output("family_identification.csv")
  coordinates <- read_output("contrast_coordinates.csv")
  projectors <- read_output("family_projectors.csv")
  loaded <- read_output("loaded_columns.csv")
  frames <- read_output("frame_counts.csv")
  strata <- read_output("profile_probability_strata.csv")
  folds <- read_output("respondent_folds.csv")
  covariates <- read_output("covariate_audit.csv")
  feature_lock <- read_output("covariate_feature_lock.csv")
  target_weights <- read_output("target_weight_diagnostics.csv")
  manifest <- yaml::read_yaml(file.path(output_dir, "design_audit_manifest.yml"))

  testthat::expect_true(all(archive$pass))
  testthat::expect_true(all(validation$pass))
  testthat::expect_identical(manifest$gate_result, "green")
  testthat::expect_true(manifest$outcome_blind)
  testthat::expect_false(manifest$outcomes_loaded)
  testthat::expect_false(manifest$outcome_models_fit)
  testthat::expect_false(any(loaded$outcome_loaded))
  testthat::expect_false(any(loaded$column %in% c("pref", "rate", "ratebin")))

  testthat::expect_equal(sample$archive_respondents, c(18030L, 14976L))
  testthat::expect_equal(sample$design_complete_respondents, c(18021L, 14966L))
  testthat::expect_equal(sample$design_complete_tasks, c(90105L, 74830L))
  testthat::expect_equal(sample$design_complete_profile_rows, c(180210L, 149660L))
  testthat::expect_equal(sample$primary_weight_eligible_respondents, c(17874L, 14846L))
  testthat::expect_equal(sample$primary_weight_eligible_profile_rows, c(178740L, 148460L))
  testthat::expect_equal(nrow(countries), 30L)
  testthat::expect_equal(as.numeric(tapply(countries$archive_respondents, countries$wave, sum)),
                         c(18030, 14976))

  testthat::expect_equal(nrow(pair_support), 72L)
  testthat::expect_true(all(pair_support$complete_cartesian))
  testthat::expect_equal(nrow(levels), 73L)
  testthat::expect_equal(strata$profile_support_size, c(136080, 170100))
  testthat::expect_equal(strata$ordered_pair_support_size, c(136080^2, 170100^2))
  testthat::expect_equal(frames$archive_respondents, c(3709L, 3809L, 3730L, 3728L))
  testthat::expect_true(all(frames$target_probability == 0.25))

  rank_key <- paste(families$wave, families$family, sep = "::")
  testthat::expect_equal(
    setNames(families$observed_interaction_rank, rank_key),
    c(
      `2016::reason_x_vulnerability` = 12L,
      `2016::religion_x_origin` = 12L,
      `2016::occupation_x_language` = 10L,
      `2022::reason_x_vulnerability` = 16L,
      `2022::religion_x_origin` = 12L,
      `2022::occupation_x_language` = 10L
    )
  )
  testthat::expect_true(all(families$identified))
  testthat::expect_gt(min(families$minimum_country_cell_count), 0)
  testthat::expect_lt(max(families$orthonormality_error), 1e-10)
  testthat::expect_lt(max(families$main_effect_orthogonality_error), 1e-10)
  testthat::expect_lt(max(families$projector_idempotence_error), 1e-10)

  # Projectors are functions of the target-weighted interaction subspaces, not
  # a reference category or a particular contrast basis.
  group_keys <- unique(coordinates[c("wave", "family")])
  set.seed(20260901)
  for (g in seq_len(nrow(group_keys))) {
    year <- group_keys$wave[[g]]
    family <- group_keys$family[[g]]
    cc <- coordinates[coordinates$wave == year & coordinates$family == family, ]
    pp <- projectors[projectors$wave == year & projectors$family == family, ]
    basis <- xtabs(basis_value ~ cell_index + coordinate, data = cc)
    stored_projector <- xtabs(value ~ row_cell_index + column_cell_index, data = pp)
    n_cells <- nrow(basis)
    reconstructed <- basis %*% t(basis) / n_cells
    testthat::expect_lt(max(abs(reconstructed - stored_projector)), 1e-12)

    rotation <- qr.Q(qr(matrix(stats::rnorm(ncol(basis)^2), ncol(basis), ncol(basis))))
    rotated <- basis %*% rotation
    rotated_projector <- rotated %*% t(rotated) / n_cells
    testthat::expect_lt(max(abs(rotated_projector - stored_projector)), 1e-12)

    cell_map <- unique(cc[c("cell_index", "a_level", "b_level")])
    cell_map <- cell_map[order(cell_map$cell_index), ]
    reference_main <- stats::model.matrix(
      ~ factor(a_level) + factor(b_level), data = cell_map
    )
    testthat::expect_lt(max(abs(stored_projector %*% reference_main)), 1e-10)

    permutation <- sample(seq_len(n_cells))
    permuted_basis <- basis[permutation, , drop = FALSE]
    testthat::expect_lt(max(abs(
      permuted_basis %*% t(permuted_basis) / n_cells -
        stored_projector[permutation, permutation]
    )), 1e-12)
  }

  testthat::expect_equal(nrow(folds), 18021L + 14966L)
  testthat::expect_equal(anyDuplicated(folds[c("wave", "ResponseId")]), 0L)
  testthat::expect_equal(sum(folds$primary_weight_eligible), 17874L + 14846L)
  fold_strata <- interaction(folds$wave, folds$country, folds$frame, drop = TRUE)
  outer_balance <- vapply(split(folds$outer_fold, fold_strata),
                          function(x) diff(range(tabulate(x, nbins = 5))) <= 1L, logical(1))
  testthat::expect_true(all(outer_balance))
  for (held_out in 1:5) {
    coverage <- vapply(split(folds, fold_strata), function(x) {
      length(unique(x$inner_fold_label[x$outer_fold != held_out])) == 5L
    }, logical(1))
    testthat::expect_true(all(coverage))
  }

  primary <- covariates[covariates$primary_feature, ]
  testthat::expect_equal(as.integer(table(primary$wave)), c(8L, 8L))
  testthat::expect_equal(sort(feature_lock$variable), sort(c(
    "cty", "gender", "age", "home_born", "educ_EISCED",
    "asy_change_home", "asy_change_europe", "immig_change_home"
  )))
  testthat::expect_equal(
    covariates$maximum[covariates$wave == 2016 & covariates$variable == "age"], 116
  )
  testthat::expect_equal(
    as.numeric(tapply(target_weights$normalized_target_weight_sum, target_weights$wave, sum)),
    c(1, 1), tolerance = 1e-12
  )
})

testthat::test_that("unexpected stale asylum audit files fail closed", {
  output_dir <- tempfile("asylum-design-audit-stale-")
  dir.create(output_dir)
  on.exit(unlink(output_dir, recursive = TRUE, force = TRUE), add = TRUE)
  file.create(file.path(output_dir, "stale_result.csv"))

  repo_root <- Sys.getenv("ASYLUM_AUDIT_REPO_ROOT")
  audit_script <- file.path(repo_root, "analysis", "asylum", "audit_design.R")
  result <- suppressWarnings(system2(
    "Rscript",
    c(audit_script, "--output-dir", output_dir),
    stdout = TRUE,
    stderr = TRUE
  ))
  testthat::expect_false(is.null(attr(result, "status")))
  testthat::expect_match(paste(result, collapse = "\n"), "unexpected generated files")
})
