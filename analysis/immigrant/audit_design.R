#!/usr/bin/env Rscript

options(stringsAsFactors = FALSE)

abort <- function(...) stop(sprintf(...), call. = FALSE)
assert <- function(ok, ...) if (!isTRUE(ok)) abort(...)

arg_value <- function(flag, default = NULL) {
  args <- commandArgs(trailingOnly = TRUE)
  hit <- which(args == flag)
  if (!length(hit)) return(default)
  assert(hit[[1]] < length(args), "Missing value after %s", flag)
  args[[hit[[1]] + 1L]]
}

script_path <- function() {
  hit <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(hit)) return(NA_character_)
  normalizePath(sub("^--file=", "", hit[[1]]), mustWork = TRUE)
}

find_repo_root <- function() {
  candidates <- c(
    normalizePath(getwd(), mustWork = TRUE),
    if (!is.na(script_path())) normalizePath(file.path(dirname(script_path()), "..", ".."), mustWork = TRUE)
  )
  for (candidate in unique(candidates)) {
    if (file.exists(file.path(candidate, "config", "datasets", "immigrant-admission.yml"))) return(candidate)
  }
  abort("Could not locate the repository root containing config/datasets/immigrant-admission.yml")
}

write_csv <- function(x, path) {
  utils::write.csv(x, path, row.names = FALSE, na = "")
}

repo_root <- find_repo_root()
contract_arg <- arg_value("--contract", "config/datasets/immigrant-admission.yml")
output_arg <- arg_value("--output-dir", "analysis/immigrant/derived")
contract_path <- if (grepl("^/", contract_arg)) contract_arg else file.path(repo_root, contract_arg)
output_dir <- if (grepl("^/", output_arg)) output_arg else file.path(repo_root, output_arg)

assert(requireNamespace("haven", quietly = TRUE), "Package 'haven' is required")
assert(requireNamespace("tidyselect", quietly = TRUE), "Package 'tidyselect' is required")
assert(requireNamespace("yaml", quietly = TRUE), "Package 'yaml' is required")

contract <- yaml::read_yaml(contract_path)
assert(identical(contract$status, "conditional_design_lock_outcomes_prohibited"), "Unexpected contract status")

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
expected_csv_files <- c(
  "all_pair_support_summary.csv",
  "archive_checksums.csv",
  "assignment_diagnostics.csv",
  "contrast_coordinates.csv",
  "covariate_audit.csv",
  "covariate_category_levels.csv",
  "covariate_feature_lock.csv",
  "design_validation.csv",
  "family_identification.csv",
  "family_projectors.csv",
  "level_lock.csv",
  "loaded_columns.csv",
  "profile_probability_strata.csv",
  "support_cells.csv",
  "support_summary.csv"
)
existing_generated <- list.files(output_dir, pattern = "\\.(csv|yml)$", full.names = FALSE)
unexpected_generated <- setdiff(existing_generated, c(expected_csv_files, "design_audit_manifest.yml"))
assert(!length(unexpected_generated),
       "Output directory contains unexpected generated files: %s",
       paste(unexpected_generated, collapse = ", "))
# Invalidate any previous successful bundle before starting. A failed rerun may
# leave old CSVs, but it cannot leave a manifest that appears current.
old_manifest <- file.path(output_dir, "design_audit_manifest.yml")
if (file.exists(old_manifest)) unlink(old_manifest)

resolve_path <- function(path) if (grepl("^/", path)) path else file.path(repo_root, path)
file_specs <- contract$provenance$files
archive_checks <- do.call(rbind, lapply(names(file_specs), function(id) {
  spec <- file_specs[[id]]
  path <- resolve_path(spec$path)
  assert(file.exists(path), "Missing locked %s file: %s", id, path)
  observed_bytes <- unname(file.info(path)$size)
  observed_md5 <- unname(tools::md5sum(path))
  data.frame(
    file_id = id,
    relative_path = spec$path,
    expected_bytes = as.numeric(spec$bytes),
    observed_bytes = observed_bytes,
    expected_md5 = spec$md5,
    observed_md5 = observed_md5,
    pass = observed_bytes == as.numeric(spec$bytes) && identical(observed_md5, spec$md5)
  )
}))
assert(all(archive_checks$pass), "At least one archive checksum or byte-size lock failed")
write_csv(archive_checks, file.path(output_dir, "archive_checksums.csv"))

allowed_columns <- unlist(contract$read_barrier$allowed_columns, use.names = FALSE)
forbidden_outcomes <- unlist(contract$read_barrier$forbidden_outcomes, use.names = FALSE)
assert(!any(allowed_columns %in% forbidden_outcomes), "An outcome appears in the design-only allowlist")
assert(!anyDuplicated(allowed_columns), "The design-only allowlist contains duplicate columns")

data_path <- resolve_path(contract$provenance$files$data$path)
raw <- haven::read_dta(data_path, col_select = tidyselect::all_of(allowed_columns))
assert(setequal(names(raw), allowed_columns), "The selected design-only columns did not match the lock")
assert(!any(names(raw) %in% forbidden_outcomes), "An outcome was loaded despite the read barrier")
raw$.source_row <- seq_len(nrow(raw))

loaded_columns <- data.frame(
  column = names(raw)[names(raw) != ".source_row"],
  role = ifelse(names(raw)[names(raw) != ".source_row"] %in%
                  vapply(contract$attributes, `[[`, character(1), "column"),
                "randomized_attribute", "design_or_preconjoint"),
  outcome_loaded = FALSE
)
write_csv(loaded_columns, file.path(output_dir, "loaded_columns.csv"))

attribute_ids <- names(contract$attributes)
attribute_columns <- vapply(contract$attributes, `[[`, character(1), "column")
attribute_numeric <- lapply(raw[attribute_columns], function(x) as.numeric(x))
names(attribute_numeric) <- attribute_columns
design_complete <- Reduce(`&`, lapply(attribute_numeric, function(x) !is.na(x)))

id_chr <- as.character(raw$CaseID)
id_split <- split(seq_len(nrow(raw)), id_chr)
complete_by_id <- vapply(id_split, function(ix) sum(design_complete[ix]), integer(1))
rows_by_id <- lengths(id_split)
assert(all(complete_by_id == 0L | complete_by_id == rows_by_id),
       "At least one respondent has a partially observed profile block")

design <- raw[design_complete, , drop = FALSE]
for (nm in attribute_columns) design[[nm]] <- as.numeric(design[[nm]])
design$CaseID <- as.numeric(design$CaseID)
design$contest_no <- as.numeric(design$contest_no)
design$weight2 <- as.numeric(design$weight2)

sample_expected <- contract$sample$expected
validation <- data.frame(check = character(), observed = character(), expected = character(), pass = logical(), detail = character())
add_check <- function(check, observed, expected, pass, detail = "") {
  validation <<- rbind(validation, data.frame(
    check = check, observed = as.character(observed), expected = as.character(expected),
    pass = isTRUE(pass), detail = detail
  ))
}

add_check("archive_rows", nrow(raw), sample_expected$all_rows, nrow(raw) == sample_expected$all_rows)
add_check("archive_respondents", length(unique(raw$CaseID)), sample_expected$archive_respondents,
          length(unique(raw$CaseID)) == sample_expected$archive_respondents)
add_check("design_complete_rows", nrow(design), sample_expected$design_complete_profile_rows,
          nrow(design) == sample_expected$design_complete_profile_rows)
add_check("design_complete_respondents", length(unique(design$CaseID)), sample_expected$design_complete_respondents,
          length(unique(design$CaseID)) == sample_expected$design_complete_respondents)
add_check("all_profile_missing_respondents", sum(complete_by_id == 0L), sample_expected$all_profile_missing_respondents,
          sum(complete_by_id == 0L) == sample_expected$all_profile_missing_respondents)

respondent_rows <- table(design$CaseID)
respondent_tasks <- tapply(design$contest_no, design$CaseID, function(x) length(unique(x)))
pair_key <- paste(design$CaseID, design$contest_no, sep = "::")
pair_rows <- table(pair_key)
add_check("rows_per_design_respondent", paste(range(respondent_rows), collapse = ":"),
          sample_expected$tasks_per_design_complete_respondent * sample_expected$profiles_per_task,
          all(respondent_rows == sample_expected$tasks_per_design_complete_respondent * sample_expected$profiles_per_task))
add_check("tasks_per_design_respondent", paste(range(respondent_tasks), collapse = ":"),
          sample_expected$tasks_per_design_complete_respondent,
          all(respondent_tasks == sample_expected$tasks_per_design_complete_respondent))
add_check("task_levels", paste(sort(unique(design$contest_no)), collapse = ","), "1,2,3,4,5",
          identical(as.integer(sort(unique(design$contest_no))), 1:5))
add_check("profiles_per_task", paste(range(pair_rows), collapse = ":"), sample_expected$profiles_per_task,
          all(pair_rows == sample_expected$profiles_per_task))
add_check("design_complete_tasks", length(pair_rows), sample_expected$design_complete_tasks,
          length(pair_rows) == sample_expected$design_complete_tasks)

design$profile_index <- ave(design$.source_row, pair_key, FUN = function(z) rank(z, ties.method = "first"))
pair_source_rows <- split(design$.source_row, pair_key)
pair_rows_contiguous <- vapply(
  pair_source_rows,
  function(x) length(x) == 2L && diff(sort(x)) == 1L,
  logical(1)
)
add_check("pair_rows_contiguous_in_locked_source", sum(pair_rows_contiguous), length(pair_rows_contiguous),
          all(pair_rows_contiguous))
add_check("derived_profile_index", paste(sort(unique(design$profile_index)), collapse = ","), "1,2",
          identical(sort(unique(as.integer(design$profile_index))), 1:2),
          "Derived from immutable source-row order; it is not interpreted as left/right.")

weight_invariant <- vapply(split(design$weight2, design$CaseID), function(x) max(x) - min(x) <= 1e-12, logical(1))
add_check("positive_finite_weights", sum(is.finite(design$weight2) & design$weight2 > 0), nrow(design),
          all(is.finite(design$weight2) & design$weight2 > 0))
add_check("respondent_invariant_weights", sum(weight_invariant), length(weight_invariant), all(weight_invariant))
add_check("respondent_weight_sum", format(sum(design$weight2[!duplicated(design$CaseID)]), digits = 12), "1407.02",
          abs(sum(design$weight2[!duplicated(design$CaseID)]) - 1407.02) < 1e-6)

level_lock <- data.frame()
for (id in attribute_ids) {
  spec <- contract$attributes[[id]]
  column <- spec$column
  codes <- as.numeric(unlist(spec$codes))
  labels <- unlist(spec$labels, use.names = FALSE)
  observed_codes <- sort(unique(design[[column]]))
  assert(identical(observed_codes, codes), "Level-code lock failed for %s", id)
  source_labels <- attr(raw[[column]], "labels")
  source_by_code <- setNames(names(source_labels), as.character(as.numeric(source_labels)))
  source_text <- unname(source_by_code[as.character(codes)])
  counts <- table(factor(design[[column]], levels = codes))
  level_lock <- rbind(level_lock, data.frame(
    attribute = id,
    column = column,
    code = codes,
    contract_label = labels,
    source_label = source_text,
    label_match = labels == source_text,
    observed_count = as.integer(counts)
  ))
}
assert(all(level_lock$label_match), "At least one value label differs from the contract")
write_csv(level_lock, file.path(output_dir, "level_lock.csv"))

allowed_education_profession <- function(e, j) j %in% c(1, 2, 3, 4, 6, 7, 9) || e >= 5
allowed_origin_reason_released <- function(o, r) r <= 2 || o >= 7

pair_cells <- function(a_col, b_col, a_codes, b_codes, allowed, family) {
  grid <- expand.grid(a_code = a_codes, b_code = b_codes, KEEP.OUT.ATTRS = FALSE)
  grid$allowed <- mapply(allowed, grid$a_code, grid$b_code)
  observed <- table(
    factor(design[[a_col]], levels = a_codes),
    factor(design[[b_col]], levels = b_codes)
  )
  grid$observed_count <- as.integer(observed[cbind(match(grid$a_code, a_codes), match(grid$b_code, b_codes))])
  grid$family <- family
  grid[, c("family", "a_code", "b_code", "allowed", "observed_count")]
}

ep_cells <- pair_cells("FeatEd", "FeatJob", 1:7, 1:11, allowed_education_profession, "education_x_profession")
or_cells <- pair_cells("FeatCountry", "FeatReason", 1:10, 1:3, allowed_origin_reason_released, "origin_x_application_reason")
le_cells <- pair_cells("FeatLang", "FeatEd", 1:4, 1:7, function(a, b) TRUE, "language_x_education")
support_cells <- rbind(ep_cells, or_cells, le_cells)
assert(all((support_cells$observed_count > 0) == support_cells$allowed), "Observed pair support differs from the released-support lock")
assert(sum(ep_cells$allowed) == 61L, "Education x profession support does not contain 61 cells")
assert(sum(or_cells$allowed) == 24L, "Origin x reason support does not contain 24 cells")
assert(sum(le_cells$allowed) == 28L, "Language x education support does not contain 28 cells")
write_csv(support_cells, file.path(output_dir, "support_cells.csv"))

all_pair_support <- do.call(rbind, lapply(utils::combn(attribute_ids, 2, simplify = FALSE), function(ids) {
  a <- contract$attributes[[ids[[1]]]]
  b <- contract$attributes[[ids[[2]]]]
  cartesian <- length(a$codes) * length(b$codes)
  observed <- nrow(unique(design[c(a$column, b$column)]))
  expected <- if (identical(ids, c("education", "profession"))) {
    61L
  } else if (identical(ids, c("origin", "application_reason"))) {
    24L
  } else {
    cartesian
  }
  data.frame(
    attribute_a = ids[[1]], attribute_b = ids[[2]],
    cartesian_cells = cartesian, expected_supported_cells = expected,
    observed_cells = observed, pass = observed == expected
  )
}))
assert(all(all_pair_support$pass), "At least one pair support differs from the contract")
write_csv(all_pair_support, file.path(output_dir, "all_pair_support_summary.csv"))

china_persecution_n <- sum(design$FeatCountry == 7 & design$FeatReason == 3)
add_check("china_x_persecution_rows", china_persecution_n, 500, china_persecution_n == 500,
          "Present in released data and code, but excluded by the article's prose.")
add_check("released_origin_reason_cells", sum(or_cells$observed_count > 0), 24,
          sum(or_cells$observed_count > 0) == 24)
add_check("released_admissible_profiles", 61L * 24L * 2L * 4L * 4L * 5L * 4L, 936960,
          61L * 24L * 2L * 4L * 4L * 5L * 4L == 936960)
add_check("article_implied_admissible_profiles", 61L * 23L * 2L * 4L * 4L * 5L * 4L, 897920,
          61L * 23L * 2L * 4L * 4L * 5L * 4L == 897920)

pair_summary <- do.call(rbind, lapply(split(support_cells, support_cells$family), function(x) data.frame(
  family = x$family[[1]],
  cartesian_cells = nrow(x),
  locked_allowed_cells = sum(x$allowed),
  observed_cells = sum(x$observed_count > 0),
  minimum_allowed_cell_count = min(x$observed_count[x$allowed]),
  maximum_forbidden_cell_count = if (any(!x$allowed)) max(x$observed_count[!x$allowed]) else 0
)))
write_csv(pair_summary, file.path(output_dir, "support_summary.csv"))

assignment_diagnostics <- data.frame()
add_assignment_block <- function(component, stratum, values, levels, expected_probability) {
  counts <- table(factor(values, levels = levels))
  n <- sum(counts)
  expected <- n * expected_probability
  assignment_diagnostics <<- rbind(assignment_diagnostics, data.frame(
    component = component,
    stratum = stratum,
    level = levels,
    observed_count = as.integer(counts),
    expected_count = expected,
    observed_probability = as.integer(counts) / n,
    expected_probability = expected_probability,
    pearson_residual = (as.integer(counts) - expected) / sqrt(expected)
  ))
}

for (nm in c("FeatEd", "FeatGender", "FeatCountry", "FeatExp", "FeatPlans", "FeatTrips", "FeatLang")) {
  levels <- sort(unique(design[[nm]]))
  add_assignment_block(nm, "unconditional", design[[nm]], levels, rep(1 / length(levels), length(levels)))
}
for (e in 1:7) {
  allowed_jobs <- if (e <= 4) c(1, 2, 3, 4, 6, 7, 9) else 1:11
  add_assignment_block("FeatJob_given_FeatEd", paste0("FeatEd=", e),
                       design$FeatJob[design$FeatEd == e], allowed_jobs,
                       rep(1 / length(allowed_jobs), length(allowed_jobs)))
}
for (o in 1:10) {
  allowed_reasons <- if (o <= 6) 1:2 else 1:3
  add_assignment_block("FeatReason_given_FeatCountry", paste0("FeatCountry=", o),
                       design$FeatReason[design$FeatCountry == o], allowed_reasons,
                       rep(1 / length(allowed_reasons), length(allowed_reasons)))
}
write_csv(assignment_diagnostics, file.path(output_dir, "assignment_diagnostics.csv"))

job_denominator <- ifelse(design$FeatEd <= 4, 7, 11)
reason_denominator <- ifelse(design$FeatCountry <= 6, 2, 3)
design$q_profile <- 1 / (7 * job_denominator * 2 * 10 * reason_denominator * 4 * 4 * 5 * 4)
expected_denominators <- c(627200, 940800, 985600, 1478400)
observed_denominators <- sort(unique(round(1 / design$q_profile)))
assert(identical(observed_denominators, expected_denominators), "Unexpected profile-probability strata")

probability_strata <- expand.grid(
  education_stratum = c("low_1_4", "high_5_7"),
  origin_stratum = c("origin_1_6", "origin_7_10"),
  KEEP.OUT.ATTRS = FALSE
)
probability_strata$education_levels <- ifelse(probability_strata$education_stratum == "low_1_4", 4, 3)
probability_strata$jobs_per_education <- ifelse(probability_strata$education_stratum == "low_1_4", 7, 11)
probability_strata$origin_levels <- ifelse(probability_strata$origin_stratum == "origin_1_6", 6, 4)
probability_strata$reasons_per_origin <- ifelse(probability_strata$origin_stratum == "origin_1_6", 2, 3)
probability_strata$admissible_profiles <- with(probability_strata,
  education_levels * jobs_per_education * origin_levels * reasons_per_origin * 2 * 4 * 4 * 5 * 4)
probability_strata$q_profile <- with(probability_strata,
  1 / (7 * jobs_per_education * 2 * 10 * reasons_per_origin * 4 * 4 * 5 * 4))
probability_strata$uniform_mu <- 1 / 936960
probability_strata$uniform_mu_over_q <- probability_strata$uniform_mu / probability_strata$q_profile
assert(abs(sum(probability_strata$admissible_profiles * probability_strata$q_profile) - 1) < 1e-12,
       "Reconstructed assignment law does not sum to one")
assert(sum(probability_strata$admissible_profiles) == 936960, "Probability strata do not enumerate 936,960 profiles")
write_csv(probability_strata, file.path(output_dir, "profile_probability_strata.csv"))

respondent_invariant <- function(values, ids) {
  all(vapply(split(values, ids), function(x) {
    token <- ifelse(is.na(x), "<NA>", as.character(x))
    length(unique(token)) == 1L
  }, logical(1)))
}

covariate_source_columns <- setdiff(allowed_columns, c("CaseID", "contest_no", "weight2", attribute_columns))
covariate_invariance <- vapply(covariate_source_columns,
                               function(nm) respondent_invariant(design[[nm]], design$CaseID), logical(1))
assert(all(covariate_invariance), "At least one purported respondent covariate varies within respondent")

first_row <- !duplicated(design$CaseID)
respondents <- design[first_row, c("CaseID", covariate_source_columns), drop = FALSE]
for (nm in covariate_source_columns) respondents[[nm]] <- as.numeric(respondents[[nm]])

reverse_self_monitor <- function(x) {
  out <- rep(NA_real_, length(x))
  out[x == 1] <- 5
  out[x == 2] <- 4
  out[x == 3] <- 3
  out[x == 4] <- 2
  out[x == 5] <- 1
  out
}
sm_parts <- cbind(
  reverse_self_monitor(respondents$W1_Q1),
  reverse_self_monitor(respondents$W1_Q2),
  reverse_self_monitor(respondents$W1_Q3)
)
respondents$SM_index <- ifelse(rowSums(is.na(sm_parts)) == 0, rowMeans(sm_parts), NA_real_)

outgroup_parts <- cbind(respondents$W1_Q5, respondents$W1_Q6, respondents$W1_Q7)
outgroup_mean <- rowMeans(outgroup_parts, na.rm = TRUE)
outgroup_mean[rowSums(!is.na(outgroup_parts)) == 0] <- NA_real_
ingroup <- ifelse(respondents$ppethm == 1, respondents$W1_Q8a,
                  ifelse(respondents$ppethm == 2, respondents$W1_Q8b, NA_real_))
respondents$ethnocentrism <- ingroup - outgroup_mean
strict_outgroup_mean <- rowMeans(outgroup_parts)
respondents$ethnocentrism_strict <- ingroup - strict_outgroup_mean

primary_raw <- unlist(contract$covariates$primary_raw, use.names = FALSE)
derived_names <- names(contract$covariates$primary_derived)
derived_sensitivity_names <- names(contract$covariates$derived_sensitivity)
feature_semantics <- contract$covariates$feature_semantics
assert(setequal(names(feature_semantics), c(primary_raw, derived_names)),
       "Covariate feature semantics must cover every and only primary raw/derived feature")
covariate_feature_lock <- do.call(rbind, lapply(names(feature_semantics), function(nm) {
  spec <- feature_semantics[[nm]]
  data.frame(
    variable = nm,
    semantic_type = spec$semantic_type,
    primary_encoding = spec$primary_encoding,
    missing_value_rule = spec$missing_value_rule,
    scaling_rule = spec$scaling_rule
  )
}))
write_csv(covariate_feature_lock, file.path(output_dir, "covariate_feature_lock.csv"))

categorical_features <- covariate_feature_lock$variable[
  grepl("categor", covariate_feature_lock$semantic_type)
]
covariate_category_levels <- do.call(rbind, lapply(categorical_features, function(nm) {
  observed_values <- sort(unique(respondents[[nm]][!is.na(respondents[[nm]])]))
  source_labels <- attr(raw[[nm]], "labels")
  source_by_code <- setNames(names(source_labels), as.character(as.numeric(source_labels)))
  labels <- unname(source_by_code[as.character(observed_values)])
  assert(!anyNA(labels), "An enrolled categorical value lacks a locked source label for %s", nm)
  enrolled_counts <- table(factor(respondents[[nm]], levels = observed_values))
  rbind(
    data.frame(
      variable = nm,
      level_order = seq_along(observed_values),
      source_code = as.character(observed_values),
      source_label = labels,
      enrolled_count = as.integer(enrolled_counts),
      is_missing_level = FALSE
    ),
    data.frame(
      variable = nm,
      level_order = length(observed_values) + 1L,
      source_code = "__MISSING__",
      source_label = "Explicit missing level",
      enrolled_count = sum(is.na(respondents[[nm]])),
      is_missing_level = TRUE
    )
  )
}))
write_csv(covariate_category_levels, file.path(output_dir, "covariate_category_levels.csv"))
audit_names <- unique(c(covariate_source_columns, derived_names, derived_sensitivity_names))
covariate_audit <- do.call(rbind, lapply(audit_names, function(nm) {
  x <- respondents[[nm]]
  classification <- if (nm %in% primary_raw) {
    "primary_raw"
  } else if (nm %in% derived_names) {
    "primary_derived"
  } else if (nm %in% derived_sensitivity_names) {
    "derived_sensitivity"
  } else {
    "derivation_component"
  }
  data.frame(
    variable = nm,
    classification = classification,
    respondents = nrow(respondents),
    missing_n = sum(is.na(x)),
    missing_fraction = mean(is.na(x)),
    unique_nonmissing = length(unique(x[!is.na(x)])),
    respondent_invariant = if (nm %in% covariate_source_columns) covariate_invariance[[nm]] else TRUE
  )
}))
expected_missing <- c(SM_index = 11L, W1_Q4 = 13L, ethnocentrism = 251L, ethnocentrism_strict = 271L)
for (nm in names(expected_missing)) {
  observed <- covariate_audit$missing_n[covariate_audit$variable == nm]
  add_check(paste0("covariate_missing_", nm), observed, expected_missing[[nm]],
            length(observed) == 1L && observed == expected_missing[[nm]])
}
write_csv(covariate_audit, file.path(output_dir, "covariate_audit.csv"))

weighted_column_projector <- function(x, p, tol = 1e-12) {
  decomposition <- qr(sqrt(p) * x, LAPACK = FALSE, tol = tol)
  rank <- decomposition$rank
  q <- qr.Q(decomposition, complete = FALSE)[, seq_len(rank), drop = FALSE]
  list(projector = tcrossprod(q), rank = rank)
}

canonical_projector_basis <- function(projector, rank, tol = 1e-10) {
  basis <- matrix(0, nrow(projector), 0L)
  for (j in seq_len(ncol(projector))) {
    candidate <- projector[, j]
    if (ncol(basis)) {
      # Reorthogonalize to make the result stable when projector columns are
      # nearly dependent. The scan order is the locked cell order.
      candidate <- candidate - basis %*% crossprod(basis, candidate)
      candidate <- candidate - basis %*% crossprod(basis, candidate)
    }
    candidate_norm <- sqrt(sum(candidate^2))
    if (candidate_norm > tol) {
      candidate <- candidate / candidate_norm
      pivot <- which(abs(candidate) > tol)[1]
      if (!is.na(pivot) && candidate[pivot] < 0) candidate <- -candidate
      basis <- cbind(basis, candidate)
      if (ncol(basis) == rank) break
    }
  }
  assert(ncol(basis) == rank, "Canonical projector scan recovered %d of %d dimensions", ncol(basis), rank)
  basis
}

make_supported_basis <- function(family, cells, p, nominal_interaction_df) {
  full_cartesian_cells <- switch(
    family,
    education_x_profession = 77L,
    origin_x_application_reason = 30L,
    language_x_education = 28L,
    abort("Unknown interaction family: %s", family)
  )
  cells <- cells[cells$allowed, c("a_code", "b_code", "observed_count"), drop = FALSE]
  cells <- cells[order(cells$a_code, cells$b_code), , drop = FALSE]
  p <- p(cells$a_code, cells$b_code)
  assert(all(is.finite(p) & p > 0) && abs(sum(p) - 1) < 1e-12,
         "Target pair probabilities are invalid for %s", family)
  a_factor <- factor(cells$a_code, levels = sort(unique(cells$a_code)))
  b_factor <- factor(cells$b_code, levels = sort(unique(cells$b_code)))
  additive <- stats::model.matrix(~ a_factor + b_factor)
  additive_geometry <- weighted_column_projector(additive, p)
  additive_rank <- additive_geometry$rank
  interaction_rank <- nrow(cells) - additive_rank
  whitened_interaction_projector <- diag(nrow(cells)) - additive_geometry$projector
  whitened_interaction_projector <- (whitened_interaction_projector + t(whitened_interaction_projector)) / 2
  q_interaction <- canonical_projector_basis(whitened_interaction_projector, interaction_rank)
  H <- q_interaction / sqrt(p)
  C <- H * p

  # Rebuild the same additive space using unrelated contrast codings. Only the
  # cell-function space may matter; treatment baselines and contrast matrices
  # must not change the projector or the locked canonical representation.
  additive_helmert_sum <- stats::model.matrix(
    ~ a_factor + b_factor,
    contrasts.arg = list(
      a_factor = stats::contr.helmert(nlevels(a_factor)),
      b_factor = stats::contr.sum(nlevels(b_factor))
    )
  )
  additive_last_baseline <- stats::model.matrix(
    ~ a_factor + b_factor,
    contrasts.arg = list(
      a_factor = stats::contr.treatment(nlevels(a_factor), base = nlevels(a_factor)),
      b_factor = stats::contr.treatment(nlevels(b_factor), base = nlevels(b_factor))
    )
  )
  a_factor_reversed <- factor(cells$a_code, levels = rev(sort(unique(cells$a_code))))
  b_factor_reversed <- factor(cells$b_code, levels = rev(sort(unique(cells$b_code))))
  additive_reversed_levels <- stats::model.matrix(~ a_factor_reversed + b_factor_reversed)
  rotation <- diag(ncol(additive))
  rotation[row(rotation) < col(rotation)] <- 0.05
  additive_rotated <- additive %*% rotation
  alternative_additive_designs <- list(
    helmert_sum = additive_helmert_sum,
    last_baseline = additive_last_baseline,
    reversed_levels = additive_reversed_levels,
    nonsingular_rotation = additive_rotated
  )
  coding_errors <- do.call(rbind, lapply(alternative_additive_designs, function(alternative) {
    alt_geometry <- weighted_column_projector(alternative, p)
    alt_interaction_projector <- diag(nrow(cells)) - alt_geometry$projector
    alt_interaction_projector <- (alt_interaction_projector + t(alt_interaction_projector)) / 2
    alt_q_interaction <- canonical_projector_basis(alt_interaction_projector, interaction_rank)
    c(
      projector = max(abs(whitened_interaction_projector - alt_interaction_projector)),
      basis = max(abs(q_interaction - alt_q_interaction))
    )
  }))
  coding_projector_error <- max(coding_errors[, "projector"])
  coding_basis_error <- max(coding_errors[, "basis"])

  function_projector <- H %*% t(H * p)
  gram_error <- max(abs(crossprod(H, H * p) - diag(ncol(H))))
  additive_error <- max(abs(crossprod(additive, H * p)))
  sum_error <- max(abs(colSums(C)))
  projector_idempotence_error <- max(abs(function_projector %*% function_projector - function_projector))
  projector_additive_error <- max(abs(function_projector %*% additive))
  observed_cell_projector <- diag(as.numeric(cells$observed_count > 0))
  row_space_residual <- max(abs((diag(nrow(cells)) - observed_cell_projector) %*% C))
  assert(gram_error < 1e-10, "Weighted Gram check failed for %s", family)
  assert(additive_error < 1e-10, "Additive orthogonality failed for %s", family)
  assert(sum_error < 1e-10, "Contrast sum check failed for %s", family)
  assert(coding_projector_error < 1e-10, "Contrast-coding projector check failed for %s", family)
  assert(coding_basis_error < 1e-10, "Contrast-coding canonical-basis check failed for %s", family)
  assert(projector_idempotence_error < 1e-10, "Interaction-projector idempotence failed for %s", family)
  assert(projector_additive_error < 1e-10, "Interaction projector does not annihilate additive functions for %s", family)
  assert(row_space_residual < 1e-10, "Supported contrasts are outside the observed-cell row space for %s", family)
  coordinates <- do.call(rbind, lapply(seq_len(ncol(C)), function(j) data.frame(
    family = family,
    coordinate = sprintf("%s_%03d", family, j),
    a_code = cells$a_code,
    b_code = cells$b_code,
    target_pair_probability = p,
    basis_function = H[, j],
    direct_contrast_weight = C[, j]
  )))
  projector_index <- expand.grid(
    row_cell = seq_len(nrow(cells)),
    column_cell = seq_len(nrow(cells)),
    KEEP.OUT.ATTRS = FALSE
  )
  projector <- data.frame(
    family = family,
    row_a_code = cells$a_code[projector_index$row_cell],
    row_b_code = cells$b_code[projector_index$row_cell],
    column_a_code = cells$a_code[projector_index$column_cell],
    column_b_code = cells$b_code[projector_index$column_cell],
    projector_value = as.vector(function_projector)
  )
  summary <- data.frame(
    family = family,
    cartesian_cells = full_cartesian_cells,
    admissible_cells = nrow(cells),
    observed_support_rank = sum(cells$observed_count > 0),
    minimum_cell_count = min(cells$observed_count),
    additive_rank = additive_rank,
    nominal_full_product_interaction_df = nominal_interaction_df,
    off_support_rank = nominal_interaction_df - ncol(H),
    identified_interaction_df = ncol(H),
    weighted_gram_error = gram_error,
    additive_orthogonality_error = additive_error,
    contrast_sum_error = sum_error,
    coding_projector_error = coding_projector_error,
    coding_basis_error = coding_basis_error,
    projector_idempotence_error = projector_idempotence_error,
    projector_additive_error = projector_additive_error,
    row_space_residual = row_space_residual
  )
  list(summary = summary, coordinates = coordinates, projector = projector)
}

ep_basis <- make_supported_basis(
  "education_x_profession", ep_cells,
  function(e, j) (1 / 7) * ifelse(e <= 4, 1 / 7, 1 / 11), 60
)
or_basis <- make_supported_basis(
  "origin_x_application_reason", or_cells,
  function(o, r) (1 / 10) * ifelse(o <= 6, 1 / 2, 1 / 3), 18
)
le_basis <- make_supported_basis(
  "language_x_education", le_cells,
  function(l, e) rep(1 / 28, length(l)), 18
)
family_identification <- rbind(ep_basis$summary, or_basis$summary, le_basis$summary)
contrast_coordinates <- rbind(ep_basis$coordinates, or_basis$coordinates, le_basis$coordinates)
family_projectors <- rbind(ep_basis$projector, or_basis$projector, le_basis$projector)
assert(identical(family_identification$identified_interaction_df, c(44L, 12L, 18L)),
       "Supported-family dimensions differ from the lock")
assert(sum(family_identification$identified_interaction_df) == 74L, "Expected 74 released-support family coordinates")
write_csv(family_identification, file.path(output_dir, "family_identification.csv"))
write_csv(contrast_coordinates, file.path(output_dir, "contrast_coordinates.csv"))
write_csv(family_projectors, file.path(output_dir, "family_projectors.csv"))

add_check("all_loaded_columns_outcome_free", sum(loaded_columns$outcome_loaded), 0, !any(loaded_columns$outcome_loaded))
add_check("supported_family_coordinate_count", sum(family_identification$identified_interaction_df), 74,
          sum(family_identification$identified_interaction_df) == 74)

assert(all(validation$pass), paste("Design audit failed:", paste(validation$check[!validation$pass], collapse = ", ")))
write_csv(validation, file.path(output_dir, "design_validation.csv"))

output_files <- file.path(output_dir, expected_csv_files)
assert(all(file.exists(output_files)), "One or more expected audit outputs were not written")
manifest <- list(
  schema_version = 1,
  created_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  study_id = contract$study_id,
  contract_status = contract$status,
  gate_result = contract$decision$result,
  outcome_values_read = FALSE,
  models_fit = FALSE,
  design_complete_respondents = length(unique(design$CaseID)),
  design_complete_tasks = length(pair_rows),
  design_complete_profile_rows = nrow(design),
  reconstructed_admissible_profiles = 936960,
  assignment_reconstruction_status = contract$assignment$reconstruction_status,
  family_dimensions = setNames(as.list(family_identification$identified_interaction_df), family_identification$family),
  unresolved_blockers = c(
    contract$support$discrepancy$statement,
    "The within-pair q x q opponent law is a prospective assumption, not generator-certified."
  ),
  output_files = lapply(output_files, function(path) list(
    file = basename(path),
    bytes = unname(file.info(path)$size),
    md5 = unname(tools::md5sum(path))
  ))
)
yaml::write_yaml(manifest, file.path(output_dir, "design_audit_manifest.yml"))

cat(sprintf(
  paste0(
    "Immigrant-admission design audit: AMBER\n",
    "  outcome values read: no\n",
    "  design universe: %d respondents, %d tasks, %d profiles\n",
    "  released admissible profiles: %d\n",
    "  supported interaction dimensions: %s\n",
    "  blockers: article/release support conflict; within-pair q x q law uncertified\n",
    "  outputs: %s\n"
  ),
  length(unique(design$CaseID)), length(pair_rows), nrow(design), 936960,
  paste(paste0(family_identification$family, "=", family_identification$identified_interaction_df), collapse = ", "),
  normalizePath(output_dir, mustWork = TRUE)
))
