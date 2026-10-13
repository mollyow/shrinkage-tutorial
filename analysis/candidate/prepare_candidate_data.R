#!/usr/bin/env Rscript

# Prepare the prespecified candidate-recruitment conjoint application.
# This script reads only the published cleaned replication CSV. It does not
# accept an input-file argument by design.

required_packages <- c("yaml", "readr", "tibble")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace,
                                               quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing_packages)) {
  stop("Install required packages before preparing the data: ",
       paste(missing_packages, collapse = ", "), call. = FALSE)
}

stopf <- function(fmt, ...) stop(sprintf(fmt, ...), call. = FALSE)
assert_that <- function(ok, fmt, ...) {
  if (!isTRUE(ok)) stopf(fmt, ...)
  invisible(TRUE)
}

script_file <- function() {
  flag <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (!length(flag)) stop("Run this file with Rscript.", call. = FALSE)
  normalizePath(sub("^--file=", "", flag[[1]]), mustWork = TRUE)
}

source(file.path(dirname(script_file()), "contract_helpers.R"), local = TRUE)

parse_args <- function(args) {
  if (any(grepl("^--input($|=)", args))) {
    stop("This confirmatory preparation script deliberately has no --input option. ",
         "It reads only the cleaned replication CSV locked in estimands.yml.",
         call. = FALSE)
  }
  out <- list(output_dir = NULL)
  i <- 1L
  while (i <= length(args)) {
    if (grepl("^--output-dir=", args[[i]])) {
      out$output_dir <- sub("^--output-dir=", "", args[[i]])
    } else if (identical(args[[i]], "--output-dir")) {
      i <- i + 1L
      if (i > length(args)) stop("--output-dir requires a value.", call. = FALSE)
      out$output_dir <- args[[i]]
    } else {
      stop("Unknown argument: ", args[[i]], call. = FALSE)
    }
    i <- i + 1L
  }
  out
}

normalise_name <- function(x) tolower(gsub("[^a-z0-9]", "", x))

resolve_column <- function(source_names, aliases, role) {
  source_key <- normalise_name(source_names)
  alias_key <- unique(normalise_name(unlist(aliases, use.names = FALSE)))
  hit <- which(source_key %in% alias_key)
  if (!length(hit)) {
    stopf("Could not detect the %s column. Accepted aliases: %s.", role,
          paste(unlist(aliases), collapse = ", "))
  }
  if (length(hit) > 1L) {
    stopf("Ambiguous %s column: %s all match the locked aliases.", role,
          paste(source_names[hit], collapse = ", "))
  }
  source_names[[hit]]
}

as_numeric_strict <- function(x, role) {
  out <- suppressWarnings(as.numeric(as.character(x)))
  bad <- is.na(out) & !is.na(x)
  if (any(bad)) stopf("Column mapped to %s contains nonnumeric values.", role)
  out
}

match_locked_levels <- function(x, spec, attribute, tolerance) {
  observed <- as_numeric_strict(x, attribute)
  locked <- as.numeric(unlist(spec$values, use.names = FALSE))
  labels <- as.character(unlist(spec$labels, use.names = FALSE))
  assert_that(length(locked) == length(labels),
              "Locked values and labels differ in length for %s.", attribute)
  nearest <- vapply(observed, function(value) {
    if (is.na(value)) return(NA_integer_)
    distance <- abs(locked - value)
    if (min(distance) > tolerance) return(NA_integer_)
    which.min(distance)
  }, integer(1))
  if (anyNA(nearest)) {
    unexpected <- sort(unique(observed[is.na(nearest)]))
    stopf("%s contains values outside the locked levels (tolerance %.3g): %s.",
          attribute, tolerance, paste(unexpected, collapse = ", "))
  }
  list(
    factor = factor(labels[nearest], levels = labels, ordered = isTRUE(spec$ordered)),
    value = locked[nearest],
    observed = observed,
    locked_values = locked,
    labels = labels
  )
}

profile_key <- function(x, attributes) {
  do.call(paste, c(lapply(x[attributes], as.character), sep = "\u001f"))
}

orient_columns <- function(x, tolerance = 1e-12) {
  for (j in seq_len(ncol(x))) {
    first <- which(abs(x[, j]) > tolerance)[1]
    if (!is.na(first) && x[first, j] < 0) x[, j] <- -x[, j]
  }
  x
}

build_family_contrasts <- function(target, family) {
  family_id <- as.character(family$id)
  family_label <- as.character(family$label)
  pair <- as.character(unlist(family$attributes, use.names = FALSE))
  if (length(pair) != 2L) stopf("Family %s must name exactly two attributes.", family_id)
  a <- pair[[1]]
  b <- pair[[2]]
  la <- levels(target[[a]])
  lb <- levels(target[[b]])
  na <- length(la)
  nb <- length(lb)
  n_grid <- nrow(target)
  n_other <- n_grid / (na * nb)
  assert_that(n_other == as.integer(n_other),
              "Target grid does not factor cleanly for family %s.", family_id)

  surface <- matrix(0, nrow = n_grid, ncol = na * nb)
  colnames(surface) <- rep("", na * nb)
  cell_meta <- vector("list", na * nb)
  k <- 0L
  for (ia in seq_along(la)) {
    for (ib in seq_along(lb)) {
      k <- k + 1L
      is_a <- as.character(target[[a]]) == la[[ia]]
      is_b <- as.character(target[[b]]) == lb[[ib]]
      surface[, k] <-
        as.numeric(is_a & is_b) / n_other -
        as.numeric(is_a) / (nb * n_other) -
        as.numeric(is_b) / (na * n_other) +
        rep(1 / n_grid, n_grid)
      cell_name <- sprintf("%s__cell_%02d_%02d", family_id, ia, ib)
      colnames(surface)[k] <- cell_name
      cell_meta[[k]] <- tibble::tibble(
        family = family_id,
        family_label = family_label,
        cell = cell_name,
        attribute_a = a,
        level_a_index = ia,
        level_a = la[[ia]],
        attribute_b = b,
        level_b_index = ib,
        level_b = lb[[ib]]
      )
    }
  }
  cell_meta <- do.call(rbind, cell_meta)

  qr_surface <- qr(surface, tol = 1e-10, LAPACK = FALSE)
  expected_rank <- (na - 1L) * (nb - 1L)
  assert_that(qr_surface$rank == expected_rank,
              "Family %s has rank %d; expected %d.", family_id,
              qr_surface$rank, expected_rank)
  coordinates <- qr.Q(qr_surface, complete = FALSE)[, seq_len(expected_rank), drop = FALSE]
  coordinates <- orient_columns(coordinates)
  coordinate_names <- sprintf("%s__coord_%02d", family_id, seq_len(expected_rank))
  colnames(coordinates) <- coordinate_names
  mapping <- solve(crossprod(coordinates), crossprod(coordinates, surface))
  rownames(mapping) <- coordinate_names
  colnames(mapping) <- colnames(surface)
  reconstruction_error <- max(abs(surface - coordinates %*% mapping))
  assert_that(reconstruction_error < 1e-10,
              "Coordinate reduction failed for family %s (error %.3g).",
              family_id, reconstruction_error)
  orthogonality_error <- max(abs(crossprod(coordinates) - diag(expected_rank)))
  assert_that(orthogonality_error < 1e-10,
              "Coordinate basis is not orthonormal for family %s (error %.3g).",
              family_id, orthogonality_error)

  coordinate_meta <- tibble::tibble(
    family = family_id,
    family_label = family_label,
    coordinate = coordinate_names,
    coordinate_index = seq_len(expected_rank),
    dimension = expected_rank,
    attribute_a = a,
    attribute_b = b
  )

  did <- NULL
  did_mapping <- NULL
  if (isTRUE(family$report_classic_did)) {
    assert_that(na == 2L && nb == 2L,
                "Classic DID requested for non-binary family %s.", family_id)
    sign <- ifelse(
      (as.character(target[[a]]) == la[[1]] & as.character(target[[b]]) == lb[[1]]) |
        (as.character(target[[a]]) == la[[2]] & as.character(target[[b]]) == lb[[2]]),
      1, -1
    )
    did <- sign / n_other
    names(did) <- target$profile_id
    did_mapping <- drop(crossprod(coordinates, did))
    names(did_mapping) <- coordinate_names
    assert_that(max(abs(did - drop(coordinates %*% did_mapping))) < 1e-10,
                "Classic DID did not lie in the coordinate span for %s.", family_id)
  }

  list(
    id = family_id,
    label = family_label,
    attributes = pair,
    surface = surface,
    cells = cell_meta,
    coordinates = coordinates,
    coordinate_meta = coordinate_meta,
    cell_mapping = mapping,
    classic_did = did,
    classic_did_mapping = did_mapping,
    classic_did_name = if (!is.null(did)) paste0(family_id, "__classic_did") else NULL,
    n_other = n_other,
    rank = expected_rank,
    reconstruction_error = reconstruction_error,
    orthogonality_error = orthogonality_error
  )
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
script <- script_file()
candidate_dir <- dirname(script)
repo_root <- normalizePath(file.path(candidate_dir, "..", ".."), mustWork = TRUE)
lock_path <- file.path(candidate_dir, "estimands.yml")
cfg <- yaml::read_yaml(lock_path)
lock_md5 <- unname(tools::md5sum(lock_path))
contract <- validate_candidate_contract(cfg)

input_relative <- as.character(cfg$data$allowed_input)
assert_that(!grepl(as.character(cfg$data$forbidden_input_pattern), input_relative,
                   fixed = TRUE),
            "The locked input points into the forbidden raw YouGov directory.")
input_path <- file.path(repo_root, input_relative)
assert_that(file.exists(input_path), "Locked replication input not found: %s", input_path)
input_path <- normalizePath(input_path, mustWork = TRUE)
expected_path <- normalizePath(file.path(repo_root, input_relative), mustWork = TRUE)
assert_that(identical(input_path, expected_path), "Resolved input differs from the locked path.")
input_md5 <- unname(tools::md5sum(input_path))
assert_that(identical(input_md5, as.character(cfg$data$md5)),
            "Input checksum mismatch. Expected %s but found %s. Refusing to prepare data.",
            as.character(cfg$data$md5), input_md5)

output_dir <- args$output_dir
if (is.null(output_dir)) output_dir <- file.path(candidate_dir, "derived")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
output_dir <- normalizePath(output_dir, mustWork = TRUE)

raw <- readr::read_csv(input_path, na = c("", "NA", "."),
                       show_col_types = FALSE, progress = FALSE,
                       name_repair = "minimal")
source_names <- names(raw)

role_names <- vapply(names(cfg$roles), function(role) {
  spec <- cfg$roles[[role]]
  resolve_column(source_names, unique(c(spec$canonical, unlist(spec$aliases))), role)
}, character(1))
attribute_names <- vapply(names(cfg$attributes), function(attribute) {
  spec <- cfg$attributes[[attribute]]
  resolve_column(source_names, unique(c(attribute, unlist(spec$aliases))), attribute)
}, character(1))
all_detected <- c(role_names, attribute_names)
assert_that(!anyDuplicated(all_detected),
            "A source column was mapped to more than one locked role or attribute.")

dat <- tibble::tibble(
  id = as.character(raw[[role_names[["respondent"]]]]),
  task = as_numeric_strict(raw[[role_names[["task"]]]], "task"),
  weight = as_numeric_strict(raw[[role_names[["weight"]]]], "weight"),
  interest = as_numeric_strict(raw[[role_names[["outcome"]]]], "interest")
)
attributes <- names(cfg$attributes)
assert_that(identical(attributes, contract$factor_names),
            "Prepared attribute order differs from the validated lockfile contract.")
tolerance <- as.numeric(cfg$preparation$numeric_tolerance)
level_lock <- vector("list", length(attributes))
for (j in seq_along(attributes)) {
  attribute <- attributes[[j]]
  attribute_spec <- cfg$attributes[[attribute]]
  source_column <- attribute_names[[attribute]]
  matched <- match_locked_levels(raw[[source_column]],
                                 attribute_spec, attribute, tolerance)
  dat[[attribute]] <- matched$factor
  dat[[paste0(attribute, "__value")]] <- matched$value
  detected <- vapply(matched$locked_values, function(value) {
    candidates <- unique(matched$observed[abs(matched$observed - value) <= tolerance])
    if (!length(candidates)) NA_real_ else candidates[[1]]
  }, numeric(1))
  level_lock[[j]] <- tibble::tibble(
    attribute = attribute,
    level_index = seq_along(matched$locked_values),
    detected_value = detected,
    locked_value = matched$locked_values,
    label = matched$labels,
    ordered = isTRUE(attribute_spec$ordered),
    source_column = source_column
  )
}
level_lock <- do.call(rbind, level_lock)

expected <- cfg$data$expected
core_columns <- c("id", "task", "weight", "interest", attributes)
checks <- tibble::tibble(check = character(), observed = character(), expected = character())
add_check <- function(name, observed, expected_value, ok) {
  if (!isTRUE(ok)) stopf("Validation failed for %s: observed %s; expected %s.",
                         name, as.character(observed), as.character(expected_value))
  checks <<- rbind(checks, tibble::tibble(
    check = name, observed = as.character(observed), expected = as.character(expected_value)
  ))
}

add_check("rows", nrow(dat), expected$rows, nrow(dat) == as.integer(expected$rows))
add_check("respondents", length(unique(dat$id)), expected$respondents,
          length(unique(dat$id)) == as.integer(expected$respondents))
add_check("core_missing_values", sum(is.na(dat[core_columns])), 0,
          sum(is.na(dat[core_columns])) == 0L)
add_check("finite_positive_weights", sum(is.finite(dat$weight) & dat$weight > 0), nrow(dat),
          all(is.finite(dat$weight) & dat$weight > 0))
weight_diagnostics <- assert_respondent_invariant_weights(
  dat$id, dat$weight, tolerance = contract$weight_tolerance
)
add_check(
  "respondent_invariant_survey_weights",
  signif(weight_diagnostics$maximum_within_respondent_range, 8),
  paste0("<= ", format(contract$weight_tolerance, scientific = TRUE)),
  weight_diagnostics$invariant
)
add_check("outcome_range", sprintf("[%.6g, %.6g]", min(dat$interest), max(dat$interest)),
          "within [0, 1]", all(is.finite(dat$interest) & dat$interest >= 0 & dat$interest <= 1))
add_check("integer_task", sum(dat$task == as.integer(dat$task)), nrow(dat),
          all(dat$task == as.integer(dat$task)))
dat$task <- as.integer(dat$task)
respondent_task_count <- table(dat$id)
add_check("tasks_per_respondent", paste(range(respondent_task_count), collapse = "-"),
          expected$tasks_per_respondent,
          all(respondent_task_count == as.integer(expected$tasks_per_respondent)))
expected_tasks <- seq_len(as.integer(expected$tasks_per_respondent))
add_check("task_positions", paste(sort(unique(dat$task)), collapse = ","),
          paste(expected_tasks, collapse = ","),
          identical(sort(unique(dat$task)), expected_tasks))
add_check("unique_respondent_task", sum(duplicated(dat[c("id", "task")])), 0,
          !anyDuplicated(dat[c("id", "task")]))

target_grid <- do.call(expand.grid, c(
  lapply(attributes, function(attribute) {
    factor(levels(dat[[attribute]]), levels = levels(dat[[attribute]]),
           ordered = is.ordered(dat[[attribute]]))
  }),
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
))
names(target_grid) <- attributes
target_grid <- tibble::as_tibble(target_grid)
for (attribute in attributes) {
  target_grid[[attribute]] <- factor(
    as.character(target_grid[[attribute]]), levels = levels(dat[[attribute]]),
    ordered = is.ordered(dat[[attribute]])
  )
}
target_grid$profile_id <- sprintf("p%04d", seq_len(nrow(target_grid)))
target_grid$target_weight <- rep(1 / nrow(target_grid), nrow(target_grid))
add_check("target_profiles", nrow(target_grid), expected$profiles,
          nrow(target_grid) == as.integer(expected$profiles))
add_check("uniform_target_weight_sum", sum(target_grid$target_weight), 1,
          abs(sum(target_grid$target_weight) - 1) < 1e-12)

target_key <- profile_key(target_grid, attributes)
observed_key <- profile_key(dat, attributes)
dat$profile_id <- target_grid$profile_id[match(observed_key, target_key)]
add_check("mapped_profile_ids", sum(!is.na(dat$profile_id)), nrow(dat),
          !anyNA(dat$profile_id))
profile_counts <- table(factor(dat$profile_id, levels = target_grid$profile_id))
add_check("observed_profiles", sum(profile_counts > 0), expected$profiles,
          sum(profile_counts > 0) == as.integer(expected$profiles))

set.seed(as.integer(cfg$preparation$fold_seed))
respondents <- sample(sort(unique(dat$id)))
nfolds <- contract$outer_folds
fold_assignments <- tibble::tibble(
  id = respondents,
  fold = rep(seq_len(nfolds), length.out = length(respondents))
)
dat$fold <- fold_assignments$fold[match(dat$id, fold_assignments$id)]
fold_cluster_counts <- table(fold_assignments$fold)
add_check("respondent_fold_balance", paste(fold_cluster_counts, collapse = ","),
          paste(rep(length(respondents) / nfolds, nfolds), collapse = ","),
          length(respondents) %% nfolds == 0L && length(unique(fold_cluster_counts)) == 1L)
add_check("respondent_fold_integrity", max(vapply(split(dat$fold, dat$id),
                                                    function(x) length(unique(x)), integer(1))),
          1, all(vapply(split(dat$fold, dat$id),
                        function(x) length(unique(x)) == 1L, logical(1))))

families <- lapply(cfg$families, function(family) build_family_contrasts(target_grid, family))
names(families) <- vapply(families, `[[`, character(1), "id")
contrast_matrix <- do.call(cbind, lapply(families, `[[`, "coordinates"))
expected_coordinates <- sum(vapply(families, `[[`, integer(1), "rank"))
add_check("amie_coordinate_dimension", ncol(contrast_matrix), contract$coordinate_dimension,
          ncol(contrast_matrix) == contract$coordinate_dimension &&
            expected_coordinates == contract$coordinate_dimension)
add_check("combined_coordinate_orthonormality",
          signif(max(abs(crossprod(contrast_matrix) - diag(ncol(contrast_matrix)))), 5),
          "< 1e-10",
          max(abs(crossprod(contrast_matrix) - diag(ncol(contrast_matrix)))) < 1e-10)

column_map <- tibble::tibble(
  kind = c(rep("role", length(role_names)), rep("attribute", length(attribute_names))),
  canonical = c(names(role_names), names(attribute_names)),
  source_column = c(unname(role_names), unname(attribute_names))
)
profile_counts_out <- tibble::tibble(
  profile_id = target_grid$profile_id,
  observed_rows = as.integer(profile_counts),
  target_weight = target_grid$target_weight
)

coordinate_meta <- do.call(rbind, lapply(families, `[[`, "coordinate_meta"))
cell_meta <- do.call(rbind, lapply(families, `[[`, "cells"))
coordinate_weights <- do.call(rbind, lapply(families, function(fam) {
  tibble::tibble(
    profile_id = rep(target_grid$profile_id, times = ncol(fam$coordinates)),
    family = fam$id,
    coordinate = rep(colnames(fam$coordinates), each = nrow(fam$coordinates)),
    direct_weight = as.vector(fam$coordinates)
  )
}))
cell_weights <- do.call(rbind, lapply(families, function(fam) {
  tibble::tibble(
    profile_id = rep(target_grid$profile_id, times = ncol(fam$surface)),
    family = fam$id,
    cell = rep(colnames(fam$surface), each = nrow(fam$surface)),
    direct_weight = as.vector(fam$surface)
  )
}))
contrast_mapping <- do.call(rbind, lapply(families, function(fam) {
  tibble::tibble(
    family = fam$id,
    coordinate = rep(rownames(fam$cell_mapping), times = ncol(fam$cell_mapping)),
    cell = rep(colnames(fam$cell_mapping), each = nrow(fam$cell_mapping)),
    mapping_weight = as.vector(fam$cell_mapping)
  )
}))
did_weights <- do.call(rbind, Filter(Negate(is.null), lapply(families, function(fam) {
  if (is.null(fam$classic_did)) return(NULL)
  family_id <- fam$id
  did_name <- fam$classic_did_name
  tibble::tibble(
    profile_id = target_grid$profile_id,
    family = family_id,
    contrast = did_name,
    direct_weight = unname(fam$classic_did)
  )
})))

bundle <- list(
  data = dat,
  target_grid = target_grid,
  attributes = attributes,
  column_map = column_map,
  level_lock = level_lock,
  fold_assignments = fold_assignments,
  contrast_matrix = contrast_matrix,
  coordinate_meta = coordinate_meta,
  cell_meta = cell_meta,
  contrast_families = families,
  validation = checks,
  contract = contract,
  config = cfg,
  provenance = list(
    input_relative = input_relative,
    input_md5 = input_md5,
    estimands_relative = file.path("analysis", "candidate", "estimands.yml"),
    estimands_md5 = lock_md5,
    assignment_mechanism = cfg$assignment$mechanism,
    assignment_verification_status = cfg$assignment$verification_status,
    assignment_probability = contract$assignment_probability,
    replication_doi = cfg$assignment$design_source$replication_doi,
    replication_license = cfg$assignment$design_source$license_status,
    prepared_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
    r_version = R.version.string
  )
)

saveRDS(bundle, file.path(output_dir, "candidate_prepared.rds"), version = 3)
readr::write_csv(column_map, file.path(output_dir, "candidate_column_map.csv"))
readr::write_csv(level_lock, file.path(output_dir, "candidate_level_lock.csv"))
readr::write_csv(checks, file.path(output_dir, "candidate_validation.csv"))
readr::write_csv(fold_assignments[order(fold_assignments$fold, fold_assignments$id), ],
                 file.path(output_dir, "candidate_folds.csv"))
readr::write_csv(target_grid, file.path(output_dir, "candidate_target_grid.csv"))
readr::write_csv(profile_counts_out, file.path(output_dir, "candidate_profile_counts.csv"))
readr::write_csv(coordinate_meta, file.path(output_dir, "candidate_coordinate_metadata.csv"))
readr::write_csv(cell_meta, file.path(output_dir, "candidate_cell_metadata.csv"))
readr::write_csv(coordinate_weights, file.path(output_dir, "candidate_contrast_coordinates.csv"))
readr::write_csv(cell_weights, file.path(output_dir, "candidate_contrast_cells.csv"))
readr::write_csv(contrast_mapping, file.path(output_dir, "candidate_contrast_mapping.csv"))
readr::write_csv(did_weights, file.path(output_dir, "candidate_classic_did_contrasts.csv"))
yaml::write_yaml(bundle$provenance, file.path(output_dir, "candidate_preparation_manifest.yml"))
yaml::write_yaml(list(
  assumption_acknowledged = cfg$assignment$assumption_acknowledged,
  mechanism = cfg$assignment$mechanism,
  verification_status = cfg$assignment$verification_status,
  across_tasks = cfg$assignment$across_tasks,
  row_marginal_profile_probability = contract$assignment_probability,
  design_source = cfg$assignment$design_source,
  exact_detected_levels = split(level_lock[c("detected_value", "locked_value", "label")],
                                level_lock$attribute)
), file.path(output_dir, "candidate_design_and_levels_lock.yml"))

message("Prepared candidate application data in: ", output_dir)
message("Validated ", nrow(dat), " rows, ", length(unique(dat$id)),
        " respondents, ", nrow(target_grid), " profiles, and ",
        ncol(contrast_matrix), " full-rank AMIE coordinates.")
