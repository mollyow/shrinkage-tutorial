# Shared utilities for the factorial_ml_comparison_v1 engine.
#
# Everything here is deterministic and outcome-free. Seeds follow the frozen
# simulation rule; respondent folds reuse the balanced string-hash algorithm
# from the outcome-blind design audits.

`%||%` <- function(x, y) if (is.null(x)) y else x

cmp_assert <- function(condition, ...) {
  if (!isTRUE(condition)) stop(sprintf(...), call. = FALSE)
  invisible(TRUE)
}

cmp_repo_root <- function(start = getwd()) {
  path <- normalizePath(start, mustWork = TRUE)
  repeat {
    if (file.exists(file.path(path, "DESCRIPTION")) &&
        dir.exists(file.path(path, "analysis", "comparison"))) {
      return(path)
    }
    parent <- dirname(path)
    cmp_assert(!identical(parent, path), "Could not locate the repository root.")
    path <- parent
  }
}

cmp_sha256 <- function(path) digest::digest(file = path, algo = "sha256")
cmp_md5 <- function(path) digest::digest(file = path, algo = "md5")

# Write an object atomically so an interrupted job never leaves a partial file.
cmp_save_rds <- function(object, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp-", Sys.getpid())
  saveRDS(object, tmp)
  cmp_assert(file.rename(tmp, path), "Could not move %s into place.", path)
  invisible(path)
}

cmp_write_csv <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  utils::write.csv(x, path, row.names = FALSE, na = "")
  invisible(path)
}

cmp_write_yaml <- function(x, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  writeLines(yaml::as.yaml(
    x, handlers = list(logical = yaml::verbatim_logical),
    indent.mapping.sequence = TRUE
  ), path)
  invisible(path)
}

cmp_utc_now <- function() format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

# Phase offsets for test fixtures sit outside every frozen phase so that
# implementation tests can never consume a calibration, smoke, or production
# random-number stream.
CMP_TEST_PHASE_OFFSET <- 90000000L

cmp_phase_offset <- function(sim_config, phase) {
  if (identical(phase, "test")) return(CMP_TEST_PHASE_OFFSET)
  offset <- sim_config$common$phase_offsets[[phase]]
  cmp_assert(!is.null(offset), "Unknown simulation phase `%s`.", phase)
  as.integer(offset)
}

cmp_purpose_offset <- function(sim_config, purpose) {
  offset <- sim_config$common$purpose_offsets[[purpose]]
  cmp_assert(!is.null(offset), "Unknown seed purpose `%s`.", purpose)
  as.integer(offset)
}

# master_seed + phase_offset + 100000*cell_id + replication + purpose_offset
cmp_seed <- function(sim_config, phase, cell_id, replication, purpose) {
  cmp_assert(replication >= 1L && replication < 10000L,
             "Replication %s is outside the seed rule's range.", replication)
  value <- as.numeric(sim_config$common$master_seed) +
    cmp_phase_offset(sim_config, phase) +
    100000 * as.numeric(cell_id) + as.numeric(replication) +
    cmp_purpose_offset(sim_config, purpose)
  cmp_assert(value < .Machine$integer.max, "Seed overflow.")
  as.integer(value)
}

# Derived seeds for individual fits: a stable hash of the parent seed and the
# fit's identity, so that adding a method never shifts another method's stream.
cmp_hashed_seed <- function(...) {
  key <- paste(vapply(list(...), function(x) paste(as.character(x), collapse = ","),
                      character(1)), collapse = "|")
  hex <- substr(digest::digest(key, algo = "sha256", serialize = FALSE), 1L, 7L)
  as.integer(strtoi(hex, base = 16L) %% 2147483000L) + 1L
}

cmp_with_seed <- function(seed, code) {
  old_exists <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (old_exists) old <- get(".Random.seed", envir = .GlobalEnv)
  on.exit({
    if (old_exists) {
      assign(".Random.seed", old, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(as.integer(seed), kind = "Mersenne-Twister", normal.kind = "Inversion",
           sample.kind = "Rejection")
  force(code)
}

# ---------------------------------------------------------------------------
# Respondent folds (same algorithm as the application design audits)
# ---------------------------------------------------------------------------

cmp_stable_string_hash <- function(x, salt, multiplier = 131, reverse_codes = FALSE) {
  vapply(x, function(token) {
    h <- as.numeric(salt)
    codes <- utf8ToInt(enc2utf8(token))
    if (reverse_codes) codes <- rev(codes)
    for (code in codes) h <- (h * multiplier + code) %% 2147483629
    h
  }, numeric(1), USE.NAMES = FALSE)
}

cmp_balanced_folds <- function(ids, k, salt, multiplier = 131, reverse_codes = FALSE) {
  ids <- as.character(ids)
  ord <- order(cmp_stable_string_hash(ids, salt, multiplier, reverse_codes), ids)
  out <- integer(length(ids))
  out[ord] <- rep(seq_len(k), length.out = length(ids))
  out
}

# Outer labels use the audit salt rule; inner labels use salt + 104729 with
# multiplier 257 and reversed codes, exactly as in the application audits.
cmp_respondent_folds <- function(ids, outer_k, inner_k, salt) {
  ids <- as.character(ids)
  cmp_assert(!anyDuplicated(ids), "Respondent identifiers must be unique.")
  out <- data.frame(
    id = ids,
    outer_fold = cmp_balanced_folds(ids, outer_k, salt),
    inner_fold = cmp_balanced_folds(ids, inner_k, salt + 104729, 257, TRUE),
    stringsAsFactors = FALSE
  )
  cmp_check_nested_folds(out, outer_k, inner_k)
  out
}

cmp_check_nested_folds <- function(folds, outer_k, inner_k) {
  cmp_assert(diff(range(tabulate(folds$outer_fold, outer_k))) <= 1L,
             "Outer folds are unbalanced.")
  cmp_assert(diff(range(tabulate(folds$inner_fold, inner_k))) <= 1L,
             "Inner fold labels are unbalanced.")
  for (k in seq_len(outer_k)) {
    train <- folds$inner_fold[folds$outer_fold != k]
    cmp_assert(length(unique(train)) == inner_k,
               "Outer training sample %d lacks an inner fold label.", k)
  }
  invisible(TRUE)
}

cmp_project_unit <- function(x) pmin(1, pmax(0, x))

cmp_weighted_mse <- function(y, prediction, weights) {
  w <- weights / mean(weights)
  sum(w * (y - prediction)^2) / sum(w)
}

cmp_elapsed <- function(start) unname(proc.time()[["elapsed"]] - start)
