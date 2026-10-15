# Protocol configs and phase authorization.
#
# The protocol lives in two configs (docs/plan.md explains them). Gate results
# are small YAML files under results/comparison-v1/; a phase runs only when the
# gates before it have passed.

CMP_CONTRACT_PATHS <- list(
  v1 = "config/comparison-v1.yml",
  simulation = "config/comparison-simulations-v1.yml",
  candidate_estimands = "analysis/candidate/estimands.yml"
)

CMP_GATE_PATHS <- list(
  c1 = "results/comparison-v1/gate-c1/gate-c1-checks.yml",
  gate_d = "results/comparison-v1/gate-d/gate-d-decision.yml",
  c2 = "results/comparison-v1/gate-c2/gate-c2-checks.yml"
)

cmp_path <- function(root, relative) file.path(root, relative)

cmp_load_contracts <- function(root) {
  out <- list(
    root = root,
    v1 = yaml::read_yaml(cmp_path(root, CMP_CONTRACT_PATHS$v1)),
    simulation = yaml::read_yaml(cmp_path(root, CMP_CONTRACT_PATHS$simulation)),
    candidate = yaml::read_yaml(cmp_path(root, CMP_CONTRACT_PATHS$candidate_estimands))
  )
  cmp_assert(identical(out$v1$protocol_id, "factorial_ml_comparison_v1"),
             "Unexpected protocol identifier.")
  out
}

cmp_gate_passed <- function(root, gate) {
  path <- cmp_path(root, CMP_GATE_PATHS[[gate]])
  if (!file.exists(path)) return(FALSE)
  rec <- yaml::read_yaml(path)
  isTRUE(if (identical(gate, "gate_d")) rec$gate_d_passed else rec$all_passed)
}

# Calibration needs C1; smoke needs Gate D; production needs C2 as well.
cmp_phase_authorization <- function(root, phase) {
  switch(
    phase,
    test = TRUE,
    calibration = cmp_gate_passed(root, "c1"),
    smoke = cmp_gate_passed(root, "c1") && cmp_gate_passed(root, "gate_d"),
    production = cmp_gate_passed(root, "c1") && cmp_gate_passed(root, "gate_d") &&
      cmp_gate_passed(root, "c2"),
    stop("Unknown phase `", phase, "`.", call. = FALSE)
  )
}

cmp_require_phase <- function(root, phase) {
  cmp_assert(cmp_phase_authorization(root, phase),
             "Phase `%s` is not authorized: an earlier gate has not passed.", phase)
  invisible(TRUE)
}

cmp_session_manifest <- function(root, extra = list()) {
  head <- tryCatch(system2("git", c("-C", shQuote(root), "rev-parse", "HEAD"),
                           stdout = TRUE, stderr = FALSE), error = function(e) NA_character_)
  dirty <- tryCatch(length(system2("git", c("-C", shQuote(root), "status", "--porcelain"),
                                   stdout = TRUE, stderr = FALSE)) > 0L,
                    error = function(e) NA)
  pkgs <- c("glmnet", "ranger", "xgboost", "Matrix", "nloptr", "digest", "yaml",
            "nanoparquet", "orderridge")
  versions <- vapply(pkgs, function(p) {
    tryCatch(as.character(utils::packageVersion(p)), error = function(e) NA_character_)
  }, character(1))
  c(list(
    created_utc = cmp_utc_now(),
    git_head = head[[1]] %||% NA_character_,
    worktree_dirty = dirty,
    r_version = R.version.string,
    platform = R.version$platform,
    blas = tryCatch(extSoftVersion()[["BLAS"]], error = function(e) NA_character_),
    package_versions = as.list(versions)
  ), extra)
}
