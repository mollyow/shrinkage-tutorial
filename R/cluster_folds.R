#' Assign complete clusters to cross-validation folds
#'
#' Each distinct cluster is assigned to exactly one fold.  Assignment is
#' randomized within coarse cluster-size strata and then greedily balanced by
#' the number of observations.  Supplying the same seed and inputs is
#' deterministic and does not alter the caller's random-number stream.
#'
#' @param cluster Vector identifying independent clusters.
#' @param nfolds Number of folds.
#' @param seed Integer seed.
#' @return Integer vector, one fold label per observation.
#' @export
cluster_folds <- function(cluster, nfolds = 5L, seed = 1L) {
  if (length(cluster) < 2L || anyNA(cluster)) {
    stop("`cluster` must contain at least two non-missing observations.", call. = FALSE)
  }
  ids <- unique(as.character(cluster))
  if (length(ids) < 2L) {
    stop("At least two independent clusters are required.", call. = FALSE)
  }
  nfolds <- as.integer(nfolds)
  if (!is.finite(nfolds) || nfolds < 2L) {
    stop("`nfolds` must be at least two.", call. = FALSE)
  }
  nfolds <- min(nfolds, length(ids))
  sizes <- tabulate(match(as.character(cluster), ids), nbins = length(ids))

  old_seed_exists <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (old_seed_exists) old_seed <- get(".Random.seed", envir = .GlobalEnv)
  on.exit({
    if (old_seed_exists) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)
  set.seed(as.integer(seed))

  # Random tie breaking followed by largest-first greedy packing.
  ord <- order(-sizes, stats::runif(length(ids)))
  loads <- integer(nfolds)
  id_fold <- integer(length(ids))
  for (j in ord) {
    candidates <- which(loads == min(loads))
    f <- sample(candidates, 1L)
    id_fold[j] <- f
    loads[f] <- loads[f] + sizes[j]
  }
  unname(id_fold[match(as.character(cluster), ids)])
}

