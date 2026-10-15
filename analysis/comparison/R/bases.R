# Target-orthonormal factor bases and family coordinates.
#
# Convention (shared with the application design audits): each factor gets
# Helmert contrasts rescaled to be centered and orthonormal under its uniform
# target marginal. A two-factor family's coordinates are the tensor products
# psi_{ja,jb}(a, b) = H_A[a, ja] * H_B[b, jb], ordered with ja varying fastest,
# and the coordinate value is theta = E_mu[psi(A, B) m]. The family loss
# ||theta_hat - theta||^2 is invariant to rotations within the family space.

cmp_helmert_basis <- function(n_levels) {
  cmp_assert(n_levels >= 2L, "A factor needs at least two levels.")
  h <- stats::contr.helmert(n_levels)
  h <- sweep(h, 2L, sqrt(colSums(h^2) / n_levels), "/")
  dimnames(h) <- NULL
  h
}

# A geometry's factors are a named list of level counts. Level codes are the
# integers 1..L; labels are carried separately for reporting.
cmp_factor_bases <- function(factor_levels) {
  lapply(factor_levels, cmp_helmert_basis)
}

cmp_family_basis <- function(bases, a, b) {
  ha <- bases[[a]]
  hb <- bases[[b]]
  grid <- expand.grid(a_code = seq_len(nrow(ha)), b_code = seq_len(nrow(hb)),
                      KEEP.OUT.ATTRS = FALSE)
  psi <- matrix(NA_real_, nrow(grid), ncol(ha) * ncol(hb))
  labels <- character(ncol(psi))
  cc <- 0L
  for (jb in seq_len(ncol(hb))) {
    for (ja in seq_len(ncol(ha))) {
      cc <- cc + 1L
      psi[, cc] <- ha[grid$a_code, ja] * hb[grid$b_code, jb]
      labels[[cc]] <- sprintf("h%02d_x_h%02d", ja, jb)
    }
  }
  colnames(psi) <- labels
  list(grid = grid, psi = psi, n_a = nrow(ha), n_b = nrow(hb))
}

# families: list of list(id=, attributes=c(a, b)).
cmp_build_families <- function(bases, families) {
  out <- lapply(families, function(f) {
    ab <- as.character(unlist(f$attributes))
    fb <- cmp_family_basis(bases, ab[[1]], ab[[2]])
    p <- rep(1 / nrow(fb$grid), nrow(fb$grid))
    gram <- crossprod(fb$psi, fb$psi * p)
    cmp_assert(max(abs(gram - diag(ncol(fb$psi)))) < 1e-10,
               "Family %s basis is not orthonormal.", f$id)
    main <- cbind(1, bases[[ab[[1]]]][fb$grid$a_code, , drop = FALSE],
                  bases[[ab[[2]]]][fb$grid$b_code, , drop = FALSE])
    cmp_assert(max(abs(crossprod(main, fb$psi * p))) < 1e-10,
               "Family %s basis is not orthogonal to its main effects.", f$id)
    c(list(id = f$id, a = ab[[1]], b = ab[[2]], dimension = ncol(fb$psi),
           cell_probability = p), fb)
  })
  names(out) <- vapply(families, `[[`, "", "id")
  out
}

cmp_coordinate_table <- function(families) {
  do.call(rbind, lapply(families, function(f) {
    data.frame(family = f$id, coordinate = colnames(f$psi),
               key = paste(f$id, colnames(f$psi), sep = "::"),
               stringsAsFactors = FALSE)
  }))
}

# Cell index for each row of a data frame under one family.
cmp_family_cell <- function(family, a_codes, b_codes) {
  as.integer(a_codes) + (as.integer(b_codes) - 1L) * family$n_a
}

# Row-level psi matrix (rows x coordinates) for observed focal codes.
cmp_family_psi_rows <- function(family, a_codes, b_codes) {
  family$psi[cmp_family_cell(family, a_codes, b_codes), , drop = FALSE]
}

# Theta for one family from cell means m_bar (cells x 1, or cells x R).
cmp_family_theta <- function(family, cell_means) {
  crossprod(family$psi * family$cell_probability, as.matrix(cell_means))
}

# Projector onto a family's interaction space in the cell coordinate system.
cmp_family_projector <- function(family) {
  family$psi %*% t(family$psi) %*% diag(family$cell_probability)
}
