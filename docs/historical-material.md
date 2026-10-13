# Historical-material boundary

The files `tutorial.Rmd`, `tutorial.md`, `tutorial.bib`, `plotting.R`, and the
root `.rda` file record the 2019 prototype. The ignored `shrinkage/` directory
contains additional manuscripts, code, figures, and result archives from that
period.

These materials are retained for provenance only. They are **non-evidentiary**
for the new project because the old implementation penalized reference-coded
and standardized coefficients, did not isolate the value of monotonicity from
unrestricted multi-penalty ridge, and used simulation workflows that do not
meet the new resampling and evaluation protocol. In particular, old results
must not be copied into the paper, package documentation, figures, or gate
assessment.

The new evidence begins with:

- source under `R/`;
- correctness tests under `tests/testthat/`;
- machine-readable configurations under `config/`;
- simulation code under `simulation/`; and
- newly generated tidy files under ignored `results/` directories.

The `Candidate recruitment/` archive is also ignored. Application code may
read the public cleaned replication CSV in place, but no raw YouGov export,
review file, presentation, or unrelated substantive-project artifact belongs
in this repository or a release tarball.

