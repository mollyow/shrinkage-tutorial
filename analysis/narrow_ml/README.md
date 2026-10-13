# Narrow orthogonal-ML feasibility screen

This directory contains an exploratory, non-confirmatory screen for the
narrower paper described in `docs/orthogonal-ml-feasibility.md`.

The exploration is now complete. The executed decision and exact results are
documented in [`docs/narrow-ml-gate-decision.md`](../../docs/narrow-ml-gate-decision.md).
The oracle-headroom gate failed, so the complete forest benchmark was not run.

The screen asks one question: **does cross-fitted outcome learning materially
reduce the design-based uncertainty of the 22 interaction coordinates that
were locked for the candidate-recruitment application?** It does not search for
new interactions and it does not treat predictive accuracy as causal evidence.

`run_candidate_precision_screen.R` compares:

- the unadjusted randomization score;
- a task-position-only weighted linear outcome model, which isolates the gain
  from ordinary outcome centering;
- an additive weighted linear outcome model;
- a weighted linear model with all two-way profile interactions and
  profile-by-demographic interactions; and
- an elastic net over that same expanded feature set, with respondent-safe
  inner folds; and
- a regression forest fit with `ranger`.

Every nuisance learner is trained with complete respondents held out. The
plug-in part averages predictions over the locked uniform 1,440-profile target,
the eight task positions, and each held-out respondent's pre-treatment
demographic stratum. Cluster scores are then aggregated with the published
survey weights.

Run from the repository root:

```sh
Rscript analysis/narrow_ml/run_candidate_precision_screen.R
```

Generated results are written under `results/narrow-ml/`, which is ignored.
The script reads only the already permitted cleaned replication file and the
locked prepared bundle. This is a feasibility diagnostic, not part of the
confirmatory candidate analysis.

`run_simulation_screen.R` adds a small three-scenario check of coefficient
precision and 95% coverage under simple, linear-heterogeneous, and nonlinear-
heterogeneous response surfaces. Its defaults are intentionally too small for
paper evidence:

```sh
Rscript analysis/narrow_ml/run_simulation_screen.R \
  --reps=30 --workers=3 --trees=100
```

## Candidate-geometry gates

The versioned production sequence is:

```sh
Rscript analysis/narrow_ml/run_candidate_geometry_benchmark.R \
  --config=analysis/narrow_ml/benchmark-headroom.yml \
  --output-dir=results/narrow-ml/benchmark-headroom \
  --workers=6

Rscript analysis/narrow_ml/evaluate_headroom_gate.R \
  --config=analysis/narrow_ml/benchmark-headroom.yml \
  --results-dir=results/narrow-ml/benchmark-headroom
```

The run completed 600 of 600 shards. It failed for scientific futility:
oracle/additive family-balanced MSE was 0.9674 (95% interval 0.9572--0.9781),
where the downstream ML target was 0.85. Under the stage rule, do not run
`benchmark-production.yml` for this DGP.
