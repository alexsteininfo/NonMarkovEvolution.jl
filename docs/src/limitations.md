# Limitations and open questions

## Deliberately out of scope

- **Untracked neutral mutations** need no state: their burden is `Poisson(m * depth)`
  against [`leaf_depths`](@ref) and their expected spectrum ``m`` times the
  [`branch_spectrum`](@ref). Deriving them afterwards is exact in distribution and lets
  one tree serve every ``m``; see [Neutral evolution](selection.md#Neutral-evolution).
- **Extinct lineages** are pruned, because under a supercritical process they dominate
  memory and contribute to no observable of the survivors. Questions about lineages that
  failed (how many clones were lost to drift) need an `on_division` hook that records
  them as they happen.
- **Spatial structure.** Cells have no position; spatial growth belongs in a different
  simulator.
- **Clone-level bookkeeping.** At ``ν \gtrsim 0.1`` every cell has a distinct fitness
  history; for ``ν \ll 1`` see
  [`BirthDeathMutation`](https://github.com/alexsteininfo/BirthDeathMutation).
- **Copy number, sequencing noise, variant calling** live downstream, for example in
  [`CopyNumberPainter.jl`](https://github.com/alexsteininfo/CopyNumberPainter.jl).
- **Inference of any kind** — keeping it apart lets estimators run on real data without
  a simulator in their dependency chain.

## Known constraints of the model

### Rates are frozen at each cell's birth

A cell's waiting times are drawn once, at its birth, and never revised. That is exact for
the model as specified (fitness cannot change mid-cycle), but anything genuinely
time-varying — density dependence, a treatment starting mid-run, interaction between
cells — is approximated as of each cell's birth; see
[Density-dependent and homeostatic growth](blocks.md#Density-dependent-and-homeostatic-growth).
The exact alternative is to change the regime by chaining blocks.

Invalidating pending events when the environment changes would make density dependence
exact. The conditional redraw it needs already exists (fresh schedules use it); what is
missing is a heap with deletion, and the rejection cost grows for cells far into their
cycle. For exponential waiting times it would be simpler: the `:thinning` loop could
evaluate a cell's rates at each tick instead of at its birth, given a rate bound that
holds for every density.

### Cell-cycle durations are independent between relatives

Real lineages show cousin–cousin correlations in cycle length (Sandler et al. 2015), so
an i.i.d. model underestimates lineage-level clustering. Inheriting cycle length would
need a per-cell latent variable passed to `birth_dist` — a small change to the cell type
and a large change to what the package claims, not made so far.

### The Gamma is a convenience, not a claim

The best-fitting empirical family is the exponentially modified Gamma (Golubev 2016). The
plain Gamma is the default because it captures the non-Markovian character with one
parameter; supply a better distribution if your conclusions depend on the shape (see the
example in [Why not the exponential?](concepts.md#Why-not-the-exponential?)).

## Conventions

These are documented choices, not settled answers:

- **Coalescence across independent founders** is taken back to the earlier founder's
  birth, which puts an artefactual spike into pooled histograms from a forest. `Inf`,
  `missing` or an error would be defensible too; sample each founder's tree separately.
- **`leaf_depths(root)` keeps its historical order** by default because stored results
  depend on it; `order = :leaves` gives the co-indexed one.
- **The realised cell-cycle distribution is biased** by competing risks and by stopping
  at a fixed size. [`cell_lifetimes`](@ref) is what an experiment would observe, not an
  estimate of `birth_dist`; a de-biased or survival-analysis estimator is not offered.
- **Fitness is unbounded** unless `fitness_update` caps it. Nothing objects when an
  additive or multiplicative rule drives division times toward zero; the
  [capped mode](selection.md#Capped-selection-(diminishing-returns)) is the opt-in answer.
- **Restarts record the tree's ancestry** once per call with
  `restart_on_extinction = true`; that costs one pass over the ancestors of the living
  cells, which is noticeable only on very large chained populations.
