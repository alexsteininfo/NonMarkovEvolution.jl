# NonMarkovEvolution.jl

Stochastic, **non-Markovian** birth–death evolution of cell populations with
mutations that may change fitness: realistic cell-cycle timing, per-cell fitness, and the
complete lineage tree of the survivors. Built for somatic evolution (copy-number or SNV
mutations), but not restricted to it.

Division and death waiting times are drawn from arbitrary distributions that may depend
on the cell's own fitness, so the cell cycle can have a realistic refractory period that
an exponential model cannot express. Each mutation draws its own fitness effect (possibly
zero), so every cell carries an individual fitness rather than a shared subclone label. What
comes back is the lineage tree itself; spectra, burdens, depths and coalescence times are
computed from it, on the whole population or on a uniform sample.

This package is a **simulator**. It does not infer trees, fit parameters, or compare
populations.

## Installation

Requires Julia 1.10 or newer.

```julia
using Pkg
Pkg.add(url = "https://github.com/alexsteininfo/NonMarkovEvolution.jl")
```

## Quickstart

```@example quickstart
using NonMarkovEvolution, Distributions, Random, Statistics

pop = initialize_population(fitness_init = 1.0)      # one founding cell

block = NonMarkovBlock(
    birth_dist     = f -> Gamma(5.0, 1 / (5 * f)),   # mean division time 1/f, CV 1/√5
    death_dist     = f -> Exponential(1 / 0.3),       # death rate 0.3, fitness-free
    effect_dist    = Exponential(0.05),               # effect size δ of one mutation
    fitness_update = (f, δ) -> f + δ,                 # additive selection
    ν              = 0.2,                             # mean mutations per daughter
    stopfunction   = pop -> popsize(pop) >= 2_000,
    restart_on_extinction = true,                     # retry if the founder line dies
)

acc = MeasurementAccumulator(MeasurementSpec(
    trajectory_dt     = 1.0,
    snapshot_triggers = [AtPopSize(500), AtEnd()],
    snapshot_stats    = [SFS(), FitnessDistribution(), MutationsPerCell()],
))

simulate!(pop, block, MersenneTwister(42); accumulator = acc)
m = finalize_measurements(acc)

(popsize(pop), round(pop.t, digits = 2), round(mean(fitness_per_cell(pop)), digits = 3))
```

The recorded trajectory and snapshots:

```@example quickstart
[(p.t, p.N_total) for p in m.trajectory][end-2:end]
```

```@example quickstart
[(s.trigger, length(s[:fitness])) for s in m.snapshots]
```

The tree is the real output:

```@example quickstart
root = single_root(pop)
(mean(mutations_per_cell(root)), mean(leaf_depths(root)), sum(branch_spectrum(root)))
```

Sequence 200 of the 2 000 cells, the way an experiment would:

```@example quickstart
out = sample_trees(pop, SamplingSpec(200); seed = 0xBEEF)
site_frequency_spectrum(out.samples[1].root)[1:5]
```

## The shape of a run

| Object | Question | Page |
|:---|:---|:---|
| [`Population`](@ref) | what exists now, and at what time | [Concepts](concepts.md) |
| [`NonMarkovBlock`](@ref) | what the cells do, and until when | [The simulation block](blocks.md) |
| [`MeasurementSpec`](@ref) | what to write down while it happens | [Output](output.md) |

[`simulate!`](@ref) advances the first under the rules of the second, recording through
the third. It mutates the population in place, so successive blocks chain onto one
lineage tree — see [Chaining blocks](blocks.md#Chaining-blocks).

## Where to go next

- [Concepts](concepts.md) — the cell, the tree, competing risks, the event queue, why
  not the exponential, and reproducibility.
- [The simulation block](blocks.md) — waiting-time modes, stop conditions, extinction,
  hooks, chaining.
- [Mutations and selection](selection.md) — the mutation channel and the selection modes.
- [Output](output.md) — the population, the tree, trajectories and snapshots.
- [Tree statistics](statistics.md) — spectra, burdens, depths, distances, coalescence.
- [Sampling](sampling.md) — drawing `n` of `N` cells and the induced tree.
- [Limitations and open questions](limitations.md).

## Related packages

- [`BirthDeathMutation`](https://github.com/alexsteininfo/BirthDeathMutation) — the low
  mutation-rate regime, where clones are the natural unit.
- [`CopyNumberPainter.jl`](https://github.com/alexsteininfo/CopyNumberPainter.jl) —
  copy-number alterations along a tree from this package.
- [`gITH-nonMarkovian`](https://github.com/alexsteininfo/gITH-nonMarkovian) — the
  analyses, figures and theory built on this simulator.
