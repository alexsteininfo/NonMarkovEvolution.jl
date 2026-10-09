# NonMarkovEvolution.jl

Stochastic, **non-Markovian** birth–death evolution of cell populations with
mutations that may change fitness: realistic cell-cycle timing, per-cell fitness, and the
complete lineage tree of the survivors. Built for somatic evolution (copy-number or SNV
mutations), but not restricted to it.

Division and death waiting times are drawn from arbitrary distributions that may depend
on the cell's own fitness, so the cell cycle can have a realistic refractory period that
a memoryless model cannot express. Each mutation draws its own fitness effect (possibly
zero), so every cell carries an individual fitness rather than a shared subclone label.

This is a **simulator**: it does not infer trees, fit parameters, or compare populations.

## Install

Requires Julia 1.10 or newer.

```julia
using Pkg
Pkg.add(url = "https://github.com/alexsteininfo/NonMarkovEvolution.jl")
```

## Quickstart

```julia
using NonMarkovEvolution, Distributions, Random

pop = initialize_population(fitness_init = 1.0)

block = NonMarkovBlock(
    birth_dist     = f -> Gamma(5.0, 1 / (5 * f)),   # mean division time 1/f, CV 1/√5
    death_dist     = f -> Exponential(1 / 0.3),       # death rate 0.3
    effect_dist    = Exponential(0.05),               # effect size of one mutation
    fitness_update = (f, δ) -> f + δ,                 # additive selection
    ν              = 0.2,                             # mean mutations per daughter
    stopfunction   = pop -> popsize(pop) >= 10_000,
    restart_on_extinction = true,
)

acc = MeasurementAccumulator(MeasurementSpec(
    trajectory_dt = 0.5, snapshot_stats = [SFS(), FitnessDistribution()]))
simulate!(pop, block, MersenneTwister(42); accumulator = acc)
m = finalize_measurements(acc)                  # trajectory + end snapshot

root = single_root(pop)
mutations_per_cell(root)        # mutation burden of every living cell
leaf_depths(root)             # divisions from founder to each living cell
branch_spectrum(root)         # topology, separated from the mutation rate

# Sequence 1 000 of the 10 000 cells, the way an experiment would.
out = sample_trees(pop, SamplingSpec(1_000); seed = 0xBEEF)
site_frequency_spectrum(out.samples[1].root)
```

Runnable examples are in [`examples/`](examples): single-cell expansion, growth with
mutations, chained two-phase runs, and an arbitrary initial condition.

## What it models

- **Competing risks, exactly.** At birth a cell draws a division and a death time and the
  earlier happens — exact for any pair of distributions, with no rate bound or time
  step, because a cell's fitness cannot change before its own event.
- **One global event heap** (the Next Reaction Method for a non-Markovian process), so
  growing to `N` cells costs `O(N log N)`. For exponential waiting times an exact
  thinning loop takes over automatically, at `O(1)` per event.
- **Built for large runs.** The tree is a struct of arrays (44 bytes per node, nothing
  for the garbage collector to trace) and saves to a compact, versioned binary file;
  growing to 10⁶ cells takes about a second.
- **Any waiting-time law** — deterministic, exponential, Gamma, Weibull, log-normal, or
  your own — and it matters: at a fixed mean cycle the growth rate falls from `b` at
  Gamma shape `k = 1` to `b·ln 2` as `k → ∞`.
- **Selection in two lines** — pick `effect_dist` and `fitness_update`: neutral,
  additive, multiplicative, winner-takes-all, capped, or deleterious load.
- **The tree is the output.** Dead cells are pruned, leaving the reduced tree of the
  survivors, from which spectra, burdens, depths, distances and coalescence times are
  computed — on the whole population or on a reproducible uniform sample.

Untracked neutral mutations, spatial structure and clone-level bookkeeping are out of
scope.
Related: [`CopyNumberPainter.jl`](https://github.com/alexsteininfo/CopyNumberPainter.jl)
(copy number along these trees),
[`BirthDeathMutation`](https://github.com/alexsteininfo/BirthDeathMutation) (the low-`ν`
regime) and [`gITH-nonMarkovian`](https://github.com/alexsteininfo/gITH-nonMarkovian)
(analyses built on this simulator).

## Documentation

The manual covers the algorithm, every block field, selection modes, recording, tree
statistics, sampling and its stability guarantees, and the model's limitations. Build it
locally:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path = "."); Pkg.instantiate()'
julia --project=docs docs/make.jl       # then open docs/build/index.html
```

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## License

See [LICENSE](LICENSE).
