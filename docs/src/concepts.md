# Concepts

## A cell is a node in a binary tree

The unit of state is a cell: one node of a lineage tree. A cell has five numbers:

| Field | Meaning |
|:---|:---|
| `id::Int64` | unique identifier, allocated in order of creation, so larger than its parent's |
| `birthtime::Float64` | absolute simulation time at which the cell was born |
| `mutations::Int64` | mutations acquired **at this cell's own birth** |
| `total_mutations::Int64` | mutations on the whole path from the root, including its own |
| `fitness::Float64` | the parent's fitness, updated once per new mutation |

`mutations` is **local**: a mutation event sitting on one node is carried by exactly the
leaves below it, which is what makes a site-frequency spectrum a single pass.
`total_mutations` is the cell's burden, stored so that burdens and distances are field
reads rather than walks to the root. `fitness` is **cumulative** for the same reason: it
is what the waiting-time distributions need, twice per division.

An `on_division` hook changes a daughter with [`set_fitness!`](@ref); nothing else about
a cell changes after its birth. A hand-built tree must keep `total_mutations`
consistent (a root's total is its own count, a child adds its own count to its
parent's).

### How the tree is stored

All nodes live in one [`LineageTree`](@ref), a **struct of arrays**: node `i` is row `i`
of the columns `parent`, `left`, `right` (row indices, `0` for none), `id`, `birthtime`,
`fitness`, `mutations` and `total`. Rows are appended in creation order, so a child
always comes after its parent. That has three consequences:

- **No per-cell objects.** A division appends two rows; the garbage collector has no
  tree of pointers to trace, whatever the population size. A node costs 44 bytes.
- **Post-order for free.** Sweeping the rows backwards visits every child before its
  parent, so spectra and leaf counts over the whole tree are one sequential pass.
- **Ids increase along lineages,** which is what [`find_mrca`](@ref) relies on.

You read the tree through [`CellNode`](@ref) handles — `node.fitness`, `node.parent`,
`node.left`, `node.data` (a [`NonMarkovCell`](@ref) value) — which implement the
AbstractTrees interface (`Leaves`, `PreOrderDFS`, `print_tree`). A handle stores the cell's
id, so it stays valid when the rows move (see below).

## The tree records the survivors

Cells that divide become internal nodes; living cells are the leaves. A cell that dies is
removed, together with every ancestor it leaves childless. What you hold is therefore the
**reduced tree of the survivors**: extinct side lineages leave nothing behind. Under a
supercritical process most cells ever born are in such lineages, and keeping them would
dominate memory while contributing to no observable of the living population.

Removal only marks a row. Once the marked rows reach half the tree (and at least 4096),
one pass **compacts** them away, keeping the order of the rest, so the cost per death
stays constant and memory follows the survivors.

!!! note "Internal nodes can have one child"
    When one daughter's lineage dies out, its parent's division still happened and stays
    in the tree with a single child. Unary nodes are legitimate, and statistics rely on
    them: [`leaf_depths`](@ref) counts every real division on a surviving lineage.
    [`sample_leaves`](@ref) keeps them for the same reason — see [Sampling](sampling.md).

## The population is the living cells

[`Population`](@ref) holds the lineage tree (`pop.tree`), the number of living cells, the
clock `pop.t` (the time of the last event) and the queue of pending events. The living
cells are the tree's leaves.

```julia
pop = initialize_population(fitness_init = 1.0)         # one founder
pop = initialize_population(100; fitness_init = 1.0)    # 100 independent founders

popsize(pop)        # number of living cells
alive_cells(pop)    # their CellNodes, in increasing id order
pop.t               # time of the most recently processed event
single_root(pop)    # the root of the lineage tree (nothing for a forest)
```

`initialize_population(N)` with `N > 1` seeds **independent founders**, so the
population is a forest of `N` trees; see
[Populations with more than one root](statistics.md#Populations-with-more-than-one-root).

## Competing risks

At a cell's birth **both** of its possible futures are drawn, and the earlier one is what
will happen:

```math
T_\text{div} \sim \texttt{birth\_dist}(f), \qquad T_\text{die} \sim \texttt{death\_dist}(f),
\qquad t_\text{event} = t_0 + \min(T_\text{div},\, T_\text{die}).
```

This is exact for *any* pair of distributions — no acceptance–rejection, no rate bound,
no time discretisation — because nothing can change a cell's fitness between its birth
and its event: mutations arise only at division, in the daughters. The flip side is that a
cell's fate never reacts to anything *external* after its birth (population size, a
treatment); see
[Density-dependent and homeostatic growth](blocks.md#Density-dependent-and-homeostatic-growth).

## The event queue

All pending events live in one global min-heap ordered by absolute time, with exactly one
event per living cell. An event is 16 bytes (time, row, division or death) and holds no
pointers. The heap is 4-ary, half as deep as a binary one, which matters once it no
longer fits in cache. The loop is:

1. Stop if `stopfunction(pop)` holds, the next event lies beyond `tmax`, or the heap is
   empty.
2. Take the earliest event, ``O(\log N)``, and advance `pop.t` to its time.
3. **Division**: append two daughters (drawing their mutations and fitness), call
   `on_division`, then draw both daughters' events. The first replaces the parent's
   event at the top of the heap; the second is pushed.
   **Death**: remove the event, and prune the cell from the tree.

Growing from one cell to ``N`` costs ``O(N \log N)``. Events at exactly the same time
(deterministic waiting times) fire in order of cell id.

### Exponential waiting times: the thinning path

Gillespie draws the time to the next event anywhere in the population from one
exponential with the summed rate. That factorisation holds only because the exponential
is memoryless. Without it, cell ages matter, and an exact algorithm must track each
cell's own residual time — which pre-drawing absolute event times and ordering them in a
heap does. This is the Next Reaction Method for a non-Markovian process.

When both waiting times *are* exponential, the package uses that shortcut instead
(`algorithm = :thinning`, chosen automatically): one exponential clock at rate ``R N``,
where ``R`` bounds every cell's ``b + d``; at each tick a uniformly chosen cell divides
with probability ``b/R``, dies with probability ``d/R``, or nothing happens. It is exact,
needs no heap, and costs ``O(1)`` per tick. ``R`` is the largest ``b + d`` seen so far, so
it rises as selection raises fitness; the fraction of empty ticks is ``1 -
\overline{b + d}/R``. See [the `algorithm` option](blocks.md#Choosing-the-algorithm).

## Why not the exponential?

The exponential's defining property is that a cell's chance of dividing in the next
instant does not depend on how long it has been alive. Mammalian cells do not behave that
way: a cell that has just divided cannot divide again for the length of a cell cycle —
8–12 h for fast tumour cells, days for normal tissue. The exponential puts substantial
probability mass in that refractory window and fits measured proliferation data poorly
(Zilman et al. 2010).

The natural replacement is the Gamma, which adds one dimensionless parameter, the shape
``k``, with ``\mathrm{CV} = 1/\sqrt{k}``: ``k = 1`` is the exponential and
``k \to \infty`` a deterministic clock. Reported coefficients of variation of the
interdivision time:

| Cell type | Typical CV | ``k = 1/\mathrm{CV}^2`` |
|:---|:---|:---|
| Well-regulated somatic (fibroblasts, epithelial) | 0.10–0.20 | 25–100 |
| Fast cancer cells (leukaemia, ovarian carcinoma) | 0.30–0.45 | 5–11 |
| Stimulated lymphocytes | 0.50–0.70 | 2–4 |

``k = 5`` (CV ≈ 0.45) is a reasonable default for rapidly cycling tumour cells.

The best-fitting empirical family is the **exponentially modified Gamma** — a Gamma
(S/G2/M) plus an exponential (the stochastic G1 wait) — across 77 datasets and 16 cell
types (Golubev 2016). `birth_dist` may return any univariate distribution, and the
simulator only ever calls `rand(rng, d)` on it, so a sampler is a few lines:

```@example emg
using NonMarkovEvolution, Distributions, Random, Statistics

struct ExpModGamma <: ContinuousUnivariateDistribution
    k::Float64; θ::Float64; λ::Float64       # Gamma shape and scale, exponential mean
end
Base.rand(rng::AbstractRNG, d::ExpModGamma) =
    rand(rng, Gamma(d.k, d.θ)) + rand(rng, Exponential(d.λ))

block = NonMarkovBlock(
    birth_dist = f -> ExpModGamma(4.0, 0.15 / f, 0.4 / f),   # mean (0.6 + 0.4)/f = 1/f
    death_dist = f -> Dirac(Inf), effect_dist = Dirac(0.0),
    fitness_update = (f, δ) -> f, ν = 0.0, stopfunction = p -> popsize(p) >= 500)
pop = simulate!(initialize_population(), block, MersenneTwister(1))
round(mean(cell_lifetimes(single_root(pop))), digits = 2)
```

(The completed lifetimes average below the mean of 1: see
[Depths, fitness and lifetimes](statistics.md#Depths,-fitness-and-lifetimes).)

### What non-exponential timing changes

Cell-cycle variability changes the growth rate itself. With division density ``g`` and
survival ``S_d``, the Malthusian rate ``r`` solves the Euler–Lotka equation

```math
2\int_0^\infty e^{-rt}\, g(t)\, S_d(t)\, \mathrm{d}t = 1 .
```

For exponentials with rates ``b`` and ``d`` this gives ``r = b - d``. For pure birth with
``T_\text{div} \sim \mathrm{Gamma}(k, 1/(kb))`` — mean ``1/b`` for every ``k`` — it gives

```math
r = k\,b\left(2^{1/k} - 1\right),
```

which falls from ``r = b`` at ``k = 1`` to ``b \ln 2 \approx 0.69\,b`` as ``k \to \infty``:
at a fixed mean cycle length, a **more variable** cycle grows **faster**, because
early-dividing lineages compound. An inference that assumes ``r = b - d`` on data
generated at ``k = 5`` is biased by about 25%. The test suite checks this rate against
simulation.

It is also why [`leaf_depths`](@ref) (divisions) and coalescence times (real time) are
kept as separate observables: under Gamma timing they carry different information.

!!! note "\"Mean equals variance\" is not a simplification here"
    Imposing ``\mathrm{Var}[T] = \mathbb{E}[T]`` on a Gamma forces ``\theta = 1``, and with
    ``\mathbb{E}[T] = 1/b`` gives ``k = 1/b`` — the exponential at ``b = 1``. The
    constraint collapses the model back to Gillespie.

## Reproducibility

`simulate!` requires an rng and consumes it in a fixed order, so a run is reproducible
from `(initial population, block, seed)`, and a chained run is identical draw for draw to
the equivalent uninterrupted one (asserted by the tests).

Per division the order is: daughter 1's mutation count and effect sizes, daughter 2's,
then `on_division` if it draws, then daughter 1's two waiting times and daughter 2's.
Anything that changes how many values a step consumes shifts everything after it:

- `Poisson(0.0)` does consume a draw, so a neutral run (`ν = 0`) still burns one value
  per daughter.
- `Dirac(x)` consumes nothing, so swapping a `Dirac` for an `Exponential` shifts the
  stream on top of changing the model.
- Both waiting times are always drawn, even when one is `Dirac(Inf)`.
- A fresh schedule (a new population, [`reset_schedule!`](@ref), an extinction restart)
  visits cells in id order; a cell that is already old may need several pairs of draws,
  because its next event is conditioned on nothing having happened to it yet.

The package holds no global mutable state: one population, block and closure per task
is safe under `Threads.@threads`, given one rng per task.

### Stability guarantees

| What | Stable across | Pinned by |
|:---|:---|:---|
| Leaf sampling: `sample_leaves` draws and `sample_trees` seed derivation | Julia and package versions (StableRNGs.jl, explicit splitmix64) | golden literals in `test/sampling.jl` |
| `simulate!` random-stream consumption | edits to this package | golden values in `test/regression.jl` |
| `leaf_depths(root)` default order | edits to this package | `test/statistics.jl` |

A `simulate!` seed does **not** reproduce across Julia or Distributions.jl versions:
`MersenneTwister` streams and several samplers have changed between releases. Passing a
`StableRNG` removes the first source of drift but not the second.

## References

- **Golubev A (2016)** — the exponentially modified Gamma fits mammalian cell-cycle data
  better than Gamma or lognormal alone; 77 datasets, 16 cell types.
  *J. Theor. Biol.* 393, 203–217.
  DOI: [10.1016/j.jtbi.2015.12.027](https://doi.org/10.1016/j.jtbi.2015.12.027)
- **Zilman A, Ganusov VV, Perelson AS (2010)** — Gamma shapes ``k = 2``–3 are needed to fit
  CD4⁺ T-cell proliferation; ``k = 1`` fits poorly in every condition tested.
  *PLoS ONE* 5(9): e12775.
  DOI: [10.1371/journal.pone.0012775](https://doi.org/10.1371/journal.pone.0012775)
- **Hahn GM (1966)** — the CV of interdivision time governs desynchronisation kinetics.
  *Biophys. J.* 6(2), 197–207.
  DOI: [10.1016/S0006-3495(66)86656-0](https://doi.org/10.1016/S0006-3495(66)86656-0)
- **Sandler O et al. (2015)** — cousin–cousin correlations in cell-cycle duration dominate
  over mother–daughter ones; i.i.d. Gamma underestimates lineage-level clustering.
  *Nature* 519, 422–425.
  DOI: [10.1038/nature14318](https://doi.org/10.1038/nature14318)
- **Yanagisawa M et al. (1985)** — per-phase CV measurement in CHO cells; G1 is the most
  variable phase. *Cytometry* 6(6), 550–558.
  DOI: [10.1002/cyto.990060609](https://doi.org/10.1002/cyto.990060609)
- **Chiorino G et al. (2001)** — Gamma-based desynchronisation for cancer cell lines;
  CV ≈ 0.2–0.4. *J. Theor. Biol.* 208(2), 185–199.
  DOI: [10.1006/jtbi.2000.2213](https://doi.org/10.1006/jtbi.2000.2213)
- **Fennell DA et al. (2005)** — apoptosis kinetics modelled with an exponential
  time-to-MOMP; the exponential is more defensible for death than for division.
  *Apoptosis* 10(3), 517–530.
  DOI: [10.1007/s10495-005-0818-2](https://doi.org/10.1007/s10495-005-0818-2)
