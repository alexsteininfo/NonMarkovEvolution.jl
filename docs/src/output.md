# Output

A finished run leaves three things: the [`Population`](@ref) (what is alive now), the
lineage tree reachable from it (everything that led there), and — if you asked for one —
a [`Measurements`](@ref) record of what happened along the way.

## The population

```julia
popsize(pop)          # number of living cells
alive_cells(pop)      # their CellNodes, in increasing id order
fitness_per_cell(pop) # one entry per living cell, same order
mutations_per_cell(pop) # same order
pop.t                 # time of the most recently processed event
pop                   # Population: 10000 cells (t = 14.212)
```

Every per-cell population function returns cells in increasing id order, so their
results are co-indexed with each other and reproducible. `pop.tree` is the
[`LineageTree`](@ref) itself; treat it as read-only except through
[`set_fitness!`](@ref).

## The tree

The tree is read through [`CellNode`](@ref) handles:

```julia
root = single_root(pop)       # nothing if the population is a forest
roots(pop)                    # every root of a forest

node.id, node.birthtime, node.fitness, node.mutations, node.total_mutations
node.data                     # the same, as a NonMarkovCell value
isalive(node)                 # a living cell (a leaf)?
node.parent                   # nothing at the root
node.left, node.right         # nothing at a leaf; one may be nothing at a unary node

division_time(node)           # when the cell divided; nothing if it is alive
cell_lifetime(node, pop.t)    # birth to division, or birth to now
alive_cells(root)             # the living leaves, in Leaves order
popsize(root)                 # how many
last_division_time(root)      # the last division in the tree — not pop.t
```

`AbstractTrees` works on a `CellNode` (`Leaves`, `PreOrderDFS`, `print_tree`), and so
does every root method on the [Tree statistics](statistics.md) page — those work on the
tree's arrays directly and are much faster than a generic traversal.
`last_division_time(root)` is earlier than `pop.t` when the run ended on a death.

A handle stays valid while the tree changes, including when removed rows are compacted
away; a handle on a cell that has since died and been removed throws when read.

## Recording while it runs

A [`MeasurementSpec`](@ref) declares what to collect, a [`MeasurementAccumulator`](@ref)
collects it during `simulate!`, and [`finalize_measurements`](@ref) packages it:

```@example rec
using NonMarkovEvolution, Distributions, Random
spec = MeasurementSpec(
    trajectory_dt     = 0.5,
    snapshot_triggers = [AtTime(3.0), AtPopSize(500), AtEnd()],
    snapshot_stats    = [SFS(), FitnessDistribution(), MutationsPerCell()],
)
acc   = MeasurementAccumulator(spec)
block = NonMarkovBlock(
    birth_dist = f -> Gamma(5.0, 1 / (5 * f)), death_dist = f -> Exponential(5.0),
    effect_dist = Exponential(0.05), fitness_update = (f, δ) -> f + δ, ν = 0.5,
    stopfunction = pop -> popsize(pop) >= 1_000, restart_on_extinction = true)
pop = simulate!(initialize_population(), block, MersenneTwister(4); accumulator = acc)
m   = finalize_measurements(acc)
[(s.trigger, s.t, length(s[:fitness])) for s in m.snapshots]
```

Defaults are `trajectory_dt = Inf` (no trajectory), `snapshot_triggers = [AtEnd()]` and
`snapshot_stats = [FitnessDistribution()]`. `finalize_measurements` copies, so the
accumulator stays usable, and one accumulator can be carried across chained blocks.

### The trajectory

A [`TrajectoryPoint`](@ref) is recorded every `trajectory_dt` time units:

| Field | Meaning |
|:---|:---|
| `t` | the grid time; the state is the exact state at that time |
| `N_total` | population size |
| `mean_fitness`, `var_fitness` | over living cells |
| `mean_mutations`, `var_mutations` | mutation burden over living cells |

```@example rec
[(p.t, p.N_total, round(p.mean_fitness, digits = 3)) for p in m.trajectory[end-2:end]]
```

Before each event is applied, every grid time strictly earlier than that event is
recorded with the current state — which is the state that held there. The state is
right-continuous: events at exactly a grid time count at that time. Nothing is recorded
once the population is empty. Each point costs ``O(N)``, so a fine grid late in a large
run is expensive; record a fine early phase and a coarse late phase in two chained
blocks if you need both.

### Snapshot triggers

| Trigger | Fires |
|:---|:---|
| [`AtEnd`](@ref)`()` | at the end of every `simulate!` call |
| [`AtTime`](@ref)`(t)` | once, with the exact state at time `t`, labelled `t` |
| [`AtPopSize`](@ref)`(N)` | once, at the first moment the size is at least `N` |

`AtTime` and `AtPopSize` fire at most once per accumulator, and `AtPopSize` is checked
before the first event too. `AtEnd` fires whenever a block exits, on the stop condition
or on extinction, so a carried accumulator gets one end snapshot per block. An `AtTime`
earlier than the start of the call it is first seen in never fires (that state is gone).
An extinction restart discards only what the current call recorded.

### Snapshot statistics

A [`SnapshotData`](@ref) holds one value per requested statistic, read by name:

| Statistic | Key | Value |
|:---|:---|:---|
| [`SFS`](@ref)`()` | `:sfs` | [`site_frequency_spectrum`](@ref)`(pop)` |
| [`FitnessDistribution`](@ref)`()` | `:fitness` | [`fitness_per_cell`](@ref)`(pop)` |
| [`MutationsPerCell`](@ref)`()` | `:mutations` | [`mutations_per_cell`](@ref)`(pop)`, co-indexed with `:fitness` |

```@example rec
endsnap = only(s for s in m.snapshots if s.trigger isa AtEnd)
(collect(keys(endsnap)), endsnap[:sfs][1:5])
```

Any other statistic is a type and a [`measure`](@ref) method away; its key is the type
name unless you also define [`statistic_name`](@ref):

```@example rec
using Statistics
struct MeanBurden <: AbstractStatistic end
NonMarkovEvolution.measure(::MeanBurden, pop) = mean(mutations_per_cell(pop))

acc2 = MeasurementAccumulator(MeasurementSpec(snapshot_triggers = [AtTime(2.0), AtEnd()],
                                              snapshot_stats = [MeanBurden()]))
simulate!(initialize_population(), block, MersenneTwister(4); accumulator = acc2)
[(s.t, round(s[:MeanBurden], digits = 2)) for s in finalize_measurements(acc2).snapshots]
```

## After the fact

Everything else is computed from the tree afterwards, which loses nothing because the
tree is a complete record of the survivors' ancestry — see [Tree statistics](statistics.md).
The one thing the tree cannot give is a quantity at an **intermediate** time: record it
with an `AtTime` snapshot and a custom statistic, or watch every division with an
`on_division` hook (see [Hooks](blocks.md#on_division)). A hook that does not draw from
the rng leaves the run unchanged.

## Persisting a run

[`save_tree`](@ref) writes a population's tree (or any subtree) to a small binary file —
the columns of the [`LineageTree`](@ref), about 44 bytes per node, written and read at
disk speed — and [`load_tree`](@ref) reads it back:

```@example rec
path = joinpath(mktempdir(), "run.nmet")
save_tree(path, pop)
tree, t = load_tree(path)
root = single_root(tree)
(filesize(path), t == pop.t, site_frequency_spectrum(root) == site_frequency_spectrum(pop))
```

The format is versioned and does not depend on Julia's `Serialization`, so files stay
readable across Julia and package versions. It stores the tree and the clock, not the
event queue: a loaded tree is for analysis, not for continuing the simulation. To keep
less, store statistics, or a [`LeafSample`](@ref) (its `root` saves the same way) — a
real tree at a fraction of the size that replays exactly from its recorded seed.
