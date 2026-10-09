# Sampling

Experiments sequence a few hundred or thousand cells out of millions. A simulated
observable is comparable to data only after the same bottleneck, so sampling is a
first-class operation. It is **post-hoc**: it applies to a finished tree, and is
deliberately not part of [`NonMarkovBlock`](@ref) or [`MeasurementSpec`](@ref).

```@example sampling
using NonMarkovEvolution, Distributions, Random, Statistics
block = NonMarkovBlock(
    birth_dist = f -> Gamma(5.0, 1 / (5 * f)), death_dist = f -> Exponential(4.0),
    effect_dist = Exponential(0.05), fitness_update = (f, δ) -> f + δ, ν = 0.5,
    stopfunction = pop -> popsize(pop) >= 2_000, restart_on_extinction = true)
pop = simulate!(initialize_population(), block, MersenneTwister(5))

out = sample_trees(pop, SamplingSpec([200, 20]); seed = 20260930)
s   = out.samples[1]                                # a LeafSample with n = 200
(s.n, s.N_full, length(site_frequency_spectrum(s.root, s.n)))
```

## One draw

[`sample_leaves`](@ref)`(root_or_pop, n; seed, replicate = 1)` draws `n` leaves uniformly
without replacement and returns the induced tree as a [`LeafSample`](@ref):

| Field | Contents |
|:---|:---|
| `root` | the induced tree |
| `n` | cells drawn |
| `N_full` | leaf count of the source tree |
| `seed` | the seed this draw used |
| `replicate` | which independent draw at this `(tree, n)` this is |
| `sampled_ids` | `NonMarkovCell.id` of each drawn cell, in draw order |

It is non-destructive, so the same tree can be drawn from again at other sizes. It costs
one pass over the source tree's leaves plus the retained nodes; [`sample_trees`](@ref)
lists the leaves once for all its draws.

## Prune, do not collapse

The induced tree keeps the sampled leaves **plus every ancestor of a sampled leaf**, and
keeps the resulting unary nodes. Every division ancestral to a sampled cell is still a
node, so a sampled cell's root-to-leaf path is unchanged:
[`mutations_per_cell`](@ref)`(s.root; includeclonal = true)` and [`leaf_depths`](@ref)
return exactly that cell's **full-tree** burden and depth. Collapsing would turn depth
into a count of bifurcations that happened to survive sampling — a property of the
sample, not of the cell. It is also the shape
[pruning](concepts.md#The-tree-records-the-survivors) leaves when a lineage dies out, so
every statistic applies to a sampled tree unchanged.

```@example sampling
full   = Dict(c.data.id => c.data.total_mutations for c in alive_cells(pop))
sample = Dict(zip([l.data.id for l in alive_cells(s.root)],
                  mutations_per_cell(s.root; includeclonal = true)))
all(sample[id] == full[id] for id in s.sampled_ids)
```

!!! warning "The root of a sampled tree is the founder, not the MRCA of the sample"
    Ancestry is retained all the way up, so `site_frequency_spectrum(s.root, s.n)` puts
    the founder's mutations in `sfs[n]` exactly as the full tree puts them in `sfs[N]`. For
    the sample's own MRCA use `find_mrca(alive_cells(s.root))`.

The sampled tree is a copy in its own [`LineageTree`](@ref): ids, times, mutation counts
and fitness are the source cells', and changing it (for example with
[`set_fitness!`](@ref)) leaves the source untouched.

## Declaring what to produce

[`SamplingSpec`](@ref) says what [`sample_trees`](@ref) builds from one finished tree:

| Intent | Spec |
|:---|:---|
| full tree only | `SamplingSpec()` |
| one sample of size `n`, nothing else | `SamplingSpec(sizes = [n], retain_full = false)` |
| full tree plus several sizes | `SamplingSpec([1000, 100])` |
| several independent draws per size | `SamplingSpec(sizes = [1000], replicates = 20)` |

Duplicate sizes are rejected (use `replicates`), and every size is checked against the
tree before any draw is made. Results are ordered by size, then replicate.

`retain_full = false` only controls what the returned bundle holds: while the
`Population` is alive, its whole tree is too, because every living cell holds a `parent`
chain back to the founder. To bound memory across a sweep, drop the population yourself
(`pop = nothing`) once it has been sampled.

## Reproducibility

`seed` is required: the draw is a pure function of `(tree, n, seed)`, and each
`LeafSample` records its seed, so any single draw replays in isolation. Supply seeds
derived from your own provenance (a simulation index, a sweep coordinate); two draws that
should differ need different seeds. Within one batch, `sample_trees` derives a distinct
per-draw seed from the base seed, the size and the replicate index.

The recipe is **stable across Julia versions**: `StableRNG(seed)` drives a partial
Fisher–Yates shuffle over `Leaves(root)` order, and per-draw seeds use an explicit
splitmix64 mix. Golden tests pin both (see
[Stability guarantees](concepts.md#Stability-guarantees)). Each call builds its own rng,
so draws are safe to run concurrently.

For one seed, draws at different `n` are **nested**: the `n = 10` draw is the first 10
cells of the `n = 100` draw. That is harmless within `sample_trees`, whose per-draw seeds
differ by size, but pass different seeds yourself when you need independent draws at
several sizes.

## Sampling needs a single-rooted tree

`sample_leaves` and `sample_trees` throw an `ArgumentError` naming the number of roots
when handed a forest. There is no uniform draw across independent trees that keeps "the
root is the founder", so sample each tree separately:

```julia
samples = [sample_leaves(r, min(100, popsize(r)); seed = i) for (i, r) in enumerate(roots(pop))]
```

## A typical sweep

```julia
using NonMarkovEvolution, Distributions, Random, Serialization

block(s) = NonMarkovBlock(
    birth_dist = f -> Gamma(5.0, 1 / (5 * f)), death_dist = f -> Exponential(1 / 0.3),
    effect_dist = Dirac(s), fitness_update = (f, δ) -> f + δ, ν = 0.2,
    stopfunction = pop -> popsize(pop) >= 100_000, restart_on_extinction = true)

for (i, s) in enumerate(0.0:0.1:0.5)
    pop = simulate!(initialize_population(), block(s), MersenneTwister(i))
    out = sample_trees(pop, SamplingSpec(sizes = [1000, 100], replicates = 10,
                                         retain_full = false); seed = 20260930 + i)
    serialize("s=$(s).jls", out.samples)   # small: the samples, not the population
end
```

`serialize` keeps the samples' metadata (`n`, `seed`, `sampled_ids`) with them, but ties
the file to Julia and package versions. For long-term storage write each tree with
[`save_tree`](@ref)`(path, sample.root)` and the metadata as a small table.

A `LeafSample` at ``n = 1000`` is a tree of a few thousand nodes, against a full history
orders of magnitude larger — and it replays exactly from its recorded seed.
