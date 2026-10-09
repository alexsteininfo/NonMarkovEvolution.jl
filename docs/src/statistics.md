# Tree statistics

Everything here is computed after a run. Two families of methods exist:

- **Population methods** take a [`Population`](@ref) and return one entry per living
  cell, in increasing id order (`alive_cells(pop)`).
- **Root methods** take a `CellNode` and return one entry per leaf, in `Leaves(root)`
  order (`alive_cells(root)`). They work on any subtree, including the induced tree of a
  [`sample_leaves`](@ref) draw.

```@example stats
using NonMarkovEvolution, Distributions, Random, Statistics
block = NonMarkovBlock(
    birth_dist = f -> Gamma(5.0, 1 / (5 * f)), death_dist = f -> Exponential(4.0),
    effect_dist = Exponential(0.05), fitness_update = (f, δ) -> f + δ, ν = 0.5,
    stopfunction = pop -> popsize(pop) >= 500, restart_on_extinction = true)
pop  = simulate!(initialize_population(), block, MersenneTwister(11))
root = single_root(pop)
popsize(root)
```

## Mutation burden

```julia
mutations_per_cell(pop)                          # full burden per living cell
mutations_per_cell(root)                         # acquired strictly below root (default)
mutations_per_cell(root; includeclonal = true)   # full burden, in Leaves order
mean_mutations(pop), var_mutations(pop)
clonal_mutations(pop)                            # carried by every living cell
```

Burdens are stored on the cells (`total_mutations`), so these are ``O(N)``. For a root
method, `includeclonal` says whether mutations carried by *every* leaf under `root` count:
with the default `false`, `root`'s own mutations and its ancestors' are left out, because
they are clonal in that subtree. For the founder of a simulated tree the two agree.
[`clonal_mutations`](@ref) is the burden of the MRCA of all living cells, `0` on a forest.

```@example stats
sub = find_mrca(alive_cells(root)[1:20])
(mean(mutations_per_cell(sub)), mean(mutations_per_cell(sub; includeclonal = true)))
```

### Dropping high-frequency variants

[`filtered_mutations_per_cell`](@ref)`(root, threshold)` leaves out the mutations of every
ancestral node subtending more than `floor(threshold * N)` of the `N` leaves under
`root` — the tree-side analogue of filtering high-frequency variants before estimating a
mutation rate, since such nodes carry no information about within-clone divergence. It
counts from each leaf up to and including `root`; `threshold = 1.0` leaves nothing out.

## Spectra

```julia
site_frequency_spectrum(pop)         # length popsize(pop); every tree of a forest counts
site_frequency_spectrum(root[, N])   # length N, default the leaf count n
branch_spectrum(root[, N])           # topological SFS
```

`sfs[k]` is the number of **mutation events** carried by exactly `k` leaves; `bs[k]` the
number of **internal nodes** subtending exactly `k` leaves. `N > n` pads with zeros —
useful for sampled trees, where the meaningful length is the sample size — and `N < n` is
an `ArgumentError`. Mutations on the root's own edge count as clonal, in `sfs[n]`.

### Separating topology from the mutation rate

`branch_spectrum` is the SFS a tree would carry at one neutral mutation per daughter per
division. For any neutral rate ``m``:

```math
\mathbb{E}[\texttt{sfs}[k]] = m \cdot \texttt{bs}[k] \quad (k \ge 2), \qquad
\mathbb{E}[\texttt{sfs}[1]] = m \cdot (\texttt{bs}[1] + n).
```

The extra ``n`` is the leaves' own edges: `bs` counts only internal nodes, so `bs[1]` is
the number of **unary** nodes (divisions whose other lineage died out), not of leaves. One
simulated tree then serves every ``m``, and a fitted ``m`` never absorbs topology.

```@example stats
using AbstractTrees
bs = branch_spectrum(root)
n_internal = count(n -> !isempty(children(n)), PreOrderDFS(root))
(sum(bs) == n_internal, bs[end])    # every internal node counted once; the root subtends all
```

## Depths, fitness and lifetimes

```julia
leaf_depths(root)                     # divisions from root to each leaf (pooled order)
leaf_depths(root; order = :leaves)    # the same, in Leaves order
leaf_fitness(root)                    # in Leaves order
cell_lifetimes(root)                  # completed lifetimes of every cell that divided
cell_lifetimes(root; include_alive = true, tnow = pop.t)
cell_lifetime(node, pop.t)            # one cell
```

[`leaf_depths`](@ref) is the primary observable of a neutral run: it counts real
divisions, unary nodes included, and gives the burden of untracked neutral mutations as
`Poisson(m * depth)`. Its default order is historical (right subtrees first) and
co-indexed with nothing; it is kept fixed because stored results depend on it. Pass
`order = :leaves` to pair depths with burdens or fitness.

`cell_lifetimes` is the realised cell-cycle distribution and is **not** an unbiased
sample of `birth_dist`. Competing risks remove cells that died first, and stopping at a
fixed size favours short cycles among the completed ones — the long ones are still in
progress — even without death (a pure-birth run with mean cycle 1.0 gives a mean
completed lifetime near 0.87). `include_alive = true` adds right-censored ages rather than
completed lifetimes, so it does not remove the bias. It is what an experiment measuring
interdivision times would see, not a check of `birth_dist`.

## Which results are co-indexed

| Order | Functions |
|:---|:---|
| increasing id (`alive_cells(pop)`) | `mutations_per_cell(pop)`, `fitness_per_cell(pop)`, `pairwise_distances(pop, idx)` indices |
| `Leaves(root)` (`alive_cells(root)`) | `mutations_per_cell(root)`, `leaf_fitness`, `filtered_mutations_per_cell`, `leaf_depths(root; order = :leaves)` |
| its own (pooled only) | `leaf_depths(root)` |

## Distances and coalescence

```julia
pairwise_distance(node1, node2)        # mutations differing between two cells
pairwise_distances(pop[, idx])         # every pair, flat Vector{Int64}
pairwise_distances(root[, idx])
coalescence_times(pop[, idx]; t = pop.t)
coalescence_times(root[, idx]; t = last_division_time(root))
```

[`pairwise_distance`](@ref) is the burden of both cells minus twice their MRCA's burden.
[`coalescence_times`](@ref) is the time from `t` back to the division of each pair's
MRCA; the two methods default to different reference times, so pass `t` to compare
across runs. For a histogram use `StatsBase.countmap(pairwise_distances(...))`.

!!! warning "These are quadratic"
    The pairwise functions build all ``\binom{N}{2}`` pairs. At ``N = 10^4`` that is
    ``5 \times 10^7``. Subsample with a recorded seed instead: the sampled tree keeps
    every ancestor, so these are the cells' true full-population distances.

```@example stats
s = sample_leaves(pop, 50; seed = 1)
(length(pairwise_distances(s.root)), round(mean(coalescence_times(s.root; t = pop.t)), digits = 2))
```

## Populations with more than one root

`initialize_population(N)` seeds `N` independent founders, so the population is a forest.

| Function | On a forest |
|:---|:---|
| `single_root(pop)` | `nothing` |
| `roots(pop)` | the distinct roots, one per tree |
| `find_mrca(pop)`, `clonal_mutations(pop)` | `nothing`, `0` |
| `site_frequency_spectrum(pop)`, per-cell functions | correct — every tree contributes |
| `sample_leaves`, `sample_trees` | throw, naming the number of roots |
| `coalescence_times`, `pairwise_distance` across trees | a convention, see below |

Two cells from different founders have no common ancestor. Their distance is the sum of
both burdens, and their coalescence time is taken back to the earlier founder's birth —
a convention that puts an artefactual spike at ``t - t_0`` into a pooled histogram. Work
root by root for per-lineage statistics:

```julia
for r in roots(pop)
    popsize(r) >= 2 && println(popsize(r), " cells, SFS ", site_frequency_spectrum(r))
end
```

## Tree utilities

```julia
find_mrca(node1, node2)      # nothing if in different trees
find_mrca(nodes)             # of a vector of nodes
find_mrca(pop)               # of all living cells
single_root(nodes), roots(nodes)

# tree construction, for fixtures: ids must increase with every node added
root = CellNode(NonMarkovCell(1, 0.0, 0, 0, 1.0))          # a new one-node tree
left_child!(root, NonMarkovCell(2, 1.0, 1, 1, 1.0))
right_child!(root, NonMarkovCell(3, 1.0, 0, 0, 1.0))
```

`find_mrca` relies on ids increasing along every lineage, which holds for every tree
`simulate!` builds; a hand-built tree must respect it too.

## Cost summary

``N`` is the number of living cells, ``D`` the typical depth, ``T`` the number of nodes.
Population methods sweep the tree's rows in order, which is the cache-friendly case;
root methods first list the subtree.

| Function | Cost |
|:---|:---|
| `mutations_per_cell`, `fitness_per_cell`, `mean_mutations` (population) | ``O(T)``, one sequential pass |
| `site_frequency_spectrum(pop)` | ``O(T)``, one reverse pass |
| root methods: spectra, `leaf_depths`, `filtered_mutations_per_cell`, `cell_lifetimes` | ``O(T)`` |
| `single_root(pop)`, `roots(pop)`, `find_mrca(pop)` | ``O(T)`` |
| `pairwise_distances`, `coalescence_times` | ``O(N^2 D)`` |

At ``N = 10^6`` the population-wide burden and SFS take about 10–15 ms.
