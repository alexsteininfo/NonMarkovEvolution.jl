# Uniform leaf sampling of lineage trees.
#
# The induced tree keeps the sampled leaves plus *every ancestor* of a sampled leaf, and
# retains the resulting unary nodes rather than collapsing them. Every division ancestral
# to a sampled cell is still a node, so a sampled cell's root-to-leaf path is unchanged:
# `mutations_per_cell(root; includeclonal = true)` and `leaf_depths` return its full-tree
# burden and divisional depth. It is the same shape `_prune!` leaves behind, so every
# statistic applies to a sampled tree unchanged.

"""
    LeafSample

One uniform draw of `n` leaves from a lineage tree, with the induced tree.

# Fields
- `root::CellNode` — root of the induced tree (its own [`LineageTree`](@ref)): the sampled leaves plus every
  ancestor of a sampled leaf, unary nodes retained. The original founder remains
  the root even when it has a single child, so `site_frequency_spectrum` counts its
  mutations in `sfs[n]` exactly as it does in `sfs[N]` for a full tree. **The root is
  therefore the founder, not the MRCA of the sample** — call `find_mrca` on the sampled
  leaves if you need that.
- `n::Int` — cells drawn.
- `N_full::Int` — leaf count of the source tree.
- `seed::UInt64` — rng seed of this draw. The draw is a pure function of
  `(tree, n, seed)`, so any single draw replays in isolation from this record.
- `replicate::Int` — which independent draw this is at this `(tree, n)`. Defaults
  to 1; distinct replicates require distinct seeds, which
  [`sample_trees`](@ref) derives for you.
- `sampled_ids::Vector{Int64}` — `NonMarkovCell.id` of each drawn cell, in draw
  order.

The induced tree is a copy: ids, birthtimes, mutation counts and fitness are those of
the source cells, and changing it (for example with [`set_fitness!`](@ref)) leaves the
source untouched.
"""
struct LeafSample
    root::CellNode
    n::Int
    N_full::Int
    seed::UInt64
    replicate::Int
    sampled_ids::Vector{Int64}
end

# Copy the rows `kept` (sorted indices, `kept[1]` the root of the copy) into a new tree,
# in the same order. Left/right slots are preserved, so a node whose left lineage was
# dropped keeps no left child — the tree stays a faithful sub-shape of the original
# rather than being re-balanced.
function _copy_rows(tree::LineageTree, kept::Vector{Int32})
    new = LineageTree()
    sizehint!(new, length(kept))
    newindex(i) = Int32(searchsortedfirst(kept, i))
    for (k, i) in enumerate(kept)
        p = k == 1 ? NOPARENT : newindex(tree.parent[i])
        _push_node!(new, p, tree.id[i], tree.birthtime[i], tree.mutations[i],
                    tree.total[i], tree.fitness[i])
        if p != NOPARENT
            if tree.left[tree.parent[i]] == i
                new.left[p] = Int32(k)
            else
                new.right[p] = Int32(k)
            end
        end
    end
    return new
end

# The first `n` entries of a uniformly random permutation of `1:N` (partial
# Fisher–Yates). Uses only `rand(rng, a:b)`, whose stream `StableRNG` guarantees across
# Julia and StableRNGs versions — unlike `randperm`, whose algorithm Julia changed in
# 1.11. Changing this function invalidates every stored draw.
function _draw_indices(rng::StableRNG, N::Int, n::Int)
    perm = collect(1:N)
    for i in 1:n
        j = rand(rng, i:N)
        perm[i], perm[j] = perm[j], perm[i]
    end
    return perm[1:n]
end

# Draw from an already-collected leaf list, so `sample_trees` traverses the tree once.
function _sample_leaves(root::CellNode, leaves::Vector{Int32},
                        n::Int, seed::UInt64, replicate::Int)
    N_full = length(leaves)
    1 <= n <= N_full || throw(ArgumentError(
        "cannot draw n = $n cells from a tree with $N_full leaves"))
    replicate >= 1 || throw(ArgumentError("replicate must be >= 1, got $replicate"))

    idx = _draw_indices(StableRNG(seed), N_full, n)

    # Mark each sampled leaf and its ancestors up to `root`, stopping at the first node
    # already marked, so marking costs the number of retained nodes.
    tree   = root.tree
    r      = _index(root)
    marked = falses(length(tree))
    kept   = Int32[]
    for k in idx
        i = leaves[k]
        while !marked[i]
            marked[i] = true
            push!(kept, i)
            i == r && break
            i = tree.parent[i]
        end
    end
    sort!(kept)

    new_root    = CellNode(_copy_rows(tree, kept), 1)
    sampled_ids = Int64[tree.id[leaves[k]] for k in idx]
    return LeafSample(new_root, n, N_full, seed, replicate, sampled_ids)
end

"""
    sample_leaves(root, n; seed, replicate = 1) -> LeafSample
    sample_leaves(population, n; seed, replicate = 1) -> LeafSample

Draw `n` of the tree's leaves uniformly without replacement and return the induced
lineage tree as a [`LeafSample`](@ref).

`seed` (any integer in `0:typemax(UInt64)`) is required and supplied by the caller: the
draw must be reproducible from values the caller itself records, such as a simulation
index. Two draws that should differ must be given different seeds. For one seed, draws at different `n` are nested: the
`n = 1` draw is the first cell of the `n = 2` draw, and so on.

Non-destructive — `root` is left untouched, because the same tree is normally
drawn from again at other sample sizes. Each call builds its own rng from `seed`, so
independent draws are safe to run concurrently from the caller's own threads.

Cost: one O(N) pass to list the leaves, plus the number of retained nodes.

!!! warning "The draw recipe is frozen"
    The draw is a pure function of `(tree, n, seed)`, stable across Julia versions:
    `StableRNG(seed)` drives a partial Fisher–Yates shuffle over `Leaves(root)` order.
    Stored samples depend on it; a golden test pins it.
"""
sample_leaves(root::CellNode, n::Integer; seed::Integer, replicate::Integer = 1) =
    _sample_leaves(root, _leaves_idx(root.tree, _index(root)), Int(n), UInt64(seed),
                   Int(replicate))

sample_leaves(population::Population, n::Integer; seed::Integer, replicate::Integer = 1) =
    sample_leaves(_sampling_root(population), n; seed = seed, replicate = replicate)

# The unique root of a population, or an `ArgumentError` naming why there is none.
function _sampling_root(population::Population)
    roots = _rootidxs(population.tree)
    isempty(roots) && throw(ArgumentError("population has no cells to sample from"))
    length(roots) == 1 || throw(ArgumentError(
        "population has $(length(roots)) independent roots (it is a forest), but " *
        "sampling needs one — sample each root's tree separately"))
    return CellNode(population.tree, only(roots))
end

# ── Declaring what to produce ────────────────────────────────────────────────

"""
    SamplingSpec(; sizes = Int[], replicates = 1, retain_full = true)
    SamplingSpec(n::Int)
    SamplingSpec(sizes::Vector{Int})

What [`sample_trees`](@ref) should produce from one finished tree.

| intent | spec |
|---|---|
| full data only | `SamplingSpec()` |
| one sample of size `n` only | `SamplingSpec(sizes = [n], retain_full = false)` |
| full data plus several sizes | `SamplingSpec([1000, 100])` |

# Keyword arguments
- `sizes` — sample sizes to draw. Empty draws nothing. No duplicates: use
  `replicates` for repeated draws at the same size.
- `replicates` — independent draws per size, each with its own derived seed.
- `retain_full` — whether the returned [`SampledTrees`](@ref) carries the full
  tree.

`retain_full = false` only controls what the returned bundle holds: while the
`Population` is alive, its whole tree is too. See the manual's *Sampling* page for
bounding memory across a sweep.
"""
struct SamplingSpec
    sizes::Vector{Int}
    replicates::Int
    retain_full::Bool

    # The only constructor, so no unvalidated `SamplingSpec` can exist.
    function SamplingSpec(sizes::AbstractVector{<:Integer},
                          replicates::Integer,
                          retain_full::Bool)
        sizes = Int[Int(n) for n in sizes]
        allunique(sizes) || throw(ArgumentError(
            "SamplingSpec: duplicate sample sizes in $sizes — use `replicates` for " *
            "repeated draws at the same size"))
        all(>=(1), sizes) || throw(ArgumentError(
            "SamplingSpec: every sample size must be >= 1, got $sizes"))
        replicates >= 1 || throw(ArgumentError(
            "SamplingSpec: replicates must be >= 1, got $replicates"))
        return new(sizes, Int(replicates), retain_full)
    end
end

function SamplingSpec(; sizes::AbstractVector{<:Integer} = Int[],
                        replicates::Integer = 1,
                        retain_full::Bool = true)
    return SamplingSpec(sizes, replicates, retain_full)
end

SamplingSpec(n::Integer) = SamplingSpec(sizes = [n])
SamplingSpec(sizes::AbstractVector{<:Integer}) = SamplingSpec(sizes = sizes)

"""
    SampledTrees

Result of [`sample_trees`](@ref): the full tree (or `nothing` when
`retain_full = false`) and every draw requested by the spec.

`samples` is ordered by the spec's `sizes`, and within a size by `replicate`.
"""
struct SampledTrees
    full::Union{CellNode, Nothing}
    samples::Vector{LeafSample}
end

# Per-draw seed: splitmix64 finalisation over (base, n, replicate). Written out rather
# than using `Base.hash`, which is not guaranteed stable across Julia versions, so that a
# base seed reproduces the same draws everywhere. Changing it invalidates stored draws.
function _splitmix64(x::UInt64)
    x += 0x9e3779b97f4a7c15
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    return x ⊻ (x >> 31)
end
_draw_seed(base::UInt64, n::Int, replicate::Int) =
    _splitmix64(_splitmix64(_splitmix64(base) ⊻ UInt64(n)) ⊻ UInt64(replicate))

"""
    sample_trees(root, spec; seed) -> SampledTrees
    sample_trees(population, spec; seed) -> SampledTrees

Apply a [`SamplingSpec`](@ref) to one finished tree.

Sizes are validated against the tree *before* any draw is made, so a size larger
than the tree fails immediately. Per-draw seeds are derived from `seed`, the size and
the replicate index, are stable across Julia versions, and are recorded on each
[`LeafSample`](@ref) so that any draw replays in isolation.

Sampling is post-hoc: it applies to a tree that has finished growing. It is
deliberately not part of `NonMarkovBlock` or `MeasurementSpec`, both of which
describe things that happen *during* `simulate!`.
"""
function sample_trees(root::CellNode, spec::SamplingSpec; seed::Integer)
    leaves = _leaves_idx(root.tree, _index(root))
    N_full = length(leaves)
    for n in spec.sizes
        n <= N_full || throw(ArgumentError(
            "SamplingSpec asks for n = $n cells but the tree has $N_full leaves"))
    end

    samples = LeafSample[]
    for n in spec.sizes, r in 1:spec.replicates
        push!(samples, _sample_leaves(root, leaves, n, _draw_seed(UInt64(seed), n, r), r))
    end
    return SampledTrees(spec.retain_full ? root : nothing, samples)
end

sample_trees(population::Population, spec::SamplingSpec; seed::Integer) =
    sample_trees(_sampling_root(population), spec; seed = seed)
