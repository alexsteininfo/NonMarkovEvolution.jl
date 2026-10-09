# Post-hoc statistics of a population or a lineage tree. Population methods return one
# entry per living cell in increasing id order (`alive_cells(pop)`); root methods in
# `Leaves(root)` order (`alive_cells(root)`), except `leaf_depths` by default. All of them
# work on the tree's index arrays: a population-wide spectrum is one reverse sweep over
# the rows (children come after their parents), a subtree one pre-order listing.

# ── Mutation burden and fitness ───────────────────────────────────────────────

"""
    mutations_per_cell(population) -> Vector{Int64}
    mutations_per_cell(root::CellNode; includeclonal = false) -> Vector{Int64}

Mutation burden of every living cell: for a population, each cell's full burden in
`alive_cells(population)` order; for a tree, in `Leaves(root)` order, where

- `includeclonal = false` (default) counts only mutations acquired strictly *below*
  `root`. Mutations on `root` itself and on its ancestors are carried by every leaf of the
  subtree, so they are clonal there and left out.
- `includeclonal = true` gives each leaf's full burden, back to the top of its tree.

For the founder of a simulated tree the two agree, because it carries no mutations.
"""
function mutations_per_cell(population::Population)
    tree = population.tree
    return Int64[tree.total[i] for i in _alive_idx(tree)]
end

function mutations_per_cell(root::CellNode; includeclonal::Bool = false)
    tree, r = root.tree, _index(root)
    offset = includeclonal ? 0 : Int64(tree.total[r])
    return Int64[tree.total[i] - offset for i in _leaves_idx(tree, r)]
end

"""
    clonal_mutations(population) -> Int64

Number of mutations carried by every living cell (the MRCA's burden), or `0` if the cells
have no common ancestor.
"""
function clonal_mutations(population::Population)
    mrca = find_mrca(population)
    return isnothing(mrca) ? 0 : mrca.total_mutations
end

"""
    mean_mutations(population) -> Float64

Mean mutation burden across living cells.
"""
mean_mutations(population::Population) = mean(_burdens(population))

"""
    var_mutations(population) -> Float64

Variance of the mutation burden across living cells.
"""
var_mutations(population::Population) = var(_burdens(population))

# Burdens and fitnesses of the living cells, in id order.
function _burdens(population::Population)
    tree = population.tree
    return Float64[tree.total[i] for i in _alive_idx(tree)]
end
function _fitnesses(population::Population)
    tree = population.tree
    return Float64[tree.fitness[i] for i in _alive_idx(tree)]
end

"""
    fitness_per_cell(population) -> Vector{Float64}

Fitness of every living cell, in `alive_cells(population)` order.
"""
fitness_per_cell(population::Population) = _fitnesses(population)

"""
    leaf_fitness(root::CellNode) -> Vector{Float64}

Fitness of every leaf under `root`, in `Leaves(root)` order — co-indexed with
[`mutations_per_cell`](@ref)`(root)`.
"""
function leaf_fitness(root::CellNode)
    tree = root.tree
    return Float64[tree.fitness[i] for i in _leaves_idx(tree, _index(root))]
end

"""
    filtered_mutations_per_cell(root::CellNode, threshold::Real) -> Vector{Int}

Per-leaf mutation burden, counted from the leaf up to and including `root`, leaving out
the mutations of any ancestral node whose leaf count exceeds `floor(threshold * N)`, where
`N` is the number of leaves under `root`. A leaf's own mutations always count.

This is the tree-side analogue of dropping high-frequency variants before estimating a
mutation rate. `threshold = 1.0` leaves nothing out, and on the top of a tree equals
`mutations_per_cell(root; includeclonal = true)`. Returned in `Leaves(root)` order.
"""
function filtered_mutations_per_cell(root::CellNode, threshold::Real)
    tree = root.tree
    nodes, ppos = _preorder_idx(tree, _index(root))
    counts    = _leafcounts(tree, nodes, ppos)
    max_count = floor(Int, threshold * counts[1])

    # Forward sweep: parents precede children, so `above[p]` is final when read.
    above  = zeros(Int, length(nodes))   # filtered burden down to and including node k
    result = Int[]
    for k in eachindex(nodes)
        i       = nodes[k]
        inherit = ppos[k] == 0 ? 0 : above[ppos[k]]
        if _isleaf(tree, i)
            push!(result, inherit + tree.mutations[i])
        else
            above[k] = inherit + (counts[k] <= max_count ? Int(tree.mutations[i]) : 0)
        end
    end
    return result
end

# ── Distances and coalescence ─────────────────────────────────────────────────

# `f(cells[i], cells[j])::T` for every pair i < j, in row-major order; `idx`, if given,
# restricts `cells` first.
function _pairwise(f, ::Type{T}, cells::AbstractVector, idx = nothing) where T
    isnothing(idx) || (cells = cells[idx])
    n   = length(cells)
    out = Vector{T}(undef, n * (n - 1) ÷ 2)
    k   = 0
    for i in 1:n, j in i+1:n
        out[k += 1] = f(cells[i], cells[j])
    end
    return out
end

"""
    pairwise_distance(node1, node2) -> Int64

Number of mutations that differ between two cells: those on the path from each cell up to
their MRCA, excluding the MRCA itself. Cells in different trees share nothing, so their
distance is the sum of both burdens.
"""
function pairwise_distance(node1::CellNode, node2::CellNode)
    mrca   = find_mrca(node1, node2)
    shared = isnothing(mrca) ? 0 : mrca.total_mutations
    return node1.total_mutations + node2.total_mutations - 2shared
end

"""
    pairwise_distances(population[, idx]) -> Vector{Int64}
    pairwise_distances(root::CellNode[, idx]) -> Vector{Int64}

[`pairwise_distance`](@ref) for every pair of living cells, as a flat vector over pairs
`i < j`. `idx` restricts to `alive_cells(population)[idx]` (or `alive_cells(root)[idx]`).
Quadratic in the number of cells: subsample with [`sample_leaves`](@ref) at scale. For a
histogram use `StatsBase.countmap(pairwise_distances(...))`.
"""
pairwise_distances(population::Population, idx = nothing) =
    _pairwise(pairwise_distance, Int64, alive_cells(population), idx)
pairwise_distances(root::CellNode, idx = nothing) =
    _pairwise(pairwise_distance, Int64, alive_cells(root), idx)

# Time from `t` back to the division of the two cells' MRCA. For cells in different
# trees this is, by convention, the time back to the earlier of the two founders' births.
function _coalescence_time(node1::CellNode, node2::CellNode, t::Real)
    node1 == node2 && return 0.0
    mrca = find_mrca(node1, node2)
    if isnothing(mrca)
        root_birth(n) = n.tree.birthtime[_rootidx(n.tree, _index(n))]
        return t - min(root_birth(node1), root_birth(node2))
    end
    return t - division_time(mrca)
end

"""
    coalescence_times(population[, idx]; t = population.t) -> Vector{Float64}
    coalescence_times(root::CellNode[, idx]; t = last_division_time(root)) -> Vector{Float64}

For every pair of living cells, the time from `t` back to the division of their MRCA.
`idx` works as for [`pairwise_distances`](@ref). The two methods default to different
reference times; pass `t` to compare across runs. Cells from different founders are
treated as coalescing at the earlier founder's birth, a convention.
"""
coalescence_times(population::Population, idx = nothing; t::Real = population.t) =
    _pairwise((a, b) -> _coalescence_time(a, b, t), Float64, alive_cells(population), idx)
coalescence_times(root::CellNode, idx = nothing; t::Real = last_division_time(root)) =
    _pairwise((a, b) -> _coalescence_time(a, b, t), Float64, alive_cells(root), idx)

# ── Spectra ───────────────────────────────────────────────────────────────────

function _check_spectrum_length(nleaves::Int, N::Int, what::String)
    nleaves <= N || throw(ArgumentError(
        "$what: tree has $nleaves leaves but N = $N — the spectrum would " *
        "overflow. Pass N >= $nleaves."))
    return nothing
end

# Add `weight(tree, i)` to `spectrum[k]` for every node `i` under `root` subtending k > 0
# leaves.
function _fill_spectrum!(spectrum::Vector, root::CellNode, weight, what::String)
    tree = root.tree
    nodes, ppos = _preorder_idx(tree, _index(root))
    counts = _leafcounts(tree, nodes, ppos)
    _check_spectrum_length(counts[1], length(spectrum), what)
    @inbounds for k in eachindex(nodes)
        c = counts[k]
        c > 0 && (spectrum[c] += weight(tree, nodes[k]))
    end
    return spectrum
end

# The same over every live node of the whole tree (all roots), in one reverse sweep:
# children have larger indices than their parents, so a node's count is final when the
# sweep reaches it.
function _fill_spectrum_all!(spectrum::Vector, tree::LineageTree, weight)
    counts = zeros(Int32, length(tree))
    parent = tree.parent
    @inbounds for i in length(tree):-1:1
        _isdead(tree, i) && continue
        c = _isleaf(tree, i) ? Int32(1) : counts[i]
        c > 0 && (spectrum[c] += weight(tree, i))
        p = parent[i]
        p > NOPARENT && (counts[p] += c)
    end
    return spectrum
end

_own_mutations(tree::LineageTree, i) = @inbounds Int64(tree.mutations[i])
_is_internal(tree::LineageTree, i) = Int(_haschildren(tree, i))

"""
    site_frequency_spectrum(population) -> Vector{Int64}
    site_frequency_spectrum(root::CellNode[, N]) -> Vector{Int64}

Mutation site-frequency spectrum: `sfs[k]` is the number of mutation events carried by
exactly `k` living cells. For a population it has length `popsize(population)` and every
tree of a forest contributes. For a tree it has length `N`, by default the number `n` of
leaves; `N > n` pads with zeros (useful for sampled trees, where the meaningful length
is the sample size), `N < n` is an error. Mutations on the root's own edge count as
clonal, in `sfs[n]`.
"""
site_frequency_spectrum(population::Population) =
    _fill_spectrum_all!(zeros(Int64, popsize(population)), population.tree, _own_mutations)

site_frequency_spectrum(root::CellNode, N::Int = popsize(root)) =
    _fill_spectrum!(zeros(Int64, N), root, _own_mutations, "site_frequency_spectrum")

"""
    branch_spectrum(root::CellNode[, N]) -> Vector{Int}

Topological site-frequency spectrum: `bs[k]` is the number of internal nodes subtending
exactly `k` leaves; leaves are not counted. Length and `N` work as for
[`site_frequency_spectrum`](@ref).

It separates topology from the mutation rate: for neutral mutations at rate `m` per
daughter per division, `E[sfs[k]] = m * bs[k]` for `k ≥ 2` and
`E[sfs[1]] = m * (bs[1] + n)`, where `n` is the number of leaves. The extra `n` is the
leaves' own edges; `bs[1]` counts only *unary* internal nodes (divisions whose other
lineage died out).
"""
branch_spectrum(root::CellNode, N::Int = popsize(root)) =
    _fill_spectrum!(zeros(Int, N), root, _is_internal, "branch_spectrum")

# ── Leaf divisional depths ────────────────────────────────────────────────────

"""
    leaf_depths(root::CellNode; order = :stack) -> Vector{Int}

Number of divisions on the path from `root` to each leaf. For neutral simulations this
is the primary quantity: a leaf's burden of untracked neutral mutations at rate `m` is
`Poisson(m * depth)`.

- `order = :stack` (default) — the historical order (right subtrees first), which is
  *not* co-indexed with anything. Kept as the default because stored results depend on
  it; valid as a pooled distribution.
- `order = :leaves` — `Leaves(root)` order, co-indexed with
  [`mutations_per_cell`](@ref)`(root)` and [`leaf_fitness`](@ref).
"""
function leaf_depths(root::CellNode; order::Symbol = :stack)
    order in (:stack, :leaves) || throw(ArgumentError(
        "leaf_depths: order must be :stack or :leaves, got :$order"))
    leaves_order = order === :leaves
    tree = root.tree
    left, right = tree.left, tree.right
    depths = Int[]
    stack  = Tuple{Int32, Int}[(_index(root), 0)]
    @inbounds while !isempty(stack)
        i, d = pop!(stack)
        l, r = left[i], right[i]
        if l == NOPARENT && r == NOPARENT
            push!(depths, d)
        elseif leaves_order     # push right first so that left pops first
            r == NOPARENT || push!(stack, (r, d + 1))
            l == NOPARENT || push!(stack, (l, d + 1))
        else
            l == NOPARENT || push!(stack, (l, d + 1))
            r == NOPARENT || push!(stack, (r, d + 1))
        end
    end
    return depths
end
