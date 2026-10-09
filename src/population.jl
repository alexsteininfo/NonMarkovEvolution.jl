# The population: a lineage tree, the clock, the number of living cells, and the queue
# of already-drawn events carried between `simulate!` calls.

"""
    Population

The lineage tree of every cell that has lived and still has living descendants
(`pop.tree`, a [`LineageTree`](@ref)), and the current simulation time `t` (the time of
the most recently processed event). The living cells are the tree's leaves; read them
with [`alive_cells`](@ref), count them with [`popsize`](@ref). Build one with
[`initialize_population`](@ref).

`simulate!` also carries a private queue of already-drawn events (one per cell, see
[`has_pending_schedule`](@ref)). Treat the tree as read-only except through
[`set_fitness!`](@ref).
"""
mutable struct Population
    tree::LineageTree
    t::Float64
    _next_id::Int64                          # id of the newest cell
    _nalive::Int
    _pending::Union{Nothing, EventQueue}     # carried event queue
    _protect::Int32                          # rows 1:_protect are never compacted
    _ndead_protected::Int                    # removed rows among them
    _queue_hint::Int                         # capacity for a freshly built queue
end

"""
    Population(tree::LineageTree, t; next_id = maximum(tree.id))

A population whose cells are the leaves of `tree` (for example one returned by
[`load_tree`](@ref)), at clock `t`. New cells get ids after `next_id`, which must be at
least the largest id in the tree. The tree must hold no removed rows.
"""
function Population(tree::LineageTree, t::Real; next_id::Integer = maximum(tree.id; init = 0))
    next_id >= maximum(tree.id; init = 0) || throw(ArgumentError(
        "Population: next_id = $next_id is below the largest id in the tree " *
        "($(maximum(tree.id))); new cells would reuse ids"))
    any(==(REMOVED), tree.parent) && throw(ArgumentError(
        "Population: the tree holds removed rows; build it with load_tree or a copy"))
    return Population(tree, Float64(t), Int64(next_id), length(_alive_idx(tree)), nothing,
                      Int32(0), 0, 0)
end

"""
    initialize_population(; fitness_init = 1.0, time = 0.0) -> Population
    initialize_population(N::Int; fitness_init = 1.0, time = 0.0) -> Population

A population of `N` (default 1) identical founding cells with fitness `fitness_init`,
born at `time`, with ids `1:N` and no mutations. Each founder is the root of its own
tree, so `N > 1` gives a forest.
"""
function initialize_population(; fitness_init::Real = 1.0, time::Real = 0.0)
    return initialize_population(1; fitness_init = fitness_init, time = time)
end

function initialize_population(N::Int; fitness_init::Real = 1.0, time::Real = 0.0)
    N >= 1 || throw(ArgumentError("initialize_population: N must be >= 1, got $N"))
    tree = LineageTree()
    for id in 1:N
        _push_node!(tree, NOPARENT, Int64(id), Float64(time), 0, 0, Float64(fitness_init))
    end
    return Population(tree, Float64(time), Int64(N), N, nothing, Int32(0), 0, 0)
end

"""
    sizehint!(population, ncells) -> population

Reserve memory for growing to about `ncells` living cells (`2 ncells` tree rows and
`ncells` queued events). Optional: without it the arrays grow by doubling, which
triggers a few full garbage collections per run (about 0.3 s of 1.1 s at 10⁶ cells on
the reference machine). Does not change the simulation.
"""
function Base.sizehint!(population::Population, ncells::Integer)
    sizehint!(population.tree, 2 * Int(ncells))
    population._queue_hint = Int(ncells)
    isnothing(population._pending) || sizehint!(population._pending, Int(ncells))
    return population
end

"""
    popsize(population) -> Int
    popsize(node::CellNode) -> Int

Number of living cells: in the population, or among the leaves under `node`.
"""
popsize(population::Population) = population._nalive
popsize(node::CellNode) = length(_leaves_idx(node.tree, _index(node)))

"""
    alive_cells(population) -> Vector{CellNode}
    alive_cells(node::CellNode) -> Vector{CellNode}

The living cells: of the population in increasing id order, or the leaves under `node`
in `Leaves(node)` order.
"""
alive_cells(population::Population) =
    CellNode[CellNode(population.tree, i) for i in _alive_idx(population.tree)]
alive_cells(node::CellNode) =
    CellNode[CellNode(node.tree, i) for i in _leaves_idx(node.tree, _index(node))]

"""
    has_pending_schedule(population) -> Bool

Whether `population` carries a queue of already-drawn events, which the next `simulate!`
call will reuse. False for a fresh population and after [`reset_schedule!`](@ref).
"""
has_pending_schedule(population::Population) = !isnothing(population._pending)

"""
    reset_schedule!(population) -> Population

Discard the queue of already-drawn events carried on `population`, so that the next
`simulate!` call redraws every living cell's next event under its block. The redraw is
conditioned on each cell's current age, so it is exact.
"""
function reset_schedule!(population::Population)
    population._pending = nothing
    return population
end

Base.show(io::IO, pop::Population) =
    print(io, "Population: $(popsize(pop)) cells (t = $(round(pop.t, digits = 3)))")

# ── Compaction ────────────────────────────────────────────────────────────────

# Compaction thresholds: compact once the removable dead rows are at least this many,
# and at least `_COMPACT_FRACTION` of the tree. Refs so tests can force compaction.
const _COMPACT_MIN = Ref(4096)
const _COMPACT_FRACTION = Ref(0.5)

@inline function _maybe_compact!(pop::Population, remap_alive!)
    tree = pop.tree
    free = tree.ndead - pop._ndead_protected
    (free >= _COMPACT_MIN[] && free >= _COMPACT_FRACTION[] * length(tree)) || return nothing
    _compact_population!(pop, remap_alive!)
    return nothing
end

@noinline function _compact_population!(pop::Population, remap_alive!)
    start = pop._protect + Int32(1)
    remap = _compact!(pop.tree, start)
    isnothing(pop._pending) || _remap!(pop._pending, remap, start)
    remap_alive!(remap, start)
    # Every removed row left is in the protected prefix.
    pop._ndead_protected = pop.tree.ndead
    return nothing
end
