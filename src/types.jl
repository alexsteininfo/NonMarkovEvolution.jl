# Core types: the tree node, the cell, the event, the population — and the explicit-stack
# traversals every other file builds on. Tree utilities live in `simulation_trees.jl`.

# ── BinaryNode ─────────────────────────────────────────────────────────────────

"""
    BinaryNode{T}

Basic unit of a binary tree, used to represent cell lineages. A node with one child is
legitimate: it is a division whose other daughter's lineage died out.

# Fields
- `data::T`
- `parent::Union{Nothing, BinaryNode{T}}`
- `left::Union{Nothing, BinaryNode{T}}`
- `right::Union{Nothing, BinaryNode{T}}`
"""
mutable struct BinaryNode{T}
    data::T
    parent::Union{Nothing, BinaryNode{T}}
    left::Union{Nothing, BinaryNode{T}}
    right::Union{Nothing, BinaryNode{T}}

    function BinaryNode{T}(data, parent = nothing, l = nothing, r = nothing) where T
        new{T}(data, parent, l, r)
    end
end
BinaryNode(data) = BinaryNode{typeof(data)}(data)

"""
    left_child!(parent::BinaryNode, data) -> BinaryNode

Create a new `BinaryNode` from `data` and assign it to `parent.left`.
"""
function left_child!(parent::BinaryNode, data)
    isnothing(parent.left) || error("left child is already assigned")
    parent.left = typeof(parent)(data, parent)
end

"""
    right_child!(parent::BinaryNode, data) -> BinaryNode

Create a new `BinaryNode` from `data` and assign it to `parent.right`.
"""
function right_child!(parent::BinaryNode, data)
    isnothing(parent.right) || error("right child is already assigned")
    parent.right = typeof(parent)(data, parent)
end

haschildren(node::BinaryNode) = !isnothing(node.left) || !isnothing(node.right)
isleaf(node::BinaryNode)      = isnothing(node.left) && isnothing(node.right)

# AbstractTrees interface, so users can call `Leaves`, `PreOrderDFS`, `print_tree`.
function AbstractTrees.children(node::BinaryNode)
    isnothing(node.left)  && return isnothing(node.right) ? () : (node.right,)
    isnothing(node.right) && return (node.left,)
    return (node.left, node.right)
end
AbstractTrees.nextsibling(child::BinaryNode) =
    (p = child.parent; isnothing(p) || child === p.right ? nothing : p.right)
AbstractTrees.prevsibling(child::BinaryNode) =
    (p = child.parent; isnothing(p) || child === p.left ? nothing : p.left)
AbstractTrees.nodevalue(n::BinaryNode) = n.data
AbstractTrees.ParentLinks(::Type{<:BinaryNode}) = StoredParents()
AbstractTrees.parent(n::BinaryNode) = n.parent
AbstractTrees.NodeType(::Type{<:BinaryNode{T}}) where {T} = HasNodeType()
AbstractTrees.nodetype(::Type{<:BinaryNode{T}}) where {T} = BinaryNode{T}
Base.eltype(::Type{<:TreeIterator{BinaryNode{T}}}) where T = BinaryNode{T}
Base.IteratorEltype(::Type{<:TreeIterator{BinaryNode{T}}}) where T = Base.HasEltype()
AbstractTrees.printnode(io::IO, node::BinaryNode) = print(io, node.data)

Base.show(io::IO, node::BinaryNode) = show(io, node.data)

# ── Fast traversals ────────────────────────────────────────────────────────────
#
# AbstractTrees' generic iterators cost about 2 µs per node on these trees, which made
# every post-hoc statistic slower than the simulation that produced the tree. Package
# code traverses with the explicit stacks below instead.

"""
    _leaves(root) -> Vector{BinaryNode}

The leaves under `root`, in exactly `AbstractTrees.Leaves(root)` order (left before
right at every node). Sampling draws index into this order, so it must never change.
"""
function _leaves(root::BinaryNode{T}) where T
    leaves = BinaryNode{T}[]
    stack  = BinaryNode{T}[root]
    while !isempty(stack)
        node = pop!(stack)
        if isleaf(node)
            push!(leaves, node)
        else
            isnothing(node.right) || push!(stack, node.right)
            isnothing(node.left)  || push!(stack, node.left)
        end
    end
    return leaves
end

"""
    _preorder(root) -> (nodes, parent_index)

Every node under `root` in pre-order (left before right), with `parent_index[i]` the
position of `nodes[i]`'s parent in `nodes` (`0` for `root`). Children always come after
their parent, so a reverse sweep is a post-order pass and a forward sweep a pre-order
pass — which is how the spectra and the filtered burden avoid recursion.
"""
function _preorder(root::BinaryNode{T}) where T
    nodes  = BinaryNode{T}[]
    parent = Int[]
    stack  = BinaryNode{T}[root]
    pstack = Int[0]
    while !isempty(stack)
        node = pop!(stack)
        push!(nodes, node)
        push!(parent, pop!(pstack))
        i = length(nodes)
        if !isnothing(node.right)
            push!(stack, node.right); push!(pstack, i)
        end
        if !isnothing(node.left)
            push!(stack, node.left); push!(pstack, i)
        end
    end
    return nodes, parent
end

# Number of leaves below each node of a `_preorder` listing.
function _leafcounts(nodes::Vector{<:BinaryNode}, parent::Vector{Int})
    counts = zeros(Int, length(nodes))
    for i in length(nodes):-1:1
        isleaf(nodes[i]) && (counts[i] = 1)
        parent[i] == 0 || (counts[parent[i]] += counts[i])
    end
    return counts
end

# The distinct roots of the trees that `nodes` belong to, in order of first appearance.
# Climbs from each node but stops at the first node already visited.
function _roots(nodes::AbstractVector{BinaryNode{T}}) where T
    visited = Base.IdSet{BinaryNode{T}}()
    found   = BinaryNode{T}[]
    for start in nodes
        node = start
        while !(node in visited)
            push!(visited, node)
            if isnothing(node.parent)
                push!(found, node)
                break
            end
            node = node.parent
        end
    end
    return found
end

function _treeroot(node::BinaryNode)
    while !isnothing(node.parent)
        node = node.parent
    end
    return node
end

# ── NonMarkovCell ──────────────────────────────────────────────────────────────

"""
    NonMarkovCell

A single cell of the simulation, stored as the `data` of a `BinaryNode`. Immutable.

# Fields
- `id::Int64` — unique cell identifier; always larger than the parent's id
- `birthtime::Float64` — simulation time at which the cell was born
- `mutations::Int64` — mutations acquired at this cell's own birth
- `total_mutations::Int64` — mutations on the whole path from the root to this cell,
  including its own: the parent's `total_mutations + mutations`
- `fitness::Float64` — cumulative fitness (parent fitness updated once per mutation)

`total_mutations` is stored so that a cell's burden is a field read, not a walk to the
root. A hand-built tree must keep it consistent: a root has `total_mutations == mutations`,
and every child adds its own `mutations` to its parent's total. To change a cell's
fitness from a hook, use [`set_fitness!`](@ref).
"""
struct NonMarkovCell
    id::Int64
    birthtime::Float64
    mutations::Int64
    total_mutations::Int64
    fitness::Float64
end

"""
    set_fitness!(node::BinaryNode{NonMarkovCell}, fitness) -> node

Replace `node`'s cell with an identical one of the given fitness. This is the supported
way for an `on_division` hook to change a daughter: it keeps `id`, `birthtime` and both
mutation counts intact. The new fitness is inherited by all later descendants.
"""
function set_fitness!(node::BinaryNode{NonMarkovCell}, fitness::Real)
    c = node.data
    node.data = NonMarkovCell(c.id, c.birthtime, c.mutations, c.total_mutations, fitness)
    return node
end

# ── CellEvent ──────────────────────────────────────────────────────────────────

# An entry in the global min-heap of pending events: the absolute time at which `node`
# divides (`is_division`) or dies. Defined here because `Population` holds the heap.
struct CellEvent
    time::Float64
    node::BinaryNode{NonMarkovCell}
    is_division::Bool
end

Base.isless(a::CellEvent, b::CellEvent) = a.time < b.time

# ── Population ─────────────────────────────────────────────────────────────────

"""
    Population

The living cells, keyed by id, and the current simulation time `t` (the time of the most
recently processed event). The lineage tree is reachable from every cell through
`parent` links. Build one with [`initialize_population`](@ref).

Treat `cells` as read-only: `simulate!` also carries a private queue of already-drawn
events (one per cell, see [`has_pending_schedule`](@ref)), and adding or removing cells
by hand makes it inconsistent until [`reset_schedule!`](@ref) is called.
"""
mutable struct Population
    cells::Dict{Int64, BinaryNode{NonMarkovCell}}
    t::Float64
    _next_id::Int64                                     # id of the newest cell
    _pending::Union{Nothing, BinaryMinHeap{CellEvent}}  # carried event queue
end

"""
    has_pending_schedule(population) -> Bool

Whether `population` carries a queue of already-drawn events, which the next `simulate!`
call will reuse. False for a fresh population and after [`reset_schedule!`](@ref).
"""
has_pending_schedule(population::Population) = !isnothing(population._pending)

"""
    popsize(population) -> Int
    popsize(root::BinaryNode) -> Int

Number of living cells: in the population, or among the leaves under `root`.
"""
popsize(population::Population) = length(population.cells)
popsize(root::BinaryNode) = length(_leaves(root))

Base.show(io::IO, pop::Population) =
    print(io, "Population: $(popsize(pop)) cells (t = $(round(pop.t, digits = 3)))")

# The cells in increasing id order: the one stable order of a population's cells.
_sorted_cells(population::Population) =
    [population.cells[id] for id in sort!(collect(keys(population.cells)))]
