# The lineage tree as a struct of arrays, the `CellNode` handles users see, and the
# index-based traversals, pruning and compaction every other file builds on.
#
# Node `i` of a `LineageTree` is row `i` of every column. Nodes are appended in creation
# order, so a child always has a larger index (and a larger id) than its parent: a
# reverse sweep over the indices visits children before parents. Removing a node only
# marks it (`parent[i] == REMOVED`); `_compact!` drops marked rows later, in one pass,
# keeping the order of the rest.

const NOPARENT = Int32(0)    # `parent` of a root; `left`/`right` of a missing child
const REMOVED  = Int32(-1)   # `parent` of a pruned node

"""
    LineageTree()

The nodes of one or more lineage trees, stored column-wise: node `i` has
`parent[i]`, `left[i]`, `right[i]` (indices, `0` for none), and the cell's `id[i]`,
`birthtime[i]`, `mutations[i]` (acquired at its birth), `total[i]` (on the whole path
from its root, including its own) and `fitness[i]`.

Ids strictly increase with the index, and every child comes after its parent. Read
nodes through [`CellNode`](@ref) handles rather than by index: indices change when
pruned nodes are compacted away, ids never do. A [`Population`](@ref) holds one in
`pop.tree`; [`load_tree`](@ref) returns one.
"""
mutable struct LineageTree
    parent::Vector{Int32}
    left::Vector{Int32}
    right::Vector{Int32}
    id::Vector{Int64}
    birthtime::Vector{Float64}
    fitness::Vector{Float64}
    mutations::Vector{Int32}
    total::Vector{Int32}
    ndead::Int                  # rows marked REMOVED and not yet compacted away
end

LineageTree() = LineageTree(Int32[], Int32[], Int32[], Int64[], Float64[], Float64[],
                            Int32[], Int32[], 0)

Base.length(tree::LineageTree) = length(tree.id)

@inline _isdead(tree::LineageTree, i::Integer) = @inbounds tree.parent[i] == REMOVED
@inline _isleaf(tree::LineageTree, i::Integer) =
    @inbounds tree.left[i] == NOPARENT && tree.right[i] == NOPARENT
@inline _haschildren(tree::LineageTree, i::Integer) = !_isleaf(tree, i)
# A living cell: a leaf that has not been removed.
@inline _isalive(tree::LineageTree, i::Integer) = _isleaf(tree, i) && !_isdead(tree, i)

# Append a node and return its index; links nothing (the caller sets the parent's slot).
function _push_node!(tree::LineageTree, parent::Int32, id::Int64, t::Float64,
                     m::Integer, total::Integer, f::Float64)
    length(tree) < typemax(Int32) || error(
        "LineageTree: more than $(typemax(Int32)) nodes; the tree cannot index them")
    push!(tree.parent, parent)
    push!(tree.left, NOPARENT)
    push!(tree.right, NOPARENT)
    push!(tree.id, id)
    push!(tree.birthtime, t)
    push!(tree.fitness, f)
    push!(tree.mutations, Int32(m))
    push!(tree.total, Int32(total))
    return Int32(length(tree))
end

function Base.sizehint!(tree::LineageTree, n::Integer)
    for v in (tree.parent, tree.left, tree.right, tree.id, tree.birthtime, tree.fitness,
              tree.mutations, tree.total)
        sizehint!(v, n)
    end
    return tree
end

# ── NonMarkovCell: the value of one node ──────────────────────────────────────

"""
    NonMarkovCell(id, birthtime, mutations, total_mutations, fitness)

The data of one node, as a value: what `node.data` returns, and what
[`add_root!`](@ref), [`left_child!`](@ref) and [`right_child!`](@ref) take.

# Fields
- `id::Int64` — unique cell identifier; always larger than the parent's id
- `birthtime::Float64` — simulation time at which the cell was born
- `mutations::Int64` — mutations acquired at this cell's own birth
- `total_mutations::Int64` — mutations on the whole path from the root to this cell,
  including its own: the parent's `total_mutations + mutations`
- `fitness::Float64` — cumulative fitness (parent fitness updated once per mutation)

A hand-built tree must keep `total_mutations` consistent: a root has
`total_mutations == mutations`, and every child adds its own `mutations` to its
parent's total.
"""
struct NonMarkovCell
    id::Int64
    birthtime::Float64
    mutations::Int64
    total_mutations::Int64
    fitness::Float64
end

# ── CellNode: a handle on one node ────────────────────────────────────────────

"""
    CellNode

A handle on one node of a [`LineageTree`](@ref): the tree, the cell's id, and the
node's index at the time the handle was made. Read the cell through properties:

| property | value |
|:---|:---|
| `node.id`, `node.birthtime`, `node.fitness` | the cell's id, birth time, fitness |
| `node.mutations`, `node.total_mutations` | its own and its root-to-cell mutation count |
| `node.parent`, `node.left`, `node.right` | `CellNode` or `nothing` |
| `node.data` | all of the above as a [`NonMarkovCell`](@ref) |

Handles stay valid while the tree changes, also across compaction (the index is
looked up again by id when it has moved). Do not read a handle on a cell that has died
and been pruned: it may still return the old values or throw, depending on whether the
row has been compacted away yet; check with [`isalive`](@ref) first. Handles taken
during an attempt that ends in an extinction restart are invalid as well: the restart
reuses those ids for new cells, so drop them in `on_restart`. Two handles are `==` when
they name the same cell of the same tree. The AbstractTrees interface is implemented, so `Leaves(node)`,
`PreOrderDFS(node)` and `print_tree(node)` work; the package's own statistics do not
need them and are much faster.
"""
struct CellNode
    tree::LineageTree
    idx::Int32         # index hint; re-resolved by id if compaction moved the node
    id::Int64
end

CellNode(tree::LineageTree, i::Integer) = CellNode(tree, Int32(i), tree.id[i])

# Current index of `node`, or an error if its cell has been removed from the tree.
@inline function _index(node::CellNode)
    tree, i = getfield(node, :tree), getfield(node, :idx)
    id = getfield(node, :id)
    @inbounds if 1 <= i <= length(tree) && tree.id[i] == id
        return i
    end
    return _resolve(tree, id)
end

@noinline function _resolve(tree::LineageTree, id::Int64)
    j = searchsortedfirst(tree.id, id)
    (j <= length(tree) && tree.id[j] == id) || throw(ArgumentError(
        "cell $id is no longer in the tree: it died and its lineage was removed"))
    return Int32(j)
end

_node(tree::LineageTree, i::Integer) =
    i == NOPARENT ? nothing : CellNode(tree, Int32(i), @inbounds tree.id[i])

function Base.getproperty(node::CellNode, s::Symbol)
    s === :id && return getfield(node, :id)
    s === :tree && return getfield(node, :tree)
    s === :idx && return getfield(node, :idx)
    tree = getfield(node, :tree)
    i = _index(node)
    s === :fitness && return tree.fitness[i]
    s === :birthtime && return tree.birthtime[i]
    s === :mutations && return Int64(tree.mutations[i])
    s === :total_mutations && return Int64(tree.total[i])
    s === :parent && return _isdead(tree, i) ? nothing : _node(tree, tree.parent[i])
    s === :left && return _node(tree, tree.left[i])
    s === :right && return _node(tree, tree.right[i])
    s === :data && return _cell(tree, i)
    throw(ArgumentError("CellNode has no property $s"))
end

Base.propertynames(::CellNode) = (:id, :birthtime, :mutations, :total_mutations, :fitness,
                                  :parent, :left, :right, :data, :tree, :idx)

_cell(tree::LineageTree, i::Integer) =
    NonMarkovCell(tree.id[i], tree.birthtime[i], tree.mutations[i], tree.total[i],
                  tree.fitness[i])

Base.:(==)(a::CellNode, b::CellNode) =
    getfield(a, :tree) === getfield(b, :tree) && getfield(a, :id) == getfield(b, :id)
Base.hash(n::CellNode, h::UInt) =
    hash(getfield(n, :id), hash(objectid(getfield(n, :tree)), h))

Base.show(io::IO, node::CellNode) = show(io, node.data)

isleaf(node::CellNode) = _isleaf(node.tree, _index(node))
haschildren(node::CellNode) = !isleaf(node)

"""
    isalive(node::CellNode) -> Bool

Whether the node is a living cell: a leaf that has not died. Every leaf of a pruned tree
is alive; in a sampled or loaded tree every leaf counts as alive.
"""
function isalive(node::CellNode)
    tree = node.tree
    j = searchsortedfirst(tree.id, node.id)
    (j <= length(tree) && tree.id[j] == node.id) || return false
    return _isalive(tree, j)
end

# ── Building trees by hand ────────────────────────────────────────────────────

function _check_new_id(tree::LineageTree, id::Integer)
    isempty(tree.id) || id > last(tree.id) || throw(ArgumentError(
        "cell ids must increase with every node added to a tree: got $id after " *
        "$(last(tree.id))"))
    return Int64(id)
end

"""
    add_root!(tree::LineageTree, cell::NonMarkovCell) -> CellNode

Add a root holding `cell` to `tree`. Its id must exceed every id already in the tree.
"""
function add_root!(tree::LineageTree, cell::NonMarkovCell)
    id = _check_new_id(tree, cell.id)
    i = _push_node!(tree, NOPARENT, id, Float64(cell.birthtime), cell.mutations,
                    cell.total_mutations, Float64(cell.fitness))
    return CellNode(tree, i, id)
end

"""
    CellNode(cell::NonMarkovCell) -> CellNode

A new single-node tree holding `cell`, returned as its root.
"""
CellNode(cell::NonMarkovCell) = add_root!(LineageTree(), cell)

function _add_child!(parent::CellNode, cell::NonMarkovCell, side::Symbol)
    tree = parent.tree
    p = _index(parent)
    slot = side === :left ? tree.left : tree.right
    slot[p] == NOPARENT || error("$side child is already assigned")
    id = _check_new_id(tree, cell.id)
    i = _push_node!(tree, p, id, Float64(cell.birthtime), cell.mutations,
                    cell.total_mutations, Float64(cell.fitness))
    slot[p] = i
    return CellNode(tree, i, id)
end

"""
    left_child!(parent::CellNode, cell::NonMarkovCell) -> CellNode

Add `cell` as `parent`'s left child. Its id must exceed every id already in the tree.
"""
left_child!(parent::CellNode, cell::NonMarkovCell) = _add_child!(parent, cell, :left)

"""
    right_child!(parent::CellNode, cell::NonMarkovCell) -> CellNode

Add `cell` as `parent`'s right child. Its id must exceed every id already in the tree.
"""
right_child!(parent::CellNode, cell::NonMarkovCell) = _add_child!(parent, cell, :right)

"""
    set_fitness!(node::CellNode, fitness) -> node

Change the cell's fitness, keeping its id, birthtime and mutation counts. The supported
way for an `on_division` hook to change a daughter: the new fitness then sets the
daughter's own waiting times and is inherited by her descendants.
"""
function set_fitness!(node::CellNode, fitness::Real)
    node.tree.fitness[_index(node)] = Float64(fitness)
    return node
end

# ── AbstractTrees interface ───────────────────────────────────────────────────

function AbstractTrees.children(node::CellNode)
    tree = node.tree
    i = _index(node)
    l, r = tree.left[i], tree.right[i]
    l == NOPARENT && return r == NOPARENT ? () : (_node(tree, r),)
    r == NOPARENT && return (_node(tree, l),)
    return (_node(tree, l), _node(tree, r))
end
AbstractTrees.parent(n::CellNode) = n.parent
AbstractTrees.ParentLinks(::Type{CellNode}) = StoredParents()
AbstractTrees.NodeType(::Type{CellNode}) = HasNodeType()
AbstractTrees.nodetype(::Type{CellNode}) = CellNode
AbstractTrees.nodevalue(n::CellNode) = n.data
AbstractTrees.printnode(io::IO, n::CellNode) = print(io, n.data)
function AbstractTrees.nextsibling(child::CellNode)
    p = child.parent
    (isnothing(p) || child == p.right) && return nothing
    return p.right
end
function AbstractTrees.prevsibling(child::CellNode)
    p = child.parent
    (isnothing(p) || child == p.left) && return nothing
    return p.left
end
Base.eltype(::Type{<:TreeIterator{CellNode}}) = CellNode
Base.IteratorEltype(::Type{<:TreeIterator{CellNode}}) = Base.HasEltype()

# ── Index traversals ──────────────────────────────────────────────────────────

"""
    _leaves_idx(tree, r) -> Vector{Int32}

The leaves under node `r`, in `AbstractTrees.Leaves` order (left before right at every
node). Sampling draws index into this order, so it must never change.
"""
function _leaves_idx(tree::LineageTree, r::Integer)
    leaves = Int32[]
    stack  = Int32[r]
    left, right = tree.left, tree.right
    @inbounds while !isempty(stack)
        i = pop!(stack)
        l, rr = left[i], right[i]
        if l == NOPARENT && rr == NOPARENT
            push!(leaves, i)
        else
            rr == NOPARENT || push!(stack, rr)
            l  == NOPARENT || push!(stack, l)
        end
    end
    return leaves
end

"""
    _preorder_idx(tree, r) -> (nodes, parentpos)

Every node under `r` in pre-order (left before right), and `parentpos[k]`, the position
of `nodes[k]`'s parent in `nodes` (`0` for `r`). Parents come before children, so a
reverse sweep is a post-order pass.
"""
function _preorder_idx(tree::LineageTree, r::Integer)
    nodes  = Int32[]
    ppos   = Int32[]
    stack  = Int32[r]
    pstack = Int32[0]
    left, right = tree.left, tree.right
    @inbounds while !isempty(stack)
        i = pop!(stack)
        push!(nodes, i)
        push!(ppos, pop!(pstack))
        k = Int32(length(nodes))
        if right[i] != NOPARENT
            push!(stack, right[i]); push!(pstack, k)
        end
        if left[i] != NOPARENT
            push!(stack, left[i]); push!(pstack, k)
        end
    end
    return nodes, ppos
end

# Number of leaves below each entry of a `_preorder_idx` listing.
function _leafcounts(tree::LineageTree, nodes::Vector{Int32}, ppos::Vector{Int32})
    counts = zeros(Int, length(nodes))
    @inbounds for k in length(nodes):-1:1
        _isleaf(tree, nodes[k]) && (counts[k] = 1)
        ppos[k] == 0 || (counts[ppos[k]] += counts[k])
    end
    return counts
end

# Root of the tree that node `i` belongs to.
function _rootidx(tree::LineageTree, i::Integer)
    i = Int32(i)
    @inbounds while tree.parent[i] > NOPARENT
        i = tree.parent[i]
    end
    return i
end

# Indices of the live roots, in index order.
_rootidxs(tree::LineageTree) =
    Int32[i for i in eachindex(tree.parent) if @inbounds tree.parent[i] == NOPARENT]

# Indices of the living cells (live leaves), in index (= id) order.
function _alive_idx(tree::LineageTree)
    out = Int32[]
    @inbounds for i in eachindex(tree.parent)
        _isalive(tree, i) && push!(out, Int32(i))
    end
    return out
end

# ── Pruning and compaction ────────────────────────────────────────────────────

"""
    _prune!(tree, i)

Remove the dead cell `i` from the tree, and then every ancestor left without children
(a root too). Removed rows are only marked; `_compact!` drops them.
"""
function _prune!(tree::LineageTree, i::Int32)
    parent, left, right = tree.parent, tree.left, tree.right
    @inbounds while true
        p = parent[i]
        parent[i] = REMOVED
        tree.ndead += 1
        p == NOPARENT && return nothing
        if left[p] == i
            left[p] = NOPARENT
        elseif right[p] == i
            right[p] = NOPARENT
        else
            error("dead cell is neither left nor right child of its parent")
        end
        (left[p] != NOPARENT || right[p] != NOPARENT) && return nothing
        i = p
    end
end

"""
    _compact!(tree, start) -> remap::Vector{Int32}

Drop the removed rows at indices `≥ start`, keeping the order of the rest, and rewrite
every link. Returns `remap`, where `remap[i - start + 1]` is the new index of old row
`i ≥ start` (`0` if it was dropped); rows before `start` keep their index. Callers holding
indices at or after `start` must translate them with it.
"""
function _compact!(tree::LineageTree, start::Int32)
    n = length(tree)
    remap = zeros(Int32, max(n - start + 1, 0))
    k = start - Int32(1)
    @inbounds for i in start:n
        _isdead(tree, i) && continue
        k += Int32(1)
        remap[i - start + 1] = k
    end
    newindex(j) = j < start ? j : (@inbounds remap[j - start + 1])
    parent, left, right = tree.parent, tree.left, tree.right
    @inbounds for i in 1:(start - 1)       # rows before `start` may link into the moved part
        left[i]  >= start && (left[i]  = newindex(left[i]))
        right[i] >= start && (right[i] = newindex(right[i]))
    end
    removed = 0
    @inbounds for i in start:n
        if _isdead(tree, i)
            removed += 1
            continue
        end
        j = remap[i - start + 1]
        p = parent[i]
        parent[j] = p > NOPARENT ? newindex(p) : p
        left[j]   = left[i]  == NOPARENT ? NOPARENT : newindex(left[i])
        right[j]  = right[i] == NOPARENT ? NOPARENT : newindex(right[i])
        tree.id[j]        = tree.id[i]
        tree.birthtime[j] = tree.birthtime[i]
        tree.fitness[j]   = tree.fitness[i]
        tree.mutations[j] = tree.mutations[i]
        tree.total[j]     = tree.total[i]
    end
    newlen = Int(k)
    for v in (parent, left, right, tree.id, tree.birthtime, tree.fitness, tree.mutations,
              tree.total)
        resize!(v, newlen)
    end
    tree.ndead -= removed
    return remap
end
