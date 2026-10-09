# Tree utilities: roots, MRCAs, division times and lifetimes.

# ── Roots ─────────────────────────────────────────────────────────────────────

"""
    roots(population) -> Vector{CellNode}
    roots(tree::LineageTree) -> Vector{CellNode}
    roots(nodes::Vector{CellNode}) -> Vector{CellNode}

The distinct roots of the trees the population's cells (or the tree's nodes, or the
given nodes) belong to, in id order. A population from `initialize_population(N)` with
`N > 1` is a forest of `N` trees; a founder whose lineage died out is no longer a root.
"""
roots(population::Population) = roots(population.tree)
roots(tree::LineageTree) = CellNode[CellNode(tree, i) for i in _rootidxs(tree)]
function roots(nodes::AbstractVector{CellNode})
    found = Set{CellNode}()
    for n in nodes
        tree = n.tree
        push!(found, _node(tree, _rootidx(tree, _index(n))))
    end
    return sort!(collect(found); by = r -> r.id)
end

"""
    single_root(population) -> Union{CellNode, Nothing}
    single_root(tree::LineageTree) -> Union{CellNode, Nothing}
    single_root(nodes::Vector{CellNode}) -> Union{CellNode, Nothing}

The unique root of the population's (the tree's, the nodes') lineage tree, or `nothing`
if they form a forest or are empty.
"""
single_root(population::Population) = single_root(population.tree)
function single_root(tree::LineageTree)
    r = _rootidxs(tree)
    return length(r) == 1 ? CellNode(tree, only(r)) : nothing
end
function single_root(nodes::AbstractVector{CellNode})
    found = roots(nodes)
    return length(found) == 1 ? only(found) : nothing
end

# ── MRCA ──────────────────────────────────────────────────────────────────────

# MRCA of nodes `a` and `b` of one tree, by index (indices increase along lineages), or 0.
@inline function _mrca_idx(tree::LineageTree, a::Int32, b::Int32)
    parent = tree.parent
    @inbounds while a != b
        a > b && ((a, b) = (b, a))
        # b has the larger index, so it cannot be an ancestor of a: climb it.
        b = parent[b]
        b <= NOPARENT && return Int32(0)
    end
    return a
end

"""
    find_mrca(node1, node2)
    find_mrca(nodes::Vector{CellNode})
    find_mrca(population)

The most recent common ancestor, or `nothing` if the nodes lie in different trees.
"""
function find_mrca(node1::CellNode, node2::CellNode)
    node1.tree === node2.tree || return nothing
    m = _mrca_idx(node1.tree, _index(node1), _index(node2))
    return _node(node1.tree, m)
end

function find_mrca(nodes::AbstractVector{CellNode})
    isempty(nodes) && return nothing
    tree = first(nodes).tree
    m = _index(first(nodes))
    for node in Iterators.drop(nodes, 1)
        node.tree === tree || return nothing
        m = _mrca_idx(tree, m, _index(node))
        m == 0 && return nothing   # forest: no shared ancestor
    end
    return _node(tree, m)
end

function find_mrca(population::Population)
    # On a single tree the MRCA of all leaves is the first node below the root with two
    # children (or the only leaf): everything above it is a chain of unary nodes.
    tree = population.tree
    r = _rootidxs(tree)
    length(r) == 1 || return nothing
    i = only(r)
    @inbounds while true
        l, rr = tree.left[i], tree.right[i]
        if l == NOPARENT && rr != NOPARENT
            i = rr
        elseif rr == NOPARENT && l != NOPARENT
            i = l
        else
            return CellNode(tree, i)
        end
    end
end

# ── Times ─────────────────────────────────────────────────────────────────────

@inline function _division_time(tree::LineageTree, i::Integer)
    @inbounds l, r = tree.left[i], tree.right[i]
    l == NOPARENT || return @inbounds tree.birthtime[l]
    r == NOPARENT || return @inbounds tree.birthtime[r]
    return NaN
end

"""
    division_time(node::CellNode) -> Union{Float64, Nothing}

The time at which the cell divided (its daughters' birthtime), or `nothing` if it is
still alive (a leaf).
"""
function division_time(node::CellNode)
    t = _division_time(node.tree, _index(node))
    return isnan(t) ? nothing : t
end

"""
    last_division_time(root::CellNode) -> Float64

The birthtime of the most recently born leaf under `root`, i.e. the time of the last
division in that tree. This is earlier than the population's clock `pop.t` if the run
ended on a death.
"""
function last_division_time(root::CellNode)
    tree = root.tree
    return maximum(i -> tree.birthtime[i], _leaves_idx(tree, _index(root)))
end

"""
    cell_lifetime(node::CellNode, tnow) -> Float64

Time from the cell's birth to its division, or to `tnow` if it is still alive. Pass the
population's clock `pop.t` as `tnow` for "how old is this cell right now".
"""
function cell_lifetime(node::CellNode, tnow::Real)
    t_end = division_time(node)
    return (isnothing(t_end) ? tnow : t_end) - node.birthtime
end

"""
    cell_lifetimes(root; include_alive = false, tnow = last_division_time(root))

The lifetime of every cell in the tree under `root`, in pre-order. By default only cells
that have divided contribute. With `include_alive = true`, living cells contribute their
age at `tnow`, which is a right-censored lifetime rather than a completed one.
"""
function cell_lifetimes(root::CellNode; include_alive::Bool = false,
                        tnow::Real = last_division_time(root))
    tree = root.tree
    nodes, _ = _preorder_idx(tree, _index(root))
    out = Float64[]
    for i in nodes
        td = _division_time(tree, i)
        if !isnan(td)
            push!(out, td - tree.birthtime[i])
        elseif include_alive
            push!(out, tnow - tree.birthtime[i])
        end
    end
    return out
end
