# Saving and loading lineage trees in a small, stable binary format.
#
# The tree is already a struct of arrays, so a file is a header followed by the columns,
# written as raw little-endian arrays. Removed rows are left out. The format does not
# depend on Julia's Serialization (which is neither stable across Julia versions nor
# independent of the package's type names), and reads and writes at disk speed.
#
# Layout, version 1:
#   magic       8 bytes  "NMEtree\0"
#   version     UInt32   1
#   n           Int64    number of nodes
#   t           Float64  population clock (NaN when a subtree was saved)
#   parent      n × Int32   (0 = root)
#   left        n × Int32
#   right       n × Int32
#   id          n × Int64
#   birthtime   n × Float64
#   fitness     n × Float64
#   mutations   n × Int32
#   total       n × Int32

const _MAGIC = UInt8['N', 'M', 'E', 't', 'r', 'e', 'e', '\0']
const _FORMAT_VERSION = UInt32(1)

"""
    save_tree(path, population)
    save_tree(path, root::CellNode)
    save_tree(path, tree::LineageTree)

Write the population's lineage tree (every root of a forest, and the clock `pop.t`), or
the subtree under `root`, to `path` in the package's binary format. Removed lineages are
left out. About 44 bytes per node, i.e. about 90 bytes per living cell of a grown
population; read it back with [`load_tree`](@ref). A saved subtree's root keeps its
`total_mutations` (the mutations above it count as clonal), so
`mutations_per_cell(root; includeclonal = true)` still gives full-tree burdens.
"""
function save_tree(path::AbstractString, population::Population)
    tree = population.tree
    kept = Int32[i for i in eachindex(tree.parent) if !_isdead(tree, i)]
    _write_tree(path, _copy_rows_forest(tree, kept), population.t)
end

function save_tree(path::AbstractString, tree::LineageTree)
    kept = Int32[i for i in eachindex(tree.parent) if !_isdead(tree, i)]
    _write_tree(path, _copy_rows_forest(tree, kept), NaN)
end

function save_tree(path::AbstractString, root::CellNode)
    tree = root.tree
    nodes, _ = _preorder_idx(tree, _index(root))
    _write_tree(path, _copy_rows(tree, sort!(nodes)), NaN)
end

# Like `_copy_rows`, for a set of rows that may hold several roots.
function _copy_rows_forest(tree::LineageTree, kept::Vector{Int32})
    new = LineageTree()
    sizehint!(new, length(kept))
    newindex(i) = Int32(searchsortedfirst(kept, i))
    for (k, i) in enumerate(kept)
        p = tree.parent[i] == NOPARENT ? NOPARENT : newindex(tree.parent[i])
        _push_node!(new, p, tree.id[i], tree.birthtime[i], tree.mutations[i],
                    tree.total[i], tree.fitness[i])
        p == NOPARENT && continue
        if tree.left[tree.parent[i]] == i
            new.left[p] = Int32(k)
        else
            new.right[p] = Int32(k)
        end
    end
    return new
end

function _write_tree(path::AbstractString, tree::LineageTree, t::Float64)
    open(path, "w") do io
        write(io, _MAGIC)
        write(io, htol(_FORMAT_VERSION))
        write(io, htol(Int64(length(tree))))
        write(io, htol(t))
        for v in (tree.parent, tree.left, tree.right, tree.id, tree.birthtime,
                  tree.fitness, tree.mutations, tree.total)
            write(io, htol.(v))
        end
    end
    return path
end

"""
    load_tree(path) -> (tree::LineageTree, t::Float64)

Read a file written by [`save_tree`](@ref). `t` is the population clock at saving
(`NaN` for a saved subtree). Get the roots with `roots(tree)` or `single_root(tree)`;
every statistic that takes a root works on them.
"""
function load_tree(path::AbstractString)
    open(path, "r") do io
        magic = read(io, length(_MAGIC))
        magic == _MAGIC || throw(ArgumentError("$path is not a NonMarkovEvolution tree file"))
        version = ltoh(read(io, UInt32))
        version == _FORMAT_VERSION || throw(ArgumentError(
            "$path has tree format version $version; this package reads version " *
            "$_FORMAT_VERSION"))
        n = Int(ltoh(read(io, Int64)))
        t = ltoh(read(io, Float64))
        col(T) = ltoh.(read!(io, Vector{T}(undef, n)))
        parent, left, right = col(Int32), col(Int32), col(Int32)
        id, birthtime, fitness = col(Int64), col(Float64), col(Float64)
        mutations, total = col(Int32), col(Int32)
        tree = LineageTree(parent, left, right, id, birthtime, fitness, mutations, total, 0)
        return tree, t
    end
end
