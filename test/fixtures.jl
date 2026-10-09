# Shared test fixtures. `runtests.jl` includes this first; each test file also includes
# it when run on its own (guarded by `@isdefined`).

# Hand-built cells must keep `total_mutations` consistent with the lineage: a root's
# total is its own count, and a child adds its own count to its parent's total.
rootnode(id, t, m, f = 1.0) = BinaryNode(NonMarkovCell(id, t, m, m, f))

function child!(side::Symbol, parent::BinaryNode, id, t, m, f = 1.0)
    cell = NonMarkovCell(id, t, m, parent.data.total_mutations + m, f)
    return side === :left ? left_child!(parent, cell) : right_child!(parent, cell)
end

# Hand-built tree with observables known by hand:
#
#   root (id 1, t 0.0, mutations 5) ├── L (id 2, t 1.0, mutations 1) ├── LL (id 4, t 2.0, mutations 2)
#                                   │                                └── LR (id 5, t 2.1, mutations 3)
#                                   └── R (id 3, t 1.2, mutations 7)
function fixture_tree()
    root = rootnode(1, 0.0, 5)
    L = child!(:left, root, 2, 1.0, 1)
    child!(:right, root, 3, 1.2, 7)
    child!(:left,  L, 4, 2.0, 2)
    child!(:right, L, 5, 2.1, 3)
    return root
end

# A small growing block with mutations, used by several test files.
function grow_block(; Nmax = 50, ν = 1.0, s_mean = 0.1, restart = false,
                    death = f -> Gamma(2.0, 20.0))
    NonMarkovBlock(
        birth_dist     = f -> Gamma(2.0, 1.0 / (2.0 * f)),
        death_dist     = death,
        stopfunction   = pop -> popsize(pop) >= Nmax,
        effect_dist    = Exponential(s_mean),
        fitness_update = (f, δ) -> f + δ,
        ν              = ν,
        restart_on_extinction = restart,
    )
end

function grown_population(; Nmax = 50, ν = 1.0, rng = MersenneTwister(1))
    pop = initialize_population(fitness_init = 1.0)
    simulate!(pop, grow_block(Nmax = Nmax, ν = ν), rng)
    return pop
end

# Burden and depth of every leaf, keyed by cell id, computed by walking the tree
# independently of the package's own functions.
function id_burden_map(root::BinaryNode{NonMarkovCell})
    m = Dict{Int64, Int}()
    for leaf in Leaves(root)
        muts, node = leaf.data.mutations, leaf
        while !isnothing(node.parent)
            node = node.parent
            muts += node.data.mutations
        end
        m[leaf.data.id] = muts
    end
    return m
end

function id_depth_map(root::BinaryNode{NonMarkovCell})
    m = Dict{Int64, Int}()
    for leaf in Leaves(root)
        d, node = 0, leaf
        while !isnothing(node.parent)
            node = node.parent
            d += 1
        end
        m[leaf.data.id] = d
    end
    return m
end
