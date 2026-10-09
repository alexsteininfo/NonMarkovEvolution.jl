# Division and death on the tree arrays.

"""
    _divide!(population, p, t, block, mutation_count, rng) -> (d1, d2)

Cell `p` divides at time `t`: append its two daughters (left, then right) to the tree
and return their indices. Each daughter independently draws `j ~ mutation_count`
mutations (`Poisson(block.ν)`, built once per `simulate!` call); for each,
`δ ~ block.effect_dist` and the fitness is updated with `block.fitness_update`. The first
daughter draws all of hers before the second.
"""
@inline function _divide!(pop::Population, p::Int32, t::Float64, block::NonMarkovBlock,
                          mutation_count::Poisson, rng::AbstractRNG)
    tree = pop.tree
    f0, total0 = @inbounds tree.fitness[p], tree.total[p]
    j1, f1 = _draw_mutations(f0, block, mutation_count, rng)
    j2, f2 = _draw_mutations(f0, block, mutation_count, rng)
    id1 = pop._next_id + 1
    pop._next_id += 2
    d1 = _push_node!(tree, p, id1, t, j1, total0 + j1, f1)
    d2 = _push_node!(tree, p, id1 + 1, t, j2, total0 + j2, f2)
    @inbounds tree.left[p]  = d1
    @inbounds tree.right[p] = d2
    pop._nalive += 1
    return d1, d2
end

@inline function _draw_mutations(f::Float64, block::NonMarkovBlock, mutation_count::Poisson,
                                 rng::AbstractRNG)
    j = rand(rng, mutation_count)
    for _ in 1:j
        f = block.fitness_update(f, rand(rng, block.effect_dist))
    end
    return j, Float64(f)
end

"""
    _die!(population, i)

Cell `i` dies: remove it and every ancestor left without living descendants.
"""
@inline function _die!(pop::Population, i::Int32)
    _prune!(pop.tree, i)
    pop._nalive -= 1
    return nothing
end
