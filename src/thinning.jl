# The exponential fast path: thinning (uniformisation) of the Markov birth–death process.
#
# With exponential waiting times a cell's next event does not depend on its age, so no
# per-cell clock is needed. The whole population is driven by one exponential clock at
# rate R·N, where R bounds every cell's total rate b + d. At each tick one cell is chosen
# uniformly; it divides with probability b/R, dies with probability d/R, and otherwise
# the tick is discarded. This is exact for any R ≥ max(b + d), and costs O(1) per tick:
# the living cells sit in a vector (swap-remove on death), with their rates beside them.

# Rate of an exponential waiting time; `Dirac(Inf)` is "never".
@inline _rate(d::Exponential) = inv(d.θ)
@inline function _rate(d::Dirac)
    isinf(d.value) || throw(ArgumentError(
        "algorithm = :thinning: a Dirac waiting time must be Dirac(Inf), got $d"))
    return 0.0
end
_rate(d) = throw(ArgumentError(
    "algorithm = :thinning needs Exponential waiting times (or Dirac(Inf) for death), " *
    "got $(typeof(d))"))

# The living cells and their division and death rates, co-indexed.
struct _AliveRates
    node::Vector{Int32}
    b::Vector{Float64}
    d::Vector{Float64}
end

function _AliveRates(pop::Population, block::NonMarkovBlock)
    tree = pop.tree
    idx  = sizehint!(_alive_idx(tree), pop._queue_hint)
    b = Float64[_rate(block.birth_dist(tree.fitness[i])) for i in idx]
    d = Float64[_rate(block.death_dist(tree.fitness[i])) for i in idx]
    return _AliveRates(idx, b, d)
end

function _run_thinning!(pop::Population, block::NonMarkovBlock, rng::AbstractRNG, acc)
    alive = _AliveRates(pop, block)
    isnothing(acc) || _check_popsize_triggers!(acc, pop)
    _thinning_loop!(pop, alive, block, Poisson(block.ν), rng, acc)
    return nothing
end

function _thinning_loop!(pop::Population, alive::_AliveRates, block::NonMarkovBlock,
                         mutation_count::Poisson, rng::AbstractRNG, acc)
    tree = pop.tree
    nodes, rb, rd = alive.node, alive.b, alive.d
    R = 0.0
    @inbounds for k in eachindex(rb)
        R = max(R, rb[k] + rd[k])
    end
    remap_alive! = function (remap, start)
        @inbounds for k in eachindex(nodes)
            i = nodes[k]
            i >= start && (nodes[k] = remap[i - start + 1])
        end
        return nothing
    end
    t = pop.t
    while !block.stopfunction(pop)
        N = length(nodes)
        N == 0 && break
        if R == 0.0                       # nothing can ever happen again
            isfinite(block.tmax) && (pop.t = max(pop.t, block.tmax))
            break
        end
        tnext = t + randexp(rng) / (R * N)
        if tnext > block.tmax
            pop.t = max(pop.t, block.tmax)
            break
        end
        t = tnext
        k = rand(rng, 1:N)
        u = rand(rng) * R
        @inbounds if u < rb[k]
            isnothing(acc) || _record_until!(acc, pop, t)
            pop.t = t
            p = nodes[k]
            d1, d2 = _divide!(pop, p, t, block, mutation_count, rng)
            isnothing(block.on_division) ||
                block.on_division(pop, CellNode(tree, p), CellNode(tree, d1),
                                  CellNode(tree, d2))
            b1, δ1 = _rate(block.birth_dist(tree.fitness[d1])), _rate(block.death_dist(tree.fitness[d1]))
            b2, δ2 = _rate(block.birth_dist(tree.fitness[d2])), _rate(block.death_dist(tree.fitness[d2]))
            nodes[k], rb[k], rd[k] = d1, b1, δ1
            push!(nodes, d2); push!(rb, b2); push!(rd, δ2)
            R = max(R, b1 + δ1, b2 + δ2)
        elseif u < rb[k] + rd[k]
            isnothing(acc) || _record_until!(acc, pop, t)
            pop.t = t
            _die!(pop, nodes[k])
            nodes[k], rb[k], rd[k] = nodes[N], rb[N], rd[N]
            pop!(nodes); pop!(rb); pop!(rd)
            _maybe_compact!(pop, remap_alive!)
        else
            continue                      # rejected tick: the state does not change
        end
        isnothing(acc) || _check_popsize_triggers!(acc, pop)
    end
    return nothing
end
