"""
    simulate!(population, block, rng; accumulator = nothing) -> Population

Advance `population` under `block` until `block.stopfunction` returns `true`, the clock
reaches `block.tmax`, or the population dies out (and is not restarted). Mutates
`population` in place and returns it, so blocks can be chained.

How events are generated depends on `block.algorithm`:

- **`:queue`.** Every living cell has exactly one pending event — its division or its
  death, whichever of its two waiting times is shorter — and events fire in global time
  order (exact ties by cell id).
  - *Chaining.* The queue of already-drawn events is carried on the population and
    reused, so a chained run is identical draw for draw to the equivalent uninterrupted
    one. Carried events keep the previous block's timing; call [`reset_schedule!`](@ref)
    to redraw every cell under the new block.
  - *Fresh schedules* (a new population, after `reset_schedule!`, after an extinction
    restart, after a `:thinning` block) visit cells in id order, and draw each cell's
    next event conditioned on nothing having happened to it before the current time.
    This is exact.
- **`:thinning`** (exponential waiting times only). One exponential clock for the whole
  population at the rate bound `R·N`; at each tick a uniformly chosen cell divides with
  probability `b/R`, dies with probability `d/R`, and otherwise nothing happens. `R` is
  the largest `b + d` of any cell so far, so the bound holds as fitness grows. Exact for
  the Markov model, and memoryless, so there is no queue to carry: a carried queue is
  discarded.

Pass a [`MeasurementAccumulator`](@ref) to record trajectories and snapshots.

The rng is required: pass a seeded one (`MersenneTwister(seed)`, `Xoshiro(seed)`, or a
`StableRNG`) so the run is reproducible.
"""
function simulate!(population::Population, block::NonMarkovBlock, rng::AbstractRNG;
                   accumulator::Union{MeasurementAccumulator, Nothing} = nothing)
    popsize(population) == 0 && block.restart_on_extinction && throw(ArgumentError(
        "simulate!: the population is already extinct, so restart_on_extinction would " *
        "retry forever"))
    snapshot = block.restart_on_extinction ? _RestartPoint(population) : nothing
    try
        _simulate!(population, block, rng, accumulator, snapshot)
    finally
        population._protect = Int32(0)
        population._ndead_protected = 0
    end
    isnothing(accumulator) || _finish_call!(accumulator, population)
    return population
end

function _simulate!(population::Population, block::NonMarkovBlock, rng::AbstractRNG,
                    accumulator, snapshot)
    isnothing(accumulator) || _start_call!(accumulator, population)
    checkpoint = isnothing(accumulator) ? nothing : _checkpoint(accumulator)
    thinning = block.algorithm === :thinning
    thinning && (population._pending = nothing)

    while true
        if thinning
            _run_thinning!(population, block, rng, accumulator)
        else
            _run_queue!(population, block, rng, accumulator)
        end
        if popsize(population) == 0 && block.restart_on_extinction
            _restore!(population, snapshot)
            isnothing(accumulator) || _rollback!(accumulator, checkpoint)
            isnothing(block.on_restart) || block.on_restart(population)
        else
            break
        end
    end
    return nothing
end

# ── Queue algorithm ───────────────────────────────────────────────────────────

function _run_queue!(pop::Population, block::NonMarkovBlock, rng::AbstractRNG, acc)
    q = pop._pending
    if isnothing(q)
        q = sizehint!(EventQueue(), max(pop._queue_hint, popsize(pop)))
        tree = pop.tree
        for i in _alive_idx(tree)                   # id order
            schedule_cell!(q, tree, i, block, rng, pop.t)
        end
        pop._pending = q
    elseif length(q) != popsize(pop)
        throw(ArgumentError(
            "population's carried event queue holds $(length(q)) events but the " *
            "population has $(popsize(pop)) alive cells — cells were added or removed " *
            "outside simulate!. Call reset_schedule!(population) to discard the queue " *
            "and redraw every cell's next event from scratch."))
    end
    isnothing(acc) || _check_popsize_triggers!(acc, pop)
    _queue_loop!(pop, q, block, Poisson(block.ν), rng, acc)
    return nothing
end

_no_alive_vector(remap, start) = nothing

function _queue_loop!(pop::Population, q::EventQueue, block::NonMarkovBlock,
                      mutation_count::Poisson, rng::AbstractRNG, acc)
    tree = pop.tree
    while !block.stopfunction(pop) && !isempty(q)
        event = first(q)
        if event.time > block.tmax
            # Stop exactly at tmax; the next event stays queued for a chained call.
            pop.t = max(pop.t, block.tmax)
            break
        end
        event.time >= pop.t || _corrupt_queue(event.time, pop.t)
        # Record grid points and AtTime snapshots due before this event, while the
        # state is still exactly the state that held at those times.
        isnothing(acc) || _record_until!(acc, pop, event.time)
        pop.t = event.time

        if event.is_division
            d1, d2 = _divide!(pop, event.node, event.time, block, mutation_count, rng)
            # Must run before drawing: the daughters' waiting times use the fitness the
            # hook leaves them with.
            isnothing(block.on_division) ||
                block.on_division(pop, CellNode(tree, event.node), CellNode(tree, d1),
                                  CellNode(tree, d2))
            e1 = _draw_event(tree, d1, block, rng)
            e2 = _draw_event(tree, d2, block, rng)
            replace_top!(q, e1)          # the parent's event leaves the queue here
            push!(q, e2)
        else
            pop!(q)
            _die!(pop, event.node)
            _maybe_compact!(pop, _no_alive_vector)
        end

        isnothing(acc) || _check_popsize_triggers!(acc, pop)
    end
    return nothing
end

@noinline _corrupt_queue(te, t) = error(
    "event at t = $te precedes the current time t = $t; the event queue is corrupt")

# ── Extinction restarts ───────────────────────────────────────────────────────
#
# A call can only append rows, change links and fitness of existing rows, and mark rows
# removed. Rows that existed at the start (the prefix) are protected from compaction for
# the whole call, so restoring is: truncate to the prefix and copy its links and fitness
# back.

struct _RestartPoint
    n::Int32
    parent::Vector{Int32}
    left::Vector{Int32}
    right::Vector{Int32}
    fitness::Vector{Float64}
    ndead::Int
    nalive::Int
    t::Float64
    next_id::Int64
end

function _RestartPoint(pop::Population)
    tree = pop.tree
    n = Int32(length(tree))
    pop._protect = n
    pop._ndead_protected = tree.ndead
    return _RestartPoint(n, copy(tree.parent), copy(tree.left), copy(tree.right),
                         copy(tree.fitness), tree.ndead, pop._nalive, pop.t, pop._next_id)
end

function _restore!(pop::Population, point::_RestartPoint)
    tree = pop.tree
    for v in (tree.parent, tree.left, tree.right, tree.id, tree.birthtime, tree.fitness,
              tree.mutations, tree.total)
        resize!(v, point.n)
    end
    copyto!(tree.parent, point.parent)
    copyto!(tree.left, point.left)
    copyto!(tree.right, point.right)
    copyto!(tree.fitness, point.fitness)
    tree.ndead          = point.ndead
    pop._ndead_protected = point.ndead
    pop._nalive         = point.nalive
    pop.t               = point.t
    pop._next_id        = point.next_id
    # The carried events belong to the failed attempt: redraw for the restored cells.
    pop._pending = nothing
    return pop
end
