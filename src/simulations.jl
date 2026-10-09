"""
    simulate!(population, block, rng; accumulator = nothing) -> Population

Advance `population` under `block` until `block.stopfunction` returns `true`, the clock
reaches `block.tmax`, or the population dies out (and is not restarted). Mutates
`population` in place and returns it, so blocks can be chained.

Every living cell has exactly one pending event — its division or its death, whichever
of its two waiting times is shorter — and events fire in global time order.

- **Chaining.** The queue of already-drawn events is carried on the population and
  reused, so a chained run is identical draw for draw to the equivalent uninterrupted
  one. Carried events keep the previous block's timing; call [`reset_schedule!`](@ref)
  to redraw every cell under the new block.
- **Fresh schedules** (a new population, after `reset_schedule!`, after an extinction
  restart) visit cells in id order, and draw each cell's next event conditioned on
  nothing having happened to it before the current time. This is exact.
- **Recording.** Pass a [`MeasurementAccumulator`](@ref) to record trajectories and
  snapshots.

The rng is required: pass a seeded one (`MersenneTwister(seed)`, or a `StableRNG`) so
the run is reproducible.
"""
function simulate!(
    population::Population,
    block::NonMarkovBlock,
    rng::AbstractRNG;
    accumulator::Union{MeasurementAccumulator, Nothing} = nothing,
)
    snapshot = block.restart_on_extinction ? _RestartPoint(population) : nothing
    mutation_count = Poisson(block.ν)

    if !isnothing(accumulator)
        _start_call!(accumulator, population)
        checkpoint = _checkpoint(accumulator)
    end

    while true
        heap = population._pending
        if isnothing(heap)
            heap = BinaryMinHeap{CellEvent}()
            for node in _sorted_cells(population)
                schedule_cell!(heap, node, block, rng, population.t)
            end
        elseif length(heap) != popsize(population)
            throw(ArgumentError(
                "population's carried event queue holds $(length(heap)) events but the " *
                "population has $(popsize(population)) alive cells — cells were added or " *
                "removed outside simulate!. Call reset_schedule!(population) to discard " *
                "the queue and redraw every cell's next event from scratch."))
        end
        # BinaryMinHeap mutates in place, so this reference stays current as events fire.
        population._pending = heap

        isnothing(accumulator) || _check_popsize_triggers!(accumulator, population)

        while !block.stopfunction(population) && !isempty(heap)
            if first(heap).time > block.tmax
                # Stop exactly at tmax; the next event stays queued for a chained call.
                population.t = max(population.t, block.tmax)
                break
            end
            event = pop!(heap)
            event.time >= population.t || error(
                "event at t = $(event.time) precedes the current time t = " *
                "$(population.t); the event queue is corrupt")
            # Record grid points and AtTime snapshots due before this event, while the
            # state is still exactly the state that held at those times.
            isnothing(accumulator) || _record_until!(accumulator, population, event.time)
            population.t = event.time

            if event.is_division
                d1, d2 = celldivision!(population, event.node, event.time, block,
                                       mutation_count, rng)
                # Must run before scheduling: schedule_cell! reads the fitness at push
                # time, so a boosted daughter's own first division uses the new fitness.
                isnothing(block.on_division) ||
                    block.on_division(population, event.node, d1, d2)
                schedule_cell!(heap, d1, block, rng)
                schedule_cell!(heap, d2, block, rng)
            else
                celldeath!(population, event.node)
            end

            isnothing(accumulator) || _check_popsize_triggers!(accumulator, population)
        end

        if popsize(population) == 0 && block.restart_on_extinction
            _restore!(population, snapshot)
            isnothing(accumulator) || _rollback!(accumulator, checkpoint)
            isnothing(block.on_restart) || block.on_restart(population)
        else
            break
        end
    end

    isnothing(accumulator) || _finish_call!(accumulator, population)
    return population
end

"""
    reset_schedule!(population) -> Population

Discard the queue of already-drawn events carried on `population`, so that the next
`simulate!` call redraws every living cell's next event under its block. The redraw is
conditioned on each cell's current age, so it is exact.

Also the recovery after adding or removing cells in `population.cells` by hand, which
otherwise leaves the queue inconsistent and makes `simulate!` throw.
"""
function reset_schedule!(population::Population)
    population._pending = nothing
    return population
end

# ── Extinction restarts ───────────────────────────────────────────────────────
#
# Instead of deep-copying the tree, record what a call can change: the starting cells'
# data, and every parent–child link on their ancestry (which `prune_tree!` may cut when a
# lineage dies). Restoring re-links those same node objects and drops everything born
# since, so the cost is the number of ancestors, with no allocation of nodes.

struct _RestartPoint
    cells::Dict{Int64, BinaryNode{NonMarkovCell}}
    data::Vector{Tuple{BinaryNode{NonMarkovCell}, NonMarkovCell}}
    links::Vector{Tuple{BinaryNode{NonMarkovCell}, BinaryNode{NonMarkovCell}, Bool}}
    t::Float64
    next_id::Int64
end

function _RestartPoint(population::Population)
    data  = [(node, node.data) for node in values(population.cells)]
    links = Tuple{BinaryNode{NonMarkovCell}, BinaryNode{NonMarkovCell}, Bool}[]
    # A pruned tree holds nothing but the living cells' ancestry, so one pre-order pass
    # per root lists exactly the links a failed attempt can cut.
    for root in _population_roots(population)
        nodes, parent = _preorder(root)
        for i in eachindex(nodes)
            parent[i] == 0 && continue
            p = nodes[parent[i]]
            push!(links, (nodes[i], p, p.left === nodes[i]))
        end
    end
    return _RestartPoint(copy(population.cells), data, links, population.t,
                         population._next_id)
end

function _restore!(population::Population, point::_RestartPoint)
    for (node, data) in point.data
        node.data  = data
        node.left  = nothing       # drop everything born during the failed attempt
        node.right = nothing
    end
    for (child, parent, isleft) in point.links
        child.parent = parent
        isleft ? (parent.left = child) : (parent.right = child)
    end
    population.cells    = copy(point.cells)
    population.t        = point.t
    population._next_id = point.next_id
    # The carried events point at the failed attempt's cells: redraw for the restored ones.
    population._pending = nothing
    return population
end
