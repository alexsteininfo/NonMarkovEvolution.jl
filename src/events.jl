# Drawing a cell's next event for the queue algorithm.

# Rejection draws allowed before `schedule_cell!` gives up on conditioning an old cell.
const _MAX_CONDITIONING_DRAWS = 100_000

"""
    _draw_event(tree, i, block, rng, tmin = tree.birthtime[i]) -> Event

The next event of cell `i` by competing risks: draw a division time from
`block.birth_dist` and a death time from `block.death_dist`, both measured from the
cell's birth, and keep the earlier one (a tie goes to division).

`tmin` is the earliest time the event may fire. For a newborn cell it is the birthtime
and the first draw is always accepted. When an *existing* cell is rescheduled (a fresh
population with old birthtimes, [`reset_schedule!`](@ref), or an extinction restart),
`tmin` is the current time and both draws are repeated until the earlier one falls at or
after `tmin`. That is exact rejection sampling from the law of the cell's next event
given that nothing happened to it before `tmin`.
"""
@inline function _draw_event(tree::LineageTree, i::Int32, block::NonMarkovBlock,
                             rng::AbstractRNG, tmin::Float64 = @inbounds tree.birthtime[i])
    f  = @inbounds tree.fitness[i]
    t0 = @inbounds tree.birthtime[i]
    for _ in 1:_MAX_CONDITIONING_DRAWS
        t_div = t0 + rand(rng, block.birth_dist(f))
        t_die = t0 + rand(rng, block.death_dist(f))
        if min(t_div, t_die) >= tmin
            return t_div <= t_die ? Event(t_div, i, true) : Event(t_die, i, false)
        end
    end
    _too_old(tree, i, tmin)
end

@noinline _too_old(tree, i, tmin) = error(
    "cell $(tree.id[i]), born at t = $(tree.birthtime[i]), could not be given an event " *
    "at or after t = $tmin in $_MAX_CONDITIONING_DRAWS draws: under this block it would " *
    "almost surely have divided or died already. Its age is incompatible with the " *
    "waiting-time distributions.")

"""
    schedule_cell!(queue, tree, i, block, rng, tmin = tree.birthtime[i])

Draw cell `i`'s next event with [`_draw_event`](@ref) and push it onto `queue`.
"""
schedule_cell!(q::EventQueue, tree::LineageTree, i::Integer, block::NonMarkovBlock,
               rng::AbstractRNG, tmin::Real = tree.birthtime[i]) =
    push!(q, _draw_event(tree, Int32(i), block, rng, Float64(tmin)))
