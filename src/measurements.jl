# Recording during `simulate!`: trajectory points on a time grid, and snapshots of
# user-chosen statistics when triggers fire.

# ── Trigger types ─────────────────────────────────────────────────────────────

"""
    AbstractTrigger

Supertype of the snapshot triggers [`AtEnd`](@ref), [`AtTime`](@ref) and
[`AtPopSize`](@ref).
"""
abstract type AbstractTrigger end

"""
    AtEnd()

Fire when a `simulate!` call exits (stop condition met or extinction) — once per call,
so an accumulator carried across chained blocks gets one snapshot per block.
"""
struct AtEnd <: AbstractTrigger end

"""
    AtTime(t)

Fire once with the population state exactly at simulation time `t`, i.e. after every
event at or before `t` and before any later one. The snapshot is labelled `t`.
"""
struct AtTime <: AbstractTrigger
    t::Float64
    function AtTime(t::Real)
        isnan(t) && throw(ArgumentError("AtTime: t must not be NaN"))
        return new(t)
    end
end

"""
    AtPopSize(N)

Fire once, at the first moment the population size is at least `N` (checked before the
first event too).
"""
struct AtPopSize <: AbstractTrigger
    N::Int
    function AtPopSize(N::Integer)
        N >= 1 || throw(ArgumentError("AtPopSize: N must be >= 1, got $N"))
        return new(N)
    end
end

# ── Statistics ────────────────────────────────────────────────────────────────

"""
    AbstractStatistic

Supertype of the statistics a snapshot can record. To add one, define a subtype and a
[`measure`](@ref) method; optionally a [`statistic_name`](@ref) method:

```julia
struct MeanFitness <: AbstractStatistic end
NonMarkovEvolution.measure(::MeanFitness, pop) = mean(fitness_per_cell(pop))
NonMarkovEvolution.statistic_name(::MeanFitness) = :mean_fitness
```
"""
abstract type AbstractStatistic end

"""
    measure(stat::AbstractStatistic, population)

Compute `stat` on the current population. Extend this for your own statistics.
"""
function measure end

"""
    statistic_name(stat::AbstractStatistic) -> Symbol

The key under which a snapshot stores `stat`'s value (`snap[name]`). Defaults to the
type name; the built-ins use `:sfs`, `:fitness` and `:mutations`.
"""
statistic_name(stat::AbstractStatistic) = Symbol(nameof(typeof(stat)))

"""
    SFS()

Snapshot statistic `:sfs` — the mutation site-frequency spectrum,
[`site_frequency_spectrum`](@ref)`(pop)`.
"""
struct SFS <: AbstractStatistic end
measure(::SFS, pop::Population) = site_frequency_spectrum(pop)
statistic_name(::SFS) = :sfs

"""
    FitnessDistribution()

Snapshot statistic `:fitness` — every living cell's fitness,
[`fitness_per_cell`](@ref)`(pop)`.
"""
struct FitnessDistribution <: AbstractStatistic end
measure(::FitnessDistribution, pop::Population) = fitness_per_cell(pop)
statistic_name(::FitnessDistribution) = :fitness

"""
    MutationsPerCell()

Snapshot statistic `:mutations` — every living cell's mutation burden,
[`mutations_per_cell`](@ref)`(pop)`, co-indexed with `:fitness`.
"""
struct MutationsPerCell <: AbstractStatistic end
measure(::MutationsPerCell, pop::Population) = mutations_per_cell(pop)
statistic_name(::MutationsPerCell) = :mutations

# ── Specification ─────────────────────────────────────────────────────────────

"""
    MeasurementSpec(; trajectory_dt = Inf, snapshot_triggers = [AtEnd()],
                      snapshot_stats = [FitnessDistribution()])

What to record during `simulate!`: a trajectory point every `trajectory_dt` time units
(`Inf` disables it), and a snapshot of every statistic in `snapshot_stats` whenever one
of `snapshot_triggers` fires. Pass it to [`MeasurementAccumulator`](@ref).
"""
struct MeasurementSpec
    trajectory_dt::Float64
    snapshot_triggers::Vector{AbstractTrigger}
    snapshot_stats::Vector{AbstractStatistic}

    function MeasurementSpec(trajectory_dt::Real, snapshot_triggers, snapshot_stats)
        # `> 0` also rejects NaN. A zero step would make trajectory recording loop forever.
        trajectory_dt > 0 || throw(ArgumentError(
            "MeasurementSpec: trajectory_dt must be > 0 (Inf disables it), got $trajectory_dt"))
        names = [statistic_name(s) for s in snapshot_stats]
        allunique(names) || throw(ArgumentError(
            "MeasurementSpec: two snapshot statistics share a name in $names"))
        for s in snapshot_stats
            hasmethod(measure, Tuple{typeof(s), Population}) || throw(ArgumentError(
                "MeasurementSpec: no `measure(::$(typeof(s)), ::Population)` method"))
        end
        return new(trajectory_dt, snapshot_triggers, snapshot_stats)
    end
end

MeasurementSpec(; trajectory_dt = Inf, snapshot_triggers = [AtEnd()],
                  snapshot_stats = [FitnessDistribution()]) =
    MeasurementSpec(Float64(trajectory_dt), Vector{AbstractTrigger}(snapshot_triggers),
                    Vector{AbstractStatistic}(snapshot_stats))

# ── Output types ──────────────────────────────────────────────────────────────

"""
    TrajectoryPoint

The exact population state at grid time `t`: size `N_total`, and mean and variance of
fitness and of the mutation burden across living cells. Variances are `NaN` while only one
cell is alive.
"""
struct TrajectoryPoint
    t::Float64
    N_total::Int
    mean_fitness::Float64
    var_fitness::Float64
    mean_mutations::Float64
    var_mutations::Float64
end

"""
    SnapshotData

A snapshot taken when a trigger fires: its time `t`, the `trigger`, and one value per
requested statistic, read by name — `snap[:sfs]`, `snap[:fitness]`, `snap[:mutations]` —
with `haskey(snap, name)` and `keys(snap)` to inspect what it holds.
"""
struct SnapshotData
    t::Float64
    trigger::AbstractTrigger
    values::Dict{Symbol, Any}
end

Base.getindex(s::SnapshotData, name::Symbol) = s.values[name]
Base.haskey(s::SnapshotData, name::Symbol)   = haskey(s.values, name)
Base.keys(s::SnapshotData)                   = keys(s.values)

"""
    Measurements

Everything an accumulator recorded: `trajectory::Vector{TrajectoryPoint}` (empty if
`trajectory_dt = Inf`) and `snapshots::Vector{SnapshotData}`, in the order they fired.
"""
struct Measurements
    trajectory::Vector{TrajectoryPoint}
    snapshots::Vector{SnapshotData}
end

# ── Accumulator ───────────────────────────────────────────────────────────────

"""
    MeasurementAccumulator(spec::MeasurementSpec)

Mutable collector that `simulate!` writes into: pass it as the `accumulator` keyword and
call [`finalize_measurements`](@ref) afterwards.

One accumulator can be carried across chained `simulate!` calls: trajectory recording
resumes at the current time, `AtTime` and `AtPopSize` fire at most once per accumulator,
and `AtEnd` fires at the end of every call. An extinction restart discards only what the
current call recorded.
"""
mutable struct MeasurementAccumulator
    spec::MeasurementSpec
    next_trajectory_t::Float64
    trajectory_points::Vector{TrajectoryPoint}
    snapshots::Vector{SnapshotData}
    fired_triggers::Set{Int}
    # Time at which the current `simulate!` call started. An `AtTime` earlier than this
    # cannot be recorded exactly any more, so it never fires.
    call_start_t::Float64
end

MeasurementAccumulator(spec::MeasurementSpec) =
    MeasurementAccumulator(spec, 0.0, TrajectoryPoint[], SnapshotData[], Set{Int}(), 0.0)

"""
    finalize_measurements(acc::MeasurementAccumulator) -> Measurements

Copy what `acc` has recorded into a `Measurements`. The accumulator stays usable.
"""
finalize_measurements(acc::MeasurementAccumulator) =
    Measurements(copy(acc.trajectory_points), copy(acc.snapshots))

# ── Recording (called by `simulate!`) ─────────────────────────────────────────
#
# Timing convention: the population state is right-continuous — the state "at time t"
# includes every event at or before t. `simulate!` calls `_record_until!` with the time
# of the next event *before* applying it, so every grid time and every `AtTime` strictly
# earlier than that event is recorded with the state that held there exactly.

function _start_call!(acc::MeasurementAccumulator, pop::Population)
    acc.call_start_t = pop.t
    # A fresh accumulator on a chained call would otherwise back-fill from t = 0.
    acc.next_trajectory_t = max(acc.next_trajectory_t, pop.t)
    return acc
end

_snapshot(acc::MeasurementAccumulator, trigger::AbstractTrigger, pop::Population, t) =
    SnapshotData(t, trigger, Dict{Symbol, Any}(statistic_name(s) => measure(s, pop)
                                               for s in acc.spec.snapshot_stats))

function _fire!(acc::MeasurementAccumulator, i::Int, trigger, pop, t)
    push!(acc.fired_triggers, i)
    push!(acc.snapshots, _snapshot(acc, trigger, pop, t))
end

# Record every trajectory point and fire every `AtTime` trigger strictly before `t_next`
# (or at or before it when `inclusive`), using the current state.
function _record_until!(acc::MeasurementAccumulator, pop::Population, t_next::Float64;
                        inclusive::Bool = false)
    due(t) = inclusive ? t <= t_next : t < t_next
    dt = acc.spec.trajectory_dt
    if !isinf(dt) && due(acc.next_trajectory_t) && popsize(pop) > 0
        f, k = _fitnesses(pop), _burdens(pop)
        mf, vf, mk, vk = mean(f), var(f), mean(k), var(k)
        while due(acc.next_trajectory_t)
            push!(acc.trajectory_points,
                  TrajectoryPoint(acc.next_trajectory_t, popsize(pop), mf, vf, mk, vk))
            acc.next_trajectory_t += dt
        end
    end
    for (i, trigger) in enumerate(acc.spec.snapshot_triggers)
        trigger isa AtTime && !(i in acc.fired_triggers) && due(trigger.t) &&
            trigger.t >= acc.call_start_t && _fire!(acc, i, trigger, pop, trigger.t)
    end
end

function _check_popsize_triggers!(acc::MeasurementAccumulator, pop::Population)
    N = popsize(pop)
    for (i, trigger) in enumerate(acc.spec.snapshot_triggers)
        trigger isa AtPopSize && !(i in acc.fired_triggers) && N >= trigger.N &&
            _fire!(acc, i, trigger, pop, pop.t)
    end
end

# At exit: the final state holds at `pop.t`, so grid points and `AtTime`s at exactly
# `pop.t` are still due; then every `AtEnd` fires (once per call, never marked fired).
function _finish_call!(acc::MeasurementAccumulator, pop::Population)
    _record_until!(acc, pop, pop.t; inclusive = true)
    for trigger in acc.spec.snapshot_triggers
        trigger isa AtEnd && push!(acc.snapshots, _snapshot(acc, trigger, pop, pop.t))
    end
end

# What the accumulator held when a `simulate!` call began, so that an extinction
# restart can discard exactly what that call recorded.
_checkpoint(acc::MeasurementAccumulator) = (
    n_points = length(acc.trajectory_points),
    n_snaps  = length(acc.snapshots),
    fired    = copy(acc.fired_triggers),
    next_t   = acc.next_trajectory_t,
)

function _rollback!(acc::MeasurementAccumulator, cp)
    resize!(acc.trajectory_points, cp.n_points)
    resize!(acc.snapshots, cp.n_snaps)
    acc.fired_triggers    = copy(cp.fired)
    acc.next_trajectory_t = cp.next_t
    return acc
end
