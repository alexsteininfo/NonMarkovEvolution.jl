"""
    NonMarkovBlock(; birth_dist, death_dist, effect_dist, fitness_update, ν,
                     stopfunction = pop -> false, tmax = Inf,
                     restart_on_extinction = false, on_division = nothing,
                     on_restart = nothing, algorithm = :auto)

Everything the cells do during one [`simulate!`](@ref) call.

- `birth_dist`, `death_dist` — `f -> Distribution`: waiting time from a cell's birth to
  its division, or to its death, given its fitness `f`. The earlier one happens.
- `effect_dist` — distribution of one mutation's effect size `δ`.
- `fitness_update` — `(f, δ) -> f′`, applied once per mutation.
- `ν` — mean mutations per daughter per division (Poisson), `ν ≥ 0`.
- `stopfunction` — `pop -> Bool`, tested before every event; the block stops when it
  returns `true`.
- `tmax` — the block also stops at exactly `pop.t = tmax`, without firing any later event.
- `restart_on_extinction` — restore the starting state and retry whenever the
  population dies out.
- `on_division` — `(pop, parent, d1, d2) -> nothing`, called after each division and
  before the daughters are scheduled. Change a daughter with [`set_fitness!`](@ref).
- `on_restart` — `pop -> nothing`, called after each extinction restart, to reset
  state kept in your hook closures. Drop any `CellNode` kept from the failed attempt
  there: the restart reuses its cells' ids.
- `algorithm` — how events are generated:
  - `:queue` — every cell draws its division and death times at birth, and a priority
    queue fires them in time order. Exact for any waiting-time distributions.
  - `:thinning` — the exponential (Markov) fast path: one global clock, a uniformly
    chosen cell, and acceptance by its rates. Needs `birth_dist` to return
    `Exponential`s and `death_dist` `Exponential`s or `Dirac(Inf)` (never dies).
    Same model, different (much cheaper) use of the random stream.
  - `:auto` (default) — `:thinning` when the return types of `birth_dist` and
    `death_dist` are both inferred as `Exponential`, otherwise `:queue`.
  `block.algorithm` holds the resolved choice.

The distributions are checked once at construction by calling them with `f = 1.0`. See
the manual pages *The simulation block* and *Mutations and selection* for waiting-time
parameterisations, stop conditions, hooks and selection modes.
"""
struct NonMarkovBlock{F1, F2, F3, F4, F5, F6, D}
    birth_dist::F1
    death_dist::F2
    stopfunction::F3
    effect_dist::D
    fitness_update::F4
    ν::Float64
    tmax::Float64
    restart_on_extinction::Bool
    on_division::F5
    on_restart::F6
    algorithm::Symbol

    function NonMarkovBlock(birth_dist::F1, death_dist::F2, stopfunction::F3,
                            effect_dist::D, fitness_update::F4, ν::Real, tmax::Real,
                            restart_on_extinction::Bool, on_division::F5,
                            on_restart::F6, algorithm::Symbol) where {F1, F2, F3, F4, F5, F6, D}
        ν >= 0 || throw(ArgumentError("NonMarkovBlock: ν must be >= 0, got $ν"))
        isnan(tmax) && throw(ArgumentError("NonMarkovBlock: tmax must not be NaN"))
        effect_dist isa Sampleable{Univariate} || throw(ArgumentError(
            "NonMarkovBlock: effect_dist must be a univariate distribution, got " *
            "$(typeof(effect_dist))"))
        for (name, dist) in (("birth_dist", birth_dist), ("death_dist", death_dist))
            d = dist(1.0)
            d isa Sampleable{Univariate} || throw(ArgumentError(
                "NonMarkovBlock: $name must map a fitness to a univariate distribution; " *
                "$name(1.0) returned a $(typeof(d))"))
        end
        resolved = _resolve_algorithm(algorithm, birth_dist, death_dist)
        return new{F1, F2, F3, F4, F5, F6, D}(birth_dist, death_dist, stopfunction,
            effect_dist, fitness_update, ν, tmax, restart_on_extinction, on_division,
            on_restart, resolved)
    end
end

function _resolve_algorithm(algorithm::Symbol, birth_dist, death_dist)
    algorithm in (:auto, :queue, :thinning) || throw(ArgumentError(
        "NonMarkovBlock: algorithm must be :auto, :queue or :thinning, got :$algorithm"))
    if algorithm === :auto
        exponential(g) = Base.promote_op(g, Float64) <: Exponential
        return exponential(birth_dist) && exponential(death_dist) ? :thinning : :queue
    end
    if algorithm === :thinning
        birth_dist(1.0) isa Exponential || throw(ArgumentError(
            "NonMarkovBlock: algorithm = :thinning needs birth_dist to return an " *
            "Exponential; birth_dist(1.0) returned a $(typeof(birth_dist(1.0)))"))
        d = death_dist(1.0)
        (d isa Exponential || (d isa Dirac && isinf(d.value))) || throw(ArgumentError(
            "NonMarkovBlock: algorithm = :thinning needs death_dist to return an " *
            "Exponential or Dirac(Inf); death_dist(1.0) returned $d"))
    end
    return algorithm
end

function NonMarkovBlock(; birth_dist, death_dist, effect_dist, fitness_update, ν,
                        stopfunction = Returns(false), tmax::Real = Inf,
                        restart_on_extinction::Bool = false, on_division = nothing,
                        on_restart = nothing, algorithm::Symbol = :auto)
    return NonMarkovBlock(birth_dist, death_dist, stopfunction, effect_dist,
                          fitness_update, ν, tmax, restart_on_extinction, on_division,
                          on_restart, algorithm)
end
