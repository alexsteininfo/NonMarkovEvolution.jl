# The simulation block

[`NonMarkovBlock`](@ref) is the complete description of what cells do during one
[`simulate!`](@ref) call. Every field is a function or a distribution you supply, so each
is independently replaceable and cheap to sweep.

```julia
block = NonMarkovBlock(
    birth_dist     = f -> Gamma(5.0, 1 / (5 * f)),   # required
    death_dist     = f -> Exponential(1 / 0.3),       # required
    effect_dist    = Exponential(0.05),               # required
    fitness_update = (f, δ) -> f + δ,                 # required
    ν              = 0.2,                             # required
    stopfunction   = pop -> popsize(pop) >= 10_000,   # default: never
    tmax           = Inf,                             # default: no time limit
    restart_on_extinction = false,
    on_division    = nothing,
    on_restart     = nothing,
)
```

| Field | Signature | Role |
|:---|:---|:---|
| `birth_dist` | `f -> Distribution` | waiting time from birth to division, given the cell's fitness |
| `death_dist` | `f -> Distribution` | waiting time from birth to death |
| `effect_dist` | `Distribution` | effect size ``δ`` of one mutation |
| `fitness_update` | `(f, δ) -> f′` | how one mutation changes fitness |
| `ν` | `Real ≥ 0` | mean mutations per daughter per division (Poisson) |
| `stopfunction` | `pop -> Bool` | stop when it returns `true` |
| `tmax` | `Real` | stop at exactly this time |
| `restart_on_extinction` | `Bool` | retry from the starting state if the population dies |
| `on_division` | `(pop, parent, d1, d2) -> nothing` | hook at every division |
| `on_restart` | `pop -> nothing` | hook after an extinction restart |

The waiting-time and mutation fields have no defaults on purpose: a default distribution
would be a scientific claim smuggled in as a convenience. The constructor checks them
once, by calling `birth_dist(1.0)` and `death_dist(1.0)`. `effect_dist`,
`fitness_update` and `ν` are covered in [Mutations and selection](selection.md).

## Waiting-time modes

`birth_dist` and `death_dist` map a cell's fitness to any univariate distribution over
non-negative reals. The useful parameterisation fixes the **mean** at ``1/(bf)`` and lets
the shape control only the variability, so that ``b`` is a rate comparable across modes:

| Mode | `birth_dist` | CV |
|:---|:---|:---|
| Deterministic | `f -> Dirac(1 / (b * f))` | ``0`` |
| Exponential (Markov) | `f -> Exponential(1 / (b * f))` | ``1`` |
| Gamma | `f -> Gamma(k, 1 / (k * b * f))` | ``1/\sqrt{k}`` |
| Weibull | `f -> Weibull(α, 1 / (b * f * mean(Weibull(α, 1.0))))` | ``\sqrt{\Gamma(1+2/α)/\Gamma(1+1/α)^2 - 1}`` |
| Log-normal | `f -> LogNormal(-log(b * f) - σ^2/2, σ)` | ``\sqrt{e^{σ^2}-1}`` |

Every row has mean ``1/(bf)``, which this block checks (and `test/events.jl` asserts):

```@example waiting
using NonMarkovEvolution, Distributions, Statistics
b, f, k, α, σ = 1.3, 1.7, 5.0, 2.5, 0.4
modes = [Dirac(1 / (b * f)), Exponential(1 / (b * f)), Gamma(k, 1 / (k * b * f)),
         Weibull(α, 1 / (b * f * mean(Weibull(α, 1.0)))), LogNormal(-log(b * f) - σ^2 / 2, σ)]
[(nameof(typeof(d)), round(mean(d) * b * f, digits = 12), round(std(d) / mean(d), digits = 3))
 for d in modes]
```

Note the Gamma parameterisation: `Gamma(k, 1/(k*b*f))`, **not** `Gamma(k, 1/(b*f))`.
`Gamma(α, θ)` has mean ``αθ``, so the second form has mean ``k/(bf)``, and changing ``k``
would silently rescale time as well as the noise. (For the Weibull row,
`mean(Weibull(α, 1.0))` is ``\Gamma(1 + 1/α)``, which avoids a SpecialFunctions
dependency.)

- **Deterministic** (`Dirac`) is the ``k \to \infty`` limit: from one cell the population
  doubles in lockstep, and every observable has a closed form — the cleanest end-to-end
  check of a pipeline. Tied events fire consecutively; a birth/death tie goes to division.
- **`Dirac(Inf)`** is the idiom for "never dies": it always loses the competing-risks
  comparison and consumes no random numbers. `Exponential(1/d)` at `d = 0` also returns
  `Inf`, but burns a draw; `Exponential(0.0)` throws.
- **Exponential** reproduces the Markov birth–death process exactly, with ``r = b - d``
  and extinction probability ``d/b`` from one cell.
- **Gamma** is the recommended default; see
  [Why not the exponential?](concepts.md#Why-not-the-exponential?) for the empirical CVs
  behind ``k = 5``. Death is often better left exponential even when division is not
  (Fennell et al. 2005), and mixing the two is fine.

### Coupling fitness to death instead of division

Nothing forces fitness into `birth_dist`:

```julia
birth_dist = f -> Gamma(k, 1 / (k * b * f)); death_dist = f -> Gamma(k, 1 / (k * d))  # faster division
birth_dist = f -> Gamma(k, 1 / (k * b));     death_dist = f -> Gamma(k, f / (k * d))  # longer survival
```

These are different processes, not reparameterisations: even tuned to a common growth
rate they leave different trees. A birth advantage deepens the fit lineage and compresses
its coalescence times; a survival advantage makes it lose fewer branches.
[`leaf_depths`](@ref) and [`coalescence_times`](@ref) together distinguish the two.

!!! danger "Fitness must stay strictly positive"
    Fitness divides into a scale parameter, so a rule that lets ``f`` reach ``0`` fails at
    the next scheduling with a `DomainError` from the distribution constructor, partway
    through the run. Deleterious effects need an explicit floor; see
    [Deleterious mutations](selection.md#Deleterious-mutations-and-mutational-load).

## Stopping

A block stops at the first of: `stopfunction(pop)` returning `true` (tested before every
event, including the first), the next event lying beyond `tmax`, or extinction.

```julia
stopfunction = pop -> popsize(pop) >= 10_000                    # fixed size
tmax         = 20.0                                             # fixed time
stopfunction = pop -> popsize(pop) >= 10_000; tmax = 20.0       # whichever comes first
```

- **A size target is hit exactly**, because the size moves by ``\pm 1`` per event.
- **`tmax` is exact.** The block stops with `pop.t == tmax`, and the next event stays
  queued, so a following block continues seamlessly. A time condition inside
  `stopfunction` (`pop -> pop.t >= T`) instead overshoots by one event; prefer `tmax`.
- **With neither**, the block runs until extinction — which never happens under pure
  birth.

!!! warning "Keep `stopfunction` cheap"
    It runs once per event, so a function that scans the population —
    `pop -> mean(fitness_per_cell(pop)) > 2` — makes the run quadratic. Track such a
    quantity incrementally in an `on_division` closure and have `stopfunction` read it.

### Growth regimes

With ``p = P(T_\text{div} < T_\text{die})``, a lineage is a branching process with mean
offspring ``2p``: supercritical for ``p > 1/2``, with extinction probability
``(1-p)/p`` from one cell; critical or subcritical, with certain extinction, otherwise.
For exponentials ``p = b/(b+d)``, giving ``d/b``. For other distributions ``p`` is a
one-dimensional integral — for Gamma division and exponential death it is the Gamma
moment-generating function at ``-d`` — which the test suite checks against simulation.
At ``b = 1, d = 0.5`` half of all single-founder attempts die out: that is the model, not
a bug.

### Density-dependent and homeostatic growth

`birth_dist` receives only fitness, but a closure can read the population's size:

```julia
K   = 10_000
pop = initialize_population()
block = NonMarkovBlock(
    birth_dist = f -> Gamma(k, 1 / (k * b * f * max(1e-6, 1 - popsize(pop) / K))),
    death_dist = f -> Gamma(k, 1 / (k * d)), tmax = 100.0,
    effect_dist = Dirac(0.0), fitness_update = (f, δ) -> f, ν = 0.0)
```

(The `max(1e-6, …)` floor matters: at ``N = K`` the bare factor is 0 and the scale `Inf`.)

!!! warning "Density dependence is frozen at each cell's birth"
    A waiting time is drawn once, at birth, and never revised. The rule above therefore
    uses the density **as of each cell's birth**, and the population overshoots ``K`` —
    even for exponential waiting times, where a true Gillespie implementation would
    resample on every rate change.

    The exact alternative is to change the *regime* rather than the rate, by chaining:
    grow to ``K`` under one block, then continue under a block with equal birth and death
    laws. That turnover is simulated exactly, but it is critical rather than regulated:
    ``N`` starts at ``K`` and performs an unbiased random walk whose variance grows with
    time (and which is eventually absorbed at 0).

```julia
grow = NonMarkovBlock(birth_dist = f -> Gamma(k, 1/(k*b*f)), death_dist = f -> Dirac(Inf),
                      stopfunction = p -> popsize(p) >= K, …)
hold = NonMarkovBlock(birth_dist = f -> Gamma(k, 1/(k*b*f)), death_dist = f -> Gamma(k, 1/(k*b)),
                      tmax = t_end, …)
simulate!(pop, grow, rng)
simulate!(pop, hold, rng)     # same tree, critical turnover: E[N] stays at K
```

## Extinction and restarting

With `restart_on_extinction = true`, `simulate!` records the population as it is when
the call begins and restores it whenever the population dies out, until an attempt
reaches the stop condition. A restart resets the cells, `pop.t`, the id counter, the
event queue and whatever the accumulator recorded during this call; earlier chained
blocks' records survive. The tree above the starting cells is restored in place — the
same node objects, re-linked — so references you hold stay valid, and the restored cells
are rescheduled conditioned on their age, so restarts are exact on any block.

- **The retry is unbounded.** A critical or subcritical block with the flag never
  terminates; check ``p > 1/2`` first.
- **Ids are reused across attempts**, since the id counter is restored too. Do not cache
  ids of cells born during a call that might restart.
- **Your closures are not reset.** A hook holding a one-shot flag spends it on an attempt
  that later dies; reset it in `on_restart` (below).

For a more selective retry (say, until a mutation clone survives drift), leave the flag off
and loop yourself, with a fresh closure per attempt:

```julia
function attempt(seed)
    pop = initialize_population()
    simulate!(pop, make_block(), MersenneTwister(seed))
    return clone_survived(pop) ? pop : nothing
end
pop = nothing
for seed in 1:1000
    pop = attempt(seed)
    isnothing(pop) || break
end
```

## Hooks

### `on_division`

`on_division(pop, parent, d1, d2)` runs once per division, **after** both daughters exist
and **before** either is scheduled, so a change applies to a daughter's own first
division. It is for deterministic, state-triggered interventions the Poisson channel
cannot express: injecting one mutation into one cell at one moment, tallying a statistic
incrementally, watching a lineage.

Change a daughter with [`set_fitness!`](@ref); the new fitness is inherited by all its
descendants. Inside the hook the parent has already been replaced by the daughters, so a
population that had `M` cells reads as `M + 1`.

```@example hooks
using NonMarkovEvolution, Distributions, Random
N_critic, s = 20, 1.0
injected    = Ref(false)
block = NonMarkovBlock(
    birth_dist     = f -> Gamma(5.0, 1 / (5 * f)),
    death_dist     = f -> Exponential(10.0),
    effect_dist    = Dirac(0.0), fitness_update = (f, δ) -> f, ν = 0.0,
    stopfunction   = pop -> popsize(pop) >= 2_000,
    restart_on_extinction = true,
    on_division = function (pop, parent, d1, d2)
        if !injected[] && popsize(pop) == N_critic + 1
            injected[] = true
            set_fitness!(d1, 1.0 + s)          # one mutation, one cell, one moment
        end
        return nothing
    end,
    on_restart = pop -> (injected[] = false; nothing),
)
pop = simulate!(initialize_population(), block, MersenneTwister(1))
count(>(1.0), fitness_per_cell(pop))       # cells descending from the boosted daughter
```

A single injected mutation can still be lost to drift — the boosted daughter may die
before it divides — so a study of its fate needs many seeds, or a retry loop that checks
for the clone. Keep hook state in the closure: the package holds no global mutable state. A hook that
does not draw from `rng` leaves the random stream untouched, so a watcher can be added to
an existing run without changing its result.

### `on_restart`

`on_restart(pop)` runs after each extinction restart, and exists to reset state your
closures own — as `injected` above. It is only called when
`restart_on_extinction = true`.

## Chaining blocks

Calling [`simulate!`](@ref) again on the same population continues the **same lineage
tree** under new rules — growth then turnover, mutagen on then off, treatment then
relapse:

```julia
simulate!(pop, growth_block, rng)    # ν = 1.0, grow to 1 000
simulate!(pop, neutral_block, rng)   # ν = 0.0, grow to 5 000 — same tree
```

Cells alive at a boundary have already survived part of a cycle, so their next event
follows the conditional law ``T \mid T > \text{age}``, not the law from birth. The
already-drawn event queue is carried on the population and reused, which respects that
trivially — each cell keeps the event it committed to — and makes a chained run identical
draw for draw to the uninterrupted one.

!!! note "Carried events keep the previous block's timing"
    A second block that changes `birth_dist` or `death_dist` therefore affects only cells
    scheduled after the boundary; the change phases in over about one cell cycle.
    `effect_dist`, `fitness_update` and `ν` are read at division time and act at once.

To make new waiting-time laws act on every cell at once, discard the queue first:

```julia
reset_schedule!(pop)
simulate!(pop, second_block, rng)
```

The redraw is exact: each cell's next event is drawn by rejection from the new block's
laws, conditioned on nothing having happened to it before `pop.t`. It costs extra draws
for old cells, and the run no longer matches an uninterrupted one draw for draw. If a
cell has outlived what the new law allows (a `Dirac` clock, say), `simulate!` raises an
error rather than looping. `reset_schedule!` is also the recovery after adding or
removing cells in `pop.cells` by hand; [`has_pending_schedule`](@ref) tells whether a
queue is carried.
