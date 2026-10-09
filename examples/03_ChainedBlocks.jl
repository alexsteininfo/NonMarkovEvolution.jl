##
## Example 3: Two chained blocks — mutagenesis, then selection without new mutations
##
## Phase 1 grows one cell to 1 000 cells with a high mutation rate (ν = 1.0), building up
## fitness diversity. Phase 2 continues the same tree to 5 000 cells with ν = 0: no new
## mutations, so any rise in mean fitness is selection acting on phase-1 variation.
##
## The second `simulate!` call reuses the events already drawn in phase 1, so the chained
## run is draw-for-draw identical to an uninterrupted one. One accumulator is carried
## across both calls; `AtEnd` fires at the end of each.
##
## Inputs : none.  Outputs: summary statistics printed to stdout.
## Run    : julia --project=. examples/03_ChainedBlocks.jl
##

using NonMarkovEvolution
using Random
using Distributions
using Statistics: mean, std

rng = MersenneTwister(99)
k   = 2.0                                         # Gamma shape: CV = 1/√2

make_block(; Nmax, ν, restart = false) = NonMarkovBlock(
    birth_dist     = f -> Gamma(k, 1.0 / (k * f)),   # mean division time 1/f
    death_dist     = f -> Exponential(20.0),
    stopfunction   = pop -> popsize(pop) >= Nmax,
    effect_dist    = Exponential(0.05),               # unused when ν = 0
    fitness_update = (f, δ) -> f + δ,
    ν              = ν,
    restart_on_extinction = restart,
)

spec = MeasurementSpec(
    trajectory_dt     = 0.5,
    snapshot_triggers = [AtEnd()],
    snapshot_stats    = [FitnessDistribution(), MutationsPerCell()],
)
acc = MeasurementAccumulator(spec)
pop = initialize_population(fitness_init = 1.0)

simulate!(pop, make_block(Nmax = 1_000, ν = 1.0, restart = true), rng; accumulator = acc)
simulate!(pop, make_block(Nmax = 5_000, ν = 0.0), rng; accumulator = acc)
m = finalize_measurements(acc)

for (phase, snap) in zip(("Phase 1: mutations, ν = 1.0", "Phase 2: selection only, ν = 0"),
                         m.snapshots)
    println("=== ", phase, " ===")
    println("Population size : ", length(snap[:fitness]))
    println("Simulation time : ", round(snap.t, digits = 2))
    println("Mean fitness    : ", round(mean(snap[:fitness]), digits = 4))
    println("Std fitness     : ", round(std(snap[:fitness]),  digits = 4))
    println("Mean mutations  : ", round(mean(snap[:mutations]), digits = 2))
    println()
end
println("Trajectory points over both phases: ", length(m.trajectory))
println("Mean fitness typically rises in phase 2 without new mutations: ",
        "fitter lineages expand.")
