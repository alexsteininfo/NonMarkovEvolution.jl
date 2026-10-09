##
## Example 2: Growth with mutations, recorded while it runs
##
## Grows one cell to 5 000 cells under additive selection and records a trajectory every
## time unit, a snapshot when the population first reaches 1 000 cells, and one at the
## end. Shows the MeasurementSpec → MeasurementAccumulator → finalize_measurements flow.
##
## Inputs : none.  Outputs: summary statistics printed to stdout.
## Run    : julia --project=. examples/02_GrowthWithMutations.jl
##

using NonMarkovEvolution
using Distributions
using Random
using Statistics: mean, std

rng = MersenneTwister(42)
k   = 5.0                                  # Gamma shape: CV = 1/√5 ≈ 0.45

block = NonMarkovBlock(
    birth_dist     = f -> Gamma(k, 1.0 / (k * f)),    # mean division time 1/f
    death_dist     = f -> Gamma(k, 10.0 / k),         # mean lifetime 10, fitness-free
    stopfunction   = pop -> popsize(pop) >= 5_000,
    effect_dist    = Exponential(0.05),
    fitness_update = (f, δ) -> f + δ,                 # additive fitness
    ν              = 0.5,                             # mean mutations per daughter
    restart_on_extinction = true,
)

spec = MeasurementSpec(
    trajectory_dt     = 1.0,
    snapshot_triggers = [AtPopSize(1_000), AtEnd()],
    snapshot_stats    = [SFS(), FitnessDistribution(), MutationsPerCell()],
)
acc = MeasurementAccumulator(spec)

pop = initialize_population(fitness_init = 1.0)
simulate!(pop, block, rng; accumulator = acc)
m = finalize_measurements(acc)

println("Final population size : ", popsize(pop))
println("Simulation time       : ", round(pop.t, digits = 2))
println("Trajectory points     : ", length(m.trajectory))
for snap in m.snapshots
    println("\nSnapshot ", snap.trigger, " at t = ", round(snap.t, digits = 2))
    println("  cells              : ", length(snap[:fitness]))
    println("  mean fitness       : ", round(mean(snap[:fitness]), digits = 4))
    println("  mean mutations/cell: ", round(mean(snap[:mutations]), digits = 2))
    println("  SFS classes > 0    : ", count(>(0), snap[:sfs]))
end
