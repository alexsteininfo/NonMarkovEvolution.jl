##
## Example 1: Single-cell expansion with mutation accumulation
##
## A single founding cell grows to K = 5 000 cells. Division times follow a Gamma with
## mean 1/f and CV 1/√2, so fitter cells divide faster. Each daughter acquires
## Poisson(ν = 0.5) mutations, each adding δ ~ Exponential(0.05) to its fitness.
## `restart_on_extinction` retries automatically if the founding lineage dies out early.
##
## Inputs : none.  Outputs: summary statistics printed to stdout.
## Run    : julia --project=. examples/01_SingleCellExpansion.jl
##

using NonMarkovEvolution
using Random
using Distributions
using Statistics: mean, std

rng = MersenneTwister(12)

K = 5_000
k = 2.0          # Gamma shape of the division time: CV = 1/√k

block = NonMarkovBlock(
    birth_dist     = f -> Gamma(k, 1.0 / (k * f)),     # mean division time 1/f
    death_dist     = f -> Exponential(20.0),            # mean lifetime 20, fitness-free
    stopfunction   = pop -> popsize(pop) >= K,
    effect_dist    = Exponential(0.05),
    fitness_update = (f, δ) -> f + δ,
    ν              = 0.5,
    restart_on_extinction = true,
)

pop = initialize_population(fitness_init = 1.0)
simulate!(pop, block, rng)

println("=== Single-cell expansion ===")
println("Population size : ", popsize(pop))
println("Simulation time : ", round(pop.t, digits = 2))

fits = fitness_per_cell(pop)
println("\nFitness across cells:")
println("  Mean   : ", round(mean(fits), digits = 4))
println("  Std    : ", round(std(fits),  digits = 4))
println("  Min    : ", round(minimum(fits), digits = 4))
println("  Max    : ", round(maximum(fits), digits = 4))

ks = mutations_per_cell(pop)
println("\nMutations per cell:")
println("  Mean   : ", round(mean(ks), digits = 2))
println("  Min    : ", minimum(ks))
println("  Max    : ", maximum(ks))
println("  Clonal : ", clonal_mutations(pop), " (shared by all cells)")

sfs = site_frequency_spectrum(pop)
sfs_nonzero = [(n, sfs[n]) for n in eachindex(sfs) if sfs[n] > 0]
println("\nMutation SFS (cells carrying, mutation events) — first 10 classes:")
for (freq, cnt) in first(sfs_nonzero, 10)
    println("  $freq cells: $cnt mutation(s)")
end
