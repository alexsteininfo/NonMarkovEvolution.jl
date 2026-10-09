##
## Example 4: Effect of the initial condition on genetic diversity
##
## Two scenarios grow to the same final size (N = 1 000) under the same block but start
## from different numbers of founders:
##
##   A: 1 founding cell    — a single tree; mutations acquired early are shared widely.
##   B: 100 founding cells — a forest of 100 independent trees of ~10 cells each.
##
## In B each founder's tree is shallow, so cells carry fewer mutations, almost every
## mutation is carried by few cells, and nothing is shared by all cells. Distances are
## computed on a reproducible random sample of 30 cells.
##
## Inputs : none.  Outputs: summary statistics printed to stdout.
## Run    : julia --project=. examples/04_ArbitraryInitialCondition.jl
##

using NonMarkovEvolution
using Random
using Distributions
using Statistics: mean

N_final = 1_000
k       = 2.0

block = NonMarkovBlock(
    birth_dist     = f -> Gamma(k, 1.0 / (k * f)),     # mean division time 1/f
    death_dist     = f -> Exponential(20.0),
    stopfunction   = pop -> popsize(pop) >= N_final,
    effect_dist    = Exponential(0.05),
    fitness_update = (f, δ) -> f + δ,
    ν              = 0.5,
    restart_on_extinction = true,
)

function summarise(label, pop, rng)
    sfs = site_frequency_spectrum(pop)
    idx = randperm(rng, popsize(pop))[1:30]          # a random, reproducible subset
    println("=== $label ===")
    println("Simulation time    : ", round(pop.t, digits = 2))
    println("Mean fitness       : ", round(mean(fitness_per_cell(pop)), digits = 4))
    println("Mean mutations/cell: ", round(mean(mutations_per_cell(pop)), digits = 2))
    println("Clonal mutations   : ", clonal_mutations(pop), "  (shared by all cells)")
    println("Singleton fraction : ", round(100 * sfs[1] / sum(sfs), digits = 1), "%")
    println("Mean pairwise dist : ", round(mean(pairwise_distances(pop, idx)), digits = 2),
            "  (30 random cells)")
    println()
end

# Separate generators: the two scenarios consume random numbers differently anyway, so
# a shared seed would not make them "matched" runs.
popA = initialize_population(fitness_init = 1.0)
simulate!(popA, block, MersenneTwister(77))
summarise("A: 1 founding cell → $N_final cells", popA, MersenneTwister(1))

popB = initialize_population(100; fitness_init = 1.0)
simulate!(popB, block, MersenneTwister(78))
summarise("B: 100 founding cells → $N_final cells", popB, MersenneTwister(1))
