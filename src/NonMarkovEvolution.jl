"""
    NonMarkovEvolution

Stochastic, non-Markovian birth–death evolution of cell populations with fitness-changing
mutations: per-cell fitness and the complete lineage tree of the survivors.
"""
module NonMarkovEvolution

using Distributions
using Statistics
using Random
using AbstractTrees
using DataStructures: BinaryMinHeap
using StableRNGs: StableRNG

export
# Block
NonMarkovBlock,

# Cells, trees and the population
NonMarkovCell,
BinaryNode,
Population,
set_fitness!,

# Simulation
simulate!,
initialize_population,
reset_schedule!,
has_pending_schedule,

# Tree utilities
alive_cells,
popsize,
roots,
single_root,
find_mrca,
left_child!,
right_child!,
division_time,
cell_lifetime,
cell_lifetimes,
last_division_time,

# Statistics
mutations_per_cell,
clonal_mutations,
mean_mutations,
var_mutations,
fitness_per_cell,
leaf_fitness,
filtered_mutations_per_cell,
pairwise_distance,
pairwise_distances,
coalescence_times,
site_frequency_spectrum,
branch_spectrum,
leaf_depths,

# Sampling
LeafSample,
sample_leaves,
SamplingSpec,
SampledTrees,
sample_trees,

# Measurements
MeasurementSpec,
MeasurementAccumulator,
Measurements,
TrajectoryPoint,
SnapshotData,
finalize_measurements,
AbstractTrigger,
AtEnd,
AtTime,
AtPopSize,
AbstractStatistic,
SFS,
FitnessDistribution,
MutationsPerCell,
measure,
statistic_name

include("types.jl")
include("blocks.jl")
include("events.jl")
include("initialisation.jl")
include("cellupdates.jl")
include("simulation_trees.jl")
include("statistics.jl")
include("sampling.jl")
include("measurements.jl")
include("simulations.jl")

end
