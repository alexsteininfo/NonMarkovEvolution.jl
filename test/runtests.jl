using NonMarkovEvolution
using Test
using Random
using AbstractTrees
using Distributions
using Statistics
using StableRNGs

include("fixtures.jl")

tests = [
    "initialisation",
    "tree",
    "events",
    "simulations",
    "regression",
    "chaining",
    "measurements",
    "statistics",
    "validation",
    "sampling",
    "thinning",
    "storage",
]

@testset "NonMarkovEvolution.jl" begin
    for test in tests
        @testset "$test" begin
            include(test * ".jl")
        end
    end
end
