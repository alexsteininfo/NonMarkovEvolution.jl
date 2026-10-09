# Saving and loading lineage trees.

@isdefined(fixture_tree) || include("fixtures.jl")

@testset "a saved population loads back identical" begin
    pop = initialize_population()
    simulate!(pop, grow_block(Nmax = 400, ν = 1.5, death = f -> Gamma(2.0, 2.0),
                              restart = true), MersenneTwister(3))
    @test pop.tree.ndead > 0                       # removed rows exist and are skipped
    mktempdir() do dir
        path = joinpath(dir, "pop.nmet")
        save_tree(path, pop)
        tree, t = load_tree(path)
        @test t === pop.t
        @test length(tree) == length(pop.tree) - pop.tree.ndead
        r0, r1 = single_root(pop), single_root(tree)
        @test [l.id for l in Leaves(r1)] == [l.id for l in Leaves(r0)]
        @test site_frequency_spectrum(r1) == site_frequency_spectrum(pop)
        @test leaf_depths(r1) == leaf_depths(r0)
        @test leaf_fitness(r1) == leaf_fitness(r0)
        @test mutations_per_cell(r1; includeclonal = true) ==
              mutations_per_cell(r0; includeclonal = true)
        @test [l.birthtime for l in Leaves(r1)] == [l.birthtime for l in Leaves(r0)]
        # Sampling draws the same cells from the loaded tree.
        @test sample_leaves(r1, 50; seed = 9).sampled_ids ==
              sample_leaves(pop, 50; seed = 9).sampled_ids
    end
end

@testset "subtrees and forests" begin
    root = fixture_tree()
    mktempdir() do dir
        path = joinpath(dir, "sub.nmet")
        save_tree(path, root.left)
        tree, t = load_tree(path)
        @test isnan(t)
        r = single_root(tree)
        @test r.id == 2
        @test [l.id for l in Leaves(r)] == [4, 5]
        @test mutations_per_cell(r; includeclonal = true) == [8, 9]

        forest = initialize_population(5)
        simulate!(forest, grow_block(Nmax = 60), MersenneTwister(1))
        save_tree(joinpath(dir, "forest.nmet"), forest)
        ftree, _ = load_tree(joinpath(dir, "forest.nmet"))
        @test [r.id for r in roots(ftree)] == [r.id for r in roots(forest)]
        @test sum(popsize, roots(ftree)) == popsize(forest)
    end
end

@testset "a file that is not a tree is rejected" begin
    mktempdir() do dir
        path = joinpath(dir, "junk")
        write(path, "this is not a tree file at all")
        @test_throws ArgumentError load_tree(path)
    end
end

@testset "a loaded tree becomes a population again, with checked ids" begin
    pop = initialize_population()
    simulate!(pop, grow_block(Nmax = 200), MersenneTwister(5))
    mktempdir() do dir
        path = joinpath(dir, "p.nmet")
        save_tree(path, pop)
        tree, t = load_tree(path)
        @test_throws ArgumentError Population(tree, t; next_id = 1)   # would reuse ids
        again = Population(tree, t)
        @test popsize(again) == popsize(pop)
        @test again._next_id == maximum(tree.id)
        simulate!(again, grow_block(Nmax = 400), MersenneTwister(6))
        @test popsize(again) >= 400 && issorted(again.tree.id) && allunique(again.tree.id)
        save_tree(joinpath(dir, "t.nmet"), again.tree)               # a whole LineageTree
        @test popsize(single_root(first(load_tree(joinpath(dir, "t.nmet"))))) == popsize(again)
    end
end
