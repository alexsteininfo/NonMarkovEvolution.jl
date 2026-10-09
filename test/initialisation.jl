@testset "single-cell initialization" begin
    pop = initialize_population(fitness_init = 2.0)
    @test popsize(pop) == 1
    @test pop.t == 0.0
    @test pop._next_id == 1
    cell = only(alive_cells(pop)).data
    @test cell.id == 1
    @test cell.fitness ≈ 2.0
    @test cell.mutations == 0
    @test cell.total_mutations == 0
    @test cell.birthtime ≈ 0.0
end

@testset "initialize_population accepts integer arguments" begin
    pop = initialize_population(3; fitness_init = 2, time = 1)
    @test all(c.data.fitness === 2.0 for c in alive_cells(pop))
    @test pop.t === 1.0
    @test_throws ArgumentError initialize_population(0)
end

@testset "N-cell initialization" begin
    pop = initialize_population(5; fitness_init = 1.5)
    @test popsize(pop) == 5
    @test pop._next_id == 5
    for node in alive_cells(pop)
        @test node.data.fitness ≈ 1.5
        @test node.data.mutations == 0
    end
end

@testset "alive_cells length and type" begin
    pop = initialize_population(3)
    cells = alive_cells(pop)
    @test length(cells) == 3
    @test eltype(cells) == CellNode
    @test [c.id for c in cells] == [1, 2, 3]
    @test all(isalive, cells)
end

@testset "default fitness is 1.0" begin
    pop = initialize_population()
    @test only(alive_cells(pop)).fitness ≈ 1.0
end
