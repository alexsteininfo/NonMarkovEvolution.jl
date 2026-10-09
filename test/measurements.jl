# Measurement recording: validation, exact timing of AtTime and trajectory points,
# trigger semantics, and behaviour across chains and restarts.

@isdefined(fixture_tree) || include("fixtures.jl")

# Deterministic pure birth: every cell divides exactly 1 time unit after its birth, so
# the size is 2^floor(t) and every observable has a closed form.
clock_block(Nmax) = NonMarkovBlock(
    birth_dist = f -> Dirac(1.0), death_dist = f -> Dirac(Inf),
    stopfunction = pop -> popsize(pop) >= Nmax, effect_dist = Dirac(0.0),
    fitness_update = (f, δ) -> f, ν = 0.0)

@testset "MeasurementSpec and triggers validate their inputs" begin
    @test_throws ArgumentError MeasurementSpec(trajectory_dt = 0.0)   # used to hang
    @test_throws ArgumentError MeasurementSpec(trajectory_dt = -1.0)
    @test_throws ArgumentError MeasurementSpec(trajectory_dt = NaN)
    @test MeasurementSpec(trajectory_dt = Inf).trajectory_dt == Inf
    @test_throws ArgumentError AtPopSize(0)
    @test_throws ArgumentError AtTime(NaN)
end

@testset "trajectory points are the exact state at their grid time" begin
    acc = MeasurementAccumulator(MeasurementSpec(trajectory_dt = 0.5,
                                                 snapshot_triggers = AbstractTrigger[]))
    pop = initialize_population()
    simulate!(pop, clock_block(4), MersenneTwister(1); accumulator = acc)
    tr = finalize_measurements(acc).trajectory
    # The state is right-continuous: the divisions at t = 1 count at t = 1.
    @test [p.t for p in tr]       == [0.0, 0.5, 1.0, 1.5, 2.0]
    @test [p.N_total for p in tr] == [1, 1, 2, 2, 4]
    @test pop.t == 2.0
end

@testset "AtTime records the state exactly at t, labelled t" begin
    spec = MeasurementSpec(snapshot_triggers = [AtTime(0.0), AtTime(1.5), AtTime(9.0)],
                           snapshot_stats = [FitnessDistribution(), MutationsPerCell()])
    acc  = MeasurementAccumulator(spec)
    simulate!(initialize_population(), clock_block(8), MersenneTwister(1); accumulator = acc)
    snaps = finalize_measurements(acc).snapshots
    @test [s.t for s in snaps] == [0.0, 1.5]            # AtTime(9.0) is never reached
    @test [length(s[:fitness]) for s in snaps] == [1, 2]
end

@testset "AtPopSize is checked before the first event" begin
    acc = MeasurementAccumulator(MeasurementSpec(snapshot_triggers = [AtPopSize(5)]))
    simulate!(initialize_population(5), clock_block(10), MersenneTwister(1);
              accumulator = acc)
    snaps = finalize_measurements(acc).snapshots
    @test length(snaps) == 1
    @test snaps[1].t == 0.0
    @test length(snaps[1][:fitness]) == 5
end

@testset "trajectory statistics match a direct computation" begin
    spec = MeasurementSpec(trajectory_dt = 1.0, snapshot_triggers = [AtEnd()],
                           snapshot_stats = [FitnessDistribution(), MutationsPerCell()])
    acc  = MeasurementAccumulator(spec)
    pop  = initialize_population()
    simulate!(pop, grow_block(Nmax = 150, ν = 1.0), MersenneTwister(4); accumulator = acc)
    m    = finalize_measurements(acc)
    last_pt, endsnap = m.trajectory[end], only(m.snapshots)
    if last_pt.t == pop.t           # only then is the last point the final state
        @test last_pt.mean_fitness ≈ mean(endsnap[:fitness])
    end
    @test endsnap[:mutations] == mutations_per_cell(pop)
    @test endsnap[:fitness] == fitness_per_cell(pop)     # co-indexed by id
    @test all(p -> p.N_total >= 1, m.trajectory)
    @test issorted([p.t for p in m.trajectory])
end

@testset "a carried accumulator: AtEnd per call, earlier records survive a restart" begin
    spec = MeasurementSpec(trajectory_dt = 0.25, snapshot_triggers = [AtEnd()],
                           snapshot_stats = [FitnessDistribution()])
    acc  = MeasurementAccumulator(spec)
    pop  = initialize_population()
    rng  = MersenneTwister(21)
    simulate!(pop, clock_block(4), rng; accumulator = acc)
    n_phase1 = length(acc.trajectory_points)
    boundary = pop.t
    risky = NonMarkovBlock(birth_dist = f -> Exponential(1.0),
        death_dist = f -> Exponential(0.9), stopfunction = p -> popsize(p) >= 30,
        effect_dist = Dirac(0.0), fitness_update = (f, δ) -> f, ν = 0.0,
        restart_on_extinction = true)
    simulate!(pop, risky, rng; accumulator = acc)
    m = finalize_measurements(acc)
    @test length(m.snapshots) == 2                          # one AtEnd per call
    @test count(p -> p.t <= boundary, m.trajectory) == n_phase1
    @test issorted([p.t for p in m.trajectory])
    @test allunique([p.t for p in m.trajectory])
end

@testset "a fresh accumulator on a chained call ignores AtTime in the past" begin
    pop = initialize_population()
    simulate!(pop, clock_block(4), MersenneTwister(1))          # now t = 2
    acc = MeasurementAccumulator(MeasurementSpec(
        snapshot_triggers = [AtTime(0.5), AtTime(2.5)]))
    simulate!(pop, clock_block(16), MersenneTwister(1); accumulator = acc)
    @test [s.t for s in finalize_measurements(acc).snapshots] == [2.5]
end

# User-defined statistics must be declared at top level (a struct cannot be defined
# inside a testset).
struct MeanFitnessStat <: AbstractStatistic end
NonMarkovEvolution.measure(::MeanFitnessStat, pop) = mean(fitness_per_cell(pop))
NonMarkovEvolution.statistic_name(::MeanFitnessStat) = :mean_fitness
struct UnnamedStat <: AbstractStatistic end
NonMarkovEvolution.measure(::UnnamedStat, pop) = popsize(pop)
struct NoMeasureStat <: AbstractStatistic end

@testset "user-defined snapshot statistics" begin
    spec = MeasurementSpec(snapshot_triggers = [AtEnd()],
                           snapshot_stats = [MeanFitnessStat(), UnnamedStat(), SFS()])
    acc  = MeasurementAccumulator(spec)
    pop  = initialize_population()
    simulate!(pop, clock_block(8), MersenneTwister(1); accumulator = acc)
    snap = only(finalize_measurements(acc).snapshots)
    @test snap[:mean_fitness] == 1.0
    @test snap[:UnnamedStat] == 8
    @test snap[:sfs] == site_frequency_spectrum(pop)
    @test Set(keys(snap)) == Set([:mean_fitness, :UnnamedStat, :sfs])
    @test !haskey(snap, :fitness)

    @test_throws ArgumentError MeasurementSpec(snapshot_stats = [NoMeasureStat()])
    @test_throws ArgumentError MeasurementSpec(snapshot_stats = [SFS(), SFS()])
end

@testset "tmax: trajectory and AtTime reach exactly tmax" begin
    block = NonMarkovBlock(birth_dist = f -> Dirac(1.0), death_dist = f -> Dirac(Inf),
        effect_dist = Dirac(0.0), fitness_update = (f, δ) -> f, ν = 0.0, tmax = 2.5)
    acc = MeasurementAccumulator(MeasurementSpec(trajectory_dt = 0.5,
                                                 snapshot_triggers = [AtTime(2.5), AtEnd()]))
    simulate!(initialize_population(), block, MersenneTwister(1); accumulator = acc)
    m = finalize_measurements(acc)
    @test last(m.trajectory).t == 2.5
    @test [p.N_total for p in m.trajectory] == [1, 1, 2, 2, 4, 4]
    @test [s.t for s in m.snapshots] == [2.5, 2.5]
end
