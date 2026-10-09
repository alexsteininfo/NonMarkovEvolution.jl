# Tests for chaining successive `simulate!` calls on one Population.
#
# `simulate!` carries the queue of already-drawn, not-yet-fired events on the population
# and reuses it. Before that fix it rebuilt the queue from scratch, redrawing each cell's
# next event anchored at its `birthtime` and ignoring the age it had already accumulated
# without an event — which both dropped the age-conditioning and let events land before
# the current `population.t`, running the clock backwards.

function chain_block(; Nmax, ν = 0.5, shape = 5.0)
    NonMarkovBlock(
        birth_dist     = f -> Gamma(shape, 1.0 / (shape * f)),
        death_dist     = f -> Gamma(shape, 10.0 / shape),
        stopfunction   = pop -> popsize(pop) >= Nmax,
        effect_dist    = Exponential(0.05),
        fitness_update = (f, δ) -> f + δ,
        ν              = ν,
    )
end

@testset "chained run is identical to the equivalent single run" begin
    # The carried queue means a chained call resumes on exactly the events the first call
    # had already drawn, and consumes the same RNG draws in the same order as an
    # uninterrupted run. So the two are not merely equal in distribution — they agree
    # bit for bit, per seed. This is the strongest available statement of correctness,
    # and it fails on the pre-fix code.
    for seed in 1:20
        single = initialize_population(fitness_init = 1.0)
        simulate!(single, chain_block(Nmax = 200), MersenneTwister(seed))

        chained = initialize_population(fitness_init = 1.0)
        rng     = MersenneTwister(seed)
        simulate!(chained, chain_block(Nmax = 100), rng)
        @test popsize(chained) >= 100
        simulate!(chained, chain_block(Nmax = 200), rng)

        @test chained.t === single.t
        @test popsize(chained) == popsize(single)
        @test chained._next_id == single._next_id
        @test sort(fitness_per_cell(chained))   == sort(fitness_per_cell(single))
        @test sort(mutations_per_cell(chained)) == sort(mutations_per_cell(single))
    end
end

@testset "time never runs backwards across a chain boundary" begin
    # The defining symptom of the old behaviour. On the pre-fix code the second call
    # redrew ~100 aged cells anchored at their birthtimes, so a dozen or more phase-2
    # events fired at times *earlier* than the boundary and `population.t` jumped
    # backwards exactly once, at the boundary. Both are checked here.
    times   = Float64[]
    watcher = function (pop, parent, d1, d2)
        push!(times, pop.t)
        return nothing
    end
    function watched(Nmax)
        b = chain_block(Nmax = Nmax)
        NonMarkovBlock(
            birth_dist = b.birth_dist, death_dist = b.death_dist,
            stopfunction = b.stopfunction, effect_dist = b.effect_dist,
            fitness_update = b.fitness_update, ν = b.ν, on_division = watcher,
        )
    end

    rng = MersenneTwister(909)
    pop = initialize_population(fitness_init = 1.0)
    simulate!(pop, watched(100), rng)
    boundary = pop.t
    n_phase1 = length(times)
    simulate!(pop, watched(200), rng)
    phase2 = times[(n_phase1 + 1):end]

    @test boundary > 1.0
    @test !isempty(phase2)
    @test issorted(times)                          # no inversion anywhere, boundary included
    @test all(>=(boundary), phase2)                # phase 2 never fires in phase 1's past
    @test minimum(phase2) >= boundary
    @test pop.t >= boundary
end

@testset "tree stays consistent across a chain boundary" begin
    rng = MersenneTwister(2718)
    pop = initialize_population(fitness_init = 1.0)
    simulate!(pop, chain_block(Nmax = 100), rng)
    simulate!(pop, chain_block(Nmax = 250), rng)

    # Structural invariants. Note these hold by construction even on the pre-fix code
    # (an event time is always its cell's birthtime plus a positive draw), so this is a
    # sanity check rather than the test that catches the chaining bug — that one is
    # "time never runs backwards" above.
    root = single_root(pop)
    @test !isnothing(root)
    for node in PreOrderDFS(root)
        if !isnothing(node.parent)
            @test node.data.birthtime >= node.parent.data.birthtime
        end
        @test cell_lifetime(node, pop.t) >= 0.0
    end
    @test all(c.data.birthtime <= pop.t for c in alive_cells(pop))
end

@testset "a fresh accumulator on a chained call does not back-fill from t=0" begin
    rng  = MersenneTwister(31415)
    pop  = initialize_population(fitness_init = 1.0)
    spec = MeasurementSpec(trajectory_dt = 0.5,
                           snapshot_triggers = [AtEnd()],
                           snapshot_stats    = [FitnessDistribution()])

    simulate!(pop, chain_block(Nmax = 100), rng; accumulator = MeasurementAccumulator(spec))
    boundary = pop.t
    @test boundary > 1.0

    acc2 = MeasurementAccumulator(spec)
    simulate!(pop, chain_block(Nmax = 250), rng; accumulator = acc2)
    m2 = finalize_measurements(acc2)

    # Every recorded point belongs to the second phase.
    @test !isempty(m2.trajectory)
    @test all(tp.t >= boundary for tp in m2.trajectory)
    # And there are only as many as the phase-2 span allows, not ~boundary/dt extra.
    @test length(m2.trajectory) <= ceil(Int, (pop.t - boundary) / 0.5) + 2
end

@testset "an accumulator reused across the chain keeps recording continuously" begin
    rng  = MersenneTwister(1618)
    pop  = initialize_population(fitness_init = 1.0)
    spec = MeasurementSpec(trajectory_dt = 0.5,
                           snapshot_triggers = [AtEnd()],
                           snapshot_stats    = [FitnessDistribution()])
    acc  = MeasurementAccumulator(spec)

    simulate!(pop, chain_block(Nmax = 100), rng; accumulator = acc)
    n1 = length(acc.trajectory_points)
    simulate!(pop, chain_block(Nmax = 250), rng; accumulator = acc)
    m = finalize_measurements(acc)

    @test length(m.trajectory) > n1
    @test issorted([tp.t for tp in m.trajectory])
    @test m.trajectory[1].t < 1.0        # phase 1 still recorded from the start
end

@testset "reset_schedule! and the queue-consistency guard" begin
    rng = MersenneTwister(5772)
    pop = initialize_population(fitness_init = 1.0)
    simulate!(pop, chain_block(Nmax = 60), rng)
    @test has_pending_schedule(pop)
    @test length(pop._pending) == popsize(pop)      # one event per cell

    # Removing a cell by hand desynchronises the queue and must be caught, not silently
    # simulated with a stale event pointing at a dead node.
    victim = first(keys(pop.cells))
    delete!(pop.cells, victim)
    @test_throws ArgumentError simulate!(pop, chain_block(Nmax = 120), rng)

    # reset_schedule! is the documented recovery.
    @test reset_schedule!(pop) === pop
    @test !has_pending_schedule(pop)
    simulate!(pop, chain_block(Nmax = 120), rng)
    @test popsize(pop) >= 120
end

@testset "a fresh population carries no queue until it is simulated" begin
    @test !has_pending_schedule(initialize_population())
    @test !has_pending_schedule(initialize_population(5))
end

# ── Fresh schedules are conditioned on each cell's age ────────────────────────

@testset "a redrawn schedule never fires in the past" begin
    # Cells born at t = 0 but scheduled at t = 2: each must be conditioned on having had
    # no event before t = 2, so no event may land earlier.
    times = Float64[]
    block = NonMarkovBlock(
        birth_dist     = f -> Gamma(5.0, 1.0 / (5.0 * f)),
        death_dist     = f -> Exponential(4.0),
        stopfunction   = pop -> popsize(pop) >= 200,
        effect_dist    = Dirac(0.0),
        fitness_update = (f, δ) -> f,
        ν              = 0.0,
        on_division    = (pop, parent, d1, d2) -> (push!(times, pop.t); nothing),
    )
    pop = initialize_population(50)
    pop.t = 2.0
    simulate!(pop, block, MersenneTwister(12))
    @test !isempty(times)
    @test minimum(times) >= 2.0
    @test issorted(times)
end

@testset "conditioning is exact: residual time of an aged exponential cell" begin
    # Memorylessness: an Exponential(1) cell of age 3 has residual time ~ Exponential(1),
    # so the conditioned event time minus 3 must have mean ≈ 1.
    heap = DataStructures.BinaryMinHeap{NonMarkovEvolution.CellEvent}()
    node = rootnode(1, 0.0, 0)
    block = NonMarkovBlock(birth_dist = f -> Exponential(1.0), death_dist = f -> Dirac(Inf),
        stopfunction = pop -> false, effect_dist = Dirac(0.0),
        fitness_update = (f, δ) -> f, ν = 0.0)
    rng = MersenneTwister(3)
    residuals = Float64[]
    for _ in 1:4000
        NonMarkovEvolution.schedule_cell!(heap, node, block, rng, 3.0)
        push!(residuals, pop!(heap).time - 3.0)
    end
    @test all(>=(0), residuals)
    @test abs(mean(residuals) - 1.0) < 0.06
end

@testset "a cell too old for its waiting times raises instead of hanging" begin
    pop = initialize_population()
    pop.t = 5.0                                   # born at 0, would have divided at 1
    block = NonMarkovBlock(birth_dist = f -> Dirac(1.0), death_dist = f -> Dirac(Inf),
        stopfunction = pop -> popsize(pop) >= 4, effect_dist = Dirac(0.0),
        fitness_update = (f, δ) -> f, ν = 0.0)
    @test_throws ErrorException simulate!(pop, block, MersenneTwister(1))
end

@testset "restart_on_extinction now works on a chained block" begin
    times = Float64[]
    watcher = (pop, parent, d1, d2) -> (push!(times, pop.t); nothing)
    grow = NonMarkovBlock(birth_dist = f -> Gamma(5.0, 0.2), death_dist = f -> Dirac(Inf),
        stopfunction = pop -> popsize(pop) >= 3, effect_dist = Dirac(0.0),
        fitness_update = (f, δ) -> f, ν = 0.0)
    risky = NonMarkovBlock(birth_dist = f -> Exponential(1.0),
        death_dist = f -> Exponential(0.9),       # near-critical: extinction is common
        stopfunction = pop -> popsize(pop) >= 40, effect_dist = Dirac(0.0),
        fitness_update = (f, δ) -> f, ν = 0.0, restart_on_extinction = true,
        on_division = watcher)
    pop = initialize_population()
    rng = MersenneTwister(21)
    simulate!(pop, grow, rng)
    boundary = pop.t
    simulate!(pop, risky, rng)
    @test popsize(pop) >= 40
    @test all(>=(boundary), times)           # restored cells never fire before the boundary
end
