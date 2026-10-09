# The exponential fast path (algorithm = :thinning): selection, exactness against theory
# and against the queue algorithm, and the features it shares with the queue loop.

markov(; b = 1.0, d = 0.5, kw...) = NonMarkovBlock(; birth_dist = f -> Exponential(1 / (b * f)),
    death_dist = f -> Exponential(1 / d), effect_dist = Dirac(0.0),
    fitness_update = (f, δ) -> f, ν = 0.0, kw...)

@testset "algorithm selection" begin
    @test markov().algorithm === :thinning                       # :auto, both Exponential
    @test markov(algorithm = :queue).algorithm === :queue
    gam = NonMarkovBlock(birth_dist = f -> Gamma(2.0, 1 / (2f)),
        death_dist = f -> Exponential(2.0), effect_dist = Dirac(0.0),
        fitness_update = (f, δ) -> f, ν = 0.0)
    @test gam.algorithm === :queue
    @test_throws ArgumentError NonMarkovBlock(birth_dist = f -> Gamma(2.0, 1 / (2f)),
        death_dist = f -> Exponential(2.0), effect_dist = Dirac(0.0),
        fitness_update = (f, δ) -> f, ν = 0.0, algorithm = :thinning)
    # Dirac death: :auto keeps the queue (a finite Dirac is not Markov); Dirac(Inf) is
    # accepted when :thinning is asked for.
    puredirac(; kw...) = NonMarkovBlock(; birth_dist = f -> Exponential(1 / f),
        death_dist = f -> Dirac(Inf), effect_dist = Dirac(0.0),
        fitness_update = (f, δ) -> f, ν = 0.0, kw...)
    @test puredirac().algorithm === :queue
    @test puredirac(algorithm = :thinning).algorithm === :thinning
    @test_throws ArgumentError NonMarkovBlock(birth_dist = f -> Exponential(1 / f),
        death_dist = f -> Dirac(3.0), effect_dist = Dirac(0.0),
        fitness_update = (f, δ) -> f, ν = 0.0, algorithm = :thinning)
    @test_throws ArgumentError markov(algorithm = :gillespie)
end

# Mean population size at time T over many runs, for one block.
function mean_size(block, runs, seed; fitness_init = 1.0)
    rng = Xoshiro(seed)
    return mean(1:runs) do _
        pop = initialize_population(fitness_init = fitness_init)
        popsize(simulate!(pop, block, rng))
    end
end

@testset "E N(t) = exp((b − d) t) for the linear birth–death process" begin
    T, runs = 3.0, 4_000
    expected = exp(0.5T)                                   # ≈ 4.48
    sd = sqrt(3 * expected * (expected - 1))                # (b+d)/(b−d) e^{rt}(e^{rt}−1)
    se = sd / sqrt(runs)
    @test abs(mean_size(markov(tmax = T), runs, 1) - expected) < 4se
    @test abs(mean_size(markov(tmax = T, algorithm = :queue), runs, 2) - expected) < 4se
end

@testset "rates follow fitness: pure birth at f = 2 grows at rate 2" begin
    T, runs = 2.0, 4_000
    block = NonMarkovBlock(birth_dist = f -> Exponential(1 / f),
        death_dist = f -> Exponential(Inf), effect_dist = Dirac(0.0),
        fitness_update = (f, δ) -> f, ν = 0.0, tmax = T)
    @test block.algorithm === :thinning
    expected = exp(2T)
    se = sqrt(expected * (expected - 1)) / sqrt(runs)
    @test abs(mean_size(block, runs, 3; fitness_init = 2.0) - expected) < 4se
end

@testset "a rising rate bound: thinning matches the queue algorithm under selection" begin
    # Mutations raise fitness during the run, so the bound R must grow with it. The size
    # cap bounds the heavy upper tail of this self-accelerating growth; it applies to
    # both algorithms alike, so the two capped laws are still equal.
    sel(alg) = NonMarkovBlock(birth_dist = f -> Exponential(1 / f),
        death_dist = f -> Exponential(4.0), effect_dist = Exponential(0.1),
        fitness_update = (f, δ) -> f + δ, ν = 0.5, tmax = 2.5,
        stopfunction = p -> popsize(p) >= 2_000, algorithm = alg)
    runs = 3_000
    sizes(alg, seed) = (rng = Xoshiro(seed);
        [popsize(simulate!(initialize_population(), sel(alg), rng)) for _ in 1:runs])
    a, q = sizes(:thinning, 4), sizes(:queue, 5)
    se = sqrt(var(a) / runs + var(q) / runs)
    @test abs(mean(a) - mean(q)) < 4se
    @test mean(a) > exp(0.75 * 2.5)          # faster than without selection (r = 0.75)
end

@testset "tmax, hooks, restarts and chaining on the thinning path" begin
    pop = simulate!(initialize_population(), markov(tmax = 2.0), Xoshiro(1))
    @test pop.t == 2.0 || popsize(pop) == 0
    @test !has_pending_schedule(pop)

    # A hook that sets a daughter's fitness to 0 stops her dividing: her birth rate is 0.
    frozen = CellNode[]
    hook = function (pop, parent, d1, d2)
        length(frozen) < 3 && (set_fitness!(d1, 0.0); push!(frozen, d1))
        return nothing
    end
    blk = NonMarkovBlock(birth_dist = f -> Exponential(1 / f), death_dist = f -> Exponential(Inf),
        effect_dist = Dirac(0.0), fitness_update = (f, δ) -> f, ν = 0.0,
        stopfunction = p -> popsize(p) >= 500, on_division = hook)
    pop = simulate!(initialize_population(), blk, Xoshiro(2))
    @test length(frozen) == 3
    @test all(n -> isalive(n) && n.fitness == 0.0, frozen)

    restarts = Ref(0)
    risky = markov(d = 0.9, stopfunction = p -> popsize(p) >= 50, restart_on_extinction = true,
                   on_restart = p -> (restarts[] += 1; nothing))
    pops = [simulate!(initialize_population(), risky, Xoshiro(s)) for s in 1:10]
    @test all(p -> popsize(p) >= 50, pops)
    @test restarts[] >= 1

    # thinning → queue → thinning: time never runs backwards and targets are reached.
    # A restart rolls the clock back to the block's start, so `on_restart` drops what
    # the failed attempt recorded.
    times = Float64[]
    mark  = Ref(0)
    watch = (pop, p, d1, d2) -> (push!(times, pop.t); nothing)
    undo  = pop -> (resize!(times, mark[]); nothing)
    gam = NonMarkovBlock(birth_dist = f -> Gamma(5.0, 1 / (5f)), death_dist = f -> Exponential(4.0),
        effect_dist = Exponential(0.05), fitness_update = (f, δ) -> f + δ, ν = 0.5,
        stopfunction = p -> popsize(p) >= 400, on_division = watch,
        restart_on_extinction = true, on_restart = undo)
    rng = Xoshiro(3)
    pop = simulate!(initialize_population(), markov(stopfunction = p -> popsize(p) >= 100,
        on_division = watch, restart_on_extinction = true, on_restart = undo), rng)
    @test popsize(pop) >= 100
    mark[] = length(times)
    simulate!(pop, gam, rng)
    @test popsize(pop) >= 400 && has_pending_schedule(pop)
    mark[] = length(times)
    simulate!(pop, markov(stopfunction = p -> popsize(p) >= 800, on_division = watch,
                          restart_on_extinction = true, on_restart = undo), rng)
    @test popsize(pop) >= 800 && !has_pending_schedule(pop)
    @test issorted(times)
    root = single_root(pop)
    @test Set(l.id for l in Leaves(root)) == Set(c.id for c in alive_cells(pop))
end

@testset "measurements on the thinning path" begin
    acc = MeasurementAccumulator(MeasurementSpec(trajectory_dt = 0.5,
        snapshot_triggers = [AtPopSize(100), AtEnd()],
        snapshot_stats = [FitnessDistribution(), SFS()]))
    pop = simulate!(initialize_population(),
                    markov(d = 0.2, stopfunction = p -> popsize(p) >= 300,
                           restart_on_extinction = true), Xoshiro(4); accumulator = acc)
    m = finalize_measurements(acc)
    @test [typeof(s.trigger) for s in m.snapshots] == [AtPopSize, AtEnd]
    @test length(m.snapshots[1][:fitness]) == 100
    @test m.snapshots[2][:sfs] == site_frequency_spectrum(pop)
    @test issorted([p.t for p in m.trajectory])
    @test all(p -> p.N_total >= 1, m.trajectory)
end
