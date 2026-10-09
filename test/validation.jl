# Statistical validation of the core algorithm against closed-form results. These test
# the model itself rather than the code's bookkeeping. Tolerances are several standard
# errors wide, and the seeds are fixed, so the tests are deterministic.

@testset "extinction probability of the Markov process is d/b" begin
    b, d, runs = 1.0, 0.5, 2000
    block = NonMarkovBlock(
        birth_dist = f -> Exponential(1 / b), death_dist = f -> Exponential(1 / d),
        stopfunction = pop -> popsize(pop) >= 40,    # extinction from 40 cells ≈ 0.5^40
        effect_dist = Dirac(0.0), fitness_update = (f, δ) -> f, ν = 0.0)
    rng = MersenneTwister(2026)
    extinct = count(1:runs) do _
        pop = initialize_population()
        simulate!(pop, block, rng)
        popsize(pop) == 0
    end
    # Standard error sqrt(0.25 / 2000) ≈ 0.011.
    @test abs(extinct / runs - d / b) < 0.045
end

@testset "competing risks: extinction probability (1-p)/p for Gamma timing" begin
    # Division ~ Gamma(3, 1/3) (mean 1), death ~ Exponential(2): p = P(T_div < T_die)
    # = E[exp(-T_div / 2)] = (1 + (1/3)/2)^(-3) = (7/6)^(-3).
    p = (7 / 6)^(-3)
    block = NonMarkovBlock(
        birth_dist = f -> Gamma(3.0, 1 / 3), death_dist = f -> Exponential(2.0),
        stopfunction = pop -> popsize(pop) >= 40, effect_dist = Dirac(0.0),
        fitness_update = (f, δ) -> f, ν = 0.0)
    rng, runs = MersenneTwister(7), 2000
    extinct = count(1:runs) do _
        pop = initialize_population()
        simulate!(pop, block, rng)
        popsize(pop) == 0
    end
    @test abs(extinct / runs - (1 - p) / p) < 0.045
end

@testset "Malthusian rate of Gamma pure birth is k·b·(2^(1/k) − 1)" begin
    # At fixed mean division time 1/b, more variable cycles grow faster: r falls from b
    # at k = 1 towards b·ln 2. Measured from the time to grow from 2 000 to 20 000 cells.
    for k in (1.0, 5.0)
        b = 1.0
        r_theory = k * b * (2^(1 / k) - 1)
        block = NonMarkovBlock(
            birth_dist = f -> Gamma(k, 1 / (k * b * f)), death_dist = f -> Dirac(Inf),
            stopfunction = pop -> popsize(pop) >= 20_000, effect_dist = Dirac(0.0),
            fitness_update = (f, δ) -> f, ν = 0.0)
        estimates = map(1:4) do seed
            acc = MeasurementAccumulator(MeasurementSpec(
                snapshot_triggers = [AtPopSize(2_000), AtPopSize(20_000)],
                snapshot_stats = AbstractStatistic[]))
            simulate!(initialize_population(), block, MersenneTwister(seed);
                      accumulator = acc)
            t1, t2 = (s.t for s in finalize_measurements(acc).snapshots)
            log(10) / (t2 - t1)
        end
        @test abs(mean(estimates) / r_theory - 1) < 0.03
    end
end

@testset "every daughter draws Poisson(ν) mutations" begin
    # Pure birth keeps every daughter in the tree, so the non-root nodes are an unbiased
    # sample of per-daughter draws: mean and variance should both be ν.
    pop = initialize_population()
    ν   = 0.7
    simulate!(pop, NonMarkovBlock(
        birth_dist = f -> Gamma(5.0, 0.2), death_dist = f -> Dirac(Inf),
        stopfunction = p -> popsize(p) >= 5_000, effect_dist = Dirac(0.0),
        fitness_update = (f, δ) -> f, ν = ν), MersenneTwister(5))
    root = single_root(pop)
    js   = [Float64(n.data.mutations) for n in PreOrderDFS(root) if n !== root]
    @test length(js) == 2 * (5_000 - 1)
    @test abs(mean(js) / ν - 1) < 0.05
    @test abs(var(js) / ν - 1) < 0.08
end
