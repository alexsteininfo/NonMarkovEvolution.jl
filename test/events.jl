@testset "CellEvent ordering" begin
    pop  = initialize_population(fitness_init = 1.0)
    node = first(values(pop.cells))
    e1 = NonMarkovEvolution.CellEvent(0.5, node, true)
    e2 = NonMarkovEvolution.CellEvent(1.5, node, false)
    @test e1 < e2
    @test !(e2 < e1)
end

@testset "schedule_cell! event times are after birthtime" begin
    rng  = MersenneTwister(7)
    pop  = initialize_population(fitness_init = 1.0)
    node = first(values(pop.cells))
    block = NonMarkovBlock(
        birth_dist     = f -> Gamma(2.0, 1.0 / f),
        death_dist     = f -> Gamma(2.0, 5.0),
        stopfunction   = pop -> false,
        effect_dist    = Exponential(0.1),
        fitness_update = (f, δ) -> f + δ,
        ν              = 0.0,
    )
    heap = DataStructures.BinaryMinHeap{NonMarkovEvolution.CellEvent}()
    for _ in 1:100
        NonMarkovEvolution.schedule_cell!(heap, node, block, rng)
    end
    while !isempty(heap)
        e = pop!(heap)
        @test e.time >= node.data.birthtime
    end
end

@testset "low death rate → births dominate events" begin
    rng  = MersenneTwister(3)
    pop  = initialize_population(fitness_init = 1.0)
    node = first(values(pop.cells))
    block = NonMarkovBlock(
        birth_dist     = f -> Exponential(1.0 / f),  # mean division time ≈ 1
        death_dist     = f -> Exponential(100.0),    # mean death time = 100 → rare
        stopfunction   = pop -> false,
        effect_dist    = Exponential(0.1),
        fitness_update = (f, δ) -> f + δ,
        ν              = 0.0,
    )
    n_birth = 0
    n_death = 0
    for _ in 1:500
        heap = DataStructures.BinaryMinHeap{NonMarkovEvolution.CellEvent}()
        NonMarkovEvolution.schedule_cell!(heap, node, block, rng)
        e = pop!(heap)
        e.is_division ? (n_birth += 1) : (n_death += 1)
    end
    @test n_birth > 450   # roughly 99% should be births
end

@testset "documented waiting-time parameterisations have the documented mean and CV" begin
    # The table in docs/src/blocks.md promises that every mode has mean 1/(b*f), so that
    # the shape parameter controls *only* the variability and `b` stays comparable across
    # modes. That is easy to get wrong: `Gamma(k, 1/(b*f))` has mean k/(b*f), which would
    # silently rescale time whenever k changed. Pin the recipes here so the docs cannot
    # drift away from what the package actually recommends.
    b, f = 1.3, 1.7
    target = 1 / (b * f)

    k = 5.0
    α = 2.5
    σ = 0.4
    # mean(Weibull(α, θ)) = θ * Γ(1 + 1/α), so Γ(1 + 1/α) == mean(Weibull(α, 1)).
    # Spelling it this way keeps the docs' recipe free of a SpecialFunctions dependency.
    Γ₁ = mean(Weibull(α, 1.0))

    modes = (
        ("Dirac",       Dirac(1 / (b * f)),                       0.0),
        ("Exponential", Exponential(1 / (b * f)),                 1.0),
        ("Gamma",       Gamma(k, 1 / (k * b * f)),                1 / sqrt(k)),
        ("Weibull",     Weibull(α, 1 / (b * f * Γ₁)),             std(Weibull(α, 1.0)) / Γ₁),
        ("LogNormal",   LogNormal(-log(b * f) - σ^2 / 2, σ),      sqrt(exp(σ^2) - 1)),
    )

    for (name, dist, cv) in modes
        @test mean(dist) ≈ target                       rtol = 1e-10
        @test std(dist) / mean(dist) ≈ cv               rtol = 1e-10
    end

    # The Gamma row is the default the docs recommend; check the shape really is the only
    # thing k changes, across the range of k the CV table spans.
    for kk in (1.0, 2.0, 5.0, 20.0, 100.0)
        d = Gamma(kk, 1 / (kk * b * f))
        @test mean(d) ≈ target rtol = 1e-10
        @test std(d) / mean(d) ≈ 1 / sqrt(kk) rtol = 1e-10
    end
    @test Gamma(1.0, 1 / (b * f)) == Gamma(1.0, 1 / (1.0 * b * f))   # k = 1 is the exponential

    # `Dirac(Inf)` is the documented idiom for "never dies": it always loses the
    # competing-risks comparison and consumes no random numbers, so a pure-birth run has
    # the same stream as the same model written any other way.
    r1 = MersenneTwister(4); @test rand(r1, Dirac(Inf)) == Inf
    r2 = MersenneTwister(4)
    @test rand(r1) == rand(r2)                     # Dirac consumed nothing
    @test_throws DomainError Exponential(0.0)
end
