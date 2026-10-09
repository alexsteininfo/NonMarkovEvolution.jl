# Golden-value regression tests.
#
# These pin the exact RNG consumption of the default code path (no hooks, no carried
# event queue), so that no refactor perturbs the random stream by a single draw: stored
# simulation results must stay reproducible from their seeds.
#
# The literals were captured on Julia 1.12.1 with Distributions 0.25.131, and still pass
# on Julia 1.12.7 with Distributions 0.25.130. They depend on the Julia RNG and the
# Gamma/Poisson/Exponential samplers, so a Julia or Distributions upgrade may
# legitimately change them (regenerate deliberately in that case); no edit to this
# package should. The stream differs on Julia 1.10, so the exact values are
# asserted on Julia >= 1.11 only.

@testset "golden: neutral configuration (Gamma k=5, d=0.5, ν=2.0, Dirac mutations)" begin
    rng = MersenneTwister(20260901)
    pop = initialize_population(fitness_init = 1.0)
    block = NonMarkovBlock(
        birth_dist     = f -> Gamma(5.0, 1.0 / (5.0 * f)),
        death_dist     = f -> Gamma(5.0, 1.0 / (5.0 * 0.5)),
        stopfunction   = pop -> popsize(pop) >= 100,
        effect_dist    = Dirac(0.0),
        fitness_update = (f, δ) -> f,
        ν              = 2.0,
    )
    simulate!(pop, block, rng)

    @test popsize(pop) == 100
    @test pop._next_id == 231
    @test all(f === 1.0 for f in fitness_per_cell(pop))
    if VERSION >= v"1.11"
        @test pop.t === 7.107669723464197
        @test sum(mutations_per_cell(pop)) == 1810
        @test sort(mutations_per_cell(pop)) == [
            8, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14,
            14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17,
            17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
            18, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 22,
            22, 23, 23, 23, 23, 23, 24, 24, 24, 25, 25, 25, 25, 25, 27, 27, 28, 29, 29, 29]
    else
        @test_skip pop.t === 7.107669723464197
        @test_skip sum(mutations_per_cell(pop)) == 1810
        @test_skip sort(mutations_per_cell(pop)) == [
            8, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14,
            14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17,
            17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
            18, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 22,
            22, 23, 23, 23, 23, 23, 24, 24, 24, 25, 25, 25, 25, 25, 27, 27, 28, 29, 29, 29]
    end
end

@testset "golden: mutation configuration (Exponential mutations, ν=0.5)" begin
    rng = MersenneTwister(20260902)
    pop = initialize_population(fitness_init = 1.0)
    block = NonMarkovBlock(
        birth_dist     = f -> Gamma(5.0, 1.0 / (5.0 * f)),
        death_dist     = f -> Gamma(5.0, 10.0 / 5.0),
        stopfunction   = pop -> popsize(pop) >= 100,
        effect_dist    = Exponential(0.05),
        fitness_update = (f, δ) -> f + δ,
        ν              = 0.5,
    )
    simulate!(pop, block, rng)

    fits = sort(fitness_per_cell(pop))
    @test popsize(pop) == 100
    @test pop._next_id == 199
    # Compared element-wise rather than via a sum: floating-point reduction order can
    # vary with compilation context, which would make a summed fixture flaky for reasons
    # that have nothing to do with the RNG stream this test exists to pin.
    expected_fits = [
        1.0001266022760236, 1.0001266022760236, 1.0173721366533395, 1.0173721366533395,
        1.0173721366533395, 1.0173721366533395, 1.0173721366533395, 1.0190965311651232,
        1.023007818659848, 1.0321936585623708, 1.0329927152084148, 1.0388025153485747,
        1.0388025153485747, 1.0388025153485747, 1.0400383632824324, 1.0400383632824324,
        1.0400383632824324, 1.0420261757752447, 1.0448825408323026, 1.0448825408323026,
        1.0450035895586651, 1.0469523139059436, 1.0469523139059436, 1.0486572247160126,
        1.057787191828064, 1.0580238639380668, 1.0580238639380668, 1.0654951736099845,
        1.0660224240168164, 1.0660224240168164, 1.0660224240168164, 1.0664538829911816,
        1.0664538829911816, 1.0707588842463762, 1.0779503899605023, 1.0788364891479538,
        1.0801409087314737, 1.0801409087314737, 1.0866084725656429, 1.0876280351871666,
        1.0880975692295145, 1.0905008538680716, 1.0905008538680716, 1.0960773809401945,
        1.103296412625665, 1.103477984409174, 1.1036207573638228, 1.1145581876483897,
        1.117471075024041, 1.1222649879183508, 1.1222649879183508, 1.1254611189916761,
        1.1321503590972686, 1.1330396387851651, 1.1330396387851651, 1.1330396387851651,
        1.1330396387851651, 1.1331690614685455, 1.1404602765101808, 1.142333445119333,
        1.144775843862109, 1.1479497073424074, 1.1491161596971162, 1.155605256345507,
        1.1562376791725666, 1.1573928837567184, 1.1628497215491826, 1.173980487065382,
        1.173980487065382, 1.1756626538496635, 1.1764326308459072, 1.195937810806653,
        1.2084680700807098, 1.2084680700807098, 1.2199965748730557, 1.2238152054522857,
        1.2350566797125822, 1.235444837373462, 1.2396013845104081, 1.2559894817562238,
        1.265503260178321, 1.2720785004925033, 1.2788083539966821, 1.2798944216105004,
        1.2806341604374558, 1.2806341604374558, 1.2891926852680935, 1.2917394070590829,
        1.2917394070590829, 1.300660505435859, 1.3260656404621207, 1.3260656404621207,
        1.3260656404621207, 1.3303535769657413, 1.3417114444773242, 1.3609592843479565,
        1.3747103628822632, 1.4052596476891026, 1.4825800163278604, 1.5009707683942342]
    if VERSION >= v"1.11"
        @test pop.t === 6.4000753169971
        @test sum(mutations_per_cell(pop)) == 388
        @test fits == expected_fits
    else
        @test_skip pop.t === 6.4000753169971
        @test_skip sum(mutations_per_cell(pop)) == 388
        @test_skip fits == expected_fits
    end
end

@testset "an explicit on_division = nothing changes nothing" begin
    # Belt and braces: passing the hook explicitly as `nothing` must be identical to
    # omitting it, i.e. the branch itself consumes no randomness.
    function _run(; hooks::Bool)
        rng = MersenneTwister(20260901)
        pop = initialize_population(fitness_init = 1.0)
        kwargs = (
            birth_dist     = f -> Gamma(5.0, 1.0 / (5.0 * f)),
            death_dist     = f -> Gamma(5.0, 1.0 / (5.0 * 0.5)),
            stopfunction   = pop -> popsize(pop) >= 100,
            effect_dist    = Dirac(0.0),
            fitness_update = (f, δ) -> f,
            ν              = 2.0,
        )
        block = hooks ? NonMarkovBlock(; kwargs..., on_division = nothing, on_restart = nothing) :
                        NonMarkovBlock(; kwargs...)
        simulate!(pop, block, rng)
        return pop
    end
    a, b = _run(hooks = false), _run(hooks = true)
    @test a.t === b.t
    @test popsize(a) == popsize(b)
    @test sort(mutations_per_cell(a)) == sort(mutations_per_cell(b))
end
