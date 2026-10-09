# Benchmark: growth from one cell to N cells with fitness-changing mutations, for several N
# and four modes, the cost of the post-hoc tree statistics on the result, and a parallel
# throughput mode (several simulations at once, one per thread).
#
# Inputs : population sizes on the command line (optional).
# Outputs: timing and memory tables printed to stdout; nothing is written to disk.
# Run:
#   julia --project=. benchmark/benchmark.jl                     # N = 10^3, 10^4, 10^5, 10^6, 10^7
#   julia --project=. benchmark/benchmark.jl 1000000 10000000    # chosen sizes
#   julia -t 16 --project=. benchmark/benchmark.jl --parallel 100000   # throughput at N = 10^5
#
# The model is the one of examples/01: Gamma(2) division times with mean 1/f,
# Exponential(20) lifetimes, Poisson(ν = 0.5) mutations per daughter, each adding
# δ ~ Exponential(0.05) to the fitness, restart_on_extinction = true. Modes:
#   gamma            the model as is (queue algorithm)
#   gamma+sizehint   the same, with sizehint!(pop, N) before the run
#   exp/thinning     Exponential division times with mean 1/f (Markov), thinning
#   exp/queue        the same Markov model forced onto the queue algorithm
#
# The sizes are run in increasing order. Each run measures the live memory per cell; a
# larger run whose estimated peak memory exceeds 60% of the currently *free* RAM is skipped
# (with the estimate printed) instead of exhausting a shared machine. Live memory is the
# GC's live-bytes counter (Base.summarysize would itself need many GB for large trees).
# --parallel N: runs 4 × nthreads simulations to N cells, first serially, then
# concurrently (Threads.@threads, one RNG per run), and reports runs per second.
using NonMarkovEvolution, Distributions, Random, Printf

const PARALLEL = "--parallel" in ARGS
const PARALLEL_N = let i = findfirst(==("--parallel"), ARGS)
    isnothing(i) ? 0 :
    i < length(ARGS) ? parse(Int, ARGS[i + 1]) :
    error("--parallel needs a population size, e.g. --parallel 100000")
end
const SIZES = sort(PARALLEL ? [PARALLEL_N] :
                   length(ARGS) >= 1 ? [parse(Int, a) for a in ARGS] :
                   [10^3, 10^4, 10^5, 10^6, 10^7])
const MEMORY_LIMIT = 0.6 * Sys.free_memory()   # bytes available for one run (shared node)
const PEAK_FACTOR = 1.5                          # peak (GC, heap growth) relative to live size

const SHAPE = 2.0                                # Gamma shape: CV of the division time 1/√2
gamma_birth(f) = Gamma(SHAPE, 1.0 / (SHAPE * f))
exp_birth(f) = Exponential(1.0 / f)
block(N; birth = gamma_birth, algorithm = :auto) = NonMarkovBlock(
    birth_dist     = birth,
    death_dist     = f -> Exponential(20.0),
    stopfunction   = pop -> popsize(pop) >= N,
    effect_dist    = Exponential(0.05),
    fitness_update = (f, δ) -> f + δ,
    ν              = 0.5,
    restart_on_extinction = true,
    algorithm      = algorithm,
)

# mode name => (block constructor, sizehint)
const MODES = [
    "gamma"          => (N -> block(N), false),
    "gamma+sizehint" => (N -> block(N), true),
    "exp/thinning"   => (N -> block(N; birth = exp_birth), false),
    "exp/queue"      => (N -> block(N; birth = exp_birth, algorithm = :queue), false),
]

# warm-up with the same block types as the measured runs (compilation is not timed)
for (_, (mk, _)) in MODES
    simulate!(initialize_population(), mk(100), Xoshiro(0))
end

gb(x) = x / 2^30

# One measured run, inside a function so that nothing (population, @timed result) survives it.
function measure(mk, hint, N)
    GC.gc()
    baseline = Base.gc_live_bytes()
    pop = initialize_population()
    hint && sizehint!(pop, N)
    stats = @timed simulate!(pop, mk(N), Xoshiro(1))
    GC.gc()
    live = Base.gc_live_bytes() - baseline
    return (time = stats.time, gctime = stats.gctime, alloc = stats.bytes, live = live,
            n = popsize(pop))
end

if PARALLEL
    run_one(N, seed) = popsize(simulate!(initialize_population(), block(N), Xoshiro(seed)))
    nt = Threads.nthreads()
    nruns = 4 * nt
    run_one(PARALLEL_N, 0)                                   # warm-up at the measured size
    GC.gc()
    tserial = @elapsed for r in 1:nruns
        run_one(PARALLEL_N, r)
    end
    GC.gc()
    tpar = @elapsed Threads.@threads for r in 1:nruns
        run_one(PARALLEL_N, r)
    end
    @printf("parallel throughput: N = %d, %d runs, %d threads (CPU threads: %d)\n",
            PARALLEL_N, nruns, nt, Sys.CPU_THREADS)
    @printf("  serial:   %8.2f s  → %8.2f runs/s\n", tserial, nruns / tserial)
    @printf("  parallel: %8.2f s  → %8.2f runs/s  (speedup %.1f× on %d threads, efficiency %.0f%%)\n",
            tpar, nruns / tpar, tserial / tpar, nt, 100 * tserial / tpar / nt)
    nt == 1 && println("  (start Julia with -t N to use N threads)")
else

@printf("Julia %s, %s, %d CPU threads\n", VERSION, Sys.cpu_info()[1].model, Sys.CPU_THREADS)
@printf("RAM: %.1f GB total, %.1f GB free, limit per run: %.1f GB\n\n",
        gb(Sys.total_memory()), gb(Sys.free_memory()), gb(MEMORY_LIMIT))
@printf("%-15s %12s %10s %8s %12s %12s %10s\n", "mode", "N", "time (s)", "GC (%)",
        "alloc (GB)", "live (GB)", "ns/cell")

for (mode, (mk, hint)) in MODES
    bytes_per_cell = nothing                     # measured on the previous run
    for N in SIZES
        if !isnothing(bytes_per_cell)
            estimate = PEAK_FACTOR * bytes_per_cell * N
            if estimate > MEMORY_LIMIT
                @printf("%-15s %12d   skipped: needs ≈ %.0f GB (limit %.0f GB)\n", mode, N,
                        gb(estimate), gb(MEMORY_LIMIT))
                continue
            end
        end
        r = measure(mk, hint, N)
        bytes_per_cell = r.live / r.n            # extrapolate from the latest (largest) run
        @printf("%-15s %12d %10.2f %8.0f %12.2f %12.2f %10.0f\n", mode, N, r.time,
                100 * r.gctime / r.time, gb(r.alloc), gb(r.live), 1e9 * r.time / N)
    end
end

# Cost of the post-hoc statistics on one grown population (each called once to compile,
# then timed). Inside a function: no global-scope overhead.
function time_statistics(pop)
    root = single_root(pop)
    tasks = [
        "mutations_per_cell"     => () -> mutations_per_cell(pop),
        "site_frequency_spectrum" => () -> site_frequency_spectrum(pop),
        "branch_spectrum"        => () -> branch_spectrum(root),
        "leaf_depths"            => () -> leaf_depths(root),
        "sample_leaves (n=1000)" => () -> sample_leaves(pop, 1000; seed = 1),
    ]
    results = Pair{String, Float64}[]
    for (name, f) in tasks
        f()
        GC.gc()
        push!(results, name => @elapsed f())
    end
    return results
end
GC.gc()
const NSTAT = min(10^6, maximum(SIZES))         # large enough that the tree is not cache-resident
pop = simulate!(initialize_population(), block(NSTAT), Xoshiro(2))
println("\npost-hoc statistics on one population of N = $NSTAT:")
for (name, t) in time_statistics(pop)
    @printf("  %-24s %8.3f s  (%5.0f ns per cell)\n", name, t, 1e9 * t / NSTAT)
end
end
