# Performance

`benchmark/benchmark.jl` times growth from one cell to ``N`` cells, the post-hoc
statistics on the grown tree, and the throughput of many independent simulations run in
parallel. The model is the one of `examples/01_SingleCellExpansion.jl`:

- division time ``\mathrm{Gamma}(2)`` with mean ``1/f``, lifetime ``\mathrm{Exponential}(20)``;
- ``\mathrm{Poisson}(ν = 0.5)`` mutations per daughter, each adding
  ``δ \sim \mathrm{Exponential}(0.05)`` to the fitness;
- `restart_on_extinction = true`, so a run's time includes any failed attempts.

## Running it

```bash
julia --project=. benchmark/benchmark.jl                      # N = 10³ … 10⁷
julia --project=. benchmark/benchmark.jl 1000000 10000000     # chosen sizes
julia -t 16 --project=. benchmark/benchmark.jl --parallel 100000
bash benchmark/run_benchmarks.sh                              # both, output kept in a log
```

- Compilation is excluded: a warm-up run with the same block type comes first.
- **Live memory** is the GC's live-bytes counter after the run and a full collection,
  minus the baseline before it. It covers the survivors' tree, the cell dictionary and
  the carried event queue.
- Sizes run in increasing order. A size whose estimated peak memory (1.5 × the live bytes
  per cell of the previous run, times ``N``) exceeds 60% of the currently *free* RAM is
  skipped, which protects shared nodes.
- `--parallel N` runs ``4 ×`` `nthreads` simulations to ``N`` cells, first one after the
  other and then with `Threads.@threads` (one RNG per run), and reports runs per second.
  Every division allocates two tree nodes, so concurrent runs share the garbage
  collector, which can limit the speed-up.

## Results

Measured on the Hyperion cluster (node `iribhm-himm01`: 2 × 28-core AMD EPYC Rome, 1.5 TB
RAM, shared and unscheduled) with Julia 1.12.7, on 9 October 2026, single samples.

| ``N`` | time | GC share | live memory | ns per cell |
|:---|---:|---:|---:|---:|
| 10³ | < 0.01 s | 0% | < 10 MB | 665 |
| 10⁴ | < 0.01 s | 0% | < 10 MB | 495 |
| 10⁵ | 0.08 s | 16% | ≈ 20 MB | 793 |
| 10⁶ | 2.3 s | 42% | ≈ 0.22 GB | 2 258 |
| 10⁷ | 40 s | 51% | ≈ 2.3 GB | 4 049 |

- **Up to 10⁵ cells a run costs about 0.5–0.8 µs per cell.** From 10⁶ on the cost per
  cell rises, to about 2.3 µs at 10⁶ and 4 µs at 10⁷. Over the same range the time spent
  in garbage collection rises to about half the run. That fits each division allocating
  two tree nodes, which the collector must trace through an ever larger live tree, but it
  was not profiled.
- **Live memory is about 230–250 bytes per surviving cell.** Extinct lineages are pruned,
  so the memory follows the survivors' tree, the cell dictionary and the event queue.
  10⁸ cells would need roughly 25 GB live, and more at the peak.

Post-hoc statistics on one population of ``N = 10^6``:

| function | time | ns per cell |
|:---|---:|---:|
| [`mutations_per_cell`](@ref) | 0.11 s | 107 |
| [`site_frequency_spectrum`](@ref) | 0.53 s | 533 |
| [`branch_spectrum`](@ref) | 0.52 s | 523 |
| [`leaf_depths`](@ref) | 0.32 s | 316 |
| [`sample_leaves`](@ref) (``n = 1000``) | 0.31 s | 314 |

Each statistic costs a fraction of the simulation that produced the tree (2.3 s), so
computing several statistics per run is cheap.

**Parallel throughput** (`julia -t 16 … --parallel 100000`): 64 runs to 10⁵ cells take
6.93 s one after the other and 1.78 s on 16 threads. That is 9.2 vs 36 runs per second, a
3.9× speed-up (24% parallel efficiency). The likely limit is the shared garbage
collector, since every run allocates its tree nodes; this was not profiled. For large
sweeps, running several independent Julia processes, each with its own collector, may
scale better than threads.

Timings on a shared node vary with the load from other users; treat them as
order-of-magnitude figures, not as a regression test.
