# Performance

`benchmark/benchmark.jl` times growth from one cell to ``N`` cells, the post-hoc
statistics on the grown tree, and the throughput of many independent simulations run in
parallel. The model is the one of `examples/01_SingleCellExpansion.jl`:

- division time ``\mathrm{Gamma}(2)`` with mean ``1/f``, lifetime ``\mathrm{Exponential}(20)``;
- ``\mathrm{Poisson}(ν = 0.5)`` mutations per daughter, each adding
  ``δ \sim \mathrm{Exponential}(0.05)`` to the fitness;
- `restart_on_extinction = true`, so a run's time includes any failed attempts.

It runs four modes:

| mode | what changes |
|:---|:---|
| `gamma` | the model as is (`:queue` algorithm) |
| `gamma+sizehint` | the same, after `sizehint!(pop, N)` |
| `exp/thinning` | exponential division times with mean ``1/f`` (a Markov model), `:thinning` |
| `exp/queue` | the same Markov model forced onto `:queue` |

## Running it

```bash
julia --project=. benchmark/benchmark.jl                      # N = 10³ … 10⁷
julia --project=. benchmark/benchmark.jl 1000000 10000000     # chosen sizes
julia -t 16 --project=. benchmark/benchmark.jl --parallel 100000
bash benchmark/run_benchmarks.sh                              # both, output kept in a log
```

- Compilation is excluded: a warm-up run with the same block types comes first.
- **Live memory** is the GC's live-bytes counter after the run and a full collection,
  minus the baseline before it. It covers the survivors' tree and the carried event
  queue.
- Sizes run in increasing order. A size whose estimated peak memory (1.5 × the live bytes
  per cell of the previous run, times ``N``) exceeds 60% of the currently *free* RAM is
  skipped, which protects shared nodes.
- `--parallel N` runs ``4 ×`` `nthreads` simulations to ``N`` cells (`gamma` mode), first
  one after the other and then with `Threads.@threads` (one RNG per run), and reports
  runs per second.

## Where the time goes

Version 0.5 stores the tree as a struct of arrays and the event queue as a 4-ary heap of
16-byte events (see [How the tree is stored](concepts.md#How-the-tree-is-stored)). A run
makes a few hundred allocations in total, all of them array growth, and nothing holds
pointers for the garbage collector to trace. What remains:

- **The event queue.** One pending event per living cell; at ``10^6`` cells the heap no
  longer fits in cache, and each event costs a few cache misses. This cost belongs to
  the model: non-exponential waiting times need a clock per cell.
- **Random numbers.** About 100 ns per division (a Gamma, an exponential and a Poisson
  draw per daughter).
- **A few full collections per run,** triggered by array growth. Their cost does not
  depend on the population: it is the time to mark the whole Julia session, which on
  the shared reference node was 75–150 ms each (75 ms with `--gcthreads=8`, 150 ms with
  1). Two ways to avoid them:
  - `sizehint!(pop, N)` before `simulate!` reserves the arrays up front;
  - `--gcthreads=4` (or more) on the `julia` command line makes each one cheaper.

The `:thinning` path needs no queue. Its cost per useful event grows with the spread of
``b + d`` across cells, because ticks are discarded in proportion to how far a cell's
rate lies below the fastest cell's.

## Results

Measured on the Hyperion cluster (node `iribhm-himm01`: 2 × 28-core AMD EPYC Rome, 1.5 TB
RAM, shared and unscheduled; load average about 40) with Julia 1.12.7, on 9 October 2026,
single samples.

| mode | ``N`` | time | GC share | live memory | ns per cell |
|:---|:---|---:|---:|---:|---:|
| `gamma` | 10⁵ | 0.06 s | 3% | ≈ 10 MB | 552 |
| `gamma` | 10⁶ | 1.17 s | 31% | ≈ 0.12 GB | 1 173 |
| `gamma` | 10⁷ | 14.5 s | 10% | ≈ 1.4 GB | 1 447 |
| `gamma+sizehint` | 10⁵ | 0.05 s | 1% | ≈ 10 MB | 456 |
| `gamma+sizehint` | 10⁶ | 0.94 s | 18% | ≈ 0.14 GB | 944 |
| `gamma+sizehint` | 10⁷ | 12.5 s | 1% | ≈ 1.3 GB | 1 248 |
| `exp/thinning` | 10⁵ | 0.03 s | 2% | ≈ 10 MB | 327 |
| `exp/thinning` | 10⁶ | 1.16 s | 29% | ≈ 0.10 GB | 1 162 |
| `exp/thinning` | 10⁷ | 12.2 s | 10% | ≈ 1.2 GB | 1 224 |
| `exp/queue` | 10⁵ | 0.04 s | 1% | ≈ 10 MB | 431 |
| `exp/queue` | 10⁶ | 1.04 s | 32% | ≈ 0.12 GB | 1 044 |
| `exp/queue` | 10⁷ | 13.0 s | 9% | ≈ 1.4 GB | 1 303 |

- **Against 0.4** (table below), growth is 1.9× faster at 10⁶ and 2.8× faster at 10⁷,
  with 40–45% less memory. The cost per cell now rises only mildly with ``N`` (1.2 µs at
  10⁶, 1.4 µs at 10⁷, against 2.3 and 4.0 µs in 0.4).
- **`sizehint!`** saves 15–20%: it removes the full collections that array growth
  triggers, whose cost on this node is fixed per collection.
- **`:thinning` is fastest for small populations** (327 against 431 ns per cell at 10⁵)
  but only on a par with the queue from 10⁶ on. In this model the fitness spread grows
  during the run, so the rate bound rises and more ticks are discarded, and each tick
  reads a randomly chosen row of a large tree.

Post-hoc statistics on one population of ``N = 10^6``:

| function | 0.4 | 0.5 |
|:---|---:|---:|
| [`mutations_per_cell`](@ref) | 0.11 s | 0.011 s |
| [`site_frequency_spectrum`](@ref) (population) | 0.53 s | 0.018 s |
| [`branch_spectrum`](@ref) (root) | 0.52 s | 0.38 s |
| [`leaf_depths`](@ref) | 0.32 s | 0.12 s |
| [`sample_leaves`](@ref) (``n = 1000``) | 0.31 s | 0.10 s |

The population-wide burden and spectrum are single sequential passes over the tree;
the root methods still list the subtree first, which is why `branch_spectrum` gained
least.

**Parallel throughput** (`julia -t 16 … --parallel 100000`): 64 runs to 10⁵ cells take
4.88 s one after the other and 1.62 s on 16 threads, 13.1 against 39.4 runs per second, a
3.0× speed-up (19% efficiency). A single run is faster than in 0.4, but threads still
scale poorly; the likely limits are memory bandwidth on a loaded node and the full
collections, which stop every thread (not profiled). Running independent processes may
scale better.

### Version 0.4, for comparison

Measured on the Hyperion cluster (node `iribhm-himm01`: 2 × 28-core AMD EPYC Rome, 1.5 TB
RAM, shared and unscheduled) with Julia 1.12.7, on 9 October 2026, single samples. In
0.4 every node was a heap object, and the living cells sat in a `Dict`.

| ``N`` | time | GC share | live memory | ns per cell |
|:---|---:|---:|---:|---:|
| 10³ | < 0.01 s | 0% | < 10 MB | 665 |
| 10⁴ | < 0.01 s | 0% | < 10 MB | 495 |
| 10⁵ | 0.08 s | 16% | ≈ 20 MB | 793 |
| 10⁶ | 2.3 s | 42% | ≈ 0.22 GB | 2 258 |
| 10⁷ | 40 s | 51% | ≈ 2.3 GB | 4 049 |

Post-hoc statistics on one population of ``N = 10^6`` took 0.11 s
([`mutations_per_cell`](@ref)), 0.53 s ([`site_frequency_spectrum`](@ref)), 0.52 s
([`branch_spectrum`](@ref)), 0.32 s ([`leaf_depths`](@ref)) and 0.31 s
([`sample_leaves`](@ref), ``n = 1000``). Parallel throughput at ``N = 10^5`` on 16 threads
was 36 runs per second, a 3.9× speed-up (24% efficiency).

Timings on a shared node vary with the load from other users; treat them as
order-of-magnitude figures, not as a regression test.
