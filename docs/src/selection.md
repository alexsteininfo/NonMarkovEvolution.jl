# Mutations and selection

Three fields of [`NonMarkovBlock`](@ref) describe mutations: `ν` is *how many*,
`effect_dist` *how big*, and `fitness_update` *how a mutation combines with what the cell
already has*. Selection models differ almost entirely in the third.

## The mutation channel

At every division, **each daughter independently**

1. draws a number of new mutations, ``j \sim \mathrm{Poisson}(ν)``;
2. for each, draws an effect ``δ \sim`` `effect_dist` and applies
   ``f \leftarrow`` `fitness_update(f, δ)`, starting from the **parent's** fitness;
3. stores ``j`` in `mutations`, the running total in `total_mutations`, and the final ``f``.

Two consequences:

- **`fitness_update` is applied once per mutation, not once to their sum.** Two mutations of
  effect ``δ`` are `update(update(f, δ), δ)`, which for a cap or a maximum is not
  `update(f, 2δ)`. That is what lets the rule express epistasis.
- **The daughters draw independently**, so a division yields ``2ν`` new mutations on
  average, and both, one or neither daughter can mutate.

The package targets ``ν \gtrsim 0.1``, where every cell has a distinct fitness history.
For ``ν \ll 1``, where clones are the natural unit, see
[`BirthDeathMutation`](https://github.com/alexsteininfo/BirthDeathMutation).

!!! note "Mutations arise only at division"
    The simulated mutations are those of the Poisson channel; their effects may be zero.
    Further neutral mutations need no state and are derived from the tree afterwards (see
    [Neutral evolution](#Neutral-evolution)). Because mutations arise only at division,
    a cell's fitness never changes between its birth and its own event, which is what
    makes the competing-risks scheduling exact.

## Selection modes

Every mode is a choice of `effect_dist` and `fitness_update`; `s` is the selection
coefficient and `f₀ = 1` the founding fitness.

| Mode | `effect_dist` | `fitness_update` |
|:---|:---|:---|
| [Neutral](#Neutral-evolution) | anything, with `ν = 0` | anything |
| [Additive, fixed](#Additive,-fixed-effect) | `Dirac(s)` | `(f, δ) -> f + δ` |
| [Additive, random](#Additive,-random-effect) | `Exponential(s)` | `(f, δ) -> f + δ` |
| [Multiplicative](#Multiplicative) | `Dirac(s)` or `Exponential(s)` | `(f, δ) -> f * (1 + δ)` |
| [Winner-takes-all](#Winner-takes-all-(max-random)) | `Exponential(s)` | `(f, δ) -> max(f, 1 + δ)` |
| [Capped](#Capped-selection-(diminishing-returns)) | `Dirac(1.0)` or `Exponential(1.0)` | `(f, δ) -> min(f * (1 + s*δ*(1 - f/M)), M)` |
| [Deleterious](#Deleterious-mutations-and-mutational-load) | `Normal(-μ, σ)` | `(f, δ) -> max(f + δ, f_min)` |

### Neutral evolution

With `ν = 0` no effects are ever drawn and every cell keeps `f₀`. `effect_dist` and
`fitness_update` are still required but never consulted. The tree is still informative:
non-exponential timing changes its shape even without selection, and
[`leaf_depths`](@ref) and [`branch_spectrum`](@ref) are the observables to compare.

Untracked neutral mutations at a rate ``m`` per daughter per division are derived from the
tree rather than simulated — exact in distribution, free, and one tree serves every ``m``:

```@example neutral
using NonMarkovEvolution, Distributions, Random
block = NonMarkovBlock(
    birth_dist = f -> Gamma(5.0, 1 / (5 * f)), death_dist = f -> Exponential(4.0),
    effect_dist = Dirac(0.0), fitness_update = (f, δ) -> f, ν = 0.0,
    stopfunction = pop -> popsize(pop) >= 1_000, restart_on_extinction = true)
pop  = simulate!(initialize_population(), block, MersenneTwister(3))
root = single_root(pop)
m    = 5.0
rng  = MersenneTwister(1)
burden = [rand(rng, Poisson(m * d)) for d in leaf_depths(root; order = :leaves)]
expected_sfs = m .* branch_spectrum(root)     # E[sfs[k]] for k ≥ 2; see branch_spectrum
(length(burden), expected_sfs[2:4])
```

`ν = 0` still consumes one random number per daughter (`Poisson(0.0)` draws), so a
neutral run does not share a stream with a model that has no mutation step.

### Additive, fixed effect

`Dirac(s)` with `(f, δ) -> f + δ`: after ``n`` mutations ``f = 1 + ns``, without a ceiling.
The simplest non-neutral model and the most tractable one: independent gain-of-function
events, each shortening the mean cycle by the same absolute amount.

### Additive, random effect

`Exponential(s)` with the same update: mean fitness tracks the fixed-effect model, but
rare large-effect mutations sweep much faster. The exponential is the maximum-entropy
choice for a positive effect of known mean. Comparing the two at matched ``s`` isolates
the effect of heterogeneity. `Exponential(0.0)` throws, so a sweep down to zero needs
`s > 0 ? Exponential(s) : Dirac(0.0)` (which also changes the random stream).

### Multiplicative

`(f, δ) -> f * (1 + δ)`: after ``n`` mutations ``f = (1+s)^n``, so log-fitness is additive.
Growth outpaces the additive rule, especially in the tail, and a single lineage can come
to dominate well before the size target; cap it if that is not wanted.

### Winner-takes-all (max-random)

`(f, δ) -> max(f, 1 + δ)`: each mutation proposes a fitness and the cell keeps the better
one. Only the running maximum of the lineage's draws matters, so after ``n`` exponential
draws the expected fitness grows like ``1 + s \ln n`` — strong diminishing returns and a
sharp sweep signature in the SFS.

### Capped selection (diminishing returns)

```julia
M              = 10.0
effect_dist    = Exponential(1.0)     # or Dirac(1.0), same mean
fitness_update = (f, δ) -> min(f * (1 + s * δ * (1 - f / M)), M)
```

A multiplicative rule with a logistic brake: the factor ``1 - f/M`` shrinks the gain as
fitness approaches the ceiling ``M``, a maximal division rate. Early mutations act almost
fully; near ``M`` further mutations are nearly neutral. Writing the magnitude as ``δ`` with
mean 1 makes the fixed and random variants a one-line switch; the random one can carry a
cell to ``M`` in one draw, giving a bimodal final fitness distribution.

### Deleterious mutations and mutational load

Nothing requires ``δ > 0``:

```julia
f_min          = 0.05
effect_dist    = Normal(-0.02, 0.05)          # mostly deleterious
fitness_update = (f, δ) -> max(f + δ, f_min)
```

!!! danger "A floor is mandatory"
    Fitness divides into a scale parameter. The moment a cell's fitness reaches 0 its
    next scheduling raises `DomainError: Gamma: the condition θ > zero(θ) is not
    satisfied`, partway through the run. Any rule whose effects can be negative must
    clamp to a strictly positive `f_min`; a multiplicative rule needs the same care when
    ``δ`` can reach ``-1``.

A floored cell divides slowly and is usually out-competed. To make load lethal rather
than crippling, let `death_dist` shorten as ``f`` falls instead.

## Choosing ν, s and k together

- **``ν`` sets the burden**: a cell carries on average ``ν`` times its divisional depth
  in mutations, since it is a daughter once per division on its path. Depths from one cell
  to ``N = 10^4`` lie between ``\log_2 N \approx 13`` (near-synchronous division) and
  ``2 \ln N \approx 18`` (exponential timing), more with death, so ``ν = 0.2`` gives 3–4
  mutations per cell. Below ``ν \approx 0.01`` a run is neutral in all but name.
- **``s`` and ``ν`` trade off** through the per-division expected gain ``νs``, until a
  non-linear `fitness_update` breaks the symmetry — which is what the capped and
  winner-takes-all modes are for.
- **``k`` changes the growth rate at a fixed mean cycle**
  (see [What non-exponential timing changes](concepts.md#What-non-exponential-timing-changes)),
  so fix ``k`` before calibrating ``s``.

A useful diagnostic is the fitness variance against its mean: under additive selection
both grow with ``νs``; under a cap the variance turns over near ``M``; under
winner-takes-all the mean saturates while the variance collapses.

## Injecting a single mutation at a chosen moment

To place **one** mutation in **one** cell at **one** moment, turn the Poisson channel off
and use the `on_division` hook; see [Hooks](blocks.md#on_division).
