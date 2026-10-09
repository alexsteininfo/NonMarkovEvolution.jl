"""
    initialize_population(; fitness_init=1.0, time=0.0) -> Population

Create a population containing a single founding cell with the given initial fitness.
The cell acquires no mutations at birth (it is the root of the lineage tree).

```julia
pop = initialize_population(fitness_init = 1.0)
```
"""
function initialize_population(; fitness_init::Real = 1.0, time::Real = 0.0)
    return initialize_population(1; fitness_init = fitness_init, time = time)
end

"""
    initialize_population(N::Int; fitness_init=1.0, time=0.0) -> Population

Create a population of `N` identical independent cells, each with `fitness_init`.
Useful for starting from a pre-existing pool rather than a single founder. Each cell is
the root of its own tree, so the population is a forest.
"""
function initialize_population(N::Int; fitness_init::Real = 1.0, time::Real = 0.0)
    N >= 1 || throw(ArgumentError("initialize_population: N must be >= 1, got $N"))
    cells = Dict{Int64, BinaryNode{NonMarkovCell}}()
    for id in 1:N
        cells[id] = BinaryNode(NonMarkovCell(id, time, 0, 0, fitness_init))
    end
    return Population(cells, Float64(time), N, nothing)
end
