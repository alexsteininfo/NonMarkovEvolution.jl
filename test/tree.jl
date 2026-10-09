# The struct-of-arrays lineage tree: handles, hand-built trees, pruning, compaction.

@isdefined(fixture_tree) || include("fixtures.jl")
const NMEt = NonMarkovEvolution

@testset "CellNode properties read the tree's columns" begin
    root = fixture_tree()
    @test root.id == 1 && root.birthtime == 0.0 && root.mutations == 5
    @test root.total_mutations == 5 && root.fitness == 1.0
    @test isnothing(root.parent)
    L = root.left
    @test L.id == 2 && L.parent == root && L.total_mutations == 6
    @test root.right.id == 3 && isnothing(root.right.left)
    @test root.data == NonMarkovCell(1, 0.0, 5, 5, 1.0)
    @test [l.id for l in Leaves(root)] == [4, 5, 3]
    @test [n.id for n in PreOrderDFS(root)] == [1, 2, 4, 5, 3]
    @test popsize(root) == 3 && popsize(L) == 2
    @test alive_cells(root) == collect(Leaves(root))
    @test length(root.tree) == 5
end

@testset "hand-built trees need increasing ids and free child slots" begin
    root = rootnode(5, 0.0, 0)
    @test_throws ArgumentError child!(:left, root, 3, 1.0, 0)
    child!(:left, root, 6, 1.0, 0)
    @test_throws ErrorException child!(:left, root, 7, 1.0, 0)
    @test_throws ArgumentError add_root!(root.tree, NonMarkovCell(6, 0.0, 0, 0, 1.0))
end

@testset "equality and hashing are by tree and id" begin
    root = fixture_tree()
    @test root.left.left == root.left.left
    @test hash(root.left) == hash(root.left)
    @test root.left != root.right
    @test root != fixture_tree()                      # same ids, different tree
    @test length(Set([root, root.left, root.left, root.right])) == 3
end

@testset "pruning a founder lineage removes its root as well" begin
    pop = initialize_population(2)
    tree = pop.tree
    NMEt._die!(pop, Int32(2))
    @test popsize(pop) == 1
    @test [c.id for c in alive_cells(pop)] == [1]
    @test [r.id for r in roots(pop)] == [1]
    @test !isalive(CellNode(tree, 2))
    @test single_root(pop).id == 1
end

@testset "compaction keeps leaf order, ids, links, and handles" begin
    # Grow with heavy death and compaction after every 16 removed rows; a handle taken
    # early must still resolve after many compactions moved its row.
    held = Ref{Any}(nothing)
    block = NonMarkovBlock(birth_dist = f -> Gamma(5.0, 1 / (5f)),
        death_dist = f -> Gamma(5.0, 1.1 / 5), stopfunction = p -> popsize(p) >= 2_000,
        effect_dist = Exponential(0.05), fitness_update = (f, δ) -> f + δ, ν = 1.0,
        restart_on_extinction = true,
        on_division = (pop, p, d1, d2) -> (popsize(pop) == 1_000 && isnothing(held[]) &&
                                           (held[] = d1); nothing),
        on_restart = p -> (held[] = nothing; nothing))
    old = (NMEt._COMPACT_MIN[], NMEt._COMPACT_FRACTION[])
    NMEt._COMPACT_MIN[], NMEt._COMPACT_FRACTION[] = 16, 0.0
    pop = try
        simulate!(initialize_population(), block, MersenneTwister(77))
    finally
        NMEt._COMPACT_MIN[], NMEt._COMPACT_FRACTION[] = old
    end
    tree = pop.tree
    @test tree.ndead < 16                      # compacted recently
    @test length(tree) < pop._next_id          # rows really were dropped
    @test issorted(tree.id)
    links_ok = true
    for i in eachindex(tree.parent)
        p = tree.parent[i]
        p == NMEt.REMOVED && continue          # fewer than 16 may remain uncompacted
        p == 0 && continue
        links_ok &= (tree.left[p] == i || tree.right[p] == i) &&
                    tree.total[i] == tree.total[p] + tree.mutations[i] && p < i
    end
    @test links_ok
    h = held[]
    @test !isnothing(h)
    if isalive(h) || NMEt._haschildren(tree, searchsortedfirst(tree.id, h.id))
        @test h.id == getfield(h, :id)
        @test h.birthtime > 0
    end
    root = single_root(pop)
    @test Set(l.id for l in Leaves(root)) == Set(c.id for c in alive_cells(pop))
end

@testset "a handle on a removed cell throws when read" begin
    pop = initialize_population(3)
    h = CellNode(pop.tree, 2)
    NMEt._die!(pop, Int32(2))
    NMEt._compact!(pop.tree, Int32(1))
    @test !isalive(h)
    @test_throws ArgumentError h.fitness
end
