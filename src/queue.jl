# The queue of pending events: a 4-ary min-heap of 16-byte, pointer-free events.
#
# Every living cell has exactly one pending event, so at 10⁶ cells the heap no longer
# fits in cache. A 4-ary heap halves its depth compared with a binary one, and a
# division replaces the top (the dividing cell's event) by the first daughter's instead
# of popping and pushing, which saves one of the three sift passes per division.

"""
    Event(time, node, is_division)

A pending event: at absolute `time`, the cell at tree index `node` divides
(`is_division`) or dies. Ordered by time; exact ties by node index, which is creation
order, so tied events fire in a defined order.
"""
struct Event
    time::Float64
    node::Int32
    is_division::Bool
end

@inline Base.isless(a::Event, b::Event) =
    a.time < b.time || (a.time == b.time && a.node < b.node)

"""
    EventQueue

Min-heap of [`Event`](@ref)s with arity 4: `push!`, `pop!`, `first`, `replace_top!`.
"""
struct EventQueue
    v::Vector{Event}
end
EventQueue() = EventQueue(Event[])

Base.length(q::EventQueue) = length(q.v)
Base.isempty(q::EventQueue) = isempty(q.v)
Base.first(q::EventQueue) = @inbounds q.v[1]
Base.sizehint!(q::EventQueue, n::Integer) = (sizehint!(q.v, n); q)

const _ARITY = 4

@inline function _sift_up!(v::Vector{Event}, i::Int, e::Event)
    @inbounds while i > 1
        p = (i - 2) ÷ _ARITY + 1
        isless(e, v[p]) || break
        v[i] = v[p]
        i = p
    end
    @inbounds v[i] = e
    return nothing
end

@inline function _sift_down!(v::Vector{Event}, i::Int, e::Event)
    n = length(v)
    @inbounds while true
        c = (i - 1) * _ARITY + 2           # first child
        c > n && break
        last = min(c + _ARITY - 1, n)
        m = c                               # smallest child
        for k in (c + 1):last
            isless(v[k], v[m]) && (m = k)
        end
        isless(v[m], e) || break
        v[i] = v[m]
        i = m
    end
    @inbounds v[i] = e
    return nothing
end

function Base.push!(q::EventQueue, e::Event)
    push!(q.v, e)
    _sift_up!(q.v, length(q.v), e)
    return q
end

function Base.pop!(q::EventQueue)
    v = q.v
    top = @inbounds v[1]
    e = pop!(v)
    isempty(v) || _sift_down!(v, 1, e)
    return top
end

"""
    replace_top!(q, e) -> q

Replace the earliest event by `e`: the same as `pop!` then `push!`, with one sift.
"""
function replace_top!(q::EventQueue, e::Event)
    _sift_down!(q.v, 1, e)
    return q
end

# Translate node indices after `_compact!(tree, start)`. Compaction keeps the order of
# the surviving rows, so the heap order (time, then index) is unchanged.
function _remap!(q::EventQueue, remap::Vector{Int32}, start::Int32)
    v = q.v
    @inbounds for k in eachindex(v)
        e = v[k]
        e.node >= start || continue
        v[k] = Event(e.time, remap[e.node - start + 1], e.is_division)
    end
    return q
end
