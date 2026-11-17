# compare_references.jl
#
# Compares two reference files key by key with `==` and reports every key that differs.
# Used to check that make_reference.jl is reproducible: run it twice, then compare the two outputs.
#
# Run: julia --project=test/reference test/reference/compare_references.jl first.jld2 second.jld2

using JLD2

"""
    compare_references(path_a, path_b) -> Vector{String}

Return the keys that are missing from one file or whose values are not `==`.
"""
function compare_references(path_a::String, path_b::String)
    a = load(path_a)
    b = load(path_b)
    differing = String[]
    for key in sort(collect(union(keys(a), keys(b))))
        if !haskey(a, key) || !haskey(b, key)
            push!(differing, key)
        elseif !(a[key] == b[key])
            push!(differing, key)
        end
    end
    println("keys in first file: ", length(a), "  keys in second file: ", length(b))
    return differing
end

if abspath(PROGRAM_FILE) == @__FILE__
    differing = compare_references(ARGS[1], ARGS[2])
    if isempty(differing)
        println("all keys equal")
    else
        println("keys that differ (", length(differing), "):")
        for key in differing
            println("  ", key)
        end
        exit(1)
    end
end
