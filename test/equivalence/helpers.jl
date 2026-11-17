using JLD2

"""
    load_inputs() -> Dict{String,Any}

Load the frozen inputs of the 0.4.1 reference run from `test/reference/inputs.jld2`.
"""
function load_inputs()::Dict{String,Any}
    path = joinpath(@__DIR__, "..", "reference", "inputs.jld2");
    return load(path);
end

"""
    load_reference() -> Dict{String,Any}

Load the frozen outputs of 0.4.1. Bounds checking changes the last bit of some sums, and `Pkg.test()` forces bounds checking on,
so we load `reference-0.4.1-checkbounds.jld2` in that mode and `reference-0.4.1.jld2` in every other mode.
"""
function load_reference()::Dict{String,Any}

    # pick the file that matches the bounds-checking mode of this run -
    file_name = "reference-0.4.1.jld2";
    if Base.JLOptions().check_bounds == 1
        file_name = "reference-0.4.1-checkbounds.jld2";
    end

    # load -
    path = joinpath(@__DIR__, "..", "reference", file_name);
    return load(path);
end
