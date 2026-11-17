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

Load the frozen outputs of 0.4.1 from `test/reference/reference-0.4.1.jld2`.
"""
function load_reference()::Dict{String,Any}
    path = joinpath(@__DIR__, "..", "reference", "reference-0.4.1.jld2");
    return load(path);
end
