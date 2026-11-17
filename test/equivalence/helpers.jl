using JLD2
using TOML

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

"""
    test_report_equals_reference(report, prefix::String)

Compare every field of a validation report with the 0.4.1 reference keys under `prefix`. The `hellinger` and `mean_hellinger`
fields are compared with the `_corrected` keys instead. Version 0.5.0 counts the largest observation in the Hellinger histogram,
which the 0.4.1 histogram dropped, so the reference script runs the same 0.4.1 `validate` calls again with the corrected distance.
"""
function test_report_equals_reference(report, prefix::String)
    for name ∈ ["ks_pvalue", "ad_pvalue", "wasserstein", "hellinger", "acf_mae", "kurtosis_sim", "kurtosis_emp",
        "ks_pass_rate", "ad_pass_rate", "mean_acf_mae", "mean_wasserstein", "mean_hellinger", "mean_kurtosis", "α"]
        key = prefix * "/" * name
        if name == "hellinger" || name == "mean_hellinger"
            key = key * "_corrected"
        end
        @test getfield(report, Symbol(name)) == reference[key]
    end
end

"""
    check_reference_environment() -> Tuple{Bool,Array{String,1}}

Check that this run uses the Julia version and the Distributions, StatsBase, and SpecialFunctions versions that wrote the reference files.
We read the versions from `test/reference/Manifest.toml`. The equivalence tests compare with `==`, and a new version of one of these packages
can change the last bit of a number, so we run the equivalence tests only when every version matches.

### Returns
- `Tuple{Bool,Array{String,1}}`: `true` if every version matches, and one line for each version that does not match
"""
function check_reference_environment()::Tuple{Bool,Array{String,1}}

    # initialize -
    manifest = TOML.parsefile(joinpath(@__DIR__, "..", "reference", "Manifest.toml"));
    mismatches = Array{String,1}();

    # the Julia version -
    recorded_julia = VersionNumber(manifest["julia_version"]);
    if VERSION != recorded_julia
        push!(mismatches, "Julia $(VERSION) (the reference used $(recorded_julia))");
    end

    # the version of each package, read from the loaded module -
    for name ∈ ["Distributions", "StatsBase", "SpecialFunctions"]
        entry = manifest["deps"][name][1];
        recorded_version = VersionNumber(entry["version"]);
        package_id = Base.PkgId(Base.UUID(entry["uuid"]), name);
        if haskey(Base.loaded_modules, package_id) == false
            push!(mismatches, "$(name) is not loaded (the reference used $(recorded_version))");
        else
            loaded_version = pkgversion(Base.loaded_modules[package_id]);
            if loaded_version != recorded_version
                push!(mismatches, "$(name) $(loaded_version) (the reference used $(recorded_version))");
            end
        end
    end

    # return -
    return (isempty(mismatches), mismatches);
end
