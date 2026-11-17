using JumpHMM
using Test
using Random
using Statistics
using LinearAlgebra

# load the frozen 0.4.1 inputs and outputs -
include(joinpath(@__DIR__, "equivalence", "helpers.jl"));
inputs = load_inputs();
# the loader picks the reference file that matches the bounds-checking mode of this run -
reference = load_reference();

# the equivalence tests compare with ==, so they run only with the Julia and package versions that wrote the reference files -
(same_environment, mismatches) = check_reference_environment();
folders = [joinpath(@__DIR__, "equivalence"), @__DIR__];
if same_environment == false
    println("Skipping the equivalence tests, because the versions differ from test/reference/Manifest.toml: ", join(mismatches, "; "));
    folders = [@__DIR__];
end

# run every equivalence test file, then every unit test file -
for folder ∈ folders
    for name ∈ sort(readdir(folder))
        if startswith(name, "test_") && endswith(name, ".jl")
            include(joinpath(folder, name));
        end
    end
end
