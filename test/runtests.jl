using JumpHMM
using Test
using Random
using Statistics
using LinearAlgebra

# the equivalence tests compare with ==, and `--check-bounds=yes` (the default of Pkg.test) changes the last bit of sums,
# so run them with: Pkg.test(julia_args = ["--check-bounds=auto"])

# load the frozen 0.4.1 inputs and outputs -
include(joinpath(@__DIR__, "equivalence", "helpers.jl"));
inputs = load_inputs();
reference = load_reference();

# run every equivalence test file, then every unit test file -
for folder ∈ [joinpath(@__DIR__, "equivalence"), @__DIR__]
    for name ∈ sort(readdir(folder))
        if startswith(name, "test_") && endswith(name, ".jl")
            include(joinpath(folder, name));
        end
    end
end
