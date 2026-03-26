using Documenter, JumpHMM, DataFrames

push!(LOAD_PATH,"../src/")

makedocs(
    sitename="JumpHMM.jl",
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [JumpHMM],
    pages = [
        "Home" => "index.md",

        "Getting Started" => "getting_started.md",

        "Single-Asset" => [
            "Fitting" => "fitting.md",
            "Tuning" => "tuning.md",
            "Simulation" => "simulation.md",
            "Decoding" => "decoding.md",
            "Validation" => "validation.md",
        ],

        "Multi-Asset" => [
            "Copulas" => "copulas.md",
            "Single-Index Model" => "sim.md",
            "Portfolio" => "portfolio.md",
        ],

        "Finance Helpers" => "finance.md",

        "API Reference" => "api.md",
    ]
)

deploydocs(
    repo = "github.com/varnerlab/JumpHMM.jl.git", branch = "gh-pages", target = "build"
)
