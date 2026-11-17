using Documenter, JumpHMM, DataFrames

push!(LOAD_PATH,"../src/")

makedocs(
    sitename = "JumpHMM.jl",
    # the API page holds every docstring (about 150 KiB), so we raise the size at which Documenter warns about a page -
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true", size_threshold_warn = 200 * 2^10),
    modules = [JumpHMM],
    checkdocs = :exports,
    warnonly = false,
    pages = [
        "Home" => "index.md",

        "Getting Started" => "getting_started.md",

        "Single asset" => [
            "The Model" => "model.md",
            "Jump Episodes" => "jump_episodes.md",
            "Tuning" => "tuning.md",
            "Simulation" => "simulation.md",
            "Validation" => "validation.md",
            "Decoding" => "decoding.md",
        ],

        "Many assets" => [
            "Sector Factor Model" => "sector_factor.md",
            "Portfolio" => "portfolio.md",
            "Copulas" => "copulas.md",
            "Vine Copula" => "vine.md",
            "Single-Index Models" => "single_index.md",
        ],

        "Helpers" => "helpers.md",

        "Migrating from 0.4" => "migrating.md",

        "API Reference" => "api.md",
    ]
)

# deploy only on the CI server, so that a local build does not warn that it cannot deploy -
if get(ENV, "CI", nothing) == "true"
    deploydocs(
        repo = "github.com/varnerlab/JumpHMM.jl.git", branch = "gh-pages", target = "build"
    )
end
