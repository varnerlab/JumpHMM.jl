# make_inputs.jl
#
# Builds inputs.jld2, the fixed price matrix that every reference check uses.
# Selection rule (no hand-picking): the four sectors with the most tickers in
# the universe; in each of those sectors, the first three tickers in
# alphabetical order. Columns are SPY first, then the 12 stocks grouped by
# sector (largest sector first).
#
# Run: julia --project=test/reference test/reference/make_inputs.jl <universe.jld2> <sectors.csv> [out.jld2]
# The universe file holds the price tables of the S&P 500 tickers, and the sectors file maps each ticker to its GICS sector.

using JLD2
using CSV
using DataFrames

const DEFAULT_OUTPUT = joinpath(@__DIR__, "inputs.jld2")

"""
    count_by_sector(tickers, sector_of) -> Dict{String,Int}

Count how many of `tickers` fall in each sector. `sector_of` maps a ticker to its sector label.
Tickers with no sector label are skipped.
"""
function count_by_sector(tickers::Vector{String}, sector_of::Dict{String,String})
    counts = Dict{String,Int}()
    for ticker in tickers
        if haskey(sector_of, ticker)
            sector = sector_of[ticker]
            counts[sector] = get(counts, sector, 0) + 1
        end
    end
    return counts
end

"""
    largest_sectors(counts, n) -> Vector{String}

Return the `n` sectors with the most tickers, largest first. Ties go to the sector name that sorts first.
"""
function largest_sectors(counts::Dict{String,Int}, n::Int)
    names = sort(collect(keys(counts)))
    # a stable sort on the alphabetical list keeps ties in alphabetical order
    order = sortperm([counts[name] for name in names]; rev=true, alg=MergeSort)
    return names[order[1:n]]
end

"""
    build_inputs(universe_path, sectors_path) -> (prices, tickers, sectors)

Apply the selection rule and return the price matrix (days x 13, SPY first), the 13 tickers, and one
sector label for each of the 12 stocks.
"""
function build_inputs(universe_path::String, sectors_path::String)
    universe = load(universe_path)
    all_prices = universe["prices"]
    all_tickers = String.(universe["tickers"])

    table = CSV.read(sectors_path, DataFrame)
    sector_of = Dict{String,String}()
    for row in eachrow(table)
        sector_of[String(row.Ticker)] = String(row.Sector)
    end

    # SPY is the market column; the sector rule applies to the stocks
    stocks = [t for t in all_tickers if t != "SPY"]
    counts = count_by_sector(stocks, sector_of)
    chosen_sectors = largest_sectors(counts, 4)

    tickers = ["SPY"]
    sectors = String[]
    for sector in chosen_sectors
        members = sort([t for t in stocks if get(sector_of, t, "") == sector])
        for ticker in members[1:3]
            push!(tickers, ticker)
            push!(sectors, sector)
        end
    end

    columns = [findfirst(==(t), all_tickers) for t in tickers]
    prices = Matrix{Float64}(all_prices[:, columns])
    return prices, tickers, sectors
end

"""
    main(universe_path, sectors_path, output_path)

Build the inputs and write them to `output_path`.
"""
function main(universe_path::String, sectors_path::String, output_path::String)
    prices, tickers, sectors = build_inputs(universe_path, sectors_path)
    jldsave(output_path; prices=prices, tickers=tickers, sectors=sectors)
    println("wrote ", output_path)
    println("prices: ", size(prices), "  tickers: ", tickers)
    println("sectors: ", sectors)
end

if abspath(PROGRAM_FILE) == @__FILE__
    # the universe and sectors files are required; the output file is optional -
    if length(ARGS) < 2
        println("usage: julia --project=test/reference test/reference/make_inputs.jl <universe.jld2> <sectors.csv> [out.jld2]")
        exit(1)
    end
    universe_path = ARGS[1]
    sectors_path = ARGS[2]
    output_path = DEFAULT_OUTPUT
    if length(ARGS) >= 3
        output_path = ARGS[3]
    end
    main(universe_path, sectors_path, output_path)
end
