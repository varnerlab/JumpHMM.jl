@testset "build equals 0.4.1 fit" begin
    P = inputs["prices"]
    cases = [("fit/", reference["finance/G_spy"]), ("fit_short/", JumpHMM.excess_growth_rates(P[1:301, 1]))]
    for (prefix, G) in cases
        m = JumpHMM.build(JumpHMM.MyJumpHiddenMarkovModel, (growth_rates = G, number_of_states = 100, ν = 5.0))
        for field in ["laplace_μ", "laplace_b", "boundaries", "transition", "stationary",
                      "emission_μ", "emission_σ", "emission_n", "emission_fallback"]
            @test getfield(m, Symbol(field)) == reference[prefix * field]
        end
        @test JumpHMM.assign_states(m.boundaries, G) == reference[prefix * "states"]
        @test m.ϵ == 0.0 && m.λ == 1.0
        if prefix == "fit_short/"
            @test any(m.emission_fallback)
        end
    end
end
