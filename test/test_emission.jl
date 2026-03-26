@testset "Emission Fitting" begin

    @testset "fit_emissions - normal case" begin
        states = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        obs = [1.0, 1.1, 0.9, 5.0, 5.1, 4.9, -3.0, -3.1, -2.9]
        emissions = fit_emissions(states, obs, 3; ν=5.0)
        @test length(emissions) == 3
        @test emissions[1].μ ≈ mean([1.0, 1.1, 0.9])
        @test emissions[2].μ ≈ mean([5.0, 5.1, 4.9])
        @test !emissions[1].is_fallback
    end

    @testset "fit_emissions - fallback for sparse states" begin
        states = [1, 1, 1, 2, 2, 2]  # state 3 has 0 observations
        obs = [1.0, 1.1, 0.9, 5.0, 5.1, 4.9]
        emissions = fit_emissions(states, obs, 3; ν=5.0, min_obs=2)
        @test emissions[3].is_fallback == true
        @test emissions[3].n_obs == 0
        @test emissions[3].μ ≈ mean(obs)
    end

    @testset "sample_emission" begin
        e = StudentTEmission(10.0, 1.0, 5.0, 100, false)
        Random.seed!(42)
        samples = [sample_emission(e) for _ in 1:10_000]
        @test abs(mean(samples) - 10.0) < 0.1
    end

end
