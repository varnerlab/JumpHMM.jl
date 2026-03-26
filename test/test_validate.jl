@testset "Validation" begin

    @testset "validate produces report" begin
        Random.seed!(42)
        n = 500
        prices = cumsum(randn(n) .* 0.01) .+ 100.0
        prices = abs.(prices)

        model = fit(JumpHiddenMarkovModel, prices; N=10, ν=5.0)
        report = validate(model, prices; n_paths=20, seed=42)

        @test report isa ValidationReport
        @test length(report.path_results) == 20
        @test 0.0 ≤ report.ks_pass_rate ≤ 1.0
        @test 0.0 ≤ report.ad_pass_rate ≤ 1.0
        @test report.mean_acf_mae ≥ 0.0
        @test report.mean_wasserstein ≥ 0.0
        @test report.mean_hellinger ≥ 0.0
    end

end
