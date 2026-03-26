@testset "Tuning" begin

    @testset "tune returns new model with jump params" begin
        # generate synthetic prices
        Random.seed!(42)
        n = 500
        prices = cumsum(randn(n) .* 0.01) .+ 100.0
        prices = abs.(prices)  # ensure positive

        model = fit(JumpHiddenMarkovModel, prices; N=10, ν=5.0)
        @test model.jump.ϵ == 0.0  # initial: no jumps

        tuned = tune(model, prices;
                     ϵ_range=[1e-3, 1e-2, 5e-2],
                     λ_range=[5.0, 10.0, 20.0],
                     n_paths=20, seed=42)

        # should be a new model with updated jump params
        @test tuned.jump.ϵ > 0.0
        @test tuned.jump.λ > 0.0
        # partition and transition should be unchanged
        @test tuned.partition.N == model.partition.N
        @test tuned.transition == model.transition
    end

end
