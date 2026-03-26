@testset "Simulation" begin

    # helper: build a small test model
    function _make_test_model(; ϵ=0.0, λ=10.0)
        partition = LaplacePartition(0.0, 1.0, 5)
        T = [0.6 0.2 0.1 0.05 0.05;
             0.1 0.6 0.2 0.05 0.05;
             0.05 0.1 0.6 0.2 0.05;
             0.05 0.05 0.1 0.6 0.2;
             0.05 0.05 0.05 0.2 0.65]
        emissions = [StudentTEmission(Float64(k), 0.5, 5.0, 100, false) for k in 1:5]
        π̄ = stationary_distribution(T)
        jump = JumpParameters(ϵ, λ)
        return JumpHiddenMarkovModel(partition, T, emissions, π̄, jump, 5.0, 0.0, 1/252)
    end

    @testset "simulate - no jumps" begin
        model = _make_test_model(ϵ=0.0)
        result = simulate(model, 100; n_paths=10, seed=42)
        @test length(result.paths) == 10
        @test length(result.paths[1].states) == 100
        @test length(result.paths[1].observations) == 100
        @test all(p -> all(.!p.jumps), result.paths)  # no jumps
    end

    @testset "simulate - with jumps" begin
        model = _make_test_model(ϵ=0.5, λ=5.0)
        result = simulate(model, 200; n_paths=50, seed=42)
        # at least some paths should have jumps
        has_jumps = any(p -> any(p.jumps), result.paths)
        @test has_jumps
    end

    @testset "simulate - seed reproducibility" begin
        model = _make_test_model(ϵ=0.1, λ=10.0)
        r1 = simulate(model, 50; n_paths=5, seed=123)
        r2 = simulate(model, 50; n_paths=5, seed=123)
        @test r1.paths[1].observations ≈ r2.paths[1].observations
        @test r1.paths[1].states == r2.paths[1].states
    end

    @testset "simulate - states in range" begin
        model = _make_test_model(ϵ=0.3, λ=5.0)
        result = simulate(model, 100; n_paths=20, seed=42)
        for p in result.paths
            @test all(1 .≤ p.states .≤ 5)
        end
    end

end
