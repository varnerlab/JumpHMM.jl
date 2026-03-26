@testset "Transition Matrix" begin

    @testset "estimate_transition" begin
        states = [1, 2, 1, 2, 1, 3, 3, 2]
        T = estimate_transition(states, 3)
        @test size(T) == (3, 3)
        # rows sum to 1
        for i in 1:3
            @test sum(T[i, :]) ≈ 1.0
        end
        # check specific counts: from state 1 → 2 (twice), 1 → 3 (once)
        @test T[1, 2] ≈ 2/3
        @test T[1, 3] ≈ 1/3
    end

    @testset "estimate_transition - uniform fallback" begin
        states = [1, 2, 1, 2]  # state 3 never appears as source
        T = estimate_transition(states, 3)
        @test T[3, :] ≈ [1/3, 1/3, 1/3]
    end

    @testset "stationary_distribution" begin
        # simple 2-state transition matrix
        T = [0.9 0.1; 0.3 0.7]
        π̄ = stationary_distribution(T; power=100)
        @test length(π̄) == 2
        @test sum(π̄) ≈ 1.0
        # fixed point: π̄ * T ≈ π̄
        @test (π̄' * T)[:] ≈ π̄ atol=1e-10
    end

end
