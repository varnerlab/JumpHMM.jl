@testset "Decoding" begin

    function _make_decode_model()
        partition = LaplacePartition(0.0, 1.0, 3)
        T = [0.8 0.15 0.05;
             0.1 0.8 0.1;
             0.05 0.15 0.8]
        # very distinct emissions so Viterbi can recover states
        emissions = [
            StudentTEmission(-10.0, 0.1, 5.0, 100, false),
            StudentTEmission(0.0, 0.1, 5.0, 100, false),
            StudentTEmission(10.0, 0.1, 5.0, 100, false),
        ]
        π̄ = stationary_distribution(T)
        jump = JumpParameters(0.0, 1.0)
        return JumpHiddenMarkovModel(partition, T, emissions, π̄, jump, 5.0, 0.0, 1/252)
    end

    @testset "decode - Viterbi recovers known states" begin
        model = _make_decode_model()
        # observations clearly from each state
        obs = [-10.1, -9.9, -10.0, 0.05, -0.02, 0.01, 10.1, 9.9, 10.0]
        states = decode(model, obs)
        @test states[1:3] == [1, 1, 1]
        @test states[4:6] == [2, 2, 2]
        @test states[7:9] == [3, 3, 3]
    end

    @testset "forward_filter - rows sum to 1" begin
        model = _make_decode_model()
        obs = [-10.0, 0.0, 10.0, -10.0, 0.0]
        alpha = forward_filter(model, obs)
        @test size(alpha) == (5, 3)
        for t in 1:5
            @test sum(alpha[t, :]) ≈ 1.0 atol=1e-10
        end
    end

    @testset "log_likelihood - finite" begin
        model = _make_decode_model()
        obs = [-10.0, 0.0, 10.0]
        ll = log_likelihood(model, obs)
        @test isfinite(ll)
    end

end
