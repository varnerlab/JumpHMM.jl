@testset "State Partition" begin

    @testset "LaplacePartition construction" begin
        lp = LaplacePartition(0.0, 1.0, 10)
        @test lp.N == 10
        @test length(lp.boundaries) == 11
        @test lp.boundaries[1] == -Inf
        @test lp.boundaries[end] == Inf
        @test issorted(lp.boundaries)
    end

    @testset "fit LaplacePartition" begin
        Random.seed!(42)
        obs = rand(Laplace(0.0, 1.0), 10_000)
        lp = fit(LaplacePartition, obs; N=50)
        @test lp.N == 50
        @test abs(lp.μ) < 0.1  # should be close to 0
        @test abs(lp.b - 1.0) < 0.1  # should be close to 1
    end

    @testset "assign_states" begin
        lp = LaplacePartition(0.0, 1.0, 10)
        Random.seed!(42)
        obs = rand(Laplace(0.0, 1.0), 10_000)
        states = assign_states(lp, obs)
        @test length(states) == 10_000
        @test all(1 .≤ states .≤ 10)
        # bins should be roughly equal
        counts = [count(==(k), states) for k in 1:10]
        @test all(c -> 500 < c < 1500, counts)  # ~1000 each
    end

    @testset "assign_states extreme values" begin
        lp = LaplacePartition(0.0, 1.0, 5)
        @test assign_states(lp, [-1e10])[1] == 1
        @test assign_states(lp, [1e10])[1] == 5
    end

end
