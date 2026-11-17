using Random

"""
    family_label(copula::JumpHMM.AbstractBivariateCopulaModel) -> String

Name the family of a bivariate copula: "gaussian", "studentt", "clayton", "gumbel", or "frank".
"""
function family_label(copula::JumpHMM.AbstractBivariateCopulaModel)::String
    if copula isa JumpHMM.MyGaussianBivariateCopulaModel
        return "gaussian";
    elseif copula isa JumpHMM.MyStudentTBivariateCopulaModel
        return "studentt";
    elseif copula isa JumpHMM.MyClaytonBivariateCopulaModel
        return "clayton";
    elseif copula isa JumpHMM.MyGumbelBivariateCopulaModel
        return "gumbel";
    elseif copula isa JumpHMM.MyFrankBivariateCopulaModel
        return "frank";
    end
    error("unknown bivariate copula family $(typeof(copula))");
end

"""
    family_parameters(copula::JumpHMM.AbstractBivariateCopulaModel) -> Array{Float64,1}

Parameters of a bivariate copula: ρ for gaussian; ρ then ν for studentt; θ for the others.
"""
function family_parameters(copula::JumpHMM.AbstractBivariateCopulaModel)::Array{Float64,1}
    if copula isa JumpHMM.MyGaussianBivariateCopulaModel
        return [copula.ρ];
    elseif copula isa JumpHMM.MyStudentTBivariateCopulaModel
        return [copula.ρ, copula.ν];
    end
    return [copula.θ];
end

@testset "vine copula equals 0.4.1" begin
    P = inputs["prices"]
    GA = JumpHMM.excess_growth_rates(P; rf = 0.0, Δt = 1/252)
    R = GA[:, 2:5]

    vine = JumpHMM.build(JumpHMM.MyVineCopulaModel, (growth_rates = R,))
    families = String[]
    parameters = Float64[]
    for tree ∈ vine.edges
        for edge ∈ tree
            push!(families, family_label(edge.copula))
            append!(parameters, family_parameters(edge.copula))
        end
    end
    @test vine.order == reference["vine/order"]
    @test families == reference["vine/families"]
    @test parameters == reference["vine/parameters"]
    Random.seed!(10)
    @test JumpHMM.sample_dependence(vine, 200) == reference["vine/U"]
end
