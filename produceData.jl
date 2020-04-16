include("functionsCreateData.jl")

## inputs
N = 3
Δ_range = [-2, -4, -8]
b₀_range = [5]
s_range = 10.0.^(range(log10(10e-6), log10(10), length=5))
nRep = 5
ones1N = AFArray(ones(1,N))
onesN1 = AFArray(ones(N,1))
onesN = AFArray(ones(N))


## Simulation

dic_detunning = []
@time @progress for Δ₀ in Δ_range
	oneΔ = AllSimulationOneΔ(N, Δ₀, b₀_range, s_range, nRep)
    push!(dic_detunning, oneΔ)
end
OneSimulation = simulation(N, dic_detunning)


cd("/home/usp/Documents/Quantum-Pair-Correlation")
save("N$(N).jld", "N$(N)", OneSimulation)

@async run(`julia toRun.jl -2`)
@async run(`julia toRun.jl -4`)
@async run(`julia toRun.jl -8`)

aa = parse(Int, readline())
