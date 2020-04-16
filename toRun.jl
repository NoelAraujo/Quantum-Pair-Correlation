include("functionsCreateData.jl")

## inputs
N = 50
b₀_range = [5]
s_range = 10.0.^(range(log10(10e-6), log10(10), length=25))
nRep = 30
ones1N = AFArray(ones(1,N))
onesN1 = AFArray(ones(N,1))
onesN = AFArray(ones(N))
#
Δ₀ = parse(Int, ARGS[1])
oneΔ = AllSimulationOneΔ(N, Δ₀, b₀_range, s_range, nRep)
println("Data saved")
