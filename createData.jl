using ArrayFire
using Distances
using LinearAlgebra
using Gaston
using SpecialFunctions
using Statistics

include("functionsToCreateAtoms.jl")
include("QPC_Evolution_GPU.jl")

## inputs
N = 25
Δ₀ = -2
b₀ = 2
s = 1e-6

## cloud parameters
Γ = 1
k₀ = [0,0,1]
radius = sqrt(6N/(b₀ * norm(k₀)^2))
density = 3N/(4π*radius^3)
dmin = radius/300

## get cloud
atoms = getAtoms_distribution(N, radius, dmin, :homogenous)

## get params
G_af, Gconj_af, Γⱼₘ_af = getScalarKernel(atoms, N)
exclusionDiagonal_af, exclusion3D_af = getExclusionMatrices(N)
Ω⁺_af_on, Ω⁻_af_on = getLaser(atoms, s, Γ, Δ₀, k₀, N)

params_on = []; push!(params_on,
							N, G_af, Gconj_af, Γⱼₘ_af,
							exclusionDiagonal_af, exclusion3D_af,
							Ω⁺_af_on, Ω⁻_af_on,
							Δ₀, Γ)

s_off = 0.0 # no laser
Ω⁺_af_off, Ω⁻_af_off = getLaser(atoms, s_off, Γ, Δ₀, k₀, N)
params_off = []; push!(params_off,
							N, G_af, Gconj_af, Γⱼₘ_af,
							exclusionDiagonal_af, exclusion3D_af,
							Ω⁺_af_off, Ω⁻_af_off,
							Δ₀, Γ)

## qpc evolution
ones1N = AFArray(ones(1,N))
onesN1 = AFArray(ones(N,1))
onesN = AFArray(ones(N))

u₀_on_af = AFArray(zeros(ComplexF64, 2*N+4*N^2))
u₀_on_af[N+1:2*N] .= -1
u₀_on_af[2*N+3*N^2+1:2*N+4*N^2] .= +1
u₀_on_af[2*N+3*N^2+1:N+1:2*N+4*N^2] .= 0.0


@time time_on, u_on, last_u_on = simple_rk4(QPC_v4_gpu, u₀_on_af, (0,100), 1500, params_on)
population_on = 0.5 .+ 0.5*[ real(mean(u_on[i][N+1:2*N])) for i in 1:length(u_on)]
plot(time_on, abs.(population_on), plotstyle="linespoints", legend="On", title="GPU (N=$(N),s=$(s))")
GC.gc()
u₀_off_af = last_u_on
t_off = (time_on[end], time_on[end] + 200)
@time time_off, u_off, last_u_off = simple_rk4(QPC_v4_gpu, u₀_off_af, t_off, 1500, params_off)
population_off = 0.5 .+ 0.5*[ real(mean(u_off[i][N+1:2*N])) for i in 1:length(u_off)]
plot!(time_off, abs.(population_off), plotstyle="linespoints", legend="Off")


## intensity curve

intensity_off = zeros(length(u_off))
Gⱼₘ = geometricFactor(deg2rad(35), atoms; k₀=1)
@progress for j=1:length(u_off)
	σ⁻ = Array(u_off[j][1:N])
	Cⱼₘ = σ⁻.*σ⁻'
	# "sum(Cⱼₘ.*Gⱼₘ)" returns imaginary values almost zero,
	# so I ignore them and store only the real part
	intensity_off[j] = real(sum(Cⱼₘ.*Gⱼₘ))
end

plot(time_off, log10.(intensity_off./intensity_off[1]), legend="", xlabel="Time", ylabel="I/I₀")

## Stand alone Function
function simulateEvolution(N, atoms, s, Δ₀, b0;θ = 35, Γ=1, k₀=[0,0,1])
	G_af, Gconj_af, Γⱼₘ_af = getScalarKernel(atoms, N)
	exclusionDiagonal_af, exclusion3D_af = getExclusionMatrices(N)
	Ω⁺_af_on, Ω⁻_af_on = getLaser(atoms, s, Γ, Δ₀, k₀, N)

	params_on = []; push!(params_on,
								N, G_af, Gconj_af, Γⱼₘ_af,
								exclusionDiagonal_af, exclusion3D_af,
								Ω⁺_af_on, Ω⁻_af_on,
								Δ₀, Γ)

	s_off = 0.0 # no laser
	Ω⁺_af_off, Ω⁻_af_off = getLaser(atoms, s_off, Γ, Δ₀, k₀, N)
	params_off = []; push!(params_off,
								N, G_af, Gconj_af, Γⱼₘ_af,
								exclusionDiagonal_af, exclusion3D_af,
								Ω⁺_af_off, Ω⁻_af_off,
								Δ₀, Γ)

	## qpc evolution
	ones1N = AFArray(ones(1,N))
	onesN1 = AFArray(ones(N,1))
	onesN = AFArray(ones(N))

	u₀_on_af = AFArray(zeros(ComplexF64, 2*N+4*N^2))
	u₀_on_af[N+1:2*N] .= -1
	u₀_on_af[2*N+3*N^2+1:2*N+4*N^2] .= +1
	u₀_on_af[2*N+3*N^2+1:N+1:2*N+4*N^2] .= 0.0


	time_on, u_on, last_u_on = simple_rk4(QPC_v4_gpu, u₀_on_af, (0,100), 1500, params_on)
	population_on = 0.5 .+ 0.5*[ real(mean(u_on[i][N+1:2*N])) for i in 1:length(u_on)]
	plot(time_on, abs.(population_on), plotstyle="linespoints", legend="On", title="N=$(N),Δ=$(Δ₀), b₀=$(b0), s=$(s)")

	u₀_off_af = last_u_on
	t_off = (time_on[end], time_on[end] + 200)
	time_off, u_off, last_u_off = simple_rk4(QPC_v4_gpu, u₀_off_af, t_off, 1500, params_off)
	population_off = 0.5 .+ 0.5*[ real(mean(u_off[i][N+1:2*N])) for i in 1:length(u_off)]
	plot!(time_off, abs.(population_off), plotstyle="linespoints", legend="Off") |> display

	u₀_on_af = time_on = u_on = last_u_on = last_u_off = 1
	GC.gc()

	intensity_off = zeros(length(u_off))
	Gⱼₘ = geometricFactor(deg2rad(θ), atoms, k₀=norm(k₀))
	for j=1:length(u_off)
		σ⁻ = Array(u_off[j][1:N])
		Cⱼₘ = σ⁻.*σ⁻'
		intensity_off[j] = real(sum(Cⱼₘ.*Gⱼₘ))
	end
	return time_off, intensity_off
end
