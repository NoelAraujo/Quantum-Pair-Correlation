using ArrayFire
using Distances
using LinearAlgebra
using JLD
using Plots; pyplot()
using ProgressMeter
using SpecialFunctions
using Statistics

include("functionsToCreateAtoms.jl")
include("structDefinitions.jl")
include("QPC_Evolution_GPU.jl")


function computeIntensityDecay(N, Δ₀, b₀, s)
    Γ = 1
    k₀ = [0,0,1]
    radius = sqrt(6N/(b₀ * norm(k₀)^2))
    density = 3N/(4π*radius^3)
    dmin = radius/300

    r = getAtoms_distribution(N, radius, dmin, :homogenous)
    atoms = (position=r, radius=radius, density=density, dmin=dmin)

    # σ_raw, timeLaserOff, IntensityLaserOff = simulateEvolution(N, atoms[:position], s, Δ₀, b₀)
	# return atoms, σ_raw, timeLaserOff, IntensityLaserOff
	timeLaserOff, IntensityLaserOff = simulateEvolution(N, atoms[:position], s, Δ₀, b₀)
	return atoms, timeLaserOff, IntensityLaserOff
end

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

	time_on, u_on, last_u_on = simple_rk4(QPC_v4_gpu, u₀_on_af, (0,100), 2000, params_on)
	# If one needs to see the population
	# population_on = computePopulation(u_on)

	u₀_off_af = last_u_on
	t_off = (time_on[end], time_on[end] + 200)
	time_off, u_off, last_u_off = simple_rk4(QPC_v4_gpu, u₀_off_af, t_off, 3500, params_off)

	# Some cleaning
	u₀_on_af = time_on = u_on = last_u_on = last_u_off = 1
	GC.gc()

	## intensity curve
	intensity_off = zeros(Float64, length(u_off))
	Gⱼₘ = geometricFactor(deg2rad(35), atoms; k₀=1)
	for j=1:length(u_off)
		intensity_off[j] = computeFieldIntensity(u_off[j], Gⱼₘ)
	end
	# return u_off, time_off, intensity_off

	u_off = 1; GC.gc()
	return time_off, intensity_off
end

function computeFieldIntensity(u, Gⱼₘ)
	Cⱼₘ = u.σ⁺σ⁻ # Following math definitions
	Cⱼₘ[diagind(Cⱼₘ)] .= (1 .+ u.σᶻ)./2 # Romain insight (no current explanation)
	# In the end I have only a real part, but I still apply real() to return a
	# single Float number and not a Complex number with null imaginary part
	intensity = real(sum(Cⱼₘ.*Gⱼₘ))
	return intensity
end
# If one need to see the population :
function computePopulation(u)
	population = (1 .+ [ real(mean(u[i].σᶻ)) for i in 1:length(u)])./2
	return population
end


## Creating everything in nested loops
function AllSimulationOneΔ(N, Δ₀, b₀_range, s_range, nRep)
    total_iterations = length(b₀_range)*length(s_range)*nRep
    p = Progress(total_iterations)
	dic_b0s = []
	for b₀ in b₀_range
		dic_saturations = []
		for s in s_range
			dic_repetitions = []
			for n in 1:nRep
				oneSimulation = computeIntensityDecay(N, Δ₀, b₀, s)
				push!(dic_repetitions, rawData(oneSimulation...))

				ProgressMeter.next!(p)
            end
			oneSaturation = saturation(s, dic_repetitions)
			push!(dic_saturations, oneSaturation)
        end
		oneb0 = opticalThickness(b₀, dic_saturations)
		push!(dic_b0s, oneb0)
    end
	oneΔ = detunning(Δ₀, dic_b0s)
	cd("/home/usp/Documents/Quantum-Pair-Correlation")
	save("N$(N)-delta$(Δ₀).jld", "N$(N)-delta$(Δ₀)", oneΔ)
	return oneΔ
end
