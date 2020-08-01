""" create_CurvesDecays_in_saturation_interval(atoms, Δ₀, s_range; 
							nSteps_on=1500, nSteps_off=1500, 
							θ = 35, Γ=1, k₀=[0,0,1])

θ in degrees
Population and Intensity curves evolves as follows:
	⋅ laser is on  in interval t = [0,100]
	⋅ laser is off in interval t = [100,200]
Those limits were the best to guarantee result stability
	⋅ laser is on : no guarantees after t > 120
	⋅ laser is off : no guarantees after t > 200
"""
function create_CurvesDecays_in_saturation_interval(atoms, Δ₀, saturation_range; 
							nSteps_on=1500, nSteps_off=1500, 
							θ = 35, Γ=1, k₀=[0,0,1])
	
    total_iterations = length(saturation_range)*(nSteps_on + nSteps_off)
    p = Progress(total_iterations)
	
	N = size(atoms,1)
	
	saturations_curves = []
	for saturation in saturation_range
		@time population_decay, intensity_decay = simulateEvolution(N, atoms, saturation, Δ₀, p;
														nSteps_on=nSteps_on, 
														nSteps_off=nSteps_off, 
														θ=θ, Γ=Γ, k₀=k₀)
		
		if all( intensity_decay.> 0) && length(intensity_decay)==nSteps_off
			push!(saturations_curves, DecayCurves(population_decay, intensity_decay))
		else
			println("Simulation failed in saturation s = $(saturation)")
			println("Initiating next simulation ...")
			break
		end
    end
	return saturations_curves
end

function simulateEvolution(N, atoms, s, Δ₀, p;
				nSteps_on=1500, nSteps_off=1500, 
				θ = 35, Γ=1, k₀=[0,0,1])

	# Laser's auxiliary matrix and vectors
	G, Gconj, Γⱼₘ = getScalarKernel(atoms, N)
	exclusionDiagonal, exclusion3D = getExclusionMatrices(N)
	Ω⁺_on, Ω⁻_on = getLaser(atoms, s, Γ, Δ₀, k₀, N)

	params_on = []; push!(params_on,
								N, G, Gconj, Γⱼₘ,
								exclusionDiagonal, exclusion3D,
								Ω⁺_on, Ω⁻_on,
								Δ₀, Γ)

	s_off = 0.0 # no laser
	Ω⁺_af_off, Ω⁻_af_off = getLaser(atoms, s_off, Γ, Δ₀, k₀, N)
	params_off = []; push!(params_off,
								N, G, Gconj, Γⱼₘ,
								exclusionDiagonal, exclusion3D,
								Ω⁺_af_off, Ω⁻_af_off,
								Δ₀, Γ)

	## Initial State : all σ⁻ down, all σᶻ up
	u₀_on = AFArray(zeros(ComplexF64, 2*N+4*N^2))
	u₀_on[N+1:2*N] .= -1
	u₀_on[2*N+3*N^2+1:2*N+4*N^2] .= +1
	u₀_on[2*N+3*N^2+1:N+1:2*N+4*N^2] .= 0.0
	Gⱼₘ = geometricFactor(deg2rad(θ), atoms; k₀=k₀)

	time_laser_should_be_on = (0,100)
	u_on, last_u_on = integrate_via_rk4(QPC, u₀_on, time_laser_should_be_on, nSteps_on, params_on, p, Gⱼₘ)
	if length(u_on) < nSteps_on
		# Some cleaning
		u₀_on = u_on = last_u_on = 1
		GC.gc()

		# I could return population(u_on) and its intensity
		# but is an unecessary data movement
		# Therefore, I return two "ones" in the right data type (Array{Float64,1})
		return  ones(1), ones(1) 
	end

	u₀_off = last_u_on
	t_off = (time_laser_should_be_on[end], time_laser_should_be_on[end] + 100)
	u_off, last_u_off = integrate_via_rk4(QPC, u₀_off, t_off,  nSteps_off, params_off, p, Gⱼₘ; isDecayCurve=true)

	## Retuning values
	population_off = computePopulation(u_off)
	
	# I could return this inside "integrate_via_rk4".
	# But I think that it makes that code more uggly.
	# Since is a fast opertation, I compute again here.
	intensity_off = zeros(Float64, length(u_off))
	for j=1:length(u_off)
		intensity_off[j] = computeFieldIntensity(u_off[j], Gⱼₘ)
	end
	
	# Some cleaning
	u₀_on = u_on = last_u_on = last_u_off = u_off = last_u_off = 1
	GC.gc()
	return  population_off, intensity_off
end

"""
Needed to Convert Population into Light intensity
"""
function geometricFactor(θ, atoms; k₀=[0,0,1])
	N = size(atoms, 1)
	Gⱼₘ = zeros(ComplexF64, N,N)
	for m=1:N
        for j=1:N # j is inside because Julia has collumn major access to matrices
			rⱼ = atoms[j,:]
            rₘ = atoms[m,:]
			rⱼₘ = rⱼ .- rₘ
			argument_bessel = norm(k₀)*sin(θ)*sqrt( rⱼₘ[1]^2 + rⱼₘ[2]^2)
			Gⱼₘ[j,m] = exp(im*norm(k₀)*rⱼₘ[3]*cos(θ))*besselj0(argument_bessel)
        end
    end
	return Gⱼₘ
end


function computeFieldIntensity(u, Gⱼₘ)
	Cⱼₘ = u.σ⁺σ⁻ # Following math definitions
	Cⱼₘ[diagind(Cⱼₘ)] .= (1 .+ u.σᶻ)./2 # Romain insight (no current explanation)
	# In the end I have only a real part, but I still apply real() to return a
	# single Float number and not a Complex number with null imaginary part
	intensity = real(sum(Cⱼₘ.*Gⱼₘ))
	return intensity
end

"""
computes the average population for an array of "sigmas_info"s
"""
function computePopulation(u)
	population = (1 .+ [ real(mean(u[i].σᶻ)) for i in 1:length(u)])./2
	return vec(population)
end


