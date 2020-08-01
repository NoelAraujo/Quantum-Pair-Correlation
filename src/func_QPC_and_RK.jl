## Build params
function getScalarKernel(r, N; Γ = 1)
    R_jk = Distances.pairwise(Euclidean(), r, r, dims=1)
    G = Array{Complex{Float64}}(undef, N, N)
    @. G = -(Γ/2)*exp(1im*R_jk)/(1im*R_jk)
    G[LinearAlgebra.diagind(G)] .= 0
    G = AFArray(-2G)

    Gconj = conj(G)
    Γⱼₘ =  real(G)

    return G, Gconj, Γⱼₘ
end

function getExclusionMatrices(N)
    # creating matrix of "Bool" uses less memory than "Integer"
    exclusionDiagonal = AFArray(ones(Bool,N,N))
    exclusionDiagonal[diagind(exclusionDiagonal)] .= false

    exclusion3D = AFArray(ones(Bool, N, N))
    exclusion3D[diagind(exclusion3D)] .= false
    exclusion3D = (exclusion3D .* reshape(exclusion3D,(N,1,N))).*reshape(exclusion3D,(1,N,N))

    return exclusionDiagonal, exclusion3D
end

# Positive and Negative components of the Laser are created once
function getLaser(r, s, Γ, Δ₀, k₀, N)
    Ω₀ = sqrt(0.5s*(Γ^2 + 4(Δ₀^2)))
    Ω⁺ = AFArray(im*Ω₀.*exp.( im.*[dot(k₀, r[i,:]) for i=1:N] ))
    Ω⁻ = AFArray(im*Ω₀.*exp.( -im.*[dot(k₀, r[i,:]) for i=1:N] ))
    return Ω⁺, Ω⁻
end

"""
QPC(t, u, params)

Follows Nicolas's equations in attached PDf file.
"""
function QPC(t, u, params)
    N = params[1]
    G = params[2]
    Gconj = params[3]
    Γⱼₘ_af = params[4]
    exclusionDiagonal = params[5]
    exclusion3D = params[6]

    Ω⁺ = params[7]
    Ω⁻ = params[8]

    Δ₀ = params[9]
    Γ = params[10]

    jₐᵤₓ,mₐᵤₓ,kₐᵤₓ = 1,2,3

	σ⁻  = u[1:N]
    σᶻ = u[N+1:2*N]
    σ⁺ = conj.(σ⁻)

    σᶻσ⁻ = reshape(u[2*N+1:2*N+N^2],(N,N))
    σ⁺σ⁻ = reshape(u[2*N+1+N^2:2*N+2*N^2],(N,N))
    σ⁻σ⁻ = reshape(u[2*N+1+2*N^2:2*N+3*N^2],(N,N))
    σᶻσᶻ = reshape(u[2*N+1+3*N^2:2*N+4*N^2],(N,N))

	σ⁻σᶻ = transpose(σᶻσ⁻)
	σ⁺σᶻ = σᶻσ⁻'

	σ⁺σ⁻σ⁻ = exclusion3D.*truncation(σ⁺, σ⁻, σ⁻, σ⁺σ⁻, σ⁺σ⁻, σ⁻σ⁻, N)
	σᶻσᶻσ⁻ = exclusion3D.*truncation(σᶻ, σᶻ, σ⁻, σᶻσᶻ, σᶻσ⁻, σᶻσ⁻, N)
	σ⁺σ⁻σᶻ = exclusion3D.*truncation(σ⁺, σ⁻, σᶻ, σ⁺σ⁻, σ⁺σᶻ, σ⁻σᶻ, N)
	σᶻσ⁻σ⁻ = exclusion3D.*truncation(σᶻ, σ⁻, σ⁻, σᶻσ⁻, σᶻσ⁻, σ⁻σ⁻, N)
	σ⁺σᶻσ⁻ = exclusion3D.*truncation(σ⁺, σᶻ, σ⁻, σ⁺σᶻ, σ⁺σ⁻, σᶻσ⁻, N)
	σᶻσ⁺σ⁻ = σ⁺σ⁻σᶻ

	# Implementation of Eq (6)-(11) in PDF
	# Eq (6)
	dₜ_σ⁻ = (im*Δ₀-1/2).*σ⁻ .+ (Ω⁺/2).*σᶻ .+ (Γ/2).*vec(sum(exclusionDiagonal.*(G.*σᶻσ⁻),dims=2))

	# Eq (7)
	 dₜ_σᶻ = Ω⁻.*σ⁻ .+ conj( Ω⁻.*σ⁻ ) .- Γ*(1 .+ σᶻ) .- Γ.*vec(sum(exclusionDiagonal.*(G.*σ⁺σ⁻) + conj.(exclusionDiagonal.*(G.*σ⁺σ⁻)),dims=2))

	# Eq (8)
	dₜ_σᶻσ⁻ = reshape(
            (
                (im*Δ₀ - 3Γ/2).*σᶻσ⁻
        		.- Γ*onesN.*transpose(σ⁻)
        		.+ σ⁻σ⁻.*Ω⁻ .- σ⁺σ⁻.*Ω⁺ .+ 0.5.*σᶻσᶻ.*transpose(Ω⁺)
        		.- Γ.*sum( reshape(G,(N,1,N)).*σ⁺σ⁻σ⁻ .+ (reshape(Gconj,(N,1,N)).*permutedims(σ⁺σ⁻σ⁻,[kₐᵤₓ,mₐᵤₓ,jₐᵤₓ])), dims=3)
        		.+ (Γ/2).*transpose(reshape(sum(G.*permutedims(σᶻσᶻσ⁻,[jₐᵤₓ,kₐᵤₓ,mₐᵤₓ]), dims=2),(N,N)))
        		.- Γ*σ⁻σᶻ.*Γⱼₘ_af - (Γ/2).*Gconj.*σ⁻
        	),(N^2)
	)

	# Eq (9)
	dₜ_σ⁺σ⁻ = reshape(
			(
                -Γ.*σ⁺σ⁻ .- 0.5*( σᶻσ⁻.*Ω⁻ - σ⁺σᶻ.*transpose(Ω⁺) )
				.+ 0.5(reshape(sum(reshape(Gconj,(N,1,N)).*permutedims(σ⁺σ⁻σᶻ, [kₐᵤₓ,mₐᵤₓ,jₐᵤₓ]), dims=3), (N,N)).+reshape(sum(reshape(G,(1,N,N)).*permutedims(σ⁺σ⁻σᶻ, [jₐᵤₓ,kₐᵤₓ,mₐᵤₓ]), dims=3), (N,N)))
				.+ 0.25Γ*(transpose(σᶻ).*G + σᶻ.*Gconj )
				.+ 0.5Γ*Γⱼₘ_af.*σᶻσᶻ
			), (N^2)
	)
	# Eq (10)
	dₜ_σ⁻σ⁻ = reshape(
        	(
                (2.0*im*Δ₀ - Γ)*σ⁻σ⁻ .+ 0.5( σᶻσ⁻.*Ω⁺ .+  transpose(σᶻσ⁻).*transpose(Ω⁺))
            	.+ (Γ/2)sum(reshape(G,(N,1,N)).*(σᶻσ⁻σ⁻) .+ reshape(G,(1,N,N)).*permutedims(σᶻσ⁻σ⁻,[mₐᵤₓ,jₐᵤₓ,kₐᵤₓ]) ,dims=3)
        	),(N^2)
    )
	# Eq (11)
	dₜ_σᶻσᶻ = reshape(
    		(
    			.-Γ*(σᶻ.*ones1N + onesN1.*transpose(σᶻ) + 2σᶻσᶻ)
    			.+ transpose(σᶻσ⁻).*Ω⁻ .+ σᶻσ⁻.*transpose(Ω⁻) .+ conj.(transpose(σᶻσ⁻).*Ω⁻ .+ σᶻσ⁻.*transpose(Ω⁻))
    			.-2Γ*reshape(real(sum(reshape(G,(N,1,N)).*(σ⁺σᶻσ⁻) .+ reshape(G,(1,N,N)).*permutedims(σ⁺σ⁻σᶻ,[kₐᵤₓ,jₐᵤₓ,mₐᵤₓ]) ,dims=3)),(N,N))
    			+ 4Γ*Γⱼₘ_af.*real(σ⁺σ⁻)
            ),(N^2)
	)
	
	# "join" with AFArray needs extra implementation from part of the user.
	# Inside the AFArray folder package, 
	# with root permission, one needs to included this code :

	# function join(dim::Integer, _first::AFArray{T,N}, 
	# 	_second::AFArray{T,N}) where {T,N}
	# 		out = RefValue{af_array}(0)
	# 		_error(ccall((:af_join,af_lib), af_err,
	# 		(Ptr{af_array},
	# 		Cint, af_array,af_array),
	# 		out, Cint(dim - 1),_first.arr, _second.arr))
	# 		AFArray{T,N}(out[])
	# 	end
	ab = join(1, dₜ_σ⁻, dₜ_σᶻ)
	ac = join(1, ab ,   dₜ_σᶻσ⁻)
	ad = join(1, ac ,   dₜ_σ⁺σ⁻)
	ae = join(1, ad ,   dₜ_σ⁻σ⁻)
	du = join(1, ae ,   dₜ_σᶻσᶻ)

	return du
end
using Base
Base.permutedims(x::AFArray, newDims::Array{Int64,1}) = reorder(x, newDims[1]-1,newDims[2]-1,newDims[3]-1,3)

function truncation(A, B, C, AB, AC, BC, N)
	f = (-2.0.*reshape(A*transpose(B),(N,N,1)).*reshape(C,(1,1,N))
			+ reshape(BC,(1,N,N)).*reshape(A,(N,1,1))
	        + reshape(AB,(N,N,1)).*reshape(C,(1,1,N))
	        + reshape(AC,(N,1,N)).*reshape(B,(1,N,1)))
	return f
end


"""
EDO Solver is Standard Runge Kutta of 4th order

Do not use adpative methods, because those methods required high
tolerances values to become pass the benchmarks - that were not recorded.

Chage the input value "nSteps" instead of creating other other solver
"""
function integrate_via_rk4(f::Function, u₀, t₀t₁, nSteps::Integer, params, p, Gⱼₘ; isDecayCurve = false)
    t₀,t₁ = t₀t₁[1], t₀t₁[2]
	t = t₀
	u = u₀
	vu = []

	N = size(Gⱼₘ,1) # Gⱼₘ is square, does not matter which size I choose
	sigmadata = make_sigma_from_u(u₀, N)
	push!(vu, sigmadata)

    h = (t₁ - t₀) / nSteps # time step for given interval

	# When decay curve, I verify if the curve is always decreasing, if not, stop
	percentages = range(0, nSteps, step=round(Int, nSteps/10))
    for i in 1:(nSteps-1)  # "-1" because "vu" already have the initial condition
		u = nextRKterm(t, u, params, h, f)

		sigmadata = make_sigma_from_u(u, N)
		push!(vu, sigmadata) # I will return only the parts that I know that are important
		ProgressMeter.next!(p; showvalues = [(:isDecayCurve, isDecayCurve), (:i,round(100*i/nSteps,digits=3))])

		# Conditions to stop if I found some curve if with anything different than positive values
		if any(isnan.(u)) || computeFieldIntensity(sigmadata, Gⱼₘ) < 0
			break
		end
		# *** The author found *** that population cuves may diverge,
		# To avoid integrating them, this function finish 
		# Therefore, the overall simulation is faster
		if isDecayCurve
			if i in percentages
				Field_before = computeFieldIntensity(vu[end - round(Int, nSteps/20)], Gⱼₘ)
				Field_now = computeFieldIntensity(vu[end], Gⱼₘ)
				if any(isnan.(u)) || Field_now > Field_before
					break
				end
			end
		end
	end
	# This function returns an array with only the values of σ⁻, σᶻ, σ⁺σ⁻
	# However, I still need to return the complete state "u", from the last interaction
    return vu, u 
end

function make_sigma_from_u(u, N)
	σ⁻ = Array(u[1:N])
	σᶻ = Array(u[N+1:2*N])
	σ⁺σ⁻ = Array(reshape(u[2*N+1+N^2:2*N+2*N^2],(N,N)))
	sigmadata = sigmas_info(σ⁻, σᶻ, σ⁺σ⁻)
	return sigmadata
end

function nextRKterm(t, u, params, h, f::Function)
	k₁ = h*f(t, u, params)
	k₂ = h*f(t + 0.5h, u + 0.5k₁, params)
	k₃ = h*f(t + 0.5h, u + 0.5k₂, params)
	k₄ = h*f(t + h   , u + k₃,    params)
	next_u = u + (k₁ + 2k₂ + 2k₃ + k₄) / 6
	return next_u
end