## Build params
function getScalarKernel(r, N; Γ = 1)
    R_jk = Distances.pairwise(Euclidean(), r, r, dims=1)
    G = Array{Complex{Float64}}(undef, N, N)
    @. G = -(Γ/2)*exp(1im*R_jk)/(1im*R_jk)
    G[LinearAlgebra.diagind(G)] .= 0
    G_af = AFArray(-2G)

    Gconj_af = conj(G_af)
    Γⱼₘ_af =  real(G_af)

    return G_af, Gconj_af, Γⱼₘ_af
end

function getExclusionMatrices(N)
    # creating matrix of "Bool" needs less space than "Integer"
    exclusionDiagonal_af = AFArray(ones(Bool,N,N))
    exclusionDiagonal_af[diagind(exclusionDiagonal_af)] .= false

    exclusion3D_af = AFArray(ones(Bool, N, N))
    exclusion3D_af[diagind(exclusion3D_af)] .= false
    exclusion3D_af = (exclusion3D_af .* reshape(exclusion3D_af,(N,1,N))).*reshape(exclusion3D_af,(1,N,N))

    return exclusionDiagonal_af, exclusion3D_af
end

function getLaser(r, s, Γ, Δ₀, k₀, N)
    Ω₀ = sqrt(0.5s*(Γ^2 + 4(Δ₀^2)))
    Ω⁺_af = AFArray(im*Ω₀.*exp.( im.*[dot(k₀, r[i,:]) for i=1:N] ))
    Ω⁻_af = AFArray(im*Ω₀.*exp.( -im.*[dot(k₀, r[i,:]) for i=1:N] ))
    return Ω⁺_af, Ω⁻_af
end

## QPC definitions
function QPC_v4_gpu(t, u, params)
    N = params[1]
    G_af = params[2]
    Gconj_af = params[3]
    Γⱼₘ_af = params[4]
    exclusionDiagonal_af = params[5]
    exclusion3D_af = params[6]

    Ω⁺_af = params[7]
    Ω⁻_af = params[8]

    Δ₀ = params[9]
    Γ = params[10]

    jₐᵤₓ,mₐᵤₓ,kₐᵤₓ = 1,2,3

	σ  = u[1:N]
    σᶻ = u[N+1:2*N]
    σ⁺ = conj.(σ)

    σᶻσ = reshape(u[2*N+1:2*N+N^2],(N,N))
    σ⁺σ = reshape(u[2*N+1+N^2:2*N+2*N^2],(N,N))
    σσ = reshape(u[2*N+1+2*N^2:2*N+3*N^2],(N,N))
    σᶻσᶻ = reshape(u[2*N+1+3*N^2:2*N+4*N^2],(N,N))

	σσᶻ = transpose(σᶻσ)
	σ⁺σᶻ = σᶻσ'

	σ⁺σσ  = exclusion3D_af.*truncation(σ⁺, σ,  σ, σ⁺σ,  σ⁺σ, σσ,  N)
	σᶻσᶻσ = exclusion3D_af.*truncation(σᶻ, σᶻ, σ, σᶻσᶻ, σᶻσ, σᶻσ, N)
	σ⁺σσᶻ = exclusion3D_af.*truncation(σ⁺, σ,  σᶻ,σ⁺σ,  σ⁺σᶻ,σσᶻ, N)
	σᶻσσ  = exclusion3D_af.*truncation(σᶻ, σ,  σ, σᶻσ,  σᶻσ, σσ,  N)
	σ⁺σᶻσ = exclusion3D_af.*truncation(σ⁺, σᶻ, σ, σ⁺σᶻ, σ⁺σ, σᶻσ, N)
	σᶻσ⁺σ = σ⁺σσᶻ

	# Implementation of Eq (6)-(11) in PDF
	# Eq (6)
	dₜ_σ = (im*Δ₀-1/2).*σ .+ (Ω⁺_af/2).*σᶻ .+ (Γ/2).*vec(sum(exclusionDiagonal_af.*(G_af.*σᶻσ),dims=2))

	# Eq (7)
	 dₜ_σᶻ = Ω⁻_af.*σ .+ conj( Ω⁻_af.*σ ) .- Γ*(1 .+ σᶻ) .- Γ.*vec(sum(exclusionDiagonal_af.*(G_af.*σ⁺σ) + conj.(exclusionDiagonal_af.*(G_af.*σ⁺σ)),dims=2))

	# Eq (8)
	dₜ_σᶻσ = reshape(
            (
                (im*Δ₀ - 3Γ/2).*σᶻσ
        		.- Γ*onesN.*transpose(σ)
        		.+ σσ.*Ω⁻_af .- σ⁺σ.*Ω⁺_af .+ 0.5.*σᶻσᶻ.*transpose(Ω⁺_af)
        		.- Γ.*sum( reshape(G_af,(N,1,N)).*σ⁺σσ .+ (reshape(Gconj_af,(N,1,N)).*permutedims(σ⁺σσ,[kₐᵤₓ,mₐᵤₓ,jₐᵤₓ])), dims=3)
        		.+(Γ/2).*transpose(reshape(sum(G_af.*permutedims(σᶻσᶻσ,[jₐᵤₓ,kₐᵤₓ,mₐᵤₓ]), dims=2),(N,N)))
        		.- Γ*σσᶻ.*Γⱼₘ_af - (Γ/2).*Gconj_af.*σ
        	),(N^2)
	)
	# Eq (9)
	dₜ_σ⁺σ = reshape(
			(
                -Γ.*σ⁺σ .- 0.5*( σᶻσ.*Ω⁻_af - σ⁺σᶻ.*transpose(Ω⁺_af) )
				.+ 0.5(reshape(sum(reshape(Gconj_af,(N,1,N)).*permutedims(σ⁺σσᶻ, [kₐᵤₓ,mₐᵤₓ,jₐᵤₓ]), dims=3), (N,N)).+reshape(sum(reshape(G_af,(1,N,N)).*permutedims(σ⁺σσᶻ, [jₐᵤₓ,kₐᵤₓ,mₐᵤₓ]), dims=3), (N,N)))
				.+ 0.25Γ*(transpose(σᶻ).*G_af + σᶻ.*Gconj_af )
				.+ 0.5Γ*Γⱼₘ_af.*σᶻσᶻ
			), (N^2)
	)
	# Eq (10)
	dₜ_σσ = reshape(
        	(
                (2.0*im*Δ₀ - Γ)*σσ .+ 0.5( σᶻσ.*Ω⁺_af .+  transpose(σᶻσ).*transpose(Ω⁺_af))
            	.+ (Γ/2)sum(reshape(G_af,(N,1,N)).*(σᶻσσ) .+ reshape(G_af,(1,N,N)).*permutedims(σᶻσσ,[mₐᵤₓ,jₐᵤₓ,kₐᵤₓ]) ,dims=3)
        	),(N^2)
    )
	# Eq (11)
	dₜ_σᶻσᶻ = reshape(
    		(
    			.-Γ*(σᶻ.*ones1N + onesN1.*transpose(σᶻ) + 2σᶻσᶻ)
    			.+ transpose(σᶻσ).*Ω⁻_af .+ σᶻσ.*transpose(Ω⁻_af) .+ conj.(transpose(σᶻσ).*Ω⁻_af .+ σᶻσ.*transpose(Ω⁻_af))
    			.-2Γ*reshape(real(sum(reshape(G_af,(N,1,N)).*(σ⁺σᶻσ) .+ reshape(G_af,(1,N,N)).*permutedims(σ⁺σσᶻ,[kₐᵤₓ,jₐᵤₓ,mₐᵤₓ]) ,dims=3)),(N,N))
    			+ 4Γ*Γⱼₘ_af.*real(σ⁺σ)
            ),(N^2)
	)

	ab = join(1, dₜ_σ, dₜ_σᶻ)
	ac = join(1, ab ,  dₜ_σᶻσ)
	ad = join(1, ac ,  dₜ_σ⁺σ)
	ae = join(1, ad ,  dₜ_σσ)
	du = join(1, ae ,  dₜ_σᶻσᶻ)

	return du
end

Base.permutedims(x::AFArray, newDims::Array{Int64,1}) = reorder(x, newDims[1]-1,newDims[2]-1,newDims[3]-1,3)

function truncation(A, B, C, AB, AC, BC, N)
	f = (-2.0.*reshape(A*transpose(B),(N,N,1)).*reshape(C,(1,1,N))
			+ reshape(BC,(1,N,N)).*reshape(A,(N,1,1))
	        + reshape(AB,(N,N,1)).*reshape(C,(1,1,N))
	        + reshape(AC,(N,1,N)).*reshape(B,(1,N,1)))
	return f
end


## EDO Solver Related

function simple_rk4(f::Function, u₀, t₀t₁, n::Integer, params)
    t₀,t₁ = t₀t₁[1], t₀t₁[2]
    vt = Vector{Float64}(undef, n + 1)
    vt[1] = t = t₀
	u = u₀
	vu = []
	sigmadata = sigmas_info(Array(u₀[1:N]), Array(u₀[N+1:2*N]), Array(reshape(u₀[2*N+1+N^2:2*N+2*N^2],(N,N))))
	push!(vu, sigmadata)
    h = (t₁ - t₀) / n

    for i in 1:n
		# println(i,"/",n)
        k₁ = h * f(t, u, params)
        k₂ = h * f(t + 0.5h, u + 0.5k₁, params)
        k₃ = h * f(t + 0.5h, u + 0.5k₂, params)
        k₄ = h * f(t + h   , u + k₃,    params)
		u = u + (k₁ + 2k₂ + 2k₃ + k₄) / 6
		if any(isnan.(u))
			break
		end
		sigmadata = sigmas_info(Array(u[1:N]), Array(u[N+1:2*N]), Array(reshape(u[2*N+1+N^2:2*N+2*N^2],(N,N))))
		push!(vu, sigmadata) # I store only the parts that I know that are important
        vt[i + 1] = t = t₀ + i*h
    end
    return vt, vu, u
end

## Convert Population into Light intensity
function intensityInPoint(atoms, σ⁻, n̂; k₀=1)
    intensity = zero(eltype(σ⁻))
    σ⁺ = conj.(σ⁻)
    for j=1:N
        rⱼ = atoms[j,:]
        for m=1:N
            rₘ = atoms[m,:]
                intensity += σ⁺[j]σ⁻[m]*exp(-im*k₀*dot(n̂, rⱼ - rₘ))
        end
    end
    return intensity
end

function getIntensityFromOneTime(atoms, σ⁻)
    intensity = zero(eltype(σ⁻))
    ϕ_range = 1:5:360 # integrate over ϕ
    for (idx, ϕ) in enumerate(ϕ_range)
        spherical_coordinate = [rad2deg(ϕ), rad2deg(35), 5*radius]
        n̂ = sph2cart(spherical_coordinate)
        intensity += intensityInPoint(atoms, σ⁻, n̂)
    end
    return real(intensity)
end

function geometricFactor(θ, atoms; k₀=1)
	N = size(atoms, 1)
	Gⱼₘ = zeros(ComplexF64, N,N)
	for m=1:N
        for j=1:N # j is inside because Julia has collumn major access to matrices
			rⱼ = atoms[j,:]
            rₘ = atoms[m,:]
			rⱼₘ = rⱼ .- rₘ
			argument_bessel = k₀*sin(θ)*sqrt( rⱼₘ[1]^2 + rⱼₘ[2]^2)
			Gⱼₘ[j,m] = exp(im*k₀*rⱼₘ[3]*cos(θ))*besselj0(argument_bessel)
        end
    end
	return Gⱼₘ
end
