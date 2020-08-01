using Distances
using DifferentialEquations
using LinearAlgebra
using Plots
pyplot()
using Statistics

function QPC_v2!(du, u, p, t)
	N = p[1]
	G = p[2]
	Γⱼₘ = p[3]
	r = p[4]
	Ω⁺ = p[5]
	Ω⁻ = p[6]
	Δ₀ = p[7]
	Γ = p[8]
	k₀ = p[9]
	exclusion3D = p[10]
	Gconj = p[11]
	exclusionDiagonal = p[12]
	jₐᵤₓ,mₐᵤₓ,kₐᵤₓ = 1,2,3

	σ  = @view u[1:N]
    σᶻ = @view u[N+1:2*N]
    σ⁺ = conj.(σ)

    σᶻσ = reshape(u[2*N+1:2*N+N^2],(N,N))
    σ⁺σ = reshape(u[2*N+1+N^2:2*N+2*N^2],(N,N))
    σσ = reshape(u[2*N+1+2*N^2:2*N+3*N^2],(N,N))
    σᶻσᶻ = reshape(u[2*N+1+3*N^2:2*N+4*N^2],(N,N))

	σσᶻ = transpose(σᶻσ)
	σ⁺σᶻ = σᶻσ'

	σ⁺σσ  = exclusao3D.*truncation(σ⁺, σ, σ, σ⁺σ, σ⁺σ, σσ, N)
	σᶻσᶻσ = exclusao3D.*truncation(σᶻ, σᶻ,σ, σᶻσᶻ,σᶻσ, σᶻσ, N)
	σ⁺σσᶻ = exclusao3D.*truncation(σ⁺, σ, σᶻ,σ⁺σ, σ⁺σᶻ, σσᶻ, N)
	σᶻσσ  = exclusao3D.*truncation(σᶻ, σ, σ, σᶻσ, σᶻσ, σσ, N)
	σ⁺σᶻσ = exclusao3D.*truncation(σ⁺, σᶻ, σ, σ⁺σᶻ, σ⁺σ, σᶻσ, N)
	σᶻσ⁺σ = σ⁺σσᶻ

	# Implementation of Eq (6)-(11) in PDF
	# Eq (6)
	du[1:N] .= (im*Δ₀-1/2).*σ .+ (Ω⁺/2).*σᶻ .+ (Γ/2).*vec(sum(exclusaoDiagonal.*(G.*σᶻσ),dims=2))

	# Eq (7)
	du[N+1:2*N] .= Ω⁻.*σ .+ conj( Ω⁻.*σ ) .- Γ*(1 .+ σᶻ) .- Γ.*vec(sum(exclusaoDiagonal.*(G.*σ⁺σ) + conj.(exclusaoDiagonal.*(G.*σ⁺σ)),dims=2))

	# Eq (8)
	du[2*N+1:2*N+N^2] .= reshape( ( (im*Δ₀ - 3Γ/2).*σᶻσ
					.- Γ*ones(N).*transpose(σ)
					.+ σσ.*Ω⁻ .- σ⁺σ.*Ω⁺ .+ 0.5.*σᶻσᶻ.*transpose(Ω⁺)
					.- Γ.*sum( reshape(G,(N,1,N)).*σ⁺σσ .+ (reshape(Gconj,(N,1,N)).*permutedims(σ⁺σσ,[kₐᵤₓ,mₐᵤₓ,jₐᵤₓ])), dims=3)
					.+(Γ/2).*transpose(reshape(sum(G.*permutedims(σᶻσᶻσ,[jₐᵤₓ,kₐᵤₓ,mₐᵤₓ]), dims=2),(N,N)))
					.- Γ*σσᶻ.*Γⱼₘ - (Γ/2).*Gconj.*σ
					),(N^2)

	)
	# Eq (9)
	du[2*N+N^2+1:2*N+2*N^2] .= reshape(
								(  -Γ.*σ⁺σ .- 0.5*( σᶻσ.*Ω⁻ - σ⁺σᶻ.*transpose(Ω⁺) )
									.+ 0.5(reshape(sum(reshape(Gconj,(N,1,N)).*permutedims(σ⁺σσᶻ, [kₐᵤₓ,mₐᵤₓ,jₐᵤₓ]), dims=3), (N,N)).+reshape(sum(reshape(G,(1,N,N)).*permutedims(σ⁺σσᶻ, [jₐᵤₓ,kₐᵤₓ,mₐᵤₓ]), dims=3), (N,N)))
									.+ 0.25Γ*(transpose(σᶻ).*G + σᶻ.*Gconj )
									.+ 0.5Γ*Γⱼₘ.*σᶻσᶻ
								), (N^2)
								)
	# Eq (10)
	du[2*N+1+2*N^2:2*N+3*N^2] .= reshape(
				( (2im*Δ₀ - Γ)*σσ .+ 0.5( σᶻσ.*Ω⁺ .+  transpose(σᶻσ).*transpose(Ω⁺))
				.+ (Γ/2)sum(reshape(G,(N,1,N)).*(σᶻσσ) .+ reshape(G,(1,N,N)).*permutedims(σᶻσσ,[mₐᵤₓ,jₐᵤₓ,kₐᵤₓ]) ,dims=3)
				)
				,(N^2))
	# Eq (11)
	du[2*N+1+3*N^2:2*N+4*N^2] .= reshape(
		(
			.-Γ*(σᶻ.*ones(1,N) + ones(N,1).*transpose(σᶻ) + 2σᶻσᶻ)
			.+ transpose(σᶻσ).*Ω⁻ .+ σᶻσ.*transpose(Ω⁻) .+ conj.(transpose(σᶻσ).*Ω⁻ .+ σᶻσ.*transpose(Ω⁻))
			.-2Γ*reshape(real(sum(reshape(G,(N,1,N)).*(σ⁺σᶻσ) .+ reshape(G,(1,N,N)).*permutedims(σ⁺σσᶻ,[kₐᵤₓ,jₐᵤₓ,mₐᵤₓ]) ,dims=3)),(N,N))
			+ 4Γ*Γⱼₘ.*real(σ⁺σ)
		), (N^2)
	)
	return nothing
end



v21 = deepcopy(u₀_off)
dv21 = deepcopy(u₀_off)
QPC_v2!(dv21, v21, p_on_v2,0.0)

@profiler solve(prob_off,RK4(), dt=(1/2)^4, adaptive=false)


function truncation(A, B, C, AB, AC, BC, N)
	N = length(A)
	f = (-2.0.*reshape(A*transpose(B),(N,N,1)).*reshape(C,(1,1,N))
			+ reshape(BC,(1,N,N)).*reshape(A,(N,1,1))
	        + reshape(AB,(N,N,1)).*reshape(C,(1,1,N))
	        + reshape(AC,(N,1,N)).*reshape(B,(1,N,1)))
	return f
end

# ------------- parameters -------------
X =  [-0.1141,  -0.0507,   -0.3428,   -0.1587,    0.1385]
Y =  [-0.0775,    0.6600,   -0.2585,   -0.6381,    0.7721]
Z =  [0.4698,   -0.0815,   -0.5313,   -0.3636,   -0.1457]
r = [X Y Z]

N = 5
Δ₀ = -5.0
Γ = 1
k₀ = [0,0,1]

R_jk = Distances.pairwise(Euclidean(), r, r, dims=1)
G = Array{Complex{Float64}}(undef, N, N)
@. G = -(Γ/2)*exp(1im*R_jk)/(1im*R_jk)
G[LinearAlgebra.diagind(G)] .= 0
G .= -2G

Gconj = conj.(G)
Γⱼₘ =  real.(G)

exclusionDiagonal = ones(Int8,N,N) .- I(N)
exclusion3D = ones(Int8, N, N) .- LinearAlgebra.I(N)
exclusion3D = (exclusion3D .* reshape(exclusion3D,(N,1,N))).*reshape(exclusion3D,(1,N,N))


Ω₀ = 10.0
Ω⁺ = im*Ω₀.*exp.( im.*[dot(k₀, r[i,:]) for i=1:N] )
Ω⁻ = im*Ω₀.*exp.( -im.*[dot(k₀, r[i,:]) for i=1:N] )
p_on_v2 = []
push!(p_on_v2, N)
push!(p_on_v2, G)
push!(p_on_v2, Γⱼₘ)
push!(p_on_v2, r)
push!(p_on_v2, Ω⁺)
push!(p_on_v2, Ω⁻)
push!(p_on_v2, Δ₀)
push!(p_on_v2, Γ)
push!(p_on_v2, k₀)
push!(p_on_v2, exclusion3D)
push!(p_on_v2, Gconj)
push!(p_on_v2, exclusionDiagonal)

Ω₀ = 0.0
Ω⁺ = im*Ω₀.*exp.( im.*[dot(k₀, r[i,:]) for i=1:N] )
Ω⁻ = im*Ω₀.*exp.( -im.*[dot(k₀, r[i,:]) for i=1:N] )
p_off_v2 = []
push!(p_off_v2, N)
push!(p_off_v2, G)
push!(p_off_v2, Γⱼₘ)
push!(p_off_v2, r)
push!(p_off_v2, Ω⁺)
push!(p_off_v2, Ω⁻)
push!(p_off_v2, Δ₀)
push!(p_off_v2, Γ)
push!(p_off_v2, k₀)
push!(p_off_v2, exclusion3D)
push!(p_off_v2, Gconj)
push!(p_off_v2, exclusionDiagonal)

u₀_on = zeros(ComplexF64, 2*N+4*N^2)
u₀_on[N+1:2*N] .= -1
u₀_on[2*N+3*N^2+1:2*N+4*N^2] .= +1
u₀_on[2*N+3*N^2+1:N+1:2*N+4*N^2] .= 0.0

# offd = ones(N, N) .- LinearAlgebra.I(N)
# Exclu3 = (offd .* reshape(offd,(N,1,N))).*reshape(offd,(1,N,N))




# -------------------------------------------------
tspan_on = (0.0, 100.0)
prob_on = ODEProblem(QPC_v2!,u₀_on,tspan_on,p_on_v2)
@time sol_on = DifferentialEquations.solve(prob_on, RK4(), dt=(1/2)^4, adaptive=false)
population_on = 0.5 .+ 0.5*[ real(mean(sol_on.u[i][N+1:2*N])) for i in eachindex(sol_on.u)]
# plot(sol_on.t, abs.(population_on), framestyle=:box, xlabel="t",label="", ylabel="0.5 + 0.5 <Population>")

tspan_off = (100.0, 140.0)
u₀_off = sol_on.u[end]
prob_off = ODEProblem(QPC_v2!,u₀_off,tspan_off,p_off_v2)
@time sol_off = DifferentialEquations.solve(prob_off,RK4(), dt=(1/2)^4, adaptive=false)
population_off = 0.5 .+ 0.5*[ real(mean(sol_off.u[i][N+1:2*N])) for i in eachindex(sol_off.u)]
# plot!(sol_off.t, abs.(population_off), framestyle=:box, xlabel="t",label="", ylabel="0.5 + 0.5 <Population>")
# plot!(yscale=:log10)

plot(timeEvolution,pop_σᶻEvolution, label="", lw=6, c=:black)
plot!(timeDecay, pop_σᶻDecay, label="", lw=6, c=:black)
xlabel!("time"); ylabel!("population")

plot!(sol_off.t, abs.(population_off), framestyle=:box, label="",lw=2, linestyle=:dot, c=:red)
plot!(sol_on.t, abs.(population_on), framestyle=:box, label="",lw=2, linestyle=:dot, c=:red)
xlims!(0,10)


# -------------------------------------
using JLD
# save("benchmarkCurve.jld",
#         "time_on", sol_on.t, "population_sigmaZ_on", abs.(population_on),
#         "time_off", sol_off.t, "population_sigmaZ_off", abs.(population_off))

d = load("benchmarkCurve.jld")
timeEvolution = d["time_on"]
pop_σᶻEvolution = d["population_sigmaZ_on"]

timeDecay = d["time_off"]
pop_σᶻDecay = d["population_sigmaZ_off"]
