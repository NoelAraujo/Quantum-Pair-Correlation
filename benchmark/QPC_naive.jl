using Distances
using DifferentialEquations
using LinearAlgebra
using Plots
pyplot()
using Statistics

function QPC_v1(u, p, t)
	N = p[1]
	G = p[2]
	Γⱼₘ = p[3]
	r = p[4]
	Ω₀ = p[5]
	Δ₀ = p[6]
	Γ = p[7]
	k₀ = p[8]
	Exclu3 = p[9]

	σ  = u[1:N]
    σᶻ = u[N+1:2*N]
    σ⁺ = conj.(σ)

    σᶻσ = reshape(u[2*N+1:2*N+N^2],(N,N))
    σ⁺σ = reshape(u[2*N+1+N^2:2*N+2*N^2],(N,N))
    σσ = reshape(u[2*N+1+2*N^2:2*N+3*N^2],(N,N))
    σᶻσᶻ = reshape(u[2*N+1+3*N^2:2*N+4*N^2],(N,N))

	σσᶻ = transpose(σᶻσ)
	σ⁺σᶻ = σᶻσ'

	σ⁺σσ  = Exclu3.*truncation(σ⁺, σ, σ, σ⁺σ, σ⁺σ, σσ)
	σᶻσᶻσ = Exclu3.*truncation(σᶻ, σᶻ,σ, σᶻσᶻ,σᶻσ, σᶻσ)
	σ⁺σσᶻ = Exclu3.*truncation(σ⁺, σ, σᶻ,σ⁺σ, σ⁺σᶻ, σσᶻ)
	σᶻσσ  = Exclu3.*truncation(σᶻ, σ, σ, σᶻσ, σᶻσ, σσ)
	σ⁺σᶻσ = Exclu3.*truncation(σ⁺, σᶻ, σ, σ⁺σᶻ, σ⁺σ, σᶻσ)
	σᶻσ⁺σ = σ⁺σσᶻ

	# Implementation of Eq (6)-(11) in PDF
	# Eq (6)
	dₜ_σ = zeros(ComplexF64, N)
    for j = 1:N
    	dₜ_σ[j] = ( (im*Δ₀ - Γ/2)*σ[j]
					+ im*(Ω₀/2)*exp(im*dot(k₀, r[j,:])).*σᶻ[j]
					+ (Γ/2)*sum(m ≠ j ? G[j,m]*σᶻσ[j,m] : 0 for m=1:N)
		)
	end

	# Eq (7)
	dₜ_σᶻ = zeros(ComplexF64, N)
	for j = 1:N
    	dₜ_σᶻ[j] = ( 	im*Ω₀*( exp.(-im*dot(k₀, r[j,:]))*σ[j] - conj(exp.(-im*dot(k₀, r[j,:]))*σ[j]) )
						- Γ*(1 + σᶻ[j])
						- Γ*sum(m ≠ j ? G[j,m]*σ⁺σ[j,m] + conj(G[j,m]*σ⁺σ[j,m]) : 0 for m=1:N)
		)
	end

	# Eq (8)
	dₜ_σᶻσ = zeros(ComplexF64, N,N)
	for j = 1:N
		for m = 1:N
				dₜ_σᶻσ[j,m] = ( (im*Δ₀ - 3Γ/2)*σᶻσ[j,m]
								- Γ*σ[m]
								+ im*Ω₀*( exp(-im*dot(k₀, r[j,:]))*σσ[j,m] - exp(im*dot(k₀, r[j,:]))*σ⁺σ[j,m] + 0.5exp(im*dot(k₀, r[m,:]))*σᶻσᶻ[j,m] )
								- Γ*sum(((k≠j)*(k≠m)) ? G[j,k].*σ⁺σσ[j,m,k] + conj(G[j,k])*σ⁺σσ[k,m,j] : 0 for k=1:N)
								+ 0.5Γ*sum((k≠j)&&(k≠m) ? G[m,k]*σᶻσᶻσ[m,j,k] : 0 for k=1:N)
								- Γ*Γⱼₘ[j,m]*σσᶻ[j,m] - 0.5Γ*conj(G[j,m])*σ[j]
				)
		end
	end
	dₜ_σᶻσ[diagind(dₜ_σᶻσ)] .= zero(eltype(dₜ_σᶻσ))

	# Eq (9)
	dₜ_σ⁺σ = zeros(ComplexF64, N,N)
	for j = 1:N
		for m = 1:N
				dₜ_σ⁺σ[j,m] = ( -Γ*σ⁺σ[j,m] - 0.5im*Ω₀*(  exp(-im*dot(k₀, r[j,:]))*σᶻσ[j,m]  - exp(im*dot(k₀, r[m,:]))*σ⁺σᶻ[j,m] )
								+ 0.5Γ*sum((k≠j)&&(k≠m) ? conj(G[j,k])*σ⁺σσᶻ[k,m,j] + G[m,k]*σ⁺σσᶻ[j,k,m] : 0 for k=1:N)
								+ 0.25Γ*(G[j,m]*σᶻ[m] + conj(G[j,m])*σᶻ[j] )
								+ 0.5Γ*Γⱼₘ[j,m]*σᶻσᶻ[j,m]
				)
		end
	end
	dₜ_σ⁺σ[diagind(dₜ_σ⁺σ)] .= zero(eltype(dₜ_σ⁺σ))

	# Eq (10)
	dₜ_σσ = zeros(ComplexF64, N,N)
	for j = 1:N
		for m = 1:N
				dₜ_σσ[j,m] = ( (2im*Δ₀ - Γ)*σσ[j,m] + 0.5im*Ω₀*(  exp(im*dot(k₀, r[j,:]))*σᶻσ[j,m]  + exp(im*dot(k₀, r[m,:]))*σᶻσ[m,j] )
								+ 0.5Γ*sum((k≠j)&&(k≠m) ? G[j,k]*σᶻσσ[j,m,k] + G[m,k]*σᶻσσ[m,j,k] : 0 for k=1:N)
				)
		end
	end
	dₜ_σσ[diagind(dₜ_σσ)] .= zero(eltype(dₜ_σσ))

	# Eq (11)
	dic_first_exp = []
	dic_second_exp = []
	dₜ_σᶻσᶻ = zeros(ComplexF64, N,N)
	for j = 1:N
		for m = 1:N
				dₜ_σᶻσᶻ[j,m] = (-Γ*( σᶻ[j] + σᶻ[m] + 2σᶻσᶻ[j,m])
								+ im*Ω₀*(  exp(-im*dot(k₀, r[j,:]))*σᶻσ[m,j]  + exp(-im*dot(k₀, r[m,:]))*σᶻσ[j,m] - conj(exp(-im*dot(k₀, r[j,:]))*σᶻσ[m,j]  + exp(-im*dot(k₀, r[m,:]))*σᶻσ[j,m]) )
								# - Γ*sum((k≠j)&&(k≠m) ? G[j,k]*σ⁺σᶻσ[j,m,k] + G[m,k]*σᶻσ⁺σ[j,m,k] + conj(G[j,k]*σ⁺σᶻσ[j,m,k] + G[m,k]*σᶻσ⁺σ[j,m,k]) : 0 for k=1:N)
								- 2Γ*real(sum((k≠j)&&(k≠m) ? G[j,k]*σ⁺σᶻσ[j,m,k] : 0 for k=1:N) + sum(transpose(G[:,k]).*permutedims(σ⁺σσᶻ, [3 1 2])[:,:,k] for k=1:N, dims=3)[j,m])
								+ 2Γ*Γⱼₘ[j,m]*( σ⁺σ[j,m] + conj(σ⁺σ[j,m]))
				)
		end
	end
	dₜ_σᶻσᶻ[diagind(dₜ_σᶻσᶻ)] .= zero(eltype(dₜ_σᶻσᶻ))

	du = zeros(eltype(u), size(u))
	du[1:N] .= dₜ_σ
    du[N+1:2*N] .= dₜ_σᶻ

    du[2*N+1:2*N+N^2] .= dₜ_σᶻσ[:]
    du[2*N+1+N^2:2*N+2*N^2] .= dₜ_σ⁺σ[:]
    du[2*N+1+2*N^2:2*N+3*N^2] .= dₜ_σσ[:]
    du[2*N+1+3*N^2:2*N+4*N^2] .= dₜ_σᶻσᶻ[:]

	return du
end


function truncation(A, B, C, AB, AC, BC)
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

Γⱼₘ =  real.(G)

offd = ones(N, N) .- LinearAlgebra.I(N)
Exclu3 = (offd .* reshape(offd,(N,1,N))).*reshape(offd,(1,N,N))

Ω₀ = 10.0
p_on = []
push!(p_on, N)
push!(p_on, G)
push!(p_on, Γⱼₘ)
push!(p_on, r)
push!(p_on, copy(Ω₀))
push!(p_on, Δ₀)
push!(p_on, Γ)
push!(p_on, k₀)
push!(p_on, Exclu3)

Ω₀ = 0.0
p_off = []
push!(p_off, N)
push!(p_off, G)
push!(p_off, Γⱼₘ)
push!(p_off, r)
push!(p_off, copy(Ω₀))
push!(p_off, Δ₀)
push!(p_off, Γ)
push!(p_off, k₀)
push!(p_off, Exclu3)

u₀_on = zeros(ComplexF64, 2*N+4*N^2)
u₀_on[N+1:2*N] .= -1
u₀_on[2*N+3*N^2+1:2*N+4*N^2] .= +1
u₀_on[2*N+3*N^2+1:N+1:2*N+4*N^2] .= 0.0

# -------------------------------------------------
tspan_on = (0.0, 100.0)
prob_on = ODEProblem(QPC_v1,u₀_on,tspan_on,p_on)
@time sol_on = DifferentialEquations.solve(prob_on)
population_on = 0.5 .+ 0.5*[ real(mean(sol_on.u[i][N+1:2*N])) for i in eachindex(sol_on.u)]
plot(sol_on.t, abs.(population_on), framestyle=:box, xlabel="t",label="", ylabel="0.5 + 0.5 <Population>")

tspan_off = (100.0, 140.0)
u₀_off = sol_on.u[end]
prob_off = ODEProblem(QPC_v1,u₀_off,tspan_off,p_off)
@time sol_off = DifferentialEquations.solve(prob_off)
population_off = 0.5 .+ 0.5*[ real(mean(sol_off.u[i][N+1:2*N])) for i in eachindex(sol_off.u)]
plot!(sol_off.t, abs.(population_off), framestyle=:box, xlabel="t",label="", ylabel="0.5 + 0.5 <Population>")
# plot!(yscale=:log10)

using DelimitedFiles
open("QPC_vJulia.txt", "w") do io
	writedlm(io, [sol_off.t abs.(population_off)])
end
