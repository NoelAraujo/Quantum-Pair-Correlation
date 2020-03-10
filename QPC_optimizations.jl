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
	Exclu3 = p[10]
	Gconj = p[11]

	σ  = @view u[1:N]
    σᶻ = @view u[N+1:2*N]
    σ⁺ = conj.(σ)

    σᶻσ = reshape(u[2*N+1:2*N+N^2],(N,N))
    σ⁺σ = reshape(u[2*N+1+N^2:2*N+2*N^2],(N,N))
    σσ = reshape(u[2*N+1+2*N^2:2*N+3*N^2],(N,N))
    σᶻσᶻ = reshape(u[2*N+1+3*N^2:2*N+4*N^2],(N,N))

	σσᶻ = transpose(σᶻσ)
	σ⁺σᶻ = σᶻσ'

	σ⁺σσ  = truncation(σ⁺, σ, σ, σ⁺σ, σ⁺σ, σσ, N)
	σᶻσᶻσ = truncation(σᶻ, σᶻ,σ, σᶻσᶻ,σᶻσ, σᶻσ, N)
	σ⁺σσᶻ = Exclu3.*truncation(σ⁺, σ, σᶻ,σ⁺σ, σ⁺σᶻ, σσᶻ, N)
	σᶻσσ  = truncation(σᶻ, σ, σ, σᶻσ, σᶻσ, σσ, N)
	σ⁺σᶻσ = truncation(σ⁺, σᶻ, σ, σ⁺σᶻ, σ⁺σ, σᶻσ, N)
	σᶻσ⁺σ = σ⁺σσᶻ

	# Implementation of Eq (6)-(11) in PDF
	# Eq (6)
	dₜ_σ = zeros(ComplexF64, N)
    for j = 1:N
		du[j] = ( (im*Δ₀ - Γ/2)*σ[j]
					+ (Ω⁺[j]/2).*σᶻ[j]
					+ (Γ/2)*sum(m ≠ j ? G[j,m]*σᶻσ[j,m] : 0 for m=1:N)
		)
	end

	# Eq (7)
    for j = 1:N
		du[j+N] = ( 	  Ω⁻[j]*σ[j] + conj( Ω⁻[j]*σ[j] )
                        - Γ*(1 + σᶻ[j])
                        - Γ*sum(m ≠ j ? G[j,m]*σ⁺σ[j,m] + conj(G[j,m]*σ⁺σ[j,m]) : 0 for m=1:N)
        )

    end

	# Eq (8)
	for j = 1:N
        m_range = deleteat!(collect(1:N), j)
        for m in m_range
				du[2N + j + (m-1)*N] = ( (im*Δ₀ - 3Γ/2)*σᶻσ[j,m]
								- Γ*σ[m]
								+ ( Ω⁻[j]*σσ[j,m] - Ω⁺[j]*σ⁺σ[j,m] + 0.5*Ω⁺[m]*σᶻσᶻ[j,m] )
								- Γ*sum(((k≠j)*(k≠m)) ? G[j,k].*σ⁺σσ[j,m,k] + Gconj[j,k]*σ⁺σσ[k,m,j] : 0 for k=1:N)
								+ 0.5Γ*sum((k≠j)&&(k≠m) ? G[m,k]*σᶻσᶻσ[m,j,k] : 0 for k=1:N)
								- Γ*Γⱼₘ[j,m]*σσᶻ[j,m] - 0.5Γ*Gconj[j,m]*σ[j]
				)
		end
	end

	# Eq (9)
	for j = 1:N
        m_range = deleteat!(collect(1:N), j)
        for m in m_range
				du[2*N+N^2 + j + (m-1)*N] = ( -Γ*σ⁺σ[j,m] - 0.5*( Ω⁻[j]*σᶻσ[j,m]  - Ω⁺[m]*σ⁺σᶻ[j,m] )
								+ 0.5Γ*sum((k≠j)&&(k≠m) ? Gconj[j,k]*σ⁺σσᶻ[k,m,j] + G[m,k]*σ⁺σσᶻ[j,k,m] : 0 for k=1:N)
								+ 0.25Γ*(G[j,m]*σᶻ[m] + Gconj[j,m]*σᶻ[j] )
								+ 0.5Γ*Γⱼₘ[j,m]*σᶻσᶻ[j,m]
				)
		end
	end

	# Eq (10)
	for j = 1:N
        m_range = deleteat!(collect(1:N), j)
        for m in m_range
				du[2*N+2*N^2 + j + (m-1)*N] = ( (2im*Δ₀ - Γ)*σσ[j,m] + 0.5(  Ω⁺[j]*σᶻσ[j,m]  + Ω⁺[m]*σᶻσ[m,j] )
								+ 0.5Γ*sum((k≠j)&&(k≠m) ? G[j,k]*σᶻσσ[j,m,k] + G[m,k]*σᶻσσ[m,j,k] : 0 for k=1:N)
				)
		end
	end

	# Eq (11)
	for j = 1:N
        m_range = deleteat!(collect(1:N), j)
        for m in m_range
				du[2*N+3*N^2 + j + (m-1)*N] = (-Γ*( σᶻ[j] + σᶻ[m] + 2σᶻσᶻ[j,m])
								+  Ω⁻[j]*σᶻσ[m,j]  + Ω⁻[m]*σᶻσ[j,m] + conj( Ω⁻[j]*σᶻσ[m,j]  + Ω⁻[m]*σᶻσ[j,m] )
								# - 2Γ*real(sum((k≠j)&&(k≠m) ? G[j,k]*σ⁺σᶻσ[j,m,k] + G[m,k]*σᶻσ⁺σ[j,m,k] : 0 for k=1:N))
								- 2Γ*real(sum((k≠j)&&(k≠m) ? G[j,k]*σ⁺σᶻσ[j,m,k] : 0 for k=1:N) + sum(transpose(G[:,k]).*permutedims(σ⁺σσᶻ, [3 1 2])[:,:,k] for k=1:N, dims=3)[j,m])
								+ 4Γ*Γⱼₘ[j,m]*real( σ⁺σ[j,m])
				)
		end
	end
end


function truncation(A, B, C, AB, AC, BC, N)
	# N = length(A)
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

offd = ones(N, N) .- LinearAlgebra.I(N)
Exclu3 = (offd .* reshape(offd,(N,1,N))).*reshape(offd,(1,N,N))


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
push!(p_on_v2, Exclu3)
push!(p_on_v2, Gconj)

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
push!(p_off_v2, Exclu3)
push!(p_off_v2, Gconj)

u₀_on = zeros(ComplexF64, 2*N+4*N^2)
u₀_on[N+1:2*N] .= -1
u₀_on[2*N+3*N^2+1:2*N+4*N^2] .= +1
u₀_on[2*N+3*N^2+1:N+1:2*N+4*N^2] .= 0.0

# -------------------------------------------------
tspan_on = (0.0, 100.0)
prob_on = ODEProblem(QPC_v2!,u₀_on,tspan_on,p_on_v2)
@time sol_on = DifferentialEquations.solve(prob_on)
population_on = 0.5 .+ 0.5*[ real(mean(sol_on.u[i][N+1:2*N])) for i in eachindex(sol_on.u)]
# plot(sol_on.t, abs.(population_on), framestyle=:box, xlabel="t",label="", ylabel="0.5 + 0.5 <Population>")

tspan_off = (100.0, 140.0)
u₀_off = sol_on.u[end]
prob_off = ODEProblem(QPC_v2!,u₀_off,tspan_off,p_off_v2)
@time sol_off = DifferentialEquations.solve(prob_off)
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
