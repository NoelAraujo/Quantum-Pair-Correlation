using ArrayFire
using Distances
using LinearAlgebra
using Plots; pyplot()
using SpecialFunctions
using Statistics

include("functionsToCreateAtoms.jl")
include("QPC_Evolution_GPU.jl")
include("structDefinitions.jl")

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


# Function to make fit
function compute_yLogFit(x,y;xmin=120, xmax=200)
	shift_for_fit = findall(   (x .> xmin).*(x .< xmax)  )
    x_interval = copy(x[shift_for_fit]) # need to ajust this
    y_interval = log.(copy(y[shift_for_fit]))

    # I want to avoid outliers
    # then I make n fits, and choose one the looks good
    arrayFits = zeros(5,2)
    I_fits = []
    for rep = 1:5
		fit_result = curve_fit(model_linear, x_interval, y_interval, [100rand(), 3rand()];
								maxIter=1_000_000, min_step_quality=1e-12,x_tol=1e-12)
		a = exp(fit_result.param[1])
		b = -1/fit_result.param[2]
		I_fit = a.*exp.(-x_interval./b)

        arrayFits[rep,1] = a
        arrayFits[rep,2] = b
        push!(I_fits, I_fit)
    end
    best_fit_idx = findmin(arrayFits[:,2])[2] # arbitraty decision

    return arrayFits[best_fit_idx,1], arrayFits[best_fit_idx,2], x_interval, I_fits[best_fit_idx]
end
model_linear(t, p) = p[1] .+ t.*p[2]
## inputs
N = 5
Δ_range = [-2,-4]
b₀_range = [1,5]
s_range = 10.0.^(range(log10(10e-6), log10(10), length=3))
nRep = 2
ones1N = AFArray(ones(1,N))
onesN1 = AFArray(ones(N,1))
onesN = AFArray(ones(N))

## Creating everything in nested loops
dic_detunning = []
@time @progress for Δ₀ in Δ_range
    dic_b0s = []
    @progress for b₀ in b₀_range
        dic_saturations = []
        @progress for s in s_range
            dic_repetitions = []
            @progress for n in 1:nRep
                oneSimulation = computeIntensityDecay(N, Δ₀, b₀, s)
                push!(dic_repetitions, rawData(oneSimulation...))
            end
            oneSaturation = saturation(s, dic_repetitions)
            push!(dic_saturations, oneSaturation)
        end
        oneb0 = opticalThickness(b₀, dic_saturations)
        push!(dic_b0s, oneb0)
    end
    oneΔ = detunning(Δ₀, dic_b0s)
    push!(dic_detunning, oneΔ)
end

OneSimulation = simulation(N, dic_detunning)

using JLD
cd("/home/usp/Documents/Quantum-Pair-Correlation")
save("N$(N).jld", "N$(N)", OneSimulation)
## Reading and Ploting

using JLD
using Plots
pyplot()
using LaTeXStrings
using LsqFit
using Printf
using Statistics
cd("/home/usp/Documents/Quantum-Pair-Correlation")
OneSimulation = load("N$(N).jld", "N$(N)")

# Example
xteste = OneSimulation.set_Δs[1].set_b₀s[1].set_saturations[end].set_repetitions[1].time_decay
yteste = OneSimulation.set_Δs[1].set_b₀s[1].set_saturations[end].set_repetitions[1].intensity_decay
yteste = yteste./yteste[1]

ae,be,xfit, yfit = compute_yLogFit(xteste, yteste; xmin=220, xmax=300);
Plots.plot(xteste .- 100, yteste, label="",  xlabel="time", ylabel="Intensity", lw=4, yscale=:log10)
Plots.plot!(xfit .- 100, yfit, label="a=$(round(ae,digits=3)), b=$(round(be,digits=3))",
 			c=:black, linestyle=:dash, lw=3)


## Fit all Figures
cd("/home/usp/Documents/Quantum-Pair-Correlation/figures")
@progress for (idxΔ₀,Δ₀) in enumerate(Δ_range)
	try
		mkdir("delta=$(Δ₀)")
		cd("delta=$(Δ₀)")
	catch
		rm("delta=$(Δ₀)", recursive=true)
		mkdir("delta=$(Δ₀)")
		cd("delta=$(Δ₀)")
	end
    @progress for (idxb₀,b₀) in enumerate(b₀_range)
		try
			mkdir("b0=$(b₀)")
			cd("b0=$(b₀)")
		catch
			rm("b0=$(b₀)")
			mkdir("b0=$(b₀)")
			cd("b0=$(b₀)")
		end
        @progress for (idxs,s) in enumerate(s_range)
			try
				mkdir("s=$(s)")
				cd("s=$(s)")
			catch
				rm("s=$(s)")
				mkdir("s=$(s)")
				cd("s=$(s)")
			end
            @progress for n in 1:nRep
				# pwd() |> display
                x = OneSimulation.set_Δs[idxΔ₀].set_b₀s[idxb₀].set_saturations[idxs].set_repetitions[n].time_decay
				y = OneSimulation.set_Δs[idxΔ₀].set_b₀s[idxb₀].set_saturations[idxs].set_repetitions[n].intensity_decay
				titleName = "N=$(N), delta=$(Δ₀), b0=$(b₀), s=$(s), rep=$(n)-$(nRep)"
				try
					afit,bfit, xfit, yfit = compute_yLogFit(x,Float64.(y);  xmin=220, xmax=300)
					Plots.plot(x .-100, y, title=titleName, label="",  xlabel="time", ylabel="Intensity", lw=4, yscale=:log10)
					Plots.plot!(xfit .-100, yfit, label="a="*@sprintf("%.3e", afit)*", b=$(round(bfit,digits=3))", c=:black, linestyle=:dash, lw=3)
					Plots.savefig(titleName*".png")
				catch
					Plots.scatter(rand(50), rand(50), title=titleName*" - Failed", xlabel="time", ylabel="Intensity", lw=4)
					savefig(titleName*" - Failed.png")
				end # try
            end
			cd("..")
        end
		cd("..")
    end
	cd("..")
end
using Notifier
notify("Created all Plots")


## Fit all Figures
cd("/home/usp/Documents/Quantum-Pair-Correlation/figures")
@progress for (idxb₀,b₀) in enumerate(b₀_range) # FIXED VALUE OF b₀
	Plots.plot()
	@progress for (idxΔ₀, Δ₀) in enumerate(Δ_range)
		s_a = []
		s_b = []
		@progress for (idxs,s) in enumerate(s_range)
			s_a_average = []
			s_b_average = []
			@progress for n in 1:nRep
				x = OneSimulation.set_Δs[idxΔ₀].set_b₀s[idxb₀].set_saturations[idxs].set_repetitions[n].time_decay
				y = OneSimulation.set_Δs[idxΔ₀].set_b₀s[idxb₀].set_saturations[idxs].set_repetitions[n].intensity_decay
				y = y./y[1]
				try
					afit,bfit, xfit, yfit = compute_yLogFit(x,Float64.(y);  xmin=220, xmax=300)
					push!(s_a_average, afit)
					push!(s_b_average, bfit)
				catch
				end
			end
			try
				push!(s_a, (s=s, average=minimum(s_a_average)))
				push!(s_b, (s=s, average=maximum(s_b_average)))
			catch
			end
		end
		x_axis = [s_a[n][:s] for n=1:length(s_a)]

		ya_axis = [s_a[n][:average] for n=1:length(s_a)]
		Plots.plot!(x_axis, ya_axis, markershape=:circle, markersize=5, label=@sprintf("Δ₀ = %.1f", Δ₀), scale=:log10, title="b₀=$(b₀)")

		# yb_axis = [s_b[n][:average] for n=1:length(s_b)]
		# Plots.plot!(x_axis, yb_axis, markershape=:circle, markersize=5, label=@sprintf("Δ₀ = %.1f", Δ₀), scale=:log10,title="b₀=$(b₀)")
	end
	Plots.plot!(xlabel="saturation",
		framestyle=:box, size=(800,600),
		guidefont=15, tickfont=15, legendfont=12)
		try
			ylabel!("Population") #|> display
			savefig("N=50, b0=$(b₀), Population.png")

			# ylabel!("Lifetime")
			# savefig("N=$(N), b0=$(b₀), Lifetime.png")
		catch
			nothing
		end
end
notify("Created Population/Lifetime")
