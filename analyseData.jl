## Reading and Ploting
using JLD
using Plots
pyplot()
using LaTeXStrings
using LsqFit
using Notifier
using Printf
using Statistics
cd("/home/usp/Documents/Quantum-Pair-Correlation")
include("structDefinitions.jl")

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
								maxIter=10_000, min_step_quality=1e-12,x_tol=1e-12)
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

## ---- begin ---
N = 3
Δ_range = [-2,-4]
dic_detunning = []
@time @progress for Δ₀ in Δ_range
	oneΔ = load("N$(N)-delta$(Δ₀).jld", "N$(N)-delta$(Δ₀)" )
    push!(dic_detunning, oneΔ)
end
OneSimulation = simulation(N, dic_detunning)

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
				y = y./y[1]
				titleName = "N=$(N), delta=$(Δ₀), b0=$(b₀), s=$(s), rep=$(n)-$(nRep)"
				try
					afit,bfit, xfit, yfit = compute_yLogFit(x, y;  xmin=220, xmax=300)
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



## Fit all Figures
cd("/home/usp/Documents/Quantum-Pair-Correlation/figures")
for (idxb₀,b₀) in enumerate(b₀_range) # FIXED VALUE OF b₀
	Plots.plot()
	for (idxΔ₀, Δ₀) in enumerate(Δ_range)
		s_a = []
		s_b = []
		for (idxs,s) in enumerate(s_range)
			s_a_average = []
			s_b_average = []
			for n in 1:nRep
				x = OneSimulation.set_Δs[idxΔ₀].set_b₀s[idxb₀].set_saturations[idxs].set_repetitions[n].time_decay
				y = OneSimulation.set_Δs[idxΔ₀].set_b₀s[idxb₀].set_saturations[idxs].set_repetitions[n].intensity_decay
				y = y./y[1]
				try
					afit,bfit, xfit, yfit = compute_yLogFit(x .-100, y; xmin=120, xmax=200)
					if bfit > 0 && bfit < 50
						push!(s_a_average, afit)
						push!(s_b_average, bfit)
					end
				catch
				end
			end
			try
				push!(s_a, (s=s, average=maximum(s_a_average)))
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
