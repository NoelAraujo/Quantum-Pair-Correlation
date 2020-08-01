#= 
Author: Noel Araujo Moreira
Created : 23/07/2020


Take the simulation in "simulation_folder" and make Plots
=#
########################################################
############## USER CAN CHANGE HERE ####################
########################################################
## Inputs
folder_path = "/home/pc2/Documents/QPC/"
simulation_folder = "2020-07-22 at 22:32:12"

#time window to create the fit
time_min = 50
time_max = 100

# creates figures of population and intensity decay
# of all clouds. All plots have a debug purpose
saveALLPlots = true # only true or false


########################################################
######## CREATES s_vs_Asub, and s_vs_tau ###############
##### Now we have more flexibility to user edition #####
########################################################
cd(folder_path)
include("struct_definitions.jl")

using DelimitedFiles
using FileIO
using LaTeXStrings
using LsqFit
using Measurements
using Statistics
ENV["GKSwstype"]="nul"
using Plots; pyplot();
function compute_yLogFit(x,y; xmin=50, xmax=100)
	shift_for_fit = findall(   (x .> xmin).*(x .< xmax)  )
    x_interval = copy(x[shift_for_fit]) # need to ajust this
    y_interval = log.(copy(y[shift_for_fit])) # take "ln()" to perform "exp()" later

    # I want to avoid outliers
    # then I make 5 fits, and choose one the looks good
	arrayFits = []
    I_fits = []
    for rep = 1:5
		fit_result = curve_fit(model_linear, x_interval, y_interval, [rand(), 2rand()];
								maxIter=10_000, min_step_quality=1e-12,x_tol=1e-12)
		errors = margin_error(fit_result, 0.05) # 2 sigma confidence
		a = measurement(fit_result.param[1], errors[1])
		b = measurement(fit_result.param[2], errors[2])
		a = exp(a) 
		b = -1/b
		
		I_fit = a.*exp.(-x_interval./b)

		push!(arrayFits, (a=copy(a), b=copy(b)))
        push!(I_fits, I_fit)
    end
	best_fit_idx = findmin([arrayFits[ii].b for ii = 1:5])[2] # arbitraty decision

    return arrayFits[best_fit_idx].a, arrayFits[best_fit_idx].b, x_interval, I_fits[best_fit_idx]
end

model_linear(t, p) = p[1] .+ t.*p[2]

using Printf
using UnicodeFun
function get_s_label(saturation)
    s_label = @sprintf("s = %1.1e",saturation)
    s_value_label = s_label[1:7]
    s_exp_label = s_label[9:end]

    raw_text = raw"10^{".*string.(s_exp_label).*"}"
    latex_text = to_latex.(raw_text)
    
    final_label = s_value_label*"⋅"*latex_text

    return final_label
end


##### Reading and defining paremeters
time_decay  = range(0,100,length=1500) # defined
s_range = load(simulation_folder*"/list_idx_stored.jld2", "s_range")
list_idx_stored = load(simulation_folder*"/list_idx_stored.jld2", "list_idx_stored")

number_saturation_steps = length(s_range)
cores_viridis = cgrad(:viridis).colors
cores_viridis = cores_viridis[  round.(Int, range(1, length(cores_viridis), length=number_saturation_steps))  ]

##### OPTIONAL
##### For each cloud, show saturation
if saveALLPlots
    folder_save_saturation_profiles = simulation_folder*"/all_saturations_profiles"
    try
        mkdir(folder_save_saturation_profiles)
        mkdir(folder_save_saturation_profiles*"/population")
        mkdir(folder_save_saturation_profiles*"/intensity")
        
        mkdir(folder_save_saturation_profiles*"/population/eps")
        mkdir(folder_save_saturation_profiles*"/intensity/eps")
        
        mkdir(folder_save_saturation_profiles*"/population/png")
        mkdir(folder_save_saturation_profiles*"/intensity/png")
    catch
    end
    
    for idx in list_idx_stored
        oneSample = load(simulation_folder*"/idx_Rep=$(idx).jld2", "array_intensitiesCurves_per_saturations")
        
        plot(xlabel=L"time \; [\Gamma]", ylabel="Population", size=(1063, 1063))
        for i=number_saturation_steps:-1:1    
            plot!(time_decay, oneSample[i].population_decay, lw=4,
                    color=cores_viridis[i], label=get_s_label(s_range[i]))
        end
        plot!(yscale=:log10, framestyle=:box, legend=:outertopright)
        plot!(guidefont = 17, tickfont=17, legendfont=14)

        savefig(folder_save_saturation_profiles*"/population/eps/idx_Rep=$(idx).eps")
        savefig(folder_save_saturation_profiles*"/population/png/idx_Rep=$(idx).png")


        plot(xlabel=L"time \; [\Gamma]", ylabel="Intensity", size=(1063, 1063))
        for i=number_saturation_steps:-1:1    
            plot!(time_decay, oneSample[i].intensity_decay, lw=4,
                    color=cores_viridis[i], label=get_s_label(s_range[i]))
        end
        plot!(yscale=:log10, framestyle=:box, legend=:outertopright)
        plot!(guidefont = 17, tickfont=17, legendfont=14)

        savefig(folder_save_saturation_profiles*"/intensity/eps/idx_Rep=$(idx).eps")
        savefig(folder_save_saturation_profiles*"/intensity/png/idx_Rep=$(idx).png")
    end
    println(" 
    ##########################################################################################
    All Plots were genered inside folder:
        $(folder_save_saturation_profiles)
    ##########################################################################################
    ")
end

##### For each saturation, take average (in log10) of all repetitions and 
##### then, compute the exponential fit in the time window specified at the begginig of this file


#------ reading data -----------
PopIntCurves_per_S = []
for idx in list_idx_stored
    global PopIntCurves_per_S
    oneSimulation = load(simulation_folder*"/idx_Rep=$(idx).jld2", "array_intensitiesCurves_per_saturations")
    push!(PopIntCurves_per_S, oneSimulation)
end

##### if you changed this value in the second script, you need to change here again
nSteps_off = 1500 

##### main loop
tuples_all_data = []
for s in 1:length(s_range)
    global tuples_all_data
    y_medio_Pop = zeros(nSteps_off)
    y_medio_Int = zeros(nSteps_off)
    nValid_Curves = 0
    for r = 1:length(list_idx_stored)
            y_Pop = PopIntCurves_per_S[r][s].population_decay
            y_Int = PopIntCurves_per_S[r][s].intensity_decay
            # y_Pop = y_Pop./y_Pop[1]
            # y_Int = y_Int./y_Int[1]

            if (all( y_Pop.> 0) && length(y_Pop)==nSteps_off && 
                all( y_Int.> 0) && length(y_Int)==nSteps_off)
                y_medio_Pop += log10.(y_Pop)
                y_medio_Int += log10.(y_Int)
                nValid_Curves += 1
            end
    end
    
    y_medio_Pop = 10.0.^(y_medio_Pop./nValid_Curves)
    y_medio_Int = 10.0.^(y_medio_Int./nValid_Curves)

    Asub_Pop,τ_Pop,xfit_Pop, yfit_Pop = compute_yLogFit(time_decay, y_medio_Pop; 
                                    xmin=time_min, xmax=time_max);
    Asub_Int,τ_Int,xfit_Int, yfit_Int = compute_yLogFit(time_decay, y_medio_Int; 
                                    xmin=time_min, xmax=time_max);
    push!(tuples_all_data, (Asub_Pop=Asub_Pop, τ_Pop=τ_Pop, Asub_Int=Asub_Int, τ_Int=τ_Int))
end

Asub_Pop = [tuples_all_data[n][:Asub_Pop] for n=1:length(s_range)]
τ_Pop = [tuples_all_data[n][:τ_Pop] for n=1:length(s_range)]

Asub_Int = [tuples_all_data[n][:Asub_Int] for n=1:length(s_range)]
τ_Int = [tuples_all_data[n][:τ_Int] for n=1:length(s_range)]


#################### Recording data ####################
##### Create folders to each data
folder_save_scaling = simulation_folder*"/scaling_figure_and_data"
try
    mkdir(folder_save_scaling)
    mkdir(folder_save_scaling*"/Pop")
    mkdir(folder_save_scaling*"/Int")
catch
end

##### Plots of Asub
plot(framestyle=:box, size=(800,800), guidefont=15, tickfont=15)
plot!(s_range, Asub_Pop, legend=:outertopright, label="",
            markershape=:circle, markersize=4, lw=3)
plot!(xlabel="s", ylabel=L"A_{sub}", scale=:log10, title="via Population")
savefig(folder_save_scaling*"/Pop/s_vs_Asub_Pop.png")

plot(framestyle=:box, size=(800,800), guidefont=15, tickfont=15)
plot!(s_range, Asub_Int, legend=:outertopright, label="",
            markershape=:circle, markersize=4, lw=3)
plot!(xlabel="s", ylabel=L"A_{sub}", scale=:log10, title="via Intensity")
savefig(folder_save_scaling*"/Int/s_vs_Asub_Int.png")

##### Plots of τ
plot(framestyle=:box, size=(800,800), guidefont=15, tickfont=15)
plot!(s_range, τ_Pop, legend=:outertopright, label="",
            markershape=:circle, markersize=4, lw=3)
plot!(xlabel="s", ylabel=L"\tau", xscale=:log10, title="via Population")
savefig(folder_save_scaling*"/Pop/s_vs_Tau_Pop.png")

plot(framestyle=:box, size=(800,800), guidefont=15, tickfont=15)
plot!(s_range, τ_Int, legend=:outertopright, label="",
            markershape=:circle, markersize=4, lw=3)
plot!(xlabel="s", ylabel=L"\tau", xscale=:log10, title="via Population")
savefig(folder_save_scaling*"/Int/s_vs_Tau_Int.png")



##### Save everything in txt files to be open easily
open(folder_save_scaling*"/Pop/s_vs_Asub_Pop_plus_Yerror.txt", "w") do io
    writedlm(io, [s_range Measurements.value.(Asub_Pop) Measurements.uncertainty.(Asub_Pop)])
end

open(folder_save_scaling*"/Pop/s_vs_tau_Pop_plus_Yerror.txt", "w") do io
    writedlm(io, [s_range Measurements.value.(τ_Pop) Measurements.uncertainty.(τ_Pop)])
end

open(folder_save_scaling*"/Int/s_vs_Asub_Int_plus_Yerror.txt", "w") do io
    writedlm(io, [s_range Measurements.value.(Asub_Int) Measurements.uncertainty.(Asub_Int)])
end

open(folder_save_scaling*"/Int/s_vs_Tau_Int_plus_Yerror.txt", "w") do io
    writedlm(io, [s_range Measurements.value.(τ_Int) Measurements.uncertainty.(τ_Int)])
end
