#= 
Author: Noel Araujo Moreira
Created : 22/07/2020


In this file the user define laser properties and 
angle of intensity measuremet
=#
########################################################
############## USER CAN CHANGE HERE ####################
########################################################
## Inputs
folder_path = "/home/pc2/Documents/QPC/"

sₘᵢₙ = 1e-4
sₘₐₓ = 10
number_saturation_steps = 10

Δ₀ = -2.0
θ = 35 # angle in degress where the intensity is measured


########################################################
## Parameters of simulation that SHOULD NOT be changed
## Really, do not change them now
########################################################


######### Loading packages and functions
cd(folder_path)


using ArrayFire
using Distances
using FileIO
using LinearAlgebra
using ProgressMeter
using SpecialFunctions
using Statistics

include("struct_definitions.jl")
include("funcsTo_simulate.jl") 
include("func_QPC_and_RK.jl")


######### Loading simulation parameters

# arrays "ones" below are used inside QPC Evolution, and 
# to avoid pass them as another parameter, I declared them
# globally
N =  load(folder_path*"simulation_parameters.jld2", "N")
const ones1N = AFArray(ones(1,N))
const onesN1 = AFArray(ones(N,1))
const onesN = AFArray(ones(N))

# reading parameter from first script
Γ = load(folder_path*"simulation_parameters.jld2", "Gamma")
k₀ = load(folder_path*"simulation_parameters.jld2", "k0")

atomic_clouds =  load(folder_path*"simulation_parameters.jld2", "atomic_clouds")
nClouds = length(atomic_clouds)

######### saturation is created now and saved
s_range = 10.0.^(range(log10(sₘᵢₙ), log10(sₘₐₓ), length=number_saturation_steps))

######### folder specific to each delta
using Dates
folderTo_save_data = Dates.format(now(), "yyyy-mm-dd at HH:MM:SS")
try
	mkdir(folderTo_save_data)
	# I want to avoid opportunities to overlap of information
	# Copyng the simulation_parameters file, and Deleting it
	# will allow me to run different simulations at the same computer 
	# (when possible in the future)
	# In addition to force the user to properly specify the data to be analysed 
	cp("simulation_parameters.jld2", folderTo_save_data*"/simulation_parameters.jld2")
	rm("simulation_parameters.jld2") 
catch
	@error("Folder where data would saved could not be created. Simulation cannot continue.")	
end



######### begin simulation
nValidCurvesStored = 0
list_idx_stored = []
for i ∈ 1:nClouds
	global nValidCurvesStored, list_idx_stored
	println("Δ : $(round(Δ₀, digits=1)), rep : $(i)/$(nClouds) | nValidCurvesStored: $(nValidCurvesStored)")
	
	atoms = atomic_clouds[i]
	array_intensitiesCurves_per_saturations = create_CurvesDecays_in_saturation_interval(atoms, Δ₀, s_range; 
												nSteps_on=1500, nSteps_off=1500, 
												θ = 35, Γ=1, k₀=[0,0,1])

	if length(array_intensitiesCurves_per_saturations)==length(s_range)
		save(folderTo_save_data*"/idx_Rep=$(i).jld2", "array_intensitiesCurves_per_saturations", 
													array_intensitiesCurves_per_saturations)
		push!(list_idx_stored, i)
		nValidCurvesStored += 1
	end
	array_intensitiesCurves_per_saturations = 1
	GC.gc()

	######### used during processing phase
	save(folderTo_save_data*"/list_idx_stored.jld2", 
		Dict("list_idx_stored" => list_idx_stored,
		"s_range"=>s_range, "Delta"=>Δ₀, "theta"=>θ ))
end


println(" 
##########################################################################################
Simulation Finished.

From $(nClouds) Clouds, only $(nValidCurvesStored) produced all saturation curves

Look for:
	Folder: $(pwd())/$(folderTo_save_data)
	Index of valid Curves: list_idx_stored.jld2
##########################################################################################
")

GC.gc()
exit()