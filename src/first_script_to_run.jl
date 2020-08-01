#= 
Author: Noel Araujo Moreira
Created : 22/07/2020


In this file the user define the atomic cloud properties
=#

########################################################
############## USER CAN CHANGE HERE ####################
########################################################
## Inputs
folder_path = "/home/pc2/Documents/QPC/"

N = 100 # simulations with large number of particles worth only if GPU
b₀ = 5 # the best optical thickness for simulation was 5
nClouds = 25


#= 
########################################################
Parameters of simulation that SHOULD NOT be changed
If you really need, be sure about your decision impact
########################################################
=#
using LinearAlgebra
Γ = 1
k₀ = [0,0,1]
radius = sqrt(6N/(b₀ * norm(k₀)^2))
density = 3N/(4π*radius^3)

# dmin is ad hoc, that is, i don't know why, but made
# intensity curves to converge
dmin = 0.5*density^(1/3) 


########################################################
## Actually creating data
cd(folder_path)

using Distances
include("funcsTo_Create_Atoms.jl")

array_atoms = []
for i=1:nClouds
    r = getAtoms_distribution(N, radius, dmin, :homogenous)
    atoms = (position=r)
    push!(array_atoms, deepcopy(atoms))
end


using FileIO # needs JLD2 package installed
save("simulation_parameters.jld2", 
    Dict("N" => N, "b0" => b₀, "Gamma"=> Γ, "folder_path"=>folder_path,
        "k0"=>k₀, "density"=>density, "dmin"=>dmin,
        "atomic_clouds"=>array_atoms))


println(" 
########################################################
Data Created. 

Look for:
    Folder: $(pwd())
    Filename: simulation_parameters.jld2
########################################################
")
