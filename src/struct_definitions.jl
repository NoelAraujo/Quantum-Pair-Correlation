mutable struct DecayCurves
#= 
Since I'm not using Adpative RungeKutta, all intensity profiles have the exact same time values
To avoid store this redundat data, I store only the intensity and population.
But I maintain the code here to future improvements if needed
=#
    # time_decay 
    population_decay::Array{Float64,1}
    intensity_decay::Array{Float64,1}
end

"""
sigmas_info stores σ⁻, σᶻ and σ⁺σ⁻, needed to population and intensity calculations
"""
mutable struct sigmas_info
    σ⁻::Array{ComplexF64,1}
    σᶻ::Array{ComplexF64,1}
    σ⁺σ⁻::Array{ComplexF64,2}
end
