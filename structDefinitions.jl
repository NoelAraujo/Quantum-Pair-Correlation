mutable struct simulation
    N_value
    set_Δs
end
mutable struct detunning
    Δ_value
    set_b₀s
end
mutable struct opticalThickness
    b₀_value
    set_saturations
end
mutable struct saturation
    saturation_value
    set_repetitions
end
mutable struct rawData
    atoms_positions
	# σ
    time_decay
    intensity_decay
end

# Used to store only the components of σ that I need
mutable struct sigmas_info
	σ⁻
	σᶻ
	σ⁺σ⁻
end

## Bottom-Up example
# oneData = rawData(rand(5,3), range(0,200, step=0.5), rand(400))
# twoData = rawData(rand(5,3), range(0,200, step=0.5), rand(400))
#
# dic_repetitions = []
# push!(dic_repetitions, oneData)
# push!(dic_repetitions, twoData)
# aSaturation = saturation(1e-6, dic_repetitions)
# bSaturation = saturation(1e-5, dic_repetitions)
#
# dic_saturations = []
# push!(dic_saturations, aSaturation)
# push!(dic_saturations, bSaturation)
# oneb₀ = opticalThickness(1, dic_saturations)
# twob₀ = opticalThickness(2, dic_saturations)
#
# dic_b0 = []
# push!(dic_b0, oneb₀)
# push!(dic_b0, twob₀)
#
# dic_detunning = []
# push!(dic_detunning,detunning(-2, dic_b0))
# push!(dic_detunning,detunning(-4, dic_b0))
# push!(dic_detunning,detunning(-8, dic_b0))
#
# N5 = simulation(5, dic_detunning)
