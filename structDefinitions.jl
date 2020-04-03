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
    time_decay
    intensity_decay
end

## Bottom-Up example
oneData = rawData(rand(5,3), range(0,200, step=0.5), rand(400))
twoData = rawData(rand(5,3), range(0,200, step=0.5), rand(400))

dic_repetitions = []
push!(dic_repetitions, oneData)
push!(dic_repetitions, twoData)
aSaturation = saturation(1e-6, dic_repetitions)
bSaturation = saturation(1e-5, dic_repetitions)

dic_saturations = []
push!(dic_saturations, aSaturation)
push!(dic_saturations, bSaturation)
oneb₀ = opticalThickness(1, dic_saturations)
twob₀ = opticalThickness(2, dic_saturations)

dic_b0 = []
push!(dic_b0, oneb₀)
push!(dic_b0, twob₀)

dic_detunning = []
push!(dic_detunning,detunning(-2, dic_b0))
push!(dic_detunning,detunning(-4, dic_b0))
push!(dic_detunning,detunning(-8, dic_b0))

N5 = simulation(5, dic_detunning)

## Creating everything in nested loops
N = 5
Δ_range = [-2,-4,-8]
b₀_range = [1,5,10,15]
s_range = [1e-6,1e-3,1e-2,1e-1,1e0,1e1]
nRep = 3
ones1N = AFArray(ones(1,N))
onesN1 = AFArray(ones(N,1))
onesN = AFArray(ones(N))


dic_detunning = []
@progress for Δ₀ in Δ_range
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

N5 = simulation(N, dic_detunning)


using LinearAlgebra
function computeIntensityDecay(N, Δ₀, b₀, s)
    Γ = 1
    k₀ = [0,0,1]
    radius = sqrt(6N/(b₀ * norm(k₀)^2))
    density = 3N/(4π*radius^3)
    dmin = radius/300

    r = getAtoms_distribution(N, radius, dmin, :homogenous)
    atoms = (position=r, radius=radius, density=density, dmin=dmin)

    timeLaserOff, IntensityLaserOff = simulateEvolution(N, atoms[:position], s, Δ₀, b₀)
    return atoms, timeLaserOff, IntensityLaserOff
end
