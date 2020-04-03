function getAtoms_distribution(N::Int64, Radius::Real, rₘᵢₙ::Real, spatialDistribution::Symbol)
    r_new_atom, r_cartesian = get_empty_arrays(N)
    nValid = 0
    for iLoop = 1:10^6
		ex = Expr(:call, Symbol("get_one_random_atom_inside_", spatialDistribution, "_sphere!"), Radius, r_new_atom)
		eval(ex)


        if is_valid_position(r_new_atom, r_cartesian[1:nValid,:], rₘᵢₙ)
            nValid = nValid + 1
            r_cartesian[nValid,:] = r_new_atom
        end
        if nValid == N
            break
        end
    end
    if nValid < N
        error("Could not generate all data after 10M interactions")
    end
    return r_cartesian
end



function get_empty_arrays(N)
	r_new_atom = zeros(3)
	r_cartesian = Array{Float64}(undef, N, 3)
    return r_new_atom, r_cartesian
end

function get_one_random_atom_inside_homogenous_sphere!(Radius, r_new_atom)
    r_new_atom[1] = 2π*rand() #azimuth
    r_new_atom[2] = asin(2*rand() - 1) #elevation
    r_new_atom[3] = Radius*(rand()^(1 ./3.)) #radii
    r_new_atom[:] = sph2cart(r_new_atom)
    nothing
end

function get_one_random_atom_inside_gaussian_sphere!(Radius, r_new_atom)
    r_new_atom[1] = 2π*rand() #azimuth
    r_new_atom[2] = asin(2*rand() - 1) #elevation
    r_new_atom[3] = abs(Radius*(randn()))^(1 ./3.) #radii
    r_new_atom[:] = sph2cart(r_new_atom)
    nothing
end

"""
    spherical_coordinate=[azimuth, elevation, r]

    azimuth = projection on XY-plane (in radians)
    "elevation" or "Polar" = projection on Z-axis (in radians)
    r = radius

	ref: https://www.mathworks.com/help/matlab/ref/sph2cart.html
"""
function sph2cart(spherical_coordinate)
    azimuth = spherical_coordinate[1]
    elevation = spherical_coordinate[2]
    radius = spherical_coordinate[3]
    x = radius*cos(elevation)*cos(azimuth)
    y = radius*cos(elevation)*sin(azimuth)
    z = radius*sin(elevation)
    return [x,y,z]
end

function is_valid_position(r_new_atom, r_cartesian, rₘᵢₙ)
    nAtoms = size(r_cartesian,1)
    is_valid = Array{Bool}(undef, nAtoms)
	return all( get_Distance_A_to_b(r_cartesian,r_new_atom) .≥ rₘᵢₙ )
end

function get_Distance_A_to_b(A,b)
    n_rows = size(A,1)
    distanceAb = Array{Float64}(undef, n_rows)
    @inbounds for i=1:n_rows
		distanceAb[i] = Distances.evaluate(Euclidean(), A[i,:], b)
    end
    return distanceAb
end





function get_scalar_GreensMatrix(N, Γ, k₀, R_jk, Δ)
	G = Array{Complex{Float64}}(undef, N, N)
	@. G = -(Γ/2)*exp(1im*k₀*R_jk)/(1im*k₀*R_jk)
	G[LinearAlgebra.diagind(G)] .= 1im*Δ - Γ/2
	return G
end
