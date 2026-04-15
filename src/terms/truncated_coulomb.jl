# Truncated Coulomb kernels for non-periodic electrostatics, following
#
#     C. A. Rozzi, D. Varsano, A. Marini, E. K. U. Gross, A. Rubio,
#     "Exact Coulomb cutoff technique for supercell calculations",
#     Phys. Rev. B 73, 205119 (2006).  https://arxiv.org/abs/cond-mat/0601031
#
# `truncated_coulomb_fourier(G_cart, ...)` returns the Fourier-space Coulomb kernel v_c(G)
# for a system whose electrostatics has been truncated in the non-periodic directions.
# It replaces the standard 4π/|G|² kernel in the Hartree and ionic-local terms.
#
# The return value by number of electrostatically-periodic directions:
#
#   3 — fully periodic: v_c(G) = 4π/|G|² (G≠0), 0 at G=0 (compensating background).
#       This is identically the standard periodic kernel; the code degenerates to the
#       usual formulas with no extra branching at call sites.
#
#   2 — 2D slab: one isolated direction orthogonal to the periodic plane.
#       v_c(G) = 4π/|G|² · (1 − exp(−|G_∥| R) cos(G_z R))
#       where G_z is the out-of-plane component, G_∥ = √(|G|²−G_z²), and R is half the
#       length of the isolated lattice vector (Rozzi et al., eq. 16).
#       G=0 set to 0 (compensating background in the periodic plane).
#
#   1 — 1D wire: NOT YET IMPLEMENTED.
#
#   0 — 0D fully isolated: spherical truncation at radius R = min(isolated edge) / 2.
#       v_c(G) = 4π/|G|² · (1 − cos(|G| R))  for G≠0,
#       v_c(0) = 2π R²  (finite limit, no compensating background).
#
# `truncated_coulomb_fourier(G_cart, n_per, R, aiso_unit)` takes only isbits parameters
# so that it is safe to use inside GPU `map` closures.  A convenience wrapper
# `truncated_coulomb_fourier(G_cart, model)` is provided for one-off calls.

"""
Truncation radius used by the Rozzi truncated Coulomb method for `model`.  Chosen as half
the minimum edge length among the electrostatically non-periodic lattice directions.
Returns `zero(T)` for fully periodic models.
"""
function truncated_coulomb_radius(model::Model{T}) where {T}
    n_periodic_electrostatics(model) == 3 && return zero(T)
    lengths = T[norm(model.lattice[:, i])
                for i = 1:3 if !is_electrostatics_periodic(model, i)]
    minimum(lengths) / 2
end

"""
Precompute the Rozzi truncated Coulomb parameters for `model`: the number of
electrostatically periodic directions `n_per`, the truncation radius `R`, and
the unit vector `aiso_unit` along the isolated direction for the 2D-slab case
(zero vector otherwise).  All fields are isbits so the result can be passed
directly into GPU-friendly kernels.
"""
function truncated_coulomb_params(model::Model{T}) where {T}
    n_per = n_periodic_electrostatics(model)
    R = T(truncated_coulomb_radius(model))
    aiso_unit = if n_per == 2
        iiso = findfirst(i -> !is_electrostatics_periodic(model, i), 1:3)::Int
        a = model.lattice[:, iiso]
        Vec3{T}(a / norm(a))
    else
        zero(Vec3{T})
    end
    (; n_per, R, aiso_unit)
end

"""
Fourier-space value of the truncated Coulomb kernel at Cartesian vector `G_cart`.
Parameters `(n_per, R, aiso_unit)` are obtained from [`truncated_coulomb_params`](@ref).
Only isbits arguments; safe to call inside GPU kernels.
"""
function truncated_coulomb_fourier(G_cart, n_per::Int, R::T, aiso_unit::Vec3{T}) where {T}
    Gsq = sum(abs2, G_cart)
    if n_per == 3
        # Fully periodic: standard Coulomb with compensating background at G=0.
        return iszero(Gsq) ? zero(T) : 4T(π) / Gsq
    elseif n_per == 0
        # Fully isolated 0D: spherical truncation at R.
        iszero(Gsq) && return 2T(π) * R^2
        Gnorm = sqrt(Gsq)
        return 4T(π) * (1 - cos(Gnorm * R)) / Gsq
    elseif n_per == 2
        # 2D slab: G_z along the isolated direction, G_∥ in the periodic plane.
        iszero(Gsq) && return zero(T)   # compensating background in the periodic plane
        Gz   = dot(G_cart, aiso_unit)
        Gpar = sqrt(max(Gsq - Gz^2, zero(T)))
        return 4T(π) * (1 - exp(-Gpar * R) * cos(Gz * R)) / Gsq
    else  # n_per == 1
        error("Truncated Coulomb for 1D-periodic wire geometries is not yet implemented.")
    end
end

"""
Convenience wrapper that extracts truncation parameters from `model` and calls
[`truncated_coulomb_fourier`](@ref).  Intended for setup code and tests; in tight
loops pre-compute with [`truncated_coulomb_params`](@ref) and pass the tuple directly.
"""
function truncated_coulomb_fourier(G_cart, model::Model)
    (; n_per, R, aiso_unit) = truncated_coulomb_params(model)
    truncated_coulomb_fourier(G_cart, n_per, R, aiso_unit)
end
