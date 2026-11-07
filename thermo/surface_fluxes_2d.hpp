#ifndef METRIPLECTIC_2D_THERMO_SURFACE_FLUXES_2D_HPP
#define METRIPLECTIC_2D_THERMO_SURFACE_FLUXES_2D_HPP

#include <cmath>
#include <cstddef>
#include "metriplectic_2d/thermo/pade_tables.hpp"

namespace metriplectic_2d::thermo {

// ============================================================================
// Shared Onsager Surface Flux Kernel (ADR-2D-008, SAT-2D / GROUND-2D)
// ============================================================================
// Single source of truth for the air-surface exchange fluxes used by BOTH the
// open SBP-SAT boundary operator (spatial/sat_boundaries_2d.hpp) and the
// per-column heated-ground reservoir (cases/solar_ground_forcing_2d.hpp).
//
// Conventions (+ = into the atmospheric column):
//   sensible_w_m2            : sensible heat flux Q_H = alpha_H * (1/T_a - 1/T_s)
//   latent_mass_flux_kg_m2_s : vapor mass flux E = M_evap * max(0, psi_l(T_s)/T_s - mu_v/T_a)
//   latent_energy_w_m2       : latent energy flux E * h_vap(T_s) (+ into air)
//   longwave_net_w_m2        : eps * sigma * (T_s^4 - T_a^4) (+ ground loses to sky)
//   entropy_prod_w_m2_k      : sigma_H + E * A_evap >= 0 (Onsager second law)
//
// The gravitational-potential term phi_grav(y=0) = g * 0 = 0 is identically zero
// at the surface, so the latent vapor enthalpy uses h_vap(T_s) = e_v(T_s) + R_v * T_s
// (the same expression the SAT operator evaluates at the bottom wall).

struct SurfaceFluxResult2D {
    double sensible_w_m2            = 0.0; // + into air
    double latent_mass_flux_kg_m2_s = 0.0; // + vapor into air
    double latent_energy_w_m2       = 0.0; // + into air
    double longwave_net_w_m2        = 0.0; // + ground loses to sky
    double entropy_prod_w_m2_k      = 0.0; // >= 0
};

static_assert(sizeof(SurfaceFluxResult2D) == 5 * sizeof(double),
              "SurfaceFluxResult2D must be a plain trivially-copyable POD");

// [UNIT_OP: GUI_BUILD_2D] [RULE_4: 38/60] [CONSERVATION: sigma=0, w_ext=EXTERNAL_QSW]
// RATIONALE: Deterministic branchless Onsager surface exchange shared by SAT and ground ODE
// PRE/POST : Pre: temp_air_k > 0, temp_surface_k > 0; Post: entropy_prod_w_m2_k >= 0
[[nodiscard]] inline SurfaceFluxResult2D compute_surface_fluxes_2d(
    double temp_air_k,
    double mu_v_air_j_kg,
    double temp_surface_k,
    double sensible_conductance_w_k_m2,
    double evaporation_mobility,
    double longwave_emissivity,
    bool enable_longwave
) noexcept {
    SurfaceFluxResult2D out{};

    // Fail closed: non-positive absolute temperatures are physically invalid.
    if (!(temp_air_k > 0.0) || !(temp_surface_k > 0.0)) {
        return out;
    }

    // Sensible heat: thermal Onsager affinity (1/T_a - 1/T_s), conductance alpha_H.
    const double thermal_affinity = (1.0 / temp_air_k) - (1.0 / temp_surface_k);
    const double q_sensible_w_m2  = sensible_conductance_w_k_m2 * thermal_affinity;
    out.sensible_w_m2       = q_sensible_w_m2;
    out.entropy_prod_w_m2_k = q_sensible_w_m2 * thermal_affinity; // sigma_H >= 0

    // One-sided Planck-potential evaporation affinity A_evap = psi_l(T_s)/T_s - mu_v/T_a.
    const double mu_v_sat_s   = PadeThermodynamicTables::psi_liquid_pade(temp_surface_k);
    const double evap_affinity = (mu_v_air_j_kg / temp_air_k) < (mu_v_sat_s / temp_surface_k)
        ? ((mu_v_sat_s / temp_surface_k) - (mu_v_air_j_kg / temp_air_k))
        : 0.0;
    out.latent_mass_flux_kg_m2_s = evaporation_mobility * evap_affinity; // >= 0

    const double h_vap_s = PadeThermodynamicTables::internal_energy_vapor(temp_surface_k)
        + ThermoConstants::kGasConstVapor * temp_surface_k;
    out.latent_energy_w_m2    = out.latent_mass_flux_kg_m2_s * h_vap_s;
    out.entropy_prod_w_m2_k  += out.latent_mass_flux_kg_m2_s * evap_affinity; // + E * A_evap >= 0

    // Net longwave radiative exchange (ground emits, atmosphere back-radiates).
    if (enable_longwave) {
        const double t4_air  = temp_air_k  * temp_air_k  * temp_air_k  * temp_air_k;
        const double t4_surf = temp_surface_k * temp_surface_k * temp_surface_k * temp_surface_k;
        out.longwave_net_w_m2 = longwave_emissivity * ThermoConstants::kStefanBoltzmann
            * (t4_surf - t4_air);
    }

    return out;
}

} // namespace metriplectic_2d::thermo

#endif // METRIPLECTIC_2D_THERMO_SURFACE_FLUXES_2D_HPP
