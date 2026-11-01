#ifndef METRIPLECTIC_2D_SPATIAL_SAT_BOUNDARIES_2D_HPP
#define METRIPLECTIC_2D_SPATIAL_SAT_BOUNDARIES_2D_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include "metriplectic_2d/core/types.hpp"
#include "metriplectic_2d/spatial/sbp_operators_2d.hpp"
#include "metriplectic_2d/thermo/moist_eos.hpp"
#include "metriplectic_2d/thermo/pade_tables.hpp"

namespace metriplectic_2d::spatial {

// ============================================================================
// Thermodynamically Open SBP-SAT Boundary Penalties & Flux Ledger (ADR-2D-008)
// ============================================================================
// Tracks exact cumulative boundary mass and energy transfers across the surface
// (y = 0) and top boundary (y = L_y) so the ClosureAuditor2D can verify open-system
// mass and energy conservation to machine precision (< 1e-14).

struct BoundaryFluxLedger2D {
    double net_dry_air_mass_flux_kg_s    = 0.0; // Instantaneous rate dM_d/dt at boundaries
    double net_water_mass_flux_kg_s      = 0.0; // Instantaneous rate dM_w/dt at boundaries
    double net_total_energy_flux_w       = 0.0; // Instantaneous rate dE_tot/dt at boundaries
    double boundary_entropy_prod_w_k     = 0.0; // Surface Onsager entropy production >= 0

    double cumulative_dry_air_mass_kg    = 0.0; // Time-integrated boundary dry mass transfer
    double cumulative_water_mass_kg      = 0.0; // Time-integrated boundary water mass transfer
    double cumulative_total_energy_j     = 0.0; // Time-integrated boundary energy transfer

    constexpr void reset_instantaneous() noexcept {
        net_dry_air_mass_flux_kg_s = 0.0;
        net_water_mass_flux_kg_s   = 0.0;
        net_total_energy_flux_w    = 0.0;
        boundary_entropy_prod_w_k  = 0.0;
    }

    constexpr void accumulate_step(double dt_s) noexcept {
        cumulative_dry_air_mass_kg += dt_s * net_dry_air_mass_flux_kg_s;
        cumulative_water_mass_kg   += dt_s * net_water_mass_flux_kg_s;
        cumulative_total_energy_j  += dt_s * net_total_energy_flux_w;
    }
};

struct SurfaceSATConfig2D {
    bool enable_precip_fallout        = false; // Upwind rain fallout at y = 0
    bool enable_surface_heat_exchange = false; // Onsager sensible & latent heat exchange at y = 0
    double surface_temp_k             = 290.0; // Prescribed surface skin temperature T_s (K)
    double sensible_conductance       = 0.0;   // Thermal Onsager conductance alpha_H (W * K / m^2)
    double evaporation_mobility       = 0.0;   // Planck-potential evaporation mobility M_evap
};

struct SATBoundaryOperators2D {
    // Applies open SBP-SAT penalties P_{y,0}^{-1} tau_SAT at the bottom surface (iy == 0)
    // and records exact boundary mass/energy effluxes in BoundaryFluxLedger2D.
    template <bool EnableKorteweg = false>
    static void apply_surface_sat_penalties(
        const core::NodeStateArray* state_grid,
        core::NodeStateArray* rhs_grid,
        const GridGeometry2D& geom,
        const thermo::MoistPadeEOS2D<EnableKorteweg>& eos,
        const SurfaceSATConfig2D& config,
        BoundaryFluxLedger2D& ledger
    ) noexcept {
        using core::idx;
        using core::StateVar;
        using thermo::PadeThermodynamicTables;
        using thermo::ThermoConstants;

        if (!config.enable_precip_fallout && !config.enable_surface_heat_exchange) {
            return;
        }

        const double py0 = geom.norm_weight_y(0); // 0.5 * dy
        const double inv_py0 = 1.0 / py0;
        const double phi_grav_0 = ThermoConstants::kGravity * geom.y_coord(0);

        for (size_t ix = 0; ix < geom.nx; ++ix) {
            const size_t node_id = geom.linear_index(ix, 0);
            const core::NodeStateArray& x0 = state_grid[node_id];
            core::NodeStateArray& rhs0     = rhs_grid[node_id];

            core::NodeEntropyArray w0{};
            const auto diag0 = eos.evaluate_cell_diagnostics(x0, w0, phi_grav_0);
            if (diag0.validity == core::EOSValidityStatus::INVALID) {
                continue;
            }

            const double temp_0 = diag0.temperature_k;
            const double rho_r0 = x0[idx(StateVar::RHO_R)];

            // 1. Upwind Precipitation Fallout Drain at y = 0 (ADR-2D-008)
            if (config.enable_precip_fallout && rho_r0 > 0.0) {
                const double w_term = PadeThermodynamicTables::rain_terminal_speed_m_s(rho_r0);
                const double mass_drain_flux = -w_term * rho_r0; // <= 0 (kg / (m^2 s))

                const double rho_tot0 = x0[idx(StateVar::RHO_D)] + x0[idx(StateVar::RHO_V)]
                    + x0[idx(StateVar::RHO_C)] + rho_r0;
                const double ux0 = x0[idx(StateVar::MOM_UX)] / rho_tot0;
                const double uy0 = x0[idx(StateVar::MOM_UY)] / rho_tot0;
                const double half_v2 = 0.5 * (ux0 * ux0 + uy0 * uy0);
                const double e_liq0  = PadeThermodynamicTables::internal_energy_liquid(temp_0);

                // Rain drops exit carrying their specific liquid internal + kinetic + potential energy
                const double spec_energy_r = e_liq0 + half_v2 + phi_grav_0;
                const double energy_drain_flux = mass_drain_flux * spec_energy_r;

                rhs0[idx(StateVar::RHO_R)]   += inv_py0 * mass_drain_flux;
                rhs0[idx(StateVar::MOM_UX)]  += inv_py0 * (mass_drain_flux * ux0);
                rhs0[idx(StateVar::MOM_UY)]  += inv_py0 * (mass_drain_flux * uy0);
                rhs0[idx(StateVar::EPS_TOT)] += inv_py0 * energy_drain_flux;

                ledger.net_water_mass_flux_kg_s += geom.dx * mass_drain_flux;
                ledger.net_total_energy_flux_w  += geom.dx * energy_drain_flux;
            }

            // 2. Onsager Surface Sensible & Planck-Potential Latent Heat Exchange (OQ-002)
            if (config.enable_surface_heat_exchange) {
                const double temp_s = config.surface_temp_k;
                const double thermal_affinity = (1.0 / temp_0) - (1.0 / temp_s);

                // Sensible heat flux Q_H = alpha_H * (1/T_0 - 1/T_s)
                // Entropy production sigma_H = Q_H * (1/T_0 - 1/T_s) >= 0 unconditionally
                const double q_sensible_w_m2 = config.sensible_conductance * thermal_affinity;
                const double sigma_thermal   = q_sensible_w_m2 * thermal_affinity;

                // One-sided Planck-potential surface evaporation:
                // Affinity A_evap = (mu_{v,sat}(T_s)/T_s - mu_{v,0}/T_0)
                // Wait: in entropy variables, w_{rho_v, 0} = (half_v2 - phi_0 - mu_{v,0}) / T_0.
                // At the surface, chemical potential affinity driving vapor influx is
                // d(-mu_v / T) = (-mu_{v,0}/T_0) - (-mu_{v,sat}(T_s)/T_s) = mu_{v,sat}(T_s)/T_s - mu_{v,0}/T_0.
                // Since mu_{v,sat}(T_s) = psi_l(T_s), this is strictly >= 0 when surface is warmer/moister.
                const double mu_v_sat_s = PadeThermodynamicTables::psi_liquid_pade(temp_s);
                const double evap_affinity = (diag0.mu_v / temp_0) < (mu_v_sat_s / temp_s)
                    ? ((mu_v_sat_s / temp_s) - (diag0.mu_v / temp_0))
                    : 0.0;
                const double evap_mass_flux = config.evaporation_mobility * evap_affinity; // >= 0
                const double h_vap_s = PadeThermodynamicTables::internal_energy_vapor(temp_s)
                    + ThermoConstants::kGasConstVapor * temp_s + phi_grav_0;
                const double evap_energy_flux = evap_mass_flux * h_vap_s;

                rhs0[idx(StateVar::RHO_V)]   += inv_py0 * evap_mass_flux;
                rhs0[idx(StateVar::EPS_TOT)] += inv_py0 * (q_sensible_w_m2 + evap_energy_flux);

                ledger.net_water_mass_flux_kg_s += geom.dx * evap_mass_flux;
                ledger.net_total_energy_flux_w  += geom.dx * (q_sensible_w_m2 + evap_energy_flux);
                ledger.boundary_entropy_prod_w_k += geom.dx * (
                    sigma_thermal + evap_mass_flux * evap_affinity
                );
            }
        }
    }
};

} // namespace metriplectic_2d::spatial

#endif // METRIPLECTIC_2D_SPATIAL_SAT_BOUNDARIES_2D_HPP
