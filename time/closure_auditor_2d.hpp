#ifndef METRIPLECTIC_2D_TIME_CLOSURE_AUDITOR_2D_HPP
#define METRIPLECTIC_2D_TIME_CLOSURE_AUDITOR_2D_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include "metriplectic_2d/core/types.hpp"
#include "metriplectic_2d/spatial/sat_boundaries_2d.hpp"
#include "metriplectic_2d/spatial/sbp_operators_2d.hpp"
#include "metriplectic_2d/thermo/moist_eos.hpp"
#include "metriplectic_2d/thermo/pade_tables.hpp"

namespace metriplectic_2d::time {

// ============================================================================
// 2D Conservation & Second-Law Closure Auditor (ADR-2D-008, ADR-2D-010)
// ============================================================================

struct ConservationIntegrals2D {
    double dry_air_mass_kg     = 0.0;
    double total_water_mass_kg = 0.0;
    double cloud_water_mass_kg = 0.0;
    double rain_water_mass_kg  = 0.0;
    double momentum_x_kg_m_s   = 0.0;
    double momentum_y_kg_m_s   = 0.0;
    double total_energy_j      = 0.0;
    double total_entropy_j_k   = 0.0;
};

struct AuditReport2D {
    core::IntegrationStatus      integration_status = core::IntegrationStatus::NOT_RUN;
    core::ClosureStatus          closure_status     = core::ClosureStatus::NOT_CHECKED;
    core::PhysicalValidityStatus physical_validity  = core::PhysicalValidityStatus::UNKNOWN;
    core::EOSValidityStatus      eos_validity       = core::EOSValidityStatus::UNKNOWN;

    double dry_mass_rel_residual     = 0.0;
    double water_mass_rel_residual   = 0.0;
    double total_energy_rel_residual = 0.0;
    double min_entropy_prod_rate_w_k = 0.0;

    ConservationIntegrals2D initial_integrals{};
    ConservationIntegrals2D final_integrals{};
};

class ClosureAuditor2D {
public:
    // Computes exact SBP quadrature integrals over the 2D (x, y) domain
    template <bool EnableKorteweg = false>
    [[nodiscard]] static ConservationIntegrals2D compute_domain_integrals(
        const core::NodeStateArray* state_grid,
        const spatial::GridGeometry2D& geom,
        const thermo::MoistPadeEOS2D<EnableKorteweg>& eos,
        core::PhysicalValidityStatus& phys_status_out,
        core::EOSValidityStatus& eos_status_out
    ) noexcept {
        using core::idx;
        using core::StateVar;
        using thermo::PadeThermodynamicTables;
        using thermo::ThermoConstants;

        ConservationIntegrals2D out{};
        phys_status_out = core::PhysicalValidityStatus::VALID;
        eos_status_out  = core::EOSValidityStatus::VALID;

        for (size_t iy = 0; iy < geom.ny; ++iy) {
            const double vol = geom.cell_volume_weight(0, iy);
            const double phi_grav = ThermoConstants::kGravity * geom.y_coord(iy);

            for (size_t ix = 0; ix < geom.nx; ++ix) {
                const size_t id = geom.linear_index(ix, iy);
                const core::NodeStateArray& x = state_grid[id];

                const double rho_d = x[idx(StateVar::RHO_D)];
                const double rho_v = x[idx(StateVar::RHO_V)];
                const double rho_c = x[idx(StateVar::RHO_C)];
                const double rho_r = x[idx(StateVar::RHO_R)];
                const double mux   = x[idx(StateVar::MOM_UX)];
                const double muy   = x[idx(StateVar::MOM_UY)];
                const double esgs  = x[idx(StateVar::RHO_E_SGS)];
                const double eps   = x[idx(StateVar::EPS_TOT)];

                if (!std::isfinite(rho_d) || !std::isfinite(rho_v) ||
                    !std::isfinite(rho_c) || !std::isfinite(rho_r) ||
                    !std::isfinite(mux) || !std::isfinite(muy) ||
                    !std::isfinite(esgs) || !std::isfinite(eps)) {
                    phys_status_out = core::PhysicalValidityStatus::INVALID;
                    eos_status_out  = core::EOSValidityStatus::INVALID;
                    return out;
                }

                if (rho_d <= 0.0 || rho_v <= 0.0 || rho_c < 0.0 || rho_r < 0.0 || esgs <= 0.0) {
                    phys_status_out = core::PhysicalValidityStatus::POSITIVITY_VIOLATED;
                }

                core::NodeEntropyArray w{};
                const auto diag = eos.evaluate_cell_diagnostics(x, w, phi_grav);
                if (diag.validity == core::EOSValidityStatus::INVALID) {
                    eos_status_out = core::EOSValidityStatus::INVALID;
                    if (phys_status_out == core::PhysicalValidityStatus::VALID) {
                        phys_status_out = core::PhysicalValidityStatus::INVALID;
                    }
                    return out;
                }
                if (diag.validity == core::EOSValidityStatus::EXTRAPOLATED &&
                    eos_status_out == core::EOSValidityStatus::VALID) {
                    eos_status_out = core::EOSValidityStatus::EXTRAPOLATED;
                }

                const double temp_k = diag.temperature_k;
                // Specific entropy density rho*s = sum_k rho_k s_k(rho_k, T)
                // Using Gibbs-Duhem: rho*s = (rho*e_int + p_gas - sum_k rho_k mu_k) / T
                const double rho_tot = rho_d + rho_v + rho_c + rho_r;
                const double half_v2 = 0.5 * (mux * mux + muy * muy) / (rho_tot * rho_tot);
                const double eps_int = eps - rho_tot * half_v2 - esgs - rho_tot * phi_grav;
                const double mu_d = PadeThermodynamicTables::psi_dry_pade(temp_k)
                    + ThermoConstants::kGasConstDry * temp_k
                    * std::log(rho_d / ThermoConstants::kRhoDryRef);
                const double rho_s = (
                    eps_int + diag.pressure_pa
                    - rho_d * mu_d
                    - rho_v * diag.mu_v
                    - (rho_c + rho_r) * diag.mu_r
                ) / temp_k;

                out.dry_air_mass_kg     += vol * rho_d;
                out.total_water_mass_kg += vol * (rho_v + rho_c + rho_r);
                out.cloud_water_mass_kg += vol * rho_c;
                out.rain_water_mass_kg  += vol * rho_r;
                out.momentum_x_kg_m_s   += vol * mux;
                out.momentum_y_kg_m_s   += vol * muy;
                out.total_energy_j      += vol * eps;
                out.total_entropy_j_k   += vol * rho_s;
            }
        }
        return out;
    }

    [[nodiscard]] static AuditReport2D evaluate_closure(
        const ConservationIntegrals2D& initial,
        const ConservationIntegrals2D& final_state,
        const spatial::BoundaryFluxLedger2D& boundary_ledger,
        double min_observed_entropy_prod_rate,
        core::PhysicalValidityStatus phys_status,
        core::EOSValidityStatus eos_status,
        double rel_tolerance = 1.0e-12
    ) noexcept {
        AuditReport2D rep{};
        rep.initial_integrals         = initial;
        rep.final_integrals           = final_state;
        rep.min_entropy_prod_rate_w_k = min_observed_entropy_prod_rate;
        rep.physical_validity         = phys_status;
        rep.eos_validity              = eos_status;

        if (phys_status != core::PhysicalValidityStatus::VALID ||
            eos_status == core::EOSValidityStatus::INVALID) {
            rep.integration_status = core::IntegrationStatus::DIVERGED;
            rep.closure_status     = core::ClosureStatus::FAILED;
            return rep;
        }

        rep.integration_status = core::IntegrationStatus::COMPLETED;

        const double expected_dry   = initial.dry_air_mass_kg + boundary_ledger.cumulative_dry_air_mass_kg;
        const double expected_water = initial.total_water_mass_kg + boundary_ledger.cumulative_water_mass_kg;
        const double expected_energy = initial.total_energy_j + boundary_ledger.cumulative_total_energy_j;

        rep.dry_mass_rel_residual = std::abs(final_state.dry_air_mass_kg - expected_dry)
            / (std::abs(initial.dry_air_mass_kg) + 1.0);
        rep.water_mass_rel_residual = std::abs(final_state.total_water_mass_kg - expected_water)
            / (std::abs(initial.total_water_mass_kg) + 1.0);
        rep.total_energy_rel_residual = std::abs(final_state.total_energy_j - expected_energy)
            / (std::abs(initial.total_energy_j) + 1.0);

        const bool mass_ok   = (rep.dry_mass_rel_residual < rel_tolerance) &&
                               (rep.water_mass_rel_residual < rel_tolerance);
        const bool energy_ok = (rep.total_energy_rel_residual < rel_tolerance);
        const bool s_prod_ok = (min_observed_entropy_prod_rate >= -1.0e-14);

        rep.closure_status = (mass_ok && energy_ok && s_prod_ok)
            ? core::ClosureStatus::PASSED
            : core::ClosureStatus::FAILED;

        return rep;
    }
};

} // namespace metriplectic_2d::time

#endif // METRIPLECTIC_2D_TIME_CLOSURE_AUDITOR_2D_HPP
