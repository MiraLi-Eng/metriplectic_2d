#ifndef METRIPLECTIC_2D_THERMO_MOIST_EOS_HPP
#define METRIPLECTIC_2D_THERMO_MOIST_EOS_HPP

#include <cmath>
#include <cstddef>
#include "metriplectic_2d/core/types.hpp"
#include "metriplectic_2d/thermo/eos_concept.hpp"
#include "metriplectic_2d/thermo/hysteretic_mobility.hpp"
#include "metriplectic_2d/thermo/pade_tables.hpp"

namespace metriplectic_2d::thermo {

// ============================================================================
// Concrete 2D Moist Multiphase Padé EOS Manifold (MoistPadeEOS2D<EnableKorteweg>)
// ============================================================================
// Operates on the 8-scalar 2D state vector (x = horizontal, y = vertical column)
// with zero virtual dispatch, zero dynamic memory allocation, and optional
// compile-time Korteweg capillary gradient coupling (ADR-2D-002, ADR-2D-003).

template <bool EnableKorteweg = false>
class MoistPadeEOS2D {
public:
    static constexpr double kMaxHalleyRelativeResidual = 1.0e-12;

    constexpr MoistPadeEOS2D() noexcept = default;
    constexpr explicit MoistPadeEOS2D(KohlerHystereticParams2D mobility_params) noexcept
        : phase_mobility_(mobility_params) {}

    // ------------------------------------------------------------------------
    // 1. Forward Helper: 8-Scalar Conservative State x from (rho_k, ux, uy, e_sgs, T)
    // ------------------------------------------------------------------------
    [[nodiscard]] static constexpr core::NodeStateArray make_conservative_state(
        double rho_d,
        double rho_v,
        double rho_c,
        double rho_r,
        double ux_vel,
        double uy_vel,
        double rho_e_sgs,
        double temp_k,
        double geopotential_j_kg = 0.0
    ) noexcept {
        using core::idx;
        using core::StateVar;

        const double rho_tot = rho_d + rho_v + rho_c + rho_r;
        const double eps_int = compute_internal_energy_density(rho_d, rho_v, rho_c, rho_r, temp_k);
        const double eps_kin = 0.5 * rho_tot * (ux_vel * ux_vel + uy_vel * uy_vel);
        const double eps_tot = eps_int + eps_kin + rho_e_sgs + rho_tot * geopotential_j_kg;

        core::NodeStateArray x{};
        x[idx(StateVar::RHO_D)]     = rho_d;
        x[idx(StateVar::RHO_V)]     = rho_v;
        x[idx(StateVar::RHO_C)]     = rho_c;
        x[idx(StateVar::RHO_R)]     = rho_r;
        x[idx(StateVar::MOM_UX)]    = rho_tot * ux_vel;
        x[idx(StateVar::MOM_UY)]    = rho_tot * uy_vel;
        x[idx(StateVar::RHO_E_SGS)] = rho_e_sgs;
        x[idx(StateVar::EPS_TOT)]   = eps_tot;
        return x;
    }

    [[nodiscard]] static constexpr double compute_internal_energy_density(
        double rho_d,
        double rho_v,
        double rho_c,
        double rho_r,
        double temp_k
    ) noexcept {
        const double rho_l = rho_c + rho_r;
        return rho_d * PadeThermodynamicTables::internal_energy_dry(temp_k)
            + rho_v * PadeThermodynamicTables::internal_energy_vapor(temp_k)
            + rho_l * PadeThermodynamicTables::internal_energy_liquid(temp_k);
    }

    // ------------------------------------------------------------------------
    // 2. Cell-Center Diagnostic Inversion: x -> (T, p, mu_k, w) (THERMO-2D-001)
    // ------------------------------------------------------------------------
    [[nodiscard]] CellThermoDiagnostics2D evaluate_cell_diagnostics(
        const core::NodeStateArray& x,
        core::NodeEntropyArray& w_out,
        double geopotential_j_kg = 0.0,
        double laplacian_rho_c = 0.0
    ) const noexcept {
        using core::idx;
        using core::StateVar;

        CellThermoDiagnostics2D diag{};
        diag.validity = core::EOSValidityStatus::INVALID;

        const double rho_d     = x[idx(StateVar::RHO_D)];
        const double rho_v     = x[idx(StateVar::RHO_V)];
        const double rho_c     = x[idx(StateVar::RHO_C)];
        const double rho_r     = x[idx(StateVar::RHO_R)];
        const double mom_ux    = x[idx(StateVar::MOM_UX)];
        const double mom_uy    = x[idx(StateVar::MOM_UY)];
        const double rho_e_sgs = x[idx(StateVar::RHO_E_SGS)];
        const double eps_tot   = x[idx(StateVar::EPS_TOT)];

        if (!std::isfinite(rho_d) || !std::isfinite(rho_v) ||
            !std::isfinite(rho_c) || !std::isfinite(rho_r) ||
            !std::isfinite(mom_ux) || !std::isfinite(mom_uy) ||
            !std::isfinite(rho_e_sgs) || !std::isfinite(eps_tot) ||
            rho_d <= 0.0 || rho_v <= 0.0 || rho_c < 0.0 || rho_r < 0.0 || rho_e_sgs < 0.0) {
            return diag;
        }

        const double rho_tot = rho_d + rho_v + rho_c + rho_r;
        const double inv_rho = 1.0 / rho_tot;
        const double ux_vel  = mom_ux * inv_rho;
        const double uy_vel  = mom_uy * inv_rho;
        const double half_v2 = 0.5 * (ux_vel * ux_vel + uy_vel * uy_vel);

        const double eps_int = eps_tot - rho_tot * half_v2 - rho_e_sgs - rho_tot * geopotential_j_kg;

        const double temp_k = solve_temperature_newton_halley(rho_d, rho_v, rho_c, rho_r, eps_int);
        if (!std::isfinite(temp_k) || temp_k <= 0.0) {
            return diag;
        }

        const double eps_recovered = compute_internal_energy_density(rho_d, rho_v, rho_c, rho_r, temp_k);
        const double rel_residual = std::abs(eps_recovered - eps_int) / (std::abs(eps_int) + 1.0);
        if (rel_residual > kMaxHalleyRelativeResidual) {
            diag.temperature_k = temp_k;
            diag.validity = core::EOSValidityStatus::INVALID;
            return diag;
        }

        diag.validity = PadeThermodynamicTables::classify_temperature_domain(temp_k);
        if (diag.validity == core::EOSValidityStatus::INVALID) {
            diag.temperature_k = temp_k;
            return diag;
        }

        const double inv_temp = 1.0 / temp_k;
        const double pressure = (rho_d * ThermoConstants::kGasConstDry
            + rho_v * ThermoConstants::kGasConstVapor) * temp_k;

        const double psi_d = PadeThermodynamicTables::psi_dry_pade(temp_k);
        const double psi_v = PadeThermodynamicTables::psi_vapor_pade(temp_k);
        const double psi_l = PadeThermodynamicTables::psi_liquid_pade(temp_k);

        const double mu_d = psi_d + ThermoConstants::kGasConstDry * temp_k
            * std::log(rho_d / ThermoConstants::kRhoDryRef);
        const double mu_v = psi_v + ThermoConstants::kGasConstVapor * temp_k
            * std::log(rho_v / ThermoConstants::kRhoVaporRef);

        double mu_c_eff = psi_l;
        if constexpr (EnableKorteweg) {
            // Variational Korteweg shift mu_c^eff = mu_c(T) - kappa * laplacian(rho_c)
            mu_c_eff -= phase_mobility_.params.kappa_korteweg_j_m5_kg2 * laplacian_rho_c;
        } else {
            (void)laplacian_rho_c;
        }
        const double mu_r = psi_l;

        const double kin_minus_phi = half_v2 - geopotential_j_kg;
        w_out[idx(StateVar::RHO_D)]     = (kin_minus_phi - mu_d) * inv_temp;
        w_out[idx(StateVar::RHO_V)]     = (kin_minus_phi - mu_v) * inv_temp;
        w_out[idx(StateVar::RHO_C)]     = (kin_minus_phi - mu_c_eff) * inv_temp;
        w_out[idx(StateVar::RHO_R)]     = (kin_minus_phi - mu_r) * inv_temp;
        w_out[idx(StateVar::MOM_UX)]    = -ux_vel * inv_temp;
        w_out[idx(StateVar::MOM_UY)]    = -uy_vel * inv_temp;
        w_out[idx(StateVar::RHO_E_SGS)] = -inv_temp;
        w_out[idx(StateVar::EPS_TOT)]   = inv_temp;

        diag.temperature_k    = temp_k;
        diag.pressure_pa      = pressure;
        diag.mu_v             = mu_v;
        diag.mu_c_eff         = mu_c_eff;
        diag.mu_r             = mu_r;
        diag.brunt_vaisala_n2 = 0.0;
        return diag;
    }

    // ------------------------------------------------------------------------
    // 3. 2D Zero-Root-Finding Quadrature-Node Gas Flux Evaluator (THERMO-2D-002)
    // ------------------------------------------------------------------------
    void evaluate_reversible_gas_flux(
        const core::NodeEntropyArray& w,
        double normal_x,
        double normal_y,
        core::NodeStateArray& flux_out
    ) const noexcept {
        using core::idx;
        using core::StateVar;

        const double w_eps  = w[idx(StateVar::EPS_TOT)];
        const double temp_k = 1.0 / w_eps;

        const double ux_vel = -w[idx(StateVar::MOM_UX)] * temp_k;
        const double uy_vel = -w[idx(StateVar::MOM_UY)] * temp_k;
        const double v_norm = ux_vel * normal_x + uy_vel * normal_y;
        const double half_v2 = 0.5 * (ux_vel * ux_vel + uy_vel * uy_vel);

        const double psi_d = PadeThermodynamicTables::psi_dry_pade(temp_k);
        const double psi_v = PadeThermodynamicTables::psi_vapor_pade(temp_k);

        const double arg_d = (half_v2 - psi_d - temp_k * w[idx(StateVar::RHO_D)])
            / (ThermoConstants::kGasConstDry * temp_k);
        const double arg_v = (half_v2 - psi_v - temp_k * w[idx(StateVar::RHO_V)])
            / (ThermoConstants::kGasConstVapor * temp_k);

        const double rho_d   = ThermoConstants::kRhoDryRef * std::exp(arg_d);
        const double rho_v   = ThermoConstants::kRhoVaporRef * std::exp(arg_v);
        const double rho_gas = rho_d + rho_v;
        const double p_gas   = (rho_d * ThermoConstants::kGasConstDry
            + rho_v * ThermoConstants::kGasConstVapor) * temp_k;

        const double dpsi_d_dt = PadeThermodynamicTables::dpsi_dt_dry_pade(temp_k);
        const double dpsi_v_dt = PadeThermodynamicTables::dpsi_dt_vapor_pade(temp_k);
        const double e_d_conj  = psi_d - temp_k * dpsi_d_dt;
        const double e_v_conj  = psi_v - temp_k * dpsi_v_dt;

        const double eps_int_gas = rho_d * e_d_conj + rho_v * e_v_conj;
        const double total_gas_energy_flux = (eps_int_gas + p_gas + rho_gas * half_v2) * v_norm;

        flux_out[idx(StateVar::RHO_D)]     = rho_d * v_norm;
        flux_out[idx(StateVar::RHO_V)]     = rho_v * v_norm;
        flux_out[idx(StateVar::RHO_C)]     = 0.0; // Advected via bar(rho_c)*v* (ADR-2D-005)
        flux_out[idx(StateVar::RHO_R)]     = 0.0; // Advected via bar(rho_r)*v* (ADR-2D-005)
        flux_out[idx(StateVar::MOM_UX)]    = rho_gas * v_norm * ux_vel + p_gas * normal_x;
        flux_out[idx(StateVar::MOM_UY)]    = rho_gas * v_norm * uy_vel + p_gas * normal_y;
        flux_out[idx(StateVar::RHO_E_SGS)] = 0.0;
        flux_out[idx(StateVar::EPS_TOT)]   = total_gas_energy_flux;
    }

    [[nodiscard]] double evaluate_tadmor_gas_potential(
        const core::NodeEntropyArray& w,
        double normal_x,
        double normal_y
    ) const noexcept {
        using core::idx;
        using core::StateVar;

        const double w_eps   = w[idx(StateVar::EPS_TOT)];
        const double temp_k  = 1.0 / w_eps;
        const double ux_vel  = -w[idx(StateVar::MOM_UX)] * temp_k;
        const double uy_vel  = -w[idx(StateVar::MOM_UY)] * temp_k;
        const double v_norm  = ux_vel * normal_x + uy_vel * normal_y;
        const double half_v2 = 0.5 * (ux_vel * ux_vel + uy_vel * uy_vel);

        const double psi_d = PadeThermodynamicTables::psi_dry_pade(temp_k);
        const double psi_v = PadeThermodynamicTables::psi_vapor_pade(temp_k);

        const double arg_d = (half_v2 - psi_d - temp_k * w[idx(StateVar::RHO_D)])
            / (ThermoConstants::kGasConstDry * temp_k);
        const double arg_v = (half_v2 - psi_v - temp_k * w[idx(StateVar::RHO_V)])
            / (ThermoConstants::kGasConstVapor * temp_k);

        const double rho_d = ThermoConstants::kRhoDryRef * std::exp(arg_d);
        const double rho_v = ThermoConstants::kRhoVaporRef * std::exp(arg_v);

        return -(rho_d * ThermoConstants::kGasConstDry + rho_v * ThermoConstants::kGasConstVapor) * v_norm;
    }

    // ------------------------------------------------------------------------
    // 4. Phase Mobility, Korteweg Force & 8-Scalar Jacobian (THERMO-2D-003/004)
    // ------------------------------------------------------------------------
    [[nodiscard]] double evaluate_phase_flux(
        const core::NodeStateArray& x,
        double delta_mu_eff,
        double temp_k
    ) const noexcept {
        const double rho_c = x[core::idx(core::StateVar::RHO_C)];
        return phase_mobility_.evaluate_flux(rho_c, delta_mu_eff, temp_k);
    }

    [[nodiscard]] core::NodeStateArray evaluate_phase_flux_jacobian(
        const core::NodeStateArray& x,
        double geopotential_j_kg = 0.0,
        double laplacian_rho_c = 0.0
    ) const noexcept {
        return phase_mobility_.evaluate_jacobian(x, geopotential_j_kg, laplacian_rho_c);
    }

    // Reversible Korteweg capillary force density F_korteweg = -rho_c * grad(kappa * lap(rho_c))
    // Zero-cost compile-time no-op when EnableKorteweg == false (ADR-2D-003)
    [[nodiscard]] constexpr std::array<double, 2> evaluate_korteweg_momentum_force(
        double rho_c,
        double d_lap_rho_c_dx,
        double d_lap_rho_c_dy
    ) const noexcept {
        if constexpr (EnableKorteweg) {
            const double coeff = -rho_c * phase_mobility_.params.kappa_korteweg_j_m5_kg2;
            return {coeff * d_lap_rho_c_dx, coeff * d_lap_rho_c_dy};
        } else {
            (void)rho_c;
            (void)d_lap_rho_c_dx;
            (void)d_lap_rho_c_dy;
            return {0.0, 0.0};
        }
    }

    [[nodiscard]] const HystereticPhaseMobility2D<EnableKorteweg>& phase_mobility() const noexcept {
        return phase_mobility_;
    }

private:
    HystereticPhaseMobility2D<EnableKorteweg> phase_mobility_{};
};

using MoistBulkPadeEOS2D     = MoistPadeEOS2D<false>;
using MoistKortewegPadeEOS2D = MoistPadeEOS2D<true>;

static_assert(MoistEOSManifold2D<MoistBulkPadeEOS2D>);
static_assert(MoistEOSManifold2D<MoistKortewegPadeEOS2D>);

} // namespace metriplectic_2d::thermo

#endif // METRIPLECTIC_2D_THERMO_MOIST_EOS_HPP
