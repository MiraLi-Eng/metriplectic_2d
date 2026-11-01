#ifndef METRIPLECTIC_2D_THERMO_HYSTERETIC_MOBILITY_HPP
#define METRIPLECTIC_2D_THERMO_HYSTERETIC_MOBILITY_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include "metriplectic_2d/core/types.hpp"
#include "metriplectic_2d/thermo/pade_tables.hpp"

namespace metriplectic_2d::thermo {

// ============================================================================
// 1. Model Parameters for C^inf Köhler Hysteresis & Optional Korteweg Policy
// ============================================================================
// Supports both mesoscale bulk phase hysteresis (kappa_korteweg == 0) and
// microscale Korteweg capillary gradient penalty (kappa_korteweg > 0) per
// ADR-2D-003.

struct KohlerHystereticParams2D {
    double gamma_active_kg2_k_j_m3_s = 2.5e-5; // Active droplet surface growth/evap coefficient
    double gamma_nucl_kg2_k_j_m3_s   = 5.0e-5; // Homogeneous/CCN Köhler activation coefficient
    double rho_c_ref_kg_m3           = 1.0e-4; // Droplet surface saturation density scale rho_{c,0}
    double delta_mu_crit_j_kg        = 3000.0; // Köhler nucleation barrier (~2.3% supersat at 285 K)
    double delta_mu_width_j_kg       = 120.0;  // Smooth C^inf transition width (dmu_crit/width = 25)
    double kappa_korteweg_j_m5_kg2   = 0.0;    // Korteweg capillary coefficient kappa (J*m^5/kg^2)
};

// ============================================================================
// 2. Branchless 4-Iteration Newton-Halley Cell-Center Temperature Kernel
// ============================================================================

inline constexpr size_t kMaxNewtonHalleyIters = 4;

namespace detail {

[[nodiscard]] constexpr double halley_residual_sub(
    double dt,
    double poly,
    double delta_eps_int
) noexcept {
    return dt * poly - delta_eps_int;
}

template <typename T>
    requires requires(const T& v) { v.real(); v.imag(); }
[[nodiscard]] inline T halley_residual_sub(
    const T& dt,
    const T& poly,
    const T& delta_eps_int
) noexcept {
    const double re = dt.real() * poly.real() - dt.imag() * poly.imag() - delta_eps_int.real();
    const double im = std::fma(
        dt.real(),
        poly.imag(),
        std::fma(dt.imag(), poly.real(), -delta_eps_int.imag())
    );
    return T(re, im);
}

[[nodiscard]] inline double stable_sech_sq(double x) noexcept {
    const double e = std::exp(-2.0 * std::abs(x));
    const double one_plus_e = 1.0 + e;
    return (4.0 * e) / (one_plus_e * one_plus_e);
}

[[nodiscard]] inline double stable_tanh(double x) noexcept {
    return std::tanh(x);
}

template <typename T>
    requires requires(const T& v) { v.real(); v.imag(); }
[[nodiscard]] inline T stable_tanh(const T& z) noexcept {
    const double tx = std::tanh(z.real());
    const double ty = std::tan(z.imag());
    const double sech2_x = stable_sech_sq(z.real());
    const double denom = 1.0 + tx * tx * ty * ty;
    return T(
        (tx * (1.0 + ty * ty)) / denom,
        (ty * sech2_x) / denom
    );
}

[[nodiscard]] inline double stable_log_ratio(double num, double den) noexcept {
    return std::log(num / den);
}

template <typename T>
    requires requires(const T& v) { v.real(); v.imag(); }
[[nodiscard]] inline T stable_log_ratio(const T& num, double den) noexcept {
    const double x = num.real();
    const double y = num.imag();
    const double r = y / x;
    return T(std::log(x / den) + 0.5 * std::log1p(r * r), std::atan2(y, x));
}

} // namespace detail

template <typename Scalar>
[[nodiscard]] constexpr Scalar solve_temperature_newton_halley(
    Scalar rho_d,
    Scalar rho_v,
    Scalar rho_c,
    Scalar rho_r,
    Scalar eps_int
) noexcept {
    const Scalar rho_l = rho_c + rho_r;

    const Scalar e_ref0 = rho_d * ThermoConstants::kEnergyDry0
        + rho_v * ThermoConstants::kEnergyVapor0
        + rho_l * ThermoConstants::kEnergyCloud0;
    const Scalar delta_eps_int = eps_int - e_ref0;

    const Scalar cv0_tot = rho_d * ThermoConstants::kCvDry0
        + rho_v * ThermoConstants::kCvVapor0
        + rho_l * ThermoConstants::kCvCloud0;
    const Scalar cv1_tot = rho_d * ThermoConstants::kCvDry1
        + rho_v * ThermoConstants::kCvVapor1
        + rho_l * ThermoConstants::kCvLiquid1;
    const Scalar cv2_tot = rho_d * ThermoConstants::kCvDry2
        + rho_v * ThermoConstants::kCvVapor2
        + rho_l * ThermoConstants::kCvLiquid2;

    Scalar dt = detail::scalar_div(delta_eps_int, cv0_tot);

    for (size_t iter = 0; iter < kMaxNewtonHalleyIters; ++iter) {
        const Scalar poly = cv0_tot + dt * (0.5 * cv1_tot + dt * (cv2_tot / 3.0));
        const Scalar f_val = detail::halley_residual_sub(dt, poly, delta_eps_int);
        const Scalar df_dt = cv0_tot + dt * (cv1_tot + dt * cv2_tot);
        const Scalar d2f_dt2 = cv1_tot + 2.0 * cv2_tot * dt;

        const Scalar newton_step = detail::scalar_div(f_val, df_dt);
        const Scalar halley_corr = 1.0 - 0.5 * newton_step * detail::scalar_div(d2f_dt2, df_dt);
        dt = dt - detail::scalar_div(newton_step, halley_corr);
    }

    return ThermoConstants::kTempRefK + dt;
}

// ============================================================================
// 3. 8-Scalar Smooth C^inf Hysteretic Mobility, Korteweg Shift & Criticality
// ============================================================================

template <bool EnableKorteweg = false>
struct HystereticPhaseMobility2D {
    KohlerHystereticParams2D params{};

    constexpr HystereticPhaseMobility2D() noexcept = default;
    constexpr explicit HystereticPhaseMobility2D(KohlerHystereticParams2D p) noexcept : params(p) {}

    template <typename Scalar>
    [[nodiscard]] Scalar nucleation_ramp(Scalar delta_mu_eff) const noexcept {
        const Scalar xi = (delta_mu_eff - params.delta_mu_crit_j_kg) / params.delta_mu_width_j_kg;
        return 0.5 * (1.0 + detail::stable_tanh(xi));
    }

    template <typename Scalar>
    [[nodiscard]] Scalar effective_mobility_coeff(Scalar rho_c, Scalar delta_mu_eff) const noexcept {
        const Scalar active_branch = params.gamma_active_kg2_k_j_m3_s
            * detail::scalar_div(rho_c, Scalar(rho_c + params.rho_c_ref_kg_m3));
        const Scalar nucl_branch = params.gamma_nucl_kg2_k_j_m3_s * nucleation_ramp(delta_mu_eff);
        return active_branch + nucl_branch;
    }

    template <typename Scalar>
    [[nodiscard]] Scalar evaluate_flux(
        Scalar rho_c,
        Scalar delta_mu_eff,
        Scalar temp_k
    ) const noexcept {
        return detail::scalar_div(delta_mu_eff, temp_k)
            * effective_mobility_coeff(rho_c, delta_mu_eff);
    }

    // Evaluates effective chemical potential affinity:
    //   delta_mu_eff = mu_v(rho_v, T) - mu_c(T) + (EnableKorteweg ? kappa * lap_rho_c : 0)
    template <typename Scalar>
    [[nodiscard]] Scalar evaluate_delta_mu_eff(
        Scalar rho_v,
        Scalar temp_k,
        Scalar laplacian_rho_c = Scalar(0.0)
    ) const noexcept {
        const Scalar psi_v = PadeThermodynamicTables::psi_vapor_pade(temp_k);
        const Scalar psi_c = PadeThermodynamicTables::psi_liquid_pade(temp_k);
        const Scalar mu_v = psi_v
            + ThermoConstants::kGasConstVapor * temp_k
            * detail::stable_log_ratio(rho_v, ThermoConstants::kRhoVaporRef);
        Scalar dmu = mu_v - psi_c;
        if constexpr (EnableKorteweg) {
            dmu = dmu + params.kappa_korteweg_j_m5_kg2 * laplacian_rho_c;
        } else {
            (void)laplacian_rho_c;
        }
        return dmu;
    }

    // Evaluates Critical Slowing Down & Second-Law diagnostics (ADR-2D-006, ADR-2D-009)
    // Near the metastable nucleation threshold (rho_c -> 0, delta_mu < delta_mu_crit),
    // the effective thermodynamic stiffness |d(J_phase)/d(rho_c)| flattens, causing
    // the relaxation timescale tau_relax -> inf before rapid nucleation jump.
    [[nodiscard]] core::CriticalSlowingMetrics evaluate_critical_slowing_metrics(
        double rho_c,
        double delta_mu_eff,
        double temp_k
    ) const noexcept {
        constexpr double kRegularizationEps = 1.0e-15;
        const double inv_temp = 1.0 / temp_k;
        const double j_phase = evaluate_flux(rho_c, delta_mu_eff, temp_k);
        const double sigma_phase = j_phase * (delta_mu_eff * inv_temp);

        const double denom_c = rho_c + params.rho_c_ref_kg_m3;
        const double dactive_drho_c = params.gamma_active_kg2_k_j_m3_s
            * (params.rho_c_ref_kg_m3 / (denom_c * denom_c));
        const double dj_drho_c = (delta_mu_eff * inv_temp) * dactive_drho_c;
        const double stiffness = std::abs(dj_drho_c);
        const double tau_relax = 1.0 / (stiffness + kRegularizationEps);

        return core::CriticalSlowingMetrics{
            .delta_mu_j_kg          = delta_mu_eff,
            .effective_stiffness_s1 = stiffness,
            .relaxation_time_s      = tau_relax,
            .entropy_prod_w_m3_k    = sigma_phase
        };
    }

    // Full 8-scalar state evaluation x -> J_phase(x) templated on Scalar
    template <typename Scalar>
    [[nodiscard]] Scalar evaluate_from_state(
        const std::array<Scalar, core::kStateVarCount>& x,
        Scalar geopotential_j_kg = Scalar(0.0),
        Scalar laplacian_rho_c = Scalar(0.0)
    ) const noexcept {
        using core::idx;
        using core::StateVar;

        const Scalar rho_d     = x[idx(StateVar::RHO_D)];
        const Scalar rho_v     = x[idx(StateVar::RHO_V)];
        const Scalar rho_c     = x[idx(StateVar::RHO_C)];
        const Scalar rho_r     = x[idx(StateVar::RHO_R)];
        const Scalar mom_ux    = x[idx(StateVar::MOM_UX)];
        const Scalar mom_uy    = x[idx(StateVar::MOM_UY)];
        const Scalar rho_e_sgs = x[idx(StateVar::RHO_E_SGS)];
        const Scalar eps_tot   = x[idx(StateVar::EPS_TOT)];

        const Scalar rho_tot = rho_d + rho_v + rho_c + rho_r;
        const Scalar mom_sq  = mom_ux * mom_ux + mom_uy * mom_uy;
        const Scalar eps_kin = detail::scalar_div(Scalar(0.5 * mom_sq), rho_tot);
        const Scalar eps_int = eps_tot - eps_kin - rho_e_sgs - rho_tot * geopotential_j_kg;

        const Scalar temp_k = solve_temperature_newton_halley(rho_d, rho_v, rho_c, rho_r, eps_int);
        const Scalar dmu_eff = evaluate_delta_mu_eff(rho_v, temp_k, laplacian_rho_c);
        return evaluate_flux(rho_c, dmu_eff, temp_k);
    }

    // Analytical 8-component Jacobian d(J_phase)/dx for 2D y-column 5x5 Block-Thomas (THERMO-2D-004)
    [[nodiscard]] core::NodeStateArray evaluate_jacobian(
        const core::NodeStateArray& x,
        double geopotential_j_kg = 0.0,
        double laplacian_rho_c = 0.0
    ) const noexcept {
        using core::idx;
        using core::StateVar;

        const double rho_d     = x[idx(StateVar::RHO_D)];
        const double rho_v     = x[idx(StateVar::RHO_V)];
        const double rho_c     = x[idx(StateVar::RHO_C)];
        const double rho_r     = x[idx(StateVar::RHO_R)];
        const double mom_ux    = x[idx(StateVar::MOM_UX)];
        const double mom_uy    = x[idx(StateVar::MOM_UY)];
        const double rho_e_sgs = x[idx(StateVar::RHO_E_SGS)];
        const double eps_tot   = x[idx(StateVar::EPS_TOT)];

        const double rho_tot   = rho_d + rho_v + rho_c + rho_r;
        const double ux_vel    = mom_ux / rho_tot;
        const double uy_vel    = mom_uy / rho_tot;
        const double half_v_sq = 0.5 * (ux_vel * ux_vel + uy_vel * uy_vel);
        const double mom_sq    = mom_ux * mom_ux + mom_uy * mom_uy;
        const double eps_kin   = 0.5 * mom_sq / rho_tot;
        const double eps_int   = eps_tot - eps_kin - rho_e_sgs - rho_tot * geopotential_j_kg;

        const double temp_k   = solve_temperature_newton_halley(rho_d, rho_v, rho_c, rho_r, eps_int);
        const double inv_temp = 1.0 / temp_k;

        const double ln_rho_v = std::log(rho_v / ThermoConstants::kRhoVaporRef);
        const double delta_mu = evaluate_delta_mu_eff(rho_v, temp_k, laplacian_rho_c);

        const double ddmu_drho_v = (ThermoConstants::kGasConstVapor * temp_k) / rho_v;
        const double dpsi_v_dt   = PadeThermodynamicTables::dpsi_dt_vapor_pade(temp_k);
        const double dpsi_c_dt   = PadeThermodynamicTables::dpsi_dt_liquid_pade(temp_k);
        const double ddmu_dt     = (dpsi_v_dt - dpsi_c_dt) + ThermoConstants::kGasConstVapor * ln_rho_v;

        const double denom_c = rho_c + params.rho_c_ref_kg_m3;
        const double active_branch = params.gamma_active_kg2_k_j_m3_s * (rho_c / denom_c);
        const double dactive_drho_c = params.gamma_active_kg2_k_j_m3_s
            * (params.rho_c_ref_kg_m3 / (denom_c * denom_c));

        const double xi      = (delta_mu - params.delta_mu_crit_j_kg) / params.delta_mu_width_j_kg;
        const double tanh_xi = std::tanh(xi);
        const double s_ramp  = 0.5 * (1.0 + tanh_xi);
        const double ds_ddmu = (0.5 * detail::stable_sech_sq(xi)) / params.delta_mu_width_j_kg;

        const double m_eff       = active_branch + params.gamma_nucl_kg2_k_j_m3_s * s_ramp;
        const double dm_eff_ddmu = params.gamma_nucl_kg2_k_j_m3_s * ds_ddmu;

        const double dj_drho_c_explicit = (delta_mu * inv_temp) * dactive_drho_c;
        const double dj_ddmu            = m_eff * inv_temp + (delta_mu * inv_temp) * dm_eff_ddmu;
        const double dj_dt_explicit     = -(delta_mu * inv_temp * inv_temp) * m_eff;
        const double dj_dt_total        = dj_dt_explicit + dj_ddmu * ddmu_dt;

        const double rho_l = rho_c + rho_r;
        const double cv_tot = rho_d * PadeThermodynamicTables::cv_dry(temp_k)
            + rho_v * PadeThermodynamicTables::cv_vapor(temp_k)
            + rho_l * PadeThermodynamicTables::cv_liquid(temp_k);
        const double inv_cv_tot = 1.0 / cv_tot;

        const double e_d = PadeThermodynamicTables::internal_energy_dry(temp_k);
        const double e_v = PadeThermodynamicTables::internal_energy_vapor(temp_k);
        const double e_l = PadeThermodynamicTables::internal_energy_liquid(temp_k);

        const double kin_minus_phi = half_v_sq - geopotential_j_kg;
        const double dt_drho_d     = (kin_minus_phi - e_d) / cv_tot;
        const double dt_drho_v     = (kin_minus_phi - e_v) / cv_tot;
        const double dt_drho_c     = (kin_minus_phi - e_l) / cv_tot;
        const double dt_drho_r     = (kin_minus_phi - e_l) / cv_tot;
        const double dt_dmom_ux    = -ux_vel / cv_tot;
        const double dt_dmom_uy    = -uy_vel / cv_tot;
        const double dt_drho_e_sgs = -inv_cv_tot;
        const double dt_deps_tot   = inv_cv_tot;

        core::NodeStateArray jac{};
        jac[idx(StateVar::RHO_D)]     = dj_dt_total * dt_drho_d;
        jac[idx(StateVar::RHO_V)]     = dj_ddmu * ddmu_drho_v + dj_dt_total * dt_drho_v;
        jac[idx(StateVar::RHO_C)]     = dj_drho_c_explicit + dj_dt_total * dt_drho_c;
        jac[idx(StateVar::RHO_R)]     = dj_dt_total * dt_drho_r;
        jac[idx(StateVar::MOM_UX)]    = dj_dt_total * dt_dmom_ux;
        jac[idx(StateVar::MOM_UY)]    = dj_dt_total * dt_dmom_uy;
        jac[idx(StateVar::RHO_E_SGS)] = dj_dt_total * dt_drho_e_sgs;
        jac[idx(StateVar::EPS_TOT)]   = dj_dt_total * dt_deps_tot;

        return jac;
    }
};

} // namespace metriplectic_2d::thermo

#endif // METRIPLECTIC_2D_THERMO_HYSTERETIC_MOBILITY_HPP
