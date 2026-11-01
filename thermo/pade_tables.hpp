#ifndef METRIPLECTIC_2D_THERMO_PADE_TABLES_HPP
#define METRIPLECTIC_2D_THERMO_PADE_TABLES_HPP

#include <cmath>
#include <cstddef>
#include "metriplectic_2d/core/types.hpp"

namespace metriplectic_2d::thermo {

// ============================================================================
// 1. Canonical SI Thermodynamic Constants & Domain Bounds (ADR-2D-001)
// ============================================================================

struct ThermoConstants {
    static constexpr double kTempValidMinK = 180.0;
    static constexpr double kTempValidMaxK = 350.0;
    static constexpr double kTempHardMinK  = 150.0;
    static constexpr double kTempHardMaxK  = 400.0;

    static constexpr double kTempRefK      = 273.15;
    static constexpr double kPressSatRefPa = 611.657;
    static constexpr double kPressDryRefPa = 100000.0;

    static constexpr double kGasConstDry   = 287.058;
    static constexpr double kGasConstVapor = 461.520;

    static constexpr double kRhoDryRef     = kPressDryRefPa / (kGasConstDry * kTempRefK);
    static constexpr double kRhoVaporRef   = kPressSatRefPa / (kGasConstVapor * kTempRefK);

    static constexpr double kCvDry0        = 717.645;
    static constexpr double kCvVapor0      = 1403.00;
    static constexpr double kCvCloud0      = 4218.00;
    static constexpr double kCvRain0       = 4218.00;

    static constexpr double kCvDry1        = 0.055;
    static constexpr double kCvDry2        = 1.8e-4;
    static constexpr double kCvVapor1      = 0.380;
    static constexpr double kCvVapor2      = 4.2e-4;
    static constexpr double kCvLiquid1     = -0.650;
    static constexpr double kCvLiquid2     = 1.4e-3;

    static constexpr double kLatentVap0    = 2.5008e6;
    static constexpr double kEnergyDry0    = kCvDry0 * kTempRefK;
    static constexpr double kEnergyCloud0  = 0.0;
    static constexpr double kEnergyRain0   = 0.0;
    static constexpr double kEnergyVapor0  = kLatentVap0 - kGasConstVapor * kTempRefK;

    static constexpr double kEntropyDry0   = kCvDry0 + kGasConstDry;
    static constexpr double kEntropyCloud0 = 0.0;
    static constexpr double kEntropyRain0  = 0.0;
    static constexpr double kEntropyVapor0 = kLatentVap0 / kTempRefK;

    // Gravitational acceleration along -y vertical column (m / s^2)
    static constexpr double kGravity       = 9.80665;
};

// ============================================================================
// 2. Branchless Rational Padé [13/12] Continued-Fraction Approximants
// ============================================================================

namespace detail {

[[nodiscard]] constexpr double scalar_div(double a, double b) noexcept {
    return a / b;
}

template <typename T>
    requires requires(const T& v) { v.real(); v.imag(); }
[[nodiscard]] inline T scalar_div(const T& a, const T& b) noexcept {
    const double br = b.real();
    const double bi = b.imag();
    if (std::abs(bi) <= 1.0e-8 * std::abs(br)) {
        const double re = a.real() / br;
        const double im = std::fma(-re, bi, a.imag()) / br;
        return T(re, im);
    }
    return a / b;
}

template <typename Scalar>
[[nodiscard]] constexpr Scalar pade_log_ratio(Scalar temp_k) noexcept {
    constexpr double t0 = ThermoConstants::kTempRefK;
    const Scalar u = scalar_div(Scalar(temp_k - t0), Scalar(temp_k + t0));
    const Scalar z = u * u;

    constexpr double c1 = 1.0 / 3.0;
    constexpr double c2 = 4.0 / 15.0;
    constexpr double c3 = 9.0 / 35.0;
    constexpr double c4 = 16.0 / 63.0;
    constexpr double c5 = 25.0 / 99.0;
    constexpr double c6 = 36.0 / 143.0;
    constexpr double c7 = 49.0 / 195.0;

    Scalar cf = 1.0 - c7 * z;
    cf = 1.0 - scalar_div(Scalar(c6 * z), cf);
    cf = 1.0 - scalar_div(Scalar(c5 * z), cf);
    cf = 1.0 - scalar_div(Scalar(c4 * z), cf);
    cf = 1.0 - scalar_div(Scalar(c3 * z), cf);
    cf = 1.0 - scalar_div(Scalar(c2 * z), cf);
    cf = 1.0 - scalar_div(Scalar(c1 * z), cf);

    return scalar_div(Scalar(2.0 * u), cf);
}

template <typename Scalar>
[[nodiscard]] constexpr Scalar pade_log_ratio_derivative(Scalar temp_k) noexcept {
    constexpr double t0 = ThermoConstants::kTempRefK;
    const Scalar sum_t = temp_k + t0;
    const Scalar u = (temp_k - t0) / sum_t;
    const Scalar du_dt = (2.0 * t0) / (sum_t * sum_t);
    const Scalar z = u * u;

    constexpr double c1 = 1.0 / 3.0;
    constexpr double c2 = 4.0 / 15.0;
    constexpr double c3 = 9.0 / 35.0;
    constexpr double c4 = 16.0 / 63.0;
    constexpr double c5 = 25.0 / 99.0;
    constexpr double c6 = 36.0 / 143.0;
    constexpr double c7 = 49.0 / 195.0;

    Scalar cf = 1.0 - c7 * z;
    Scalar dcf_dz = -c7;

    auto step_cf = [&](double ck) noexcept {
        const Scalar inv_cf = 1.0 / cf;
        const Scalar next_cf = 1.0 - (ck * z) * inv_cf;
        const Scalar next_dcf_dz = -ck * (cf - z * dcf_dz) * (inv_cf * inv_cf);
        cf = next_cf;
        dcf_dz = next_dcf_dz;
    };

    step_cf(c6);
    step_cf(c5);
    step_cf(c4);
    step_cf(c3);
    step_cf(c2);
    step_cf(c1);

    const Scalar df_du = 2.0 * (cf - 2.0 * z * dcf_dz) / (cf * cf);
    return df_du * du_dt;
}

template <typename Scalar>
[[nodiscard]] constexpr Scalar species_energy(
    Scalar temp_k,
    double e0,
    double cv0,
    double cv1,
    double cv2
) noexcept {
    const Scalar dt = temp_k - ThermoConstants::kTempRefK;
    return e0 + dt * (cv0 + dt * (0.5 * cv1 + dt * (cv2 / 3.0)));
}

template <typename Scalar>
[[nodiscard]] constexpr Scalar species_cv(
    Scalar temp_k,
    double cv0,
    double cv1,
    double cv2
) noexcept {
    const Scalar dt = temp_k - ThermoConstants::kTempRefK;
    return cv0 + dt * (cv1 + dt * cv2);
}

template <typename Scalar>
[[nodiscard]] constexpr Scalar species_ref_entropy_pade(
    Scalar temp_k,
    double s0,
    double cv0,
    double cv1,
    double cv2
) noexcept {
    constexpr double t0 = ThermoConstants::kTempRefK;
    const Scalar dt = temp_k - t0;
    const Scalar ln_ratio = pade_log_ratio(temp_k);
    const double c_log = cv0 - cv1 * t0 + cv2 * t0 * t0;
    const double c_lin = cv1 - cv2 * t0;
    const double c_quad = 0.5 * cv2;
    return s0 + c_log * ln_ratio + c_lin * dt + c_quad * (temp_k * temp_k - t0 * t0);
}

template <typename Scalar>
[[nodiscard]] constexpr Scalar species_dpsi_dt_pade(
    Scalar temp_k,
    double s0,
    double cv0,
    double cv1,
    double cv2,
    double gas_const_r
) noexcept {
    constexpr double t0 = ThermoConstants::kTempRefK;
    const double c_log = cv0 - cv1 * t0 + cv2 * t0 * t0;
    const double c_lin = cv1 - cv2 * t0;
    const double c_quad = 0.5 * cv2;

    const Scalar ds_ref = species_ref_entropy_pade(temp_k, 0.0, cv0, cv1, cv2);
    const Scalar d_ds_ref_dt = c_log * pade_log_ratio_derivative(temp_k)
        + c_lin + 2.0 * c_quad * temp_k;
    const Scalar de_dt = species_cv(temp_k, cv0, cv1, cv2);
    return (gas_const_r - s0) - ds_ref + (de_dt - temp_k * d_ds_ref_dt);
}

} // namespace detail

// ============================================================================
// 3. Constexpr Padé/Chebyshev Constitutive Spline Functions
// ============================================================================

struct PadeThermodynamicTables {
    template <typename Scalar>
    [[nodiscard]] static constexpr Scalar internal_energy_dry(Scalar temp_k) noexcept {
        return detail::species_energy(
            temp_k,
            ThermoConstants::kEnergyDry0,
            ThermoConstants::kCvDry0,
            ThermoConstants::kCvDry1,
            ThermoConstants::kCvDry2
        );
    }

    template <typename Scalar>
    [[nodiscard]] static constexpr Scalar internal_energy_vapor(Scalar temp_k) noexcept {
        return detail::species_energy(
            temp_k,
            ThermoConstants::kEnergyVapor0,
            ThermoConstants::kCvVapor0,
            ThermoConstants::kCvVapor1,
            ThermoConstants::kCvVapor2
        );
    }

    template <typename Scalar>
    [[nodiscard]] static constexpr Scalar internal_energy_liquid(Scalar temp_k) noexcept {
        return detail::species_energy(
            temp_k,
            ThermoConstants::kEnergyCloud0,
            ThermoConstants::kCvCloud0,
            ThermoConstants::kCvLiquid1,
            ThermoConstants::kCvLiquid2
        );
    }

    template <typename Scalar>
    [[nodiscard]] static constexpr Scalar cv_dry(Scalar temp_k) noexcept {
        return detail::species_cv(
            temp_k,
            ThermoConstants::kCvDry0,
            ThermoConstants::kCvDry1,
            ThermoConstants::kCvDry2
        );
    }

    template <typename Scalar>
    [[nodiscard]] static constexpr Scalar cv_vapor(Scalar temp_k) noexcept {
        return detail::species_cv(
            temp_k,
            ThermoConstants::kCvVapor0,
            ThermoConstants::kCvVapor1,
            ThermoConstants::kCvVapor2
        );
    }

    template <typename Scalar>
    [[nodiscard]] static constexpr Scalar cv_liquid(Scalar temp_k) noexcept {
        return detail::species_cv(
            temp_k,
            ThermoConstants::kCvCloud0,
            ThermoConstants::kCvLiquid1,
            ThermoConstants::kCvLiquid2
        );
    }

    template <typename Scalar>
    [[nodiscard]] static constexpr Scalar psi_dry_pade(Scalar temp_k) noexcept {
        constexpr double t0 = ThermoConstants::kTempRefK;
        constexpr double psi0 = ThermoConstants::kEnergyDry0
            + ThermoConstants::kGasConstDry * t0
            - t0 * ThermoConstants::kEntropyDry0;
        const Scalar dt = temp_k - t0;
        const Scalar de = detail::species_energy(
            temp_k, 0.0, ThermoConstants::kCvDry0, ThermoConstants::kCvDry1, ThermoConstants::kCvDry2
        );
        const Scalar ds = detail::species_ref_entropy_pade(
            temp_k, 0.0, ThermoConstants::kCvDry0, ThermoConstants::kCvDry1, ThermoConstants::kCvDry2
        );
        return psi0 + (ThermoConstants::kGasConstDry - ThermoConstants::kEntropyDry0) * dt
            + (de - temp_k * ds);
    }

    template <typename Scalar>
    [[nodiscard]] static constexpr Scalar psi_vapor_pade(Scalar temp_k) noexcept {
        constexpr double t0 = ThermoConstants::kTempRefK;
        const Scalar dt = temp_k - t0;
        const Scalar de = detail::species_energy(
            temp_k, 0.0, ThermoConstants::kCvVapor0, ThermoConstants::kCvVapor1, ThermoConstants::kCvVapor2
        );
        const Scalar ds = detail::species_ref_entropy_pade(
            temp_k, 0.0, ThermoConstants::kCvVapor0, ThermoConstants::kCvVapor1, ThermoConstants::kCvVapor2
        );
        return (ThermoConstants::kGasConstVapor - ThermoConstants::kEntropyVapor0) * dt
            + (de - temp_k * ds);
    }

    template <typename Scalar>
    [[nodiscard]] static constexpr Scalar psi_liquid_pade(Scalar temp_k) noexcept {
        const Scalar de = detail::species_energy(
            temp_k, 0.0, ThermoConstants::kCvCloud0, ThermoConstants::kCvLiquid1, ThermoConstants::kCvLiquid2
        );
        const Scalar ds = detail::species_ref_entropy_pade(
            temp_k, 0.0, ThermoConstants::kCvCloud0, ThermoConstants::kCvLiquid1, ThermoConstants::kCvLiquid2
        );
        return de - temp_k * ds;
    }

    template <typename Scalar>
    [[nodiscard]] static constexpr Scalar dpsi_dt_dry_pade(Scalar temp_k) noexcept {
        return detail::species_dpsi_dt_pade(
            temp_k,
            ThermoConstants::kEntropyDry0,
            ThermoConstants::kCvDry0,
            ThermoConstants::kCvDry1,
            ThermoConstants::kCvDry2,
            ThermoConstants::kGasConstDry
        );
    }

    template <typename Scalar>
    [[nodiscard]] static constexpr Scalar dpsi_dt_vapor_pade(Scalar temp_k) noexcept {
        return detail::species_dpsi_dt_pade(
            temp_k,
            ThermoConstants::kEntropyVapor0,
            ThermoConstants::kCvVapor0,
            ThermoConstants::kCvVapor1,
            ThermoConstants::kCvVapor2,
            ThermoConstants::kGasConstVapor
        );
    }

    template <typename Scalar>
    [[nodiscard]] static constexpr Scalar dpsi_dt_liquid_pade(Scalar temp_k) noexcept {
        return detail::species_dpsi_dt_pade(
            temp_k,
            ThermoConstants::kEntropyCloud0,
            ThermoConstants::kCvCloud0,
            ThermoConstants::kCvLiquid1,
            ThermoConstants::kCvLiquid2,
            0.0
        );
    }

    [[nodiscard]] static double psi_dry_exact_log(double temp_k) noexcept {
        constexpr double t0 = ThermoConstants::kTempRefK;
        constexpr double psi0 = ThermoConstants::kEnergyDry0
            + ThermoConstants::kGasConstDry * t0
            - t0 * ThermoConstants::kEntropyDry0;
        const double dt = temp_k - t0;
        const double ln_ratio = std::log(temp_k / t0);
        const double c_log = ThermoConstants::kCvDry0
            - ThermoConstants::kCvDry1 * t0
            + ThermoConstants::kCvDry2 * t0 * t0;
        const double c_lin = ThermoConstants::kCvDry1 - ThermoConstants::kCvDry2 * t0;
        const double c_quad = 0.5 * ThermoConstants::kCvDry2;
        const double ds_ref = c_log * ln_ratio + c_lin * dt + c_quad * (temp_k * temp_k - t0 * t0);
        const double de_d = detail::species_energy(
            temp_k, 0.0, ThermoConstants::kCvDry0, ThermoConstants::kCvDry1, ThermoConstants::kCvDry2
        );
        return psi0 + (ThermoConstants::kGasConstDry - ThermoConstants::kEntropyDry0) * dt
            + de_d - temp_k * ds_ref;
    }

    [[nodiscard]] static double psi_vapor_exact_log(double temp_k) noexcept {
        constexpr double t0 = ThermoConstants::kTempRefK;
        const double dt = temp_k - t0;
        const double ln_ratio = std::log(temp_k / t0);
        const double c_log = ThermoConstants::kCvVapor0
            - ThermoConstants::kCvVapor1 * t0
            + ThermoConstants::kCvVapor2 * t0 * t0;
        const double c_lin = ThermoConstants::kCvVapor1 - ThermoConstants::kCvVapor2 * t0;
        const double c_quad = 0.5 * ThermoConstants::kCvVapor2;
        const double ds_ref = c_log * ln_ratio + c_lin * dt + c_quad * (temp_k * temp_k - t0 * t0);
        const double de_v = detail::species_energy(
            temp_k, 0.0, ThermoConstants::kCvVapor0, ThermoConstants::kCvVapor1, ThermoConstants::kCvVapor2
        );
        return (ThermoConstants::kGasConstVapor - ThermoConstants::kEntropyVapor0) * dt
            + de_v - temp_k * ds_ref;
    }

    template <typename Scalar>
    [[nodiscard]] static Scalar saturation_vapor_pressure_pa(Scalar temp_k) noexcept {
        using std::exp;
        const Scalar dpsi = psi_vapor_pade(temp_k) - psi_liquid_pade(temp_k);
        const Scalar rho_v_sat = ThermoConstants::kRhoVaporRef
            * exp(-dpsi / (ThermoConstants::kGasConstVapor * temp_k));
        return rho_v_sat * ThermoConstants::kGasConstVapor * temp_k;
    }

    template <typename Scalar>
    [[nodiscard]] static constexpr Scalar rain_terminal_speed_m_s(Scalar rho_r) noexcept {
        constexpr double w_max = 9.65;
        constexpr double rho_r_scale = 1e-3;
        const Scalar q = rho_r / (rho_r + rho_r_scale);
        return w_max * q * (0.35 + 0.65 * q) / (1.0 + 0.15 * q);
    }

    [[nodiscard]] static constexpr core::EOSValidityStatus classify_temperature_domain(
        double temp_k
    ) noexcept {
        if (!std::isfinite(temp_k)) {
            return core::EOSValidityStatus::INVALID;
        }
        if (temp_k >= ThermoConstants::kTempValidMinK && temp_k <= ThermoConstants::kTempValidMaxK) {
            return core::EOSValidityStatus::VALID;
        }
        if (temp_k >= ThermoConstants::kTempHardMinK && temp_k <= ThermoConstants::kTempHardMaxK) {
            return core::EOSValidityStatus::EXTRAPOLATED;
        }
        return core::EOSValidityStatus::INVALID;
    }
};

} // namespace metriplectic_2d::thermo

#endif // METRIPLECTIC_2D_THERMO_PADE_TABLES_HPP
