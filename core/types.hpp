#ifndef METRIPLECTIC_2D_CORE_TYPES_HPP
#define METRIPLECTIC_2D_CORE_TYPES_HPP

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>

#if __has_include(<stdfloat>)
#include <stdfloat>
#endif

namespace metriplectic_2d::core {

// ============================================================================
// 1. Canonical SI Quantity & Orthogonal Status Dimensions (ADR-2D-001)
// ============================================================================
// Bare "VALID" or "SUCCESS" without a status dimension qualifier is prohibited.

struct Quantity {
    double value;
    std::string_view unit;

    [[nodiscard]] constexpr bool has_finite_value() const noexcept {
        return std::isfinite(value);
    }
};

enum class IntegrationStatus : uint8_t {
    NOT_RUN = 0,
    COMPLETED = 1,
    DIVERGED = 2,
    CANCELLED = 3
};

enum class ClosureStatus : uint8_t {
    NOT_CHECKED = 0,
    PASSED = 1,
    FAILED = 2
};

enum class PhysicalValidityStatus : uint8_t {
    UNKNOWN = 0,
    VALID = 1,
    POSITIVITY_VIOLATED = 2,
    INVALID = 3
};

enum class EOSValidityStatus : uint8_t {
    UNKNOWN = 0,
    VALID = 1,
    EXTRAPOLATED = 2,
    INVALID = 3
};

// ============================================================================
// 2. Dedicated 8-Scalar 2D Vertical-Slice State Vector (ADR-2D-002)
// ============================================================================
// Coordinate Convention:
//   x = Horizontal axis
//   y = Vertical atmospheric column (Phi_grav = g * y)
//
// Each NodeStateArray is 8 * sizeof(double) == 64 bytes (1 CPU cache line).

enum class StateVar : size_t {
    RHO_D     = 0, // Dry air density (kg/m^3)
    RHO_V     = 1, // Water vapor density (kg/m^3)
    RHO_C     = 2, // Cloud liquid water density (kg/m^3)
    RHO_R     = 3, // Rain water density (kg/m^3)
    MOM_UX    = 4, // Horizontal x-momentum density rho * u_x (kg/(m^2 s))
    MOM_UY    = 5, // Vertical column y-momentum density rho * u_y (kg/(m^2 s))
    RHO_E_SGS = 6, // Sub-grid turbulent kinetic energy density rho * e_sgs (J/m^3)
    EPS_TOT   = 7, // Total energy density eps (J/m^3)
    COUNT     = 8
};

inline constexpr size_t kStateVarCount = static_cast<size_t>(StateVar::COUNT);
static_assert(kStateVarCount == 8, "2D state vector must have exactly 8 scalars (64 bytes)");

using NodeStateArray   = std::array<double, kStateVarCount>;
using NodeEntropyArray = std::array<double, kStateVarCount>;

static_assert(sizeof(NodeStateArray) == 64, "NodeStateArray must match a 64-byte cache line");

[[nodiscard]] constexpr size_t idx(StateVar var) noexcept {
    return static_cast<size_t>(var);
}

// ============================================================================
// 3. Critical Slowing Down & Hysteresis Telemetry Diagnostics (ADR-2D-006/009)
// ============================================================================

struct CriticalSlowingMetrics {
    double delta_mu_j_kg;          // Effective affinity mu_v - mu_c^eff (J/kg)
    double effective_stiffness_s1; // |d(J_phase)/d(rho_c)| proxy for |d^2 V_eff / d psi^2| (1/s)
    double relaxation_time_s;      // tau_relax = 1 / (effective_stiffness_s1 + eps) (s)
    double entropy_prod_w_m3_k;    // sigma_phase = J_phase * (delta_mu / T) >= 0 (W/(m^3 K))
};

// ============================================================================
// 4. Hardware-Portable bfloat16_t Wrapper
// ============================================================================

#if defined(__STDCPP_BFLOAT16_T__)
using bfloat16_t = std::bfloat16_t;
#else
struct bfloat16_t {
    uint16_t bits = 0;

    constexpr bfloat16_t() noexcept = default;

    static bfloat16_t from_float(float val) noexcept {
        uint32_t u32 = std::bit_cast<uint32_t>(val);
        uint32_t rounding_bias = 0x00007FFFu + ((u32 >> 16u) & 1u);
        bfloat16_t out;
        out.bits = static_cast<uint16_t>((u32 + rounding_bias) >> 16u);
        return out;
    }

    [[nodiscard]] float to_float() const noexcept {
        uint32_t u32 = static_cast<uint32_t>(bits) << 16u;
        return std::bit_cast<float>(u32);
    }
};
#endif

} // namespace metriplectic_2d::core

#endif // METRIPLECTIC_2D_CORE_TYPES_HPP
