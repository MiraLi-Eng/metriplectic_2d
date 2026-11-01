#ifndef METRIPLECTIC_2D_THERMO_EOS_CONCEPT_HPP
#define METRIPLECTIC_2D_THERMO_EOS_CONCEPT_HPP

#include <concepts>
#include "metriplectic_2d/core/types.hpp"

namespace metriplectic_2d::thermo {

// ============================================================================
// 2D Cell-Center Thermodynamic Diagnostics (CONTRACTS_2D.md Section 2)
// ============================================================================

struct CellThermoDiagnostics2D {
    double temperature_k;      // T (K)
    double pressure_pa;        // p = (rho_d R_d + rho_v R_v) T (Pa)
    double mu_v;               // Vapor chemical potential (J/kg)
    double mu_c_eff;           // Effective cloud liquid chemical potential mu_c(T) - kappa*lap(rho_c) (J/kg)
    double mu_r;               // Rain chemical potential (J/kg)
    double brunt_vaisala_n2;   // Vertical column N^2 along y (1/s^2)
    core::EOSValidityStatus validity;
};

// ============================================================================
// MoistEOSManifold2D C++20 Concept Port (Zero-Virtual-Dispatch Boundary)
// ============================================================================

template <typename T>
concept MoistEOSManifold2D = requires(
    const T& eos,
    const core::NodeStateArray& x,
    const core::NodeEntropyArray& w,
    core::NodeEntropyArray& w_out,
    core::NodeStateArray& flux_out,
    double normal_x,
    double normal_y
) {
    // 4-iteration branchless Newton-Halley cell-center diagnostic inversion
    { eos.evaluate_cell_diagnostics(x, w_out) } noexcept -> std::same_as<CellThermoDiagnostics2D>;

    // 2D zero-root-finding quadrature evaluation w(tau_q) -> f_rev(w(tau_q), nx, ny)
    { eos.evaluate_reversible_gas_flux(w, normal_x, normal_y, flux_out) } noexcept -> std::same_as<void>;

    // Smooth C^inf one-sided hysteretic phase mobility J_phase(rho_c, dmu_eff, T)
    { eos.evaluate_phase_flux(x, double{}, double{}) } noexcept -> std::same_as<double>;
};

} // namespace metriplectic_2d::thermo

#endif // METRIPLECTIC_2D_THERMO_EOS_CONCEPT_HPP
