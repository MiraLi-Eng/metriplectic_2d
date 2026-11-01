#ifndef METRIPLECTIC_2D_BRACKETS_REVERSIBLE_L_2D_HPP
#define METRIPLECTIC_2D_BRACKETS_REVERSIBLE_L_2D_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include "metriplectic_2d/core/types.hpp"
#include "metriplectic_2d/spatial/sbp_operators_2d.hpp"
#include "metriplectic_2d/thermo/moist_eos.hpp"
#include "metriplectic_2d/thermo/pade_tables.hpp"

namespace metriplectic_2d::brackets {

// ============================================================================
// 2D Reversible Lie-Poisson Bracket L(x) dE/dx (ADR-2D-004, ADR-2D-005)
// ============================================================================
// Discretizes reversible Euler convection, condensate advection, sub-grid TKE
// advection, vertical gravity (-rho * g * e_y), and optional Korteweg capillary
// stress on the 2D (x = horizontal, y = vertical column) grid.
//
// Provably satisfies discrete GENERIC degeneracy & conservation to machine
// precision (< 1e-14):
//   1. Mass & Energy Conservation: sum_{x,y} P_x P_y (L(x) dE/dx)_{rho_k, eps} == 0
//   2. Entropy Degeneracy:         sum_{x,y} P_x P_y w^T (L(x) dE/dx) == 0

template <bool EnableKorteweg = false>
class ReversibleBracketL2D {
public:
    // 4-point Gauss-Legendre quadrature on [0, 1] for Tadmor-Fjordholm path integral
    static constexpr std::array<double, 4> kQuadTau = {
        0.06943184420297371239,
        0.33000947820757186760,
        0.66999052179242813240,
        0.93056815579702628761
    };
    static constexpr std::array<double, 4> kQuadWeight = {
        0.17392742256872692869,
        0.32607257743127307131,
        0.32607257743127307131,
        0.17392742256872692869
    };

    struct TwoPointFluxResult2D {
        core::NodeStateArray flux{};
        double grav_coeff_left  = 0.0; // A_{LR} for compatible gravitational force
        double grav_coeff_right = 0.0; // A_{RL} for compatible gravitational force
        double kort_coeff_left  = 0.0; // K_{LR} for compatible Korteweg capillary force
        double kort_coeff_right = 0.0; // K_{RL} for compatible Korteweg capillary force
    };

    // Evaluates the symmetric two-point entropy-conservative flux f^*(x_L, x_R, nx, ny)
    // between states L and R, including:
    //   1. Gaseous Tadmor-Fjordholm path integral (rho_d, rho_v, m_gas, eps_gas)
    //   2. Symmetric arithmetic-mean condensate fluxes bar(rho_c)*v_n^*, bar(rho_r)*v_n^*
    //      paired with path-integrated liquid energy/momentum so (w_L - w_R)^T f_cond^* == 0
    //   3. Symmetric SGS kinetic energy advection bar(rho*e_sgs)*v_n^*
    //   4. Symmetric geopotential energy flux bar(Phi)_{LR} * f_{rho, tot}^*
    //   5. Optional Korteweg capillary energy flux -bar(kappa*lap(rho_c))_{LR} * f_{rho_c}^*
    [[nodiscard]] static TwoPointFluxResult2D evaluate_two_point_ec_flux(
        const core::NodeStateArray& x_l,
        const core::NodeStateArray& x_r,
        const core::NodeEntropyArray& w_local_l,
        const core::NodeEntropyArray& w_local_r,
        double phi_eff_l,
        double phi_eff_r,
        double normal_x,
        double normal_y,
        const thermo::MoistPadeEOS2D<EnableKorteweg>& eos,
        double kort_pot_l = 0.0,
        double kort_pot_r = 0.0
    ) noexcept {
        using core::idx;
        using core::StateVar;
        using thermo::PadeThermodynamicTables;

        TwoPointFluxResult2D res{};

        double path_ux         = 0.0;
        double path_uy         = 0.0;
        double path_vn         = 0.0;
        double path_e_liq_tot  = 0.0;
        double a_lr_gas        = 0.0;
        double a_rl_gas        = 0.0;

        for (size_t q = 0; q < 4; ++q) {
            const double tau    = kQuadTau[q];
            const double weight = kQuadWeight[q];

            core::NodeEntropyArray w_q{};
            for (size_t k = 0; k < core::kStateVarCount; ++k) {
                w_q[k] = w_local_l[k] + tau * (w_local_r[k] - w_local_l[k]);
            }

            core::NodeStateArray flux_gas_q{};
            eos.evaluate_reversible_gas_flux(w_q, normal_x, normal_y, flux_gas_q);
            for (size_t k = 0; k < core::kStateVarCount; ++k) {
                res.flux[k] += weight * flux_gas_q[k];
            }

            const double w_eps_q = w_q[idx(StateVar::EPS_TOT)];
            const double temp_q  = 1.0 / w_eps_q;
            const double ux_q    = -w_q[idx(StateVar::MOM_UX)] * temp_q;
            const double uy_q    = -w_q[idx(StateVar::MOM_UY)] * temp_q;
            const double vn_q    = ux_q * normal_x + uy_q * normal_y;
            const double half_v2 = 0.5 * (ux_q * ux_q + uy_q * uy_q);

            // Exact Legendre-conjugate liquid internal energy e_{l,conj}(T) = psi_l(T) - T * dpsi_l/dT
            const double psi_l_q     = PadeThermodynamicTables::psi_liquid_pade(temp_q);
            const double dpsi_l_dt_q = PadeThermodynamicTables::dpsi_dt_liquid_pade(temp_q);
            const double e_l_conj_q  = psi_l_q - temp_q * dpsi_l_dt_q;

            path_ux        += weight * ux_q;
            path_uy        += weight * uy_q;
            path_vn        += weight * vn_q;
            path_e_liq_tot += weight * (e_l_conj_q + half_v2);

            // Recover gas density at quadrature node for compatible gravity factor A_{LR}
            const double psi_d_q = PadeThermodynamicTables::psi_dry_pade(temp_q);
            const double psi_v_q = PadeThermodynamicTables::psi_vapor_pade(temp_q);
            const double arg_d = (half_v2 - psi_d_q - temp_q * w_q[idx(StateVar::RHO_D)])
                / (thermo::ThermoConstants::kGasConstDry * temp_q);
            const double arg_v = (half_v2 - psi_v_q - temp_q * w_q[idx(StateVar::RHO_V)])
                / (thermo::ThermoConstants::kGasConstVapor * temp_q);
            const double rho_gas_q = thermo::ThermoConstants::kRhoDryRef * std::exp(arg_d)
                + thermo::ThermoConstants::kRhoVaporRef * std::exp(arg_v);

            a_lr_gas += weight * (1.0 - tau) * (rho_gas_q * temp_q);
            a_rl_gas += weight * tau * (rho_gas_q * temp_q);
        }

        // 2. Symmetric arithmetic-mean condensate fluxes bar(rho_c)*v_n^*, bar(rho_r)*v_n^* (ADR-2D-005)
        const double rho_c_bar = 0.5 * (x_l[idx(StateVar::RHO_C)] + x_r[idx(StateVar::RHO_C)]);
        const double rho_r_bar = 0.5 * (x_l[idx(StateVar::RHO_R)] + x_r[idx(StateVar::RHO_R)]);
        const double rho_l_bar = rho_c_bar + rho_r_bar;

        const double flux_rho_c = rho_c_bar * path_vn;
        const double flux_rho_r = rho_r_bar * path_vn;
        const double flux_rho_l = flux_rho_c + flux_rho_r;

        res.flux[idx(StateVar::RHO_C)]  += flux_rho_c;
        res.flux[idx(StateVar::RHO_R)]  += flux_rho_r;
        res.flux[idx(StateVar::MOM_UX)] += flux_rho_l * path_ux;
        res.flux[idx(StateVar::MOM_UY)] += flux_rho_l * path_uy;
        res.flux[idx(StateVar::EPS_TOT)] += flux_rho_l * path_e_liq_tot;

        // 3. Sub-grid kinetic energy advection (since w_sgs + w_eps == 0 identically)
        const double sgs_bar  = 0.5 * (x_l[idx(StateVar::RHO_E_SGS)] + x_r[idx(StateVar::RHO_E_SGS)]);
        const double flux_sgs = sgs_bar * path_vn;
        res.flux[idx(StateVar::RHO_E_SGS)] += flux_sgs;
        res.flux[idx(StateVar::EPS_TOT)]   += flux_sgs;

        // 4. Geopotential energy flux bar(Phi)_{LR} * f_{rho, tot}^*
        const double flux_rho_tot = res.flux[idx(StateVar::RHO_D)]
            + res.flux[idx(StateVar::RHO_V)]
            + flux_rho_c
            + flux_rho_r;
        const double phi_bar = 0.5 * (phi_eff_l + phi_eff_r);
        res.flux[idx(StateVar::EPS_TOT)] += phi_bar * flux_rho_tot;

        // Factor path_vn = -c_l * w_{uy, L} - c_r * w_{uy, R} for condensate gravity & Korteweg coupling
        double c_l_cond = 0.0;
        double c_r_cond = 0.0;
        double k_l_kort = 0.0;
        double k_r_kort = 0.0;
        for (size_t q = 0; q < 4; ++q) {
            const double tau    = kQuadTau[q];
            const double weight = kQuadWeight[q];
            const double w_eps_q = w_local_l[idx(StateVar::EPS_TOT)]
                + tau * (w_local_r[idx(StateVar::EPS_TOT)] - w_local_l[idx(StateVar::EPS_TOT)]);
            const double temp_q = 1.0 / w_eps_q;
            c_l_cond += weight * (1.0 - tau) * (rho_l_bar * temp_q);
            c_r_cond += weight * tau * (rho_l_bar * temp_q);
            if constexpr (EnableKorteweg) {
                k_l_kort += weight * (1.0 - tau) * (rho_c_bar * temp_q);
                k_r_kort += weight * tau * (rho_c_bar * temp_q);
            }
        }

        if constexpr (EnableKorteweg) {
            const double kort_bar = 0.5 * (kort_pot_l + kort_pot_r);
            res.flux[idx(StateVar::EPS_TOT)] -= kort_bar * flux_rho_c;
            res.kort_coeff_left  = k_l_kort;
            res.kort_coeff_right = k_r_kort;
        } else {
            (void)kort_pot_l;
            (void)kort_pot_r;
        }

        res.grav_coeff_left  = a_lr_gas + c_l_cond;
        res.grav_coeff_right = a_rl_gas + c_r_cond;
        return res;
    }

    // Evaluates the full 2D reversible bracket tendency rhs_out += L(x) dE/dx
    // and populates true cell-center entropy variables w_grid_out (including -Phi_grav / T).
    static core::EOSValidityStatus apply(
        const core::NodeStateArray* state_grid,
        core::NodeEntropyArray* w_grid_out,
        core::NodeStateArray* rhs_grid,
        const spatial::GridGeometry2D& geom,
        const thermo::MoistPadeEOS2D<EnableKorteweg>& eos
    ) noexcept {
        using core::idx;
        using core::StateVar;
        using spatial::SBPOperators2D;
        using thermo::ThermoConstants;

        core::EOSValidityStatus worst_status = core::EOSValidityStatus::VALID;

        // Step 1: Evaluate cell-center diagnostics and true entropy variables w = dS/dx
        for (size_t iy = 0; iy < geom.ny; ++iy) {
            const double phi_grav = ThermoConstants::kGravity * geom.y_coord(iy);
            for (size_t ix = 0; ix < geom.nx; ++ix) {
                const size_t node_id = geom.linear_index(ix, iy);
                double lap_rho_c = 0.0;
                if constexpr (EnableKorteweg) {
                    lap_rho_c = SBPOperators2D::laplacian_2d_node_var(
                        state_grid, geom, ix, iy, StateVar::RHO_C
                    );
                }
                const auto diag = eos.evaluate_cell_diagnostics(
                    state_grid[node_id], w_grid_out[node_id], phi_grav, lap_rho_c
                );
                if (diag.validity == core::EOSValidityStatus::INVALID) {
                    return core::EOSValidityStatus::INVALID;
                }
                if (diag.validity == core::EOSValidityStatus::EXTRAPOLATED) {
                    worst_status = core::EOSValidityStatus::EXTRAPOLATED;
                }
            }
        }

        // Helper to evaluate capillary potential Psi_kort = kappa * laplacian(rho_c)
        auto eval_kort_pot = [&](size_t ix, size_t iy) noexcept -> double {
            if constexpr (EnableKorteweg) {
                const double lap_rho_c = SBPOperators2D::laplacian_2d_node_var(
                    state_grid, geom, ix, iy, StateVar::RHO_C
                );
                return eos.phase_mobility().params.kappa_korteweg_j_m5_kg2 * lap_rho_c;
            } else {
                (void)ix;
                (void)iy;
                return 0.0;
            }
        };

        // Helper to recover local (zero-geopotential, zero-Korteweg) entropy array tilde(w)
        // from true entropy array w_grid_out at (ix, iy)
        auto make_local_w = [&](size_t ix, size_t iy) noexcept -> core::NodeEntropyArray {
            const size_t id = geom.linear_index(ix, iy);
            core::NodeEntropyArray w_loc = w_grid_out[id];
            const double w_eps = w_loc[idx(StateVar::EPS_TOT)];
            const double phi_grav = ThermoConstants::kGravity * geom.y_coord(iy);
            const double shift_grav = phi_grav * w_eps;
            w_loc[idx(StateVar::RHO_D)] += shift_grav;
            w_loc[idx(StateVar::RHO_V)] += shift_grav;
            w_loc[idx(StateVar::RHO_C)] += shift_grav;
            w_loc[idx(StateVar::RHO_R)] += shift_grav;
            if constexpr (EnableKorteweg) {
                w_loc[idx(StateVar::RHO_C)] -= eval_kort_pot(ix, iy) * w_eps;
            }
            return w_loc;
        };

        // Step 2: Horizontal 4th-order periodic SBP flux differencing along x
        // Each undirected pair (ix, (ix + m) mod NX) for m in {1, 2} is visited once
        const double inv_dx = 1.0 / geom.dx;
        constexpr std::array<double, 2> kWeightsX = {
            2.0 * SBPOperators2D::kQxWeight1,
            2.0 * SBPOperators2D::kQxWeight2
        };

        for (size_t iy = 0; iy < geom.ny; ++iy) {
            const double phi_y = ThermoConstants::kGravity * geom.y_coord(iy);
            for (size_t ix = 0; ix < geom.nx; ++ix) {
                const size_t id_l = geom.linear_index(ix, iy);
                const core::NodeEntropyArray w_loc_l = make_local_w(ix, iy);
                const double kort_l = eval_kort_pot(ix, iy);

                for (size_t m_idx = 0; m_idx < 2; ++m_idx) {
                    const ptrdiff_t step_m = static_cast<ptrdiff_t>(m_idx + 1);
                    const size_t ix_r = geom.wrap_x(static_cast<ptrdiff_t>(ix) + step_m);
                    const size_t id_r = geom.linear_index(ix_r, iy);
                    const core::NodeEntropyArray w_loc_r = make_local_w(ix_r, iy);
                    const double kort_r = eval_kort_pot(ix_r, iy);

                    const auto pair_res = evaluate_two_point_ec_flux(
                        state_grid[id_l], state_grid[id_r],
                        w_loc_l, w_loc_r,
                        phi_y, phi_y,
                        1.0, 0.0,
                        eos,
                        kort_l, kort_r
                    );

                    const double coeff = 0.5 * kWeightsX[m_idx] * inv_dx;
                    for (size_t k = 0; k < core::kStateVarCount; ++k) {
                        const double df = coeff * pair_res.flux[k];
                        rhs_grid[id_l][k] -= df;
                        rhs_grid[id_r][k] += df;
                    }

                    if constexpr (EnableKorteweg) {
                        // Compatible horizontal Korteweg capillary force matching eos.evaluate_korteweg_momentum_force
                        const double w_eps_bar = 0.5 * (
                            w_grid_out[id_l][idx(StateVar::EPS_TOT)] +
                            w_grid_out[id_r][idx(StateVar::EPS_TOT)]
                        );
                        const double d_lap_dx = (kort_r - kort_l)
                            / eos.phase_mobility().params.kappa_korteweg_j_m5_kg2;
                        const auto f_kort_l = eos.evaluate_korteweg_momentum_force(
                            w_eps_bar * pair_res.kort_coeff_left, -d_lap_dx, 0.0
                        );
                        const auto f_kort_r = eos.evaluate_korteweg_momentum_force(
                            w_eps_bar * pair_res.kort_coeff_right, -d_lap_dx, 0.0
                        );
                        rhs_grid[id_l][idx(StateVar::MOM_UX)] += coeff * f_kort_l[0];
                        rhs_grid[id_r][idx(StateVar::MOM_UX)] += coeff * f_kort_r[0];
                    }
                }
            }
        }

        // Step 3: Vertical 2nd-order SBP flux differencing & compatible gravity along y-columns
        for (size_t ix = 0; ix < geom.nx; ++ix) {
            // 3a. Interior vertical edges (iy, iy + 1) with S_{y, iy, iy+1} = +1/2
            for (size_t iy = 0; iy + 1 < geom.ny; ++iy) {
                const size_t id_l = geom.linear_index(ix, iy);
                const size_t id_r = geom.linear_index(ix, iy + 1);
                const double py_l = geom.norm_weight_y(iy);
                const double py_r = geom.norm_weight_y(iy + 1);

                const double phi_l = ThermoConstants::kGravity * geom.y_coord(iy);
                const double phi_r = ThermoConstants::kGravity * geom.y_coord(iy + 1);

                const core::NodeEntropyArray w_loc_l = make_local_w(ix, iy);
                const core::NodeEntropyArray w_loc_r = make_local_w(ix, iy + 1);
                const double kort_l = eval_kort_pot(ix, iy);
                const double kort_r = eval_kort_pot(ix, iy + 1);

                const auto pair_res = evaluate_two_point_ec_flux(
                    state_grid[id_l], state_grid[id_r],
                    w_loc_l, w_loc_r,
                    phi_l, phi_r,
                    0.0, 1.0,
                    eos,
                    kort_l, kort_r
                );

                // Flux-differencing contribution: 2 * S_{y, iy, iy+1} == 1.0
                for (size_t k = 0; k < core::kStateVarCount; ++k) {
                    rhs_grid[id_l][k] -= pair_res.flux[k] / py_l;
                    rhs_grid[id_r][k] += pair_res.flux[k] / py_r;
                }

                // Compatible gravitational body force on MOM_UY (guarantees w^T P L dE/dx == 0)
                const double w_eps_bar = 0.5 * (
                    w_grid_out[id_l][idx(StateVar::EPS_TOT)] +
                    w_grid_out[id_r][idx(StateVar::EPS_TOT)]
                );
                const double dphi = phi_r - phi_l;
                const double grav_impulse_l = -w_eps_bar * pair_res.grav_coeff_left * dphi;
                const double grav_impulse_r = -w_eps_bar * pair_res.grav_coeff_right * dphi;

                rhs_grid[id_l][idx(StateVar::MOM_UY)] += grav_impulse_l / py_l;
                rhs_grid[id_r][idx(StateVar::MOM_UY)] += grav_impulse_r / py_r;

                if constexpr (EnableKorteweg) {
                    const double d_lap_dy = (kort_r - kort_l)
                        / eos.phase_mobility().params.kappa_korteweg_j_m5_kg2;
                    const auto f_kort_l = eos.evaluate_korteweg_momentum_force(
                        w_eps_bar * pair_res.kort_coeff_left, 0.0, -d_lap_dy
                    );
                    const auto f_kort_r = eos.evaluate_korteweg_momentum_force(
                        w_eps_bar * pair_res.kort_coeff_right, 0.0, -d_lap_dy
                    );
                    rhs_grid[id_l][idx(StateVar::MOM_UY)] += f_kort_l[1] / py_l;
                    rhs_grid[id_r][idx(StateVar::MOM_UY)] += f_kort_r[1] / py_r;
                }
            }

            // 3b. Impermeable wall pressure boundary terms at iy = 0 (B = -1) and iy = NY-1 (B = +1)
            // Cancels the Tadmor boundary potential psi_y = -p_gas * u_y / T identically!
            {
                const size_t id_bot = geom.linear_index(ix, 0);
                const double py_bot = geom.norm_weight_y(0);
                const double temp_bot = 1.0 / w_grid_out[id_bot][idx(StateVar::EPS_TOT)];
                const double p_gas_bot = (
                    state_grid[id_bot][idx(StateVar::RHO_D)] * ThermoConstants::kGasConstDry +
                    state_grid[id_bot][idx(StateVar::RHO_V)] * ThermoConstants::kGasConstVapor
                ) * temp_bot;
                // -B_{0,0} * p_gas_bot / P_{y,0} = +p_gas_bot / P_{y,0}
                rhs_grid[id_bot][idx(StateVar::MOM_UY)] += p_gas_bot / py_bot;
            }
            {
                const size_t id_top = geom.linear_index(ix, geom.ny - 1);
                const double py_top = geom.norm_weight_y(geom.ny - 1);
                const double temp_top = 1.0 / w_grid_out[id_top][idx(StateVar::EPS_TOT)];
                const double p_gas_top = (
                    state_grid[id_top][idx(StateVar::RHO_D)] * ThermoConstants::kGasConstDry +
                    state_grid[id_top][idx(StateVar::RHO_V)] * ThermoConstants::kGasConstVapor
                ) * temp_top;
                // -B_{NY-1, NY-1} * p_gas_top / P_{y, NY-1} = -p_gas_top / P_{y, NY-1}
                rhs_grid[id_top][idx(StateVar::MOM_UY)] -= p_gas_top / py_top;
            }
        }

        return worst_status;
    }
};

} // namespace metriplectic_2d::brackets

#endif // METRIPLECTIC_2D_BRACKETS_REVERSIBLE_L_2D_HPP
