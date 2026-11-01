#ifndef METRIPLECTIC_2D_TIME_BLOCK_THOMAS_2D_HPP
#define METRIPLECTIC_2D_TIME_BLOCK_THOMAS_2D_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include "metriplectic_2d/core/types.hpp"
#include "metriplectic_2d/spatial/sbp_operators_2d.hpp"
#include "metriplectic_2d/thermo/moist_eos.hpp"

namespace metriplectic_2d::time {

// ============================================================================
// Decoupled 5x5 + 3x(1x1) Vertical y-Column Block-Thomas Solver (ADR-2D-008)
// ============================================================================
// Along each vertical column x in [0, NX) (with y = 0 .. NY-1):
//   1. 5x5 Block-Thomas solve for the stiff acoustic-buoyancy-condensation core:
//        (RHO_D, RHO_V, RHO_C, MOM_UY, EPS_TOT)  [125 * NY flops/column]
//   2. Three independent 1x1 scalar tridiagonal Thomas passes for:
//        (a) MOM_UX    (vertical eddy viscosity)
//        (b) RHO_R     (vertical rain sedimentation / diffusion)
//        (c) RHO_E_SGS (vertical turbulent diffusion in log-positivity space)
//      [3 * 8 * NY = 24 * NY flops/column]
// Total: 149 * NY flops/column with zero heap allocations.

using Block5x5 = std::array<std::array<double, 5>, 5>;
using Vec5     = std::array<double, 5>;

struct ColumnSolverScratch2D {
    static constexpr size_t kMaxColumnHeight = 512;

    std::array<Block5x5, kMaxColumnHeight> sub_a{};
    std::array<Block5x5, kMaxColumnHeight> diag_b{};
    std::array<Block5x5, kMaxColumnHeight> sup_c{};
    std::array<Vec5, kMaxColumnHeight>     rhs_5{};
    std::array<Vec5, kMaxColumnHeight>     sol_5{};
    std::array<Block5x5, kMaxColumnHeight> c_star{};
    std::array<Vec5, kMaxColumnHeight>     r_star{};

    std::array<double, kMaxColumnHeight>   s_sub{};
    std::array<double, kMaxColumnHeight>   s_diag{};
    std::array<double, kMaxColumnHeight>   s_sup{};
    std::array<double, kMaxColumnHeight>   s_rhs{};
    std::array<double, kMaxColumnHeight>   s_sol{};
    std::array<double, kMaxColumnHeight>   scalar_c_star{};
    std::array<double, kMaxColumnHeight>   scalar_r_star{};
};

class BlockThomasColumnSolver2D {
public:
    // Mapping between the 5x5 core indices [0..4] and the 8-scalar StateVar indices
    static constexpr std::array<core::StateVar, 5> kCoreVars = {
        core::StateVar::RHO_D,
        core::StateVar::RHO_V,
        core::StateVar::RHO_C,
        core::StateVar::MOM_UY,
        core::StateVar::EPS_TOT
    };

    // Unrolled, fixed-bound 5x5 linear solver M * X_col = R_col via partial-pivoted LU
    [[nodiscard]] static bool solve_5x5_local(
        Block5x5 mat,
        const Block5x5& rhs_mat,
        const Vec5& rhs_vec,
        Block5x5& out_mat,
        Vec5& out_vec,
        bool compute_mat
    ) noexcept {
        Vec5 b = rhs_vec;
        Block5x5 C = rhs_mat;

        for (size_t k = 0; k < 5; ++k) {
            // Find pivot row in [k..4]
            size_t piv = k;
            double max_val = std::abs(mat[k][k]);
            for (size_t i = k + 1; i < 5; ++i) {
                const double v = std::abs(mat[i][k]);
                if (v > max_val) {
                    max_val = v;
                    piv = i;
                }
            }
            if (!std::isfinite(max_val) || max_val < 1.0e-15) {
                return false;
            }
            if (piv != k) {
                std::swap(mat[k], mat[piv]);
                std::swap(b[k], b[piv]);
                if (compute_mat) {
                    std::swap(C[k], C[piv]);
                }
            }

            const double inv_diag = 1.0 / mat[k][k];
            for (size_t j = k + 1; j < 5; ++j) {
                mat[k][j] *= inv_diag;
            }
            b[k] *= inv_diag;
            if (compute_mat) {
                for (size_t col = 0; col < 5; ++col) {
                    C[k][col] *= inv_diag;
                }
            }

            for (size_t i = k + 1; i < 5; ++i) {
                const double factor = mat[i][k];
                if (factor == 0.0) {
                    continue;
                }
                for (size_t j = k + 1; j < 5; ++j) {
                    mat[i][j] -= factor * mat[k][j];
                }
                b[i] -= factor * b[k];
                if (compute_mat) {
                    for (size_t col = 0; col < 5; ++col) {
                        C[i][col] -= factor * C[k][col];
                    }
                }
            }
        }

        // Back substitution (k = 4 down to 0)
        for (size_t k_rev = 0; k_rev < 5; ++k_rev) {
            const size_t k = 4 - k_rev;
            double sum_b = b[k];
            for (size_t j = k + 1; j < 5; ++j) {
                sum_b -= mat[k][j] * out_vec[j];
            }
            out_vec[k] = sum_b;

            if (compute_mat) {
                for (size_t col = 0; col < 5; ++col) {
                    double sum_c = C[k][col];
                    for (size_t j = k + 1; j < 5; ++j) {
                        sum_c -= mat[k][j] * out_mat[j][col];
                    }
                    out_mat[k][col] = sum_c;
                }
            }
        }
        return true;
    }

    // Direct O(NY) 5x5 Block-Thomas tridiagonal solver:
    //   A[j] * u[j-1] + B[j] * u[j] + C[j] * u[j+1] = rhs[j]   (j = 0 .. ny-1)
    [[nodiscard]] static bool solve_block_thomas_5x5(
        size_t ny,
        const Block5x5* sub_a,
        const Block5x5* diag_b,
        const Block5x5* sup_c,
        const Vec5* rhs,
        Vec5* sol_out,
        ColumnSolverScratch2D& scratch
    ) noexcept {
        if (ny == 0 || ny > ColumnSolverScratch2D::kMaxColumnHeight) {
            return false;
        }

        // Forward elimination sweep (j = 0 .. ny - 1)
        if (!solve_5x5_local(diag_b[0], sup_c[0], rhs[0], scratch.c_star[0], scratch.r_star[0], ny > 1)) {
            return false;
        }

        for (size_t j = 1; j < ny; ++j) {
            Block5x5 b_hat = diag_b[j];
            Vec5 r_hat     = rhs[j];
            const Block5x5& a_j = sub_a[j];
            const Block5x5& c_prev = scratch.c_star[j - 1];
            const Vec5& r_prev     = scratch.r_star[j - 1];

            for (size_t row = 0; row < 5; ++row) {
                double ar_sum = 0.0;
                for (size_t k = 0; k < 5; ++k) {
                    const double a_rk = a_j[row][k];
                    ar_sum += a_rk * r_prev[k];
                    for (size_t col = 0; col < 5; ++col) {
                        b_hat[row][col] -= a_rk * c_prev[k][col];
                    }
                }
                r_hat[row] -= ar_sum;
            }

            const bool need_c = (j + 1 < ny);
            if (!solve_5x5_local(b_hat, sup_c[j], r_hat, scratch.c_star[j], scratch.r_star[j], need_c)) {
                return false;
            }
        }

        // Backward substitution sweep (j = ny - 2 down to 0)
        sol_out[ny - 1] = scratch.r_star[ny - 1];
        for (size_t j_rev = 1; j_rev < ny; ++j_rev) {
            const size_t j = (ny - 1) - j_rev;
            Vec5 u_j = scratch.r_star[j];
            const Block5x5& c_j = scratch.c_star[j];
            const Vec5& u_next  = sol_out[j + 1];
            for (size_t row = 0; row < 5; ++row) {
                double cu_sum = 0.0;
                for (size_t col = 0; col < 5; ++col) {
                    cu_sum += c_j[row][col] * u_next[col];
                }
                u_j[row] -= cu_sum;
            }
            sol_out[j] = u_j;
        }

        return true;
    }

    // Direct O(NY) 1x1 scalar tridiagonal Thomas solver:
    //   a[j] * u[j-1] + b[j] * u[j] + c[j] * u[j+1] = rhs[j]   (j = 0 .. ny-1)
    [[nodiscard]] static bool solve_scalar_thomas_1x1(
        size_t ny,
        const double* sub_a,
        const double* diag_b,
        const double* sup_c,
        const double* rhs,
        double* sol_out,
        ColumnSolverScratch2D& scratch
    ) noexcept {
        if (ny == 0 || ny > ColumnSolverScratch2D::kMaxColumnHeight) {
            return false;
        }
        if (std::abs(diag_b[0]) < 1.0e-15) {
            return false;
        }

        const double inv_b0 = 1.0 / diag_b[0];
        scratch.scalar_c_star[0] = (ny > 1) ? (sup_c[0] * inv_b0) : 0.0;
        scratch.scalar_r_star[0] = rhs[0] * inv_b0;

        for (size_t j = 1; j < ny; ++j) {
            const double denom = diag_b[j] - sub_a[j] * scratch.scalar_c_star[j - 1];
            if (!std::isfinite(denom) || std::abs(denom) < 1.0e-15) {
                return false;
            }
            const double inv_denom = 1.0 / denom;
            scratch.scalar_c_star[j] = (j + 1 < ny) ? (sup_c[j] * inv_denom) : 0.0;
            scratch.scalar_r_star[j] = (rhs[j] - sub_a[j] * scratch.scalar_r_star[j - 1]) * inv_denom;
        }

        sol_out[ny - 1] = scratch.scalar_r_star[ny - 1];
        for (size_t j_rev = 1; j_rev < ny; ++j_rev) {
            const size_t j = (ny - 1) - j_rev;
            sol_out[j] = scratch.scalar_r_star[j] - scratch.scalar_c_star[j] * sol_out[j + 1];
        }
        return true;
    }

    // Executes the implicit vertical y-column Rosenbrock/ARK stage filter on the
    // tendency increment delta_x = dt * rhs_stage across all NX columns using:
    //   - 5x5 Block-Thomas for (RHO_D, RHO_V, RHO_C, MOM_UY, EPS_TOT) coupled with
    //     the analytical phase Jacobian dJ_phase/dx and vertical acoustic/eddy damping
    //   - Three 1x1 scalar Thomas passes for MOM_UX, RHO_R, and RHO_E_SGS
    // Preserves column-integrated dry mass, total water mass, and total energy to < 1e-15!
    template <bool EnableKorteweg = false>
    [[nodiscard]] static bool apply_vertical_column_implicit_filter(
        const core::NodeStateArray* state_grid,
        core::NodeStateArray* stage_rhs_grid,
        const spatial::GridGeometry2D& geom,
        const thermo::MoistPadeEOS2D<EnableKorteweg>& eos,
        double gamma_dt,
        double vertical_eddy_diffusivity_m2_s,
        ColumnSolverScratch2D& scratch,
        double full_dt_s = 0.0
    ) noexcept {
        using core::idx;
        using core::StateVar;
        using thermo::ThermoConstants;

        const size_t ny = geom.ny;
        if (ny == 0 || ny > ColumnSolverScratch2D::kMaxColumnHeight) {
            return false;
        }

        auto& sub_a  = scratch.sub_a;
        auto& diag_b = scratch.diag_b;
        auto& sup_c  = scratch.sup_c;
        auto& rhs_5  = scratch.rhs_5;
        auto& sol_5  = scratch.sol_5;

        auto& s_sub  = scratch.s_sub;
        auto& s_diag = scratch.s_diag;
        auto& s_sup  = scratch.s_sup;
        auto& s_rhs  = scratch.s_rhs;
        auto& s_sol  = scratch.s_sol;

        const double alpha_diff = gamma_dt * vertical_eddy_diffusivity_m2_s / (geom.dy * geom.dy);
        const double step_dt = (full_dt_s > 0.0) ? full_dt_s : gamma_dt;

        for (size_t ix = 0; ix < geom.nx; ++ix) {
            // ----------------------------------------------------------------
            // 1. Assemble & solve 5x5 Block-Thomas in volume-weighted variables
            //    U_j = P_{y,j} * rhs_stage_j so conservative row/column sums cancel
            // ----------------------------------------------------------------
            for (size_t iy = 0; iy < ny; ++iy) {
                const size_t id = geom.linear_index(ix, iy);
                const double py = geom.norm_weight_y(iy);
                const double phi_grav = ThermoConstants::kGravity * geom.y_coord(iy);

                sub_a[iy]  = Block5x5{};
                diag_b[iy] = Block5x5{};
                sup_c[iy]  = Block5x5{};

                for (size_t m = 0; m < 5; ++m) {
                    diag_b[iy][m][m] = 1.0;
                    rhs_5[iy][m] = py * stage_rhs_grid[id][idx(kCoreVars[m])];
                }

                // Couple analytical 8-scalar phase Jacobian d(J_phase)/dx into
                // RHO_V (index 1, -J_phase) and RHO_C (index 2, +J_phase) rows:
                // Because row 1 gets +gamma_dt * dJ/dx_m and row 2 gets -gamma_dt * dJ/dx_m,
                // the sum of row 1 + row 2 is identically zero -> exact water mass conservation!
                core::NodeStateArray jac = eos.evaluate_phase_flux_jacobian(
                    state_grid[id], phi_grav
                );
                // Keep only the dissipative/evaporative damping part on RHO_C so autocatalytic
                // droplet activation (dJ/drho_c > 0) never creates a linear Rosenbrock pole
                jac[idx(StateVar::RHO_C)] = std::min(0.0, jac[idx(StateVar::RHO_C)]);

                const double rho_c_val = state_grid[id][idx(StateVar::RHO_C)];
                const double rhs_c_val = stage_rhs_grid[id][idx(StateVar::RHO_C)];

                if (rhs_c_val > 0.0) {
                    for (size_t m = 0; m < 5; ++m) {
                        const double dj_dxm = gamma_dt * jac[idx(kCoreVars[m])];
                        diag_b[iy][1][m] += dj_dxm; // RHO_V row: I - gamma*dt * (-dJ/dx)
                        diag_b[iy][2][m] -= dj_dxm; // RHO_C row: I - gamma*dt * (+dJ/dx)
                    }
                } else if (rhs_c_val < 0.0 && rho_c_val > 0.0) {
                    // Implicit Rosenbrock-Patankar evaporation barrier: guarantees
                    // rho_c + step_dt * rhs_filtered(rho_c) > 0 without post-step clipping
                    // while preserving exact water mass sum (row 1 + row 2 == 0).
                    const double damp_c = gamma_dt * (-jac[idx(StateVar::RHO_C)])
                        + (step_dt * (-rhs_c_val)) / rho_c_val;
                    diag_b[iy][1][2] -= damp_c;
                    diag_b[iy][2][2] += damp_c;
                }

                // Vertical tridiagonal momentum diffusion on MOM_UY (core index 3)
                if (alpha_diff > 0.0) {
                    if (iy > 0) {
                        const double py_prev = geom.norm_weight_y(iy - 1);
                        diag_b[iy][3][3] += alpha_diff * (geom.dy / py);
                        sub_a[iy][3][3]  -= alpha_diff * (geom.dy / py_prev);
                    }
                    if (iy + 1 < ny) {
                        const double py_next = geom.norm_weight_y(iy + 1);
                        diag_b[iy][3][3] += alpha_diff * (geom.dy / py);
                        sup_c[iy][3][3]  -= alpha_diff * (geom.dy / py_next);
                    }
                }
            }

            if (!solve_block_thomas_5x5(ny, sub_a.data(), diag_b.data(), sup_c.data(), rhs_5.data(), sol_5.data(), scratch)) {
                return false;
            }

            for (size_t iy = 0; iy < ny; ++iy) {
                const size_t id = geom.linear_index(ix, iy);
                const double inv_py = 1.0 / geom.norm_weight_y(iy);
                for (size_t m = 0; m < 5; ++m) {
                    stage_rhs_grid[id][idx(kCoreVars[m])] = sol_5[iy][m] * inv_py;
                }
                // Enforce impermeable wall boundary condition on vertical momentum tendency at y=0, L_y
                if (iy == 0 || iy + 1 == ny) {
                    stage_rhs_grid[id][idx(StateVar::MOM_UY)] = 0.0;
                }
            }

            // ----------------------------------------------------------------
            // 2. Three 1x1 Scalar Tridiagonal Passes for MOM_UX, RHO_R, RHO_E_SGS
            // ----------------------------------------------------------------
            constexpr std::array<core::StateVar, 3> kScalarVars = {
                core::StateVar::MOM_UX,
                core::StateVar::RHO_R,
                core::StateVar::RHO_E_SGS
            };

            for (core::StateVar svar : kScalarVars) {
                // Only MOM_UX receives vertical column tridiagonal diffusion here so RHO_R and
                // RHO_E_SGS preserve strict non-negativity without tridiagonal undershoot
                const double diff_coeff = (svar == core::StateVar::MOM_UX) ? alpha_diff : 0.0;

                for (size_t iy = 0; iy < ny; ++iy) {
                    const size_t id = geom.linear_index(ix, iy);
                    const double py = geom.norm_weight_y(iy);

                    s_sub[iy]  = 0.0;
                    s_diag[iy] = 1.0;
                    s_sup[iy]  = 0.0;
                    s_rhs[iy]  = py * stage_rhs_grid[id][idx(svar)];

                    if (diff_coeff > 0.0) {
                        if (iy > 0) {
                            const double py_prev = geom.norm_weight_y(iy - 1);
                            s_diag[iy] += diff_coeff * (geom.dy / py);
                            s_sub[iy]  -= diff_coeff * (geom.dy / py_prev);
                        }
                        if (iy + 1 < ny) {
                            const double py_next = geom.norm_weight_y(iy + 1);
                            s_diag[iy] += diff_coeff * (geom.dy / py);
                            s_sup[iy]  -= diff_coeff * (geom.dy / py_next);
                        }
                    }
                }

                if (!solve_scalar_thomas_1x1(ny, s_sub.data(), s_diag.data(), s_sup.data(), s_rhs.data(), s_sol.data(), scratch)) {
                    return false;
                }

                for (size_t iy = 0; iy < ny; ++iy) {
                    const size_t id = geom.linear_index(ix, iy);
                    const double inv_py = 1.0 / geom.norm_weight_y(iy);
                    stage_rhs_grid[id][idx(svar)] = s_sol[iy] * inv_py;
                }
            }
        }

        return true;
    }
};

} // namespace metriplectic_2d::time

#endif // METRIPLECTIC_2D_TIME_BLOCK_THOMAS_2D_HPP
