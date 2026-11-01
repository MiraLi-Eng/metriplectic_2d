#ifndef METRIPLECTIC_2D_SPATIAL_SBP_OPERATORS_2D_HPP
#define METRIPLECTIC_2D_SPATIAL_SBP_OPERATORS_2D_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include "metriplectic_2d/core/types.hpp"

namespace metriplectic_2d::spatial {

// ============================================================================
// 2D Flat Cartesian Anisotropic 4th-Horiz / 2nd-Vert FD-SBP Operators
// (ADR-2D-004, SBP-2D-001)
// ============================================================================
// Coordinate convention:
//   x (index i in [0, NX)): Horizontal coordinate, periodic, 4th-order SBP D_x
//     Interior stencil: (D_x u)_i = (1/dx) * (1/12 u_{i-2} - 2/3 u_{i-1} + 2/3 u_{i+1} - 1/12 u_{i+2})
//     P_x = dx * I, Q_x = -Q_x^T with weights q_1 = 2/3, q_2 = -1/12.
//
//   y (index j in [0, NY)): Vertical atmospheric column, bounded [0, L_y], 2nd-order SBP D_y
//     Norm weights: P_{y,0} = 0.5 * dy, P_{y,j} = dy (1 <= j <= NY-2), P_{y,NY-1} = 0.5 * dy
//     Q_y + Q_y^T = B_y = diag(-1, 0, ..., 0, +1).

struct GridGeometry2D {
    size_t nx = 64;
    size_t ny = 32;
    double dx = 100.0; // Horizontal grid spacing (m)
    double dy = 100.0; // Vertical column spacing (m)

    [[nodiscard]] constexpr size_t total_nodes() const noexcept {
        return nx * ny;
    }

    [[nodiscard]] constexpr size_t linear_index(size_t ix, size_t iy) const noexcept {
        return iy * nx + ix;
    }

    [[nodiscard]] constexpr size_t wrap_x(ptrdiff_t ix) const noexcept {
        const ptrdiff_t n = static_cast<ptrdiff_t>(nx);
        ptrdiff_t r = ix % n;
        if (r < 0) {
            r += n;
        }
        return static_cast<size_t>(r);
    }

    [[nodiscard]] constexpr double y_coord(size_t iy) const noexcept {
        return static_cast<double>(iy) * dy;
    }

    [[nodiscard]] constexpr double x_coord(size_t ix) const noexcept {
        return static_cast<double>(ix) * dx;
    }

    // Diagonal SBP norm weight P_{y, iy} along the vertical column (m)
    [[nodiscard]] constexpr double norm_weight_y(size_t iy) const noexcept {
        if (iy == 0 || iy + 1 == ny) {
            return 0.5 * dy;
        }
        return dy;
    }

    // 2D cell quadrature volume weight P_{x, ix} * P_{y, iy} (m^2)
    [[nodiscard]] constexpr double cell_volume_weight(size_t /*ix*/, size_t iy) const noexcept {
        return dx * norm_weight_y(iy);
    }
};

struct SBPOperators2D {
    static constexpr double kQxWeight1 =  2.0 / 3.0;
    static constexpr double kQxWeight2 = -1.0 / 12.0;

    // 4th-order periodic horizontal derivative (D_x u) at (ix, iy)
    [[nodiscard]] static constexpr double derivative_x_scalar(
        const double* field,
        const GridGeometry2D& geom,
        size_t ix,
        size_t iy
    ) noexcept {
        const ptrdiff_t i = static_cast<ptrdiff_t>(ix);
        const size_t im2 = geom.wrap_x(i - 2);
        const size_t im1 = geom.wrap_x(i - 1);
        const size_t ip1 = geom.wrap_x(i + 1);
        const size_t ip2 = geom.wrap_x(i + 2);

        const double u_im2 = field[geom.linear_index(im2, iy)];
        const double u_im1 = field[geom.linear_index(im1, iy)];
        const double u_ip1 = field[geom.linear_index(ip1, iy)];
        const double u_ip2 = field[geom.linear_index(ip2, iy)];

        return (kQxWeight1 * (u_ip1 - u_im1) + kQxWeight2 * (u_ip2 - u_im2)) / geom.dx;
    }

    // 2nd-order 3-point vertical column derivative (D_y u) at (ix, iy)
    [[nodiscard]] static constexpr double derivative_y_scalar(
        const double* field,
        const GridGeometry2D& geom,
        size_t ix,
        size_t iy
    ) noexcept {
        if (iy == 0) {
            const double u0 = field[geom.linear_index(ix, 0)];
            const double u1 = field[geom.linear_index(ix, 1)];
            return (u1 - u0) / geom.dy;
        }
        if (iy + 1 == geom.ny) {
            const double u_nm2 = field[geom.linear_index(ix, geom.ny - 2)];
            const double u_nm1 = field[geom.linear_index(ix, geom.ny - 1)];
            return (u_nm1 - u_nm2) / geom.dy;
        }
        const double u_jm1 = field[geom.linear_index(ix, iy - 1)];
        const double u_jp1 = field[geom.linear_index(ix, iy + 1)];
        return 0.5 * (u_jp1 - u_jm1) / geom.dy;
    }

    // Symmetric SBP 2D Laplacian (D_xx + D_yy) u for optional Korteweg capillarity (ADR-2D-003)
    // Uses homogeneous Neumann boundary condition at y = 0 and y = L_y so sum_{x,y} P u_lap == 0.
    [[nodiscard]] static constexpr double laplacian_2d_node_var(
        const core::NodeStateArray* state_grid,
        const GridGeometry2D& geom,
        size_t ix,
        size_t iy,
        core::StateVar var
    ) noexcept {
        const size_t v_idx = core::idx(var);
        const ptrdiff_t i = static_cast<ptrdiff_t>(ix);
        const size_t im1 = geom.wrap_x(i - 1);
        const size_t ip1 = geom.wrap_x(i + 1);

        const double u_c   = state_grid[geom.linear_index(ix, iy)][v_idx];
        const double u_xm1 = state_grid[geom.linear_index(im1, iy)][v_idx];
        const double u_xp1 = state_grid[geom.linear_index(ip1, iy)][v_idx];
        const double d2x   = (u_xp1 - 2.0 * u_c + u_xm1) / (geom.dx * geom.dx);

        double d2y = 0.0;
        const double inv_dy2 = 1.0 / (geom.dy * geom.dy);
        if (iy == 0) {
            const double u_yp1 = state_grid[geom.linear_index(ix, 1)][v_idx];
            d2y = 2.0 * (u_yp1 - u_c) * inv_dy2;
        } else if (iy + 1 == geom.ny) {
            const double u_ym1 = state_grid[geom.linear_index(ix, geom.ny - 2)][v_idx];
            d2y = 2.0 * (u_ym1 - u_c) * inv_dy2;
        } else {
            const double u_ym1 = state_grid[geom.linear_index(ix, iy - 1)][v_idx];
            const double u_yp1 = state_grid[geom.linear_index(ix, iy + 1)][v_idx];
            d2y = (u_yp1 - 2.0 * u_c + u_ym1) * inv_dy2;
        }
        return d2x + d2y;
    }

    // Verifies discrete SBP summation-by-parts identities on the 2D mesh:
    //   1. Periodic horizontal skew-symmetry: sum_i P_x u_i (D_x v)_i + v_i (D_x u)_i == 0
    //   2. Vertical column IBP: sum_j P_{y,j} [u_j (D_y v)_j + v_j (D_y u)_j] == u_{NY-1}v_{NY-1} - u_0 v_0
    [[nodiscard]] static double verify_sbp_identities(
        const double* u_field,
        const double* v_field,
        const GridGeometry2D& geom
    ) noexcept {
        double max_ibp_residual = 0.0;

        // Check horizontal periodic skew-symmetry on every row iy
        for (size_t iy = 0; iy < geom.ny; ++iy) {
            double row_sum = 0.0;
            for (size_t ix = 0; ix < geom.nx; ++ix) {
                const double u = u_field[geom.linear_index(ix, iy)];
                const double v = v_field[geom.linear_index(ix, iy)];
                const double dx_u = derivative_x_scalar(u_field, geom, ix, iy);
                const double dx_v = derivative_x_scalar(v_field, geom, ix, iy);
                row_sum += geom.dx * (u * dx_v + v * dx_u);
            }
            max_ibp_residual = std::max(max_ibp_residual, std::abs(row_sum));
        }

        // Check vertical SBP integration-by-parts on every column ix
        for (size_t ix = 0; ix < geom.nx; ++ix) {
            double col_sum = 0.0;
            for (size_t iy = 0; iy < geom.ny; ++iy) {
                const double u = u_field[geom.linear_index(ix, iy)];
                const double v = v_field[geom.linear_index(ix, iy)];
                const double dy_u = derivative_y_scalar(u_field, geom, ix, iy);
                const double dy_v = derivative_y_scalar(v_field, geom, ix, iy);
                col_sum += geom.norm_weight_y(iy) * (u * dy_v + v * dy_u);
            }
            const double bnd_top = u_field[geom.linear_index(ix, geom.ny - 1)]
                * v_field[geom.linear_index(ix, geom.ny - 1)];
            const double bnd_bot = u_field[geom.linear_index(ix, 0)]
                * v_field[geom.linear_index(ix, 0)];
            max_ibp_residual = std::max(max_ibp_residual, std::abs(col_sum - (bnd_top - bnd_bot)));
        }

        return max_ibp_residual;
    }
};

} // namespace metriplectic_2d::spatial

#endif // METRIPLECTIC_2D_SPATIAL_SBP_OPERATORS_2D_HPP
