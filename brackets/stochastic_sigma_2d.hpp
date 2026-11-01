#ifndef METRIPLECTIC_2D_BRACKETS_STOCHASTIC_SIGMA_2D_HPP
#define METRIPLECTIC_2D_BRACKETS_STOCHASTIC_SIGMA_2D_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include "metriplectic_2d/core/prng.hpp"
#include "metriplectic_2d/core/types.hpp"
#include "metriplectic_2d/spatial/sbp_operators_2d.hpp"

namespace metriplectic_2d::brackets {

// ============================================================================
// 2D Macroscopic SGS Fluctuation-Dissipation Bracket Sigma(x) (ADR-2D-007)
// ============================================================================
// Discretizes the local SBP square-root stochastic forcing operator:
//   Sigma_eff(x) = D^T P^(1/2) Lambda(x)^(1/2) xi_white
// acting on face-centered stateless counter-based normal draws, paired with
// logarithmic Feller boundary scaling on sub-grid kinetic energy rho*e_sgs.
//
// Guarantees:
//   1. Zero net mass, momentum, or total energy drift (conservative face fluxes).
//   2. Strict positivity rho * e_sgs > 0 via logarithmic update and Feller sqrt(e_sgs).

struct StochasticFDTConfig2D {
    bool enable_stochastic      = false;
    uint64_t seed               = 0x2D5F4D4554524950ULL;
    double noise_amplitude_scale = 1.0e-3; // Dimensionless mesoscale FDT scaling factor
    double feller_sgs_noise_amp  = 5.0e-3; // Feller sqrt(rho*e_sgs) amplitude
};

class StochasticBracketSigma2D {
public:
    // Injects conservative face-centered stochastic momentum/energy fluxes into rhs_grid
    template <core::StatelessPRNGPort2D PRNG = core::StatelessCounterPRNG2D>
    static void apply_face_stochastic_fluxes(
        const core::NodeStateArray* state_grid,
        core::NodeStateArray* rhs_grid,
        const spatial::GridGeometry2D& geom,
        const StochasticFDTConfig2D& config,
        uint64_t timestep,
        uint32_t ark_stage,
        double dt_s,
        const PRNG& prng = PRNG{}
    ) noexcept {
        using core::idx;
        using core::StateVar;

        if (!config.enable_stochastic || dt_s <= 0.0) {
            return;
        }

        const double inv_sqrt_dt = 1.0 / std::sqrt(dt_s);

        // 1. Horizontal face-centered stochastic shear stress fluctuations
        for (size_t iy = 0; iy < geom.ny; ++iy) {
            const double py = geom.norm_weight_y(iy);
            const double vol = geom.dx * py;
            for (size_t ix = 0; ix < geom.nx; ++ix) {
                const size_t ix_r = geom.wrap_x(static_cast<ptrdiff_t>(ix) + 1);
                const size_t id_l = geom.linear_index(ix, iy);
                const size_t id_r = geom.linear_index(ix_r, iy);

                const double e_sgs_bar = std::max(
                    0.0,
                    0.5 * (state_grid[id_l][idx(StateVar::RHO_E_SGS)] +
                           state_grid[id_r][idx(StateVar::RHO_E_SGS)])
                );
                // Feller boundary degeneracy: amplitude vanishes smoothly as sqrt(e_sgs) -> 0
                const double sigma_amp = config.noise_amplitude_scale
                    * std::sqrt(2.0 * e_sgs_bar / geom.dx) * inv_sqrt_dt;

                const uint64_t face_id = static_cast<uint64_t>(id_l) * 2ULL;
                const double xi_x = prng.sample_normal(config.seed, timestep, ark_stage, face_id);
                const double xi_y = prng.sample_normal(config.seed, timestep, ark_stage, face_id + 1ULL);

                const double j_stoch_ux = sigma_amp * xi_x;
                const double j_stoch_uy = sigma_amp * xi_y;

                const double scale = py / vol;
                rhs_grid[id_l][idx(StateVar::MOM_UX)] += scale * j_stoch_ux;
                rhs_grid[id_r][idx(StateVar::MOM_UX)] -= scale * j_stoch_ux;
                rhs_grid[id_l][idx(StateVar::MOM_UY)] += scale * j_stoch_uy;
                rhs_grid[id_r][idx(StateVar::MOM_UY)] -= scale * j_stoch_uy;
            }
        }

        // 2. Vertical column face-centered stochastic shear stress fluctuations
        for (size_t ix = 0; ix < geom.nx; ++ix) {
            for (size_t iy = 0; iy + 1 < geom.ny; ++iy) {
                const size_t id_bot = geom.linear_index(ix, iy);
                const size_t id_top = geom.linear_index(ix, iy + 1);
                const double vol_bot = geom.cell_volume_weight(ix, iy);
                const double vol_top = geom.cell_volume_weight(ix, iy + 1);

                const double e_sgs_bar = std::max(
                    0.0,
                    0.5 * (state_grid[id_bot][idx(StateVar::RHO_E_SGS)] +
                           state_grid[id_top][idx(StateVar::RHO_E_SGS)])
                );
                const double sigma_amp = config.noise_amplitude_scale
                    * std::sqrt(2.0 * e_sgs_bar / geom.dy) * inv_sqrt_dt;

                const uint64_t face_id = 0x10000000ULL + static_cast<uint64_t>(id_bot) * 2ULL;
                const double xi_x = prng.sample_normal(config.seed, timestep, ark_stage, face_id);
                const double xi_y = prng.sample_normal(config.seed, timestep, ark_stage, face_id + 1ULL);

                const double j_stoch_ux = sigma_amp * xi_x;
                const double j_stoch_uy = sigma_amp * xi_y;

                rhs_grid[id_bot][idx(StateVar::MOM_UX)] += (geom.dx / vol_bot) * j_stoch_ux;
                rhs_grid[id_top][idx(StateVar::MOM_UX)] -= (geom.dx / vol_top) * j_stoch_ux;
                rhs_grid[id_bot][idx(StateVar::MOM_UY)] += (geom.dx / vol_bot) * j_stoch_uy;
                rhs_grid[id_top][idx(StateVar::MOM_UY)] -= (geom.dx / vol_top) * j_stoch_uy;
            }
        }
    }

    // Evolves sub-grid TKE in logarithmic coordinates psi_sgs = ln(rho * e_sgs)
    // with Feller square-root scaling so rho * e_sgs > 0 is strictly preserved (ADR-2D-007).
    template <core::StatelessPRNGPort2D PRNG = core::StatelessCounterPRNG2D>
    [[nodiscard]] static double step_log_sgs_energy_feller(
        double rho_e_sgs_in,
        double deterministic_rhs_sgs,
        const StochasticFDTConfig2D& config,
        uint64_t timestep,
        uint32_t ark_stage,
        uint64_t node_linear_id,
        double dt_s,
        const PRNG& prng = PRNG{}
    ) noexcept {
        constexpr double kMolecularFloorJPerM3 = 1.0e-12;
        const double e_safe = std::max(rho_e_sgs_in, kMolecularFloorJPerM3);
        const double psi_sgs = std::log(e_safe);

        // Deterministic drift in log-space: d(ln e)/dt = (1/e) * de/dt
        // Use Padé-regularized ratio so stiff decay never underflows
        const double dpsi_det = (deterministic_rhs_sgs * dt_s) / (e_safe + std::abs(deterministic_rhs_sgs) * dt_s);

        double dpsi_stoch = 0.0;
        if (config.enable_stochastic && dt_s > 0.0) {
            const double dw = std::sqrt(dt_s) * prng.sample_normal(
                config.seed ^ 0x5555555555555555ULL, timestep, ark_stage, node_linear_id
            );
            // Feller square-root volatility sigma_feller / sqrt(e_sgs) with Ito drift -0.5 * sigma^2
            const double vol = config.feller_sgs_noise_amp / std::sqrt(e_safe + 1.0);
            dpsi_stoch = vol * dw - 0.5 * vol * vol * dt_s;
        }

        return std::exp(psi_sgs + dpsi_det + dpsi_stoch);
    }
};

} // namespace metriplectic_2d::brackets

#endif // METRIPLECTIC_2D_BRACKETS_STOCHASTIC_SIGMA_2D_HPP
