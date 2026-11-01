#ifndef METRIPLECTIC_2D_TIME_HEVI_ARK_2D_HPP
#define METRIPLECTIC_2D_TIME_HEVI_ARK_2D_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

#include "metriplectic_2d/brackets/onsager_m_2d.hpp"
#include "metriplectic_2d/brackets/reversible_l_2d.hpp"
#include "metriplectic_2d/brackets/stochastic_sigma_2d.hpp"
#include "metriplectic_2d/core/arena.hpp"
#include "metriplectic_2d/core/types.hpp"
#include "metriplectic_2d/spatial/sat_boundaries_2d.hpp"
#include "metriplectic_2d/spatial/sbp_operators_2d.hpp"
#include "metriplectic_2d/telemetry/gnn_ring_buffer_2d.hpp"
#include "metriplectic_2d/thermo/moist_eos.hpp"
#include "metriplectic_2d/time/block_thomas_2d.hpp"
#include "metriplectic_2d/time/closure_auditor_2d.hpp"

namespace metriplectic_2d::time {

// ============================================================================
// 2D Horizontally Explicit / Vertically Implicit IMEX-ARK Coordinator
// (ADR-2D-008, HEVI-2D-001)
// ============================================================================
// Executes a 2-stage Heun/ARS-style IMEX-ARK rollout on the 2D (x, y) slice:
//   - Pre-allocates all working buffers inside StaticArena2D prior to lock().
//   - Zero dynamic heap allocations occur during the step loop.
//   - Couples L(x) dE/dx, M(x) dS/dx, Sigma(x) dW, open SAT boundaries, and the
//     5x5 + 3x(1x1) vertical y-column Block-Thomas solver.

struct SimulationConfig2D {
    spatial::GridGeometry2D geom{};
    double dt_s                              = 0.05;
    uint64_t total_steps                     = 10;
    double ark_gamma                         = 0.29289321881345248; // 1 - 1/sqrt(2) (L-stable SDIRK)
    double vertical_eddy_diffusivity_m2_s    = 0.05;
    brackets::OnsagerTransportParams2D onsager_params{};
    brackets::StochasticFDTConfig2D stochastic_config{};
    spatial::SurfaceSATConfig2D sat_config{};
};

struct ArenaWorkspace2D {
    std::span<core::NodeStateArray>         x_stage{};
    std::span<core::NodeEntropyArray>       w_grid{};
    std::span<core::NodeStateArray>         rhs_stage1{};
    std::span<core::NodeStateArray>         rhs_stage2{};
    std::span<core::CriticalSlowingMetrics> crit_metrics{};
    std::span<ColumnSolverScratch2D>        col_scratch{};

    [[nodiscard]] bool allocate_from(core::StaticArena2D& arena, size_t total_nodes) noexcept {
        x_stage      = arena.allocate_aligned<core::NodeStateArray>(total_nodes);
        w_grid       = arena.allocate_aligned<core::NodeEntropyArray>(total_nodes);
        rhs_stage1   = arena.allocate_aligned<core::NodeStateArray>(total_nodes);
        rhs_stage2   = arena.allocate_aligned<core::NodeStateArray>(total_nodes);
        crit_metrics = arena.allocate_aligned<core::CriticalSlowingMetrics>(total_nodes);
        col_scratch  = arena.allocate_aligned<ColumnSolverScratch2D>(1);

        return !x_stage.empty() && !w_grid.empty() && !rhs_stage1.empty() &&
               !rhs_stage2.empty() && !crit_metrics.empty() && !col_scratch.empty();
    }
};

template <bool EnableKorteweg = false>
class HEVIARKCoordinator2D {
public:
    // Evaluates a single combined stage tendency rhs_out = L + M + Sigma + SAT
    // followed by the vertical y-column 5x5 + 3x(1x1) Block-Thomas implicit filter.
    [[nodiscard]] static bool evaluate_hevi_stage(
        const core::NodeStateArray* state_in,
        core::NodeEntropyArray* w_buf,
        core::NodeStateArray* rhs_out,
        core::CriticalSlowingMetrics* crit_out,
        const SimulationConfig2D& cfg,
        const thermo::MoistPadeEOS2D<EnableKorteweg>& eos,
        uint64_t step_idx,
        uint32_t stage_idx,
        spatial::BoundaryFluxLedger2D& stage_ledger_out,
        double& stage_entropy_prod_out,
        ColumnSolverScratch2D& scratch
    ) noexcept {
        const size_t n_nodes = cfg.geom.total_nodes();
        for (size_t i = 0; i < n_nodes; ++i) {
            rhs_out[i] = core::NodeStateArray{};
        }
        stage_ledger_out.reset_instantaneous();

        // 1. Reversible Lie-Poisson bracket L(x) dE/dx (also populates w_buf = dS/dx)
        const auto eos_status = brackets::ReversibleBracketL2D<EnableKorteweg>::apply(
            state_in, w_buf, rhs_out, cfg.geom, eos
        );
        if (eos_status == core::EOSValidityStatus::INVALID) {
            return false;
        }

        // 2. PSD Dyadic Onsager bracket M(x) dS/dx
        const double s_prod_interior = brackets::OnsagerBracketM2D<EnableKorteweg>::apply(
            state_in, w_buf, rhs_out, crit_out, cfg.geom, eos, cfg.onsager_params
        );

        // 3. Macroscopic SGS-FDT stochastic bracket Sigma(x) dW
        brackets::StochasticBracketSigma2D::apply_face_stochastic_fluxes(
            state_in, rhs_out, cfg.geom, cfg.stochastic_config, step_idx, stage_idx, cfg.dt_s
        );

        // 4. Open SBP-SAT boundary penalties at y = 0
        spatial::SATBoundaryOperators2D::apply_surface_sat_penalties(
            state_in, rhs_out, cfg.geom, eos, cfg.sat_config, stage_ledger_out
        );

        stage_entropy_prod_out = s_prod_interior + stage_ledger_out.boundary_entropy_prod_w_k;

        // 5. Decoupled 5x5 + 3x(1x1) vertical y-column Block-Thomas implicit filter
        const double gamma_dt = cfg.ark_gamma * cfg.dt_s;
        if (gamma_dt > 0.0) {
            if (!BlockThomasColumnSolver2D::apply_vertical_column_implicit_filter(
                    state_in,
                    rhs_out,
                    cfg.geom,
                    eos,
                    gamma_dt,
                    cfg.vertical_eddy_diffusivity_m2_s,
                    scratch,
                    cfg.dt_s
                )) {
                return false;
            }
        }

        return true;
    }

    // Applies a stage state update x_out = x_base + dt_weight * rhs with
    // logarithmic Feller positivity on RHO_E_SGS and zero post-step clipping
    // (positivity of RHO_C and RHO_R is guaranteed analytically by the one-sided
    // mobility barrier and implicit 5x5 Block-Thomas Rosenbrock-Patankar damping).
    static void advance_state_positively(
        const core::NodeStateArray* x_base,
        const core::NodeStateArray* rhs,
        core::NodeStateArray* x_out,
        const SimulationConfig2D& cfg,
        uint64_t step_idx,
        uint32_t stage_idx,
        double dt_weight
    ) noexcept {
        using core::idx;
        using core::StateVar;

        const size_t n_nodes = cfg.geom.total_nodes();
        for (size_t id = 0; id < n_nodes; ++id) {
            core::NodeStateArray next = x_base[id];
            const core::NodeStateArray& k_rhs = rhs[id];

            next[idx(StateVar::RHO_D)]   += dt_weight * k_rhs[idx(StateVar::RHO_D)];
            next[idx(StateVar::RHO_V)]   += dt_weight * k_rhs[idx(StateVar::RHO_V)];
            next[idx(StateVar::RHO_C)]   += dt_weight * k_rhs[idx(StateVar::RHO_C)];
            next[idx(StateVar::RHO_R)]   += dt_weight * k_rhs[idx(StateVar::RHO_R)];
            next[idx(StateVar::MOM_UX)]  += dt_weight * k_rhs[idx(StateVar::MOM_UX)];
            next[idx(StateVar::MOM_UY)]  += dt_weight * k_rhs[idx(StateVar::MOM_UY)];
            next[idx(StateVar::EPS_TOT)] += dt_weight * k_rhs[idx(StateVar::EPS_TOT)];

            // Evolve RHO_E_SGS in logarithmic coordinates psi_sgs = ln(rho*e_sgs) with Feller scaling
            // Note: because EPS_TOT is updated exclusively by conservative flux divergences,
            // total energy is invariant while internal energy absorbs the TKE exchange automatically!
            next[idx(StateVar::RHO_E_SGS)] = brackets::StochasticBracketSigma2D::step_log_sgs_energy_feller(
                x_base[id][idx(StateVar::RHO_E_SGS)],
                k_rhs[idx(StateVar::RHO_E_SGS)],
                cfg.stochastic_config,
                step_idx,
                stage_idx,
                static_cast<uint64_t>(id),
                dt_weight
            );

            x_out[id] = next;
        }
    }

    // Executes the complete multi-step 2D HEVI-ARK rollout and returns the AuditReport2D.
    // Optionally captures each step into a GNNTelemetryPipeline2D ring buffer.
    template <size_t TELEM_NX = 64, size_t TELEM_NY = 32, size_t TELEM_DEPTH = 16>
    [[nodiscard]] static AuditReport2D run_simulation(
        core::NodeStateArray* state_grid,
        ArenaWorkspace2D& ws,
        const SimulationConfig2D& cfg,
        const thermo::MoistPadeEOS2D<EnableKorteweg>& eos,
        spatial::BoundaryFluxLedger2D& total_boundary_ledger,
        telemetry::GNNTelemetryPipeline2D<TELEM_NX, TELEM_NY, TELEM_DEPTH>* telemetry_ring = nullptr
    ) noexcept {
        core::PhysicalValidityStatus init_phys{};
        core::EOSValidityStatus init_eos{};
        const ConservationIntegrals2D init_integrals = ClosureAuditor2D::compute_domain_integrals(
            state_grid, cfg.geom, eos, init_phys, init_eos
        );

        total_boundary_ledger = spatial::BoundaryFluxLedger2D{};
        double min_entropy_prod_rate = 1.0e30;

        for (uint64_t step = 0; step < cfg.total_steps; ++step) {
            spatial::BoundaryFluxLedger2D ledger_s1{};
            spatial::BoundaryFluxLedger2D ledger_s2{};
            double s_prod_1 = 0.0;
            double s_prod_2 = 0.0;

            // Stage 1: Evaluate tendency k1 at state^n
            if (!evaluate_hevi_stage(
                    state_grid,
                    ws.w_grid.data(),
                    ws.rhs_stage1.data(),
                    ws.crit_metrics.data(),
                    cfg,
                    eos,
                    step,
                    0U,
                    ledger_s1,
                    s_prod_1,
                    ws.col_scratch[0]
                )) {
                AuditReport2D fail_rep{};
                fail_rep.integration_status = core::IntegrationStatus::DIVERGED;
                fail_rep.closure_status     = core::ClosureStatus::FAILED;
                fail_rep.physical_validity  = core::PhysicalValidityStatus::INVALID;
                fail_rep.eos_validity       = core::EOSValidityStatus::INVALID;
                return fail_rep;
            }

            // Predictor state x^{(1)} = x^n + dt * k1
            advance_state_positively(
                state_grid, ws.rhs_stage1.data(), ws.x_stage.data(), cfg, step, 0U, cfg.dt_s
            );

            // Stage 2: Evaluate tendency k2 at x^{(1)}
            if (!evaluate_hevi_stage(
                    ws.x_stage.data(),
                    ws.w_grid.data(),
                    ws.rhs_stage2.data(),
                    ws.crit_metrics.data(),
                    cfg,
                    eos,
                    step,
                    1U,
                    ledger_s2,
                    s_prod_2,
                    ws.col_scratch[0]
                )) {
                AuditReport2D fail_rep{};
                fail_rep.integration_status = core::IntegrationStatus::DIVERGED;
                fail_rep.closure_status     = core::ClosureStatus::FAILED;
                fail_rep.physical_validity  = core::PhysicalValidityStatus::INVALID;
                fail_rep.eos_validity       = core::EOSValidityStatus::INVALID;
                return fail_rep;
            }

            // Combine stages: k_avg = 0.5 * (k1 + k2)
            const size_t n_nodes = cfg.geom.total_nodes();
            for (size_t i = 0; i < n_nodes; ++i) {
                for (size_t c = 0; c < core::kStateVarCount; ++c) {
                    ws.rhs_stage1[i][c] = 0.5 * (ws.rhs_stage1[i][c] + ws.rhs_stage2[i][c]);
                }
            }

            advance_state_positively(
                state_grid, ws.rhs_stage1.data(), state_grid, cfg, step, 1U, cfg.dt_s
            );

            // Accumulate exact boundary flux ledger for 2nd-order trapezoidal stage weights (0.5, 0.5)
            total_boundary_ledger.net_dry_air_mass_flux_kg_s = 0.5 * (
                ledger_s1.net_dry_air_mass_flux_kg_s + ledger_s2.net_dry_air_mass_flux_kg_s
            );
            total_boundary_ledger.net_water_mass_flux_kg_s = 0.5 * (
                ledger_s1.net_water_mass_flux_kg_s + ledger_s2.net_water_mass_flux_kg_s
            );
            total_boundary_ledger.net_total_energy_flux_w = 0.5 * (
                ledger_s1.net_total_energy_flux_w + ledger_s2.net_total_energy_flux_w
            );
            total_boundary_ledger.accumulate_step(cfg.dt_s);

            const double step_s_prod = 0.5 * (s_prod_1 + s_prod_2);
            min_entropy_prod_rate = std::min(min_entropy_prod_rate, step_s_prod);

            // Optional zero-allocation telemetry capture
            if (telemetry_ring != nullptr &&
                cfg.geom.nx == TELEM_NX && cfg.geom.ny == TELEM_NY) {
                telemetry_ring->capture_step(
                    step + 1,
                    static_cast<double>(step + 1) * cfg.dt_s,
                    state_grid,
                    ws.crit_metrics.data(),
                    cfg.geom.dy
                );
            }
        }

        if (cfg.total_steps == 0) {
            min_entropy_prod_rate = 0.0;
        }

        core::PhysicalValidityStatus final_phys{};
        core::EOSValidityStatus final_eos{};
        const ConservationIntegrals2D final_integrals = ClosureAuditor2D::compute_domain_integrals(
            state_grid, cfg.geom, eos, final_phys, final_eos
        );

        return ClosureAuditor2D::evaluate_closure(
            init_integrals,
            final_integrals,
            total_boundary_ledger,
            min_entropy_prod_rate,
            final_phys,
            final_eos
        );
    }
};

} // namespace metriplectic_2d::time

#endif // METRIPLECTIC_2D_TIME_HEVI_ARK_2D_HPP
