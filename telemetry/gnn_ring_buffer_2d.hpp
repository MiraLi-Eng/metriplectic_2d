#ifndef METRIPLECTIC_2D_TELEMETRY_GNN_RING_BUFFER_2D_HPP
#define METRIPLECTIC_2D_TELEMETRY_GNN_RING_BUFFER_2D_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include "metriplectic_2d/core/types.hpp"
#include "metriplectic_2d/thermo/hysteretic_mobility.hpp"
#include "metriplectic_2d/thermo/pade_tables.hpp"

namespace metriplectic_2d::telemetry {

// ============================================================================
// 1. 1D Horizontal x-Column Summary & Critical Slowing Down Moments (ADR-2D-009)
// ============================================================================

enum class ColumnMoment2D : size_t {
    PRECIPITABLE_WATER_KG_M2 = 0, // Vertical integral of rho_v dy
    CLOUD_WATER_PATH_KG_M2   = 1, // Vertical integral of rho_c dy
    RAIN_WATER_PATH_KG_M2    = 2, // Vertical integral of rho_r dy
    SURFACE_PRESSURE_PA      = 3, // p(x, y=0)
    SURFACE_TEMP_K           = 4, // T(x, y=0)
    COLUMN_TKE_J_M2          = 5, // Vertical integral of (0.5*rho*|v|^2 + rho*e_sgs) dy
    COLUMN_ENTROPY_PROD      = 6, // Vertical integral of sigma_phase dy >= 0
    MAX_SUPERSAT_DMU_J_KG    = 7, // Column max(mu_v - mu_c^eff)
    MIN_PHASE_STIFFNESS_S1   = 8, // Column min(|dJ_phase/drho_c|) -> 0 at criticality
    MAX_RELAXATION_TIME_S    = 9, // Column max(tau_relax) -> inf at criticality
    WELFORD_VARIANCE_DMU     = 10,// Online Welford variance of column supersaturation
    LAG1_AUTOCORRELATION_DMU = 11,// Online lag-1 autocorrelation ACF_1 in [-1, 1]
    COUNT                    = 12
};

inline constexpr size_t kColumnMomentCount2D = static_cast<size_t>(ColumnMoment2D::COUNT);

// ============================================================================
// 2. Online Welford Variance & Lag-1 Autocorrelation Tracker (Zero Allocation)
// ============================================================================
// Detects Critical Slowing Down (tau_relax -> inf => variance spike & ACF_1 -> 1)
// prior to rapid phase transitions in each vertical y-column along x.

struct OnlineCriticalityTracker {
    uint64_t sample_count = 0;
    double mean           = 0.0;
    double m2             = 0.0; // Sum of squared deviations (X_t - mean)^2
    double c1             = 0.0; // Lag-1 cross-deviation sum (X_t - mean)*(X_{t-1} - mean)
    double prev_dev       = 0.0;

    constexpr void update(double x_val) noexcept {
        ++sample_count;
        const double delta = x_val - mean;
        mean += delta / static_cast<double>(sample_count);
        const double delta2 = x_val - mean;
        m2 += delta * delta2;
        if (sample_count > 1) {
            c1 += prev_dev * delta2;
        }
        prev_dev = delta2;
    }

    [[nodiscard]] constexpr double variance() const noexcept {
        if (sample_count < 2) {
            return 0.0;
        }
        return m2 / static_cast<double>(sample_count - 1);
    }

    [[nodiscard]] constexpr double lag1_autocorrelation() const noexcept {
        if (sample_count < 3 || m2 <= 1.0e-30) {
            return 0.0;
        }
        const double acf = c1 / m2;
        return std::clamp(acf, -1.0, 1.0);
    }
};

// ============================================================================
// 3. Static Zero-Allocation 2D Telemetry Ring Buffer (GNNTelemetryPipeline2D)
// ============================================================================
// For default NX = 256, NY = 128, RING_DEPTH = 32:
//   - 2x2 downsampled cells: 64 * 128 * 8 * 2 B = 131,072 B (128 KB / step)
//   - 1D column moments:     256 * 12 * 4 B     =  12,288 B ( 12 KB / step)
//   - Total 32-step ring buffer:                  ~4.58 MB total static footprint!

template <size_t NX = 256, size_t NY = 128, size_t RING_DEPTH = 32>
struct GNNTelemetryPipeline2D {
    static_assert(NX % 2 == 0 && NY % 2 == 0, "NX and NY must be divisible by 2");

    static constexpr size_t CX = NX / 2;
    static constexpr size_t CY = NY / 2;
    static constexpr size_t CHANNELS = core::kStateVarCount; // 8 channels

    struct CompressedSnapshot2D {
        uint64_t step_index = 0;
        double simulation_time_s = 0.0;
        alignas(64) core::bfloat16_t cells[CY][CX][CHANNELS]{};
        alignas(64) float column_moments[NX][kColumnMomentCount2D]{};
    };

    alignas(64) CompressedSnapshot2D ring_buffer[RING_DEPTH]{};
    alignas(64) OnlineCriticalityTracker column_trackers[NX]{};
    size_t head_index = 0;

    // Captures a 2x2 block-averaged bfloat16_t snapshot from a flat SOA/AOS 2D grid
    // node_states[y * NX + x] and updates column Critical Slowing Down statistics.
    void capture_step(
        uint64_t step,
        double time_s,
        const core::NodeStateArray* grid_nodes,
        const core::CriticalSlowingMetrics* grid_crit_metrics,
        double dy_m
    ) noexcept {
        CompressedSnapshot2D& snap = ring_buffer[head_index];
        snap.step_index = step;
        snap.simulation_time_s = time_s;

        // 1. In-register 2x2 block average into bfloat16_t
        for (size_t cy = 0; cy < CY; ++cy) {
            const size_t y0 = 2 * cy;
            const size_t y1 = y0 + 1;
            for (size_t cx = 0; cx < CX; ++cx) {
                const size_t x0 = 2 * cx;
                const size_t x1 = x0 + 1;

                const core::NodeStateArray& n00 = grid_nodes[y0 * NX + x0];
                const core::NodeStateArray& n01 = grid_nodes[y0 * NX + x1];
                const core::NodeStateArray& n10 = grid_nodes[y1 * NX + x0];
                const core::NodeStateArray& n11 = grid_nodes[y1 * NX + x1];

                for (size_t ch = 0; ch < CHANNELS; ++ch) {
                    const float avg = static_cast<float>(
                        0.25 * (n00[ch] + n01[ch] + n10[ch] + n11[ch])
                    );
                    snap.cells[cy][cx][ch] = core::bfloat16_t::from_float(avg);
                }
            }
        }

        // 2. Integrate y-column summary moments & Critical Slowing Down statistics for each x
        for (size_t x = 0; x < NX; ++x) {
            double pw_kg_m2   = 0.0;
            double cwp_kg_m2  = 0.0;
            double rwp_kg_m2  = 0.0;
            double tke_j_m2   = 0.0;
            double s_prod_int = 0.0;
            double max_dmu    = -1.0e30;
            double min_stiff  = 1.0e30;
            double max_tau    = 0.0;
            double surf_p_pa  = 0.0;
            double surf_t_k   = 0.0;

            for (size_t y = 0; y < NY; ++y) {
                const size_t idx_2d = y * NX + x;
                const core::NodeStateArray& node = grid_nodes[idx_2d];
                const core::CriticalSlowingMetrics& cm = grid_crit_metrics[idx_2d];

                const double rho_d = node[core::idx(core::StateVar::RHO_D)];
                const double rho_v = node[core::idx(core::StateVar::RHO_V)];
                const double rho_c = node[core::idx(core::StateVar::RHO_C)];
                const double rho_r = node[core::idx(core::StateVar::RHO_R)];
                const double mux   = node[core::idx(core::StateVar::MOM_UX)];
                const double muy   = node[core::idx(core::StateVar::MOM_UY)];
                const double esgs  = node[core::idx(core::StateVar::RHO_E_SGS)];
                const double eps   = node[core::idx(core::StateVar::EPS_TOT)];
                const double rho   = rho_d + rho_v + rho_c + rho_r;
                const double e_kin = 0.5 * (mux * mux + muy * muy) / rho;

                if (y == 0) {
                    const double eps_int0 = eps - e_kin - esgs;
                    surf_t_k  = thermo::solve_temperature_newton_halley(rho_d, rho_v, rho_c, rho_r, eps_int0);
                    surf_p_pa = (rho_d * thermo::ThermoConstants::kGasConstDry
                        + rho_v * thermo::ThermoConstants::kGasConstVapor) * surf_t_k;
                }

                // Use SBP vertical quadrature weight P_{y,y} (0.5*dy at boundaries, dy in interior)
                const double wy = (y == 0 || y + 1 == NY) ? (0.5 * dy_m) : dy_m;

                pw_kg_m2   += rho_v * wy;
                cwp_kg_m2  += rho_c * wy;
                rwp_kg_m2  += rho_r * wy;
                tke_j_m2   += (e_kin + esgs) * wy;
                s_prod_int += cm.entropy_prod_w_m3_k * wy;

                max_dmu   = std::max(max_dmu, cm.delta_mu_j_kg);
                min_stiff = std::min(min_stiff, cm.effective_stiffness_s1);
                max_tau   = std::max(max_tau, cm.relaxation_time_s);
            }

            column_trackers[x].update(max_dmu);

            float* col = snap.column_moments[x];
            col[static_cast<size_t>(ColumnMoment2D::PRECIPITABLE_WATER_KG_M2)] = static_cast<float>(pw_kg_m2);
            col[static_cast<size_t>(ColumnMoment2D::CLOUD_WATER_PATH_KG_M2)]   = static_cast<float>(cwp_kg_m2);
            col[static_cast<size_t>(ColumnMoment2D::RAIN_WATER_PATH_KG_M2)]    = static_cast<float>(rwp_kg_m2);
            col[static_cast<size_t>(ColumnMoment2D::SURFACE_PRESSURE_PA)]      = static_cast<float>(surf_p_pa);
            col[static_cast<size_t>(ColumnMoment2D::SURFACE_TEMP_K)]           = static_cast<float>(surf_t_k);
            col[static_cast<size_t>(ColumnMoment2D::COLUMN_TKE_J_M2)]          = static_cast<float>(tke_j_m2);
            col[static_cast<size_t>(ColumnMoment2D::COLUMN_ENTROPY_PROD)]      = static_cast<float>(s_prod_int);
            col[static_cast<size_t>(ColumnMoment2D::MAX_SUPERSAT_DMU_J_KG)]    = static_cast<float>(max_dmu);
            col[static_cast<size_t>(ColumnMoment2D::MIN_PHASE_STIFFNESS_S1)]   = static_cast<float>(min_stiff);
            col[static_cast<size_t>(ColumnMoment2D::MAX_RELAXATION_TIME_S)]    = static_cast<float>(max_tau);
            col[static_cast<size_t>(ColumnMoment2D::WELFORD_VARIANCE_DMU)]     = static_cast<float>(column_trackers[x].variance());
            col[static_cast<size_t>(ColumnMoment2D::LAG1_AUTOCORRELATION_DMU)] = static_cast<float>(column_trackers[x].lag1_autocorrelation());
        }

        head_index = (head_index + 1) % RING_DEPTH;
    }
};

} // namespace metriplectic_2d::telemetry

#endif // METRIPLECTIC_2D_TELEMETRY_GNN_RING_BUFFER_2D_HPP
