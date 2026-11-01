#ifndef METRIPLECTIC_2D_CASES_WEATHER_CASE_2D_HPP
#define METRIPLECTIC_2D_CASES_WEATHER_CASE_2D_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include "metriplectic_2d/core/arena.hpp"
#include "metriplectic_2d/core/types.hpp"
#include "metriplectic_2d/spatial/sat_boundaries_2d.hpp"
#include "metriplectic_2d/telemetry/gnn_ring_buffer_2d.hpp"
#include "metriplectic_2d/thermo/hysteretic_mobility.hpp"
#include "metriplectic_2d/thermo/moist_eos.hpp"
#include "metriplectic_2d/time/closure_auditor_2d.hpp"
#include "metriplectic_2d/time/hevi_ark_2d.hpp"

namespace metriplectic_2d::cases {

struct WeatherCaseConfig2D {
    time::SimulationConfig2D sim_config{};
    thermo::KohlerHystereticParams2D hysteresis_params{};

    // Hydrostatic base sounding & moist convective perturbation parameters
    double surface_temp_k            = 289.0;  // Base surface temperature at y = 0 (K)
    double lapse_rate_k_m            = 0.0045; // Vertical temperature lapse rate -dT/dy (K/m)
    double surface_dry_density_kg_m3 = 1.18;   // Base dry air density at y = 0 (kg/m^3)
    double dry_scale_height_m        = 8500.0; // Dry air exponential scale height (m)
    double base_relative_humidity    = 0.82;   // Background relative humidity in [0, 1]
    double bubble_temp_excess_k      = 2.5;    // Warm thermal core temperature perturbation (K)
    double bubble_rh_excess          = 0.26;   // Supersaturated core RH boost (core RH ~ 1.08)
    double bubble_seed_cloud_kg_m3   = 5.0e-4; // Initial cloud liquid water seed in core (kg/m^3)
    double initial_rain_band_kg_m3   = 1.5e-3; // Initial precipitable rain band density (kg/m^3)
    double initial_sgs_tke_j_m3      = 0.8;    // Initial sub-grid turbulent kinetic energy (J/m^3)
};

struct WeatherCaseResult2D {
    time::AuditReport2D audit{};
    spatial::BoundaryFluxLedger2D boundary_ledger{};

    // Summary diagnostics extracted from GNNTelemetryPipeline2D & boundary ledger
    double peak_cloud_water_path_kg_m2   = 0.0;
    double cumulative_precip_fallout_kg  = 0.0;
    double max_supersat_dmu_j_kg         = 0.0;
    double min_phase_stiffness_s1        = 0.0;
    double max_relaxation_time_s         = 0.0;
    double max_welford_variance_dmu      = 0.0;
    double max_lag1_autocorrelation_dmu  = 0.0;
    double mean_surface_pressure_pa      = 0.0;
    double mean_surface_temp_k           = 0.0;
};

class WeatherSimulationCase2D {
public:
    // [UNIT_OP: CASE_2D_FINITE] [RULE_4: 3/60] [CONSERVATION: sigma=0.0, w_ext=BLOWDOWN]
    // RATIONALE: Bounded compile-time IEEE-754 finiteness check avoiding runtime library calls
    // PRE/POST : Pre: val is IEEE-754 float; Post: Returns true iff val is non-NaN and finite
    [[nodiscard]] static constexpr bool is_finite(double val) noexcept {
        return !(val != val) && ((val - val) == 0.0);
    }

    // [UNIT_OP: CASE_2D_CONFIG] [RULE_4: 34/60] [CONSERVATION: sigma=0.0, w_ext=BLOWDOWN]
    // RATIONALE: Construct deterministic default 2D weather simulation configuration with bounded bounds
    // PRE/POST : Pre: nx >= 2, ny >= 2; Post: Valid conservative atmospheric sounding configuration
    [[nodiscard]] static constexpr WeatherCaseConfig2D make_default_config(
        size_t nx = 64,
        size_t ny = 32
    ) noexcept {
        WeatherCaseConfig2D cfg{};
        cfg.sim_config.geom.nx = (nx >= 2) ? nx : 2;
        cfg.sim_config.geom.ny = (ny >= 2) ? ny : 2;
        cfg.sim_config.geom.dx = 100.0;
        cfg.sim_config.geom.dy = 100.0;
        cfg.sim_config.dt_s = 0.02;
        cfg.sim_config.total_steps = 20;
        cfg.sim_config.onsager_params.enable_rain_sedimentation = false;
        cfg.sim_config.onsager_params.autoconversion_thresh = 1.0;
        cfg.sim_config.stochastic_config.enable_stochastic = false;
        cfg.sim_config.stochastic_config.noise_amplitude_scale = 5.0e-4;
        cfg.sim_config.stochastic_config.feller_sgs_noise_amp  = 2.0e-3;
        cfg.sim_config.sat_config.enable_precip_fallout = false;
        cfg.sim_config.sat_config.enable_surface_heat_exchange = false;
        cfg.sim_config.sat_config.surface_temp_k = 292.0;
        cfg.sim_config.sat_config.sensible_conductance = 250.0;
        cfg.sim_config.sat_config.evaporation_mobility = 1.0e-7;

        cfg.hysteresis_params = thermo::KohlerHystereticParams2D{};

        cfg.surface_temp_k            = 289.0;
        cfg.lapse_rate_k_m            = 0.0045;
        cfg.surface_dry_density_kg_m3 = 1.18;
        cfg.dry_scale_height_m        = 8500.0;
        cfg.base_relative_humidity    = 0.82;
        cfg.bubble_temp_excess_k      = 2.5;
        cfg.bubble_rh_excess          = 0.26;
        cfg.bubble_seed_cloud_kg_m3   = 5.0e-4;
        cfg.initial_rain_band_kg_m3   = 1.5e-3;
        cfg.initial_sgs_tke_j_m3      = 0.8;

        return cfg;
    }

    // [UNIT_OP: CASE_2D_ALIGN] [RULE_4: 4/60] [CONSERVATION: sigma=0.0, w_ext=BLOWDOWN]
    // RATIONALE: Align arena byte offset to 64-byte hardware cache line boundary
    // PRE/POST : Pre: kAlign is power of 2; Post: Return value >= val and multiple of 64
    [[nodiscard]] static constexpr size_t align_up_64(size_t val) noexcept {
        constexpr size_t kAlign = 64;
        return (val + kAlign - 1) & ~(kAlign - 1);
    }

    // [UNIT_OP: CASE_2D_BUDGET] [RULE_4: 19/60] [CONSERVATION: sigma=0.0, w_ext=BLOWDOWN]
    // RATIONALE: Compute exact StaticArena2D byte budget with 64-byte alignment padding <= 16 MB
    // PRE/POST : Pre: cfg has valid grid nodes; Post: Return value > 0 and <= 16 MB for tested meshes
    template <size_t NX = 64, size_t NY = 32, size_t RING_DEPTH = 16>
    [[nodiscard]] static constexpr size_t required_arena_bytes(
        const WeatherCaseConfig2D& cfg
    ) noexcept {
        const size_t n_nodes = (cfg.sim_config.geom.total_nodes() > 0)
            ? cfg.sim_config.geom.total_nodes()
            : (NX * NY);

        constexpr size_t kAlign = 64;
        size_t total = 0;

        total = align_up_64(total) + n_nodes * sizeof(core::NodeStateArray);
        total = align_up_64(total) + n_nodes * sizeof(core::NodeStateArray);
        total = align_up_64(total) + n_nodes * sizeof(core::NodeEntropyArray);
        total = align_up_64(total) + n_nodes * sizeof(core::NodeStateArray);
        total = align_up_64(total) + n_nodes * sizeof(core::NodeStateArray);
        total = align_up_64(total) + n_nodes * sizeof(core::CriticalSlowingMetrics);
        total = align_up_64(total) + sizeof(time::ColumnSolverScratch2D);
        total = align_up_64(total) + sizeof(telemetry::GNNTelemetryPipeline2D<NX, NY, RING_DEPTH>);

        total += 16 * kAlign;
        return align_up_64(total);
    }

    // [UNIT_OP: CASE_2D_INIT] [RULE_4: 52/60] [CONSERVATION: sigma=0.0, w_ext=BLOWDOWN]
    // RATIONALE: Initialize physical 8-scalar vertical-slice state with thermal bubble & squall band
    // PRE/POST : Pre: state_grid non-null, nx >= 2, ny >= 2; Post: densities > 0, uy=0, returns true
    [[nodiscard]] static bool initialize_state(
        core::NodeStateArray* state_grid,
        const WeatherCaseConfig2D& cfg
    ) noexcept {
        if (state_grid == nullptr) {
            return false;
        }
        const spatial::GridGeometry2D& geom = cfg.sim_config.geom;
        const size_t nx = geom.nx;
        const size_t ny = geom.ny;
        if (nx < 2 || ny < 2 || geom.dx <= 0.0 || geom.dy <= 0.0) {
            return false;
        }

        const double lx = geom.x_coord(nx);
        const double ly = geom.y_coord(ny - 1);
        const double xc = 0.5 * lx;
        const double yc = 0.35 * ly;
        const double sx = 0.15 * lx;
        const double sy = 0.15 * ly;

        const double rain_active = (cfg.sim_config.onsager_params.enable_rain_sedimentation ||
                                    cfg.sim_config.sat_config.enable_precip_fallout) ? 1.0 : 0.0;

        for (size_t iy = 0; iy < ny; ++iy) {
            const double y = geom.y_coord(iy);
            const double phi_grav = thermo::ThermoConstants::kGravity * y;
            const double temp_bg  = cfg.surface_temp_k - cfg.lapse_rate_k_m * y;
            const double rho_d    = cfg.surface_dry_density_kg_m3 * std::exp(-y / cfg.dry_scale_height_m);

            for (size_t ix = 0; ix < nx; ++ix) {
                const double x = geom.x_coord(ix);
                const size_t id = geom.linear_index(ix, iy);

                const double dx_dist = x - xc;
                const double rx = dx_dist / sx;
                const double ry = (y - yc) / sy;
                const double h_x = std::exp(-rx * rx);
                const double bubble = h_x * std::exp(-ry * ry);

                const double temp_k = temp_bg + cfg.bubble_temp_excess_k * bubble;
                const double p_sat = thermo::PadeThermodynamicTables::saturation_vapor_pressure_pa(temp_k);
                const double rho_v_sat = p_sat / (thermo::ThermoConstants::kGasConstVapor * temp_k);

                const double moist_env = std::max(bubble, h_x);
                const double rh = cfg.base_relative_humidity + cfg.bubble_rh_excess * moist_env;
                const double rho_v = rh * rho_v_sat;
                const double rho_c = cfg.bubble_seed_cloud_kg_m3 * bubble;
                const double rain_band = std::exp(-rx * rx);
                const double rho_r = rain_active * cfg.initial_rain_band_kg_m3 * rain_band * std::exp(-y / 2000.0);

                const double ux = 0.0;
                const double uy = 0.0;

                state_grid[id] = thermo::MoistBulkPadeEOS2D::make_conservative_state(
                    rho_d, rho_v, rho_c, rho_r, ux, uy, cfg.initial_sgs_tke_j_m3, temp_k, phi_grav
                );
            }
        }
        return true;
    }

    // [UNIT_OP: CASE_2D_EXTRACT] [RULE_4: 44/60] [CONSERVATION: sigma=0.0, w_ext=BLOWDOWN]
    // RATIONALE: Extract column moments and Critical Slowing Down diagnostics from latest GNN snapshot
    // PRE/POST : Pre: column_moments non-null, NX > 0; Post: result summary diagnostics populated, returns true
    template <size_t NX = 64>
    [[nodiscard]] static bool extract_snapshot_diagnostics(
        const float column_moments[NX][telemetry::kColumnMomentCount2D],
        WeatherCaseResult2D& result
    ) noexcept {
        if (column_moments == nullptr || NX == 0) {
            return false;
        }

        double peak_cwp  = 0.0;
        double max_dmu   = -1.0e30;
        double min_stiff = 1.0e30;
        double max_tau   = 0.0;
        double max_var   = 0.0;
        double max_acf   = -1.0;
        double sum_p     = 0.0;
        double sum_t     = 0.0;

        for (size_t x = 0; x < NX; ++x) {
            const float* col = column_moments[x];
            const double cwp = col[static_cast<size_t>(telemetry::ColumnMoment2D::CLOUD_WATER_PATH_KG_M2)];
            const double dmu = col[static_cast<size_t>(telemetry::ColumnMoment2D::MAX_SUPERSAT_DMU_J_KG)];
            const double stf = col[static_cast<size_t>(telemetry::ColumnMoment2D::MIN_PHASE_STIFFNESS_S1)];
            const double tau = col[static_cast<size_t>(telemetry::ColumnMoment2D::MAX_RELAXATION_TIME_S)];
            const double var = col[static_cast<size_t>(telemetry::ColumnMoment2D::WELFORD_VARIANCE_DMU)];
            const double acf = col[static_cast<size_t>(telemetry::ColumnMoment2D::LAG1_AUTOCORRELATION_DMU)];
            const double ps  = col[static_cast<size_t>(telemetry::ColumnMoment2D::SURFACE_PRESSURE_PA)];
            const double ts  = col[static_cast<size_t>(telemetry::ColumnMoment2D::SURFACE_TEMP_K)];

            peak_cwp  = std::max(peak_cwp, cwp);
            max_dmu   = std::max(max_dmu, dmu);
            min_stiff = std::min(min_stiff, stf);
            max_tau   = std::max(max_tau, tau);
            max_var   = std::max(max_var, var);
            max_acf   = std::max(max_acf, acf);
            sum_p    += ps;
            sum_t    += ts;
        }

        result.peak_cloud_water_path_kg_m2   = peak_cwp;
        result.max_supersat_dmu_j_kg         = max_dmu;
        result.min_phase_stiffness_s1        = min_stiff;
        result.max_relaxation_time_s         = max_tau;
        result.max_welford_variance_dmu      = max_var;
        result.max_lag1_autocorrelation_dmu  = max_acf;
        result.mean_surface_pressure_pa      = sum_p / static_cast<double>(NX);
        result.mean_surface_temp_k           = sum_t / static_cast<double>(NX);
        return true;
    }

    // [UNIT_OP: CASE_2D_ROLLOUT] [RULE_4: 44/60] [CONSERVATION: sigma=0.0, w_ext=BLOWDOWN]
    // RATIONALE: Allocate arena buffers, lock arena, initialize state, execute HEVI-ARK rollout
    // PRE/POST : Pre: arena unlocked with capacity; Post: arena locked, zero heap allocations, result valid
    template <size_t NX = 64, size_t NY = 32, size_t RING_DEPTH = 16>
    [[nodiscard]] static WeatherCaseResult2D run_weather_case(
        core::StaticArena2D& arena,
        const WeatherCaseConfig2D& cfg
    ) noexcept {
        const size_t n_nodes = cfg.sim_config.geom.total_nodes();
        auto state_span = arena.allocate_aligned<core::NodeStateArray>(n_nodes);

        time::ArenaWorkspace2D ws{};
        const bool ws_ok = ws.allocate_from(arena, n_nodes);

        using TelemetryRing = telemetry::GNNTelemetryPipeline2D<NX, NY, RING_DEPTH>;
        auto telem_span = arena.allocate_aligned<TelemetryRing>(1);

        arena.lock();

        WeatherCaseResult2D result{};
        if (state_span.empty() || !ws_ok || telem_span.empty()) {
            result.audit.integration_status = core::IntegrationStatus::DIVERGED;
            result.audit.closure_status     = core::ClosureStatus::FAILED;
            return result;
        }

        const bool init_ok = initialize_state(state_span.data(), cfg);
        if (!init_ok) {
            result.audit.integration_status = core::IntegrationStatus::DIVERGED;
            result.audit.closure_status     = core::ClosureStatus::FAILED;
            return result;
        }

        const thermo::MoistBulkPadeEOS2D eos{cfg.hysteresis_params};
        result.audit = time::HEVIARKCoordinator2D<false>::run_simulation<NX, NY, RING_DEPTH>(
            state_span.data(),
            ws,
            cfg.sim_config,
            eos,
            result.boundary_ledger,
            &telem_span[0]
        );

        result.cumulative_precip_fallout_kg = std::max(
            0.0, -result.boundary_ledger.cumulative_water_mass_kg
        );

        if (cfg.sim_config.total_steps > 0) {
            const size_t last_snap_idx = (telem_span[0].head_index + RING_DEPTH - 1) % RING_DEPTH;
            (void)extract_snapshot_diagnostics<NX>(
                telem_span[0].ring_buffer[last_snap_idx].column_moments,
                result
            );
        }

        return result;
    }
};

} // namespace metriplectic_2d::cases

#endif // METRIPLECTIC_2D_CASES_WEATHER_CASE_2D_HPP
