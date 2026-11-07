#ifndef METRIPLECTIC_2D_CASES_SOLAR_GROUND_FORCING_2D_HPP
#define METRIPLECTIC_2D_CASES_SOLAR_GROUND_FORCING_2D_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include "metriplectic_2d/thermo/surface_fluxes_2d.hpp"

namespace metriplectic_2d::cases {

// ============================================================================
// Deterministic Daylight / Solar-Flux & Per-Column Heated-Ground Forcing
// (GROUND-2D, SAT-2D) — plane-parallel solar geometry + per-column ground ODE.
// ============================================================================
// The ground skin temperature T_g[ix] is a separate per-column reservoir. The
// shortwave absorbed flux Q_sw is a PRESCRIBED EXTERNAL source (w_ext): it heats
// the ground directly and never enters the air-domain conserved sum. The
// ground<->air exchange (sensible + latent) is computed by the SHARED
// compute_surface_fluxes_2d kernel, so the air-side SAT gain equals the ground
// ODE loss exactly; the net longwave (ground loses to sky) is an external sink.

struct SolarGroundConfig2D {
    double latitude_deg                 = 35.0;   // Geographic latitude (deg N)
    double day_of_year                  = 172.0;  // Day of year (172 ~ summer solstice)
    double start_local_hour             = 6.0;    // Local solar time at t = 0 (hours, 06:00)
    double solar_constant_w_m2          = 1361.0; // Total solar irradiance S0 (W/m^2)
    double solar_flux_multiplier        = 1.0;    // Interactive user scale factor on Q_sw (>= 0)
    double clear_sky_transmittance      = 0.75;   // Atmospheric shortwave transmittance
    double albedo                       = 0.18;   // Base land surface shortwave albedo
    double water_albedo                 = 0.06;   // Open water body shortwave albedo
    double mountain_albedo              = 0.36;   // High-altitude rock/snow ridge albedo
    double cloud_attenuation_per_kg_m2  = 0.08;   // k_cloud optical depth per column water path (m^2/kg)
    double ground_heat_capacity_j_m2_k  = 2.0e5;  // C_g land thermal inertia (J/(m^2 K))
    double water_heat_capacity_j_m2_k   = 1.0e6;  // C_w water body thermal inertia (J/(m^2 K))
    double water_evap_multiplier        = 4.0;    // Evaporation mobility boost over water bodies
    double emissivity                   = 0.95;   // Longwave surface emissivity
    bool   enable_longwave              = true;   // Enable ground<->sky net longwave cooling
    double min_ground_temp_k            = 220.0;  // Hard clamp lower bound (K)
    double max_ground_temp_k            = 340.0;  // Hard clamp upper bound (K)
    double solar_time_scale             = 1.0;    // >1 accelerates the demo solar clock
};

// Per-column landform elevation and water-body surface descriptor.
struct SurfaceColumnProps2D {
    double terrain_elevation_m = 0.0; // Landform height above base y=0 [0, 2000] (m)
    double water_fraction      = 0.0; // Water coverage in [0, 1] (0 = land/rock, 1 = water body)
    double slope_dx            = 0.0; // Horizontal terrain slope dh/dx (dimensionless)
};

// Cached solar geometry at a given simulation time (radians).
struct SolarGeometry2D {
    double hour_angle_rad  = 0.0;
    double declination_rad = 0.0;
    double sin_elevation   = 0.0;
};

inline constexpr double kSolarPi      = 3.14159265358979323846;
inline constexpr double kSolarDeg2Rad = kSolarPi / 180.0;
inline constexpr double kObliquityDeg = 23.44;    // Earth axial tilt (deg)
inline constexpr double kSecondsPerHour = 3600.0;

// [UNIT_OP: GUI_BUILD_2D] [RULE_4: 6/60] [CONSERVATION: sigma=0, w_ext=EXTERNAL_QSW]
// RATIONALE: Standard solar declination (Cooper approximation) in radians
// PRE/POST : Pre: day_of_year finite; Post: declination in [-23.44deg, +23.44deg]
[[nodiscard]] inline double solar_declination_rad(double day_of_year) noexcept {
    return kObliquityDeg * kSolarDeg2Rad
        * std::sin(2.0 * kSolarPi * (day_of_year - 81.0) / 365.0);
}

// [UNIT_OP: GUI_BUILD_2D] [RULE_4: 4/60] [CONSERVATION: sigma=0, w_ext=EXTERNAL_QSW]
// RATIONALE: Hour angle H = (pi/12) * (t_solar/3600 - 12), noon = 0
// PRE/POST : Pre: local_time_seconds finite; Post: hour angle continuous in time
[[nodiscard]] inline double solar_hour_angle_rad(double local_time_seconds) noexcept {
    return (kSolarPi / 12.0) * (local_time_seconds / kSecondsPerHour - 12.0);
}

// [UNIT_OP: GUI_BUILD_2D] [RULE_4: 7/60] [CONSERVATION: sigma=0, w_ext=EXTERNAL_QSW]
// RATIONALE: sin_elevation = sin(lat)sin(dec) + cos(lat)cos(dec)cos(H)
// PRE/POST : Pre: all angles finite; Post: sin_elevation in [-1, 1]
[[nodiscard]] inline double solar_sin_elevation(
    double latitude_rad,
    double declination_rad,
    double hour_angle_rad
) noexcept {
    return std::sin(latitude_rad) * std::sin(declination_rad)
         + std::cos(latitude_rad) * std::cos(declination_rad) * std::cos(hour_angle_rad);
}

// [UNIT_OP: GUI_BUILD_2D] [RULE_4: 10/60] [CONSERVATION: sigma=0, w_ext=EXTERNAL_QSW]
// RATIONALE: Aggregate solar geometry from config at a given simulation time
// PRE/POST : Pre: cfg valid, simulation_time_s >= 0; Post: geometry fields finite
[[nodiscard]] inline SolarGeometry2D compute_solar_geometry(
    const SolarGroundConfig2D& cfg,
    double simulation_time_s
) noexcept {
    const double t_solar = cfg.start_local_hour * kSecondsPerHour
        + simulation_time_s * cfg.solar_time_scale;

    SolarGeometry2D geo{};
    geo.hour_angle_rad  = solar_hour_angle_rad(t_solar);
    geo.declination_rad = solar_declination_rad(cfg.day_of_year);
    geo.sin_elevation   = solar_sin_elevation(
        cfg.latitude_deg * kSolarDeg2Rad, geo.declination_rad, geo.hour_angle_rad
    );
    return geo;
}

// [UNIT_OP: GUI_BUILD_2D] [RULE_4: 14/60] [CONSERVATION: sigma=0, w_ext=EXTERNAL_QSW]
// RATIONALE: Absorbed shortwave at the ground Q_sw = scale * tau_c * (1-a) * S_toa * exp(-tau);
//            tau = k_cloud * (CWP + RWP) attenuates by column cloud/rain water path
// PRE/POST : Pre: cfg valid; Post: Q_sw >= 0 (zero at night or fully opaque column)
[[nodiscard]] inline double compute_absorbed_shortwave_w_m2(
    const SolarGroundConfig2D& cfg,
    double simulation_time_s,
    double column_cloud_water_path_kg_m2,
    double column_rain_water_path_kg_m2
) noexcept {
    const SolarGeometry2D geo = compute_solar_geometry(cfg, simulation_time_s);

    const double flux_scale = std::max(0.0, cfg.solar_flux_multiplier);
    const double s_toa = flux_scale * cfg.solar_constant_w_m2 * std::max(0.0, geo.sin_elevation);
    const double tau   = cfg.cloud_attenuation_per_kg_m2
        * (column_cloud_water_path_kg_m2 + column_rain_water_path_kg_m2);
    const double transmitted = cfg.clear_sky_transmittance * (1.0 - cfg.albedo)
        * s_toa * std::exp(-tau);
    return std::max(0.0, transmitted);
}

// [UNIT_OP: GUI_BUILD_2D] [RULE_4: 14/60] [CONSERVATION: sigma=0, w_ext=EXTERNAL_QSW]
// RATIONALE: Effective surface albedo blending water body, lowland soil, and alpine elevation.
// PRE/POST : Pre: cfg valid; Post: effective albedo clamped to [0.02, 0.90]
[[nodiscard]] inline double compute_column_albedo(
    const SolarGroundConfig2D& cfg,
    const SurfaceColumnProps2D& col
) noexcept {
    const double w_frac = std::clamp(col.water_fraction, 0.0, 1.0);
    const double h_norm = std::clamp(col.terrain_elevation_m / 1800.0, 0.0, 1.0);
    const double land_albedo = cfg.albedo + h_norm * (cfg.mountain_albedo - cfg.albedo);
    const double eff_albedo  = (1.0 - w_frac) * land_albedo + w_frac * cfg.water_albedo;
    return std::clamp(eff_albedo, 0.02, 0.90);
}

// [UNIT_OP: GUI_BUILD_2D] [RULE_4: 14/60] [CONSERVATION: sigma=0, w_ext=EXTERNAL_QSW]
// RATIONALE: Effective thermal inertia C_g blending rocky mountain skin, soil, and water body.
// PRE/POST : Pre: cfg valid; Post: heat capacity strictly positive (J/(m^2 K))
[[nodiscard]] inline double compute_column_heat_capacity_j_m2_k(
    const SolarGroundConfig2D& cfg,
    const SurfaceColumnProps2D& col
) noexcept {
    const double w_frac = std::clamp(col.water_fraction, 0.0, 1.0);
    const double h_norm = std::clamp(col.terrain_elevation_m / 1800.0, 0.0, 1.0);
    const double land_cg = cfg.ground_heat_capacity_j_m2_k * (1.0 - 0.35 * h_norm);
    const double eff_cg  = (1.0 - w_frac) * land_cg + w_frac * cfg.water_heat_capacity_j_m2_k;
    return std::max(2.0e4, eff_cg);
}

// [UNIT_OP: GUI_BUILD_2D] [RULE_4: 18/60] [CONSERVATION: sigma=0, w_ext=EXTERNAL_QSW]
// RATIONALE: Per-column absorbed shortwave including water/mountain albedo and slope aspect.
// PRE/POST : Pre: cfg valid; Post: Q_sw >= 0
[[nodiscard]] inline double compute_column_shortwave_w_m2(
    const SolarGroundConfig2D& cfg,
    const SurfaceColumnProps2D& col,
    double simulation_time_s,
    double column_cloud_water_path_kg_m2,
    double column_rain_water_path_kg_m2
) noexcept {
    const SolarGeometry2D geo = compute_solar_geometry(cfg, simulation_time_s);
    const double flux_scale = std::max(0.0, cfg.solar_flux_multiplier);
    const double s_toa = flux_scale * cfg.solar_constant_w_m2 * std::max(0.0, geo.sin_elevation);
    const double tau = cfg.cloud_attenuation_per_kg_m2
        * (column_cloud_water_path_kg_m2 + column_rain_water_path_kg_m2);

    const double albedo_col = compute_column_albedo(cfg, col);
    // Sunlit slope modulation: east-facing slopes (slope_dx < 0) get morning boost (H < 0)
    const double slope_factor = std::clamp(1.0 + 0.25 * col.slope_dx * std::sin(geo.hour_angle_rad), 0.6, 1.4);
    const double alt_boost = 1.0 + 0.06 * std::clamp(col.terrain_elevation_m / 1800.0, 0.0, 1.0);
    const double transmitted = cfg.clear_sky_transmittance * alt_boost * slope_factor
        * (1.0 - albedo_col) * s_toa * std::exp(-tau);
    return std::max(0.0, transmitted);
}

// [UNIT_OP: GUI_BUILD_2D] [RULE_4: 24/60] [CONSERVATION: sigma=0, w_ext=EXTERNAL_QSW]
// RATIONALE: Forward-Euler ground energy balance:
//            C_g dT_g/dt = Q_sw - Q_sens - Q_lat - Q_lw.
//            Q_sens/Q_lat use the SHARED surface flux kernel so the ground loss equals
//            the air-side SAT gain exactly (Q_lat uses full vapor enthalpy h_vap(T_s),
//            not just L_v, preserving air-domain energy closure). Q_lw is an external
//            radiative sink; CFL-safe because C_g ~ 2e5 J/(m^2 K) >> dt.
// PRE/POST : Pre: dt_s > 0, T_g > 0; Post: T_g clamped to [min,max] physical bounds
[[nodiscard]] inline double advance_ground_temp_with_capacity_k(
    double temp_ground_k,
    double q_sw_w_m2,
    double temp_air_k,
    double mu_v_air_j_kg,
    double sensible_conductance_w_k_m2,
    double evaporation_mobility,
    double heat_capacity_j_m2_k,
    const SolarGroundConfig2D& cfg,
    double dt_s
) noexcept {
    if (!(temp_ground_k > 0.0) || !(dt_s > 0.0) || !(heat_capacity_j_m2_k > 0.0)) {
        return temp_ground_k; // Fail closed: no ground temperature advance
    }

    const thermo::SurfaceFluxResult2D flux = thermo::compute_surface_fluxes_2d(
        temp_air_k,
        mu_v_air_j_kg,
        temp_ground_k,
        sensible_conductance_w_k_m2,
        evaporation_mobility,
        cfg.emissivity,
        cfg.enable_longwave
    );

    const double net_heating_w_m2 = q_sw_w_m2
        - flux.sensible_w_m2
        - flux.latent_energy_w_m2
        - flux.longwave_net_w_m2;

    double next_k = temp_ground_k
        + (dt_s / heat_capacity_j_m2_k) * net_heating_w_m2;

    if (next_k < cfg.min_ground_temp_k) { next_k = cfg.min_ground_temp_k; }
    if (next_k > cfg.max_ground_temp_k) { next_k = cfg.max_ground_temp_k; }
    return next_k;
}

// [UNIT_OP: GUI_BUILD_2D] [RULE_4: 14/60] [CONSERVATION: sigma=0, w_ext=EXTERNAL_QSW]
// RATIONALE: Forward-Euler ground advance using default cfg.ground_heat_capacity_j_m2_k.
// PRE/POST : Pre: dt_s > 0, T_g > 0; Post: T_g clamped to [min,max] physical bounds
[[nodiscard]] inline double advance_ground_temp_k(
    double temp_ground_k,
    double q_sw_w_m2,
    double temp_air_k,
    double mu_v_air_j_kg,
    double sensible_conductance_w_k_m2,
    double evaporation_mobility,
    const SolarGroundConfig2D& cfg,
    double dt_s
) noexcept {
    return advance_ground_temp_with_capacity_k(
        temp_ground_k, q_sw_w_m2, temp_air_k, mu_v_air_j_kg,
        sensible_conductance_w_k_m2, evaporation_mobility,
        cfg.ground_heat_capacity_j_m2_k, cfg, dt_s
    );
}

} // namespace metriplectic_2d::cases

#endif // METRIPLECTIC_2D_CASES_SOLAR_GROUND_FORCING_2D_HPP
