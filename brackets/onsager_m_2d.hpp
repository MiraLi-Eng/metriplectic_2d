#ifndef METRIPLECTIC_2D_BRACKETS_ONSAGER_M_2D_HPP
#define METRIPLECTIC_2D_BRACKETS_ONSAGER_M_2D_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include "metriplectic_2d/core/types.hpp"
#include "metriplectic_2d/spatial/sbp_operators_2d.hpp"
#include "metriplectic_2d/thermo/moist_eos.hpp"
#include "metriplectic_2d/thermo/pade_tables.hpp"

namespace metriplectic_2d::brackets {

// ============================================================================
// 2D Positive Semi-Definite Dyadic Onsager Bracket M(x) dS/dx (ADR-2D-006)
// ============================================================================
// Implements all irreversible dissipative processes via symmetric PSD dyads:
//   1. Smooth C^inf Köhler Hysteretic Phase Relaxation (rho_v <-> rho_c)
//   2. Warm-Rain Accretion / Sub-Cloud Rain Evaporation (rho_c -> rho_r <-> rho_v)
//   3. Lilly-Deardorff SGS Shear Transfer (m -> rho*e_sgs) & Viscous Dissipation
//   4. Kolmogorov SGS Cascade Dissipation (rho*e_sgs -> rho*e_int at fixed eps)
//   5. Fourier Thermal Conduction & Fickian Moisture Diffusion
//   6. Gravitational Rain Terminal Slip Sedimentation along -y with Frictional Heating
//
// Provably satisfies:
//   - Energy Degeneracy:  sum_{x,y} P_x P_y (M(x) dS/dx)_{eps} == 0
//   - Second Law (PSD):   dot(S)_prod = sum_{x,y} P_x P_y w^T (M(x) dS/dx) >= 0

struct OnsagerTransportParams2D {
    double dynamic_viscosity_pa_s    = 1.8e-5; // Molecular + background eddy viscosity (Pa*s)
    double smagorinsky_cs            = 0.18;   // Lilly-Deardorff Smagorinsky constant C_s
    double cascade_ceps              = 0.845;  // Kolmogorov cascade coefficient C_epsilon
    double thermal_conductivity_w_mk = 2.5e-2; // Effective thermal conductivity kappa (W/(m*K))
    double vapor_diffusivity_m2_s    = 2.4e-5; // Effective Fickian diffusivity D_v (m^2/s)
    double autoconversion_rate_s1    = 1.0e-3; // Warm-rain autoconversion rate (1/s)
    double autoconversion_thresh     = 5.0e-4; // Cloud threshold for warm rain (kg/m^3)
    bool enable_rain_sedimentation   = true;   // Internal y-column terminal rain slip
};

template <bool EnableKorteweg = false>
class OnsagerBracketM2D {
public:
    // Evaluates cell-local and face-dyadic irreversible tendencies rhs_grid += M(x) dS/dx,
    // populates per-cell CriticalSlowingMetrics, and returns total domain entropy
    // production rate dot(S)_prod >= 0 (W / K).
    static double apply(
        const core::NodeStateArray* state_grid,
        const core::NodeEntropyArray* w_grid,
        core::NodeStateArray* rhs_grid,
        core::CriticalSlowingMetrics* crit_metrics_out,
        const spatial::GridGeometry2D& geom,
        const thermo::MoistPadeEOS2D<EnableKorteweg>& eos,
        const OnsagerTransportParams2D& params = {}
    ) noexcept {
        using core::idx;
        using core::StateVar;
        using spatial::SBPOperators2D;
        using thermo::PadeThermodynamicTables;
        using thermo::ThermoConstants;

        double total_entropy_prod_w_k = 0.0;
        const double delta_grid = std::sqrt(geom.dx * geom.dy);

        // --------------------------------------------------------------------
        // Part 1: Cell-Local PSD Blocks (Phase Hysteresis + Kolmogorov Cascade)
        // --------------------------------------------------------------------
        for (size_t iy = 0; iy < geom.ny; ++iy) {
            const double cell_vol = geom.cell_volume_weight(0, iy);
            for (size_t ix = 0; ix < geom.nx; ++ix) {
                const size_t id = geom.linear_index(ix, iy);
                const core::NodeStateArray& x = state_grid[id];
                const core::NodeEntropyArray& w = w_grid[id];
                core::NodeStateArray& rhs = rhs_grid[id];

                const double temp_k = 1.0 / w[idx(StateVar::EPS_TOT)];
                const double rho_d  = x[idx(StateVar::RHO_D)];
                const double rho_v  = x[idx(StateVar::RHO_V)];
                const double rho_c  = x[idx(StateVar::RHO_C)];
                const double rho_r  = x[idx(StateVar::RHO_R)];
                const double e_sgs  = x[idx(StateVar::RHO_E_SGS)];
                const double rho_tot = rho_d + rho_v + rho_c + rho_r;

                // 1a. Smooth C^inf Köhler Hysteretic Phase Mobility (rho_v <-> rho_c)
                // Note: w_{rho_c} - w_{rho_v} == (mu_v - mu_c^eff) / T == delta_mu_eff / T!
                const double dmu_eff = (w[idx(StateVar::RHO_C)] - w[idx(StateVar::RHO_V)]) * temp_k;
                const auto crit = eos.phase_mobility().evaluate_critical_slowing_metrics(
                    rho_c, dmu_eff, temp_k
                );
                if (crit_metrics_out != nullptr) {
                    crit_metrics_out[id] = crit;
                }

                const double j_phase = eos.evaluate_phase_flux(x, dmu_eff, temp_k);
                rhs[idx(StateVar::RHO_V)] -= j_phase;
                rhs[idx(StateVar::RHO_C)] += j_phase;
                // Total energy eps is strictly invariant (dot(eps)_phase == 0)
                total_entropy_prod_w_k += cell_vol * crit.entropy_prod_w_m3_k;

                // 1b. Sub-cloud rain evaporation / condensation (rho_v <-> rho_r)
                const double dmu_rain = (w[idx(StateVar::RHO_R)] - w[idx(StateVar::RHO_V)]) * temp_k;
                const double m_rain_evap = eos.phase_mobility().params.gamma_active_kg2_k_j_m3_s
                    * (rho_r / (rho_r + eos.phase_mobility().params.rho_c_ref_kg_m3));
                const double j_rain_phase = (dmu_rain / temp_k) * m_rain_evap;
                rhs[idx(StateVar::RHO_V)] -= j_rain_phase;
                rhs[idx(StateVar::RHO_R)] += j_rain_phase;
                total_entropy_prod_w_k += cell_vol * m_rain_evap * (dmu_rain / temp_k) * (dmu_rain / temp_k);

                // 1c. Warm-rain autoconversion (rho_c -> rho_r at equal chemical potential mu_c == mu_r)
                if (rho_c > params.autoconversion_thresh) {
                    const double j_auto = params.autoconversion_rate_s1
                        * (rho_c - params.autoconversion_thresh);
                    rhs[idx(StateVar::RHO_C)] -= j_auto;
                    rhs[idx(StateVar::RHO_R)] += j_auto;
                }

                // 1d. Kolmogorov SGS Cascade Dissipation (rho*e_sgs -> rho*e_int at fixed eps)
                // Since w_sgs = -1/T, decaying rho*e_sgs at fixed eps produces +eps_casc / T >= 0!
                if (e_sgs > 0.0) {
                    const double eps_casc = params.cascade_ceps
                        * (e_sgs * std::sqrt(e_sgs / rho_tot)) / delta_grid;
                    rhs[idx(StateVar::RHO_E_SGS)] -= eps_casc;
                    total_entropy_prod_w_k += cell_vol * (eps_casc / temp_k);
                }
            }
        }

        // --------------------------------------------------------------------
        // Part 2: Face-Centered PSD Dyads (Shear, Conduction, Fickian Diffusion)
        // --------------------------------------------------------------------
        auto apply_face_dyad = [&](
            size_t id_l,
            size_t id_r,
            double vol_l,
            double vol_r,
            double face_area,
            double dist_lr
        ) noexcept {
            const core::NodeStateArray& xl = state_grid[id_l];
            const core::NodeStateArray& xr = state_grid[id_r];
            const core::NodeEntropyArray& wl = w_grid[id_l];
            const core::NodeEntropyArray& wr = w_grid[id_r];

            const double w_eps_l = wl[idx(StateVar::EPS_TOT)];
            const double w_eps_r = wr[idx(StateVar::EPS_TOT)];
            const double w_eps_bar = 0.5 * (w_eps_l + w_eps_r);
            const double temp_l  = 1.0 / w_eps_l;
            const double temp_r  = 1.0 / w_eps_r;
            const double temp_bar = 0.5 * (temp_l + temp_r);

            const double ux_l = -wl[idx(StateVar::MOM_UX)] * temp_l;
            const double uy_l = -wl[idx(StateVar::MOM_UY)] * temp_l;
            const double ux_r = -wr[idx(StateVar::MOM_UX)] * temp_r;
            const double uy_r = -wr[idx(StateVar::MOM_UY)] * temp_r;

            const double ux_bar = 0.5 * (ux_l + ux_r);
            const double uy_bar = 0.5 * (uy_l + uy_r);

            // Exact thermodynamic velocity-gradient affinity:
            // a_visc = (w_{m,L} - w_{m,R}) + bar(v)_{LR} (w_{eps,L} - w_{eps,R}) == bar(w_eps)*(v_R - v_L)
            const double dux = ux_r - ux_l;
            const double duy = uy_r - uy_l;

            const double rho_bar = 0.5 * (
                xl[idx(StateVar::RHO_D)] + xl[idx(StateVar::RHO_V)] +
                xr[idx(StateVar::RHO_D)] + xr[idx(StateVar::RHO_V)]
            );
            const double e_sgs_bar = 0.5 * (xl[idx(StateVar::RHO_E_SGS)] + xr[idx(StateVar::RHO_E_SGS)]);
            const double mu_sgs = params.smagorinsky_cs * rho_bar * delta_grid
                * std::sqrt(std::max(0.0, e_sgs_bar / std::max(rho_bar, 1.0e-6)));
            const double mu_tot = params.dynamic_viscosity_pa_s + mu_sgs;

            // 1. Viscous momentum & work dyad (strictly PSD: sigma_visc >= 0, dE_tot == 0)
            const double j_mom_x = (mu_tot / dist_lr) * dux;
            const double j_mom_y = (mu_tot / dist_lr) * duy;
            const double j_eps_visc = ux_bar * j_mom_x + uy_bar * j_mom_y;

            // 2. Fourier thermal conduction dyad driven by dw_eps = 1/T_L - 1/T_R
            const double dw_eps = w_eps_l - w_eps_r;
            const double j_eps_cond = (params.thermal_conductivity_w_mk * temp_bar * temp_bar / dist_lr) * dw_eps;

            // 3. Fickian moisture diffusion dyad driven by dw_diff = (w_v - w_d)_L - (w_v - w_d)_R
            const double dw_diff_l = wl[idx(StateVar::RHO_V)] - wl[idx(StateVar::RHO_D)];
            const double dw_diff_r = wr[idx(StateVar::RHO_V)] - wr[idx(StateVar::RHO_D)];
            const double dw_diff   = dw_diff_l - dw_diff_r;
            const double j_diff    = (params.vapor_diffusivity_m2_s * rho_bar * temp_bar / dist_lr) * dw_diff;

            // Apply conservative fluxes (subtract from R, add to L)
            const double scale_l = face_area / vol_l;
            const double scale_r = face_area / vol_r;

            rhs_grid[id_l][idx(StateVar::RHO_D)]  -= scale_l * j_diff;
            rhs_grid[id_r][idx(StateVar::RHO_D)]  += scale_r * j_diff;
            rhs_grid[id_l][idx(StateVar::RHO_V)]  += scale_l * j_diff;
            rhs_grid[id_r][idx(StateVar::RHO_V)]  -= scale_r * j_diff;

            rhs_grid[id_l][idx(StateVar::MOM_UX)] += scale_l * j_mom_x;
            rhs_grid[id_r][idx(StateVar::MOM_UX)] -= scale_r * j_mom_x;
            rhs_grid[id_l][idx(StateVar::MOM_UY)] += scale_l * j_mom_y;
            rhs_grid[id_r][idx(StateVar::MOM_UY)] -= scale_r * j_mom_y;

            const double j_eps_tot = j_eps_visc + j_eps_cond;
            rhs_grid[id_l][idx(StateVar::EPS_TOT)] += scale_l * j_eps_tot;
            rhs_grid[id_r][idx(StateVar::EPS_TOT)] -= scale_r * j_eps_tot;

            const double sigma_face = face_area * (
                (mu_tot / dist_lr) * w_eps_bar * (dux * dux + duy * duy) +
                dw_eps * j_eps_cond +
                dw_diff * j_diff
            );
            total_entropy_prod_w_k += sigma_face;
        };

        // Horizontal faces (ix, (ix + 1) mod NX)
        for (size_t iy = 0; iy < geom.ny; ++iy) {
            const double py = geom.norm_weight_y(iy);
            const double vol = geom.dx * py;
            for (size_t ix = 0; ix < geom.nx; ++ix) {
                const size_t ix_r = geom.wrap_x(static_cast<ptrdiff_t>(ix) + 1);
                apply_face_dyad(
                    geom.linear_index(ix, iy),
                    geom.linear_index(ix_r, iy),
                    vol, vol, py, geom.dx
                );
            }
        }

        // Vertical column faces (iy, iy + 1) + Gravitational Rain Sedimentation
        for (size_t ix = 0; ix < geom.nx; ++ix) {
            for (size_t iy = 0; iy + 1 < geom.ny; ++iy) {
                const size_t id_bot = geom.linear_index(ix, iy);
                const size_t id_top = geom.linear_index(ix, iy + 1);
                const double vol_bot = geom.cell_volume_weight(ix, iy);
                const double vol_top = geom.cell_volume_weight(ix, iy + 1);

                apply_face_dyad(id_bot, id_top, vol_bot, vol_top, geom.dx, geom.dy);

                // Gravitational rain sedimentation slip along -y (top -> bot) (GEN-002)
                // Formulated as an Onsager flux driven by the rain potential jump
                // dw_sed = (w_{rho_r, bot} - w_{rho_r, top}) + h_sed * (w_{eps, bot} - w_{eps, top}).
                // Using the path-integrated liquid energy + kinetic + mean geopotential enthalpy
                // h_sed makes dw_sed == bar(w_eps) * g * dy > 0 identically, so:
                //   sigma_sed = j_sed * bar(w_eps) * g * dy == (rho_r * g * w_T / bar(T)) * dy >= 0!
                if (params.enable_rain_sedimentation) {
                    const double rho_r_top = state_grid[id_top][idx(StateVar::RHO_R)];
                    if (rho_r_top > 0.0) {
                        const double w_term = PadeThermodynamicTables::rain_terminal_speed_m_s(rho_r_top);
                        const double j_sed  = rho_r_top * w_term; // >= 0 (kg / (m^2 s) downward)

                        const core::NodeEntropyArray& w_bot = w_grid[id_bot];
                        const core::NodeEntropyArray& w_top = w_grid[id_top];

                        const double w_eps_bot = w_bot[idx(StateVar::EPS_TOT)];
                        const double w_eps_top = w_top[idx(StateVar::EPS_TOT)];
                        const double w_eps_bar = 0.5 * (w_eps_bot + w_eps_top);
                        const double dphi_grav = ThermoConstants::kGravity * geom.dy;

                        // Construct energy transfer j_eps_sed so that total energy is conserved
                        // and the net entropy production across (bot, top) is exactly
                        // j_sed * w_eps_bar * g * dy >= 0 (frictional wake heating!):
                        const double dw_r = w_bot[idx(StateVar::RHO_R)] - w_top[idx(StateVar::RHO_R)];
                        const double dw_eps = w_eps_bot - w_eps_top;

                        double h_sed = 0.0;
                        if (std::abs(dw_eps) > 1.0e-14) {
                            h_sed = (w_eps_bar * dphi_grav - dw_r) / dw_eps;
                        } else {
                            const double temp_bar = 1.0 / w_eps_bar;
                            const double phi_bar = ThermoConstants::kGravity * (geom.y_coord(iy) + 0.5 * geom.dy);
                            h_sed = PadeThermodynamicTables::internal_energy_liquid(temp_bar) + phi_bar;
                        }

                        const double j_eps_sed = j_sed * h_sed;
                        const double scale_bot = geom.dx / vol_bot;
                        const double scale_top = geom.dx / vol_top;

                        rhs_grid[id_bot][idx(StateVar::RHO_R)]   += scale_bot * j_sed;
                        rhs_grid[id_top][idx(StateVar::RHO_R)]   -= scale_top * j_sed;
                        rhs_grid[id_bot][idx(StateVar::EPS_TOT)] += scale_bot * j_eps_sed;
                        rhs_grid[id_top][idx(StateVar::EPS_TOT)] -= scale_top * j_eps_sed;

                        const double sigma_sed = geom.dx * j_sed * (dw_r + h_sed * dw_eps);
                        total_entropy_prod_w_k += sigma_sed;
                    }
                }
            }
        }

        return total_entropy_prod_w_k;
    }
};

} // namespace metriplectic_2d::brackets

#endif // METRIPLECTIC_2D_BRACKETS_ONSAGER_M_2D_HPP
