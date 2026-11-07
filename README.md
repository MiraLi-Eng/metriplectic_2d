# metriplectic_2d

[![Standard: C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B20)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: Passing](https://img.shields.io/badge/Tests-100%25%20Passing-brightgreen.svg)]()

A header-only C++20 scientific computing kernel for 2D metriplectic fluid dynamics and moist-atmospheric vertical-slice weather simulation.

`metriplectic_2d` couples non-equilibrium moist thermodynamics, high-order Summation-by-Parts (SBP) spatial operators, and Horizontally Explicit, Vertically Implicit (HEVI-ARK) time integration within a thermodynamically consistent metriplectic bracket architecture.

---

## Metriplectic Bracket Framework

Physical state evolution is governed by the metriplectic equation of motion:

$$\frac{du}{dt} = \mathcal{L}(u)\,\frac{\delta H}{\delta u} + \mathcal{M}(u)\,\frac{\delta S}{\delta u} + \Sigma(u)\,\xi(t)$$

where:
* **$u = (\rho_d, \rho_v, \rho_c, \rho_r, \rho u_x, \rho u_y, \varepsilon_{\text{tot}})^T$** is the canonical conserved state vector containing dry air, water vapor, cloud water, rain water densities, momentum components, and total internal/kinetic energy.
* **$H(u)$** is the total Hamiltonian (energy functional).
* **$S(u)$** is the total thermodynamic entropy functional.
* **$\mathcal{L}(u)$** is the skew-symmetric Poisson bracket operator governing reversible, adiabatic advection and pressure-buoyancy dynamics.
* **$\mathcal{M}(u)$** is the symmetric positive semi-definite Onsager operator governing irreversible dissipation, heat conduction, moisture diffusion, and phase relaxation.
* **$\Sigma(u)\,\xi(t)$** is the stochastic Langevin fluctuation operator satisfying the Fluctuation-Dissipation Theorem (FDT).

### Exact Thermodynamic Invariants
The kernel enforces structural degeneracy conditions:
$$\mathcal{L}(u)\,\frac{\delta S}{\delta u} = 0 \quad \text{(entropy is a Casimir of the Poisson bracket)}$$
$$\mathcal{M}(u)\,\frac{\delta H}{\delta u} = 0 \quad \text{(Hamiltonian is in the nullspace of the Onsager bracket)}$$

These conditions guarantee unconditional discrete compliance with the fundamental laws of thermodynamics:
1. **First Law (Energy Conservation):** $\frac{dH}{dt} = 0$ (under isolated boundary conditions, or exactly balanced by prescribed external fluxes $\dot{w}_{\text{ext}}$).
2. **Second Law (Non-Negative Entropy Production):** $\sigma_{\text{gen}} = \left\langle \frac{\delta S}{\delta u}, \mathcal{M}(u) \frac{\delta S}{\delta u} \right\rangle \ge 0$ unconditionally.

---

## Key Capabilities & Features

* **Thermodynamics & Non-Equilibrium Moist Physics:**
  * Multi-component moist equation of state (`thermo/moist_eos.hpp`).
  * High-order rational Padé approximants (`thermo/pade_tables.hpp`) for vapor saturation pressure $e_{\text{sat}}(T)$, chemical potentials $\psi_l(T)$, and latent heat.
  * Hysteretic phase transitions (`thermo/hysteretic_mobility.hpp`) modeling subgrid condensation, supersaturation, and rain formation without non-physical equilibrium snapping.
  * Unified Onsager surface fluxes (`thermo/surface_fluxes_2d.hpp`) computing sensible heat exchange, Planck-potential evaporation, and longwave Stefan-Boltzmann radiative cooling.

* **High-Order Energy-Stable Discretization:**
  * Diagonal-norm Summation-by-Parts (SBP) finite difference operators (`spatial/sbp_operators_2d.hpp`).
  * Simultaneous Approximation Term (SAT) weak boundary conditions (`spatial/sat_boundaries_2d.hpp`) preserving provable global stability.

* **Stiff-Wave Multi-Rate Time Integration:**
  * Horizontally Explicit, Vertically Implicit Additive Runge-Kutta (HEVI-ARK) integrator (`time/hevi_ark_2d.hpp`).
  * Exact $O(N_y)$ vertical block-tridiagonal linear solver (`time/block_thomas_2d.hpp`) for fast acoustic and acoustic-gravity waves.
  * Real-time thermodynamic closure auditor (`time/closure_auditor_2d.hpp`) monitoring conservation and entropy production step-by-step.

* **Solar & Surface Topography Modeling:**
  * Diurnal solar radiation forcing (`cases/solar_ground_forcing_2d.hpp`) with orbital geometry (Cooper declination, hour angle), atmospheric transmittance, and cloud attenuation.
  * Multi-surface albedo and heat capacity models (land, water bodies, mountain ridge snow).
  * Per-column heated ground ODE coupled conservation-consistently to the atmospheric column.

* **Engineering & Safety Standards:**
  * Zero dynamic heap churn post-initialization (`core/arena.hpp`).
  * Strict lifetime hygiene; `-fno-exceptions` and `-fno-rtti` compatible.
  * Full safety-critical coding standards and practices documented in [docs/SAFETY_STANDARDS.md](docs/SAFETY_STANDARDS.md).

---

## Directory Architecture

```
include/metriplectic_2d/
├── core/
│   ├── arena.hpp               # Cache-aligned static memory arena (zero heap)
│   ├── prng.hpp                # Deterministic PCG32 pseudo-random generator
│   └── types.hpp               # Canonical state vectors, dimensions, geometry types
├── brackets/
│   ├── reversible_l_2d.hpp     # Skew-symmetric Poisson bracket (ideal hydrodynamics)
│   ├── onsager_m_2d.hpp        # Positive semi-definite Onsager bracket (dissipation)
│   └── stochastic_sigma_2d.hpp # Fluctuation-Dissipation Langevin stochastic bracket
├── spatial/
│   ├── sbp_operators_2d.hpp    # High-order Summation-by-Parts differentiation
│   └── sat_boundaries_2d.hpp   # SBP-SAT simultaneous approximation boundary operator
├── thermo/
│   ├── eos_concept.hpp         # C++20 concepts for thermodynamic equations of state
│   ├── moist_eos.hpp           # Non-ideal multi-phase moist air EOS
│   ├── pade_tables.hpp         # Rational Padé approximation tables for saturation & potentials
│   ├── hysteretic_mobility.hpp # Hysteretic condensation and rain autoconversion
│   └── surface_fluxes_2d.hpp   # Unified Onsager sensible & Planck-potential latent fluxes
├── time/
│   ├── hevi_ark_2d.hpp         # HEVI-ARK stiff-wave split time integrator
│   ├── block_thomas_2d.hpp     # Vectorized block-tridiagonal Thomas solver
│   └── closure_auditor_2d.hpp  # Step-by-step thermodynamic conservation auditor
├── cases/
│   ├── weather_case_2d.hpp     # Standard vertical-slice benchmark initializations
│   └── solar_ground_forcing_2d.hpp # Diurnal shortwave solar and ground ODE forcing
├── telemetry/
│   └── gnn_ring_buffer_2d.hpp  # Zero-allocation ring buffer state telemetry tap
└── docs/
    └── SAFETY_STANDARDS.md     # Code safety and deterministic execution guidelines
```

---

## Quick Start & Integration

`metriplectic_2d` is an interface/header-only library requiring a modern C++20 compiler (Clang 16+, GCC 12+, or Apple Clang 15+).

### CMake Usage

Add the include directory as an interface library in your `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.24)
project(my_weather_simulation CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(metriplectic_2d INTERFACE)
target_include_directories(metriplectic_2d INTERFACE
    path/to/metriplectic_2d/include
)
target_compile_options(metriplectic_2d INTERFACE
    -Wall -Wextra -Wpedantic -Werror
    -fno-exceptions -fno-rtti
    -ffp-contract=off
    -O3
)

add_executable(sim main.cpp)
target_link_libraries(sim PRIVATE metriplectic_2d)
```

### Basic Usage Example

```cpp
#include "metriplectic_2d/core/types.hpp"
#include "metriplectic_2d/cases/weather_case_2d.hpp"
#include "metriplectic_2d/time/hevi_ark_2d.hpp"
#include "metriplectic_2d/time/closure_auditor_2d.hpp"

using namespace metriplectic_2d;

int main() {
    constexpr size_t Nx = 128;
    constexpr size_t Ny = 64;
    
    // Configure vertical-slice domain
    core::GridGeometry2D geom{
        .nx = Nx,
        .ny = Ny,
        .dx = 100.0, // 100m horizontal resolution
        .dy = 50.0   // 50m vertical resolution
    };

    // Initialize state on cache-aligned static arena
    core::StaticArena<1024 * 1024 * 16> arena;
    auto* state = arena.allocate<core::StateVector2D<Nx, Ny>>();
    
    // Initialize standard convective bubble test case
    cases::initialize_moist_bubble(*state, geom);

    // Initialize HEVI-ARK stepper and closure auditor
    time::HeviArkIntegrator2D<Nx, Ny> integrator(geom);
    time::ClosureAuditor2D auditor;

    double dt = 0.1; // seconds
    for (int step = 0; step < 1000; ++step) {
        integrator.step(*state, dt);
        
        auto audit_result = auditor.audit(*state, geom);
        // Verify energy conservation |dE/dt| <= eps and entropy generation >= 0
        if (!audit_result.is_physical()) {
            return 1;
        }
    }

    return 0;
}
```

---

## Verification & Test Suite

The library includes an exhaustive suite of unit and physical verification tests verifying mathematical invariance, SBP operator adjoint properties, Onsager positive-semi-definiteness, and conservation laws.

To build and run the test suite:

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
ctest --output-on-failure
```

### Verified Properties
* **Energy conservation:** Hamiltonian total energy invariant within $10^{-12}$ relative tolerance under adiabatic flows.
* **Discrete SBP property:** Skew-symmetry of $Q$ and exact boundary projection via $B = \text{diag}(-1, 0, \dots, 0, 1)$.
* **Entropy monotonicity:** Onsager dissipation satisfies $\sigma_{\text{gen}} \ge 0$ unconditionally across all phase transitions and turbulent diffusive regimes.
* **Padé approximation accuracy:** Inversion accuracy within machine precision across temperatures from $200\,\text{K}$ to $340\,\text{K}$.

---

## Roadmap & Planned Work

- [ ] **Boundary Interaction & Wave Reflection ("Fishtank" Effect):** Mitigate unphysical acoustic and gravity wave reflection and artificial boundary confinement at domain borders (absorbing sponge layers / buffer zones and non-reflecting open radiation SAT boundary conditions).
- [ ] **Atmospheric Radiative Transfer:** Model multi-layer atmospheric absorption of solar radiation and Stefan-Boltzmann thermal longwave transmission/cooling between the ground and atmospheric layers (including nocturnal heat transfer and greenhouse re-radiation).

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
