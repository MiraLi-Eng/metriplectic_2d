# metriplectic_2d

Header-only C++20 kernel library for a 2D metriplectic fluid / moist-weather
vertical-slice simulator. Provides the bracket formulation (reversible,
irreversible/Onsager, stochastic), SBP spatial operators, thermodynamic EOS
and closure auditing, and a HEVI-ARK time integrator.

## Layout

| Directory   | Contents |
|-------------|----------|
| `core/`     | Cache-aligned static arena, PRNG, canonical types |
| `brackets/` | Reversible Poisson, Onsager/irreversible, and stochastic brackets |
| `spatial/`  | SBP operators and SAT boundary treatment |
| `thermo/`   | Moist EOS, hysteretic mobility, Padé tables, EOS concept |
| `time/`     | HEVI-ARK integrator, block-Thomas solver, closure auditor |
| `cases/`    | Locked weather simulation case definitions |
| `telemetry/`| Ring-buffer GNN telemetry tap |

## Build

Consume as an INTERFACE library (C++20, `-fno-exceptions -fno-rtti`):

```cmake
add_library(metriplectic_2d_kernel INTERFACE)
target_include_directories(metriplectic_2d_kernel INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(metriplectic_2d_kernel INTERFACE
    -Wall -Wextra -Wpedantic -Werror -fno-exceptions -fno-rtti -ffp-contract=off -O3)
```
