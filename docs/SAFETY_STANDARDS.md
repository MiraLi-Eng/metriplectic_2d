# Safety & Coding Standards

This document outlines the coding and safety guidelines applied across the `metriplectic_2d` kernel.

## NASA/JPL Power of 10 Compliance

The kernel follows the NASA/JPL Power of 10 safety-critical rules adapted for modern C++20:

1. **Simple Control Flow (Rule 1):**
   * Acyclic execution paths; no `goto`, `setjmp`, or unstructured jumps.
   * Recursion is forbidden across all algorithms.

2. **Fixed Loop Upper Bounds (Rule 2):**
   * All iterative loops have statically verifiable or compile-time upper bounds.
   * Linear solvers (such as the block-Thomas algorithm) and iterative root-finders feature hard iteration limits with fail-closed return paths.

3. **Zero Heap Churn Post-Initialization (Rule 3):**
   * No dynamic memory allocation (`new`, `malloc`, dynamic resizing) during simulation time-stepping.
   * All scratch buffers and workspace vectors use cache-aligned static arenas (`core/arena.hpp`) or stack allocation.

4. **Short Functions (Rule 4):**
   * Functions are concise, modular, and single-purpose, typically fitting within standard line limits.

5. **High Assertion Density (Rule 5):**
   * Extensive precondition, postcondition, and invariant checks verify physical validity (e.g., positive densities, valid temperatures, non-negative entropy generation).

6. **Minimal Scope (Rule 6):**
   * Variables are declared at the smallest possible scope and marked `const` or `constexpr` wherever practical.

7. **Validated Return Values & Pointers (Rule 7 & 8):**
   * Pointer dereferences are guarded; array indices are checked against grid bounds.

8. **Restricted Preprocessor (Rule 9):**
   * Preprocessor usage is strictly limited to header guards and feature detection.

9. **Pedantic Zero-Warning Compilation (Rule 10):**
   * Clean compilation under strict compiler flags:
     `-Wall -Wextra -Wpedantic -Werror -Werror=non-virtual-dtor -fno-exceptions -fno-rtti -ffp-contract=off -O3`.
