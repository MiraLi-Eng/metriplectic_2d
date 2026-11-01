#ifndef METRIPLECTIC_2D_CORE_PRNG_HPP
#define METRIPLECTIC_2D_CORE_PRNG_HPP

#include <cmath>
#include <concepts>
#include <cstdint>
#include <numbers>

namespace metriplectic_2d::core {

// ============================================================================
// StatelessPRNGPort2D Concept & Counter-Based Normal Generator (MEM-2D-002)
// ============================================================================

template <typename T>
concept StatelessPRNGPort2D = requires(
    const T& prng,
    uint64_t seed,
    uint64_t timestep,
    uint32_t ark_stage,
    uint64_t face_linear_id
) {
    { prng.sample_normal(seed, timestep, ark_stage, face_linear_id) } noexcept -> std::same_as<double>;
};

struct StatelessCounterPRNG2D {
    [[nodiscard]] static constexpr uint64_t splitmix64(uint64_t x) noexcept {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31U);
    }

    [[nodiscard]] static constexpr uint64_t hash_key(
        uint64_t seed,
        uint64_t timestep,
        uint32_t ark_stage,
        uint64_t face_linear_id,
        uint64_t stream_offset
    ) noexcept {
        uint64_t h = splitmix64(seed ^ 0xD1B54A32D192ED03ULL);
        h = splitmix64(h ^ timestep);
        h = splitmix64(h ^ (static_cast<uint64_t>(ark_stage) * 0x94D049BB133111EBULL));
        h = splitmix64(h ^ face_linear_id);
        return splitmix64(h ^ stream_offset);
    }

    // Pure function returning N(0, 1) via branchless Box-Muller transform
    [[nodiscard]] double sample_normal(
        uint64_t seed,
        uint64_t timestep,
        uint32_t ark_stage,
        uint64_t face_linear_id
    ) const noexcept {
        const uint64_t bits_u1 = hash_key(seed, timestep, ark_stage, face_linear_id, 0ULL);
        const uint64_t bits_u2 = hash_key(seed, timestep, ark_stage, face_linear_id, 1ULL);

        // Strictly positive uniform u1 in (0, 1] to prevent log(0)
        const double u1 = (static_cast<double>(bits_u1 >> 11U) + 1.0) * 0x1.0p-53;
        const double u2 = static_cast<double>(bits_u2 >> 11U) * 0x1.0p-53;

        const double radius = std::sqrt(-2.0 * std::log(u1));
        const double theta  = 2.0 * std::numbers::pi * u2;
        return radius * std::cos(theta);
    }
};

static_assert(StatelessPRNGPort2D<StatelessCounterPRNG2D>);

} // namespace metriplectic_2d::core

#endif // METRIPLECTIC_2D_CORE_PRNG_HPP
