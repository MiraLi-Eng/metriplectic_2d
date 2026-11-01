#ifndef METRIPLECTIC_2D_CORE_ARENA_HPP
#define METRIPLECTIC_2D_CORE_ARENA_HPP

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

namespace metriplectic_2d::core {

// ============================================================================
// 64-Byte Cache-Aligned Static Memory Arena (ADR-2D-009, MEM-2D-001)
// ============================================================================
// Pre-allocates a single contiguous 64B-aligned memory block prior to
// time-stepping and hands out cache-aligned std::span views. Once lock() is
// called, any further sub-allocation fails closed (returning an empty span).

class StaticArena2D {
public:
    static constexpr size_t kCacheLineBytes = 64;

    StaticArena2D() noexcept = default;

    explicit StaticArena2D(size_t capacity_bytes) noexcept {
        (void)initialize(capacity_bytes);
    }

    ~StaticArena2D() noexcept {
        if (base_ptr_ != nullptr) {
            std::free(base_ptr_);
            base_ptr_ = nullptr;
        }
    }

    StaticArena2D(const StaticArena2D&) = delete;
    StaticArena2D& operator=(const StaticArena2D&) = delete;

    [[nodiscard]] bool initialize(size_t capacity_bytes) noexcept {
        if (base_ptr_ != nullptr || capacity_bytes == 0) {
            return false;
        }
        const size_t aligned_cap = align_up(capacity_bytes, kCacheLineBytes);
        void* raw = nullptr;
        if (posix_memalign(&raw, kCacheLineBytes, aligned_cap) != 0 || raw == nullptr) {
            return false;
        }
        base_ptr_ = static_cast<std::byte*>(raw);
        capacity_bytes_ = aligned_cap;
        offset_bytes_ = 0;
        locked_ = false;
        return true;
    }

    template <typename T>
    [[nodiscard]] std::span<T> allocate_aligned(size_t count) noexcept {
        if (locked_ || base_ptr_ == nullptr || count == 0) {
            return {};
        }
        const size_t aligned_offset = align_up(offset_bytes_, kCacheLineBytes);
        const size_t bytes_needed = count * sizeof(T);
        if (aligned_offset + bytes_needed > capacity_bytes_) {
            return {};
        }
        T* ptr = reinterpret_cast<T*>(base_ptr_ + aligned_offset);
        offset_bytes_ = aligned_offset + bytes_needed;
        for (size_t i = 0; i < count; ++i) {
            ptr[i] = T{};
        }
        return std::span<T>(ptr, count);
    }

    void lock() noexcept {
        locked_ = true;
    }

    void reset_unlocked() noexcept {
        offset_bytes_ = 0;
        locked_ = false;
    }

    [[nodiscard]] constexpr bool is_locked() const noexcept { return locked_; }
    [[nodiscard]] constexpr size_t capacity_bytes() const noexcept { return capacity_bytes_; }
    [[nodiscard]] constexpr size_t used_bytes() const noexcept { return offset_bytes_; }
    [[nodiscard]] constexpr size_t remaining_bytes() const noexcept {
        return capacity_bytes_ - offset_bytes_;
    }

private:
    [[nodiscard]] static constexpr size_t align_up(size_t value, size_t alignment) noexcept {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    std::byte* base_ptr_ = nullptr;
    size_t capacity_bytes_ = 0;
    size_t offset_bytes_ = 0;
    bool locked_ = false;
};

} // namespace metriplectic_2d::core

#endif // METRIPLECTIC_2D_CORE_ARENA_HPP
