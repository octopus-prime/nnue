#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cstdint>
#include <ranges>
#include <span>
#include <utility>

namespace nnue {

template <std::size_t N>
class network {
   public:
    constexpr static inline std::size_t L1 = N;

   private:
    constexpr static inline std::size_t L2 = 16;
    constexpr static inline std::size_t L3 = 32;

    alignas(64) std::int8_t weights1[L1][L2];
    alignas(64) std::int32_t biases1[L2];

    alignas(64) std::int8_t weights2[2 * L2][L3];
    alignas(64) std::int32_t biases2[L3];

    alignas(64) std::int8_t weights3[1][L3];
    alignas(64) std::int32_t biases3[1];

   public:
    network() noexcept;

    std::int32_t eval(const std::span<const std::uint8_t, L1> input) const noexcept;
};

using small_network = network<128>;
using big_network = network<3072>;

template <typename T, typename U, std::size_t I>
static inline auto span_cast(const std::span<U, I> span) noexcept {
    static_assert(sizeof(T) % sizeof(U) == 0);
    const auto O = I / (sizeof(T) / sizeof(U));
    return std::span<T, O>{reinterpret_cast<T*>(span.data()), O};
}

}  // namespace nnue
