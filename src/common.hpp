#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <ranges>
#include <span>

namespace nnue {

template <typename T, typename U, std::size_t I>
static inline auto span_cast(const std::span<U, I> span) noexcept {
    static_assert(sizeof(T) % sizeof(U) == 0);
    const auto O = I / (sizeof(T) / sizeof(U));
    return std::span<T, O>{reinterpret_cast<T*>(span.data()), O};
}

}  // namespace nnue
