#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <ranges>
#include <span>
#include <fstream>
#include <memory>

namespace nnue {

using namespace std::literals;

constexpr std::string_view big_nnue_filename = "/home/mike/workspace2/Stockfish/src/nn-1111cefa1111.nnue"sv;
constexpr std::string_view small_nnue_filename = "/home/mike/workspace2/Stockfish/src/nn-37f18f62d772.nnue"sv;

template <typename T, typename U, std::size_t I>
    requires(sizeof(T) % sizeof(U) == 0)
auto span_cast(const std::span<U, I> span) noexcept {
    constexpr auto O = I / (sizeof(T) / sizeof(U));
    return std::span<T, O>{reinterpret_cast<T*>(span.data()), O};
}

}  // namespace nnue
