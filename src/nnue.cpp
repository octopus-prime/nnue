#include "nnue.hpp"

#include <chrono>
#include <print>

#include "affine_tranform.hpp"
#include "clipped_relu.hpp"
#include "sqr_clipped_relu.hpp"

namespace nnue {

template <std::size_t N>
network<N>::network() noexcept {
    // constexpr auto get_weight_index_scrambled = [](auto i) { return (i / 4) % (N / 4) * M * 4 + i / N * 4 + i % 4; };

    std::ranges::fill(biases1, 10);

    // 'identity' matrix
    alignas(64) int8_t w1[L2][L1];
    for (auto i = 0ul; i < L2; ++i)
        for (auto j = 0ul; j < L1; ++j)
            w1[i][j] = 100 * (i == j);
    w1[L2 - 3][L1 - 3] = 1;  // seen in accumation ?!
    w1[L2 - 2][L1 - 2] = 1;  // seen in accumation ?!
    w1[L2 - 1][L1 - 1] = 1;  // seen in accumation ?!

    // scramble matrix
    for (auto i = 0ul; i < L1 * L2; i++)
        (&weights1[0][0])[get_weight_index_scrambled<L1, L2>(i)] = (&w1[0][0])[i];

    std::ranges::fill(biases2, 20);

    // 'identity' matrix
    alignas(64) int8_t w2[L3][L2 * 2];
    for (auto i = 0ul; i < L3; ++i)
        for (auto j = 0ul; j < 2 * L2; ++j)
            w2[i][j] = 100 * (i == j);
    // w2[L3-3][L2-3] = 1; // seen in accumation ?!
    // w2[L3-2][L2-2] = 1; // seen in accumation ?!
    // w2[L3-1][L2-1] = 1; // seen in accumation ?!

    // scramble matrix
    for (auto i = 0ul; i < 2 * L2 * L3; i++)
        (&weights2[0][0])[get_weight_index_scrambled<2 * L2, L3>(i)] = (&w2[0][0])[i];

    std::ranges::fill(biases3, 30);

    for (auto i = 0ul; i < L3; ++i)
        weights3[0][i] = i;
}

template <std::size_t N>
std::int32_t network<N>::eval(const std::span<const std::uint8_t, L1> input) const noexcept {
    alignas(64) std::array<std::int32_t, L2> l2a;
    affine_tranform<true>(std::span{input}, std::span{weights1}, std::span{biases1}, std::span{l2a});

    alignas(64) std::array<std::uint8_t, 2 * L2> l2c;
    sqr_clipped_relu(std::span{std::as_const(l2a)}, std::span{l2c}.template first<L2>());
    clipped_relu(std::span{std::as_const(l2a)}, std::span{l2c}.template last<L2>());

    alignas(64) std::array<std::int32_t, L3> l3a;
    affine_tranform<false>(std::span{std::as_const(l2c)}, std::span{weights2}, std::span{biases2}, std::span{l3a});

    alignas(64) std::array<std::uint8_t, L3> l3c;
    clipped_relu(std::span{std::as_const(l3a)}, std::span{l3c});

    alignas(64) std::array<std::int32_t, 1> l4a;
    affine_tranform(std::span{std::as_const(l3c)}, std::span{weights3}, std::span{biases3}, std::span{l4a});

    return l4a[0];
}

}  // namespace nnue

template class nnue::network<128>;
template class nnue::network<3072>;
