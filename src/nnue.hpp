#pragma once

#include <affine_tranform.hpp>
#include <clipped_relu.hpp>
#include <sqr_clipped_relu.hpp>

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

template <std::size_t N>
network<N>::network() noexcept {
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
    alignas(64) std::int32_t l2transformed[L2];
    alignas(64) std::uint8_t l2clipped[2 * L2];
    alignas(64) std::int32_t l3transformed[L3];
    alignas(64) std::uint8_t l3clipped[L3];
    alignas(64) std::int32_t l4transformed[1];

    affine_tranform<true>(std::span{std::as_const(input)}, std::span{weights1}, std::span{biases1}, std::span{l2transformed});
    sqr_clipped_relu(std::span{std::as_const(l2transformed)}, std::span{l2clipped}.template first<L2>());
    clipped_relu(std::span{std::as_const(l2transformed)}, std::span{l2clipped}.template last<L2>());
    affine_tranform<false>(std::span{std::as_const(l2clipped)}, std::span{weights2}, std::span{biases2}, std::span{l3transformed});
    clipped_relu(std::span{std::as_const(l3transformed)}, std::span{l3clipped});
    affine_tranform(std::span{std::as_const(l3clipped)}, std::span{weights3}, std::span{biases3}, std::span{l4transformed});

    return l4transformed[0];
}

}  // namespace nnue
