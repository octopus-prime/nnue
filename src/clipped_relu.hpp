#pragma once

#include "nnue.hpp"

namespace nnue {

// N = 32
template <std::size_t N>
    requires(N % sizeof(__m256i) == 0)
void clipped_relu(const std::span<const std::int32_t, N> input, const std::span<std::uint8_t, N> output) noexcept {
    using vec_t = __m256i;

    const auto in = span_cast<const vec_t>(input);
    const auto out = span_cast<vec_t>(output);

    for (auto i = 0ul; i < in.size() / 4; ++i) {
        const vec_t words0 = _mm256_srli_epi16(_mm256_packus_epi32(in[i * 4 + 0], in[i * 4 + 1]), 6);
        const vec_t words1 = _mm256_srli_epi16(_mm256_packus_epi32(in[i * 4 + 2], in[i * 4 + 3]), 6);
        out[i] = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(words0, words1), _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0));
    }

    // for (auto i = 0ul; i < N; ++i)
    //     output[i] = static_cast<std::uint8_t>(std::clamp(input[i] >> 6, 0, 127));
}

// N = 16
template <std::size_t N>
    requires(N % sizeof(__m256i) != 0 && N % sizeof(__m128i) == 0)
void clipped_relu(const std::span<const std::int32_t, N> input, const std::span<std::uint8_t, N> output) noexcept {
    using vec_t = __m128i;

    const auto in = span_cast<const vec_t>(input);
    const auto out = span_cast<vec_t>(output);

    for (auto i = 0ul; i < in.size() / 4; ++i) {
        const vec_t words0 = _mm_srli_epi16(_mm_packus_epi32(in[i * 4 + 0], in[i * 4 + 1]), 6);
        const vec_t words1 = _mm_srli_epi16(_mm_packus_epi32(in[i * 4 + 2], in[i * 4 + 3]), 6);
        out[i] = _mm_packs_epi16(words0, words1);
    }

    // for (auto i = 0ul; i < N; ++i)
    //     output[i] = static_cast<std::uint8_t>(std::clamp(input[i] >> 6, 0, 127));
}

void test_clipped_relu_16();
void test_clipped_relu_32();

}  // namespace nnue
