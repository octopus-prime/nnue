#pragma once

#include "nnue.hpp"

namespace nnue {

// N = 16
template <std::size_t N>
    requires(N % sizeof(__m128i) == 0)
void sqr_clipped_relu(const std::span<const std::int32_t, N> input, const std::span<std::uint8_t, N> output) noexcept {
    using vec_t = __m128i;

    const auto in = span_cast<const vec_t>(input);
    const auto out = span_cast<vec_t>(output);

    for (auto i = 0ul; i < in.size() / 4; ++i) {
        vec_t words0 = _mm_packs_epi32(in[i * 4 + 0], in[i * 4 + 1]);
        vec_t words1 = _mm_packs_epi32(in[i * 4 + 2], in[i * 4 + 3]);
        words0 = _mm_srli_epi16(_mm_mulhi_epi16(words0, words0), 3);
        words1 = _mm_srli_epi16(_mm_mulhi_epi16(words1, words1), 3);
        out[i] = _mm_packs_epi16(words0, words1);
    }

    // for (auto i = 0ul; i < N; ++i)
    //     output[i] = static_cast<std::uint8_t>(std::min(127ll, ((long long) (input[i]) * input[i]) >> (2 * 6 + 7)));
}

void test_sqr_clipped_relu_16();

}  // namespace nnue
