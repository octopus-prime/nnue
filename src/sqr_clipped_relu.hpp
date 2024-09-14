#pragma once

#include <common.hpp>

namespace nnue {

// N = 16
template <std::size_t N>
    requires(N % sizeof(__m128i) == 0)
void sqr_clipped_relu(const std::span<const std::int32_t, N> input, const std::span<std::uint8_t, N> output) noexcept {
    const auto in = span_cast<const __m128i>(input) | std::views::chunk(4);
    const auto out = span_cast<__m128i>(output);

    std::ranges::transform(in, out.begin(), [](auto&& chunk) {
        __m128i words0 = _mm_packs_epi32(chunk[0], chunk[1]);
        __m128i words1 = _mm_packs_epi32(chunk[2], chunk[3]);
        words0 = _mm_srli_epi16(_mm_mulhi_epi16(words0, words0), 3);
        words1 = _mm_srli_epi16(_mm_mulhi_epi16(words1, words1), 3);
        return _mm_packs_epi16(words0, words1);
    });

    // for (auto i = 0ul; i < N; ++i)
    //     output[i] = static_cast<std::uint8_t>(std::min(127ll, ((long long) (input[i]) * input[i]) >> (2 * 6 + 7)));
}

}  // namespace nnue
