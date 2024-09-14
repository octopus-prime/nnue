#pragma once

#include <common.hpp>

namespace nnue {

alignas(64) static inline const std::array<std::array<std::uint16_t, 8>, 256> lookup_indices = []() {
    std::array<std::array<std::uint16_t, 8>, 256> v{};
    for (unsigned i = 0; i < 256; ++i) {
        std::uint64_t j = i, k = 0;
        while (j) {
            auto n = std::countr_zero(j);
            v[i][k++] = n;
            j ^= 1ull << n;
        }
    }
    return v;
}();

template <std::size_t I, std::size_t O>
static inline auto get_weight_index_scrambled(std::size_t i) noexcept {
    return (i / 4) % (I / 4) * O * 4 + i / I * 4 + i % 4;
}

template <std::size_t I, std::size_t O>
static auto find_nnz(const std::span<const std::int32_t, I> input, const std::span<std::uint16_t, O> out) noexcept {
    using vec_t = __m256i;
#define vec_nnz(a) _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(a, _mm256_setzero_si256())))

    using vec128_t = __m128i;
#define vec128_zero() _mm_setzero_si128()
#define vec128_set_16(a) _mm_set1_epi16(a)
#define vec128_add(a, b) _mm_add_epi16(a, b)

    constexpr auto InputSimdWidth = sizeof(vec_t) / sizeof(std::int32_t);
    constexpr auto ChunkSize = std::max(InputSimdWidth, 8ul);
    constexpr auto NumChunks = I / ChunkSize;
    constexpr auto InputsPerChunk = ChunkSize / InputSimdWidth;
    constexpr auto OutputsPerChunk = ChunkSize / 8;

    const auto inputVector = span_cast<const vec_t>(input);
    auto count = 0;
    vec128_t base = vec128_zero();
    const vec128_t increment = vec128_set_16(8);
    for (auto i = 0ul; i < NumChunks; ++i) {
        unsigned nnz = 0;
        for (auto j = 0ul; j < InputsPerChunk; ++j) {
            const vec_t inputChunk = inputVector[i * InputsPerChunk + j];
            nnz |= unsigned(vec_nnz(inputChunk)) << (j * InputSimdWidth);
        }
        for (auto j = 0ul; j < OutputsPerChunk; ++j) {
            const auto lookup = (nnz >> (j * 8)) & 0xFF;
            const auto offsets = *reinterpret_cast<const vec128_t*>(&lookup_indices[lookup]);
            *reinterpret_cast<vec128_t*>(&out[count]) = vec128_add(base, offsets);
            count += std::popcount(lookup);
            base = vec128_add(base, increment);
        }
    }
    return out.subspan(0, count);

#undef vec_nnz
#undef vec128_zero
#undef vec128_set_16
#undef vec128_add
}

// column major
template <bool sparse, std::size_t I, std::size_t O>
    requires(I % 4 == 0 && O % (sizeof(__m256i) / sizeof(std::int32_t)) == 0)
void affine_tranform(const std::span<const std::uint8_t, I> input, const std::span<const std::int8_t[O], I> weights, const std::span<const std::int32_t, O> biases, const std::span<std::int32_t, O> output) noexcept {
#define vec_set_32(a) _mm256_set1_epi32(a)
#define vec_add_dpbusd_32(acc, x, y) _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(x, y), _mm256_set1_epi16(1)))

    constexpr auto OutputSimdWidth = sizeof(__m256i) / sizeof(std::int32_t);
    constexpr auto NumChunks = I / 4;
    constexpr auto NumRegs = O / OutputSimdWidth;

    const auto input32 = span_cast<const std::int32_t>(input);
    const auto biasvec = span_cast<const __m256i>(biases);
    const auto outptr = span_cast<__m256i>(output);

    __m256i acc[NumRegs];
    const auto f = [&](auto i) -> void {
        const auto in0 = vec_set_32(input32[i]);
        const auto col0 = span_cast<const __m256i>(std::span{weights[i * 4]});
        for (auto k = 0ul; k < NumRegs; ++k)
            acc[k] = vec_add_dpbusd_32(acc[k], in0, col0[k]);
    };

    if constexpr (sparse) {
        alignas(64) std::uint16_t buf[NumChunks];
        auto nnz = find_nnz(input32, std::span{buf});
        std::ranges::copy(biasvec, acc);
        std::ranges::for_each(nnz, f);
        std::ranges::copy(acc, outptr.begin());
    } else {
        std::ranges::copy(biasvec, acc);
        std::ranges::for_each(std::views::iota(0ul, NumChunks), f);
        std::ranges::copy(acc, outptr.begin());
    }

#undef vec_set_32
#undef vec_add_dpbusd_32

    // for (auto i = 0ul; i < O; ++i)
    //     output[i] = biases[i];
    // for (auto i = 0ul; i < I; ++i)
    //     // if (input[i]) {
    //         for (auto j = 0ul; j < O; ++j)
    //             output[j] += weights[j][i] * input[i];
    //     // }
}

// row major
template <std::size_t I, std::size_t O>
    requires(I % sizeof(__m256i) == 0)
void affine_tranform(const std::span<const std::uint8_t, I> input, const std::span<const std::int8_t[I], O> weights, const std::span<const std::int32_t, O> biases, const std::span<std::int32_t, O> output) noexcept {
    const auto hadd = [](const __m256i sum) -> std::int32_t {
        __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
        sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_BADC));
        sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_CDAB));
        return _mm_cvtsi128_si32(sum128);
    };

    constexpr auto NumChunks = I / sizeof(__m256i);

    const auto in = span_cast<const __m256i>(input);

    for (auto i = 0ul; i < O; ++i) {
        const auto row = span_cast<const __m256i>(std::span{weights[i]});
        __m256i sum = _mm256_setzero_si256();
        for (auto j = 0ul; j < NumChunks; ++j)
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(_mm256_maddubs_epi16(in[j], row[j]), _mm256_set1_epi16(1)));
        output[i] = biases[i] + hadd(sum);
    }

    // for (auto i = 0ul; i < O; ++i)
    //     output[i] = biases[i];
    // for (auto i = 0ul; i < I; ++i)
    //     for (auto j = 0ul; j < O; ++j)
    //         output[j] += weights[j][i] * input[i];
}

}  // namespace nnue
