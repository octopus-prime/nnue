#include "test_mul_clipped_relu.hpp"
#include <nnue/mul_clipped_relu.hpp>

namespace nnue {

void test_mul_clipped_relu_64() {
    constexpr auto N = 64;
    alignas(N) const std::int16_t input[2 * N] = {1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201, 211, 221, 231, 241, 251, 261, 271, 281, 291, 301, 311, 1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201, 211, 221, 231, 241, 251, 261, 271, 281, 291, 301, 311, 1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201, 211, 221, 231, 241, 251, 261, 271, 281, 291, 301, 311, 1, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201, 211, 221, 231, 241, 251, 261, 271, 281, 291, 301, 311};
    alignas(N) std::uint8_t output[N];
    alignas(N) const std::uint8_t expected[N] = {0, 0, 0, 1, 3, 5, 7, 9, 50, 57, 63, 71, 78, 86, 95, 104, 12, 16, 19, 24, 28, 33, 38, 44, 113, 123, 126, 126, 126, 126, 126, 126, 0, 0, 0, 1, 3, 5, 7, 9, 50, 57, 63, 71, 78, 86, 95, 104, 12, 16, 19, 24, 28, 33, 38, 44, 113, 123, 126, 126, 126, 126, 126, 126};

    mul_clipped_relu(std::span{input}, std::span{output});

    for (auto i = 0ul; i < N; ++i)
        std::printf("%d ", output[i]);
    std::printf("\n");

    if (!std::ranges::equal(output, expected))
        throw std::runtime_error{"test_mul_clipped_relu_64 failed"};
}

}  // namespace nnue
