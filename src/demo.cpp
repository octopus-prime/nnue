#include <chrono>
#include <print>

#include "affine_tranform.hpp"
#include "clipped_relu.hpp"
#include "nnue.hpp"
#include "sqr_clipped_relu.hpp"

int main() {
    nnue::test_clipped_relu_16();
    nnue::test_clipped_relu_32();
    nnue::test_sqr_clipped_relu_16();
    nnue::test_affine_tranform_32_1();
    nnue::test_affine_tranform_32_32();

    using network = nnue::big_network;
    constexpr auto N = network::L1;
    constexpr auto Q = 1000000;
    const network net{};

    alignas(64) uint8_t input[N];
    std::ranges::fill(input, 0);
    for (auto i = 0ul; i < N; ++i)
        input[i] = (i + 1) % 127;

    int32_t scores[Q];

    const int32_t score = net.eval(std::span{std::as_const(input)});
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < Q; ++i)
        scores[i] = net.eval(std::span{std::as_const(input)});
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto t = (t1 - t0) / Q;
    std::println("elapsed time: {}", t);
    std::println("score = {}", score);
    std::println("score = {}", scores[0]);
    std::println("score = {}", scores[999999]);
}
