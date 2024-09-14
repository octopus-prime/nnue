#include <chrono>
#include <nnue.hpp>
#include <print>

int main() {
    using network = nnue::big_network;
    constexpr auto N = network::L1;
    constexpr auto Q = 1000000;
    const network net{};

    alignas(64) std::uint8_t input[N];
    std::ranges::fill(input, 0);
    for (auto i = 0ul; i < N; ++i)
        input[i] = (i + 1) % 127;
    std::int32_t scores[Q];

    const std::int32_t score = net.eval(std::span{std::as_const(input)});
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
