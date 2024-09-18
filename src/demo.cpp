#include <chrono>
#include <nnue.hpp>
#include <print>

void eval_network() {
    using network = nnue::big_network;
    constexpr auto N = network::L1;
    constexpr auto Q = 1000000;
    const network net{};

    alignas(64) std::uint8_t input[N];
    std::ranges::fill(input, 0);
    for (auto i = 0ul; i < N; ++i)
        input[i] = (i + 1) % 127;
    std::int32_t scores[Q];

    const std::int32_t score = net.eval(std::span{input} | std::views::as_const);
    const auto t0 = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < Q; ++i)
        scores[i] = net.eval(std::span{input} | std::views::as_const);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto t = (t1 - t0) / Q;

    std::println("elapsed time: {}", t);
    std::println("score = {}", score);
    std::println("score = {}", scores[0]);
    std::println("score = {}", scores[999999]);
}

void eval_nnue() {
    using nnue = nnue::small_nnue;
    constexpr auto N = nnue::L1;
    constexpr auto Q = 2000000;
    const nnue net{};

    std::println("version = {}", net.header().version);
    std::println("hash = {}", net.header().hash);
    std::println("description = {}", net.header().description);

    alignas(64) std::uint8_t input[N];
    std::int32_t scores[Q];

    std::ranges::fill(input, 0);

    for (auto j = 0ul; j <= 4; ++j) {
        std::println("input = {}%", 100.0 * j / 4);

        for (auto i = 0ul; i < N * j / 4; ++i)
            input[i] = (i + 1) % 127;

        const std::int32_t score = net[0].eval(std::span{input} | std::views::as_const);
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < Q; ++i)
            scores[i] = net[i % 8].eval(std::span{input} | std::views::as_const);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const auto t = (t1 - t0) / Q;

        std::println("time = {}", t);
        std::println("score = {}", score);
        std::println("score = {}", scores[0]);
        std::println("score = {}", scores[999999]);
    }
}

int main() {
    eval_network();
    eval_nnue();
}
