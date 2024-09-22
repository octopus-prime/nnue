#include <chrono>
#include <nnue.hpp>
#include <print>

void evaluate_nnue() {
    using nnue = nnue::small_nnue;

    std::println("nnue={}", sizeof(nnue));
    std::println("accumulator={}", sizeof(nnue::Accumulator));
    std::println("features={}", sizeof(nnue::Features));
    std::println("networks={}=8*{}", sizeof(nnue::Network) * 8, sizeof(nnue::Network));

    constexpr auto N = nnue::L1;
    constexpr auto Q = 1000;
    const nnue ai{};

    std::println("version = {}", ai.get_header().version);
    std::println("hash = {}", ai.get_header().hash);
    std::println("description = {}", ai.get_header().description);

    nnue::Accumulator accumulator;
    std::ranges::fill(accumulator.accumulation[0], 0);
    std::ranges::fill(accumulator.accumulation[1], 0);

    for (auto j = 0ul; j <= 4; ++j) {
        std::println("input = {}%", 100.0 * j / 4);

        std::ranges::fill_n(accumulator.accumulation[0], j * N / 4, 1000);
        std::ranges::fill_n(accumulator.accumulation[1], j * N / 4, 1000);

        std::int32_t scores[Q];
        const std::int32_t score = ai.evaluate(32, accumulator);
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (auto i = 0; i < Q; ++i)
            scores[i] = ai.evaluate(i % 30 + 2, accumulator);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const auto t = (t1 - t0) / Q;

        std::println("time = {}", t);
        std::println("score = {}", score);
        std::println("score = {}", scores[0]);
        std::println("score = {}", scores[Q-1]);
    }
}

int main() {
    try {
        evaluate_nnue();
    } catch (const std::exception& e) {
        std::println("error: {}", e.what());
    }
}
