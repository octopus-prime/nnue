#include <chrono>
#include <nnue/nnue.hpp>

void evaluate_nnue() {
    using namespace nnue;
    using nnue = big_nnue;

    const nnue ai;

    std::printf("version = %d\n", ai.version());
    std::printf("hash = %d\n", ai.hash());
    std::printf("description = %s\n", ai.description().data());

    const auto evaluate = [&ai](){
        std::uint16_t white_features[32] = {
            make_index<WHITE>(SQ_A1, SQ_C2, W_PAWN),
            make_index<WHITE>(SQ_A1, SQ_D4, B_ROOK)};
        std::uint16_t black_features[32] = {
            make_index<BLACK>(SQ_B8, SQ_C2, W_PAWN),
            make_index<BLACK>(SQ_B8, SQ_D4, B_ROOK)};

        nnue::Accumulator accumulator;
        ai.refresh<WHITE>(accumulator, std::span{white_features}.first(2));
        ai.refresh<BLACK>(accumulator, std::span{black_features}.first(2));

        return ai.evaluate<WHITE>(accumulator, 4);
    };

    const std::int32_t score = evaluate();

    constexpr auto N = 1000000;
    std::vector<std::int32_t> scores(N);
    const auto t0 = std::chrono::high_resolution_clock::now();
    std::ranges::generate(scores, evaluate);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto t = (t1 - t0) / N;

    std::printf("time = %ldns\n", t.count());
    std::printf("score = %d\n", score);
    std::printf("score = %d\n", scores[0]);
    std::printf("score = %d\n", scores[N - 1]);
}

int main() {
    try {
        evaluate_nnue();
    } catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());
    }
}
