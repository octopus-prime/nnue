#pragma once

#include <nnue/nnue.hpp>

namespace nnue {

void test_nnue() {
    using namespace nnue;
    using nnue = big_nnue;

    const nnue ai;

    constexpr std::uint16_t white_features[32] = {
        make_index<WHITE>(SQ_A1, SQ_A1, W_KING),
        make_index<WHITE>(SQ_A1, SQ_C3, W_PAWN),
        make_index<WHITE>(SQ_A1, SQ_B8, B_KING),
        make_index<WHITE>(SQ_A1, SQ_D4, B_ROOK)
    };
    constexpr std::uint16_t black_features[32] = {
        make_index<BLACK>(SQ_B8, SQ_B8, B_KING),
        make_index<BLACK>(SQ_B8, SQ_D4, B_ROOK),
        make_index<BLACK>(SQ_B8, SQ_A1, W_KING),
        make_index<BLACK>(SQ_B8, SQ_C3, W_PAWN)
    };

    nnue::Accumulator accumulator;
    ai.refresh<WHITE>(accumulator, std::span{white_features}.first(4));
    ai.refresh<BLACK>(accumulator, std::span{black_features}.first(4));

    const std::int32_t white_score = ai.evaluate<WHITE>(accumulator, 4);
    const std::int32_t black_score = ai.evaluate<BLACK>(accumulator, 4);

    std::printf("%d\n", white_score);
    std::printf("%d\n", black_score);

    if (white_score != -887)
        throw std::runtime_error{"test_nnue (white) failed"};

    if (black_score != 1317)
        throw std::runtime_error{"test_nnue (black) failed"};
}

}  // namespace nnue
