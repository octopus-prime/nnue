#pragma once

#include <nnue/nnue.hpp>

namespace nnue {

void test_nnue() {
    using NNUE = big_nnue;

    const NNUE nnue;

    const auto version = nnue.version();
    const auto hash = nnue.hash();
    const auto description = nnue.description();

    std::printf("%d\n", version);
    std::printf("%d\n", hash);
    std::printf("%s\n", description.data());

    if (version != 2062757664)
        throw std::runtime_error{"test_nnue (version) failed"};

    if (hash != 470819058)
        throw std::runtime_error{"test_nnue (hash) failed"};
    
    if (!description.ends_with("with the https://github.com/official-stockfish/nnue-pytorch trainer."))
        throw std::runtime_error{"test_nnue (description) failed"};

    const std::uint16_t white_features[32] = {
        make_index<WHITE>(SQ_A1, SQ_A1, W_KING),
        make_index<WHITE>(SQ_A1, SQ_C3, W_PAWN),
        make_index<WHITE>(SQ_A1, SQ_B8, B_KING),
        make_index<WHITE>(SQ_A1, SQ_D4, B_ROOK)
    };
    const std::uint16_t black_features[32] = {
        make_index<BLACK>(SQ_B8, SQ_B8, B_KING),
        make_index<BLACK>(SQ_B8, SQ_D4, B_ROOK),
        make_index<BLACK>(SQ_B8, SQ_A1, W_KING),
        make_index<BLACK>(SQ_B8, SQ_C3, W_PAWN)
    };

    NNUE::Accumulator accumulator;
    nnue.refresh<WHITE>(accumulator, std::span{white_features}.first(4));
    nnue.refresh<BLACK>(accumulator, std::span{black_features}.first(4));

    const std::int32_t white_score = nnue.evaluate<WHITE>(accumulator, 4);
    const std::int32_t black_score = nnue.evaluate<BLACK>(accumulator, 4);

    std::printf("%d\n", white_score);
    std::printf("%d\n", black_score);

    if (white_score != -887)
        throw std::runtime_error{"test_nnue (white) failed"};

    if (black_score != 1317)
        throw std::runtime_error{"test_nnue (black) failed"};
}

}  // namespace nnue
