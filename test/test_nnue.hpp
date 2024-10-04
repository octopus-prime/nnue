#pragma once

#include <nnue/nnue.hpp>

namespace nnue {

void test_nnue() {
    using nnue = big_nnue;

    const nnue nnue_;

    const auto version = nnue_.version();
    const auto hash = nnue_.hash();
    const auto description = nnue_.description();

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

    nnue::Accumulator accumulator;
    nnue_.refresh<WHITE>(accumulator, std::span{white_features}.first(4));
    nnue_.refresh<BLACK>(accumulator, std::span{black_features}.first(4));

    const std::int32_t white_score = nnue_.evaluate<WHITE>(accumulator, 4);
    const std::int32_t black_score = nnue_.evaluate<BLACK>(accumulator, 4);

    std::printf("%d\n", white_score);
    std::printf("%d\n", black_score);

    if (white_score != -887)
        throw std::runtime_error{"test_nnue (white) failed"};

    if (black_score != 1317)
        throw std::runtime_error{"test_nnue (black) failed"};
}

}  // namespace nnue
