#pragma once

namespace demo {

class evaluator {
    using Accumulator = NNUE::Accumulator;

    NNUE nnue;

    template <int Perspective>
    std::int32_t evaluate(const std::span<const std::uint8_t, 64> board, Accumulator& accumulator) const noexcept {
        const auto white_king = std::distance(board.begin(), std::ranges::find(board, W_KING));
        const auto black_king = std::distance(board.begin(), std::ranges::find(board, B_KING));

        std::uint16_t active_features[2][32];
        std::size_t piece_count = 0;

        for (auto [square, piece] : std::views::enumerate(board))
            if (piece != NO_PIECE) {
                active_features[WHITE][piece_count] = make_index<WHITE>(white_king, square, piece);
                active_features[BLACK][piece_count] = make_index<BLACK>(black_king, square, piece);
                ++piece_count;
            }

        nnue.refresh<WHITE>(accumulator, std::span{active_features[WHITE]}.first(piece_count));
        nnue.refresh<BLACK>(accumulator, std::span{active_features[BLACK]}.first(piece_count));

        return nnue.evaluate<Perspective>(accumulator, piece_count);
    }

    template <int Perspective>
    std::int32_t evaluate(const std::span<const std::uint8_t, 64> board, Accumulator& accumulator, const Accumulator& previous, const std::span<const dirty_piece> dirty_pieces) const noexcept {
        const auto white_king = std::distance(board.begin(), std::ranges::find(board, W_KING));
        const auto black_king = std::distance(board.begin(), std::ranges::find(board, B_KING));
        const auto piece_count = 64 - std::ranges::count(board, NO_PIECE);

        std::uint16_t added_features[2][3];
        std::uint16_t removed_features[2][3];

        for (auto [index, dirty_piece] : std::views::enumerate(dirty_pieces)) {
            // note: just demo code - no caputres, promotions, rochades, king moves, etc.
            removed_features[WHITE][index] = make_index<WHITE>(white_king, dirty_piece.from, dirty_piece.piece);
            removed_features[BLACK][index] = make_index<BLACK>(black_king, dirty_piece.from, dirty_piece.piece);
            added_features[WHITE][index] = make_index<WHITE>(white_king, dirty_piece.to, dirty_piece.piece);
            added_features[BLACK][index] = make_index<BLACK>(black_king, dirty_piece.to, dirty_piece.piece);
        }

        nnue.update<WHITE>(accumulator, previous, std::span{removed_features[WHITE]}.first(dirty_pieces.size()), std::span{added_features[WHITE]}.first(dirty_pieces.size()));
        nnue.update<BLACK>(accumulator, previous, std::span{removed_features[BLACK]}.first(dirty_pieces.size()), std::span{added_features[BLACK]}.first(dirty_pieces.size()));

        return nnue.evaluate<Perspective>(accumulator, piece_count);
    }

public:
    template <int Perspective>
    std::int32_t evaluate(const position& position) const noexcept {
        if (position.previous != nullptr)
            return evaluate<Perspective>(std::span{position.board}, position.accumulator, position.previous->accumulator, std::span{position.dirty_pieces}.first(position.dirty_pieces_count));
        else
            return evaluate<Perspective>(std::span{position.board}, position.accumulator);
    }
};

}  // namespace demo
