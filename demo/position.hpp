#pragma once

namespace demo {

class position {
    const position* previous;
    std::uint8_t board[64];
    std::size_t dirty_pieces_count;
    dirty_piece dirty_pieces[3];
    mutable NNUE::Accumulator accumulator;

    auto get_dirty_pieces() const noexcept {
        return std::span{dirty_pieces}.first(dirty_pieces_count);
    }

    auto find_white_king() const noexcept {
        return std::distance(board, std::ranges::find(board, W_KING));
    }

    auto find_black_king() const noexcept {
        return std::distance(board, std::ranges::find(board, B_KING));
    }

public:
    position(const position* previous, const std::span<const std::uint8_t, 64> board) : previous{previous}, dirty_pieces_count{0} {
        std::ranges::copy(board, this->board);
    }

    position(const position* previous, const std::span<const std::uint8_t, 64> board, const std::span<const dirty_piece> dirty_pieces) : previous{previous}, dirty_pieces_count{dirty_pieces.size()} {
        std::ranges::copy(board, this->board);
        std::ranges::copy(dirty_pieces, this->dirty_pieces);
    }

    bool has_king_moved() const noexcept {
        return std::ranges::any_of(get_dirty_pieces(), [](const auto& piece) { return piece == W_KING || piece == B_KING; }, &dirty_piece::piece);
    }

    bool has_previous() const noexcept {
        return previous != nullptr;
    }

    const position& get_previous() const noexcept {
        return *previous;
    }

    NNUE::Accumulator& get_accumulator() const noexcept {
        return accumulator;
    }

    auto count_pieces() const noexcept {
        return 64ul - std::ranges::count(board, NO_PIECE);
    }

    auto get_active_features(const std::span<std::uint16_t[32], 2> active_features) const noexcept {
        const auto white_king = find_white_king();
        const auto black_king = find_black_king();

        std::size_t active_count = 0;

        for (auto&& [square, piece] : std::views::enumerate(board))
            if (piece != NO_PIECE) {
                active_features[WHITE][active_count] = make_index<WHITE>(white_king, square, piece);
                active_features[BLACK][active_count] = make_index<BLACK>(black_king, square, piece);
                ++active_count;
            }

        return std::to_array({std::span{active_features[WHITE]}.first(active_count), std::span{active_features[BLACK]}.first(active_count)});
    }

    auto get_changed_features(const std::span<std::uint16_t[3], 2> removed_features, const std::span<std::uint16_t[3], 2> added_features) const noexcept {
        const auto white_king = find_white_king();
        const auto black_king = find_black_king();

        std::size_t removed_count = 0;
        std::size_t added_count = 0;

        for (const auto& dirty_piece : get_dirty_pieces()) {
            if (dirty_piece.from != NO_PIECE) {
                removed_features[WHITE][removed_count] = make_index<WHITE>(white_king, dirty_piece.from, dirty_piece.piece);
                removed_features[BLACK][removed_count] = make_index<BLACK>(black_king, dirty_piece.from, dirty_piece.piece);
                ++removed_count;
            }
            if (dirty_piece.to != NO_PIECE) {
                added_features[WHITE][added_count] = make_index<WHITE>(white_king, dirty_piece.to, dirty_piece.piece);
                added_features[BLACK][added_count] = make_index<BLACK>(black_king, dirty_piece.to, dirty_piece.piece);
                ++added_count;
            }
        }

        return std::tuple {
            std::to_array({std::span{removed_features[WHITE]}.first(removed_count), std::span{removed_features[BLACK]}.first(removed_count)}),
            std::to_array({std::span{added_features[WHITE]}.first(added_count), std::span{added_features[BLACK]}.first(added_count)})
        };
    }
};

}  // namespace demo
