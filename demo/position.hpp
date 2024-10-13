#pragma once

namespace demo {

struct position {
    const position* previous;
    std::uint8_t board[64];
    std::size_t dirty_pieces_count;
    dirty_piece dirty_pieces[3];
    mutable NNUE::Accumulator accumulator;

    position(const position* previous, const std::span<const std::uint8_t, 64> board) : previous{previous}, dirty_pieces_count{0} {
        std::ranges::copy(board, this->board);
    }

    position(const position* previous, const std::span<const std::uint8_t, 64> board, const std::span<const dirty_piece> dirty_pieces) : previous{previous}, dirty_pieces_count{dirty_pieces.size()} {
        std::ranges::copy(board, this->board);
        std::ranges::copy(dirty_pieces, this->dirty_pieces);
    }
};

}  // namespace demo
