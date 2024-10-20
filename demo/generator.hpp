#pragma once

namespace demo {

// a toy generator that generates some pawn moves for the first 2 plies
class generator {
public:
    template <int Perspective>
    void generate(const std::span<dirty_piece, 8> dirty_pieces) const noexcept {
        if constexpr (Perspective == WHITE) {
            constexpr std::uint8_t squares[] = {SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2};
            std::ranges::transform(squares, dirty_pieces.begin(), [&](std::uint8_t from) {
                const std::uint8_t to = from + 16;
                return dirty_piece{from, to, W_PAWN};
            });
        } else {
            constexpr std::uint8_t squares[] = {SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7};
            std::ranges::transform(squares, dirty_pieces.begin(), [&](std::uint8_t from) {
                const std::uint8_t to = from - 16;
                return dirty_piece{from, to, B_PAWN};
            });
        }
    }
};

}  // namespace demo
