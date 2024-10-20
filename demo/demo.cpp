#include <nnue/nnue.hpp>

using namespace nnue;
using NNUE = big_nnue;

#include "dirty_piece.hpp"
#include "position.hpp"
#include "evaluator.hpp"
#include "generator.hpp"
#include "searcher.hpp"

using namespace demo;

#include <chrono>

void run_demo() {
    const searcher searcher;

    constexpr std::uint8_t board[64] = {
        W_ROOK,   W_KNIGHT, W_BISHOP, W_QUEEN,  W_KING,   W_BISHOP, W_KNIGHT, W_ROOK,
        W_PAWN,   W_PAWN,   W_PAWN,   W_PAWN,   W_PAWN,   W_PAWN,   W_PAWN,   W_PAWN,
        NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE,
        NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE,
        NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE,
        NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE,
        B_PAWN,   B_PAWN,   B_PAWN,   B_PAWN,   B_PAWN,   B_PAWN,   B_PAWN,   B_PAWN,
        B_ROOK,   B_KNIGHT, B_BISHOP, B_QUEEN,  B_KING,   B_BISHOP, B_KNIGHT, B_ROOK
    };

    const position position {std::span{board}};

    const auto t0 = std::chrono::high_resolution_clock::now();
    const auto best = searcher.search<WHITE>(position, 2);
    const auto t1 = std::chrono::high_resolution_clock::now();

    std::printf("best = %d\n", best);
    std::printf("time = %ldns\n", (t1 - t0).count());
}

int main() {
    try {
        run_demo();
    } catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());
    }
}
