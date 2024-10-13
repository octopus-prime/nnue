#include <nnue/nnue.hpp>

using namespace nnue;
using NNUE = big_nnue;

#include "dirty_piece.hpp"
#include "position.hpp"
#include "evaluator.hpp"

using namespace demo;

#include <chrono>

template <int Perspective>
void run_evaluation(const evaluator& evaluator, const position& position) noexcept;

void demo_evaluation() {
    const evaluator evaluator;

    // start position

    std::uint8_t board[64] = {
        W_ROOK,   W_KNIGHT, W_BISHOP, W_QUEEN,  W_KING,   W_BISHOP, W_KNIGHT, W_ROOK,
        W_PAWN,   W_PAWN,   W_PAWN,   W_PAWN,   W_PAWN,   W_PAWN,   W_PAWN,   W_PAWN,
        NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE,
        NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE,
        NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE,
        NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE,
        B_PAWN,   B_PAWN,   B_PAWN,   B_PAWN,   B_PAWN,   B_PAWN,   B_PAWN,   B_PAWN,
        B_ROOK,   B_KNIGHT, B_BISHOP, B_QUEEN,  B_KING,   B_BISHOP, B_KNIGHT, B_ROOK
    };

    const position position0 {nullptr, std::span{board}};

    run_evaluation<WHITE>(evaluator, position0);

    // make a move

    std::swap(board[SQ_E2], board[SQ_E4]);
    const dirty_piece dirty_pieces[3] = {
        {SQ_E2, SQ_E4, W_PAWN}
    };

    const position position1 {&position0, std::span{board}, std::span{dirty_pieces}.first(1)};

    run_evaluation<BLACK>(evaluator, position1);
}

template <int Perspective>
void run_evaluation(const evaluator& evaluator, const position& position) noexcept {
    constexpr auto pawn = 208.f;

    const auto evaluation = [&](){
        return evaluator.evaluate<Perspective>(position); 
    };

    const std::int32_t score = evaluation();
    std::printf("score = %d (%.2f pawns)\n", score, score / pawn);

    constexpr auto N = 1000000;
    std::vector<std::int32_t> scores(N);
    const auto t0 = std::chrono::high_resolution_clock::now();
    std::ranges::generate(scores, evaluation);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto t = (t1 - t0) / N;
    std::printf("score = %d (%.2f pawns)\n", scores[0], scores[0] / pawn);
    std::printf("score = %d (%.2f pawns)\n", scores[N - 1], scores[N - 1] / pawn);
    std::printf("time = %ldns\n", t.count());
}

int main() {
    try {
        demo_evaluation();
    } catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());
    }
}
