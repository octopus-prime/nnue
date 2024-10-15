#include <nnue/nnue.hpp>

using namespace nnue;
using NNUE = big_nnue;

#include "dirty_piece.hpp"
#include "position.hpp"
#include "evaluator.hpp"

using namespace demo;

#include <chrono>

template <int Perspective>
bool run_evaluation(const evaluator& evaluator, const position& position, const std::string_view intend) noexcept;

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

    run_evaluation<WHITE>(evaluator, position0, ""sv);

    // make some moves

    for (std::uint8_t from : {SQ_A2, SQ_B2, SQ_C2, SQ_D2, SQ_E2, SQ_F2, SQ_G2, SQ_H2}) {
        const std::uint8_t to = from + 16; // A4 ... H4
        const dirty_piece dirty_pieces[3] = {
            {from, to, W_PAWN}
        };

        std::swap(board[from], board[to]);

        const position position1 {&position0, std::span{board}, std::span{dirty_pieces}.first(1)};

        run_evaluation<BLACK>(evaluator, position1, " "sv);

        for (std::uint8_t from : {SQ_A7, SQ_B7, SQ_C7, SQ_D7, SQ_E7, SQ_F7, SQ_G7, SQ_H7}) {
            const std::uint8_t to = from - 16; // A5 ... H5
            const dirty_piece dirty_pieces[3] = {
                {from, to, B_PAWN}
            };

            std::swap(board[from], board[to]);

            const position position2 {&position1, std::span{board}, std::span{dirty_pieces}.first(1)};

            run_evaluation<WHITE>(evaluator, position2, "  "sv);

            std::swap(board[to], board[from]);
        }

        std::swap(board[to], board[from]);
    }
}

template <int Perspective>
bool run_evaluation(const evaluator& evaluator, const position& position, const std::string_view intend) noexcept {
    constexpr auto pawn = 208.f;

    const auto evaluation = [&](){
        return evaluator.evaluate<Perspective>(position); 
    };

    const std::int32_t score = evaluation();

    constexpr auto N = 1000;
    std::int32_t scores[N];
    const auto t0 = std::chrono::high_resolution_clock::now();
    std::ranges::generate(scores, evaluation);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto t = (t1 - t0) / N;

    std::printf("%sscore = %d (%.2f pawns) (%ldns)\n", intend.data(), score, score / pawn, t.count());

    return scores[0] == score && scores[N - 1] == score;
}

int main() {
    try {
        demo_evaluation();
    } catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());
    }
}
