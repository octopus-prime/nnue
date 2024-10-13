#include <chrono>
#include <nnue/nnue.hpp>

using namespace nnue;
using NNUE = big_nnue;

struct dirty_piece {
    std::uint8_t from;
    std::uint8_t to;
    std::uint8_t piece;
};

class demo_evaluator {
    using Accumulator = NNUE::Accumulator;

    NNUE nnue;

public:
    template <int Perspective>
    std::int32_t evaluate(Accumulator& accumulator, const std::span<const std::uint8_t, 64> board) const noexcept {
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
    std::int32_t evaluate(Accumulator& accumulator, const Accumulator& previous, const std::span<const std::uint8_t, 64> board, const std::span<const dirty_piece> dirty_pieces) const noexcept {
        const auto white_king = std::distance(board.begin(), std::ranges::find(board, W_KING));
        const auto black_king = std::distance(board.begin(), std::ranges::find(board, B_KING));
        const auto piece_count = 64 - std::ranges::count(board, NO_PIECE);

        std::uint16_t added_features[2][3];
        std::uint16_t removed_features[2][3];

        for (auto [index, dp] : std::views::enumerate(dirty_pieces)) {
            removed_features[WHITE][index] = make_index<WHITE>(white_king, dp.from, dp.piece);
            removed_features[BLACK][index] = make_index<BLACK>(black_king, dp.from, dp.piece);
            added_features[WHITE][index] = make_index<WHITE>(white_king, dp.to, dp.piece);
            added_features[BLACK][index] = make_index<BLACK>(black_king, dp.to, dp.piece);
        }

        nnue.update<WHITE>(accumulator, previous, std::span{removed_features[WHITE]}.first(dirty_pieces.size()), std::span{added_features[WHITE]}.first(dirty_pieces.size()));
        nnue.update<BLACK>(accumulator, previous, std::span{removed_features[BLACK]}.first(dirty_pieces.size()), std::span{added_features[BLACK]}.first(dirty_pieces.size()));

        return nnue.evaluate<Perspective>(accumulator, piece_count);
    }
};

void evaluate_nnue() {

    const demo_evaluator evaluator;
    NNUE::Accumulator accumulator[2];

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

    // evaluate with refresh

    {
        const auto evaluate = [&](){
            return evaluator.evaluate<WHITE>(accumulator[0], std::span{board}); 
        };

        const std::int32_t score = evaluate();
        std::printf("score = %d (%.2f pawns)\n", score, score / 208.f);

        constexpr auto N = 1000000;
        std::vector<std::int32_t> scores(N);
        const auto t0 = std::chrono::high_resolution_clock::now();
        std::ranges::generate(scores, evaluate);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const auto t = (t1 - t0) / N;
        std::printf("score = %d (%.2f pawns)\n", scores[0], scores[0] / 208.f);
        std::printf("score = %d (%.2f pawns)\n", scores[N - 1], scores[N - 1] / 208.f);
        std::printf("time = %ldns\n", t.count());
    }

    // make a move

    std::swap(board[SQ_E2], board[SQ_E4]);
    const dirty_piece dirty_pieces[3] = {
        {SQ_E2, SQ_E4, W_PAWN}
    };

    // evaluate with update

    {
        const auto evaluate = [&](){
            return evaluator.evaluate<BLACK>(accumulator[1], accumulator[0], std::span{board}, std::span{dirty_pieces}.first(1));
        };

        const std::int32_t score = evaluate();
        std::printf("score = %d (%.2f pawns)\n", score, score / 208.f);

        constexpr auto N = 1000000;
        std::vector<std::int32_t> scores(N);
        const auto t0 = std::chrono::high_resolution_clock::now();
        std::ranges::generate(scores, evaluate);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const auto t = (t1 - t0) / N;
        std::printf("score = %d (%.2f pawns)\n", scores[0], scores[0] / 208.f);
        std::printf("score = %d (%.2f pawns)\n", scores[N - 1], scores[N - 1] / 208.f);
        std::printf("time = %ldns\n", t.count());
    }
}

int main() {
    try {
        evaluate_nnue();
    } catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());
    }
}
