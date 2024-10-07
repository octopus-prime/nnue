#include <chrono>
#include <nnue/nnue.hpp>

using namespace nnue;

class slow_evaluator {
    using NNUE = big_nnue;

    NNUE nnue;

public:
    template <int Perspective>
    std::int32_t evaluate(const std::span<const int, 64> board) const noexcept {
        const auto white_king = std::distance(board.begin(), std::ranges::find(board, W_KING));
        const auto black_king = std::distance(board.begin(), std::ranges::find(board, B_KING));

        std::uint16_t white_features[32];
        std::uint16_t black_features[32];
        std::size_t piece_count = 0;
        for (auto [square, piece] : std::views::enumerate(board))
            if (piece != NO_PIECE) {
                white_features[piece_count] = make_index<WHITE>(white_king, square, piece);
                black_features[piece_count] = make_index<BLACK>(black_king, square, piece);
                ++piece_count;
            }

        NNUE::Accumulator accumulator;
        nnue.refresh<WHITE>(accumulator, std::span{white_features}.first(piece_count));
        nnue.refresh<BLACK>(accumulator, std::span{black_features}.first(piece_count));

        return nnue.evaluate<Perspective>(accumulator, piece_count);
    }
};

void evaluate_nnue() {
    const slow_evaluator evaluator;

    const int board[64] = {
        W_ROOK,   W_KNIGHT, W_BISHOP, W_QUEEN,  W_KING,   W_BISHOP, W_KNIGHT, W_ROOK,
        W_PAWN,   W_PAWN,   W_PAWN,   W_PAWN,   W_PAWN,   W_PAWN,   W_PAWN,   W_PAWN,
        NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE,
        NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE,
        NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE,
        NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE, NO_PIECE,
        B_PAWN,   B_PAWN,   B_PAWN,   B_PAWN,   B_PAWN,   B_PAWN,   B_PAWN,   B_PAWN,
        B_ROOK,   B_KNIGHT, B_BISHOP, B_QUEEN,  B_KING,   B_BISHOP, B_KNIGHT, B_ROOK
    };

    const auto evaluate = [&](){
        return evaluator.evaluate<WHITE>(std::span{board}); 
    };

    const std::int32_t score = evaluate();
    std::printf("score = %d (%f pawns)\n", score, score / 208.f);

    constexpr auto N = 1000000;
    std::vector<std::int32_t> scores(N);
    const auto t0 = std::chrono::high_resolution_clock::now();
    std::ranges::generate(scores, evaluate);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto t = (t1 - t0) / N;
    std::printf("score = %d (%f pawns)\n", scores[0], scores[0] / 208.f);
    std::printf("score = %d (%f pawns)\n", scores[N - 1], scores[N - 1] / 208.f);
    std::printf("time = %ldns\n", t.count());
}

int main() {
    try {
        evaluate_nnue();
    } catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());
    }
}
