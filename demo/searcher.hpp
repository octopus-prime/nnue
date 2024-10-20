#pragma once

namespace demo {

class searcher {
    generator generator;
    evaluator evaluator;

public:
    template <int Perspective>
    std::int32_t search(const position& position, std::int32_t alpha, std::int32_t beta, std::int32_t depth) const noexcept {
        if (depth == 0)
            return evaluator.evaluate<Perspective>(position);

        dirty_piece dirty_pieces[8];
        generator.generate<Perspective>(dirty_pieces);

        for (const dirty_piece& dirty_piece : dirty_pieces) {
            const class position successor{std::addressof(position), std::span{std::addressof(dirty_piece), 1}};
            const int score = -search<1 - Perspective>(successor, -beta, -alpha, depth - 1);

            if (score >= beta)
                return beta;
            if (score > alpha)
                alpha = score;
        }

        return alpha;
    }

    template <int Perspective>
    std::int32_t search(const position& position, std::int32_t depth) const noexcept {
        constexpr std::int32_t alpha = -1000000;
        constexpr std::int32_t beta = +1000000;
        std::int32_t score = 0;

        for (std::int32_t iteration = 0; iteration <= depth; ++iteration) {
            score = search<Perspective>(position, alpha, beta, iteration);
            std::printf("score[%d] = %d\n", iteration, score);
        }

        return score;
    }
};

}  // namespace demo
