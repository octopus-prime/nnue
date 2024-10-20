#pragma once

namespace demo {

class evaluator {
    using Accumulator = NNUE::Accumulator;

    NNUE nnue;

    void refresh(const position& position) const noexcept {
        auto& accumulator = position.get_accumulator();

        std::uint16_t active_features_buffer[2][32];

        const auto&& active_features = position.get_active_features(active_features_buffer);

        nnue.refresh<WHITE>(accumulator, active_features[WHITE]);
        nnue.refresh<BLACK>(accumulator, active_features[BLACK]);
    }

    void update(const position& position) const noexcept {
        auto& accumulator = position.get_accumulator();
        const auto& previous = position.get_previous().get_accumulator();

        std::uint16_t removed_features_buffer[2][3];
        std::uint16_t added_features_buffer[2][3];

        const auto&& [removed_features, added_features] = position.get_changed_features(removed_features_buffer, added_features_buffer);

        nnue.update<WHITE>(accumulator, previous, removed_features[WHITE], added_features[WHITE]);
        nnue.update<BLACK>(accumulator, previous, removed_features[BLACK], added_features[BLACK]);
    }

    void prepare(const position& position) const noexcept {
        auto& accumulator = position.get_accumulator();
        if (accumulator.computed)
            return;

        if (!position.has_previous() || position.has_king_moved()) {
            refresh(position);
        } else {
            prepare(position.get_previous());
            update(position);
        }

        accumulator.computed = true;
    }

public:
    template <int Perspective>
    std::int32_t evaluate(const position& position) const noexcept {
        prepare(position);
        return nnue.evaluate<Perspective>(position.get_accumulator(), position.count_pieces());
    }
};

}  // namespace demo
