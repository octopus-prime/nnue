#pragma once

#include <header.hpp>
#include <features.hpp>
#include <network.hpp>
#include <mul_clipped_relu.hpp>

namespace nnue {

template <std::size_t N>
struct accumulator {
    alignas(64) std::int16_t accumulation[2][N];
    bool         computed[2];
};

template <std::size_t N>
class nnue {

   public:
    using Accumulator = accumulator<N>;
    using Features = features<N>;
    using Network = network<N>;

    constexpr static inline std::size_t L1 = Network::L1;

   private:
    std::unique_ptr<header> header_;
    std::unique_ptr<Features> features;
    std::unique_ptr<Network> networks[8];

   public:
    nnue();

    nnue(const std::string_view filename) {
        std::ifstream stream{filename.data(), std::ios::binary};

        header_ = std::make_unique<header>(stream);
        features = std::make_unique<Features>(stream);
        std::ranges::generate(networks, [&]() {
            return std::make_unique<Network>(stream);
        });

        if (!stream || stream.fail() || stream.peek() != std::ios::traits_type::eof())
            throw std::runtime_error("failed to read network");
    }

    const header& get_header() const noexcept {
        return *header_;
    }

    const Features& get_features() const noexcept {
        return *features;
    }

    std::int32_t evaluate(const std::size_t piece_count, const Accumulator& accumulator) const noexcept {
        const auto bucket = (piece_count - 1) / 4;
        alignas(64) std::uint8_t l1clipped[L1];

        mul_clipped_relu(std::span{accumulator.accumulation[0]}, std::span{l1clipped}.template first<L1 / 2>());
        mul_clipped_relu(std::span{accumulator.accumulation[1]}, std::span{l1clipped}.template last<L1 / 2>());

        return networks[bucket]->evaluate(std::span{l1clipped} | std::views::as_const);
    }
};

template <>
nnue<128>::nnue() : nnue{"/home/mike/workspace2/Stockfish/src/nn-37f18f62d772.nnue"} {
}

template <>
nnue<3072>::nnue() : nnue{"/home/mike/workspace2/Stockfish/src/nn-1111cefa1111.nnue"} {
}

using small_nnue = nnue<128>;
using big_nnue = nnue<3072>;

}  // namespace nnue
