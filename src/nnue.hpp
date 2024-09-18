#pragma once

#include <fstream>
#include <memory>
#include <network.hpp>
#include <string>

namespace nnue {

struct header {
    std::uint32_t version;
    std::uint32_t hash;
    std::string description;
};

template <std::size_t N>
class nnue {
    using Network = network<N>;

   public:
    constexpr static inline std::size_t L1 = Network::L1;

   private:
    header header_;
    std::unique_ptr<Network> networks[8];

   public:
    nnue();

    nnue(const std::string_view filename) {
        std::ifstream stream{filename.data(), std::ios::binary};

        // read header
        std::uint32_t size;
        stream.read(reinterpret_cast<char*>(&header_.version), sizeof(header_.version));
        stream.read(reinterpret_cast<char*>(&header_.hash), sizeof(header_.hash));
        stream.read(reinterpret_cast<char*>(&size), sizeof(size));
        header_.description.resize(size);
        stream.read(header_.description.data(), size);

        // read networks
        std::ranges::generate(networks, [&]() {
            return std::make_unique<Network>(stream);
        });

        if (stream.fail())
            throw std::runtime_error("failed to read network");
    }

    const header& header() const noexcept {
        return header_;
    }

    const Network& operator[](const std::size_t i) const noexcept {
        return *networks[i];
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
