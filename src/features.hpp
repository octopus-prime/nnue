#pragma once

#include <common.hpp>

namespace nnue {

// Read N signed integers from the stream s, putting them in the array out.
// The stream is assumed to be compressed using the signed LEB128 format.
// See https://en.wikipedia.org/wiki/LEB128 for a description of the compression scheme.

template<typename T, std::size_t N>
    requires std::is_signed_v<T>
inline void read_leb_128(std::istream& stream, const std::span<T, N> buffer) {

    // Check the presence of our LEB128 magic string
    char leb128MagicString[17];
    stream.read(leb128MagicString, 17);

    if (std::string_view{"COMPRESSED_LEB128"} != std::string_view{leb128MagicString})
        throw std::runtime_error(std::string{"no 'COMPRESSED_LEB128' was '"} + leb128MagicString + "'");

    const std::uint32_t BUF_SIZE = 4096;
    std::uint8_t        buf[BUF_SIZE];

    std::uint32_t bytes_left;
    stream.read((char*) &bytes_left, sizeof(bytes_left));

    std::uint32_t buf_pos = BUF_SIZE;
    for (std::size_t i = 0; i < buffer.size(); ++i)
    {
        T result = 0;
        size_t  shift  = 0;
        do
        {
            if (buf_pos == BUF_SIZE)
            {
                stream.read(reinterpret_cast<char*>(buf), std::min(bytes_left, BUF_SIZE));
                buf_pos = 0;
            }

            std::uint8_t byte = buf[buf_pos++];
            --bytes_left;
            result |= (byte & 0x7f) << shift;
            shift += 7;

            if ((byte & 0x80) == 0)
            {
                buffer[i] = (sizeof(T) * 8 <= shift || (byte & 0x40) == 0)
                         ? result
                         : result | ~((1 << shift) - 1);
                break;
            }
        } while (shift < sizeof(T) * 8);
    }

    // assert(bytes_left == 0);
}

template <std::size_t N>
struct features {
    constexpr static inline std::size_t L0 = 22528;
    constexpr static inline std::size_t L1 = N;

    alignas(64) std::int16_t weights0[L0][L1];
    alignas(64) std::int16_t biases0[L1];

    features(std::istream& stream) {
        const auto biases = std::span{biases0};
        const auto weights = std::span<std::int16_t, L1 * L0>{&weights0[0][0], L1 * L0};
        std::uint32_t header;
        std::int32_t psqtWeights[8 * L0];

        stream.read(reinterpret_cast<char*>(&header), sizeof(header));
        read_leb_128(stream, biases);
        read_leb_128(stream, weights);
        read_leb_128(stream, std::span{psqtWeights});

        // permute
        std::ranges::for_each(span_cast<std::uint64_t>(biases) | std::views::chunk(8), [](auto&& chunk){
            std::swap(chunk[2], chunk[4]);
            std::swap(chunk[3], chunk[5]);
        });
        std::ranges::for_each(span_cast<std::uint64_t>(weights) | std::views::chunk(8), [](auto&& chunk){
            std::swap(chunk[2], chunk[4]);
            std::swap(chunk[3], chunk[5]);
        });

        // scale
        std::ranges::for_each(biases, [](auto&& bias){
            bias *= 2; 
        });
        std::ranges::for_each(weights, [](auto&& weight){
            weight *= 2; 
        });
    }
};

}  // namespace nnue
