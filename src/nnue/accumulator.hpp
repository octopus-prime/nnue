#pragma once

#include <nnue/common.hpp>

namespace nnue {

template <std::size_t N>
struct accumulator {
    alignas(64)  std::int16_t accumulation[2][N];
    bool         computed[2];
};

}  // namespace nnue
