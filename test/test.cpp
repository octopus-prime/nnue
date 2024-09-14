#include "test_clipped_relu.hpp"
#include "test_sqr_clipped_relu.hpp"
#include "test_affine_tranform.hpp"

int main() {
    nnue::test_clipped_relu_16();
    nnue::test_clipped_relu_32();
    nnue::test_sqr_clipped_relu_16();
    nnue::test_affine_tranform_32_1();
    nnue::test_affine_tranform_32_32();
    nnue::test_affine_tranform_32_32_2();
}
