#include "utils/efg_cutils.h"
#include "sort_vert.h"

namespace efg {

void sort_vertices_wrapper(int b, int n, int m, const float *vertices, const bool *mask, const int *num_valid, int* idx);

at::Tensor sort_vertices(at::Tensor vertices, at::Tensor mask, at::Tensor num_valid){
    CHECK_CONTIGUOUS(vertices);
    CHECK_CONTIGUOUS(mask);
    CHECK_CONTIGUOUS(num_valid);
    CHECK_CUDA(vertices);
    CHECK_CUDA(mask);
    CHECK_CUDA(num_valid);
    CHECK_IS_FLOAT(vertices);
    CHECK_IS_BOOL(mask);
    CHECK_IS_INT(num_valid);

    int b = vertices.size(0);
    int n = vertices.size(1);
    int m = vertices.size(2);
    at::Tensor idx = torch::zeros({b, n, MAX_NUM_VERT_IDX}, at::device(vertices.device()).dtype(at::ScalarType::Int));

    sort_vertices_wrapper(b, n, m, vertices.data_ptr<float>(), mask.data_ptr<bool>(), num_valid.data_ptr<int>(), idx.data_ptr<int>());

    return idx;
}

}
