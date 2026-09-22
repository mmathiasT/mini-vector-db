#pragma once
#include <vector>
#include "vector.h"

struct IVFIndex {
    std::vector<Vec> centroids;
    std::vector<std::vector<int>> inverted_lists;
    std::vector<Vec> data;
};

IVFIndex build_ivf_index(const std::vector<Vec>& data, int nlist, int max_iters = 50, unsigned seed = 42);
std::vector<int> ivf_search(const IVFIndex& index, const Vec& query, int k, int nprobe);
