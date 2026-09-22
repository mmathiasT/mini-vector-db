#pragma once
#include <vector>
#include <string>
#include "vector.h"

struct IVFIndex {
    std::vector<Vec> centroids;
    std::vector<std::vector<int>> inverted_lists;
    std::vector<Vec> data;
};

IVFIndex build_ivf_index(const std::vector<Vec>& data, int nlist, int max_iters = 50, unsigned seed = 42);
std::vector<int> ivf_search(const IVFIndex& index, const Vec& query, int k, int nprobe);
void ivf_insert(IVFIndex& index, const Vec& new_vector);
void save_ivf_index(const std::string& path, const IVFIndex& index);
IVFIndex load_ivf_index(const std::string& path);