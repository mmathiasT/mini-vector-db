#pragma once
#include <vector>
#include "vector.h"

// Exact O(N) search over the given candidate indices, used as ground-truth baseline
// for comparing against approximate (IVF) search.
std::vector<int> brute_force_knn(const Vec& query, const std::vector<Vec>& data, const std::vector<int>& candidate_indices, int k);

// Convenience helper: {0, 1, ..., count - 1}, used to search the entire dataset.
std::vector<int> all_indices(size_t count);
