#pragma once
#include <vector>
#include "vector.h"

// Exact O(N) search, used as ground-truth baseline for comparing against approximate (IVF) search.
std::vector<int> brute_force_knn(const Vec& query, const std::vector<Vec>& db, int k);
