#pragma once
#include <vector>
#include "vector.h"

struct KMeansResult {
    std::vector<Vec> centroids;
    std::vector<int> assignments;
};

KMeansResult kmeans(const std::vector<Vec>& data, int k, int max_iters = 50, unsigned seed = 42);
