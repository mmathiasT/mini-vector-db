#include "knn.h"
#include <algorithm>
#include <utility>

std::vector<int> brute_force_knn(const Vec& query, const std::vector<Vec>& data, const std::vector<int>& candidate_indices, int k) {
    std::vector<std::pair<float, int>> distances;

     for (auto idx : candidate_indices) {
        float d = squared_l2(query, data[idx]);
        distances.push_back({d, data[idx].id});
    }

    std::sort(distances.begin(), distances.end());

    int result_count = std::min(k, static_cast<int>(distances.size()));
    std::vector<int> result;
    result.reserve(result_count);
    for (int i = 0; i < result_count; i++) {
        result.push_back(distances[i].second);
    }

    return result;
}

std::vector<int> all_indices(size_t count) {
    std::vector<int> indices(count);
    for (size_t i = 0; i < count; i++) {
        indices[i] = i;
    }
    return indices;
}
