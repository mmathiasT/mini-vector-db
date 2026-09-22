#include "knn.h"
#include <algorithm>
#include <utility>

std::vector<int> brute_force_knn(const Vec& query, const std::vector<Vec>& db, int k) {
    std::vector<std::pair<float, int>> distances;

    for (const Vec& v : db) {
        float d = squared_l2(query, v);
        distances.push_back({d, v.id});
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
