#include "ivf.h"
#include "kmeans.h"
#include "knn.h"
#include <algorithm>
#include <utility>

IVFIndex build_ivf_index(const std::vector<Vec>& data, int nlist, int max_iters, unsigned seed) {
    KMeansResult kmeans_result = kmeans(data, nlist, max_iters, seed);

    IVFIndex index;
    index.centroids = kmeans_result.centroids;
    index.data = data;
    std::vector<std::vector<int>> inv_list(nlist);

    for (size_t i = 0; i < data.size(); i++) {
        int cluster = kmeans_result.assignments[i];
        inv_list[cluster].push_back(i);
    }
    index.inverted_lists = inv_list;

    return index;
}

std::vector<int> ivf_search(const IVFIndex& index, const Vec& query, int k, int nprobe) {
    std::vector<std::pair<float, int>> all_centroids;
    std::vector<int> closest_centroids;

    for (size_t i = 0; i < index.centroids.size(); i++) {
        all_centroids.push_back({squared_l2(query, index.centroids[i]), static_cast<int>(i)});
    }

    std::sort(all_centroids.begin(), all_centroids.end());

    for (int i = 0; i < nprobe; i++) {
        closest_centroids.push_back(all_centroids[i].second);
    }

    std::vector<Vec> candidates;
    for (int centroid_id : closest_centroids) {
        for (size_t j = 0; j < index.inverted_lists[centroid_id].size(); j++) {
            int vector_idx = index.inverted_lists[centroid_id][j];
            candidates.push_back(index.data[vector_idx]);
        }
    }

    return brute_force_knn(query, candidates, k);
}
