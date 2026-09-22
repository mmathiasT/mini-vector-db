#include "kmeans.h"
#include <random>
#include <algorithm>
#include <limits>
#include <stdexcept>

static int nearest_centroid(const Vec& point, const std::vector<Vec>& centroids) {
    int closest_centroid_idx = 0;
    float min_distance = std::numeric_limits<float>::max();

    for (size_t i = 0; i < centroids.size(); i++) {
        float distance = squared_l2(point, centroids[i]);
        if (distance < min_distance) {
            closest_centroid_idx = i;
            min_distance = distance;
        }
    }
    return closest_centroid_idx;
}

static Vec average(const std::vector<Vec>& points, int dim) {
    Vec result;
    for (int i = 0; i < dim; i++) {
        float sum = 0;
        for (const Vec& point : points) {
            sum += point.data[i];
        }
        result.data.push_back(sum / float(points.size()));
    }
    return result;
}

KMeansResult kmeans(const std::vector<Vec>& data, int k, int max_iters, unsigned seed) {
    if (data.empty() || k > static_cast<int>(data.size())) {
        throw std::invalid_argument("Invalid data or k parameter");
    }

    int dim = static_cast<int>(data[0].data.size());

    std::vector<int> indices(data.size());
    for (size_t i = 0; i < data.size(); i++) indices[i] = i;

    std::mt19937 gen(seed);
    std::shuffle(indices.begin(), indices.end(), gen);

    std::vector<Vec> centroids;
    for (int i = 0; i < k; i++) {
        centroids.push_back(data[indices[i]]);
    }

    std::vector<std::vector<int>> clusters(k);
    std::vector<int> assignments(data.size());

    while (max_iters--) {
        for (size_t i = 0; i < data.size(); i++) {
            assignments[i] = nearest_centroid(data[i], centroids);
            clusters[assignments[i]].push_back(i);
        }

        for (int i = 0; i < k; i++) {
            if (!clusters[i].empty()) {
                std::vector<Vec> cluster_points;
                for (size_t j = 0; j < clusters[i].size(); j++) {
                    cluster_points.push_back(data[clusters[i][j]]);
                }
                centroids[i] = average(cluster_points, dim);
            }
            clusters[i].clear();
        }
    }
    KMeansResult result = {centroids, assignments};
    return result;
}
