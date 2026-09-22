#include "vector.h"
#include <random>
#include <cmath>

float squared_l2(const Vec& a, const Vec& b) {
    float result = 0;
    for (size_t i = 0; i < a.data.size(); i++) {
        float diff = a.data[i] - b.data[i];
        result += (diff * diff);
    }
    return result;
}

float cosine_similarity(const Vec& a, const Vec& b) {
    float dot_product = 0;
    float norm_a = 0;
    float norm_b = 0;
    for (size_t i = 0; i < a.data.size(); i++) {
        dot_product += a.data[i] * b.data[i];
        norm_a += a.data[i] * a.data[i];
        norm_b += b.data[i] * b.data[i];
    }
    norm_a = sqrt(norm_a);
    norm_b = sqrt(norm_b);
    return dot_product / (norm_a * norm_b);
}

std::vector<Vec> generate_random_vectors(int count, int dim, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f); // Random floats in [-1, 1), used to fill vector components.

    std::vector<Vec> result;

    for (int i = 0; i < count; i++) {
        Vec new_vec;
        new_vec.id = i;
        for (int j = 0; j < dim; j++) {
            new_vec.data.push_back(dist(gen));
        }
        result.push_back(new_vec);
    }
    return result;
}
