#pragma once
#include <vector>

struct Vec {
    int id;
    std::vector<float> data;
};

float squared_l2(const Vec& a, const Vec& b);
float cosine_similarity(const Vec& a, const Vec& b);

std::vector<Vec> generate_random_vectors(int count, int dim, unsigned seed = 42);
