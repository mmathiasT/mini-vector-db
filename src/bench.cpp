#include "vector.h"
#include "knn.h"
#include "ivf.h"
#include <iostream>
#include <chrono>
#include <algorithm>

float recall(const std::vector<int>& exact, const std::vector<int>& approx) {
    int hits = 0;
    for (int id : exact) {
        if (std::find(approx.begin(), approx.end(), id) != approx.end()) {
            hits++;
        }
    }
    return static_cast<float>(hits) / exact.size();
}

int main() {
    std::vector<Vec> data = generate_random_vectors(10000, 64);
    IVFIndex index = build_ivf_index(data, 50);

    int num_queries = 250;
    int k = 10;
    std::vector<int> nprobe_values = {1, 5, 10, 20, 30, 40, 50};

    for (int nprobe : nprobe_values) {
        float total_recall = 0;
        long long total_ivf_time_us = 0;
        long long total_bf_time_us = 0;

        for (int q = 0; q < num_queries; q++) {
            Vec query = data[q];

            auto bf_start = std::chrono::high_resolution_clock::now();
            auto exact = brute_force_knn(query, data, all_indices(data.size()), k);
            auto bf_end = std::chrono::high_resolution_clock::now();

            auto ivf_start = std::chrono::high_resolution_clock::now();
            auto approx = ivf_search(index, query, k, nprobe);
            auto ivf_end = std::chrono::high_resolution_clock::now();

            total_recall += recall(exact, approx);
            total_bf_time_us += std::chrono::duration_cast<std::chrono::microseconds>(bf_end - bf_start).count();
            total_ivf_time_us += std::chrono::duration_cast<std::chrono::microseconds>(ivf_end - ivf_start).count();
        }

        long long avg_bf_time = total_bf_time_us / num_queries;
        long long avg_ivf_time = total_ivf_time_us / num_queries;
        float speedup = static_cast<float>(avg_bf_time) / avg_ivf_time;

        std::cout << "nprobe=" << nprobe
                  << " recall=" << (total_recall / num_queries * 100) << "%"
                  << " avg_ivf_time=" << avg_ivf_time << "us"
                  << " avg_bf_time=" << avg_bf_time << "us"
                  << " speedup=" << speedup << "x\n";
    }
}