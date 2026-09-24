#include "vector.h"
#include "io.h"
#include "knn.h"
#include <iostream>

int main() {
    std::vector<Vec> original = generate_random_vectors(1000, 32);

    save_vectors("data/test.bin", original);
    std::vector<Vec> loaded = load_vectors("data/test.bin");

    if (loaded.size() != original.size()) {
        std::cout << "ERROR: sizes differ!\n";
        return 1;
    }

    std::vector<int> result_original = brute_force_knn(original[0], original, all_indices(original.size()), 5);
    std::vector<int> result_loaded   = brute_force_knn(loaded[0], loaded, all_indices(loaded.size()), 5);

    if (result_original == result_loaded) {
        std::cout << "OK\n";
    } else {
        std::cout << "ERROR: results differ!\n";
    }

    return 0;
}