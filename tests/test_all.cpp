#include "../src/vector.h"
#include "../src/knn.h"
#include "../src/io.h"
#include "../src/kmeans.h"
#include "../src/ivf.h"

#include <cassert>
#include <iostream>
#include <cmath>
#include <algorithm>

void test_squared_l2() {
    Vec a{0, {0.0f, 0.0f}};
    Vec b{1, {3.0f, 4.0f}};
    assert(std::abs(squared_l2(a, b) - 25.0f) < 1e-4f);
    std::cout << "test_squared_l2 OK\n";
}

void test_cosine_similarity() {
    Vec a{0, {1.0f, 0.0f}};
    Vec b{1, {1.0f, 0.0f}};
    Vec c{2, {0.0f, 1.0f}};
    assert(std::abs(cosine_similarity(a, b) - 1.0f) < 1e-4f);
    assert(std::abs(cosine_similarity(a, c) - 0.0f) < 1e-4f);
    std::cout << "test_cosine_similarity OK\n";
}

void test_brute_force_knn_finds_self() {
    std::vector<Vec> data = generate_random_vectors(200, 16);
    auto result = brute_force_knn(data[0], data, 1);
    assert(result.size() == 1);
    assert(result[0] == data[0].id);
    std::cout << "test_brute_force_knn_finds_self OK\n";
}

void test_save_load_vectors_roundtrip() {
    std::vector<Vec> original = generate_random_vectors(500, 32);
    save_vectors("data/test_vectors.bin", original);
    std::vector<Vec> loaded = load_vectors("data/test_vectors.bin");

    assert(loaded.size() == original.size());
    auto r1 = brute_force_knn(original[0], original, 5);
    auto r2 = brute_force_knn(loaded[0], loaded, 5);
    assert(r1 == r2);
    std::cout << "test_save_load_vectors_roundtrip OK\n";
}

void test_kmeans_separates_clusters() {
    std::vector<Vec> data;
    for (int i = 0; i < 50; i++) {
        Vec v{i, {10.0f + (i % 3) * 0.1f, 10.0f, 10.0f, 10.0f}};
        data.push_back(v);
    }
    for (int i = 50; i < 100; i++) {
        Vec v{i, {-10.0f + (i % 3) * 0.1f, -10.0f, -10.0f, -10.0f}};
        data.push_back(v);
    }

    KMeansResult result = kmeans(data, 2);

    bool first_half_consistent = true;
    for (int i = 1; i < 50; i++) {
        if (result.assignments[i] != result.assignments[0]) first_half_consistent = false;
    }
    bool second_half_consistent = true;
    for (int i = 51; i < 100; i++) {
        if (result.assignments[i] != result.assignments[50]) second_half_consistent = false;
    }
    bool groups_differ = result.assignments[0] != result.assignments[50];

    assert(first_half_consistent);
    assert(second_half_consistent);
    assert(groups_differ);
    std::cout << "test_kmeans_separates_clusters OK\n";
}

void test_ivf_full_nprobe_matches_brute_force() {
    std::vector<Vec> data = generate_random_vectors(1000, 16);
    IVFIndex index = build_ivf_index(data, 10);

    Vec query = data[0];
    int k = 5;

    auto exact = brute_force_knn(query, data, k);
    auto approx = ivf_search(index, query, k, /*nprobe=*/10);

    std::sort(exact.begin(), exact.end());
    std::sort(approx.begin(), approx.end());
    assert(exact == approx);
    std::cout << "test_ivf_full_nprobe_matches_brute_force OK\n";
}

void test_ivf_insert_is_searchable() {
    std::vector<Vec> data = generate_random_vectors(500, 8);
    IVFIndex index = build_ivf_index(data, 10);

    Vec new_vector;
    new_vector.id = 9999;
    new_vector.data = data[0].data;

    ivf_insert(index, new_vector);

    auto result = ivf_search(index, new_vector, 3, 10);
    bool found = false;
    for (int id : result) if (id == 9999) found = true;
    assert(found);
    std::cout << "test_ivf_insert_is_searchable OK\n";
}

void test_ivf_index_save_load_roundtrip() {
    std::vector<Vec> data = generate_random_vectors(1000, 16);
    IVFIndex original = build_ivf_index(data, 15);

    save_ivf_index("data/test_ivf_index", original);
    IVFIndex loaded = load_ivf_index("data/test_ivf_index");

    assert(original.centroids.size() == loaded.centroids.size());
    assert(original.data.size() == loaded.data.size());
    assert(original.inverted_lists.size() == loaded.inverted_lists.size());
    for (size_t i = 0; i < original.inverted_lists.size(); i++) {
        assert(original.inverted_lists[i] == loaded.inverted_lists[i]);
    }

    Vec query = data[0];
    auto r1 = ivf_search(original, query, 5, 5);
    auto r2 = ivf_search(loaded, query, 5, 5);
    assert(r1 == r2);
    std::cout << "test_ivf_index_save_load_roundtrip OK\n";
}

int main() {
    test_squared_l2();
    test_cosine_similarity();
    test_brute_force_knn_finds_self();
    test_save_load_vectors_roundtrip();
    test_kmeans_separates_clusters();
    test_ivf_full_nprobe_matches_brute_force();
    test_ivf_insert_is_searchable();
    test_ivf_index_save_load_roundtrip();

    std::cout << "\n All tests passed\n";
    return 0;
}
