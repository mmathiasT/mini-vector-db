#include "ivf.h"
#include "kmeans.h"
#include "knn.h"
#include "io.h"
#include <algorithm>
#include <fstream>
#include <utility>

IVFIndex build_ivf_index(const std::vector<Vec>& data, int nlist, int max_iters, unsigned seed) {
    KMeansResult kmeans_result = kmeans(data, nlist, max_iters, seed);

    IVFIndex index;
    index.centroids = kmeans_result.centroids;
    index.data = data;
    std::vector<std::vector<int>> inv_list(nlist);
    index.deleted = std::vector<bool>(data.size(), false);

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

    std::vector<int> candidates;
    for (int centroid_id : closest_centroids) {
        for (size_t j = 0; j < index.inverted_lists[centroid_id].size(); j++) {
            int vector_idx = index.inverted_lists[centroid_id][j];
            if (!index.deleted[vector_idx]) {
                candidates.push_back(vector_idx);
            }
        }
    }

    return brute_force_knn(query, index.data, candidates, k);
}

void ivf_insert(IVFIndex& index, const Vec& new_vector) {
    int closest_centroid_index = closest_centroid(new_vector, index.centroids);

    index.data.push_back(new_vector);
    index.deleted.push_back(false);

    index.inverted_lists[closest_centroid_index].push_back(index.data.size() - 1);

    index.inserts_since_rebuild++;

    int threshold = std::max(10, static_cast<int>(index.data.size()) / 5);
    if (index.inserts_since_rebuild >= threshold) {
        int nlist = static_cast<int>(index.centroids.size());
        index = ivf_rebuild(index, nlist);
        index.inserts_since_rebuild = 0;
    }
}

void save_ivf_index(const std::string& path, const IVFIndex& index) {
    save_vectors(path + ".centroids", index.centroids);
    save_vectors(path + ".data", index.data);
    save_bool_vector(path + ".deleted", index.deleted);

    std::string path_inverted_list = path + ".inverted_lists";

    std::ofstream out(path_inverted_list, std::ios::binary);

    int num_lists = index.inverted_lists.size();
    out.write(reinterpret_cast<const char*>(&num_lists), sizeof(num_lists));

    for(size_t i = 0; i < index.inverted_lists.size(); i++) {
        int num_list_i = index.inverted_lists[i].size();
        out.write(reinterpret_cast<const char*>(&num_list_i), sizeof(num_list_i));

        for(int j = 0; j < num_list_i; j++) {
            int x = index.inverted_lists[i][j];
            out.write(reinterpret_cast<const char*>(&x), sizeof(x));
        }   
    }
}

IVFIndex load_ivf_index(const std::string& path) {
    IVFIndex ivf_index;

    std::string path_centroids = path + ".centroids";
    std::string path_data = path + ".data";
    std::string path_deleted = path + ".deleted";

    ivf_index.centroids = load_vectors(path_centroids);
    ivf_index.data = load_vectors(path_data);
    ivf_index.deleted = load_bool_vector(path_deleted);

    std::string path_inverted_lists = path + ".inverted_lists";
    std::ifstream in(path_inverted_lists, std::ios::binary);

    int num_lists = 0;
    in.read(reinterpret_cast<char*>(&num_lists), sizeof(num_lists));

    ivf_index.inverted_lists.resize(num_lists);

    for(int i = 0; i < num_lists; i++) {
        int num_list_i = 0;
        in.read(reinterpret_cast<char*>(&num_list_i), sizeof(num_list_i));

        for(int j = 0; j < num_list_i; j++) {
            int next_num;
            in.read(reinterpret_cast<char*>(&next_num), sizeof(next_num));
            ivf_index.inverted_lists[i].push_back(next_num);
        }
    }

    return ivf_index;
}

bool ivf_delete(IVFIndex& ivf_index, int id) {
    for(size_t i = 0; i < ivf_index.data.size(); i++) {
        if(ivf_index.data[i].id == id) {
            ivf_index.deleted[i] = true;
            return true;
        }
    }
    return false;
}

IVFIndex ivf_rebuild(const IVFIndex& index, int nlist, int max_iters, unsigned seed) {
    std::vector<Vec> live_data;
    for (size_t i = 0; i < index.data.size(); i++) {
        if (!index.deleted[i]) {
            live_data.push_back(index.data[i]);
        }
    }
    return build_ivf_index(live_data, nlist, max_iters, seed);
}