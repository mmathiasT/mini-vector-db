#include "vector.h"
#include "io.h"
#include "ivf.h"
#include "server.h"
#include <algorithm>
#include <cmath>
#include <iostream>

int main() {
    std::vector<Vec> data = load_vectors("tools/embeddings.bin");
    if (data.empty()) {
        std::cout << "No embeddings found at tools/embeddings.bin\n";
        return 1;
    }

    int cluster_count = static_cast<int>(std::sqrt(data.size()));
    IVFIndex index = build_ivf_index(data, cluster_count);

    std::cout << "Loaded " << data.size() << " embeddings (dim=" << data[0].data.size()
              << "), built index with cluster_count=" << cluster_count << "\n";

    run_server(index, 8080);
}