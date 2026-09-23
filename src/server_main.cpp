#include "vector.h"
#include "ivf.h"
#include "server.h"

int main() {
    std::vector<Vec> data = generate_random_vectors(200, 8);
    IVFIndex index = build_ivf_index(data, 5);
    run_server(index, 8080);
}