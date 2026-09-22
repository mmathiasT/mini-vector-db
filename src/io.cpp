#include "io.h"
#include <fstream>

void save_vectors(const std::string& path, const std::vector<Vec>& vectors) {
    std::ofstream out(path, std::ios::binary);

    for (const Vec& v : vectors) {
        int id = v.id;
        out.write(reinterpret_cast<const char*>(&id), sizeof(id));

        int dim = v.data.size();
        out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));

        out.write(reinterpret_cast<const char*>(v.data.data()), v.data.size() * sizeof(float));
    }
}

std::vector<Vec> load_vectors(const std::string& path) {
    std::vector<Vec> result;
    std::ifstream in(path, std::ios::binary);

    while (true) {
        int id;
        in.read(reinterpret_cast<char*>(&id), sizeof(id));

        if (!in) {
            break;
        }

        int dim;
        in.read(reinterpret_cast<char*>(&dim), sizeof(dim));

        Vec v;
        v.id = id;
        v.data.resize(dim);
        in.read(reinterpret_cast<char*>(v.data.data()), v.data.size() * sizeof(float));

        result.push_back(v);
    }
    return result;
}
