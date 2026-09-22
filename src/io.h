#pragma once
#include <string>
#include <vector>
#include "vector.h"

void save_vectors(const std::string& path, const std::vector<Vec>& vectors);
std::vector<Vec> load_vectors(const std::string& path);
