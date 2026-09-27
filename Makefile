CXX = g++
CXXFLAGS = -std=c++17 -Wall

# Shared library code (everything except the files with their own main()).
SRC = src/vector.cpp src/knn.cpp src/io.cpp src/kmeans.cpp src/ivf.cpp

.PHONY: all test bench server client gpt-search run-tests clean

all: mini_vector_db

mini_vector_db: src/main.cpp $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

test: tests/test_all.cpp $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o test_all

run-tests: test
	./test_all

bench: src/bench.cpp $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

server: src/server_main.cpp src/server.cpp $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

client: src/client.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

gpt-search: src/gpt_search_main.cpp src/server.cpp $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o gpt_search_server

clean:
	rm -f mini_vector_db test_all bench server client gpt_search_server
