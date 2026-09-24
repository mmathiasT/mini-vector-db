# mini-vector-db

A minimal vector database written from scratch in C++ — IVF (Inverted File Index) approximate nearest neighbor search built on custom k-means clustering, with binary file persistence and a TCP server/client.

Built as a learning project to understand the mechanics behind real vector databases from first principles.

## Features

- Vector representation with L2 and cosine distance
- Exact brute-force k-NN search (baseline / ground truth)
- K-means clustering implemented from scratch (Lloyd's algorithm)
- IVF index: cluster-based approximate nearest neighbor search with a configurable `nprobe`
- Insert into an existing index without rebuilding
- Binary persistence (save/load) for both raw vectors and the full IVF index
- TCP server + CLI client — query and insert over the network with a simple text protocol
- Benchmark suite measuring recall and speedup vs brute-force
- 8 automated tests covering the whole pipeline

## Architecture

```
src/
├── vector.h/.cpp    — Vec struct, distance functions, random vector generation
├── knn.h/.cpp        — brute-force k-NN (ground truth baseline)
├── io.h/.cpp          — binary save/load for raw vectors
├── kmeans.h/.cpp    — k-means clustering
├── ivf.h/.cpp         — IVF index: build, search, insert, persistence
├── server.h/.cpp    — TCP server (QUERY / INSERT protocol)
├── server_main.cpp   — starts the server with a built IVF index
├── client.cpp         — standalone CLI client
├── bench.cpp           — recall/speed benchmark
└── main.cpp             — I/O sanity check
tests/test_all.cpp    — end-to-end test suite
```

## How IVF search works

1. **Build**: run k-means on the dataset to get `nlist` centroids, then group every vector into an inverted list per centroid.
2. **Query**: compute the distance from the query to all centroids, pick the `nprobe` closest clusters, then brute-force search only within those clusters.

This trades exactness for speed: a smaller `nprobe` means fewer clusters are searched (faster), but the true nearest neighbor might live in a cluster that wasn't checked (lower recall). Setting `nprobe = nlist` searches every cluster and is mathematically equivalent to brute-force.

## Benchmark results

10,000 random 64-dimensional vectors, `nlist=50`, `k=10`, averaged over 250 queries:

| nprobe | recall | avg IVF time | avg brute-force time | speedup |
|---|---|---|---|---|
| 1  | 21.7%  | 243us  | 8050us | 33.1x |
| 5  | 45.9%  | 954us  | 8465us | 8.9x  |
| 10 | 63.0%  | 1711us | 8147us | 4.8x  |
| 20 | 82.1%  | 3712us | 8689us | 2.3x  |
| 30 | 92.4%  | 5344us | 8360us | 1.6x  |
| 40 | 97.9%  | 7487us | 8741us | 1.2x  |
| 50 | 100%   | 9041us | 8395us | 0.9x  |

`nprobe=50` (= `nlist`) hits exactly 100% recall — confirms the IVF search is a correct approximation of brute-force, not a different algorithm. Below full `nprobe`, there's a clear speed/recall trade-off: `nprobe=10` gets ~5x faster at ~63% recall, a reasonable operating point for many applications.

## Building and running

Using the Makefile:

```bash
make            # builds mini_vector_db (main sanity check)
make run-tests  # builds and runs the test suite
make bench      # builds the benchmark
make server     # builds the server
make client     # builds the client
make clean      # removes all built binaries
```

Or manually:

```bash
g++ -std=c++17 -Wall src/main.cpp src/vector.cpp src/knn.cpp src/io.cpp src/kmeans.cpp src/ivf.cpp -o mini_vector_db
g++ -std=c++17 -Wall tests/test_all.cpp src/vector.cpp src/knn.cpp src/io.cpp src/kmeans.cpp src/ivf.cpp -o test_all
g++ -std=c++17 -Wall src/bench.cpp src/vector.cpp src/knn.cpp src/io.cpp src/kmeans.cpp src/ivf.cpp -o bench
g++ -std=c++17 -Wall src/server_main.cpp src/server.cpp src/vector.cpp src/knn.cpp src/io.cpp src/kmeans.cpp src/ivf.cpp -o server
g++ -std=c++17 -Wall src/client.cpp -o client
```

Server + client demo:

```bash
./server &
./client QUERY 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 5 3
./client INSERT 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
```

## Protocol

The server accepts one command per TCP connection, as a single newline-terminated line:

```
QUERY <dim floats> <k> <nprobe>   -> space-separated list of nearest neighbor ids
INSERT <dim floats>                 -> "OK"
```

## Known simplifications

This is a learning project, not a production system. Deliberately left out:

- No authentication or encryption on the server — anyone who can reach the port can query/insert
- Single-threaded server — one client handled at a time
- No index rebuild/rebalancing after inserts — cluster quality degrades slowly as more vectors are added without a corresponding centroid update
- No delete operation
