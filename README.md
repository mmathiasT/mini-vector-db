# mini-vector-db

A small vector database written from scratch in C++. It uses an IVF (Inverted File Index) with custom k-means clustering to find similar vectors fast, plus file storage and a TCP server/client.

I built this to learn how vector databases work, from the ground up.

It also has a semantic search demo, built on [gpt-from-scratch](https://github.com/mmathiasT/gpt-from-scratch), a Character-Level GPT I trained myself. See [Semantic search demo](#semantic-search-demo) below.

## Features

- Vectors with L2 and cosine distance
- Exact brute-force k-NN search (the "correct answer" baseline)
- K-means clustering from scratch (Lloyd's algorithm)
- IVF index for fast approximate search, with a `nprobe` setting you can adjust
- Insert into an existing index, no full rebuild needed
- Soft delete (marks vectors as deleted, doesn't shift the index)
- Auto rebuild: after ~20% growth, redoes k-means on live data to fix clusters and drop deleted vectors
- Save/load for raw vectors and the full index, including delete flags
- TCP server + CLI client for query/insert/delete
- Benchmark comparing speed and accuracy against brute-force
- 12 automated tests
- Semantic search demo using GPT word embeddings

## Architecture

```
src/
├── vector.h/.cpp      — Vec struct, distance functions, random vector generation
├── knn.h/.cpp         — brute-force k-NN (ground truth baseline)
├── io.h/.cpp          — binary save/load for raw vectors
├── kmeans.h/.cpp      — k-means clustering
├── ivf.h/.cpp         — IVF index: build, search, insert, delete, rebuild, persistence
├── server.h/.cpp      — TCP server (QUERY / INSERT / DELETE protocol)
├── server_main.cpp    — starts the server with a built IVF index
├── client.cpp         — standalone CLI client
├── bench.cpp          — recall/speed benchmark
├── gpt_search_main.cpp — starts the server with GPT word embeddings (see below)
└── main.cpp           — I/O sanity check
tests/test_all.cpp     — end-to-end test suite
tools/                  — Python side of the semantic search demo
├── gpt_model_copy.py   — copy of the GPT model class (unmodified)
├── embed_corpus.py      — embeds Shakespeare's unique words, writes embeddings.bin
├── query_client.py       — embeds a query and sends it to the server
└── input.txt              — Shakespeare corpus (public domain)
```

## How IVF search works

1. **Build**: run k-means to get `cluster_count` centroids. Put every vector in the group of its closest centroid.
2. **Query**: compare the query to all centroids, pick the `nprobe` closest ones, then search only inside those groups.

This trades correctness for speed. A smaller `nprobe` checks fewer groups — faster, but you might miss the real nearest neighbor (lower recall). `nprobe = cluster_count` checks every group, so it gives the same result as brute-force.

### Automatic rebuild

Inserting doesn't move the centroids, so clusters slowly get worse as more data comes in — and deleted vectors still take up space. `ivf_insert` counts inserts since the last rebuild. Once that count hits ~20% of the dataset size, it calls `ivf_rebuild`: takes all non-deleted vectors and reruns `build_ivf_index` from scratch, making fresh centroids and dropping deleted vectors for good. A rebuild costs as much as a full build (`O(max_iters × N × cluster_count)`), so it only happens now and then, not on every insert.

## Benchmark results

10,000 random 64-dimensional vectors, `cluster_count=50`, `k=10`, averaged over 250 queries:

| nprobe | recall | avg IVF time | avg brute-force time | speedup |
|---|---|---|---|---|
| 1  | 21.7%  | 243us  | 8050us | 33.1x |
| 5  | 45.9%  | 954us  | 8465us | 8.9x  |
| 10 | 63.0%  | 1711us | 8147us | 4.8x  |
| 20 | 82.1%  | 3712us | 8689us | 2.3x  |
| 30 | 92.4%  | 5344us | 8360us | 1.6x  |
| 40 | 97.9%  | 7487us | 8741us | 1.2x  |
| 50 | 100%   | 9041us | 8395us | 0.9x  |

At `nprobe=50` (= `cluster_count`), recall is exactly 100% — this shows IVF search gives a correct approximation of brute-force, not a different algorithm. Below full `nprobe`, there's a clear trade-off: `nprobe=10` is ~5x faster at ~63% recall, a good setting for many cases.

## Building and running

Using the Makefile:

```bash
make            # builds mini_vector_db (main sanity check)
make run-tests  # builds and runs the test suite
make bench      # builds the benchmark
make server     # builds the server
make client     # builds the client
make gpt-search # builds the semantic search server (see below)
make clean      # removes all built binaries
```

Or by hand:

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
./client DELETE 0
```

## Protocol

One command per TCP connection, sent as a single line ending in `\n`:

```
QUERY <dim floats> <k> <nprobe>   -> space-separated list of nearest neighbor ids
INSERT <dim floats>               -> "OK"
DELETE <id>                       -> "OK" or "NOT FOUND"
```

## Semantic search demo

Word-level search over Shakespeare's text, using embeddings from [gpt-from-scratch](https://github.com/mmathiasT/gpt-from-scratch) — a decoder-only Transformer I built and trained myself.

`tools/gpt_model_copy.py` is an unchanged copy of that project's model class. It's copied here so this repo doesn't depend on the other one's file layout. `embed_corpus.py` runs each unique word through the model (up to the final layer norm, skipping the vocabulary layer), averages the per-character outputs into one 256-number vector per word, and saves them in the same binary format `save_vectors` already uses. So the C++ side needs no new code to read them.

```bash
# one-time setup: copy your own trained checkpoint
mkdir -p tools/checkpoints
cp /path/to/gpt-from-scratch/checkpoints/gpt_best_model.pt tools/checkpoints/

# build the embeddings (uses the gpt project's venv, needs torch)
cd tools
/path/to/gpt-from-scratch/.venv/bin/python3 embed_corpus.py

# start the search server
cd ..
make gpt-search
./gpt_search_server &

# query it
cd tools
/path/to/gpt-from-scratch/.venv/bin/python3 query_client.py king
```

The query and every word in the index go through the exact same embedding function, so comparing them makes sense.

**Known limitation:** the model reads text left-to-right only (causal attention), and the final vector is just an average over all character positions. So words that share a prefix get very similar vectors, even if their meaning is different — e.g. "apple" and "applied" come out as similar. Early characters carry little context in this kind of model, so a shared prefix dominates the average. A model that reads both directions, or using only the last character's vector instead of averaging all of them, would fix this.

## Known simplifications

This is a learning project, not a production system:

- No login or encryption — anyone who can reach the port can query/insert
- One client at a time — the server is single-threaded
- Delete just flags a vector until the next rebuild, it doesn't remove it right away
- The rebuild threshold (~20% growth) is a fixed number, not adjustable
- No automated tests for the Python side (semantic search demo)
- The trained model checkpoint isn't included (too large) — bring your own from [gpt-from-scratch](https://github.com/mmathiasT/gpt-from-scratch)
