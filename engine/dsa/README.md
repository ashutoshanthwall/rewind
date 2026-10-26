# DSA files by member

Each member's most DSA-heavy file. These are real source files: CMake compiles them into
the engine, so edit them here. Declarations are in the matching header in `include/`.

| Member | File | Header | Core DSA |
|---|---|---|---|
| 1 | `member1_inverted_index.cpp` | `data.h` | Hash table (open addressing, linear probing, FNV-1a, power-of-two capacity, rehash at load factor 0.7); inverted index with sorted posting lists, two-pointer and galloping intersection (AND), two-pointer union (OR) |
| 2 | `member2_sorting.cpp` | `search.h` | Introsort (median-of-three quicksort, heap-sort fallback, insertion-sort finish); binary heap (min and max); bounded top-K selection; the exact brute-force index that uses them |
| 3 | `member3_hnsw.cpp` | `hnsw.h` | HNSW graph: skip-list-style random levels, adjacency lists per layer, two-heap best-first search, heuristic neighbour selection, pruning with in-degree tracking |
| 4 | `member4_arrays.cpp` | `common.h` | Growable arrays with doubling (amortised O(1) push) and deep copy; bit array packed 64 flags per word |

Evidence: G12 (`evaluation/results/g12_caption_index.txt`) for Member 1, G6/G7 for
Member 2, G8/G9 for Member 3, and the `arrays` checks in `tests/test_storage.cpp` for
Member 4.
