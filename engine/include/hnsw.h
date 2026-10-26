// hnsw.h  (Member 3: HNSW approximate search and benchmarking)
// The HNSW graph index, and the measurement helpers: latency statistics, Recall@K,
// Hit@K, and a description of the machine for benchmark reports.
#ifndef HNSW_H
#define HNSW_H

#include <string>

#include "common.h"
#include "data.h"
#include "search.h"

// ---------------------------------------------------------------- HNSW (dsa/member3_hnsw.cpp)
// Hierarchical Navigable Small World graph (Malkov & Yashunin), written from scratch.
// Nodes keep only ids and neighbour lists; vectors are read from the EmbeddingStore.
//
//   level(new node) = floor(-ln(U) * mL), U uniform in (0, 1], mL = 1 / ln(M)
//   insert: greedy descent on the layers above the new node's level, then on each layer
//           from its level down to 0: search the layer with efConstruction, select M
//           neighbours, link both ways, and prune any list that grew past its maximum
//           (M, or 2M on layer 0)
//   search: greedy descent to layer 1, then search layer 0 with ef = max(efSearch, k)
//
// Not safe to use from two threads at once: search() reuses the visited-marker array.
class HNSWIndex : public RetrievalIndex {
public:
    HNSWIndex();
    ~HNSWIndex();
    HNSWIndex(const HNSWIndex&) = delete;  // owns a large graph: never copied
    HNSWIndex& operator=(const HNSWIndex&) = delete;

    // Starts an empty graph over `store` (which must outlive the index). Throws Error for
    // bad parameters. Calling it again discards the old graph.
    void setup(const EmbeddingStore& store, const HnswParams& params);

    void insert(int id, const float* embedding);
    void search(const float* query, int queryDim, int k, CandidateArray& out) const;
    int size() const;
    long long memoryBytes() const;

    void setEfSearch(int ef);
    const HnswParams& params() const;
    const EmbeddingStore& store() const;

    // Looking inside the graph (validation, saving, tests).
    int entryPoint() const;
    int maxLevel() const;
    bool contains(int id) const;
    int levelOf(int id) const;  // -1 if the node is not in the graph
    const IntArray& neighbors(int id, int layer) const;  // throws Error for a bad layer
    int inDegreeOf(int id, int layer) const;  // incoming edges, kept up to date on insert
    int maxDegree(int layer) const;
    long long distanceComputations() const;  // since the last reset (analysis only)
    void resetDistanceComputations() const;

    // Saving and loading (used by IndexManager).
    void getRandomState(uint64_t* words, int& position) const;
    // Puts back node `id` with `level`; its lists are then filled through restoreLinks().
    void restoreNode(int id, int level);
    IntArray& restoreLinks(int id, int layer);
    // Finishes loading: recounts in-degrees, checks every link, sets entry point and RNG.
    void finishRestore(int entry, int topLevel, const uint64_t* rngWords, int rngPosition);

private:
    struct Node {
        int level;          // -1 = not inserted
        IntArray* links;    // links[layer] for layer 0..level
        int* inDegree;      // inDegree[layer] for layer 0..level
    };

    float dist(const float* q, int id) const;
    int randomLevel();
    Neighbor greedyClosest(const float* q, Neighbor entry, int layer) const;
    // SEARCH-LAYER (paper Algorithm 2): up to ef results, closest first, into `out`.
    void searchLayer(const float* q, const NeighborArray& entryPoints, int ef, int layer,
                     NeighborArray& out) const;
    // SELECT-NEIGHBORS: the m closest (naive) or the paper's heuristic (Algorithm 4).
    // `candidates` must be closest first, with distances measured from the base node.
    void selectNeighbors(const NeighborArray& candidates, int m, NeighborArray& out) const;
    // Prunes the list of `id` on `layer` back to its maximum degree.
    void shrinkNeighbors(int id, int layer);
    void nextVisitEpoch() const;
    void freeGraph();

    const EmbeddingStore* vectors;  // not owned
    const float* firstRow;          // row id starts at firstRow + id * dims
    int dims;
    HnswParams settings;
    double mL;
    Random64 rng;
    Node* nodes;  // nodes[id] for every row of the store
    int numNodes;
    BitArray present;  // present.get(id): node id has been inserted
    int count;  // nodes inserted so far
    int entry;
    int topLevel;

    // search() is const but needs scratch space: a node is visited in the current search
    // when visited[id] == epoch. `mutable` lets const functions change these fields.
    mutable unsigned int* visited;
    mutable unsigned int epoch;
    mutable long long distCount;
};

// ---------------------------------------------------------------- measurement (src/benchmark.cpp)

struct LatencyStats {
    double medianMs = 0, p95Ms = 0, meanMs = 0, minMs = 0, maxMs = 0;
    int samples = 0;
};

// Sorts values[0..n-1] in place and summarises them.
LatencyStats summarize(double* values, int n);

// Timing protocol: `warmup` queries are run and thrown away, then `repetitions` passes
// over all queries (repetition-outer, so the same query never runs twice in a row and
// hits a warm cache). Each query's latency is the median of its repetitions; result goes
// in perQueryMs[0..queries.size()-1]. Only RetrievalIndex::search is timed.
void timeQueries(const RetrievalIndex& index, const EmbeddingStore& queries, int k, int warmup,
                 int repetitions, double* perQueryMs);

// |approx ∩ exact| / k over the first k entries of each list.
double recallAtK(const CandidateArray& approx, const CandidateArray& exact, int k);
// True if `target` is among the first k results.
bool hitAtK(const CandidateArray& results, int target, int k);

struct SystemInfo {
    std::string cpuModel;
    int logicalCpus = 0;
    int physicalCores = 0;
    long long ramBytes = 0;
    std::string os, kernel, compiler, buildType, cxxFlags, gitCommit, cpuGovernor;
    std::string cpuAffinity;  // CPUs this process may run on
    std::string loadAverage;  // when collected
    std::string timestampUtc;
};

SystemInfo collectSystemInfo(const std::string& repoRoot);
long long peakRssBytes();     // VmHWM from /proc/self/status
long long currentRssBytes();  // VmRSS

#endif
