// collection.h  (Rewind, step 7.1)
// Several named HNSW indexes in one process (Rewind: one per camera per branch), each over its
// own embedding store, with one time value per row for time-range filtering and a lock so that
// one index is never used by two threads at once (HNSWIndex::search reuses scratch memory).
// Ids stay the store's row indices 0..N-1; the caller keeps the id -> metadata map.
//
// Filtered search (decided 26 Oct, PROGRESS.md step 7.1; HNSW internals are not changed):
//   no time range        -> HNSW top k
//   time range [t0, t1]  -> n = rows with t0 <= time <= t1, want = min(k, n)
//     n == 0                       -> no results ("empty")
//     n <= k * oversample          -> exact search over those n rows ("exact_small")
//     otherwise HNSW top k * oversample, keep the in-range hits (best first, at most want);
//       kept >= minFraction * want -> those ("hnsw_filtered")
//       else                       -> exact search over the n rows ("exact_fallback")
#ifndef COLLECTION_H
#define COLLECTION_H

#include <mutex>
#include <string>

#include "common.h"
#include "data.h"
#include "hnsw.h"
#include "search.h"

struct FilterParams {
    int oversample = 4;            // HNSW candidates = k * oversample when a time range is set
    double minFraction = 1.0;      // fewer in-range hits than this share of min(k, n) -> exact
};

// How one search was answered (for tests, benchmarks and the API's "why" fields).
struct SearchReport {
    std::string method;  // hnsw | empty | exact_small | hnsw_filtered | exact_fallback | exact
    int inRange = -1;    // rows inside the time range (-1: no range)
    int candidates = 0;  // HNSW candidates requested (0: HNSW not used)
};

class NamedIndex {
public:
    NamedIndex(const std::string& name, const std::string& model);
    ~NamedIndex();
    NamedIndex(const NamedIndex&) = delete;
    NamedIndex& operator=(const NamedIndex&) = delete;

    // Vectors: from a float32 file + manifest (checked as EmbeddingStore::load), or copied
    // from a buffer. Either one discards any graph and times set before.
    void loadVectors(const std::string& binPath, const std::string& manifestPath, int dim,
                     double normTolerance);
    void setVectors(int dim, long long rows, const float* data, double normTolerance);
    // One finite time per row (seconds; Rewind: wall clock as Unix seconds).
    void setTimes(const double* t, long long n);

    double build(const HnswParams& params);  // returns seconds; validates the graph
    void save(const std::string& path) const;
    void load(const std::string& path);  // graph only; vectors must be loaded first

    // HNSW search, filtered when hasRange (rules above). ef = max(efSearch, candidates).
    void search(const float* query, int queryDim, int k, int efSearch, bool hasRange, double t0,
                double t1, const FilterParams& filter, CandidateArray& out, SearchReport& report);
    // Exact (brute-force) top k over every row, or over the rows inside [t0, t1].
    void exact(const float* query, int queryDim, int k, bool hasRange, double t0, double t1,
               CandidateArray& out, SearchReport& report);

    const std::string& name() const;
    const std::string& model() const;
    int size() const;
    int dim() const;
    bool hasGraph() const;
    double timeOf(int id) const;
    const EmbeddingStore& store() const;

private:
    void requireReady(bool needGraph) const;
    void rangeIds(double t0, double t1, IntArray& out) const;
    void exactOver(const float* query, int queryDim, int k, const IntArray& ids, CandidateArray& out) const;

    std::string indexName;
    std::string modelName;
    EmbeddingStore vectors;
    HNSWIndex graph;
    BruteForceIndex* everything;  // exact search over all rows (benchmarks, ground truth)
    double* times;
    int numTimes;
    bool built;
    mutable std::mutex lock;  // held by every search, build, save and load
};

// Named indexes owned by one object. Names are unique.
class IndexCollection {
public:
    IndexCollection();
    ~IndexCollection();
    IndexCollection(const IndexCollection&) = delete;
    IndexCollection& operator=(const IndexCollection&) = delete;

    NamedIndex& add(const std::string& name, const std::string& model);  // throws if the name exists
    NamedIndex& get(const std::string& name);  // throws Error naming the missing index
    bool has(const std::string& name) const;
    int size() const;
    const std::string& nameAt(int i) const;

private:
    NamedIndex** items;
    int count;
    int cap;
};

#endif
