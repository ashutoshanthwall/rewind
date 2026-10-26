// search.h  (Member 2: similarity, exact search, top-K ranking)
// Sorting, the binary heap, top-K selection, the shared index interface, the exact
// (brute-force) index, and the SearchEngine that turns query text into results.
#ifndef SEARCH_H
#define SEARCH_H

#include <string>

#include "common.h"
#include "data.h"

// ---------------------------------------------------------------- sorting and heaps
// (dsa/member2_sorting.cpp)

// Sorts items[0..n-1] closest first (by closer()), in place, with introsort.
void sortNeighbors(Neighbor* items, int n);

// Binary heap stored in an array: the children of slot i are 2i+1 and 2i+2 and its parent
// is (i-1)/2. Every parent ranks at least as high as its children, so the top is slot 0.
// With closestOnTop = true it is a min-heap on distance (next node to explore); with
// false it is a max-heap (the worst of the results kept so far).
class NeighborHeap {
public:
    NeighborHeap(bool closestOnTop);
    void push(const Neighbor& item);  // O(log n)
    void pop();                       // O(log n)
    const Neighbor& top() const;      // O(1)
    int size() const;
    bool empty() const;
    void reserve(int n);

private:
    bool ranksAbove(const Neighbor& a, const Neighbor& b) const;  // should a sit above b?
    void siftUp(int i);
    void siftDown(int i);

    NeighborArray items;
    bool closestOnTop;
};

// Keeps the k closest items seen so far in a max-heap whose top is the worst one kept, so
// each push is O(log k) and a scan of N items is O(N log k) instead of a full sort.
class TopK {
public:
    TopK(int k);
    void push(const Neighbor& item);
    int size() const;
    void extractSorted(NeighborArray& out);  // closest first; empties the heap

private:
    int k;
    NeighborHeap heap;
};

// ---------------------------------------------------------------- index interface (src/search.cpp)
// BruteForceIndex's code is in dsa/member2_sorting.cpp.

// Implemented by BruteForceIndex (exact) and HNSWIndex (approximate). Results come back by
// descending score; equal scores are ordered by ascending id.
class RetrievalIndex {
public:
    virtual ~RetrievalIndex();
    // `embedding` must be the store's row for `id` (indexes keep only the id).
    virtual void insert(int id, const float* embedding) = 0;
    virtual void search(const float* query, int queryDim, int k, CandidateArray& out) const = 0;
    virtual int size() const = 0;
    virtual long long memoryBytes() const = 0;
};

// Converts closest-first neighbours into results with score = 1 - dist. Two different
// distances can round to the same score, so the final order is fixed again here.
void toCandidates(const NeighborArray& closestFirst, CandidateArray& out);

// Checks shared by both indexes; they throw QueryError / Error with a clear message.
void checkSearchArgs(const float* query, int queryDim, int k, int dim, int size);
void checkInsert(const EmbeddingStore& store, int id, const float* embedding,
                 const BitArray& present);

// Exact baseline and ground truth: scores every stored vector against the query and keeps
// the best k with TopK. O(N log k) per query.
class BruteForceIndex : public RetrievalIndex {
public:
    BruteForceIndex(const EmbeddingStore& store);
    void insert(int id, const float* embedding);
    void insertAll();  // every store row, in id order
    void search(const float* query, int queryDim, int k, CandidateArray& out) const;
    int size() const;
    long long memoryBytes() const;

private:
    const EmbeddingStore* store;  // not owned; the store must outlive the index
    IntArray ids;
    BitArray present;
};

// ---------------------------------------------------------------- search engine (src/search.cpp)

// Turns query text into a raw embedding (not necessarily unit length).
class QueryEncoder {
public:
    virtual ~QueryEncoder();
    virtual void encode(const std::string& text, FloatArray& out) = 0;
};

struct SearchResponse {
    CandidateArray hits;       // best first
    double retrievalMs = 0.0;  // time spent in RetrievalIndex::search only
};

// Query text -> clean text -> embedding -> checked unit vector -> index search -> results.
// It does not know which index it talks to. It keeps pointers to the metadata, store,
// index and encoder, which must all outlive it.
class SearchEngine {
public:
    SearchEngine(const MetadataManager& metadata, const EmbeddingStore& store,
                 const RetrievalIndex& index, QueryEncoder* encoder);

    // Trims and collapses whitespace; throws QueryError for empty text.
    static std::string normalizeQueryText(const std::string& text);
    void checkK(int k) const;  // throws QueryError unless 0 < k <= index size
    void embedQuery(const std::string& text, FloatArray& out) const;
    void prepareVector(const FloatArray& raw, FloatArray& out) const;  // check + normalise

    void search(const std::string& text, int k, SearchResponse& out) const;
    void searchEmbedding(const FloatArray& unitQuery, int k, SearchResponse& out) const;
    const ImageRecord& record(int id) const;

private:
    const MetadataManager* metadata;
    const EmbeddingStore* store;
    const RetrievalIndex* index;
    QueryEncoder* encoder;  // may be nullptr when only precomputed vectors are searched
};

#endif
