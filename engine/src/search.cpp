// Index interface helpers and the SearchEngine (search.h). The brute-force index is in
// dsa/member2_sorting.cpp.
#include "search.h"

#include <cctype>
#include <cmath>

RetrievalIndex::~RetrievalIndex() {}
QueryEncoder::~QueryEncoder() {}

// ---------------------------------------------------------------- shared helpers

static bool rankedBefore(const Candidate& a, const Candidate& b) {
    return a.score > b.score || (a.score == b.score && a.id < b.id);
}

void toCandidates(const NeighborArray& closestFirst, CandidateArray& out) {
    out.clear();
    out.reserve(closestFirst.size());
    for (int i = 0; i < closestFirst.size(); i++) {
        Candidate c;
        c.id = closestFirst[i].id;
        c.score = distanceToScore(closestFirst[i].dist);
        out.push(c);
    }
    // The input is already in order except where rounding made two scores equal, so
    // insertion sort finishes in about one pass.
    for (int i = 1; i < out.size(); i++) {
        Candidate item = out[i];
        int j = i;
        while (j > 0 && rankedBefore(item, out[j - 1])) {
            out[j] = out[j - 1];
            j--;
        }
        out[j] = item;
    }
}

void checkSearchArgs(const float* query, int queryDim, int k, int dim, int size) {
    if (queryDim != dim)
        throw QueryError("query dimension mismatch: got " + std::to_string(queryDim) +
                         ", index expects " + std::to_string(dim));
    if (!allFinite(query, queryDim)) throw QueryError("query vector contains NaN or Inf");
    if (k <= 0) throw QueryError("K must be positive, got " + std::to_string(k));
    if (k > size)
        throw QueryError("K=" + std::to_string(k) + " exceeds the number of indexed vectors (" +
                         std::to_string(size) + ")");
}

void checkInsert(const EmbeddingStore& store, int id, const float* embedding,
                 const BitArray& present) {
    if (id < 0 || id >= store.size())
        throw Error("insert: id " + std::to_string(id) + " is not a row of the store (N=" +
                    std::to_string(store.size()) + ")");
    const float* row = store.row(id);
    if (embedding != row) {
        for (int i = 0; i < store.dim(); i++)
            if (embedding[i] != row[i])
                throw Error("insert: embedding for id " + std::to_string(id) +
                            " differs from the store row (indexes reference the store)");
    }
    if (present.get(id)) throw Error("insert: id " + std::to_string(id) + " already inserted");
}

// ---------------------------------------------------------------- search engine

SearchEngine::SearchEngine(const MetadataManager& meta, const EmbeddingStore& s,
                           const RetrievalIndex& idx, QueryEncoder* enc) {
    if (meta.size() != s.size())
        throw Error("metadata has " + std::to_string(meta.size()) +
                    " records but the embedding store has " + std::to_string(s.size()) + " rows");
    metadata = &meta;
    store = &s;
    index = &idx;
    encoder = enc;
}

std::string SearchEngine::normalizeQueryText(const std::string& text) {
    std::string out;
    bool pendingSpace = false;
    for (size_t i = 0; i < text.size(); i++) {
        if (std::isspace((unsigned char)text[i])) {
            pendingSpace = !out.empty();
        } else {
            if (pendingSpace) out += ' ';
            out += text[i];
            pendingSpace = false;
        }
    }
    if (out.empty()) throw QueryError("query is empty or whitespace-only");
    return out;
}

void SearchEngine::checkK(int k) const {
    if (k <= 0) throw QueryError("K must be a positive integer, got " + std::to_string(k));
    if (k > index->size())
        throw QueryError("K=" + std::to_string(k) + " exceeds the number of indexed images (N=" +
                         std::to_string(index->size()) + ")");
}

void SearchEngine::prepareVector(const FloatArray& raw, FloatArray& out) const {
    if (raw.size() != store->dim())
        throw QueryError("query embedding dimension mismatch: got " + std::to_string(raw.size()) +
                         ", expected " + std::to_string(store->dim()));
    for (int i = 0; i < raw.size(); i++)
        if (!std::isfinite(raw[i]))
            throw QueryError(std::string("query embedding contains ") +
                             (std::isnan(raw[i]) ? "NaN" : "Inf") + " at element " +
                             std::to_string(i));
    out = raw;
    if (norm(out.data(), out.size()) == 0.0) throw QueryError("query embedding has zero norm");
    normalize(out.data(), out.size());
}

void SearchEngine::embedQuery(const std::string& text, FloatArray& out) const {
    std::string clean = normalizeQueryText(text);
    if (encoder == nullptr) throw QueryError("no text encoder configured for text queries");
    FloatArray raw;
    encoder->encode(clean, raw);
    prepareVector(raw, out);
}

void SearchEngine::searchEmbedding(const FloatArray& unitQuery, int k, SearchResponse& out) const {
    checkK(k);
    double start = nowMs();
    index->search(unitQuery.data(), unitQuery.size(), k, out.hits);
    out.retrievalMs = nowMs() - start;
}

void SearchEngine::search(const std::string& text, int k, SearchResponse& out) const {
    checkK(k);  // fail fast, before running the encoder
    FloatArray query;
    embedQuery(text, query);
    searchEmbedding(query, k, out);
}

const ImageRecord& SearchEngine::record(int id) const {
    return metadata->at(id);
}
