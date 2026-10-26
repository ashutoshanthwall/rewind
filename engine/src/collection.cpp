// Named indexes with time-range filtering (collection.h). Rewind, step 7.1.
#include "collection.h"

#include <cmath>

#include "storage.h"

NamedIndex::NamedIndex(const std::string& name, const std::string& model) {
    if (name.empty()) throw Error("index name must not be empty");
    indexName = name;
    modelName = model;
    everything = nullptr;
    times = nullptr;
    numTimes = 0;
    built = false;
}

NamedIndex::~NamedIndex() {
    delete everything;
    delete[] times;
}

void NamedIndex::loadVectors(const std::string& binPath, const std::string& manifestPath, int dim,
                             double normTolerance) {
    std::lock_guard<std::mutex> guard(lock);
    built = false;
    delete everything;
    everything = nullptr;
    delete[] times;  // times belong to the old rows
    times = nullptr;
    numTimes = 0;
    vectors.load(binPath, manifestPath, dim, modelName, normTolerance);
    everything = new BruteForceIndex(vectors);
    everything->insertAll();
}

void NamedIndex::setVectors(int dim, long long rows, const float* data, double normTolerance) {
    std::lock_guard<std::mutex> guard(lock);
    built = false;
    delete everything;
    everything = nullptr;
    delete[] times;  // times belong to the old rows
    times = nullptr;
    numTimes = 0;
    vectors.setRows(dim, rows, data, normTolerance);
    everything = new BruteForceIndex(vectors);
    everything->insertAll();
}

void NamedIndex::setTimes(const double* t, long long n) {
    std::lock_guard<std::mutex> guard(lock);
    if (n != vectors.size())
        throw Error(indexName + ": " + std::to_string(n) + " times for " +
                    std::to_string(vectors.size()) + " rows");
    for (long long i = 0; i < n; i++)
        if (!std::isfinite(t[i])) throw Error(indexName + ": time of row " + std::to_string(i) + " is not finite");
    delete[] times;
    times = new double[n > 0 ? n : 1];
    for (long long i = 0; i < n; i++) times[i] = t[i];
    numTimes = (int)n;
}

double NamedIndex::build(const HnswParams& params) {
    std::lock_guard<std::mutex> guard(lock);
    if (vectors.size() == 0) throw Error(indexName + ": no vectors to build on");
    built = false;
    graph.setup(vectors, params);
    double seconds = IndexManager::build(graph);
    IndexManager::validate(graph);
    built = true;
    return seconds;
}

void NamedIndex::save(const std::string& path) const {
    std::lock_guard<std::mutex> guard(lock);
    if (!built) throw Error(indexName + ": no graph to save");
    IndexManager::save(graph, modelName, path);
}

void NamedIndex::load(const std::string& path) {
    std::lock_guard<std::mutex> guard(lock);
    if (vectors.size() == 0) throw Error(indexName + ": load the vectors before the graph");
    built = false;
    IndexManager::load(path, vectors, modelName, graph);
    built = true;
}

void NamedIndex::requireReady(bool needGraph) const {
    if (vectors.size() == 0) throw Error(indexName + ": no vectors loaded");
    if (needGraph && !built) throw Error(indexName + ": no HNSW graph (build or load it first)");
}

void NamedIndex::rangeIds(double t0, double t1, IntArray& out) const {
    if (numTimes != vectors.size())
        throw QueryError(indexName + ": time filter requested but no times are set");
    if (!(t0 <= t1)) throw QueryError(indexName + ": time range start is after its end (or NaN)");
    out.clear();
    for (int i = 0; i < numTimes; i++)
        if (times[i] >= t0 && times[i] <= t1) out.push(i);
}

void NamedIndex::exactOver(const float* query, int queryDim, int k, const IntArray& ids,
                           CandidateArray& out) const {
    BruteForceIndex subset(vectors);
    for (int i = 0; i < ids.size(); i++) subset.insert(ids[i], vectors.row(ids[i]));
    subset.search(query, queryDim, k < ids.size() ? k : ids.size(), out);
}

void NamedIndex::search(const float* query, int queryDim, int k, int efSearch, bool hasRange,
                        double t0, double t1, const FilterParams& filter, CandidateArray& out,
                        SearchReport& report) {
    std::lock_guard<std::mutex> guard(lock);
    requireReady(true);
    if (filter.oversample < 1) throw QueryError("filter oversample must be >= 1");
    if (!(filter.minFraction >= 0.0 && filter.minFraction <= 1.0))
        throw QueryError("filter minFraction must be in [0, 1]");
    int n = vectors.size();
    if (!hasRange) {
        int kk = k > n ? n : k;  // an index with fewer rows than k returns all of them
        graph.setEfSearch(efSearch);
        graph.search(query, queryDim, kk, out);
        report.method = "hnsw";
        report.inRange = -1;
        report.candidates = kk;
        return;
    }
    if (k <= 0) throw QueryError("K must be positive, got " + std::to_string(k));
    IntArray ids;
    rangeIds(t0, t1, ids);
    report.inRange = ids.size();
    report.candidates = 0;
    if (ids.size() == 0) {
        out.clear();
        report.method = "empty";
        return;
    }
    int want = k < ids.size() ? k : ids.size();
    long long over = (long long)k * filter.oversample;
    if (ids.size() <= over) {
        exactOver(query, queryDim, k, ids, out);
        report.method = "exact_small";
        return;
    }
    int kc = over < n ? (int)over : n;
    CandidateArray cand;
    graph.setEfSearch(efSearch);
    graph.search(query, queryDim, kc, cand);
    report.candidates = kc;
    out.clear();
    for (int i = 0; i < cand.size() && out.size() < want; i++)
        if (times[cand[i].id] >= t0 && times[cand[i].id] <= t1) out.push(cand[i]);
    if (out.size() >= filter.minFraction * want) {
        report.method = "hnsw_filtered";
        return;
    }
    exactOver(query, queryDim, k, ids, out);
    report.method = "exact_fallback";
}

void NamedIndex::exact(const float* query, int queryDim, int k, bool hasRange, double t0, double t1,
                       CandidateArray& out, SearchReport& report) {
    std::lock_guard<std::mutex> guard(lock);
    requireReady(false);
    report.method = "exact";
    report.candidates = 0;
    if (!hasRange) {
        report.inRange = -1;
        everything->search(query, queryDim, k > vectors.size() ? vectors.size() : k, out);
        return;
    }
    if (k <= 0) throw QueryError("K must be positive, got " + std::to_string(k));
    IntArray ids;
    rangeIds(t0, t1, ids);
    report.inRange = ids.size();
    if (ids.size() == 0) {
        out.clear();
        return;
    }
    exactOver(query, queryDim, k, ids, out);
}

const std::string& NamedIndex::name() const { return indexName; }
const std::string& NamedIndex::model() const { return modelName; }
int NamedIndex::size() const { return vectors.size(); }
int NamedIndex::dim() const { return vectors.dim(); }
bool NamedIndex::hasGraph() const { return built; }
const EmbeddingStore& NamedIndex::store() const { return vectors; }

double NamedIndex::timeOf(int id) const {
    if (id < 0 || id >= numTimes) throw Error(indexName + ": no time for id " + std::to_string(id));
    return times[id];
}

// ---------------------------------------------------------------- collection

IndexCollection::IndexCollection() {
    items = nullptr;
    count = 0;
    cap = 0;
}

IndexCollection::~IndexCollection() {
    for (int i = 0; i < count; i++) delete items[i];
    delete[] items;
}

NamedIndex& IndexCollection::add(const std::string& name, const std::string& model) {
    if (has(name)) throw Error("an index named '" + name + "' already exists");
    if (count == cap) {
        int newCap = cap == 0 ? 4 : cap * 2;
        NamedIndex** bigger = new NamedIndex*[newCap];
        for (int i = 0; i < count; i++) bigger[i] = items[i];
        delete[] items;
        items = bigger;
        cap = newCap;
    }
    items[count] = new NamedIndex(name, model);
    return *items[count++];
}

NamedIndex& IndexCollection::get(const std::string& name) {
    for (int i = 0; i < count; i++)
        if (items[i]->name() == name) return *items[i];
    throw Error("no index named '" + name + "'");
}

bool IndexCollection::has(const std::string& name) const {
    for (int i = 0; i < count; i++)
        if (items[i]->name() == name) return true;
    return false;
}

int IndexCollection::size() const { return count; }

const std::string& IndexCollection::nameAt(int i) const {
    if (i < 0 || i >= count) throw Error("collection index " + std::to_string(i) + " out of range");
    return items[i]->name();
}
