#include "hnsw.h"

#include <cmath>

static const double TWO_POW_MINUS_53 = 1.0 / 9007199254740992.0;

HNSWIndex::HNSWIndex() {
    vectors = nullptr;
    firstRow = nullptr;
    dims = 0;
    mL = 0.0;
    nodes = nullptr;
    numNodes = 0;
    count = 0;
    entry = -1;
    topLevel = -1;
    visited = nullptr;
    epoch = 0;
    distCount = 0;
}

HNSWIndex::~HNSWIndex() {
    freeGraph();
}

void HNSWIndex::freeGraph() {
    for (int i = 0; i < numNodes; i++) {
        delete[] nodes[i].links;
        delete[] nodes[i].inDegree;
    }
    delete[] nodes;
    delete[] visited;
    nodes = nullptr;
    visited = nullptr;
    numNodes = 0;
}

void HNSWIndex::setup(const EmbeddingStore& store, const HnswParams& params) {
    if (params.M < 2) throw Error("HNSW M must be >= 2, got " + std::to_string(params.M));
    if (params.efConstruction < 1) throw Error("HNSW efConstruction must be >= 1");
    if (params.efSearch < 1) throw Error("HNSW efSearch must be >= 1");
    freeGraph();
    vectors = &store;
    firstRow = store.size() > 0 ? store.row(0) : nullptr;
    dims = store.dim();
    settings = params;
    mL = 1.0 / std::log((double)params.M);
    rng.seed(params.seed);
    numNodes = store.size();
    nodes = new Node[numNodes > 0 ? numNodes : 1];
    for (int i = 0; i < numNodes; i++) {
        nodes[i].level = -1;
        nodes[i].links = nullptr;
        nodes[i].inDegree = nullptr;
    }
    present.reset(numNodes, false);
    visited = new unsigned int[numNodes > 0 ? numNodes : 1];
    for (int i = 0; i < numNodes; i++) visited[i] = 0;
    epoch = 0;
    distCount = 0;
    count = 0;
    entry = -1;
    topLevel = -1;
}

void HNSWIndex::setEfSearch(int ef) {
    if (ef < 1) throw Error("efSearch must be >= 1, got " + std::to_string(ef));
    settings.efSearch = ef;
}

float HNSWIndex::dist(const float* q, int id) const {
    distCount++;
    return distance(q, firstRow + (long long)id * dims, dims);
}

int HNSWIndex::randomLevel() {
    double u = ((double)(rng.next() >> 11) + 1.0) * TWO_POW_MINUS_53;
    return (int)std::floor(-std::log(u) * mL);
}

void HNSWIndex::nextVisitEpoch() const {
    epoch++;
    if (epoch == 0) {
        for (int i = 0; i < numNodes; i++) visited[i] = 0;
        epoch = 1;
    }
}

int HNSWIndex::maxDegree(int layer) const {
    return layer == 0 ? 2 * settings.M : settings.M;
}

Neighbor HNSWIndex::greedyClosest(const float* q, Neighbor current, int layer) const {
    bool improved = true;
    while (improved) {
        improved = false;
        Neighbor best = current;
        const IntArray& links = nodes[current.id].links[layer];
        for (int i = 0; i < links.size(); i++) {
            Neighbor n;
            n.dist = dist(q, links[i]);
            n.id = links[i];
            if (closer(n, best)) best = n;
        }
        if (best.id != current.id) {
            current = best;
            improved = true;
        }
    }
    return current;
}

void HNSWIndex::searchLayer(const float* q, const NeighborArray& entryPoints, int ef, int layer,
                            NeighborArray& out) const {
    nextVisitEpoch();
    NeighborHeap candidates(true);
    NeighborHeap results(false);
    for (int i = 0; i < entryPoints.size(); i++) {
        visited[entryPoints[i].id] = epoch;
        candidates.push(entryPoints[i]);
        results.push(entryPoints[i]);
        if (results.size() > ef) results.pop();
    }
    while (!candidates.empty()) {
        Neighbor c = candidates.top();
        if (closer(results.top(), c)) break;
        candidates.pop();
        const IntArray& links = nodes[c.id].links[layer];
        for (int i = 0; i < links.size(); i++) {
            int e = links[i];
            if (visited[e] == epoch) continue;
            visited[e] = epoch;
            Neighbor n;
            n.dist = dist(q, e);
            n.id = e;
            if (results.size() < ef || closer(n, results.top())) {
                candidates.push(n);
                results.push(n);
                if (results.size() > ef) results.pop();
            }
        }
    }
    int n = results.size();
    out.clear();
    out.reserve(n);
    for (int i = 0; i < n; i++) out.push(Neighbor());
    for (int i = n - 1; i >= 0; i--) {
        out[i] = results.top();
        results.pop();
    }
}

void HNSWIndex::selectNeighbors(const NeighborArray& candidates, int m, NeighborArray& out) const {
    out.clear();
    if (!settings.selectHeuristic) {
        for (int i = 0; i < candidates.size() && i < m; i++) out.push(candidates[i]);
        return;
    }
    NeighborArray discarded;
    out.reserve(m);
    for (int i = 0; i < candidates.size(); i++) {
        if (out.size() >= m) break;
        const Neighbor& e = candidates[i];
        const float* ev = firstRow + (long long)e.id * dims;
        bool good = true;
        for (int r = 0; r < out.size(); r++) {
            distCount++;
            if (distance(ev, firstRow + (long long)out[r].id * dims, dims) < e.dist) {
                good = false;
                break;
            }
        }
        if (good)
            out.push(e);
        else if (settings.keepPrunedConnections)
            discarded.push(e);
    }
    for (int i = 0; i < discarded.size() && out.size() < m; i++) out.push(discarded[i]);
}

static bool isKept(const NeighborArray& kept, int id) {
    for (int i = 0; i < kept.size(); i++)
        if (kept[i].id == id) return true;
    return false;
}

void HNSWIndex::shrinkNeighbors(int id, int layer) {
    IntArray& links = nodes[id].links[layer];
    const float* base = firstRow + (long long)id * dims;
    NeighborArray cands;
    cands.reserve(links.size());
    for (int i = 0; i < links.size(); i++) {
        Neighbor n;
        n.dist = dist(base, links[i]);
        n.id = links[i];
        cands.push(n);
    }
    sortNeighbors(cands.data(), cands.size());
    NeighborArray kept;
    selectNeighbors(cands, maxDegree(layer), kept);
    if (settings.protectLastInEdge) {
        for (int i = 0; i < cands.size(); i++) {
            const Neighbor& c = cands[i];
            if (isKept(kept, c.id) || nodes[c.id].inDegree[layer] > 1) continue;
            if (kept.size() < maxDegree(layer)) {
                kept.push(c);
                continue;
            }
            for (int j = kept.size() - 1; j >= 0; j--) {
                if (nodes[kept[j].id].inDegree[layer] > 1) {
                    kept[j] = c;
                    break;
                }
            }
        }
        sortNeighbors(kept.data(), kept.size());
    }
    for (int i = 0; i < links.size(); i++)
        if (!isKept(kept, links[i])) nodes[links[i]].inDegree[layer]--;
    links.clear();
    for (int i = 0; i < kept.size(); i++) links.push(kept[i].id);
}

void HNSWIndex::insert(int id, const float* embedding) {
    checkInsert(*vectors, id, embedding, present);
    const float* q = firstRow + (long long)id * dims;
    int level = randomLevel();
    Node& node = nodes[id];
    node.level = level;
    node.links = new IntArray[level + 1];
    node.inDegree = new int[level + 1];
    for (int l = 0; l <= level; l++) {
        node.links[l].reserve(maxDegree(l) + 1);
        node.inDegree[l] = 0;
    }
    present.set(id, true);
    count++;

    if (entry < 0) {
        entry = id;
        topLevel = level;
        return;
    }

    Neighbor ep;
    ep.dist = dist(q, entry);
    ep.id = entry;
    for (int layer = topLevel; layer > level; layer--) ep = greedyClosest(q, ep, layer);

    NeighborArray entryPoints;
    entryPoints.push(ep);
    NeighborArray found, chosen;
    int startLayer = level < topLevel ? level : topLevel;
    for (int layer = startLayer; layer >= 0; layer--) {
        searchLayer(q, entryPoints, settings.efConstruction, layer, found);
        selectNeighbors(found, settings.M, chosen);
        IntArray& mine = node.links[layer];
        for (int i = 0; i < chosen.size(); i++) {
            mine.push(chosen[i].id);
            nodes[chosen[i].id].inDegree[layer]++;
        }
        for (int i = 0; i < chosen.size(); i++) {
            int other = chosen[i].id;
            IntArray& theirs = nodes[other].links[layer];
            theirs.push(id);
            node.inDegree[layer]++;
            if (theirs.size() > maxDegree(layer)) shrinkNeighbors(other, layer);
        }
        entryPoints = found;
    }
    if (level > topLevel) {
        topLevel = level;
        entry = id;
    }
}

void HNSWIndex::search(const float* query, int queryDim, int k, CandidateArray& out) const {
    checkSearchArgs(query, queryDim, k, dims, count);
    Neighbor ep;
    ep.dist = dist(query, entry);
    ep.id = entry;
    for (int layer = topLevel; layer > 0; layer--) ep = greedyClosest(query, ep, layer);
    NeighborArray start, found;
    start.push(ep);
    int ef = settings.efSearch > k ? settings.efSearch : k;
    searchLayer(query, start, ef, 0, found);
    found.shrink(k);
    toCandidates(found, out);
}

int HNSWIndex::size() const { return count; }
const HnswParams& HNSWIndex::params() const { return settings; }
const EmbeddingStore& HNSWIndex::store() const { return *vectors; }
int HNSWIndex::entryPoint() const { return entry; }
int HNSWIndex::maxLevel() const { return topLevel; }
long long HNSWIndex::distanceComputations() const { return distCount; }
void HNSWIndex::resetDistanceComputations() const { distCount = 0; }

bool HNSWIndex::contains(int id) const {
    return id >= 0 && id < numNodes && present.get(id);
}

int HNSWIndex::levelOf(int id) const {
    return contains(id) ? nodes[id].level : -1;
}

const IntArray& HNSWIndex::neighbors(int id, int layer) const {
    if (!contains(id) || layer < 0 || layer > nodes[id].level)
        throw Error("neighbors: node " + std::to_string(id) + " has no layer " + std::to_string(layer));
    return nodes[id].links[layer];
}

int HNSWIndex::inDegreeOf(int id, int layer) const {
    if (!contains(id) || layer < 0 || layer > nodes[id].level)
        throw Error("inDegreeOf: node " + std::to_string(id) + " has no layer " + std::to_string(layer));
    return nodes[id].inDegree[layer];
}

long long HNSWIndex::memoryBytes() const {
    long long bytes = (long long)numNodes * sizeof(Node) + present.memoryBytes() +
                      (long long)numNodes * sizeof(unsigned int);
    for (int i = 0; i < numNodes; i++) {
        if (nodes[i].level < 0) continue;
        int layers = nodes[i].level + 1;
        bytes += (long long)layers * (sizeof(IntArray) + sizeof(int));
        for (int l = 0; l < layers; l++) bytes += (long long)nodes[i].links[l].capacity() * sizeof(int);
    }
    return bytes;
}

void HNSWIndex::getRandomState(uint64_t* words, int& position) const {
    rng.getState(words, position);
}

void HNSWIndex::restoreNode(int id, int level) {
    if (level < 0) return;
    Node& node = nodes[id];
    node.level = level;
    node.links = new IntArray[level + 1];
    node.inDegree = new int[level + 1];
    for (int l = 0; l <= level; l++) {
        node.links[l].reserve(maxDegree(l) + 1);
        node.inDegree[l] = 0;
    }
    present.set(id, true);
    count++;
}

IntArray& HNSWIndex::restoreLinks(int id, int layer) {
    return nodes[id].links[layer];
}

void HNSWIndex::finishRestore(int entryId, int maxLevelValue, const uint64_t* rngWords,
                              int rngPosition) {
    for (int i = 0; i < numNodes; i++) {
        for (int l = 0; l <= nodes[i].level; l++) {
            const IntArray& links = nodes[i].links[l];
            for (int j = 0; j < links.size(); j++) {
                int e = links[j];
                if (e < 0 || e >= numNodes || nodes[e].level < l)
                    throw Error("restore: link to a node absent from layer " + std::to_string(l));
                nodes[e].inDegree[l]++;
            }
        }
    }
    entry = entryId;
    topLevel = maxLevelValue;
    rng.setState(rngWords, rngPosition);
}
