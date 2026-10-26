// Member 3: HNSW graph invariants, exactness at high ef, recall (gate G8), determinism.
#include <iomanip>
#include <iostream>

#include "hnsw.h"
#include "tests.h"

static HnswParams params(int M, int efC, int efS, bool heuristic) {
    HnswParams p;
    p.M = M;
    p.efConstruction = efC;
    p.efSearch = efS;
    p.seed = 42;
    p.selectHeuristic = heuristic;
    return p;
}

static void buildAll(HNSWIndex& h, const EmbeddingStore& store, const HnswParams& p) {
    h.setup(store, p);
    for (int i = 0; i < store.size(); i++) h.insert(i, store.row(i));
}

// Degree caps, no self loops, no duplicate links, links only to nodes on that layer,
// in-degrees equal to a full recount, and every node reachable on layer 0.
static bool invariantsHold(const HNSWIndex& h, int n) {
    for (int id = 0; id < n; id++) {
        if (!h.contains(id)) return false;
        for (int l = 0; l <= h.levelOf(id); l++) {
            const IntArray& nb = h.neighbors(id, l);
            if (nb.size() > h.maxDegree(l)) return false;
            for (int i = 0; i < nb.size(); i++) {
                if (nb[i] == id || h.levelOf(nb[i]) < l) return false;
                for (int j = 0; j < i; j++)
                    if (nb[j] == nb[i]) return false;
            }
        }
    }
    for (int l = 0; l <= h.maxLevel(); l++) {
        IntArray count(n, 0);
        for (int id = 0; id < n; id++)
            if (h.levelOf(id) >= l) {
                const IntArray& nb = h.neighbors(id, l);
                for (int i = 0; i < nb.size(); i++) count[nb[i]]++;
            }
        for (int id = 0; id < n; id++)
            if (h.levelOf(id) >= l && count[id] != h.inDegreeOf(id, l)) return false;
    }
    BitArray seen(n, false);
    IntArray queue;
    queue.push(h.entryPoint());
    seen.set(h.entryPoint(), true);
    for (int head = 0; head < queue.size(); head++) {
        const IntArray& nb = h.neighbors(queue[head], 0);
        for (int i = 0; i < nb.size(); i++)
            if (!seen.get(nb[i])) {
                seen.set(nb[i], true);
                queue.push(nb[i]);
            }
    }
    return queue.size() == n;
}

// G8 part 1: with efSearch = N the search visits the whole graph, so HNSW must return
// exactly what brute force returns.
static void exactWithHighEf(int n, int d, int k, bool heuristic) {
    EmbeddingStore store;
    store.setData(d, randomUnitVectors(n, d, 99 + n), 1e-5);
    BruteForceIndex bf(store);
    bf.insertAll();
    HNSWIndex h;
    buildAll(h, store, params(4, 16, n, heuristic));
    check(invariantsHold(h, n), "graph invariants, N=" + std::to_string(n));
    FloatArray qs = randomUnitVectors(50, d, 7 + n);
    int identical = 0;
    CandidateArray a, b;
    for (int t = 0; t < 50; t++) {
        bf.search(qs.data() + t * d, d, k, a);
        h.search(qs.data() + t * d, d, k, b);
        bool same = a.size() == b.size();
        for (int i = 0; i < a.size() && same; i++) same = a[i].id == b[i].id && a[i].score == b[i].score;
        if (same) identical++;
    }
    bool selfOk = true;
    for (int i = 0; i < n; i++) {  // every stored vector as its own query
        bf.search(store.row(i), d, k, a);
        h.search(store.row(i), d, k, b);
        for (int r = 0; r < a.size(); r++) selfOk = selfOk && a[r].id == b[r].id;
    }
    check(identical == 50 && selfOk, "HNSW == brute force at efSearch = N, N=" + std::to_string(n));
    std::cout << "  [G8] N=" << n << " D=" << d << " k=" << k << " efSearch=" << n
              << (heuristic ? " heuristic" : " naive") << ": " << identical
              << "/50 random queries identical to brute force (+" << n << " self-queries)\n";
}

void testHnsw() {
    exactWithHighEf(10, 8, 5, true);
    exactWithHighEf(10, 8, 10, true);
    exactWithHighEf(10, 8, 10, false);
    exactWithHighEf(100, 8, 10, true);
    exactWithHighEf(100, 8, 10, false);

    // G8 part 2: recall on 5000 random vectors grows with efSearch.
    const int N = 5000, D = 32;
    EmbeddingStore store;
    store.setData(D, randomUnitVectors(N, D, 2024), 1e-5);
    BruteForceIndex bf(store);
    bf.insertAll();
    FloatArray qs = randomUnitVectors(200, D, 77);
    int sweep[] = {20, 40, 80, 120};
    for (int variant = 0; variant < 2; variant++) {
        bool heuristic = variant == 0;
        HNSWIndex h;
        buildAll(h, store, params(16, 200, 20, heuristic));
        check(invariantsHold(h, N), "graph invariants, N=5000");
        std::cout << "  [G8] N=5000 D=32 M=16 efC=200 " << (heuristic ? "heuristic" : "naive    ")
                  << " Recall@10:";
        double previous = 0.0;
        bool growing = true;
        CandidateArray a, b;
        for (int s = 0; s < 4; s++) {
            h.setEfSearch(sweep[s]);
            int hits = 0;
            for (int t = 0; t < 200; t++) {
                bf.search(qs.data() + t * D, D, 10, a);
                h.search(qs.data() + t * D, D, 10, b);
                for (int i = 0; i < b.size(); i++)
                    for (int j = 0; j < a.size(); j++)
                        if (b[i].id == a[j].id) hits++;
            }
            double recall = hits / 2000.0;
            std::cout << "  ef=" << sweep[s] << ":" << std::fixed << std::setprecision(4) << recall;
            growing = growing && recall + 1e-12 >= previous;
            previous = recall;
        }
        std::cout << "\n";
        check(growing, "recall never drops as efSearch grows");
        check(previous >= 0.99, "recall at efSearch=120 is at least 0.99");
    }

    // Same seed, same data: the same graph.
    EmbeddingStore s800;
    s800.setData(16, randomUnitVectors(800, 16, 5), 1e-5);
    HNSWIndex x, y;
    buildAll(x, s800, params(8, 64, 32, true));
    buildAll(y, s800, params(8, 64, 32, true));
    bool same = x.entryPoint() == y.entryPoint();
    for (int i = 0; i < 800 && same; i++) {
        same = x.levelOf(i) == y.levelOf(i);
        for (int l = 0; l <= x.levelOf(i) && same; l++) same = x.neighbors(i, l).equals(y.neighbors(i, l));
    }
    check(same, "fixed seed gives an identical graph");

    // P(level >= 1) = 1/M = 1/16: about 312 of 5000 nodes; allow a wide binomial band.
    EmbeddingStore s5000;
    s5000.setData(8, randomUnitVectors(5000, 8, 3), 1e-5);
    HNSWIndex levels;
    buildAll(levels, s5000, params(16, 32, 16, true));
    int upper = 0;
    for (int i = 0; i < 5000; i++)
        if (levels.levelOf(i) >= 1) upper++;
    check(upper > 240 && upper < 390, "level distribution: " + std::to_string(upper) + " nodes above layer 0");

    EmbeddingStore tiny;
    tiny.setData(4, randomUnitVectors(20, 4, 1), 1e-5);
    bool threw = false;
    try {
        HNSWIndex bad;
        bad.setup(tiny, params(1, 10, 10, true));
    } catch (const Error&) {
        threw = true;
    }
    check(threw, "M < 2 rejected");
    HNSWIndex h;
    buildAll(h, tiny, params(4, 10, 10, true));
    FloatArray q = randomUnitVectors(1, 4, 9);
    CandidateArray out;
    int errors = 0;
    try {
        h.search(q.data(), 4, 0, out);
    } catch (const QueryError&) {
        errors++;
    }
    try {
        h.search(q.data(), 4, 21, out);
    } catch (const QueryError&) {
        errors++;
    }
    try {
        h.insert(3, tiny.row(3));
    } catch (const Error&) {
        errors++;
    }
    check(errors == 3, "bad K and duplicate insert rejected");
    h.search(q.data(), 4, 20, out);
    check(out.size() == 20, "k larger than efSearch: ef becomes k");
}
