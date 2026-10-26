// Member 2: vector math, sorting, heap, top-K (gate G6), brute force, search engine.
#include <cmath>
#include <iostream>

#include "hnsw.h"
#include "search.h"
#include "tests.h"

static bool near(double a, double b, double tolerance) {
    return std::fabs(a - b) <= tolerance;
}

void testVectorMath() {
    float a[] = {1, 2, 3, 4}, b[] = {5, 6, 7, 8};
    check(dot(a, b, 4) == 70.0f, "dot by hand: 5 + 12 + 21 + 32 = 70");
    float c[] = {0.5f, -0.25f}, d[] = {-2.0f, 4.0f};
    check(dot(c, d, 2) == -2.0f, "dot with negatives: -1 - 1 = -2");
    float e[] = {3, 4};
    check(norm(e, 2) == 5.0, "norm of (3, 4) is 5");
    float e1[] = {1, 0}, e2[] = {0, 1}, m1[] = {-1, 0};
    check(distance(e1, e1, 2) == 0.0f, "identical vectors: distance 0");
    check(distance(e1, e2, 2) == 1.0f, "orthogonal: distance 1");
    check(distance(e1, m1, 2) == 2.0f, "opposite: distance 2");
    check(distanceToScore(distance(e1, e2, 2)) == 0.0f, "score = 1 - distance");
    normalize(e, 2);
    check(near(e[0], 0.6, 1e-7) && near(e[1], 0.8, 1e-7), "normalize (3, 4) -> (0.6, 0.8)");

    FloatArray v = randomUnitVectors(200, 512, 7);
    bool idempotent = true;
    for (int i = 0; i < 200; i++) {
        FloatArray once(512, 0.0f);
        for (int j = 0; j < 512; j++) once[j] = v[i * 512 + j];
        FloatArray twice = once;
        normalize(twice.data(), 512);
        idempotent = idempotent && near(norm(once.data(), 512), 1.0, 1e-6);
        for (int j = 0; j < 512; j++) idempotent = idempotent && near(once[j], twice[j], 1e-7);
    }
    check(idempotent, "normalising a unit vector changes nothing");

    float zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float withNan[] = {1.0f, NAN};
    int rejected = 0;
    try {
        normalize(zero, 8);
    } catch (const Error&) {
        rejected++;
    }
    try {
        normalize(withNan, 2);
    } catch (const Error&) {
        rejected++;
    }
    check(rejected == 2, "normalize rejects zero and NaN vectors");
}

static bool isSorted(const Neighbor* a, int n) {
    for (int i = 1; i < n; i++)
        if (closer(a[i], a[i - 1])) return false;
    return true;
}

void testSorting() {
    Random64 rng;
    rng.seed(3);
    bool ok = true;
    for (int n = 0; n < 300; n += 7) {
        NeighborArray a;
        for (int i = 0; i < n; i++) {
            Neighbor x;
            x.dist = (float)(rng.next() % 50);  // many equal distances: ties broken by id
            x.id = i;
            a.push(x);
        }
        sortNeighbors(a.data(), a.size());
        ok = ok && isSorted(a.data(), a.size());
    }
    check(ok, "random inputs with many ties come out sorted");

    // Adversarial inputs: already sorted, reversed, all equal distances.
    NeighborArray sorted, reversed, equal;
    for (int i = 0; i < 5000; i++) {
        Neighbor x;
        x.dist = (float)i;
        x.id = i;
        sorted.push(x);
        x.dist = (float)(5000 - i);
        reversed.push(x);
        x.dist = 1.0f;
        x.id = 5000 - i;
        equal.push(x);
    }
    sortNeighbors(sorted.data(), sorted.size());
    sortNeighbors(reversed.data(), reversed.size());
    sortNeighbors(equal.data(), equal.size());
    check(isSorted(sorted.data(), 5000) && isSorted(reversed.data(), 5000) &&
              isSorted(equal.data(), 5000),
          "sorted, reversed and all-equal inputs");

    NeighborHeap minHeap(true), maxHeap(false);
    for (int i = 0; i < 100; i++) {
        Neighbor x;
        x.dist = (float)((i * 37) % 100);
        x.id = i;
        minHeap.push(x);
        maxHeap.push(x);
    }
    bool minOrder = true, maxOrder = true;
    Neighbor last = minHeap.top();
    while (!minHeap.empty()) {
        minOrder = minOrder && !closer(minHeap.top(), last);
        last = minHeap.top();
        minHeap.pop();
    }
    last = maxHeap.top();
    while (!maxHeap.empty()) {
        maxOrder = maxOrder && !closer(last, maxHeap.top());
        last = maxHeap.top();
        maxHeap.pop();
    }
    check(minOrder, "min-heap pops closest first");
    check(maxOrder, "max-heap pops farthest first");
}

void testTopK() {
    TopK t(3);
    float dists[] = {0.5f, 0.1f, 0.9f, 0.1f, 0.3f};
    int ids[] = {4, 9, 1, 2, 7};
    for (int i = 0; i < 5; i++) {
        Neighbor n;
        n.dist = dists[i];
        n.id = ids[i];
        t.push(n);
    }
    NeighborArray r;
    t.extractSorted(r);
    check(r.size() == 3 && r[0].id == 2 && r[1].id == 9 && r[2].id == 7,
          "top-3 by hand; tie at 0.1 broken by smaller id");

    // G6: top-K against a full sort, 100 trials of 1000 vectors, with ~5% exact duplicate
    // rows so that exact distance ties happen.
    const int N = 1000, D = 32;
    Random64 rng;
    rng.seed(123);
    int exact = 0;
    for (int trial = 0; trial < 100; trial++) {
        FloatArray data = randomUnitVectors(N, D, 1000 + trial);
        for (int dup = 0; dup < 50; dup++) {
            int src = (int)(rng.next() % N), dst = (int)(rng.next() % N);
            for (int j = 0; j < D; j++) data[dst * D + j] = data[src * D + j];
        }
        FloatArray q = randomUnitVectors(1, D, 5000 + trial);
        int k = 1 + trial % 50;
        NeighborArray all;
        TopK top(k);
        for (int i = 0; i < N; i++) {
            Neighbor n;
            n.dist = distance(q.data(), data.data() + i * D, D);
            n.id = i;
            all.push(n);
            top.push(n);
        }
        NeighborArray got;
        top.extractSorted(got);
        sortNeighbors(all.data(), all.size());
        bool same = got.size() == k;
        for (int i = 0; i < k && same; i++) same = got[i].id == all[i].id && got[i].dist == all[i].dist;
        if (same) exact++;
    }
    check(exact == 100, "G6: top-K equals the full sort in every trial");
    std::cout << "  [G6] top-K vs full sort: " << exact << "/100 trials exact\n";
}

static bool searchThrows(const RetrievalIndex& index, const float* q, int dim, int k) {
    try {
        CandidateArray out;
        index.search(q, dim, k, out);
    } catch (const QueryError&) {
        return true;
    }
    return false;
}

void testBruteForce() {
    const int N = 500, D = 16;
    EmbeddingStore store;
    store.setData(D, randomUnitVectors(N, D, 11), 1e-5);
    BruteForceIndex bf(store);
    bf.insertAll();
    FloatArray q = randomUnitVectors(20, D, 12);
    bool ok = true;
    CandidateArray got;
    for (int t = 0; t < 20; t++) {
        const float* qs = q.data() + t * D;
        bf.search(qs, D, 10, got);
        // Reference: score every vector, then pick the best 10 by repeated selection.
        float* scores = new float[N];
        bool* used = new bool[N];
        for (int i = 0; i < N; i++) {
            scores[i] = 1.0f - (1.0f - dot(qs, store.row(i), D));
            used[i] = false;
        }
        for (int r = 0; r < 10; r++) {
            int best = -1;
            for (int i = 0; i < N; i++)
                if (!used[i] && (best < 0 || scores[i] > scores[best])) best = i;
            used[best] = true;
            ok = ok && got[r].id == best && got[r].score == scores[best];
        }
        delete[] scores;
        delete[] used;
    }
    check(ok, "brute force matches a full scan, ids and scores");

    FloatArray v = randomUnitVectors(6, 4, 3);
    int dups[] = {4, 1, 5};  // rows 1, 4, 5 become copies of row 2
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++) v[dups[i] * 4 + j] = v[2 * 4 + j];
    EmbeddingStore tied;
    tied.setData(4, v, 1e-5);
    BruteForceIndex tbf(tied);
    tbf.insertAll();
    tbf.search(tied.row(2), 4, 4, got);
    check(got[0].id == 1 && got[1].id == 2 && got[2].id == 4 && got[3].id == 5 &&
              got[0].score == got[3].score,
          "exact ties come back by ascending id");

    EmbeddingStore small;
    small.setData(4, randomUnitVectors(5, 4, 1), 1e-5);
    BruteForceIndex sbf(small);
    sbf.insertAll();
    FloatArray q1 = randomUnitVectors(1, 4, 2);
    check(searchThrows(sbf, q1.data(), 4, 0) && searchThrows(sbf, q1.data(), 4, -3) &&
              searchThrows(sbf, q1.data(), 4, 6),
          "K <= 0 and K > N rejected");
    check(!searchThrows(sbf, q1.data(), 4, 5), "K = N allowed");
    FloatArray q8 = randomUnitVectors(1, 8, 2);
    check(searchThrows(sbf, q8.data(), 8, 1), "wrong query dimension rejected");
    FloatArray badQ = q1;
    badQ[1] = NAN;
    check(searchThrows(sbf, badQ.data(), 4, 1), "NaN query rejected");
    badQ[1] = INFINITY;
    check(searchThrows(sbf, badQ.data(), 4, 1), "Inf query rejected");
    int insertErrors = 0;
    try {
        sbf.insert(0, small.row(0));  // already inserted
    } catch (const Error&) {
        insertErrors++;
    }
    try {
        sbf.insert(9, small.row(0));  // not a row of the store
    } catch (const Error&) {
        insertErrors++;
    }
    try {
        BruteForceIndex fresh(small);
        fresh.insert(1, small.row(2));  // wrong vector for this id
    } catch (const Error&) {
        insertErrors++;
    }
    check(insertErrors == 3, "bad inserts rejected");
}

// A fake encoder: returns a fixed vector and remembers the text it was given.
class FakeEncoder : public QueryEncoder {
public:
    FloatArray vector;
    std::string lastText;
    void encode(const std::string& text, FloatArray& out) {
        lastText = text;
        out = vector;
    }
};

// The QueryError message from a search, or "" if it worked.
static std::string queryError(const SearchEngine& e, const std::string& text, int k) {
    try {
        SearchResponse r;
        e.search(text, k, r);
    } catch (const QueryError& ex) {
        return ex.what();
    }
    return "";
}

static bool has(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

void testSearchEngine() {
    check(SearchEngine::normalizeQueryText("  a   dog\tin\n snow  ") == "a dog in snow",
          "query text: whitespace trimmed and collapsed");
    bool threw = false;
    try {
        SearchEngine::normalizeQueryText(" \t\n ");
    } catch (const QueryError&) {
        threw = true;
    }
    check(threw, "whitespace-only query rejected");

    const int N = 300, D = 16;
    EmbeddingStore store;
    store.setData(D, randomUnitVectors(N, D, 21), 1e-5);
    MetadataManager meta;
    meta.allocate(N);
    for (int i = 0; i < N; i++) {
        meta.edit(i).internalId = i;
        meta.edit(i).imageId = "img" + std::to_string(i);
    }
    BruteForceIndex bf(store);
    bf.insertAll();
    HNSWIndex hnsw;
    HnswParams p;
    p.M = 8;
    p.efConstruction = 64;
    p.efSearch = N;  // ef >= N: HNSW must give the exact answer
    hnsw.setup(store, p);
    for (int i = 0; i < N; i++) hnsw.insert(i, store.row(i));

    FakeEncoder enc;
    enc.vector = randomUnitVectors(1, D, 5);
    for (int i = 0; i < D; i++) enc.vector[i] *= 3.0f;  // not unit length: engine must normalise
    SearchEngine a(meta, store, bf, &enc), b(meta, store, hnsw, &enc);
    SearchResponse ra, rb;
    a.search("  a dog  ", 5, ra);
    b.search("a dog", 5, rb);
    check(enc.lastText == "a dog", "encoder receives the cleaned text");
    bool same = ra.hits.size() == 5;
    for (int i = 0; i < 5 && same; i++)
        same = ra.hits[i].id == rb.hits[i].id &&
               a.record(ra.hits[i].id).imageId == "img" + std::to_string(ra.hits[i].id) &&
               ra.hits[i].score >= -1.0f && ra.hits[i].score <= 1.0f + 1e-6f;
    check(same, "both indexes give the same results, mapped to metadata");
    check(ra.retrievalMs >= 0.0, "retrieval time recorded");

    check(has(queryError(a, "   ", 5), "empty"), "error: empty query");
    check(has(queryError(a, "dog", 0), "positive"), "error: K = 0");
    check(has(queryError(a, "dog", 301), "exceeds"), "error: K > N");
    enc.vector = randomUnitVectors(1, 8, 1);
    check(has(queryError(a, "dog", 5), "dimension mismatch"), "error: wrong embedding size");
    enc.vector = randomUnitVectors(1, D, 1);
    enc.vector[3] = NAN;
    check(has(queryError(a, "dog", 5), "NaN"), "error: NaN embedding");
    enc.vector[3] = INFINITY;
    check(has(queryError(a, "dog", 5), "Inf"), "error: Inf embedding");
    enc.vector = FloatArray(D, 0.0f);
    check(has(queryError(a, "dog", 5), "zero norm"), "error: zero embedding");
}
