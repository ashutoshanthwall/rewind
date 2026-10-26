// Rewind (step 7.1): 64-bit store sizes, named indexes, time-range filtering against
// filtered brute force, save/load, and the per-index lock.
#include <climits>
#include <cmath>
#include <thread>

#include "collection.h"
#include "storage.h"
#include "tests.h"

static bool throwsWith(void (*f)(), const std::string& text) {
    try {
        f();
    } catch (const Error& e) {
        return std::string(e.what()).find(text) != std::string::npos;
    }
    return false;
}

void testStoreSizes() {
    // 5 M x 512 floats = 2,560,000,000: past INT_MAX, which the old int multiply overflowed.
    long long n = floatCount(5000000, 512);
    check(n == 2560000000LL, "floatCount(5 M, 512) is exact in 64 bits");
    check(n > (long long)INT_MAX, "and is larger than INT_MAX (the old int product overflowed)");
    check(floatCount(2000000, 768) == 1536000000LL, "floatCount(2 M, 768)");
    check(floatCount(0, 512) == 0, "zero rows");
    check(throwsWith([] { floatCount(-1, 512); }, "row count"), "negative rows rejected");
    check(throwsWith([] { floatCount(10, 0); }, "dimension"), "dim 0 rejected");
    check(throwsWith([] { floatCount(3000000000LL, 512); }, "ids are int"), "more rows than int ids rejected");

    // setRows copies, validates and hashes like load() does for a file.
    FloatArray v = randomUnitVectors(20, 16, 5);
    EmbeddingStore s;
    s.setRows(16, 20, v.data(), 1e-5);
    check(s.size() == 20 && s.dim() == 16 && s.row(7)[3] == v[7 * 16 + 3], "setRows rows, dim, values");
    check(s.sourceSha256() == sha256Hex(v.data(), 20 * 16 * 4), "setRows records the SHA-256 of the bytes");
    FloatArray bad = randomUnitVectors(3, 4, 6);
    bad[5] *= 2.0f;
    bool threw = false;
    try {
        EmbeddingStore t;
        t.setRows(4, 3, bad.data(), 1e-5);
    } catch (const Error& e) {
        threw = std::string(e.what()).find("row 1") != std::string::npos;
    }
    check(threw, "setRows validates every row");
}

// Exact top k over the rows with t0 <= time <= t1, as plain loops (independent of the engine's
// BruteForceIndex): sorted by score, ties by id.
static void reference(const float* data, int n, int d, const double* t, double t0, double t1,
                      const float* q, int k, IntArray& ids) {
    NeighborArray all;
    for (int i = 0; i < n; i++) {
        if (t[i] < t0 || t[i] > t1) continue;
        Neighbor x;
        x.dist = distance(q, data + (long long)i * d, d);
        x.id = i;
        all.push(x);
    }
    sortNeighbors(all.data(), all.size());
    ids.clear();
    for (int i = 0; i < all.size() && i < k; i++) ids.push(all[i].id);
}

static double overlap(const CandidateArray& got, const IntArray& want) {
    if (want.size() == 0) return got.size() == 0 ? 1.0 : 0.0;
    int hit = 0;
    for (int i = 0; i < got.size(); i++)
        if (want.contains(got[i].id)) hit++;
    return (double)hit / want.size();
}

void testFilteredSearch() {
    const int n = 3000, d = 32, k = 20;
    FloatArray v = randomUnitVectors(n, d, 11);
    double* t = new double[n];
    for (int i = 0; i < n; i++) t[i] = 1000.0 + i;  // one row per second
    NamedIndex idx("appearance/G330", "test-model");
    idx.setVectors(d, n, v.data(), 1e-5);
    idx.setTimes(t, n);
    HnswParams p;
    p.M = 16;
    p.efConstruction = 100;
    idx.build(p);
    FilterParams f;
    f.oversample = 4;
    f.minFraction = 1.0;
    FloatArray queries = randomUnitVectors(30, d, 12);

    // Ranges: none, wide (HNSW + filter), narrow (<= k * oversample rows: exact), very narrow
    // (fewer rows than k), empty, and one that touches both ends.
    struct Range { bool has; double t0, t1; const char* method; };
    Range ranges[] = {{false, 0, 0, "hnsw"},
                      {true, 1000.0, 3500.0, "hnsw_filtered"},
                      {true, 2000.0, 2000.0 + k * 4 - 1, "exact_small"},
                      {true, 1500.0, 1509.0, "exact_small"},
                      {true, 9000.0, 9100.0, "empty"},
                      {true, 0.0, 1e9, "hnsw"}};
    for (const Range& r : ranges) {
        double recall = 0.0;
        bool allInRange = true, sizesOk = true, methodOk = true, everyQueryExact = true;
        for (int qi = 0; qi < 30; qi++) {
            const float* q = queries.data() + qi * d;
            CandidateArray got;
            SearchReport rep;
            idx.search(q, d, k, 200, r.has, r.t0, r.t1, f, got, rep);
            IntArray want;
            reference(v.data(), n, d, t, r.has ? r.t0 : -1e300, r.has ? r.t1 : 1e300, q, k, want);
            recall += overlap(got, want) / 30.0;
            if (overlap(got, want) != 1.0) everyQueryExact = false;
            for (int i = 0; i < got.size(); i++)
                if (r.has && (t[got[i].id] < r.t0 || t[got[i].id] > r.t1)) allInRange = false;
            if (got.size() != want.size()) sizesOk = false;
            std::string expected = r.method;
            // a range covering every row goes through HNSW + filter, not the no-range path
            if (r.has && expected == "hnsw") expected = "hnsw_filtered";
            if (rep.method != expected) methodOk = false;
        }
        std::string name = std::string(r.has ? "range " : "no range ") + r.method;
        check(allInRange, name + ": every result is inside the range");
        check(sizesOk, name + ": min(k, rows in range) results");
        check(methodOk, name + ": answered by the documented method");
        check(recall >= 0.95, name + ": recall vs filtered brute force >= 0.95 (got " + std::to_string(recall) + ")");
        if (std::string(r.method).rfind("exact", 0) == 0 || std::string(r.method) == "empty")
            check(everyQueryExact, name + ": exact paths are exact");
    }

    // Fallback: oversample 1 and a range holding 1/3 of the rows -> HNSW returns k candidates of
    // which ~1/3 are in range, fewer than k -> exact search over the range.
    FilterParams strict;
    strict.oversample = 1;
    strict.minFraction = 1.0;
    int fallbacks = 0, exactQueries = 0;
    for (int qi = 0; qi < 30; qi++) {
        const float* q = queries.data() + qi * d;
        CandidateArray got;
        SearchReport rep;
        idx.search(q, d, k, 200, true, 1000.0, 1999.0, strict, got, rep);
        if (rep.method == "exact_fallback") fallbacks++;
        IntArray want;
        reference(v.data(), n, d, t, 1000.0, 1999.0, q, k, want);
        if (overlap(got, want) == 1.0) exactQueries++;
        check(rep.inRange == 1000, "inRange counts the rows in [t0, t1] (inclusive)");
    }
    check(fallbacks == 30, "too few in-range HNSW hits -> exact fallback (" + std::to_string(fallbacks) + "/30)");
    check(exactQueries == 30, "the fallback is exact");
    // minFraction 0 accepts whatever HNSW kept.
    strict.minFraction = 0.0;
    CandidateArray got;
    SearchReport rep;
    idx.search(queries.data(), d, k, 200, true, 1000.0, 1999.0, strict, got, rep);
    check(rep.method == "hnsw_filtered" && got.size() < k, "minFraction 0: filtered HNSW hits kept as they are");

    // exact(): brute force through the same index equals the reference.
    idx.exact(queries.data(), d, k, false, 0, 0, got, rep);
    IntArray want;
    reference(v.data(), n, d, t, -1e300, 1e300, queries.data(), k, want);
    bool same = got.size() == k;
    for (int i = 0; i < got.size() && same; i++) same = got[i].id == want[i];
    check(same, "exact(): same ids in the same order as the reference");

    // Bad input fails loudly.
    bool threw = false;
    try {
        idx.search(queries.data(), d, k, 200, true, 2000.0, 1000.0, f, got, rep);
    } catch (const QueryError&) {
        threw = true;
    }
    check(threw, "time range with start after end rejected");
    threw = false;
    try {
        double bad[2] = {0.0, NAN};
        NamedIndex tiny("x", "m");
        FloatArray two = randomUnitVectors(2, 4, 1);
        tiny.setVectors(4, 2, two.data(), 1e-5);
        tiny.setTimes(bad, 2);
    } catch (const Error&) {
        threw = true;
    }
    check(threw, "non-finite time rejected");
    threw = false;
    try {
        idx.setTimes(t, n - 1);
    } catch (const Error&) {
        threw = true;
    }
    check(threw, "wrong number of times rejected");
    // New vectors discard the old times: a time filter then fails loudly instead of using stale times.
    idx.setVectors(d, n, v.data(), 1e-5);
    idx.build(p);
    threw = false;
    try {
        idx.search(queries.data(), d, k, 200, true, 1000.0, 2000.0, f, got, rep);
    } catch (const QueryError& e) {
        threw = std::string(e.what()).find("no times") != std::string::npos;
    }
    check(threw, "setVectors discards the old times");
    delete[] t;
}

void testCollectionPersistence() {
    const int n = 2000, d = 24;
    FloatArray v = randomUnitVectors(n, d, 21);
    double* t = new double[n];
    for (int i = 0; i < n; i++) t[i] = i * 0.5;
    IndexCollection c;
    NamedIndex& a = c.add("action/G424", "lb");
    c.add("appearance/G424", "clip");
    check(c.size() == 2 && c.has("action/G424") && !c.has("action/G436"), "named indexes");
    check(c.nameAt(1) == "appearance/G424", "names kept in insertion order");
    bool threw = false;
    try {
        c.add("action/G424", "lb");
    } catch (const Error&) {
        threw = true;
    }
    check(threw, "duplicate name rejected");
    threw = false;
    try {
        c.get("nope");
    } catch (const Error& e) {
        threw = std::string(e.what()).find("nope") != std::string::npos;
    }
    check(threw, "missing name rejected with its name");

    a.setVectors(d, n, v.data(), 1e-5);
    a.setTimes(t, n);
    HnswParams p;
    a.build(p);
    std::string path = tempFile("collection_action.hnsw");
    a.save(path);

    // Reload into a second index over the same vectors: identical answers.
    NamedIndex b("action/G424", "lb");
    b.setVectors(d, n, v.data(), 1e-5);
    b.setTimes(t, n);
    b.load(path);
    FloatArray queries = randomUnitVectors(10, d, 22);
    FilterParams f;
    bool same = true;
    for (int qi = 0; qi < 10; qi++) {
        CandidateArray x, y;
        SearchReport rx, ry;
        a.search(queries.data() + qi * d, d, 50, 200, qi % 2 == 1, 100.0, 700.0, f, x, rx);
        b.search(queries.data() + qi * d, d, 50, 200, qi % 2 == 1, 100.0, 700.0, f, y, ry);
        same = same && x.size() == y.size() && rx.method == ry.method;
        for (int i = 0; i < x.size() && same; i++) same = x[i].id == y[i].id && x[i].score == y[i].score;
    }
    check(same, "save/load: identical results, filtered and unfiltered");

    // A graph saved for other vectors is refused (stale index).
    FloatArray other = randomUnitVectors(n, d, 99);
    NamedIndex s("action/G424", "lb");
    s.setVectors(d, n, other.data(), 1e-5);
    threw = false;
    try {
        s.load(path);
    } catch (const PersistenceError&) {
        threw = true;
    }
    check(threw, "graph for other vectors rejected (stale)");
    NamedIndex m("action/G424", "other-model");
    m.setVectors(d, n, v.data(), 1e-5);
    threw = false;
    try {
        m.load(path);
    } catch (const PersistenceError&) {
        threw = true;
    }
    check(threw, "graph for another model rejected");
    delete[] t;
}

void testIndexLock() {
    // Two threads search one index at the same time with different ef values; with the lock,
    // every answer equals the one computed alone.
    const int n = 4000, d = 32, k = 30;
    FloatArray v = randomUnitVectors(n, d, 31);
    NamedIndex idx("appearance/G436", "m");
    idx.setVectors(d, n, v.data(), 1e-5);
    HnswParams p;
    p.efConstruction = 80;
    idx.build(p);
    FloatArray queries = randomUnitVectors(40, d, 32);
    FilterParams f;
    CandidateArray expected[2][40];
    int efs[2] = {30, 300};
    for (int w = 0; w < 2; w++)
        for (int qi = 0; qi < 40; qi++) {
            SearchReport r;
            idx.search(queries.data() + qi * d, d, k, efs[w], false, 0, 0, f, expected[w][qi], r);
        }
    bool same[2] = {true, true};
    auto worker = [&](int w) {
        for (int rep = 0; rep < 20; rep++)
            for (int qi = 0; qi < 40; qi++) {
                CandidateArray got;
                SearchReport r;
                idx.search(queries.data() + qi * d, d, k, efs[w], false, 0, 0, f, got, r);
                for (int i = 0; i < k; i++)
                    if (got[i].id != expected[w][qi][i].id) same[w] = false;
            }
    };
    std::thread a(worker, 0), b(worker, 1);
    a.join();
    b.join();
    check(same[0] && same[1], "concurrent searches on one index give the single-thread answers");
}
