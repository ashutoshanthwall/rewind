// Member 4: arrays, random numbers, SHA-256, index files, text-encoder process.
#include <fstream>
#include <iostream>
#include <sstream>

#include "storage.h"
#include "tests.h"

void testArrays() {
    IntArray a;
    check(a.size() == 0 && a.capacity() == 0, "new array is empty");
    for (int i = 0; i < 100; i++) a.push(i * i);
    check(a.size() == 100, "100 pushes give size 100");
    check(a.capacity() == 128, "capacity doubles 4, 8, ..., 128");
    check(a[7] == 49 && a.back() == 99 * 99, "items keep their values while growing");
    check(a.contains(81) && !a.contains(82), "contains");

    IntArray b = a;  // copy constructor: a separate block
    b[0] = -1;
    check(a[0] == 0, "copy does not share memory with the original");
    IntArray c;
    c = b;  // copy assignment
    check(c.equals(b) && !c.equals(a), "copy assignment and equals");
    c = c;
    check(c.size() == 100, "self-assignment keeps the items");
    c.clear();
    check(c.size() == 0 && c.capacity() >= 100, "clear keeps the block");
    IntArray filled(5, 7);
    check(filled.size() == 5 && filled[4] == 7, "size/value constructor");

    NeighborArray n;
    Neighbor x;
    x.dist = 0.5f;
    x.id = 3;
    n.push(x);
    for (int i = 0; i < 20; i++) n.push(n[0]);  // pushing an own item while the array grows
    check(n.size() == 21 && n[20].id == 3 && n[20].dist == 0.5f, "push of own item survives growth");
    n.shrink(5);
    check(n.size() == 5, "shrink");

    StringArray s;
    s.push("dog");
    for (int i = 0; i < 10; i++) s.push(s[0]);
    check(s.size() == 11 && s[10] == "dog" && s.contains("dog") && !s.contains("cat"),
          "string array push, own item, contains");

    BitArray bits(130, false);
    bits.set(0, true);
    bits.set(63, true);
    bits.set(64, true);
    bits.set(129, true);
    check(bits.get(0) && bits.get(63) && bits.get(64) && bits.get(129), "bits set across words");
    check(!bits.get(1) && !bits.get(65) && !bits.get(128), "other bits stay clear");
    bits.set(63, false);
    check(!bits.get(63) && bits.get(64), "clearing one bit leaves its neighbour");
    check(bits.memoryBytes() == 3 * 8, "130 bits use 3 words");
    BitArray ones(70, true);
    check(ones.get(69), "all-true constructor");
}

void testRandom() {
    // The C++ standard says the 10000th number from a default-seeded mt19937_64 (seed 5489)
    // is 9981545732273789042, so this checks our hand-written version against it.
    Random64 rng;
    rng.seed(5489);
    uint64_t value = 0;
    for (int i = 0; i < 10000; i++) value = rng.next();
    check(value == 9981545732273789042ULL, "MT19937-64 matches the standard's 10000th value");

    Random64 a, b;
    a.seed(42);
    b.seed(42);
    for (int i = 0; i < 500; i++) a.next();
    uint64_t words[Random64::STATE_SIZE];
    int position = 0;
    a.getState(words, position);
    b.setState(words, position);
    bool same = true;
    for (int i = 0; i < 1000; i++) same = same && a.next() == b.next();
    check(same, "saved and restored state continues the same sequence");
}

void testSha256() {
    check(sha256Hex("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          "SHA-256 of empty input");
    check(sha256Hex("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "SHA-256 of 'abc'");
    std::string m = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    check(sha256Hex(m.data(), (long long)m.size()) ==
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
          "SHA-256 of the 448-bit NIST message");
    std::string million(1000000, 'a');
    check(sha256Hex(million.data(), (long long)million.size()) ==
              "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
          "SHA-256 of one million 'a'");

    std::string s(1000, '\0');
    for (int i = 0; i < 1000; i++) s[i] = (char)(i * 7);
    Sha256 h;
    for (int off = 0; off < 1000; off += 37) h.update(s.data() + off, off + 37 <= 1000 ? 37 : 1000 - off);
    unsigned char digest[32];
    h.finish(digest);
    check(toHex(digest, 32) == sha256Hex(s.data(), 1000), "pieces give the same hash as one call");
}

// ---------------------------------------------------------------- persistence

static void makeStore(EmbeddingStore& store, int n, int d, uint64_t seed) {
    FloatArray v = randomUnitVectors(n, d, seed);
    store.setData(d, v, 1e-5, sha256Hex(v.data(), (long long)v.size() * 4));
}

static HnswParams smallParams() {
    HnswParams p;
    p.M = 8;
    p.efConstruction = 64;
    p.efSearch = 40;
    p.seed = 7;
    return p;
}

static void flipByte(const std::string& path, long long offset) {
    std::fstream f(path.c_str(), std::ios::in | std::ios::out | std::ios::binary);
    f.seekg(offset);
    char c;
    f.read(&c, 1);
    c = (char)(c ^ 0x5A);
    f.seekp(offset);
    f.write(&c, 1);
}

static void truncateFile(const std::string& path, long long newSize) {
    std::stringstream contents;
    {
        std::ifstream in(path.c_str(), std::ios::binary);
        contents << in.rdbuf();
    }
    std::string bytes = contents.str();
    std::ofstream out(path.c_str(), std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), newSize);
}

// The PersistenceError message from loading, or "" if the load worked.
static std::string loadError(const std::string& path, const EmbeddingStore& store,
                             const std::string& model) {
    try {
        HNSWIndex idx;
        IndexManager::load(path, store, model, idx);
    } catch (const PersistenceError& e) {
        return e.what();
    }
    return "";
}

static bool has(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

void testPersistence() {
    EmbeddingStore store;
    makeStore(store, 1500, 16, 1);
    HNSWIndex idx;
    idx.setup(store, smallParams());
    IndexManager::build(idx);
    std::string path = tempFile("roundtrip.idx");
    IndexManager::save(idx, "m", path);
    HNSWIndex loaded;
    IndexManager::load(path, store, "m", loaded);
    FloatArray qs = randomUnitVectors(50, 16, 2);
    bool same = true;
    CandidateArray a, b;
    for (int t = 0; t < 50; t++) {
        idx.search(qs.data() + t * 16, 16, 10, a);
        loaded.search(qs.data() + t * 16, 16, 10, b);
        for (int i = 0; i < 10; i++) same = same && a[i].id == b[i].id && a[i].score == b[i].score;
    }
    check(same, "saved and loaded index give identical results");
    uint64_t w1[Random64::STATE_SIZE], w2[Random64::STATE_SIZE];
    int p1 = 0, p2 = 0;
    idx.getRandomState(w1, p1);
    loaded.getRandomState(w2, p2);
    bool sameRng = p1 == p2;
    for (int i = 0; i < Random64::STATE_SIZE; i++) sameRng = sameRng && w1[i] == w2[i];
    check(sameRng, "random-number state survives save/load");
    check(IndexManager::validate(loaded).reachableLayer0 == 1500, "loaded graph is connected");

    EmbeddingStore small;
    makeStore(small, 600, 8, 3);
    HNSWIndex smallIdx;
    smallIdx.setup(small, smallParams());
    IndexManager::build(smallIdx);
    std::string bad = tempFile("bad.idx");

    IndexManager::save(smallIdx, "m", bad);
    flipByte(bad, fileSize(bad) / 2);
    check(has(loadError(bad, small, "m"), "corrupted"), "flipped byte is reported as corruption");

    IndexManager::save(smallIdx, "m", bad);
    truncateFile(bad, fileSize(bad) - 100);
    check(has(loadError(bad, small, "m"), "truncated"), "short file is reported as truncated");

    IndexManager::save(smallIdx, "m", bad);
    flipByte(bad, 0);
    check(has(loadError(bad, small, "m"), "magic"), "bad magic number is reported");

    IndexManager::save(smallIdx, "m", bad);
    check(has(loadError(bad, small, "other-model"), "stale"), "different model is stale");
    EmbeddingStore otherContent, bigger, otherDim;
    makeStore(otherContent, 600, 8, 4);  // same shape, different vectors
    check(has(loadError(bad, otherContent, "m"), "stale"), "different embeddings are stale");
    makeStore(bigger, 601, 8, 3);
    check(has(loadError(bad, bigger, "m"), "stale"), "different row count is stale");
    makeStore(otherDim, 600, 16, 3);
    check(has(loadError(bad, otherDim, "m"), "dimension mismatch"), "different dimension is reported");
    check(has(loadError(tempFile("does_not_exist.idx"), small, "m"), "not found"),
          "missing file is reported");
}

void testTextEncoder() {
    std::string root = SIR_SOURCE_DIR;
    std::string python = root + "/.venv/bin/python";
    if (!fileExists(python)) {
        std::cout << "  (skipped: no .venv)\n";
        return;
    }
    TextEncoderProcess encoder(python, root + "/scripts/text_encoder_server.py",
                               root + "/config/config.json", 512);
    FloatArray v, v2;
    encoder.encode("a dog playing in snow", v);
    check(v.size() == 512, "encoder returns 512 numbers");
    check(allFinite(v.data(), v.size()) && norm(v.data(), v.size()) > 0.1, "embedding is finite, non-zero");
    encoder.encode("a dog playing in snow", v2);
    bool same = true;
    for (int i = 0; i < 512; i++) same = same && v[i] == v2[i];
    check(same, "same text gives the same embedding");
    bool threw = false;
    try {
        encoder.encode("", v);
    } catch (const QueryError&) {
        threw = true;
    }
    check(threw, "empty text is rejected");
}
