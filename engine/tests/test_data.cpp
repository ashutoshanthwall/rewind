// Member 1: JSON, embedding store, metadata, hash table, inverted index.
#include <cmath>
#include <fstream>

#include "data.h"
#include "tests.h"

static std::string writeFile(const std::string& name, const std::string& body) {
    std::string path = tempFile(name);
    std::ofstream(path.c_str()) << body;
    return path;
}

// True if loading `path` as JSON throws an Error.
static bool jsonFails(const std::string& path) {
    try {
        JsonDocument doc(path);
    } catch (const Error&) {
        return true;
    }
    return false;
}

void testJson() {
    std::string path = writeFile("a.json",
        "{\"name\": \"caf\\u00e9 \\\"x\\\"\", \"n\": -12.5e1, \"ok\": true, \"none\": null,"
        " \"list\": [1, 2, 3], \"inner\": {\"deep\": [[], {}]}, \"emoji\": \"\\ud83d\\ude00\"}");
    JsonDocument doc(path);
    const JsonNode* root = doc.root();
    check(jsonString(root, "name") == "caf\xc3\xa9 \"x\"", "string escapes and \\u00e9 -> UTF-8");
    check(jsonNumber(root, "n") == -125.0, "number with exponent");
    check(jsonBool(root, "ok"), "true");
    check(jsonGet(root, "none")->type == JsonNode::NULL_VALUE, "null");
    check(jsonString(root, "emoji") == "\xf0\x9f\x98\x80", "surrogate pair -> 4-byte UTF-8");
    int sum = 0, count = 0;
    for (const JsonNode* x = jsonGet(root, "list")->child; x != nullptr; x = x->next) {
        sum += (int)x->number;
        count++;
    }
    check(count == 3 && sum == 6, "array children in order");
    check(jsonFind(root, "missing") == nullptr, "jsonFind returns nullptr for a missing key");
    bool threw = false;
    try {
        jsonInt(root, "missing");
    } catch (const Error& e) {
        threw = std::string(e.what()).find("missing") != std::string::npos;
    }
    check(threw, "jsonInt names the missing key");

    check(jsonFails(writeFile("b1.json", "{\"a\": 1,}")), "trailing comma rejected");
    check(jsonFails(writeFile("b2.json", "{\"a\": [1, 2}")), "mismatched bracket rejected");
    check(jsonFails(writeFile("b3.json", "{\"a\": \"unterminated}")), "unterminated string rejected");
    check(jsonFails(writeFile("b4.json", "[1] [2]")), "text after the document rejected");
    check(jsonFails(tempFile("no_such_file.json")), "missing file rejected");
}

// True if setData throws an Error whose message contains `part`.
static bool storeFails(int dim, const FloatArray& data, const std::string& part) {
    try {
        EmbeddingStore s;
        s.setData(dim, data, 1e-5);
    } catch (const Error& e) {
        return std::string(e.what()).find(part) != std::string::npos;
    }
    return false;
}

void testEmbeddingStore() {
    FloatArray v = randomUnitVectors(10, 8, 1);
    EmbeddingStore s;
    s.setData(8, v, 1e-5);
    check(s.size() == 10 && s.dim() == 8, "rows and dim");
    check(s.row(3)[5] == v[3 * 8 + 5], "row(i) reads the right values");
    check(s.row(1) + 8 == s.row(2), "rows are contiguous (no per-row copies)");

    FloatArray bad = randomUnitVectors(5, 4, 2);
    bad[3 * 4 + 1] = NAN;
    check(storeFails(4, bad, "row 3"), "NaN error names the row");
    check(storeFails(4, bad, "NaN"), "NaN error says NaN");
    bad = randomUnitVectors(5, 4, 2);
    bad[1] = INFINITY;
    check(storeFails(4, bad, "Inf"), "Inf rejected");
    check(storeFails(4, FloatArray(8, 0.0f), "zero-norm"), "zero vector rejected");
    FloatArray unnormalised = randomUnitVectors(2, 4, 3);
    unnormalised[0] *= 1.01f;
    check(storeFails(4, unnormalised, "norm"), "vector that is not unit length rejected");
    check(storeFails(4, FloatArray(7, 0.5f), "multiple"), "size not a multiple of dim rejected");

    bool threw = false;
    try {
        validateVector(v.data(), 8, 16, 1e-5, "query");
    } catch (const Error&) {
        threw = true;
    }
    check(threw, "validateVector: dimension mismatch");
}

void testMetadata() {
    std::string p = writeFile("meta.json",
        "{\"count\": 2, \"records\": ["
        "{\"internal_id\": 0, \"image_id\": \"100\", \"path\": \"img/100.jpg\", \"width\": 500,"
        " \"height\": 375, \"split\": \"train\", \"captions\": [\"a dog\", \"a cat\"]},"
        "{\"internal_id\": 1, \"image_id\": \"200\", \"path\": \"img/200.jpg\", \"width\": 333,"
        " \"height\": 500, \"split\": \"test\", \"captions\": [\"a car\"]}]}");
    MetadataManager m;
    m.load(p, "/root");
    check(m.size() == 2, "two records");
    check(m.at(1).imageId == "200" && m.at(1).path == "/root/img/200.jpg", "fields and joined path");
    check(m.at(0).captions.size() == 2, "captions");
    bool threw = false;
    try {
        m.at(2);
    } catch (const Error&) {
        threw = true;
    }
    check(threw, "id out of range rejected");

    std::string gap = writeFile("gap.json",
        "{\"count\": 1, \"records\": [{\"internal_id\": 1, \"image_id\": \"x\", \"path\": \"x.jpg\","
        " \"width\": 1, \"height\": 1, \"captions\": [\"c\"]}]}");
    threw = false;
    try {
        MetadataManager g;
        g.load(gap, "/");
    } catch (const Error&) {
        threw = true;
    }
    check(threw, "non-contiguous ids rejected");
    threw = false;
    try {
        MetadataManager g;
        g.load(writeFile("cnt.json", "{\"count\": 5, \"records\": []}"), "/");
    } catch (const Error&) {
        threw = true;
    }
    check(threw, "wrong count rejected");

    std::string real = std::string(SIR_SOURCE_DIR) + "/data/processed/metadata.json";
    if (fileExists(real)) {
        MetadataManager r;
        r.load(real, SIR_SOURCE_DIR);
        check(r.size() > 0 && fileExists(r.at(0).path), "real metadata.json loads");
    }
}

static uint64_t constantHash(const std::string&) {
    return 0;  // every key lands in slot 0: the worst case for linear probing
}

void testHashTable() {
    HashTable t;
    check(t.find("x") == nullptr, "empty table");
    for (int i = 0; i < 1000; i++) t.getOrInsert("k" + std::to_string(i)).push(i);
    check(t.size() == 1000, "1000 keys");
    bool allFound = true;
    for (int i = 0; i < 1000; i++) {
        const IntArray* v = t.find("k" + std::to_string(i));
        allFound = allFound && v != nullptr && (*v)[0] == i;
    }
    check(allFound, "every key finds its value after many rehashes");
    check(t.find("k1000") == nullptr, "absent key");
    t.getOrInsert("k5").push(99);
    check(t.size() == 1000 && t.find("k5")->size() == 2, "existing key: no new entry");
    check((t.capacity() & (t.capacity() - 1)) == 0, "capacity is a power of two");
    check(t.loadFactor() <= 0.7, "load factor stays <= 0.7");

    HashTable worst(constantHash);
    for (int i = 0; i < 200; i++) worst.getOrInsert(std::to_string(i)).push(i * i);
    bool ok = true;
    for (int i = 0; i < 200; i++) ok = ok && (*worst.find(std::to_string(i)))[0] == i * i;
    check(ok, "correct even when every key collides");
    // One cluster: the key at distance d from slot 0 needs d + 1 probes, mean (n + 1) / 2.
    check(worst.averageProbeLength() == 100.5, "mean probe length (200 + 1) / 2");
}

static bool sameList(const IntArray& a, int n, const int* values) {
    if (a.size() != n) return false;
    for (int i = 0; i < n; i++)
        if (a[i] != values[i]) return false;
    return true;
}

void testInvertedIndex() {
    StringArray w;
    tokenize("A Dog, in the SNOW!", w);
    check(w.size() == 2 && w[0] == "dog" && w[1] == "snow", "tokenize: lowercase, stop words");
    tokenize("two-year-old  child's", w);
    check(w.size() == 5 && w[3] == "child" && w[4] == "s", "tokenize: punctuation splits words");
    tokenize("caf\xc3\xa9 au lait", w);
    check(w.size() == 3 && w[0] == "caf\xc3\xa9", "tokenize: UTF-8 stays whole");
    tokenize("the a of ...", w);
    check(w.size() == 0, "tokenize: only stop words");

    InvertedIndex idx;
    idx.add(0, "a dog runs");
    idx.add(0, "the dog jumps");  // second caption of image 0: no duplicate posting
    idx.add(2, "a cat");
    idx.add(5, "dog and cat");
    const int dog[] = {0, 5};
    check(sameList(*idx.postings("dog"), 2, dog), "posting list sorted, no duplicates");
    check(idx.postings("bird") == nullptr, "unknown word");
    check(idx.postingCount() == 6, "6 postings");
    bool threw = false;
    try {
        idx.add(4, "late");
    } catch (const Error&) {
        threw = true;
    }
    check(threw, "ids out of order rejected");

    InvertedIndex q;
    q.add(0, "dog snow");
    q.add(1, "dog beach");
    q.add(2, "cat snow");
    q.add(3, "dog cat snow");
    IntArray out;
    const int dogSnow[] = {0, 3}, beachCat[] = {1, 2, 3}, beach[] = {1};
    q.matchAll("dog snow", out);
    check(sameList(out, 2, dogSnow), "AND query");
    q.matchAll("Snow, DOG, snow", out);
    check(sameList(out, 2, dogSnow), "AND ignores case and repeats");
    q.matchAll("dog bird", out);
    check(out.size() == 0, "unknown word empties an AND");
    q.matchAll("the", out);
    check(out.size() == 0, "only stop words: no results");
    q.matchAny("beach cat", out);
    check(sameList(out, 3, beachCat), "OR query");
    q.matchAny("bird beach", out);
    check(sameList(out, 1, beach), "OR skips unknown words");

    // Merge and galloping intersection must agree on random sorted lists.
    Random64 rng;
    rng.seed(7);
    bool agree = true;
    IntArray a, b, m, g;
    for (int trial = 0; trial < 200; trial++) {
        a.clear();
        b.clear();
        for (int x = 0; x < 5000; x++) {
            if (rng.next() % 40 == 0) a.push(x);
            if (rng.next() % (1 + trial % 5) == 0) b.push(x);
        }
        intersectMerge(a, b, m);
        intersectGallop(a, b, g);
        agree = agree && m.equals(g);
    }
    check(agree, "merge and galloping intersections agree");

    // Random documents vs a linear scan.
    const char* vocab[] = {"dog", "cat", "man", "woman", "red", "ball", "snow", "beach", "car", "child"};
    StringArray* docs = new StringArray[2000];
    InvertedIndex r;
    for (int d = 0; d < 2000; d++) {
        std::string text;
        int len = 1 + (int)(rng.next() % 6);
        for (int t = 0; t < len; t++) {
            int pick = (int)((rng.next() % 4) * (rng.next() % 4));  // low words more common
            docs[d].push(vocab[pick]);
            text += std::string(vocab[pick]) + " ";
        }
        r.add(d, text);
    }
    bool matches = true;
    IntArray all, any, wantAll, wantAny;
    for (int t = 0; t < 300; t++) {
        StringArray words;
        std::string query;
        int len = 1 + (int)(rng.next() % 3);
        for (int i = 0; i < len; i++) {
            words.push(vocab[rng.next() % 10]);
            query += words[i] + " ";
        }
        wantAll.clear();
        wantAny.clear();
        for (int d = 0; d < 2000; d++) {
            int found = 0;
            for (int i = 0; i < words.size(); i++)
                if (docs[d].contains(words[i])) found++;
            if (found == words.size()) wantAll.push(d);
            if (found > 0) wantAny.push(d);
        }
        r.matchAll(query, all);
        r.matchAny(query, any);
        matches = matches && all.equals(wantAll) && any.equals(wantAny);
    }
    delete[] docs;
    check(matches, "300 random AND/OR queries match a linear scan");

    MetadataManager meta;
    meta.allocate(2);
    meta.edit(0).internalId = 0;
    meta.edit(0).captions.push("held out zebra");
    meta.edit(0).captions.push("a dog");
    meta.edit(1).internalId = 1;
    meta.edit(1).captions.push("a cat");
    meta.edit(1).captions.push("zebra crossing");
    InvertedIndex skip0, keepAll;
    skip0.build(meta, 0);
    keepAll.build(meta, -1);
    const int one[] = {1}, both[] = {0, 1};
    skip0.matchAll("zebra", out);
    check(sameList(out, 1, one), "held-out caption 0 is not indexed");
    keepAll.matchAll("zebra", out);
    check(sameList(out, 2, both), "skipCaption -1 indexes every caption");
}
