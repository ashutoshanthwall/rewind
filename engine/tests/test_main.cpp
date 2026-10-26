// Runs every test and prints a summary. Each failed check is printed with its message.
#include <cmath>
#include <iostream>

#include "tests.h"

static int passed = 0;
static int failed = 0;

void check(bool condition, const std::string& what) {
    if (condition) {
        passed++;
    } else {
        failed++;
        std::cout << "  FAILED: " << what << "\n";
    }
}

// Box-Muller: two uniform numbers in (0, 1] give one normally distributed number.
static double gaussian(Random64& rng) {
    double u1 = ((double)(rng.next() >> 11) + 1.0) / 9007199254740992.0;
    double u2 = ((double)(rng.next() >> 11) + 1.0) / 9007199254740992.0;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

FloatArray randomUnitVectors(int n, int d, uint64_t seed) {
    Random64 rng;
    rng.seed(seed);
    FloatArray v(n * d, 0.0f);
    for (int i = 0; i < n * d; i++) v[i] = (float)gaussian(rng);
    for (int i = 0; i < n; i++) normalize(v.data() + i * d, d);
    return v;
}

std::string tempFile(const std::string& name) {
    makeDirs("/tmp/sir_tests");
    return "/tmp/sir_tests/" + name;
}

static void runTest(const std::string& name, void (*test)()) {
    int failedBefore = failed;
    std::cout << name << "\n";
    try {
        test();
    } catch (const Error& e) {
        check(false, name + " threw: " + e.what());
    } catch (const std::exception& e) {
        check(false, name + " threw: " + e.what());
    }
    if (failed > failedBefore) std::cout << "  -> " << (failed - failedBefore) << " failed\n";
}

int main() {
    runTest("arrays", testArrays);
    runTest("random", testRandom);
    runTest("json", testJson);
    runTest("embedding store", testEmbeddingStore);
    runTest("metadata", testMetadata);
    runTest("hash table", testHashTable);
    runTest("inverted index", testInvertedIndex);
    runTest("vector math", testVectorMath);
    runTest("sorting", testSorting);
    runTest("top-k", testTopK);
    runTest("brute force", testBruteForce);
    runTest("search engine", testSearchEngine);
    runTest("hnsw", testHnsw);
    runTest("sha256", testSha256);
    runTest("persistence", testPersistence);
    runTest("text encoder", testTextEncoder);
    runTest("store sizes (64-bit)", testStoreSizes);
    runTest("filtered search", testFilteredSearch);
    runTest("collection persistence", testCollectionPersistence);
    runTest("index lock", testIndexLock);
    std::cout << "\n" << passed << " checks passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
