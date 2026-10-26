// tests.h: a minimal test framework. check() records a pass or a failure; main() in
// test_main.cpp runs every test function below and exits with 1 if any check failed.
#ifndef TESTS_H
#define TESTS_H

#include <string>

#include "common.h"

void check(bool condition, const std::string& what);

// n random unit vectors of dimension d (Gaussian, then normalised), row after row.
FloatArray randomUnitVectors(int n, int d, uint64_t seed);

// A writable scratch file path under /tmp/sir_tests.
std::string tempFile(const std::string& name);

// Member 1 (test_data.cpp)
void testJson();
void testEmbeddingStore();
void testMetadata();
void testHashTable();
void testInvertedIndex();
// Member 2 (test_search.cpp)
void testVectorMath();
void testSorting();
void testTopK();
void testBruteForce();
void testSearchEngine();
// Member 3 (test_hnsw.cpp)
void testHnsw();
// Member 4 (test_storage.cpp)
void testArrays();
void testRandom();
void testSha256();
void testPersistence();
void testTextEncoder();
// Rewind, step 7.1 (test_collection.cpp)
void testStoreSizes();
void testFilteredSearch();
void testCollectionPersistence();
void testIndexLock();

#endif
