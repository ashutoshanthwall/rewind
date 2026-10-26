// common.h
// Building blocks shared by every part of the engine:
//   - error classes (thrown with a clear message, caught in main)
//   - growable arrays and a bit array (written in dsa/member4_arrays.cpp)
//   - vector math on embeddings, a millisecond timer, and a random number generator
//   - small helpers for paths and files
#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <string>

// ---------------------------------------------------------------- errors

class Error {
public:
    Error(const std::string& message);
    const char* what() const;

private:
    std::string message;
};

// A bad query from the user (empty text, bad K, ...). The CLI reports it and keeps going.
class QueryError : public Error {
public:
    QueryError(const std::string& message);
};

// A missing, corrupted or stale index file.
class PersistenceError : public Error {
public:
    PersistenceError(const std::string& message);
};

// ---------------------------------------------------------------- search result types

// One stored vector and its distance to the query. Lower distance = closer.
struct Neighbor {
    float dist;
    int id;
};

// One search result as shown to the user. Higher score = more similar.
struct Candidate {
    int id;
    float score;
};

// "a is closer than b": by distance, and by smaller id when the distances are equal, so
// that results never depend on the order things were inserted.
bool closer(const Neighbor& a, const Neighbor& b);

// ---------------------------------------------------------------- growable arrays
// Each array keeps its items in one block from new[]. When the block is full, push()
// allocates a block twice as big and copies the items over, so pushing n items costs
// O(n) copies in total (amortised O(1) per push). operator[] does not check the index.

class IntArray {
public:
    IntArray();
    IntArray(int size, int value);  // `size` copies of `value`
    IntArray(const IntArray& other);
    IntArray& operator=(const IntArray& other);
    ~IntArray();

    void push(int value);
    void pop();
    void clear();
    void reserve(int newCapacity);
    int size() const;
    int capacity() const;
    int& operator[](int i);
    int operator[](int i) const;
    int back() const;
    bool contains(int value) const;  // linear search
    bool equals(const IntArray& other) const;
    const int* data() const;

private:
    int* items;
    int count;
    int cap;
};

class FloatArray {
public:
    FloatArray();
    FloatArray(int size, float value);
    FloatArray(const FloatArray& other);
    FloatArray& operator=(const FloatArray& other);
    ~FloatArray();

    void push(float value);
    void clear();
    void reserve(int newCapacity);
    int size() const;
    float& operator[](int i);
    float operator[](int i) const;
    float* data();
    const float* data() const;

private:
    float* items;
    int count;
    int cap;
};

class StringArray {
public:
    StringArray();
    StringArray(const StringArray& other);
    StringArray& operator=(const StringArray& other);
    ~StringArray();

    void push(const std::string& value);
    void clear();
    int size() const;
    std::string& operator[](int i);
    const std::string& operator[](int i) const;
    bool contains(const std::string& value) const;  // linear search

private:
    std::string* items;
    int count;
    int cap;
};

class NeighborArray {
public:
    NeighborArray();
    NeighborArray(const NeighborArray& other);
    NeighborArray& operator=(const NeighborArray& other);
    ~NeighborArray();

    void push(const Neighbor& value);
    void pop();
    void clear();
    void reserve(int newCapacity);
    void shrink(int newSize);  // keeps only the first newSize items
    int size() const;
    int capacity() const;
    Neighbor& operator[](int i);
    const Neighbor& operator[](int i) const;
    Neighbor* data();

private:
    Neighbor* items;
    int count;
    int cap;
};

class CandidateArray {
public:
    CandidateArray();
    CandidateArray(const CandidateArray& other);
    CandidateArray& operator=(const CandidateArray& other);
    ~CandidateArray();

    void push(const Candidate& value);
    void clear();
    void reserve(int newCapacity);
    int size() const;
    Candidate& operator[](int i);
    const Candidate& operator[](int i) const;

private:
    Candidate* items;
    int count;
    int cap;
};

// A fixed number of true/false flags packed 64 per word (1 bit each instead of 1 byte).
class BitArray {
public:
    BitArray();
    BitArray(int size, bool value);
    BitArray(const BitArray& other);
    BitArray& operator=(const BitArray& other);
    ~BitArray();

    void reset(int size, bool value);  // resize and set every flag to `value`
    bool get(int i) const;
    void set(int i, bool value);
    int size() const;
    long long memoryBytes() const;

private:
    uint64_t* words;
    int numWords;
    int numBits;
};

// ---------------------------------------------------------------- vector math
// Distance used everywhere inside the engine: dist(a, b) = 1 - dot(a, b) on unit vectors,
// lower is better. Score = 1 - dist (higher is better) appears only in results.
// dot() lives in one .cpp file so that every distance in the program is computed by the
// same compiled code and is bit-identical for the same pair of vectors.

float dot(const float* a, const float* b, int n);
float distance(const float* a, const float* b, int n);
float distanceToScore(float dist);
double norm(const float* a, int n);  // sum of squares in double, then sqrt
bool allFinite(const float* a, int n);  // no NaN and no Inf
void normalize(float* a, int n);  // scale to length 1; throws on zero or NaN/Inf input

// ---------------------------------------------------------------- timer and random numbers

double nowMs();  // milliseconds from a monotonic clock (only differences are meaningful)

// Mersenne Twister MT19937-64, written out by hand. Same output sequence as the C++
// standard's std::mt19937_64 for the same seed, so index graphs are reproducible.
class Random64 {
public:
    static const int STATE_SIZE = 312;

    Random64();
    void seed(uint64_t value);
    uint64_t next();
    // The full state (312 words + position) for saving to and restoring from a file.
    void getState(uint64_t* words, int& position) const;
    void setState(const uint64_t* words, int position);

private:
    void twist();
    uint64_t state[STATE_SIZE];
    int index;
};

// ---------------------------------------------------------------- paths and files

std::string joinPath(const std::string& dir, const std::string& name);
std::string parentDir(const std::string& path);  // "a/b/c.txt" -> "a/b"
std::string absolutePath(const std::string& path);
bool fileExists(const std::string& path);
long long fileSize(const std::string& path);  // -1 if the file cannot be opened
void makeDirs(const std::string& path);  // like mkdir -p

#endif
