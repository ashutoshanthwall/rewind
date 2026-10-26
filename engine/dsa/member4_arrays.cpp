#include "common.h"

static int grownCapacity(int cap) {
    return cap == 0 ? 4 : 2 * cap;
}

IntArray::IntArray() {
    items = nullptr;
    count = 0;
    cap = 0;
}

IntArray::IntArray(int size, int value) {
    items = size > 0 ? new int[size] : nullptr;
    count = size;
    cap = size;
    for (int i = 0; i < size; i++) items[i] = value;
}

IntArray::IntArray(const IntArray& other) {
    items = other.count > 0 ? new int[other.count] : nullptr;
    count = other.count;
    cap = other.count;
    for (int i = 0; i < count; i++) items[i] = other.items[i];
}

IntArray& IntArray::operator=(const IntArray& other) {
    if (this == &other) return *this;
    if (cap < other.count) {
        delete[] items;
        items = new int[other.count];
        cap = other.count;
    }
    count = other.count;
    for (int i = 0; i < count; i++) items[i] = other.items[i];
    return *this;
}

IntArray::~IntArray() {
    delete[] items;
}

void IntArray::reserve(int newCapacity) {
    if (newCapacity <= cap) return;
    int* bigger = new int[newCapacity];
    for (int i = 0; i < count; i++) bigger[i] = items[i];
    delete[] items;
    items = bigger;
    cap = newCapacity;
}

void IntArray::push(int value) {
    if (count == cap) reserve(grownCapacity(cap));
    items[count] = value;
    count++;
}

void IntArray::pop() { count--; }
void IntArray::clear() { count = 0; }
int IntArray::size() const { return count; }
int IntArray::capacity() const { return cap; }
int& IntArray::operator[](int i) { return items[i]; }
int IntArray::operator[](int i) const { return items[i]; }
int IntArray::back() const { return items[count - 1]; }
const int* IntArray::data() const { return items; }

bool IntArray::contains(int value) const {
    for (int i = 0; i < count; i++)
        if (items[i] == value) return true;
    return false;
}

bool IntArray::equals(const IntArray& other) const {
    if (count != other.count) return false;
    for (int i = 0; i < count; i++)
        if (items[i] != other.items[i]) return false;
    return true;
}

FloatArray::FloatArray() {
    items = nullptr;
    count = 0;
    cap = 0;
}

FloatArray::FloatArray(int size, float value) {
    items = size > 0 ? new float[size] : nullptr;
    count = size;
    cap = size;
    for (int i = 0; i < size; i++) items[i] = value;
}

FloatArray::FloatArray(const FloatArray& other) {
    items = other.count > 0 ? new float[other.count] : nullptr;
    count = other.count;
    cap = other.count;
    for (int i = 0; i < count; i++) items[i] = other.items[i];
}

FloatArray& FloatArray::operator=(const FloatArray& other) {
    if (this == &other) return *this;
    if (cap < other.count) {
        delete[] items;
        items = new float[other.count];
        cap = other.count;
    }
    count = other.count;
    for (int i = 0; i < count; i++) items[i] = other.items[i];
    return *this;
}

FloatArray::~FloatArray() {
    delete[] items;
}

void FloatArray::reserve(int newCapacity) {
    if (newCapacity <= cap) return;
    float* bigger = new float[newCapacity];
    for (int i = 0; i < count; i++) bigger[i] = items[i];
    delete[] items;
    items = bigger;
    cap = newCapacity;
}

void FloatArray::push(float value) {
    if (count == cap) reserve(grownCapacity(cap));
    items[count] = value;
    count++;
}

void FloatArray::clear() { count = 0; }
int FloatArray::size() const { return count; }
float& FloatArray::operator[](int i) { return items[i]; }
float FloatArray::operator[](int i) const { return items[i]; }
float* FloatArray::data() { return items; }
const float* FloatArray::data() const { return items; }

StringArray::StringArray() {
    items = nullptr;
    count = 0;
    cap = 0;
}

StringArray::StringArray(const StringArray& other) {
    items = other.count > 0 ? new std::string[other.count] : nullptr;
    count = other.count;
    cap = other.count;
    for (int i = 0; i < count; i++) items[i] = other.items[i];
}

StringArray& StringArray::operator=(const StringArray& other) {
    if (this == &other) return *this;
    if (cap < other.count) {
        delete[] items;
        items = new std::string[other.count];
        cap = other.count;
    }
    count = other.count;
    for (int i = 0; i < count; i++) items[i] = other.items[i];
    return *this;
}

StringArray::~StringArray() {
    delete[] items;
}

void StringArray::push(const std::string& value) {
    if (count == cap) {
        std::string copy = value;
        int newCap = grownCapacity(cap);
        std::string* bigger = new std::string[newCap];
        for (int i = 0; i < count; i++) bigger[i] = items[i];
        delete[] items;
        items = bigger;
        cap = newCap;
        items[count] = copy;
    } else {
        items[count] = value;
    }
    count++;
}

void StringArray::clear() { count = 0; }
int StringArray::size() const { return count; }
std::string& StringArray::operator[](int i) { return items[i]; }
const std::string& StringArray::operator[](int i) const { return items[i]; }

bool StringArray::contains(const std::string& value) const {
    for (int i = 0; i < count; i++)
        if (items[i] == value) return true;
    return false;
}

NeighborArray::NeighborArray() {
    items = nullptr;
    count = 0;
    cap = 0;
}

NeighborArray::NeighborArray(const NeighborArray& other) {
    items = other.count > 0 ? new Neighbor[other.count] : nullptr;
    count = other.count;
    cap = other.count;
    for (int i = 0; i < count; i++) items[i] = other.items[i];
}

NeighborArray& NeighborArray::operator=(const NeighborArray& other) {
    if (this == &other) return *this;
    if (cap < other.count) {
        delete[] items;
        items = new Neighbor[other.count];
        cap = other.count;
    }
    count = other.count;
    for (int i = 0; i < count; i++) items[i] = other.items[i];
    return *this;
}

NeighborArray::~NeighborArray() {
    delete[] items;
}

void NeighborArray::reserve(int newCapacity) {
    if (newCapacity <= cap) return;
    Neighbor* bigger = new Neighbor[newCapacity];
    for (int i = 0; i < count; i++) bigger[i] = items[i];
    delete[] items;
    items = bigger;
    cap = newCapacity;
}

void NeighborArray::push(const Neighbor& value) {
    Neighbor copy = value;
    if (count == cap) reserve(grownCapacity(cap));
    items[count] = copy;
    count++;
}

void NeighborArray::pop() { count--; }
void NeighborArray::clear() { count = 0; }

void NeighborArray::shrink(int newSize) {
    if (newSize < count) count = newSize;
}

int NeighborArray::size() const { return count; }
int NeighborArray::capacity() const { return cap; }
Neighbor& NeighborArray::operator[](int i) { return items[i]; }
const Neighbor& NeighborArray::operator[](int i) const { return items[i]; }
Neighbor* NeighborArray::data() { return items; }

CandidateArray::CandidateArray() {
    items = nullptr;
    count = 0;
    cap = 0;
}

CandidateArray::CandidateArray(const CandidateArray& other) {
    items = other.count > 0 ? new Candidate[other.count] : nullptr;
    count = other.count;
    cap = other.count;
    for (int i = 0; i < count; i++) items[i] = other.items[i];
}

CandidateArray& CandidateArray::operator=(const CandidateArray& other) {
    if (this == &other) return *this;
    if (cap < other.count) {
        delete[] items;
        items = new Candidate[other.count];
        cap = other.count;
    }
    count = other.count;
    for (int i = 0; i < count; i++) items[i] = other.items[i];
    return *this;
}

CandidateArray::~CandidateArray() {
    delete[] items;
}

void CandidateArray::reserve(int newCapacity) {
    if (newCapacity <= cap) return;
    Candidate* bigger = new Candidate[newCapacity];
    for (int i = 0; i < count; i++) bigger[i] = items[i];
    delete[] items;
    items = bigger;
    cap = newCapacity;
}

void CandidateArray::push(const Candidate& value) {
    Candidate copy = value;
    if (count == cap) reserve(grownCapacity(cap));
    items[count] = copy;
    count++;
}

void CandidateArray::clear() { count = 0; }
int CandidateArray::size() const { return count; }
Candidate& CandidateArray::operator[](int i) { return items[i]; }
const Candidate& CandidateArray::operator[](int i) const { return items[i]; }

BitArray::BitArray() {
    words = nullptr;
    numWords = 0;
    numBits = 0;
}

BitArray::BitArray(int size, bool value) {
    words = nullptr;
    numWords = 0;
    numBits = 0;
    reset(size, value);
}

BitArray::BitArray(const BitArray& other) {
    numWords = other.numWords;
    numBits = other.numBits;
    words = numWords > 0 ? new uint64_t[numWords] : nullptr;
    for (int i = 0; i < numWords; i++) words[i] = other.words[i];
}

BitArray& BitArray::operator=(const BitArray& other) {
    if (this == &other) return *this;
    delete[] words;
    numWords = other.numWords;
    numBits = other.numBits;
    words = numWords > 0 ? new uint64_t[numWords] : nullptr;
    for (int i = 0; i < numWords; i++) words[i] = other.words[i];
    return *this;
}

BitArray::~BitArray() {
    delete[] words;
}

void BitArray::reset(int size, bool value) {
    delete[] words;
    numBits = size;
    numWords = (size + 63) / 64;
    words = numWords > 0 ? new uint64_t[numWords] : nullptr;
    uint64_t fill = value ? ~(uint64_t)0 : 0;
    for (int i = 0; i < numWords; i++) words[i] = fill;
}

bool BitArray::get(int i) const {
    return (words[i / 64] >> (i % 64)) & 1;
}

void BitArray::set(int i, bool value) {
    uint64_t bit = (uint64_t)1 << (i % 64);
    if (value)
        words[i / 64] |= bit;
    else
        words[i / 64] &= ~bit;
}

int BitArray::size() const { return numBits; }
long long BitArray::memoryBytes() const { return (long long)numWords * 8; }
