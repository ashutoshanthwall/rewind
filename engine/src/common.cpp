// Errors, vector math, timer, random numbers, and path helpers (declared in common.h).
#include "common.h"

#include <climits>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <sys/stat.h>

// ---------------------------------------------------------------- errors

Error::Error(const std::string& text) { message = text; }
const char* Error::what() const { return message.c_str(); }
QueryError::QueryError(const std::string& text) : Error(text) {}
PersistenceError::PersistenceError(const std::string& text) : Error(text) {}

// ---------------------------------------------------------------- vector math

float dot(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

float distance(const float* a, const float* b, int n) {
    return 1.0f - dot(a, b, n);
}

float distanceToScore(float dist) {
    return 1.0f - dist;
}

double norm(const float* a, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)a[i] * (double)a[i];
    return std::sqrt(sum);
}

bool allFinite(const float* a, int n) {
    for (int i = 0; i < n; i++)
        if (!std::isfinite(a[i])) return false;
    return true;
}

void normalize(float* a, int n) {
    if (!allFinite(a, n)) throw Error("cannot normalise a vector containing NaN/Inf");
    double length = norm(a, n);
    if (length == 0.0) throw Error("cannot normalise a zero-norm vector");
    for (int i = 0; i < n; i++) a[i] = (float)((double)a[i] / length);
}

// ---------------------------------------------------------------- timer

double nowMs() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1000.0 + (double)t.tv_nsec / 1e6;
}

// ---------------------------------------------------------------- Random64 (MT19937-64)
// Constants from Matsumoto and Nishimura's 64-bit Mersenne Twister.

static const int MIDDLE = 156;
static const uint64_t MATRIX_A = 0xB5026F5AA96619E9ULL;
static const uint64_t UPPER_MASK = 0xFFFFFFFF80000000ULL;  // top 33 bits
static const uint64_t LOWER_MASK = 0x7FFFFFFFULL;          // bottom 31 bits

Random64::Random64() { seed(5489); }

void Random64::seed(uint64_t value) {
    state[0] = value;
    for (int i = 1; i < STATE_SIZE; i++)
        state[i] = 6364136223846793005ULL * (state[i - 1] ^ (state[i - 1] >> 62)) + (uint64_t)i;
    index = STATE_SIZE;
}

// Refills all 312 words at once; next() then hands them out one by one.
void Random64::twist() {
    for (int i = 0; i < STATE_SIZE; i++) {
        uint64_t x = (state[i] & UPPER_MASK) | (state[(i + 1) % STATE_SIZE] & LOWER_MASK);
        uint64_t xA = x >> 1;
        if (x & 1) xA ^= MATRIX_A;
        state[i] = state[(i + MIDDLE) % STATE_SIZE] ^ xA;
    }
    index = 0;
}

uint64_t Random64::next() {
    if (index >= STATE_SIZE) twist();
    uint64_t x = state[index];
    index++;
    // Tempering: mixes the bits so every output bit depends on many state bits.
    x ^= (x >> 29) & 0x5555555555555555ULL;
    x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
    x ^= (x << 37) & 0xFFF7EEE000000000ULL;
    x ^= x >> 43;
    return x;
}

void Random64::getState(uint64_t* words, int& position) const {
    for (int i = 0; i < STATE_SIZE; i++) words[i] = state[i];
    position = index;
}

void Random64::setState(const uint64_t* words, int position) {
    for (int i = 0; i < STATE_SIZE; i++) state[i] = words[i];
    index = position;
}

// ---------------------------------------------------------------- paths and files

std::string joinPath(const std::string& dir, const std::string& name) {
    if (dir.empty()) return name;
    if (dir[dir.size() - 1] == '/') return dir + name;
    return dir + "/" + name;
}

std::string parentDir(const std::string& path) {
    size_t slash = path.find_last_of('/');
    if (slash == std::string::npos) return ".";
    if (slash == 0) return "/";
    return path.substr(0, slash);
}

std::string absolutePath(const std::string& path) {
    char buffer[PATH_MAX];
    if (realpath(path.c_str(), buffer) == nullptr) return path;
    return buffer;
}

bool fileExists(const std::string& path) {
    struct stat info;
    return stat(path.c_str(), &info) == 0;
}

long long fileSize(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) return -1;
    return (long long)info.st_size;
}

void makeDirs(const std::string& path) {
    for (size_t i = 1; i <= path.size(); i++)
        if (i == path.size() || path[i] == '/') mkdir(path.substr(0, i).c_str(), 0755);
}
