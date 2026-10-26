// storage.h  (Member 4: persistence, integrity, text-encoder process, command-line tools)
// SHA-256, the index file format, the Python text-encoder helper process, and helpers
// shared by the command-line programs.
#ifndef STORAGE_H
#define STORAGE_H

#include <string>

#include "common.h"
#include "data.h"
#include "hnsw.h"
#include "search.h"

// ---------------------------------------------------------------- SHA-256 (src/sha256.cpp)
// SHA-256 (FIPS 180-4), written from scratch. Checks that an index was built from the exact
// embedding file named in its header, and checksums index files. Tested on NIST vectors.
class Sha256 {
public:
    Sha256();
    void update(const void* data, long long n);
    void finish(unsigned char digest[32]);

private:
    void block(const unsigned char* p);  // processes one 64-byte block
    uint32_t h[8];
    unsigned char buffer[64];
    int bufferLength;
    uint64_t totalBytes;
};

std::string toHex(const unsigned char* bytes, int n);
std::string sha256Hex(const void* data, long long n);
std::string sha256File(const std::string& path);

// ---------------------------------------------------------------- index files (src/index_manager.cpp)
// File format (little-endian), version 2:
//   magic        8 bytes  "SIRHNSW\0"
//   version      u32
//   model        u32 length + UTF-8 bytes
//   dim          u64
//   store rows   u64      rows in the EmbeddingStore the index was built on
//   vectors      u64      nodes in the index
//   M, efConstruction, efSearch   i32 x3
//   seed         u64
//   heuristic, keepPrunedConnections, protectLastInEdge   u8 x3
//   source hash  64 bytes SHA-256 (hex) of the embedding file
//   entry, maxLevel   i32 x2
//   rng state    u32 position, then 312 x u64
//   payload size u64
//   payload      per node id 0..rows-1: i32 level, then per layer: u32 n, i32 ids[n]
//   checksum     32 bytes SHA-256 of every byte before it
// Loading rejects: wrong magic or version, truncation, checksum mismatch (corruption), and
// a header that does not match the current embeddings (model, dim, row count, hash).

struct IndexValidation {
    int nodes = 0;
    int reachableLayer0 = 0;
    int maxLevel = -1;
    double meanDegreeLayer0 = 0.0;
    IntArray nodesPerLevel;
    IntArray unreachableLayer0;  // nodes with no layer-0 path from the entry point
    int inDegreeZeroLayer0 = 0;  // nodes that no other node links to on layer 0
};

class IndexManager {
public:
    static const uint32_t FORMAT_VERSION = 2;

    // Inserts every row of the index's store; returns the build time in seconds.
    static double build(HNSWIndex& index);
    static void save(const HNSWIndex& index, const std::string& modelName, const std::string& path);
    // Reads `path` into `index` (built over `store`); throws PersistenceError.
    static void load(const std::string& path, const EmbeddingStore& store,
                     const std::string& modelName, HNSWIndex& index);
    // Structure checks (throws Error when one fails) plus statistics.
    static IndexValidation validate(const HNSWIndex& index);
    // build + validate + save; returns the build time in seconds.
    static double rebuild(HNSWIndex& index, const std::string& modelName, const std::string& path);
};

// ---------------------------------------------------------------- text encoder (src/text_encoder.cpp)
// Starts the Python CLIP text-encoder helper (scripts/text_encoder_server.py) once and
// talks to it over a Unix socket pair. The protocol is described in that script. The
// destructor asks the helper to exit, waits, and sends SIGTERM / SIGKILL if it does not.
class TextEncoderProcess : public QueryEncoder {
public:
    TextEncoderProcess(const std::string& python, const std::string& script,
                       const std::string& config, int expectedDim,
                       const std::string& logFile = "/dev/null");
    ~TextEncoderProcess();
    TextEncoderProcess(const TextEncoderProcess&) = delete;
    TextEncoderProcess& operator=(const TextEncoderProcess&) = delete;

    void encode(const std::string& text, FloatArray& out);

private:
    void sendAll(const void* data, int n);
    void receiveAll(void* data, int n);

    int pid;
    int sock;
    int dim;
};

// ---------------------------------------------------------------- command-line helpers (src/app.cpp)

// Command-line arguments: words are positional; "--key value" or "--flag" are options.
struct Args {
    StringArray positional;
    StringArray keys;
    StringArray values;

    bool has(const std::string& key) const;
    std::string get(const std::string& key, const std::string& fallback) const;
    int getInt(const std::string& key, int fallback) const;
};

// `flags` lists the options that take no value.
Args parseArgs(int argc, char** argv, const StringArray& flags);
AppConfig loadAppConfig(const Args& args);  // --config, default config/config.json
void loadImageStore(const AppConfig& config, EmbeddingStore& out);
void loadQueryStore(const AppConfig& config, EmbeddingStore& out);

// One text query through both indexes, the same way for every front end: embed once,
// time each index `timingRuns` times and keep the median, then compare the two top-K lists.
struct Comparison {
    std::string query;  // cleaned text that was sent to the encoder
    int k = 0;
    double encodeMs = 0.0;
    SearchResponse exact, approx;
    double exactMs = 0.0, approxMs = 0.0;
    double recall = 0.0;  // |approx ∩ exact| / k
    double speedup = 0.0;
};

void compareBackends(const std::string& text, int k, const SearchEngine& exact,
                     const SearchEngine& approx, int timingRuns, Comparison& out);

// For writing JSON by hand: a quoted, escaped string, and a number as text.
std::string jsonQuote(const std::string& s);
std::string jsonNumberText(double value);

#endif
