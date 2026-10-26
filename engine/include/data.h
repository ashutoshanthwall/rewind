// data.h  (Member 1: data ingestion, vector management, caption search)
// JSON reading, configuration, image metadata, the embedding store, and the caption
// inverted index with its hash table.
#ifndef DATA_H
#define DATA_H

#include <string>

#include "common.h"

// ---------------------------------------------------------------- JSON (src/json.cpp)
// A parsed JSON value. The children of an object or array form a linked list: `child`
// points to the first child and each child's `next` to the following one (a
// first-child / next-sibling tree), so a node needs no array of children.
struct JsonNode {
    enum Type { NULL_VALUE, BOOLEAN, NUMBER, STRING, ARRAY, OBJECT };

    Type type;
    std::string key;   // member name, when this node is a member of an object
    std::string text;  // STRING value
    double number;     // NUMBER value
    bool flag;         // BOOLEAN value
    JsonNode* child;
    JsonNode* next;
};

// Owns the tree parsed from one file and frees it in the destructor.
class JsonDocument {
public:
    JsonDocument(const std::string& path);  // throws Error on a missing or malformed file
    ~JsonDocument();
    JsonDocument(const JsonDocument&) = delete;  // never copied (it owns the tree)
    JsonDocument& operator=(const JsonDocument&) = delete;
    const JsonNode* root() const;

private:
    JsonNode* top;
};

// Lookups in an object. jsonFind returns nullptr for a missing key; the others throw an
// Error naming the key when it is missing or has the wrong type.
const JsonNode* jsonFind(const JsonNode* object, const std::string& key);
const JsonNode* jsonGet(const JsonNode* object, const std::string& key);
std::string jsonString(const JsonNode* object, const std::string& key);
double jsonNumber(const JsonNode* object, const std::string& key);
int jsonInt(const JsonNode* object, const std::string& key);
bool jsonBool(const JsonNode* object, const std::string& key);

// ---------------------------------------------------------------- configuration (src/data.cpp)

struct HnswParams {
    int M = 16;                    // max neighbours per node on layers >= 1 (2M on layer 0)
    int efConstruction = 200;      // candidate-list size while inserting
    int efSearch = 80;             // candidate-list size while searching
    uint64_t seed = 42;            // random-level seed
    bool selectHeuristic = true;   // true: paper's heuristic selection, false: M closest
    bool keepPrunedConnections = true;  // heuristic only: refill to M with discarded ones
    bool protectLastInEdge = true;      // pruning never removes a node's only incoming edge
};

struct EvaluationParams {
    uint64_t seed = 0;
    int warmupQueries = 20;
    int repetitions = 5;
    int heldoutCaptionIndex = 0;  // caption used as the held-out query text for each image
    IntArray efSearchSweep;
};

struct AppConfig {
    std::string root;  // project root: the parent of the config file's directory
    std::string configFile;
    std::string modelName;
    int dim = 0;
    double normTolerance = 1e-5;
    HnswParams hnsw;
    EvaluationParams evaluation;
    // Absolute paths.
    std::string metadata, embeddings, manifest, index, queries, queryGroundTruth,
        queryEmbeddings, queryManifest, benchmarkResults, textEncoderServer, python;
};

AppConfig loadConfig(const std::string& configFile);

// ---------------------------------------------------------------- image metadata (src/data.cpp)

struct ImageRecord {
    int internalId = -1;
    std::string imageId;
    std::string path;  // absolute
    int width = 0;
    int height = 0;
    std::string split;
    StringArray captions;
};

// Owns the record of every image. Search works on ids 0..N-1; record i is image id i.
class MetadataManager {
public:
    MetadataManager();
    ~MetadataManager();
    MetadataManager(const MetadataManager&) = delete;
    MetadataManager& operator=(const MetadataManager&) = delete;

    void load(const std::string& metadataJson, const std::string& root);
    void allocate(int count);  // `count` empty records, to be filled with edit()
    ImageRecord& edit(int id);
    void checkIds() const;  // throws unless record i has internalId i for every i
    const ImageRecord& at(int id) const;  // throws Error when out of range
    int size() const;

private:
    ImageRecord* records;
    int count;
};

// ---------------------------------------------------------------- embeddings (src/data.cpp)

// Mirror of data/embeddings/manifest.json (and evaluation/query_manifest.json).
struct EmbeddingManifest {
    std::string model;
    int dim = 0;
    int rows = 0;
    bool l2Normalized = false;
    std::string dtype;
    std::string sha256;
    std::string createdUtc;
    std::string gitCommit;
};

EmbeddingManifest loadManifest(const std::string& path);

// Owns every embedding vector exactly once, as one row-major float block [rows x dim].
// Row i belongs to image id i. Indexes keep only ids and read rows through row(i).
class EmbeddingStore {
public:
    EmbeddingStore();
    ~EmbeddingStore();
    EmbeddingStore(const EmbeddingStore&) = delete;  // 60 MB: never copied by accident
    EmbeddingStore& operator=(const EmbeddingStore&) = delete;

    // Reads a raw little-endian float32 file and checks it against its manifest and the
    // expected dimension and model: size, dim, row count, dtype, normalised flag, SHA-256,
    // and every row for NaN / Inf / zero norm / norm != 1.
    void load(const std::string& binPath, const std::string& manifestPath, int expectedDim,
              const std::string& expectedModel, double normTolerance);
    // Copies `values` (rows * dim floats) in and validates every row (used by tests).
    void setData(int dim, const FloatArray& values, double normTolerance,
                 const std::string& sourceSha256 = "");

    const float* row(int i) const;
    int size() const;  // number of rows
    int dim() const;
    long long memoryBytes() const;
    const std::string& sourceSha256() const;

private:
    void validate(double normTolerance) const;

    float* values;
    int numRows;
    int numDims;
    std::string sha256;
};

// Throws an Error describing the first problem: wrong dimension, NaN, Inf, zero norm,
// or a norm outside 1 +- normTolerance.
void validateVector(const float* v, int size, int expectedDim, double normTolerance,
                    const std::string& what);

// ---------------------------------------------------------------- caption search
// (dsa/member1_inverted_index.cpp)

typedef uint64_t (*HashFunction)(const std::string& key);
uint64_t fnv1aHash(const std::string& key);

// Hash table from a word to its posting list (sorted image ids), using open addressing
// with linear probing. Slots live in one array whose length is a power of two; a key's
// home slot is hash(key) & (capacity - 1), and a collision moves on to the next slot.
class HashTable {
public:
    HashTable(HashFunction function = fnv1aHash);
    ~HashTable();
    HashTable(const HashTable&) = delete;
    HashTable& operator=(const HashTable&) = delete;

    IntArray& getOrInsert(const std::string& key);  // adds an empty list for a new key
    const IntArray* find(const std::string& key) const;  // nullptr if absent
    int size() const;
    int capacity() const;
    double loadFactor() const;
    double averageProbeLength() const;  // slots a successful lookup inspects, on average

private:
    struct Slot {
        std::string key;
        IntArray value;
        bool used = false;
    };
    int homeSlot(const std::string& key) const;
    void grow();

    Slot* slots;
    int numSlots;
    int count;
    HashFunction hash;
};

// Lowercased words of `text` in order, stop words ("a", "the", ...) removed.
void tokenize(const std::string& text, StringArray& words);

// Intersections and union of two sorted id lists without duplicates; results go in `out`.
void intersectMerge(const IntArray& a, const IntArray& b, IntArray& out);
void intersectGallop(const IntArray& small, const IntArray& large, IntArray& out);
void unionMerge(const IntArray& a, const IntArray& b, IntArray& out);

// For every caption word, the sorted list of image ids whose captions contain it.
class InvertedIndex {
public:
    InvertedIndex();

    // Indexes every caption except caption number `skipCaption` (-1 keeps all). Passing
    // the held-out caption index keeps the query captions out of the index.
    void build(const MetadataManager& meta, int skipCaption);
    // Adds the words of `text` to image `id`. Ids must come in non-decreasing order.
    void add(int id, const std::string& text);

    const IntArray* postings(const std::string& term) const;  // nullptr if unknown
    void matchAll(const std::string& query, IntArray& out) const;  // AND of all words
    void matchAny(const std::string& query, IntArray& out) const;  // OR of all words

    int vocabularySize() const;
    long long postingCount() const;
    const HashTable& table() const;

private:
    HashTable terms;
    long long totalPostings;
    int lastId;
};

#endif
