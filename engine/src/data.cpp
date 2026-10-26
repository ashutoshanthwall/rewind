// Configuration, image metadata, embedding manifest and the embedding store (data.h).
#include "data.h"

#include <cmath>
#include <fstream>
#include <sstream>

#include "storage.h"

static_assert(sizeof(float) == 4, "embedding files hold 4-byte floats");

// ---------------------------------------------------------------- configuration

AppConfig loadConfig(const std::string& configFile) {
    if (!fileExists(configFile)) throw Error("cannot open config file: " + configFile);
    AppConfig c;
    try {
        JsonDocument doc(configFile);
        const JsonNode* j = doc.root();
        c.configFile = absolutePath(configFile);
        c.root = parentDir(parentDir(c.configFile));

        const JsonNode* model = jsonGet(j, "model");
        c.modelName = jsonString(model, "name");
        c.dim = jsonInt(model, "embedding_dim");
        if (c.dim <= 0) throw Error("model.embedding_dim must be > 0");
        c.normTolerance = jsonNumber(jsonGet(j, "embedding"), "norm_tolerance");

        const JsonNode* h = jsonGet(j, "hnsw");
        c.hnsw.M = jsonInt(h, "M");
        c.hnsw.efConstruction = jsonInt(h, "ef_construction");
        c.hnsw.efSearch = jsonInt(h, "ef_search");
        c.hnsw.seed = (uint64_t)jsonNumber(h, "seed");
        c.hnsw.selectHeuristic = jsonBool(h, "select_heuristic");
        c.hnsw.keepPrunedConnections = jsonBool(h, "keep_pruned_connections");
        c.hnsw.protectLastInEdge = jsonBool(h, "protect_last_in_edge");

        const JsonNode* e = jsonGet(j, "evaluation");
        c.evaluation.seed = (uint64_t)jsonNumber(e, "seed");
        c.evaluation.warmupQueries = jsonInt(e, "warmup_queries");
        c.evaluation.repetitions = jsonInt(e, "repetitions");
        c.evaluation.heldoutCaptionIndex = jsonInt(e, "heldout_caption_index");
        const JsonNode* sweep = jsonGet(e, "ef_search_sweep");
        if (sweep->type != JsonNode::ARRAY) throw Error("evaluation.ef_search_sweep must be an array");
        for (const JsonNode* ef = sweep->child; ef != nullptr; ef = ef->next)
            c.evaluation.efSearchSweep.push((int)ef->number);

        const JsonNode* p = jsonGet(j, "paths");
        c.metadata = joinPath(c.root, jsonString(p, "metadata"));
        c.embeddings = joinPath(c.root, jsonString(p, "embeddings"));
        c.manifest = joinPath(c.root, jsonString(p, "manifest"));
        c.index = joinPath(c.root, jsonString(p, "index"));
        c.queries = joinPath(c.root, jsonString(p, "queries"));
        c.queryGroundTruth = joinPath(c.root, jsonString(p, "query_ground_truth"));
        c.queryEmbeddings = joinPath(c.root, jsonString(p, "query_embeddings"));
        c.queryManifest = joinPath(c.root, jsonString(p, "query_manifest"));
        c.benchmarkResults = joinPath(c.root, jsonString(p, "benchmark_results"));
        c.textEncoderServer = joinPath(c.root, jsonString(p, "text_encoder_server"));
        c.python = joinPath(c.root, jsonString(p, "python"));
    } catch (const Error& e) {
        throw Error("config " + configFile + ": " + e.what());
    }
    return c;
}

// ---------------------------------------------------------------- metadata

MetadataManager::MetadataManager() {
    records = nullptr;
    count = 0;
}

MetadataManager::~MetadataManager() {
    delete[] records;
}

void MetadataManager::allocate(int n) {
    delete[] records;
    records = n > 0 ? new ImageRecord[n] : nullptr;
    count = n;
}

ImageRecord& MetadataManager::edit(int id) {
    return records[id];
}

void MetadataManager::checkIds() const {
    for (int i = 0; i < count; i++)
        if (records[i].internalId != i)
            throw Error("metadata ids are not contiguous: record " + std::to_string(i) +
                        " has internal_id " + std::to_string(records[i].internalId));
}

void MetadataManager::load(const std::string& metadataJson, const std::string& root) {
    if (!fileExists(metadataJson)) throw Error("cannot open metadata file: " + metadataJson);
    try {
        JsonDocument doc(metadataJson);
        const JsonNode* list = jsonGet(doc.root(), "records");
        int n = 0;
        for (const JsonNode* r = list->child; r != nullptr; r = r->next) n++;
        if (jsonInt(doc.root(), "count") != n)
            throw Error("metadata count field disagrees with number of records");

        allocate(n);
        int i = 0;
        for (const JsonNode* r = list->child; r != nullptr; r = r->next) {
            ImageRecord& rec = records[i];
            rec.internalId = jsonInt(r, "internal_id");
            rec.imageId = jsonString(r, "image_id");
            rec.path = joinPath(root, jsonString(r, "path"));
            rec.width = jsonInt(r, "width");
            rec.height = jsonInt(r, "height");
            const JsonNode* split = jsonFind(r, "split");
            rec.split = (split != nullptr && split->type == JsonNode::STRING) ? split->text : "";
            const JsonNode* caps = jsonGet(r, "captions");
            if (caps->type != JsonNode::ARRAY)
                throw Error("captions of record " + std::to_string(rec.internalId) + " is not an array");
            for (const JsonNode* cap = caps->child; cap != nullptr; cap = cap->next)
                rec.captions.push(cap->text);
            i++;
        }
    } catch (const Error& e) {
        throw Error("malformed metadata " + metadataJson + ": " + e.what());
    }
    checkIds();
}

const ImageRecord& MetadataManager::at(int id) const {
    if (id < 0 || id >= count)
        throw Error("internal id " + std::to_string(id) + " out of range [0, " +
                    std::to_string(count) + ")");
    return records[id];
}

int MetadataManager::size() const {
    return count;
}

// ---------------------------------------------------------------- manifest

EmbeddingManifest loadManifest(const std::string& path) {
    if (!fileExists(path)) throw Error("cannot open embedding manifest: " + path);
    EmbeddingManifest m;
    try {
        JsonDocument doc(path);
        const JsonNode* j = doc.root();
        m.model = jsonString(j, "model");
        m.dim = jsonInt(j, "dim");
        m.rows = jsonInt(j, "rows");
        m.l2Normalized = jsonBool(j, "l2_normalized");
        m.dtype = jsonString(j, "dtype");
        m.sha256 = jsonString(j, "sha256");
        if (jsonFind(j, "created_utc") != nullptr) m.createdUtc = jsonString(j, "created_utc");
        if (jsonFind(j, "git_commit") != nullptr) m.gitCommit = jsonString(j, "git_commit");
    } catch (const Error& e) {
        throw Error("malformed manifest " + path + ": " + e.what());
    }
    return m;
}

// ---------------------------------------------------------------- embedding store

long long floatCount(long long rows, long long dim) {
    if (dim <= 0) throw Error("embedding dimension must be > 0, got " + std::to_string(dim));
    if (rows < 0) throw Error("row count must be >= 0, got " + std::to_string(rows));
    if (rows > 2147483647LL)
        throw Error(std::to_string(rows) + " rows: ids are int, at most 2147483647 rows");
    if (rows > 0 && dim > 9223372036854775807LL / 4 / rows)
        throw Error(std::to_string(rows) + " x " + std::to_string(dim) + " floats overflow 64 bits");
    return rows * dim;
}

void validateVector(const float* v, int size, int expectedDim, double normTolerance,
                    const std::string& what) {
    if (size != expectedDim)
        throw Error(what + ": dimension mismatch (got " + std::to_string(size) + ", expected " +
                    std::to_string(expectedDim) + ")");
    for (int i = 0; i < size; i++) {
        if (std::isnan(v[i])) throw Error(what + ": NaN at element " + std::to_string(i));
        if (std::isinf(v[i])) throw Error(what + ": Inf at element " + std::to_string(i));
    }
    double n = norm(v, size);
    if (n == 0.0) throw Error(what + ": zero-norm vector");
    if (std::fabs(n - 1.0) > normTolerance) {
        std::ostringstream os;
        os.precision(9);
        os << what << ": L2 norm " << n << " outside 1 +- " << normTolerance
           << " (vectors must be unit-normalised)";
        throw Error(os.str());
    }
}

EmbeddingStore::EmbeddingStore() {
    values = nullptr;
    numRows = 0;
    numDims = 0;
}

EmbeddingStore::~EmbeddingStore() {
    delete[] values;
}

void EmbeddingStore::validate(double normTolerance) const {
    for (int i = 0; i < numRows; i++)
        validateVector(row(i), numDims, numDims, normTolerance, "embedding row " + std::to_string(i));
}

void EmbeddingStore::setData(int dim, const FloatArray& data, double normTolerance,
                             const std::string& sourceSha256) {
    if (dim <= 0) throw Error("embedding dimension must be > 0");
    if (data.size() % dim != 0)
        throw Error("embedding buffer of " + std::to_string(data.size()) +
                    " floats is not a multiple of dim " + std::to_string(dim));
    delete[] values;
    numDims = dim;
    numRows = data.size() / dim;
    values = new float[data.size() > 0 ? data.size() : 1];
    for (int i = 0; i < data.size(); i++) values[i] = data[i];
    sha256 = sourceSha256;
    validate(normTolerance);
}

void EmbeddingStore::setRows(int dim, long long rows, const float* data, double normTolerance) {
    long long n = floatCount(rows, dim);
    if (n > 0 && data == nullptr) throw Error("setRows: null data pointer");
    delete[] values;
    numDims = dim;
    numRows = (int)rows;
    values = new float[n > 0 ? n : 1];
    for (long long i = 0; i < n; i++) values[i] = data[i];
    sha256 = sha256Hex(values, n * 4);
    validate(normTolerance);
}

void EmbeddingStore::load(const std::string& binPath, const std::string& manifestPath,
                          int expectedDim, const std::string& expectedModel,
                          double normTolerance) {
    EmbeddingManifest m = loadManifest(manifestPath);
    if (m.dim != expectedDim)
        throw Error("dimension mismatch: manifest says " + std::to_string(m.dim) +
                    ", config says " + std::to_string(expectedDim));
    if (m.model != expectedModel)
        throw Error("model mismatch: manifest '" + m.model + "', config '" + expectedModel + "'");
    if (m.dtype != "float32") throw Error("unsupported dtype " + m.dtype);
    if (!m.l2Normalized) throw Error("manifest says embeddings are not L2-normalised");

    long long bytes = fileSize(binPath);
    if (bytes < 0) throw Error("cannot open embedding file " + binPath);
    long long expected = (long long)m.rows * m.dim * 4;
    if (bytes != expected)
        throw Error(binPath + ": size " + std::to_string(bytes) + " bytes, manifest implies " +
                    std::to_string(expected) + " (truncated or wrong file)");

    long long n = floatCount(m.rows, m.dim);
    delete[] values;
    numRows = m.rows;
    numDims = m.dim;
    values = new float[n > 0 ? n : 1];
    std::ifstream in(binPath.c_str(), std::ios::binary);
    if (!in.read((char*)values, bytes)) throw Error("short read from " + binPath);
    std::string sha = sha256Hex(values, bytes);
    if (sha != m.sha256)
        throw Error(binPath + ": SHA-256 " + sha + " does not match manifest " + m.sha256);
    sha256 = sha;
    validate(normTolerance);
}

const float* EmbeddingStore::row(int i) const {
    return values + (long long)i * numDims;
}

int EmbeddingStore::size() const { return numRows; }
int EmbeddingStore::dim() const { return numDims; }
long long EmbeddingStore::memoryBytes() const { return (long long)numRows * numDims * 4; }
const std::string& EmbeddingStore::sourceSha256() const { return sha256; }
