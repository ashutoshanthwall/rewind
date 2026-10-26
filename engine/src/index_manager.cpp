// Index lifecycle: build, save, load, validate, rebuild (storage.h).
// The file is assembled in a std::string used as a byte buffer, then written in one go.
#include "storage.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

static const char MAGIC[8] = {'S', 'I', 'R', 'H', 'N', 'S', 'W', '\0'};

// ---------------------------------------------------------------- writing bytes

static void putBytes(std::string& out, const void* p, long long n) {
    out.append((const char*)p, n);
}
static void putU8(std::string& out, uint8_t v) { putBytes(out, &v, 1); }
static void putU32(std::string& out, uint32_t v) { putBytes(out, &v, 4); }
static void putI32(std::string& out, int32_t v) { putBytes(out, &v, 4); }
static void putU64(std::string& out, uint64_t v) { putBytes(out, &v, 8); }

static void putString(std::string& out, const std::string& s) {
    putU32(out, (uint32_t)s.size());
    putBytes(out, s.data(), (long long)s.size());
}

// ---------------------------------------------------------------- reading bytes

// Reads values one after another from buf[0..end-1], throwing if the file is too short.
class Reader {
public:
    Reader(const std::string& buffer, long long endPos, const std::string& fileName)
        : buf(buffer), file(fileName) {
        end = endPos;
        pos = 0;
    }

    void getBytes(void* p, long long n) {
        need(n);
        memcpy(p, buf.data() + pos, n);
        pos += n;
    }
    uint8_t getU8() { uint8_t v; getBytes(&v, 1); return v; }
    uint32_t getU32() { uint32_t v; getBytes(&v, 4); return v; }
    int32_t getI32() { int32_t v; getBytes(&v, 4); return v; }
    uint64_t getU64() { uint64_t v; getBytes(&v, 8); return v; }

    std::string getFixed(long long n) {
        need(n);
        std::string s = buf.substr(pos, n);
        pos += n;
        return s;
    }
    std::string getString(uint32_t maxLength) {
        uint32_t n = getU32();
        if (n > maxLength) throw PersistenceError(file + ": corrupted header (string length)");
        return getFixed(n);
    }
    long long position() const { return pos; }

private:
    void need(long long n) {
        if (pos + n > end)
            throw PersistenceError(file + ": index file is truncated (needed " +
                                   std::to_string(pos + n) + " bytes, have " +
                                   std::to_string(end) + ")");
    }
    const std::string& buf;
    std::string file;
    long long pos;
    long long end;
};

// ---------------------------------------------------------------- build / save

double IndexManager::build(HNSWIndex& index) {
    const EmbeddingStore& store = index.store();
    double start = nowMs();
    for (int i = 0; i < store.size(); i++) index.insert(i, store.row(i));
    return (nowMs() - start) / 1000.0;
}

void IndexManager::save(const HNSWIndex& index, const std::string& modelName,
                        const std::string& path) {
    const EmbeddingStore& store = index.store();
    if (store.sourceSha256().size() != 64)
        throw PersistenceError("cannot save: the embedding store has no source SHA-256");
    std::string out;
    putBytes(out, MAGIC, 8);
    putU32(out, FORMAT_VERSION);
    putString(out, modelName);
    putU64(out, (uint64_t)store.dim());
    putU64(out, (uint64_t)store.size());
    putU64(out, (uint64_t)index.size());
    const HnswParams& p = index.params();
    putI32(out, p.M);
    putI32(out, p.efConstruction);
    putI32(out, p.efSearch);
    putU64(out, p.seed);
    putU8(out, p.selectHeuristic ? 1 : 0);
    putU8(out, p.keepPrunedConnections ? 1 : 0);
    putU8(out, p.protectLastInEdge ? 1 : 0);
    putBytes(out, store.sourceSha256().data(), 64);
    putI32(out, index.entryPoint());
    putI32(out, index.maxLevel());
    uint64_t words[Random64::STATE_SIZE];
    int position = 0;
    index.getRandomState(words, position);
    putU32(out, (uint32_t)position);
    for (int i = 0; i < Random64::STATE_SIZE; i++) putU64(out, words[i]);

    std::string payload;
    for (int id = 0; id < store.size(); id++) {
        int level = index.levelOf(id);
        putI32(payload, level);
        for (int l = 0; l <= level; l++) {
            const IntArray& nb = index.neighbors(id, l);
            putU32(payload, (uint32_t)nb.size());
            putBytes(payload, nb.data(), (long long)nb.size() * 4);
        }
    }
    putU64(out, (uint64_t)payload.size());
    out += payload;

    Sha256 hasher;
    hasher.update(out.data(), (long long)out.size());
    unsigned char digest[32];
    hasher.finish(digest);
    putBytes(out, digest, 32);

    makeDirs(parentDir(path));
    std::string tmp = path + ".tmp";
    {
        std::ofstream file(tmp.c_str(), std::ios::binary | std::ios::trunc);
        if (!file.write(out.data(), (long long)out.size()))
            throw PersistenceError("failed writing " + tmp);
    }
    // Write to a temporary file, then rename: a crash never leaves a half-written index.
    if (std::rename(tmp.c_str(), path.c_str()) != 0)
        throw PersistenceError("failed to rename " + tmp + " to " + path);
}

// ---------------------------------------------------------------- load

void IndexManager::load(const std::string& path, const EmbeddingStore& store,
                        const std::string& modelName, HNSWIndex& index) {
    const std::string& f = path;
    if (!fileExists(path))
        throw PersistenceError("index file not found: " + f + " (build it with sir_build_index)");
    std::string buf;
    {
        std::ifstream in(path.c_str(), std::ios::binary);
        std::stringstream contents;
        contents << in.rdbuf();
        buf = contents.str();
    }
    long long size = (long long)buf.size();
    if (size < 8 + 4 + 32)
        throw PersistenceError(f + ": index file is truncated (" + std::to_string(size) + " bytes)");
    if (memcmp(buf.data(), MAGIC, 8) != 0)
        throw PersistenceError(f + ": not an index file (bad magic number)");

    long long body = size - 32;
    Reader r(buf, body, f);
    r.getFixed(8);
    uint32_t version = r.getU32();
    if (version != FORMAT_VERSION)
        throw PersistenceError(f + ": unsupported format version " + std::to_string(version) +
                               " (expected " + std::to_string(FORMAT_VERSION) +
                               "; rebuild it with sir_build_index)");
    std::string model = r.getString(256);
    uint64_t dim = r.getU64();
    uint64_t rows = r.getU64();
    uint64_t vectors = r.getU64();
    HnswParams p;
    p.M = r.getI32();
    p.efConstruction = r.getI32();
    p.efSearch = r.getI32();
    p.seed = r.getU64();
    p.selectHeuristic = r.getU8() != 0;
    p.keepPrunedConnections = r.getU8() != 0;
    p.protectLastInEdge = r.getU8() != 0;
    std::string sourceSha = r.getFixed(64);
    int entry = r.getI32();
    int maxLevel = r.getI32();
    uint32_t rngPosition = r.getU32();
    uint64_t rngWords[Random64::STATE_SIZE];
    for (int i = 0; i < Random64::STATE_SIZE; i++) rngWords[i] = r.getU64();
    uint64_t payloadBytes = r.getU64();
    if (r.position() + (long long)payloadBytes != body)
        throw PersistenceError(f + ": index file is truncated or has trailing data (header declares " +
                               std::to_string(r.position() + (long long)payloadBytes + 32) +
                               " bytes, file has " + std::to_string(size) + ")");

    Sha256 hasher;
    hasher.update(buf.data(), body);
    unsigned char digest[32];
    hasher.finish(digest);
    if (memcmp(digest, buf.data() + body, 32) != 0)
        throw PersistenceError(f + ": index file is corrupted (SHA-256 checksum mismatch)");

    if (model != modelName)
        throw PersistenceError(f + ": stale index: built for model '" + model +
                               "', current model is '" + modelName + "' - rebuild it");
    if (dim != (uint64_t)store.dim())
        throw PersistenceError(f + ": dimension mismatch: index dim " + std::to_string(dim) +
                               ", embeddings dim " + std::to_string(store.dim()));
    if (rows != (uint64_t)store.size())
        throw PersistenceError(f + ": stale index: built on " + std::to_string(rows) +
                               " embeddings, current file has " + std::to_string(store.size()));
    if (sourceSha != store.sourceSha256())
        throw PersistenceError(f + ": stale index: built from embeddings " +
                               sourceSha.substr(0, 16) + "..., current embeddings are " +
                               store.sourceSha256().substr(0, 16) + "... - rebuild it");
    if (rngPosition > (uint32_t)Random64::STATE_SIZE)
        throw PersistenceError(f + ": corrupted random-number state");

    index.setup(store, p);
    for (int i = 0; i < (int)rows; i++) {
        int level = r.getI32();
        if (level < -1 || level > 64)
            throw PersistenceError(f + ": corrupted node level at id " + std::to_string(i));
        index.restoreNode(i, level);
        for (int l = 0; l <= level; l++) {
            uint32_t n = r.getU32();
            if (n > (uint32_t)index.maxDegree(l))
                throw PersistenceError(f + ": corrupted degree at id " + std::to_string(i));
            IntArray& links = index.restoreLinks(i, l);
            for (uint32_t c = 0; c < n; c++) {
                int e = r.getI32();
                if (e < 0 || e >= (int)rows)
                    throw PersistenceError(f + ": corrupted link at id " + std::to_string(i));
                links.push(e);
            }
        }
    }
    index.finishRestore(entry, maxLevel, rngWords, (int)rngPosition);
    if ((uint64_t)index.size() != vectors)
        throw PersistenceError(f + ": node count does not match header");
    validate(index);
}

// ---------------------------------------------------------------- validate / rebuild

IndexValidation IndexManager::validate(const HNSWIndex& index) {
    IndexValidation v;
    int n = index.store().size();
    v.nodes = index.size();
    v.maxLevel = index.maxLevel();
    if (v.nodes == 0) return v;
    if (!index.contains(index.entryPoint()) || index.levelOf(index.entryPoint()) != v.maxLevel)
        throw Error("invalid entry point " + std::to_string(index.entryPoint()));
    v.nodesPerLevel = IntArray(v.maxLevel + 1, 0);
    long long degreeSum = 0;
    for (int id = 0; id < n; id++) {
        if (!index.contains(id)) continue;
        int level = index.levelOf(id);
        if (level > v.maxLevel) throw Error("node " + std::to_string(id) + " above the max level");
        for (int l = 0; l <= level; l++) {
            v.nodesPerLevel[l]++;
            const IntArray& nb = index.neighbors(id, l);
            if (nb.size() > index.maxDegree(l))
                throw Error("node " + std::to_string(id) + " exceeds max degree on layer " +
                            std::to_string(l));
            for (int i = 0; i < nb.size(); i++) {
                int e = nb[i];
                if (e == id) throw Error("self loop at node " + std::to_string(id));
                if (index.levelOf(e) < l)
                    throw Error("node " + std::to_string(id) + " links to " + std::to_string(e) +
                                " which is absent from layer " + std::to_string(l));
                for (int j = 0; j < i; j++)  // lists are short (at most 2M), so a scan is enough
                    if (nb[j] == e) throw Error("duplicate link at node " + std::to_string(id));
            }
            if (l == 0) degreeSum += nb.size();
        }
    }
    v.meanDegreeLayer0 = (double)degreeSum / v.nodes;

    // Breadth-first search on layer 0 from the entry point. Every node enters the queue at
    // most once, so a plain array with a read position `head` works as the FIFO queue.
    BitArray seen(n, false);
    IntArray queue;
    queue.reserve(n);
    queue.push(index.entryPoint());
    seen.set(index.entryPoint(), true);
    for (int head = 0; head < queue.size(); head++) {
        const IntArray& nb = index.neighbors(queue[head], 0);
        for (int i = 0; i < nb.size(); i++) {
            if (!seen.get(nb[i])) {
                seen.set(nb[i], true);
                queue.push(nb[i]);
            }
        }
    }
    v.reachableLayer0 = queue.size();

    IntArray inDegree(n, 0);
    for (int id = 0; id < n; id++) {
        if (!index.contains(id)) continue;
        const IntArray& nb = index.neighbors(id, 0);
        for (int i = 0; i < nb.size(); i++) inDegree[nb[i]]++;
    }
    for (int id = 0; id < n; id++) {
        if (!index.contains(id)) continue;
        if (!seen.get(id)) v.unreachableLayer0.push(id);
        if (inDegree[id] == 0 && id != index.entryPoint()) v.inDegreeZeroLayer0++;
    }
    return v;
}

double IndexManager::rebuild(HNSWIndex& index, const std::string& modelName,
                             const std::string& path) {
    double seconds = build(index);
    validate(index);
    save(index, modelName, path);
    return seconds;
}
