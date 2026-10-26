// Interactive natural-language image search: brute force vs HNSW side by side.
//
//   sir_cli [--config f] [--k 5] [--ef N] [--encoder-log file] [query words...]
//   sir_cli --serve [--config f] [--encoder-log file]
//
// With a query on the command line it answers once; otherwise it reads queries from stdin
// until end of input or an empty line. Text is embedded by the Python CLIP helper; all
// retrieval, ranking and metadata lookup is C++.
//
// --serve is the engine behind the web page (scripts/web_server.py). It first prints one
// JSON line describing the collection, then reads requests "k<TAB>ef<TAB>query", one per
// line, and answers each with one JSON line, until end of input. stdout carries only these
// JSON lines; progress messages go to stderr.
#include <iomanip>
#include <iostream>
#include <string>

#include "storage.h"

static const int TIMING_RUNS = 5;  // each query is timed several times; the median is shown
static const int MAX_WEB_K = 50;     // keeps the web page readable; the engine allows K <= N
static const int MAX_WEB_EF = 1000;

static void printResults(const std::string& title, const SearchEngine& engine,
                         const SearchResponse& r, double ms) {
    std::cout << "\n" << title << "  (retrieval " << std::fixed << std::setprecision(3) << ms
              << " ms)\n";
    for (int i = 0; i < r.hits.size(); i++) {
        const ImageRecord& rec = engine.record(r.hits[i].id);
        std::cout << "  " << i + 1 << ". " << std::setprecision(4) << r.hits[i].score << "  "
                  << rec.imageId << "  " << rec.path
                  << (fileExists(rec.path) ? "" : "  [MISSING ON DISK]") << "\n";
    }
}

static void answer(const std::string& query, int k, const SearchEngine& bfEngine,
                   const SearchEngine& hnswEngine) {
    Comparison c;
    compareBackends(query, k, bfEngine, hnswEngine, TIMING_RUNS, c);
    std::cout << "\nquery: \"" << c.query << "\"   (text embedding " << std::fixed
              << std::setprecision(1) << c.encodeMs << " ms, excluded from retrieval latency)";
    printResults("Brute force (exact) top-" + std::to_string(k), bfEngine, c.exact, c.exactMs);
    printResults("HNSW top-" + std::to_string(k), hnswEngine, c.approx, c.approxMs);
    std::cout << "\nRecall@" << k << " (HNSW vs brute force): " << std::setprecision(2) << c.recall
              << "   speedup: " << std::setprecision(1) << c.speedup << "x   (median of "
              << TIMING_RUNS << " runs each)\n";
}

// ---------------------------------------------------------------- --serve mode

static std::string selectionName(const HnswParams& p) {
    if (!p.selectHeuristic) return "naive";
    return p.keepPrunedConnections ? "heuristic+keepPruned" : "heuristic";
}

static bool containsId(const CandidateArray& hits, int id) {
    for (int i = 0; i < hits.size(); i++)
        if (hits[i].id == id) return true;
    return false;
}

// One index's results as a JSON array. "in_other" says whether the other index also
// returned the image, so the page can mark the differences.
static std::string resultsJson(const SearchEngine& engine, const SearchResponse& r,
                               const SearchResponse& other) {
    std::string out = "[";
    for (int i = 0; i < r.hits.size(); i++) {
        const Candidate& c = r.hits[i];
        const ImageRecord& rec = engine.record(c.id);
        if (i > 0) out += ",";
        out += "{\"rank\":" + std::to_string(i + 1) + ",\"id\":" + std::to_string(c.id) +
               ",\"image_id\":" + jsonQuote(rec.imageId) + ",\"score\":" + jsonNumberText(c.score) +
               ",\"caption\":" + jsonQuote(rec.captions.size() > 0 ? rec.captions[0] : "") +
               ",\"in_other\":" + (containsId(other.hits, c.id) ? "true" : "false") +
               ",\"url\":\"/img/" + std::to_string(c.id) + "\"}";
    }
    return out + "]";
}

static std::string errorJson(int status, const std::string& message) {
    return "{\"status\":" + std::to_string(status) + ",\"error\":" + jsonQuote(message) + "}";
}

// Answers one request line "k<TAB>ef<TAB>query".
static std::string serveRequest(const std::string& line, HNSWIndex& hnsw,
                                const SearchEngine& bfEngine, const SearchEngine& hnswEngine) {
    size_t tab1 = line.find('\t');
    size_t tab2 = tab1 == std::string::npos ? std::string::npos : line.find('\t', tab1 + 1);
    if (tab2 == std::string::npos) return errorJson(400, "malformed request");
    int k = 0, ef = 0;
    try {
        k = std::stoi(line.substr(0, tab1));
        ef = std::stoi(line.substr(tab1 + 1, tab2 - tab1 - 1));
    } catch (const std::exception&) {
        return errorJson(400, "k and ef must be integers");
    }
    std::string query = line.substr(tab2 + 1);
    try {
        if (k > MAX_WEB_K)
            throw QueryError("K is limited to " + std::to_string(MAX_WEB_K) + " in the web UI");
        if (ef < 1 || ef > MAX_WEB_EF)
            throw QueryError("efSearch must be between 1 and " + std::to_string(MAX_WEB_EF));
        hnsw.setEfSearch(ef);
        Comparison c;
        compareBackends(query, k, bfEngine, hnswEngine, TIMING_RUNS, c);
        return "{\"status\":200,\"query\":" + jsonQuote(c.query) + ",\"k\":" + std::to_string(k) +
               ",\"ef\":" + std::to_string(ef) + ",\"encode_ms\":" + jsonNumberText(c.encodeMs) +
               ",\"recall\":" + jsonNumberText(c.recall) + ",\"speedup\":" + jsonNumberText(c.speedup) +
               ",\"exact\":{\"ms\":" + jsonNumberText(c.exactMs) +
               ",\"results\":" + resultsJson(bfEngine, c.exact, c.approx) + "}" +
               ",\"hnsw\":{\"ms\":" + jsonNumberText(c.approxMs) +
               ",\"results\":" + resultsJson(hnswEngine, c.approx, c.exact) + "}}";
    } catch (const QueryError& e) {
        return errorJson(400, e.what());
    } catch (const Error& e) {
        return errorJson(500, e.what());
    }
}

static void serve(HNSWIndex& hnsw, const EmbeddingStore& store, const AppConfig& cfg,
                  const SearchEngine& bfEngine, const SearchEngine& hnswEngine) {
    const HnswParams& p = hnsw.params();
    std::cout << "{\"images\":" << store.size() << ",\"dim\":" << store.dim()
              << ",\"model\":" << jsonQuote(cfg.modelName) << ",\"M\":" << p.M
              << ",\"ef_construction\":" << p.efConstruction
              << ",\"ef_search_default\":" << cfg.hnsw.efSearch << ",\"max_k\":" << MAX_WEB_K
              << ",\"max_ef\":" << MAX_WEB_EF << ",\"selection\":" << jsonQuote(selectionName(p))
              << "}" << std::endl;
    std::string line;
    while (std::getline(std::cin, line))
        std::cout << serveRequest(line, hnsw, bfEngine, hnswEngine) << std::endl;
}

int main(int argc, char** argv) {
    try {
        StringArray flags;
        flags.push("serve");
        Args args = parseArgs(argc, argv, flags);
        AppConfig cfg = loadAppConfig(args);
        int k = args.getInt("k", 5);
        std::cerr << "loading metadata, embeddings and index...\n";
        MetadataManager meta;
        meta.load(cfg.metadata, cfg.root);
        EmbeddingStore store;
        loadImageStore(cfg, store);
        BruteForceIndex bf(store);
        bf.insertAll();
        HNSWIndex hnsw;
        IndexManager::load(args.get("index", cfg.index), store, cfg.modelName, hnsw);
        hnsw.setEfSearch(args.getInt("ef", cfg.hnsw.efSearch));
        std::cerr << "starting text encoder...\n";
        TextEncoderProcess encoder(cfg.python, cfg.textEncoderServer, cfg.configFile, cfg.dim,
                                   args.get("encoder-log", "/dev/null"));
        SearchEngine bfEngine(meta, store, bf, &encoder);
        SearchEngine hnswEngine(meta, store, hnsw, &encoder);
        std::cerr << "ready: " << store.size() << " images, HNSW M=" << hnsw.params().M
                  << " efSearch=" << hnsw.params().efSearch << "\n";

        if (args.has("serve")) {
            serve(hnsw, store, cfg, bfEngine, hnswEngine);
            return 0;
        }
        if (args.positional.size() > 0) {
            std::string q;
            for (int i = 0; i < args.positional.size(); i++)
                q += (q.empty() ? "" : " ") + args.positional[i];
            answer(q, k, bfEngine, hnswEngine);
            return 0;
        }
        std::string line;
        while (true) {
            std::cout << "\nsearch> " << std::flush;
            if (!std::getline(std::cin, line) || line.empty()) break;
            try {
                answer(line, k, bfEngine, hnswEngine);
            } catch (const QueryError& e) {
                std::cout << "invalid query: " << e.what() << "\n";
            }
        }
        return 0;
    } catch (const QueryError& e) {
        std::cerr << "invalid query: " << e.what() << '\n';
        return 2;
    } catch (const Error& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
