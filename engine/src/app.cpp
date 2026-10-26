// Helpers shared by the command-line programs (storage.h): argument parsing, loading the
// standard files, and the brute-force vs HNSW comparison used by sir_cli.
#include "storage.h"

#include <cmath>
#include <cstdio>
#include <sstream>

// ---------------------------------------------------------------- arguments

bool Args::has(const std::string& key) const {
    return keys.contains(key);
}

std::string Args::get(const std::string& key, const std::string& fallback) const {
    for (int i = 0; i < keys.size(); i++)
        if (keys[i] == key) return values[i];
    return fallback;
}

int Args::getInt(const std::string& key, int fallback) const {
    if (!has(key)) return fallback;
    return std::stoi(get(key, ""));
}

Args parseArgs(int argc, char** argv, const StringArray& flags) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        if (s.size() > 2 && s[0] == '-' && s[1] == '-') {
            std::string key = s.substr(2);
            a.keys.push(key);
            if (flags.contains(key) || i + 1 >= argc)
                a.values.push("1");
            else
                a.values.push(argv[++i]);
        } else {
            a.positional.push(s);
        }
    }
    return a;
}

AppConfig loadAppConfig(const Args& args) {
    return loadConfig(args.get("config", "config/config.json"));
}

void loadImageStore(const AppConfig& c, EmbeddingStore& out) {
    out.load(c.embeddings, c.manifest, c.dim, c.modelName, c.normTolerance);
}

void loadQueryStore(const AppConfig& c, EmbeddingStore& out) {
    out.load(c.queryEmbeddings, c.queryManifest, c.dim, c.modelName, c.normTolerance);
}

// ---------------------------------------------------------------- brute force vs HNSW

// Runs the search `runs` times and returns the median retrieval time; the last response
// is left in `out`.
static double medianRetrievalMs(const SearchEngine& engine, const FloatArray& query, int k,
                                int runs, SearchResponse& out) {
    double* ms = new double[runs];
    for (int i = 0; i < runs; i++) {
        engine.searchEmbedding(query, k, out);
        ms[i] = out.retrievalMs;
    }
    double median = summarize(ms, runs).medianMs;
    delete[] ms;
    return median;
}

void compareBackends(const std::string& text, int k, const SearchEngine& exact,
                     const SearchEngine& approx, int timingRuns, Comparison& out) {
    exact.checkK(k);
    out.query = SearchEngine::normalizeQueryText(text);
    out.k = k;
    double start = nowMs();
    FloatArray query;
    exact.embedQuery(out.query, query);
    out.encodeMs = nowMs() - start;
    out.exactMs = medianRetrievalMs(exact, query, k, timingRuns, out.exact);
    out.approxMs = medianRetrievalMs(approx, query, k, timingRuns, out.approx);
    int overlap = 0;
    for (int i = 0; i < out.approx.hits.size(); i++)
        for (int j = 0; j < out.exact.hits.size(); j++)
            if (out.approx.hits[i].id == out.exact.hits[j].id) overlap++;
    out.recall = (double)overlap / k;
    out.speedup = out.exactMs / out.approxMs;
}

// ---------------------------------------------------------------- writing JSON

std::string jsonQuote(const std::string& s) {
    std::string out = "\"";
    for (size_t i = 0; i < s.size(); i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') {
            out += '\\';
            out += (char)c;
        } else if (c < 0x20) {
            char buf[8];
            snprintf(buf, sizeof buf, "\\u%04x", c);
            out += buf;
        } else {
            out += (char)c;
        }
    }
    return out + "\"";
}

std::string jsonNumberText(double value) {
    if (!std::isfinite(value)) return "0";  // JSON has no NaN or Infinity
    std::ostringstream os;
    os.precision(10);
    os << value;
    return os.str();
}
