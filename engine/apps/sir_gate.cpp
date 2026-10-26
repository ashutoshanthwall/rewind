// Verification helper for gates that compare C++ against Python/NumPy or across
// processes. Subcommands:
//   dump-rows <id>...                 exact row values (hex floats)            [G5]
//   bf-topk --n 50 --k 10 --out f     brute-force top-k for the first n queries [G7]
//   persist-save --index f --out r    build HNSW, save, record n query results  [G10]
//   persist-check --index f --expect r   load in a fresh process, compare       [G10]
//   load-index --index f              try to load an index, report the outcome  [G10]
//   caption-index --n 200             caption inverted index vs a linear scan    [G12]
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "storage.h"

static std::string hexFloat(float f) {
    char buf[64];
    snprintf(buf, sizeof buf, "%a", (double)f);
    return buf;
}

static std::string resultsText(const HNSWIndex& idx, const EmbeddingStore& queries, int n, int k) {
    std::ostringstream os;
    CandidateArray hits;
    for (int q = 0; q < n; q++) {
        idx.search(queries.row(q), queries.dim(), k, hits);
        for (int i = 0; i < hits.size(); i++)
            os << q << ' ' << hits[i].id << ' ' << hexFloat(hits[i].score) << '\n';
    }
    return os.str();
}

// Builds the caption inverted index, then answers n random AND and OR queries with it and
// with a linear scan over every image's words; the answers must be identical. Query words
// come from random indexed captions, so most queries have matches.
static int captionIndexGate(const AppConfig& cfg, int n) {
    MetadataManager meta;
    meta.load(cfg.metadata, cfg.root);
    int skip = cfg.evaluation.heldoutCaptionIndex;
    double start = nowMs();
    InvertedIndex idx;
    idx.build(meta, skip);
    double buildMs = nowMs() - start;

    // Baseline data: each image's distinct words (a forward index).
    StringArray* docs = new StringArray[meta.size()];
    StringArray words;
    for (int i = 0; i < meta.size(); i++) {
        const StringArray& caps = meta.at(i).captions;
        for (int c = 0; c < caps.size(); c++) {
            if (c == skip) continue;
            tokenize(caps[c], words);
            for (int w = 0; w < words.size(); w++)
                if (!docs[i].contains(words[w])) docs[i].push(words[w]);
        }
    }

    Random64 rng;
    rng.seed(cfg.evaluation.seed);
    StringArray queries;
    while (queries.size() < n) {
        const StringArray& caps = meta.at((int)(rng.next() % (uint64_t)meta.size())).captions;
        int c = (int)(rng.next() % (uint64_t)caps.size());
        if (c == skip) continue;
        tokenize(caps[c], words);
        if (words.size() == 0) continue;
        std::string q;
        for (uint64_t t = 1 + rng.next() % 3; t > 0; t--)
            q += words[(int)(rng.next() % (uint64_t)words.size())] + ' ';
        queries.push(q);
    }

    int mismatches = 0;
    long long hitsAll = 0, hitsAny = 0;
    double indexMs = 0, scanMs = 0;
    IntArray all, any, wantAll, wantAny;
    for (int qi = 0; qi < queries.size(); qi++) {
        const std::string& q = queries[qi];
        tokenize(q, words);
        double t = nowMs();
        idx.matchAll(q, all);
        idx.matchAny(q, any);
        indexMs += nowMs() - t;
        t = nowMs();
        wantAll.clear();
        wantAny.clear();
        for (int d = 0; d < meta.size(); d++) {
            int found = 0;
            for (int w = 0; w < words.size(); w++)
                if (docs[d].contains(words[w])) found++;
            if (found == words.size()) wantAll.push(d);
            if (found > 0) wantAny.push(d);
        }
        scanMs += nowMs() - t;
        if (!all.equals(wantAll)) mismatches++;
        if (!any.equals(wantAny)) mismatches++;
        hitsAll += all.size();
        hitsAny += any.size();
    }
    delete[] docs;

    const HashTable& table = idx.table();
    std::cout << std::fixed << std::setprecision(3) << "images=" << meta.size()
              << " skipped_caption=" << skip << " vocabulary=" << idx.vocabularySize()
              << " postings=" << idx.postingCount() << " build_ms=" << buildMs << '\n'
              << "hash table: capacity=" << table.capacity() << " load_factor=" << table.loadFactor()
              << " avg_probe_length=" << table.averageProbeLength() << '\n'
              << "queries=" << n << " (1-3 words) mean_hits AND=" << (double)hitsAll / n
              << " OR=" << (double)hitsAny / n << '\n'
              << "AND+OR per query: index_us=" << 1000 * indexMs / n
              << " linear_scan_us=" << 1000 * scanMs / n << " speedup=" << std::setprecision(1)
              << scanMs / indexMs << "x\n"
              << "result mismatches vs linear scan: " << mismatches << " -> "
              << (mismatches == 0 ? "PASS" : "FAIL") << '\n';
    return mismatches == 0 ? 0 : 1;
}

static int run(int argc, char** argv) {
    Args args = parseArgs(argc, argv, StringArray());
    if (args.positional.size() == 0) {
        std::cerr << "usage: sir_gate <dump-rows|bf-topk|persist-save|persist-check|load-index|"
                     "caption-index>\n";
        return 2;
    }
    std::string cmd = args.positional[0];
    AppConfig cfg = loadAppConfig(args);
    if (cmd == "caption-index") return captionIndexGate(cfg, args.getInt("n", 200));

    EmbeddingStore store;
    loadImageStore(cfg, store);
    if (cmd == "dump-rows") {
        std::cout << "rows=" << store.size() << " dim=" << store.dim() << '\n';
        for (int p = 1; p < args.positional.size(); p++) {
            int id = std::stoi(args.positional[p]);
            if (id < 0 || id >= store.size()) throw Error("row id out of range");
            std::cout << id;
            const float* row = store.row(id);
            for (int j = 0; j < store.dim(); j++) std::cout << ' ' << hexFloat(row[j]);
            std::cout << '\n';
        }
        return 0;
    }
    EmbeddingStore queries;
    loadQueryStore(cfg, queries);
    int n = args.getInt("n", 50), k = args.getInt("k", 10);

    if (cmd == "bf-topk") {
        BruteForceIndex bf(store);
        bf.insertAll();
        std::ofstream out(args.get("out", "bf_topk.txt").c_str());
        CandidateArray hits;
        for (int q = 0; q < n; q++) {
            bf.search(queries.row(q), queries.dim(), k, hits);
            for (int i = 0; i < hits.size(); i++)
                out << q << ' ' << hits[i].id << ' ' << hexFloat(hits[i].score) << '\n';
        }
        std::cout << "wrote brute-force top-" << k << " for " << n << " queries\n";
        return 0;
    }
    if (cmd == "persist-save") {
        HNSWIndex idx;
        idx.setup(store, cfg.hnsw);
        double seconds = IndexManager::build(idx);
        IndexManager::save(idx, cfg.modelName, args.get("index", ""));
        std::ofstream out(args.get("out", "").c_str());
        out << resultsText(idx, queries, n, k);
        std::cout << "built in " << seconds << " s, saved, recorded " << n << " query results\n";
        return 0;
    }
    if (cmd == "persist-check") {
        HNSWIndex idx;
        IndexManager::load(args.get("index", ""), store, cfg.modelName, idx);
        std::ifstream in(args.get("expect", "").c_str());
        std::stringstream want;
        want << in.rdbuf();
        std::string got = resultsText(idx, queries, n, k);
        std::istringstream a(want.str()), b(got);
        std::string la, lb;
        int lines = 0, diff = 0;
        while (std::getline(a, la)) {
            std::getline(b, lb);
            lines++;
            if (la != lb) diff++;
        }
        bool same = diff == 0 && lines == n * k;
        std::cout << "compared " << lines << " (query, rank) results: " << diff
                  << " differences -> " << (same ? "IDENTICAL" : "MISMATCH") << '\n';
        return same ? 0 : 1;
    }
    if (cmd == "load-index") {
        try {
            HNSWIndex idx;
            IndexManager::load(args.get("index", ""), store, cfg.modelName, idx);
            std::cout << "loaded OK: " << idx.size() << " nodes\n";
            return 0;
        } catch (const PersistenceError& e) {
            std::cout << "load rejected: " << e.what() << '\n';
            return 3;
        }
    }
    std::cerr << "unknown subcommand " << cmd << '\n';
    return 2;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const Error& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
