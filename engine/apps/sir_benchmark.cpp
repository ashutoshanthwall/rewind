// Exact vs approximate retrieval benchmark (single thread, precomputed query embeddings).
//
//   sir_benchmark [--config f] [--out-dir d] [--measure-encoder] [--pin-cpu N]
//
// --pin-cpu pins this process to one CPU (sched_setaffinity) so every index runs on the
// same core type; on hybrid CPUs, P- and E-cores differ greatly in speed. The optional
// text-encoder helper is started before pinning so it does not inherit the affinity.
//
// Measures brute force, and HNSW at every efSearch in the sweep with three neighbour-
// selection variants: latency (median / p95 / mean of per-query medians), speedup,
// Recall@5/@10 against brute force, distance evaluations per query, build time, memory
// (VmHWM and memoryBytes()), and the held-out caption quality track R@1/5/10.
// Writes CSV + JSON + a markdown summary. Every number is measured here.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sched.h>
#include <sstream>

#include "storage.h"

struct Variant {
    std::string name;
    HnswParams params;
};

struct Row {
    std::string backend, variant;
    int efSearch = 0;
    LatencyStats lat;
    double speedup = 1.0, recall5 = 1.0, recall10 = 1.0, perfect10 = 1.0;
    double distEvals = 0.0, buildSeconds = 0.0;
    long long structBytes = 0, peakRss = 0;
    int reachable = 0;
    double r1 = 0, r5 = 0, r10 = 0;  // quality track
};

// Column 4 (internal_id) of evaluation/query_ground_truth.csv; -1 for hand-written queries.
static IntArray loadGroundTruth(const std::string& csv) {
    std::ifstream in(csv.c_str());
    if (!in) throw Error("cannot open " + csv);
    std::string line;
    std::getline(in, line);  // header
    IntArray gt;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string idx, kind, imageId, internalId;
        std::getline(ss, idx, ',');
        std::getline(ss, kind, ',');
        std::getline(ss, imageId, ',');
        std::getline(ss, internalId, ',');
        gt.push(std::stoi(internalId));
    }
    return gt;
}

static StringArray loadQueryTexts(const std::string& txt) {
    std::ifstream in(txt.c_str());
    StringArray q;
    std::string line;
    while (std::getline(in, line))
        if (!line.empty()) q.push(line);
    return q;
}

static void qualityInto(Row& row, const CandidateArray* results, int nq, const IntArray& gt) {
    int n = 0, h1 = 0, h5 = 0, h10 = 0;
    for (int q = 0; q < nq; q++) {
        if (gt[q] < 0) continue;  // hand-written queries have no ground truth
        n++;
        if (hitAtK(results[q], gt[q], 1)) h1++;
        if (hitAtK(results[q], gt[q], 5)) h5++;
        if (hitAtK(results[q], gt[q], 10)) h10++;
    }
    row.r1 = (double)h1 / n;
    row.r5 = (double)h5 / n;
    row.r10 = (double)h10 / n;
}

static std::string fmt(double v, int digits) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(digits) << v;
    return os.str();
}

static std::string mib(long long bytes) {
    return fmt((double)bytes / (1024.0 * 1024.0), 2);
}

static int run(int argc, char** argv) {
    StringArray flags;
    flags.push("measure-encoder");
    Args args = parseArgs(argc, argv, flags);
    AppConfig cfg = loadAppConfig(args);
    std::string outDir = args.get("out-dir", cfg.benchmarkResults);
    makeDirs(outDir);
    TextEncoderProcess* encoder = nullptr;
    if (args.has("measure-encoder"))
        encoder = new TextEncoderProcess(cfg.python, cfg.textEncoderServer, cfg.configFile, cfg.dim);
    if (args.has("pin-cpu")) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(args.getInt("pin-cpu", 0), &set);
        if (sched_setaffinity(0, sizeof set, &set) != 0) throw Error("sched_setaffinity failed for --pin-cpu");
    }
    SystemInfo sys = collectSystemInfo(cfg.root);
    const int K = 10;
    int warm = cfg.evaluation.warmupQueries, reps = cfg.evaluation.repetitions;
    const IntArray& sweep = cfg.evaluation.efSearchSweep;

    std::cout << "CPU: " << sys.cpuModel << " | logical CPUs " << sys.logicalCpus
              << " | physical cores " << sys.physicalCores << " | RAM " << mib(sys.ramBytes)
              << " MiB\nOS: " << sys.os << " (" << sys.kernel << ")\ncompiler: " << sys.compiler
              << " | " << sys.buildType << " " << sys.cxxFlags << "\ngit: " << sys.gitCommit
              << " | governor " << sys.cpuGovernor << " | affinity CPUs " << sys.cpuAffinity
              << " | loadavg " << sys.loadAverage << "\n";

    EmbeddingStore store, queries;
    loadImageStore(cfg, store);
    loadQueryStore(cfg, queries);
    int nq = queries.size();
    IntArray gt = loadGroundTruth(cfg.queryGroundTruth);
    if (gt.size() != nq)
        throw Error("query ground truth has " + std::to_string(gt.size()) + " rows but there are " +
                    std::to_string(nq) + " queries");
    long long rssLoaded = peakRssBytes();
    std::cout << "images " << store.size() << " x " << store.dim() << ", queries " << nq
              << ", warm-up " << warm << ", repetitions " << reps << ", K " << K << "\n";

    Row* rows = new Row[1 + 3 * sweep.size()];
    int numRows = 0;
    double* perQuery = new double[nq];

    // ---------------------------------------------------------------- brute force
    BruteForceIndex bf(store);
    double buildStart = nowMs();
    bf.insertAll();
    double bfBuild = (nowMs() - buildStart) / 1000.0;
    CandidateArray* exact = new CandidateArray[nq];
    for (int q = 0; q < nq; q++) bf.search(queries.row(q), queries.dim(), K, exact[q]);
    Row& bfRow = rows[numRows++];
    bfRow.backend = "brute_force";
    bfRow.variant = "exact";
    timeQueries(bf, queries, K, warm, reps, perQuery);
    bfRow.lat = summarize(perQuery, nq);
    bfRow.distEvals = store.size();
    bfRow.buildSeconds = bfBuild;
    bfRow.structBytes = bf.memoryBytes();
    bfRow.peakRss = peakRssBytes();
    bfRow.reachable = store.size();
    qualityInto(bfRow, exact, nq, gt);
    std::cout << "brute force: median " << fmt(bfRow.lat.medianMs, 3) << " ms, p95 "
              << fmt(bfRow.lat.p95Ms, 3) << " ms, mean " << fmt(bfRow.lat.meanMs, 3) << " ms | R@1 "
              << fmt(bfRow.r1, 4) << "\n";

    // ---------------------------------------------------------------- HNSW
    Variant variants[3];
    variants[0].name = "heuristic+keepPruned";
    variants[0].params = cfg.hnsw;
    variants[0].params.selectHeuristic = true;
    variants[0].params.keepPrunedConnections = true;
    variants[1].name = "heuristic";
    variants[1].params = variants[0].params;
    variants[1].params.keepPrunedConnections = false;
    variants[2].name = "naive";
    variants[2].params = variants[0].params;
    variants[2].params.selectHeuristic = false;
    variants[2].params.keepPrunedConnections = false;

    CandidateArray* approx = new CandidateArray[nq];
    for (int v = 0; v < 3; v++) {
        HNSWIndex h;
        h.setup(store, variants[v].params);
        double buildSeconds = IndexManager::build(h);
        IndexValidation val = IndexManager::validate(h);
        long long rssBuilt = peakRssBytes();
        std::cout << "HNSW " << variants[v].name << ": built in " << fmt(buildSeconds, 2)
                  << " s, reachable " << val.reachableLayer0 << "/" << val.nodes << ", mean L0 degree "
                  << fmt(val.meanDegreeLayer0, 2) << "\n";
        for (int s = 0; s < sweep.size(); s++) {
            int ef = sweep[s];
            h.setEfSearch(ef);
            Row& r = rows[numRows++];
            r.backend = "hnsw";
            r.variant = variants[v].name;
            r.efSearch = ef;
            h.resetDistanceComputations();
            double s5 = 0, s10 = 0, perfect = 0;
            for (int q = 0; q < nq; q++) {
                h.search(queries.row(q), queries.dim(), K, approx[q]);
                s5 += recallAtK(approx[q], exact[q], 5);
                double r10 = recallAtK(approx[q], exact[q], 10);
                s10 += r10;
                if (r10 == 1.0) perfect++;
            }
            r.distEvals = (double)h.distanceComputations() / nq;
            r.recall5 = s5 / nq;
            r.recall10 = s10 / nq;
            r.perfect10 = perfect / nq;
            qualityInto(r, approx, nq, gt);
            timeQueries(h, queries, K, warm, reps, perQuery);
            r.lat = summarize(perQuery, nq);
            r.speedup = bfRow.lat.medianMs / r.lat.medianMs;
            r.buildSeconds = buildSeconds;
            r.structBytes = h.memoryBytes();
            r.peakRss = rssBuilt;
            r.reachable = val.reachableLayer0;
            std::cout << "  ef=" << std::setw(3) << ef << "  median " << fmt(r.lat.medianMs, 4)
                      << " ms  p95 " << fmt(r.lat.p95Ms, 4) << "  speedup " << fmt(r.speedup, 1)
                      << "x  R@5 " << fmt(r.recall5, 4) << "  R@10 " << fmt(r.recall10, 4)
                      << "  dist/q " << fmt(r.distEvals, 0) << "  quality R@1 " << fmt(r.r1, 4) << "\n";
        }
    }

    // ---------------------------------------------------------------- text encoder latency
    LatencyStats enc;
    if (encoder != nullptr) {
        StringArray texts = loadQueryTexts(cfg.queries);
        FloatArray v;
        for (int w = 0; w < 5; w++) encoder->encode(texts[w], v);
        int n = texts.size() < 100 ? texts.size() : 100;
        double* ms = new double[n];
        for (int i = 0; i < n; i++) {
            double start = nowMs();
            encoder->encode(texts[i], v);
            ms[i] = nowMs() - start;
        }
        enc = summarize(ms, n);
        delete[] ms;
        std::cout << "text encoding (Python helper, CPU, incl. IPC; " << enc.samples
                  << " queries): median " << fmt(enc.medianMs, 2) << " ms, p95 " << fmt(enc.p95Ms, 2)
                  << " ms\n";
    }

    // ---------------------------------------------------------------- outputs
    {
        std::ofstream c(joinPath(outDir, "latency_recall.csv").c_str());
        c << "backend,variant,M,ef_construction,ef_search,median_ms,p95_ms,mean_ms,min_ms,"
             "max_ms,speedup_vs_bf_median,recall_at_5,recall_at_10,frac_queries_recall10_eq_1,"
             "dist_evals_per_query,build_seconds,memory_bytes_structure,peak_rss_bytes,"
             "reachable_layer0\n";
        for (int i = 0; i < numRows; i++) {
            const Row& r = rows[i];
            bool isHnsw = r.backend == "hnsw";
            c << r.backend << ',' << r.variant << ',' << (isHnsw ? cfg.hnsw.M : 0) << ','
              << (isHnsw ? cfg.hnsw.efConstruction : 0) << ',' << r.efSearch << ','
              << fmt(r.lat.medianMs, 5) << ',' << fmt(r.lat.p95Ms, 5) << ',' << fmt(r.lat.meanMs, 5)
              << ',' << fmt(r.lat.minMs, 5) << ',' << fmt(r.lat.maxMs, 5) << ',' << fmt(r.speedup, 3)
              << ',' << fmt(r.recall5, 5) << ',' << fmt(r.recall10, 5) << ',' << fmt(r.perfect10, 5)
              << ',' << fmt(r.distEvals, 1) << ',' << fmt(r.buildSeconds, 3) << ',' << r.structBytes
              << ',' << r.peakRss << ',' << r.reachable << '\n';
        }
        std::ofstream qcsv(joinPath(outDir, "quality.csv").c_str());
        qcsv << "backend,variant,ef_search,n_queries,R_at_1,R_at_5,R_at_10\n";
        int nHeld = 0;
        for (int i = 0; i < gt.size(); i++)
            if (gt[i] >= 0) nHeld++;
        for (int i = 0; i < numRows; i++) {
            const Row& r = rows[i];
            qcsv << r.backend << ',' << r.variant << ',' << r.efSearch << ',' << nHeld << ','
                 << fmt(r.r1, 5) << ',' << fmt(r.r5, 5) << ',' << fmt(r.r10, 5) << '\n';
        }
    }
    {
        std::ofstream j(joinPath(outDir, "system.json").c_str());
        j << std::setprecision(10) << "{\n"
          << "  \"cpu_model\": " << jsonQuote(sys.cpuModel) << ",\n"
          << "  \"logical_cpus\": " << sys.logicalCpus << ",\n"
          << "  \"physical_cores\": " << sys.physicalCores << ",\n"
          << "  \"ram_bytes\": " << sys.ramBytes << ",\n"
          << "  \"os\": " << jsonQuote(sys.os) << ",\n"
          << "  \"kernel\": " << jsonQuote(sys.kernel) << ",\n"
          << "  \"compiler\": " << jsonQuote(sys.compiler) << ",\n"
          << "  \"build_type\": " << jsonQuote(sys.buildType) << ",\n"
          << "  \"cxx_flags\": " << jsonQuote(sys.cxxFlags) << ",\n"
          << "  \"git_commit\": " << jsonQuote(sys.gitCommit) << ",\n"
          << "  \"cpu_governor\": " << jsonQuote(sys.cpuGovernor) << ",\n"
          << "  \"cpu_affinity\": " << jsonQuote(sys.cpuAffinity) << ",\n"
          << "  \"loadavg_at_start\": " << jsonQuote(sys.loadAverage) << ",\n"
          << "  \"timestamp_utc\": " << jsonQuote(sys.timestampUtc) << ",\n"
          << "  \"hnsw_seed\": " << cfg.hnsw.seed << ",\n"
          << "  \"M\": " << cfg.hnsw.M << ",\n"
          << "  \"ef_construction\": " << cfg.hnsw.efConstruction << ",\n"
          << "  \"queries\": " << nq << ",\n"
          << "  \"warmup_queries\": " << warm << ",\n"
          << "  \"repetitions\": " << reps << ",\n"
          << "  \"k\": " << K << ",\n"
          << "  \"images\": " << store.size() << ",\n"
          << "  \"dim\": " << store.dim() << ",\n"
          << "  \"embedding_store_bytes\": " << store.memoryBytes() << ",\n"
          << "  \"peak_rss_after_loading_bytes\": " << rssLoaded << ",\n"
          << "  \"final_peak_rss_bytes\": " << peakRssBytes();
        if (enc.samples > 0)
            j << ",\n  \"text_encoder_ms\": {\"median\": " << enc.medianMs << ", \"p95\": " << enc.p95Ms
              << ", \"mean\": " << enc.meanMs << ", \"samples\": " << enc.samples << "}";
        j << "\n}\n";
    }

    std::ofstream md(joinPath(outDir, "summary.md").c_str());
    md << "# Benchmark results\n\nGenerated by `sir_benchmark` at " << sys.timestampUtc << " (git "
       << sys.gitCommit << "). All numbers measured in this run.\n\n"
       << "| Environment | |\n|---|---|\n| CPU | " << sys.cpuModel << " (" << sys.logicalCpus
       << " logical CPUs, " << sys.physicalCores << " physical cores) |\n| RAM | " << mib(sys.ramBytes)
       << " MiB |\n| OS | " << sys.os << ", " << sys.kernel << " |\n| Compiler | " << sys.compiler
       << " |\n| Flags | " << sys.buildType << " `" << sys.cxxFlags << "` |\n| CPU governor / affinity | "
       << sys.cpuGovernor << " / CPUs " << sys.cpuAffinity << " |\n| Load average at start | "
       << sys.loadAverage << " |\n\n"
       << "Protocol: " << store.size() << " images x " << store.dim() << " dims; " << nq
       << " queries (5 hand-written + held-out captions); " << warm << " warm-up queries discarded; "
       << reps << " timed repetitions per query (repetition-outer); per-query latency = median of "
          "its repetitions; statistics across queries. Single thread, scalar distance "
          "kernel shared by both backends. K = " << K << ". HNSW M = " << cfg.hnsw.M
       << ", efConstruction = " << cfg.hnsw.efConstruction << ", seed = " << cfg.hnsw.seed
       << ", last-in-edge protection on.\n\n"
       << "## Latency and recall vs brute force\n\n"
       << "| Backend | Selection | efSearch | Median ms | p95 ms | Mean ms | Speedup | "
          "Recall@5 | Recall@10 | Queries w/ R@10=1 | Dist evals/query | Build s |\n"
       << "|---|---|---|---|---|---|---|---|---|---|---|---|\n";
    for (int i = 0; i < numRows; i++) {
        const Row& r = rows[i];
        md << "| " << r.backend << " | " << r.variant << " | "
           << (r.efSearch ? std::to_string(r.efSearch) : "-") << " | " << fmt(r.lat.medianMs, 4)
           << " | " << fmt(r.lat.p95Ms, 4) << " | " << fmt(r.lat.meanMs, 4) << " | "
           << fmt(r.speedup, 1) << "x | " << fmt(r.recall5, 4) << " | " << fmt(r.recall10, 4)
           << " | " << fmt(100.0 * r.perfect10, 1) << "% | " << fmt(r.distEvals, 0) << " | "
           << fmt(r.buildSeconds, 2) << " |\n";
    }
    md << "\n## Neighbour selection at efSearch = 80\n\n| Selection | Recall@10 | Median ms | "
          "Dist evals/query | Build s | Graph bytes |\n|---|---|---|---|---|---|\n";
    for (int i = 0; i < numRows; i++) {
        const Row& r = rows[i];
        if (r.efSearch == 80)
            md << "| " << r.variant << " | " << fmt(r.recall10, 4) << " | " << fmt(r.lat.medianMs, 4)
               << " | " << fmt(r.distEvals, 0) << " | " << fmt(r.buildSeconds, 2) << " | "
               << r.structBytes << " |\n";
    }
    md << "\n## Memory\n\n| Structure | memoryBytes() | MiB |\n|---|---|---|\n"
       << "| EmbeddingStore (shared by both backends) | " << store.memoryBytes() << " | "
       << mib(store.memoryBytes()) << " |\n| BruteForceIndex (id list) | " << bfRow.structBytes
       << " | " << mib(bfRow.structBytes) << " |\n";
    for (int i = 0; i < numRows; i++) {
        const Row& r = rows[i];
        if (r.backend == "hnsw" && r.efSearch == sweep[0])
            md << "| HNSWIndex graph (" << r.variant << ") | " << r.structBytes << " | "
               << mib(r.structBytes) << " |\n";
    }
    md << "\nPeak RSS (VmHWM, whole process, monotone): after loading embeddings + queries "
       << mib(rssLoaded) << " MiB; after brute-force timing " << mib(bfRow.peakRss) << " MiB; at end "
       << mib(peakRssBytes())
       << " MiB. VmHWM includes the C++ runtime, both embedding files, ground-truth result "
          "lists and allocator slack, so it is larger than the sum of memoryBytes().\n"
       << "\n## Absolute retrieval quality (held-out caption #1 -> its image)\n\n"
       << "| Backend | Selection | efSearch | R@1 | R@5 | R@10 |\n|---|---|---|---|---|---|\n";
    for (int i = 0; i < numRows; i++) {
        const Row& r = rows[i];
        md << "| " << r.backend << " | " << r.variant << " | "
           << (r.efSearch ? std::to_string(r.efSearch) : "-") << " | " << fmt(r.r1, 4) << " | "
           << fmt(r.r5, 4) << " | " << fmt(r.r10, 4) << " |\n";
    }
    if (enc.samples > 0)
        md << "\n## Query embedding latency (excluded from retrieval latency)\n\nPython CLIP "
              "text-encoder helper on CPU, including socket IPC, "
           << enc.samples << " queries: median " << fmt(enc.medianMs, 2) << " ms, p95 "
           << fmt(enc.p95Ms, 2) << " ms, mean " << fmt(enc.meanMs, 2) << " ms.\n";
    std::cout << "wrote " << joinPath(outDir, "summary.md") << "\n";

    delete[] approx;
    delete[] exact;
    delete[] perQuery;
    delete[] rows;
    delete encoder;
    return 0;
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
