// Builds the HNSW index over the full embedding store, validates the graph, saves it, and
// reports Recall@10 against brute force for each efSearch in the sweep (gate G9).
//
//   sir_build_index [--config f] [--out path] [--M 16] [--ef-construction 200]
//                   [--select-heuristic heuristic|naive] [--no-recall]
#include <iomanip>
#include <iostream>

#include "storage.h"

static int run(int argc, char** argv) {
    StringArray flags;
    flags.push("no-recall");
    Args args = parseArgs(argc, argv, flags);
    AppConfig cfg = loadAppConfig(args);
    HnswParams p = cfg.hnsw;
    p.M = args.getInt("M", p.M);
    p.efConstruction = args.getInt("ef-construction", p.efConstruction);
    if (args.has("select-heuristic")) {
        std::string mode = args.get("select-heuristic", "");
        if (mode != "heuristic" && mode != "naive")
            throw Error("--select-heuristic must be 'heuristic' or 'naive'");
        p.selectHeuristic = mode == "heuristic";
    }
    if (args.has("keep-pruned")) p.keepPrunedConnections = args.get("keep-pruned", "1") == "1";
    if (args.has("protect-in-edge")) p.protectLastInEdge = args.get("protect-in-edge", "1") == "1";
    std::string out = args.get("out", cfg.index);

    EmbeddingStore store;
    loadImageStore(cfg, store);
    std::cout << "embeddings: " << store.size() << " x " << store.dim() << " (sha256 "
              << store.sourceSha256().substr(0, 16) << "...)\n"
              << "HNSW: M=" << p.M << " efConstruction=" << p.efConstruction << " seed=" << p.seed
              << " selection=" << (p.selectHeuristic ? "heuristic" : "naive")
              << " keepPruned=" << p.keepPrunedConnections
              << " protectLastInEdge=" << p.protectLastInEdge << '\n';
    HNSWIndex idx;
    idx.setup(store, p);
    double seconds = IndexManager::rebuild(idx, cfg.modelName, out);
    IndexValidation v = IndexManager::validate(idx);
    std::cout << std::fixed << std::setprecision(2) << "built in " << seconds << " s; saved to "
              << out << " (" << fileSize(out) / 1024 << " KiB)\n"
              << "graph: nodes=" << v.nodes << " maxLevel=" << v.maxLevel
              << " meanDegree(L0)=" << v.meanDegreeLayer0 << " reachable(L0)=" << v.reachableLayer0
              << "/" << v.nodes << '\n'
              << "nodes per level:";
    for (int l = 0; l < v.nodesPerLevel.size(); l++) std::cout << " L" << l << "=" << v.nodesPerLevel[l];
    std::cout << "\nlayer-0 nodes with in-degree 0 (excluding entry): " << v.inDegreeZeroLayer0;
    if (v.unreachableLayer0.size() > 0) {
        // Explain each unreachable node: its exact nearest neighbours.
        BruteForceIndex probe(store);
        probe.insertAll();
        CandidateArray near;
        for (int i = 0; i < v.unreachableLayer0.size(); i++) {
            int u = v.unreachableLayer0[i];
            std::cout << "\n  unreachable id=" << u << " level=" << idx.levelOf(u)
                      << " outDegree=" << idx.neighbors(u, 0).size() << " exact NN:";
            probe.search(store.row(u), store.dim(), 3, near);
            for (int j = 0; j < near.size(); j++)
                if (near[j].id != u)
                    std::cout << ' ' << near[j].id << "(sim " << std::setprecision(5) << near[j].score
                              << std::setprecision(2) << ")";
        }
    }
    std::cout << "\nmemoryBytes(): HNSW graph " << idx.memoryBytes() << ", embeddings "
              << store.memoryBytes() << '\n';
    if (args.has("no-recall")) return 0;

    EmbeddingStore queries;
    loadQueryStore(cfg, queries);
    BruteForceIndex bf(store);
    bf.insertAll();
    IntArray* truth = new IntArray[queries.size()];  // exact top-10 ids of each query
    CandidateArray hits;
    for (int q = 0; q < queries.size(); q++) {
        bf.search(queries.row(q), queries.dim(), 10, hits);
        for (int i = 0; i < hits.size(); i++) truth[q].push(hits[i].id);
    }
    std::cout << "Recall@10 vs brute force over " << queries.size() << " queries:\n";
    for (int s = 0; s < cfg.evaluation.efSearchSweep.size(); s++) {
        int ef = cfg.evaluation.efSearchSweep[s];
        idx.setEfSearch(ef);
        long long found = 0;
        idx.resetDistanceComputations();
        for (int q = 0; q < queries.size(); q++) {
            idx.search(queries.row(q), queries.dim(), 10, hits);
            for (int i = 0; i < hits.size(); i++)
                if (truth[q].contains(hits[i].id)) found++;
        }
        std::cout << "  efSearch=" << std::setw(4) << ef << "  Recall@10=" << std::setprecision(4)
                  << (double)found / (10.0 * queries.size())
                  << "  mean distance evaluations/query=" << std::setprecision(0)
                  << (double)idx.distanceComputations() / queries.size() << std::setprecision(2)
                  << '\n';
    }
    delete[] truth;
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
