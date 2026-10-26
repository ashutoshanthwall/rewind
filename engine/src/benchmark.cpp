// Latency statistics, recall metrics and machine description for benchmarks (hnsw.h).
#include "hnsw.h"

#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <sched.h>
#include <sstream>
#include <sys/utsname.h>
#include <unistd.h>

#ifndef SIR_CXX_FLAGS
#define SIR_CXX_FLAGS "unknown"
#endif
#ifndef SIR_BUILD_TYPE
#define SIR_BUILD_TYPE "unknown"
#endif

// ---------------------------------------------------------------- latency and recall

LatencyStats summarize(double* values, int n) {
    LatencyStats s;
    s.samples = n;
    if (n == 0) return s;
    for (int i = 1; i < n; i++) {  // insertion sort (n is at most a few thousand)
        double item = values[i];
        int j = i;
        while (j > 0 && values[j - 1] > item) {
            values[j] = values[j - 1];
            j--;
        }
        values[j] = item;
    }
    s.medianMs = n % 2 == 1 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
    int rank = (int)std::ceil(0.95 * n);  // nearest-rank 95th percentile
    s.p95Ms = values[(rank < 1 ? 1 : rank) - 1];
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += values[i];
    s.meanMs = sum / n;
    s.minMs = values[0];
    s.maxMs = values[n - 1];
    return s;
}

void timeQueries(const RetrievalIndex& index, const EmbeddingStore& queries, int k, int warmup,
                 int repetitions, double* perQueryMs) {
    int nq = queries.size();
    int dim = queries.dim();
    CandidateArray results;
    volatile long long sink = 0;  // `volatile`: the compiler may not skip the searches
    for (int w = 0; w < warmup; w++) {
        index.search(queries.row(w % nq), dim, k, results);
        sink = sink + results.size();
    }
    double* times = new double[(long long)nq * repetitions];
    for (int r = 0; r < repetitions; r++) {
        for (int q = 0; q < nq; q++) {
            double start = nowMs();
            index.search(queries.row(q), dim, k, results);
            double end = nowMs();
            sink = sink + results[0].id;
            times[(long long)q * repetitions + r] = end - start;
        }
    }
    for (int q = 0; q < nq; q++)
        perQueryMs[q] = summarize(times + (long long)q * repetitions, repetitions).medianMs;
    delete[] times;
}

double recallAtK(const CandidateArray& approx, const CandidateArray& exact, int k) {
    IntArray truth;
    for (int i = 0; i < k && i < exact.size(); i++) truth.push(exact[i].id);
    int hits = 0;
    for (int i = 0; i < k && i < approx.size(); i++)
        if (truth.contains(approx[i].id)) hits++;
    return (double)hits / k;
}

bool hitAtK(const CandidateArray& results, int target, int k) {
    for (int i = 0; i < k && i < results.size(); i++)
        if (results[i].id == target) return true;
    return false;
}

// ---------------------------------------------------------------- system information

static std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\n\"");
    size_t e = s.find_last_not_of(" \t\n\"");
    return b == std::string::npos ? "" : s.substr(b, e - b + 1);
}

static bool startsWith(const std::string& s, const std::string& prefix) {
    return s.compare(0, prefix.size(), prefix) == 0;
}

static std::string readFirstLine(const std::string& path) {
    std::ifstream in(path.c_str());
    std::string line;
    std::getline(in, line);
    return trim(line);
}

static std::string runCommand(const std::string& cmd) {
    FILE* pipe = popen(cmd.c_str(), "r");
    if (pipe == nullptr) return "unknown";
    std::string out;
    char buf[256];
    while (fgets(buf, sizeof buf, pipe) != nullptr) out += buf;
    pclose(pipe);
    return trim(out);
}

static long long statusField(const std::string& key) {  // kB value from /proc/self/status
    std::ifstream in("/proc/self/status");
    std::string line;
    while (std::getline(in, line)) {
        if (startsWith(line, key + ":")) {
            std::istringstream is(line.substr(key.size() + 1));
            long long kb = 0;
            is >> kb;
            return kb * 1024;
        }
    }
    return 0;
}

long long peakRssBytes() { return statusField("VmHWM"); }
long long currentRssBytes() { return statusField("VmRSS"); }

SystemInfo collectSystemInfo(const std::string& repoRoot) {
    SystemInfo s;
    std::ifstream cpu("/proc/cpuinfo");
    std::string line, physId;
    StringArray cores;  // distinct "physical id:core id" pairs
    while (std::getline(cpu, line)) {
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = trim(line.substr(0, colon)), val = trim(line.substr(colon + 1));
        if (key == "model name" && s.cpuModel.empty()) s.cpuModel = val;
        if (key == "physical id") physId = val;
        if (key == "core id" && !cores.contains(physId + ":" + val)) cores.push(physId + ":" + val);
    }
    s.logicalCpus = (int)sysconf(_SC_NPROCESSORS_ONLN);
    s.physicalCores = cores.size();

    std::ifstream mem("/proc/meminfo");
    while (std::getline(mem, line)) {
        if (startsWith(line, "MemTotal:")) {
            std::istringstream is(line.substr(9));
            long long kb = 0;
            is >> kb;
            s.ramBytes = kb * 1024;
        }
    }
    std::ifstream osr("/etc/os-release");
    while (std::getline(osr, line))
        if (startsWith(line, "PRETTY_NAME=")) s.os = trim(line.substr(12));
    struct utsname u;
    if (uname(&u) == 0) s.kernel = std::string(u.sysname) + " " + u.release + " " + u.machine;
#if defined(__clang__)
    s.compiler = std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
    s.compiler = std::string("GCC ") + __VERSION__;
#endif
    s.buildType = SIR_BUILD_TYPE;
    s.cxxFlags = trim(SIR_CXX_FLAGS);
    std::string git = "git -C '" + repoRoot + "' ";
    s.gitCommit = runCommand(git + "rev-parse HEAD 2>/dev/null");
    if (!runCommand(git + "status --porcelain --untracked-files=no 2>/dev/null").empty())
        s.gitCommit += "-dirty";
    s.cpuGovernor = readFirstLine("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
        std::string aff;
        for (int c = 0; c < CPU_SETSIZE; c++)
            if (CPU_ISSET(c, &set)) aff += (aff.empty() ? "" : ",") + std::to_string(c);
        s.cpuAffinity = aff;
    }
    s.loadAverage = readFirstLine("/proc/loadavg");
    time_t now = time(nullptr);
    char ts[32];
    strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
    s.timestampUtc = ts;
    return s;
}
