#include "search.h"

bool closer(const Neighbor& a, const Neighbor& b) {
    return a.dist < b.dist || (a.dist == b.dist && a.id < b.id);
}

static const int INSERTION_THRESHOLD = 16;

static void swapItems(Neighbor& a, Neighbor& b) {
    Neighbor temp = a;
    a = b;
    b = temp;
}

static void insertionSort(Neighbor* a, int n) {
    for (int i = 1; i < n; i++) {
        Neighbor item = a[i];
        int j = i;
        while (j > 0 && closer(item, a[j - 1])) {
            a[j] = a[j - 1];
            j--;
        }
        a[j] = item;
    }
}

static void heapSiftDown(Neighbor* a, int n, int i) {
    Neighbor item = a[i];
    while (true) {
        int child = 2 * i + 1;
        if (child >= n) break;
        if (child + 1 < n && closer(a[child], a[child + 1])) child++;
        if (!closer(item, a[child])) break;
        a[i] = a[child];
        i = child;
    }
    a[i] = item;
}

static void heapSort(Neighbor* a, int n) {
    for (int i = n / 2 - 1; i >= 0; i--) heapSiftDown(a, n, i);
    for (int end = n - 1; end > 0; end--) {
        swapItems(a[0], a[end]);
        heapSiftDown(a, end, 0);
    }
}

static int partition(Neighbor* a, int n) {
    int mid = n / 2, last = n - 1;
    if (closer(a[mid], a[0])) swapItems(a[mid], a[0]);
    if (closer(a[last], a[mid])) {
        swapItems(a[last], a[mid]);
        if (closer(a[mid], a[0])) swapItems(a[mid], a[0]);
    }
    Neighbor pivot = a[mid];
    int i = -1, j = n;
    while (true) {
        do i++; while (closer(a[i], pivot));
        do j--; while (closer(pivot, a[j]));
        if (i >= j) return j + 1;
        swapItems(a[i], a[j]);
    }
}

static void introsortLoop(Neighbor* a, int n, int depthLeft) {
    while (n > INSERTION_THRESHOLD) {
        if (depthLeft == 0) {
            heapSort(a, n);
            return;
        }
        depthLeft--;
        int s = partition(a, n);
        if (s < n - s) {
            introsortLoop(a, s, depthLeft);
            a += s;
            n -= s;
        } else {
            introsortLoop(a + s, n - s, depthLeft);
            n = s;
        }
    }
}

void sortNeighbors(Neighbor* items, int n) {
    if (n < 2) return;
    int log2n = 0;
    for (int m = n; m > 1; m /= 2) log2n++;
    introsortLoop(items, n, 2 * log2n);
    insertionSort(items, n);
}

NeighborHeap::NeighborHeap(bool closestFirst) {
    closestOnTop = closestFirst;
}

bool NeighborHeap::ranksAbove(const Neighbor& a, const Neighbor& b) const {
    return closestOnTop ? closer(a, b) : closer(b, a);
}

void NeighborHeap::siftUp(int i) {
    Neighbor item = items[i];
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (!ranksAbove(item, items[parent])) break;
        items[i] = items[parent];
        i = parent;
    }
    items[i] = item;
}

void NeighborHeap::siftDown(int i) {
    int n = items.size();
    Neighbor item = items[i];
    while (true) {
        int child = 2 * i + 1;
        if (child >= n) break;
        if (child + 1 < n && ranksAbove(items[child + 1], items[child])) child++;
        if (!ranksAbove(items[child], item)) break;
        items[i] = items[child];
        i = child;
    }
    items[i] = item;
}

void NeighborHeap::push(const Neighbor& item) {
    items.push(item);
    siftUp(items.size() - 1);
}

void NeighborHeap::pop() {
    items[0] = items[items.size() - 1];
    items.pop();
    if (items.size() > 0) siftDown(0);
}

const Neighbor& NeighborHeap::top() const { return items[0]; }
int NeighborHeap::size() const { return items.size(); }
bool NeighborHeap::empty() const { return items.size() == 0; }
void NeighborHeap::reserve(int n) { items.reserve(n); }

TopK::TopK(int count) : heap(false) {
    k = count;
    heap.reserve(k + 1);
}

void TopK::push(const Neighbor& item) {
    if (k <= 0) return;
    if (heap.size() < k) {
        heap.push(item);
    } else if (closer(item, heap.top())) {
        heap.pop();
        heap.push(item);
    }
}

int TopK::size() const { return heap.size(); }

void TopK::extractSorted(NeighborArray& out) {
    int n = heap.size();
    out.clear();
    out.reserve(n);
    for (int i = 0; i < n; i++) out.push(Neighbor());
    for (int i = n - 1; i >= 0; i--) {
        out[i] = heap.top();
        heap.pop();
    }
}

BruteForceIndex::BruteForceIndex(const EmbeddingStore& s) : present(s.size(), false) {
    store = &s;
}

void BruteForceIndex::insert(int id, const float* embedding) {
    checkInsert(*store, id, embedding, present);
    ids.push(id);
    present.set(id, true);
}

void BruteForceIndex::insertAll() {
    ids.reserve(store->size());
    for (int i = 0; i < store->size(); i++) insert(i, store->row(i));
}

void BruteForceIndex::search(const float* query, int queryDim, int k, CandidateArray& out) const {
    checkSearchArgs(query, queryDim, k, store->dim(), ids.size());
    TopK top(k);
    for (int i = 0; i < ids.size(); i++) {
        Neighbor n;
        n.dist = distance(query, store->row(ids[i]), queryDim);
        n.id = ids[i];
        top.push(n);
    }
    NeighborArray best;
    top.extractSorted(best);
    toCandidates(best, out);
}

int BruteForceIndex::size() const {
    return ids.size();
}

long long BruteForceIndex::memoryBytes() const {
    return (long long)ids.capacity() * 4 + present.memoryBytes();
}
