#include "data.h"

uint64_t fnv1aHash(const std::string& key) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < key.size(); i++) {
        h ^= (unsigned char)key[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

HashTable::HashTable(HashFunction function) {
    slots = nullptr;
    numSlots = 0;
    count = 0;
    hash = function;
}

HashTable::~HashTable() {
    delete[] slots;
}

int HashTable::homeSlot(const std::string& key) const {
    return (int)(hash(key) & (uint64_t)(numSlots - 1));
}

void HashTable::grow() {
    Slot* old = slots;
    int oldSize = numSlots;
    numSlots = oldSize == 0 ? 8 : 2 * oldSize;
    slots = new Slot[numSlots];
    for (int i = 0; i < oldSize; i++) {
        if (!old[i].used) continue;
        int s = homeSlot(old[i].key);
        while (slots[s].used) s = (s + 1) & (numSlots - 1);
        slots[s] = old[i];
    }
    delete[] old;
}

IntArray& HashTable::getOrInsert(const std::string& key) {
    if (10 * (count + 1) > 7 * numSlots) grow();
    int s = homeSlot(key);
    while (slots[s].used) {
        if (slots[s].key == key) return slots[s].value;
        s = (s + 1) & (numSlots - 1);
    }
    slots[s].used = true;
    slots[s].key = key;
    count++;
    return slots[s].value;
}

const IntArray* HashTable::find(const std::string& key) const {
    if (numSlots == 0) return nullptr;
    int s = homeSlot(key);
    while (slots[s].used) {
        if (slots[s].key == key) return &slots[s].value;
        s = (s + 1) & (numSlots - 1);
    }
    return nullptr;
}

int HashTable::size() const { return count; }
int HashTable::capacity() const { return numSlots; }

double HashTable::loadFactor() const {
    return numSlots == 0 ? 0.0 : (double)count / numSlots;
}

double HashTable::averageProbeLength() const {
    if (count == 0) return 0.0;
    long long total = 0;
    for (int i = 0; i < numSlots; i++)
        if (slots[i].used) total += ((i - homeSlot(slots[i].key)) & (numSlots - 1)) + 1;
    return (double)total / count;
}

static const char* STOP_WORDS[] = {"a",  "an", "and", "are", "at", "for", "in",
                                   "is", "of", "on",  "the", "to", "with"};
static const int NUM_STOP_WORDS = 13;

static bool isStopWord(const std::string& word) {
    for (int i = 0; i < NUM_STOP_WORDS; i++)
        if (word == STOP_WORDS[i]) return true;
    return false;
}

static bool isWordByte(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c >= 0x80;
}

void tokenize(const std::string& text, StringArray& words) {
    words.clear();
    std::string word;
    for (size_t i = 0; i <= text.size(); i++) {
        unsigned char c = i < text.size() ? (unsigned char)text[i] : ' ';
        if (isWordByte(c)) {
            if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
            word += (char)c;
        } else if (!word.empty()) {
            if (!isStopWord(word)) words.push(word);
            word.clear();
        }
    }
}

void intersectMerge(const IntArray& a, const IntArray& b, IntArray& out) {
    out.clear();
    int i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] < b[j]) {
            i++;
        } else if (b[j] < a[i]) {
            j++;
        } else {
            out.push(a[i]);
            i++;
            j++;
        }
    }
}

static int gallop(const IntArray& list, int from, int x) {
    int low = from, high = from, step = 1;
    while (high < list.size() && list[high] < x) {
        low = high + 1;
        high += step;
        step *= 2;
    }
    if (high > list.size()) high = list.size();
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (list[mid] < x)
            low = mid + 1;
        else
            high = mid;
    }
    return low;
}

void intersectGallop(const IntArray& small, const IntArray& large, IntArray& out) {
    out.clear();
    int pos = 0;
    for (int i = 0; i < small.size(); i++) {
        pos = gallop(large, pos, small[i]);
        if (pos == large.size()) break;
        if (large[pos] == small[i]) out.push(small[i]);
    }
}

void unionMerge(const IntArray& a, const IntArray& b, IntArray& out) {
    out.clear();
    int i = 0, j = 0;
    while (i < a.size() || j < b.size()) {
        if (j == b.size() || (i < a.size() && a[i] < b[j])) {
            out.push(a[i]);
            i++;
        } else if (i == a.size() || b[j] < a[i]) {
            out.push(b[j]);
            j++;
        } else {
            out.push(a[i]);
            i++;
            j++;
        }
    }
}

InvertedIndex::InvertedIndex() {
    totalPostings = 0;
    lastId = -1;
}

void InvertedIndex::build(const MetadataManager& meta, int skipCaption) {
    for (int i = 0; i < meta.size(); i++) {
        const ImageRecord& r = meta.at(i);
        for (int c = 0; c < r.captions.size(); c++)
            if (c != skipCaption) add(i, r.captions[c]);
    }
}

void InvertedIndex::add(int id, const std::string& text) {
    if (id < lastId)
        throw Error("inverted index: id " + std::to_string(id) + " added after id " +
                    std::to_string(lastId) + " (ids must be added in order)");
    lastId = id;
    StringArray words;
    tokenize(text, words);
    for (int w = 0; w < words.size(); w++) {
        IntArray& list = terms.getOrInsert(words[w]);
        if (list.size() == 0 || list.back() != id) {
            list.push(id);
            totalPostings++;
        }
    }
}

const IntArray* InvertedIndex::postings(const std::string& term) const {
    return terms.find(term);
}

static bool queryLists(const InvertedIndex& index, const std::string& query,
                       const IntArray** lists, int& numLists) {
    StringArray words;
    tokenize(query, words);
    bool allFound = true;
    numLists = 0;
    for (int w = 0; w < words.size(); w++) {
        bool repeated = false;
        for (int v = 0; v < w; v++)
            if (words[v] == words[w]) repeated = true;
        if (repeated) continue;
        const IntArray* list = index.postings(words[w]);
        if (list == nullptr)
            allFound = false;
        else
            lists[numLists++] = list;
    }
    return allFound;
}

static int countWords(const std::string& query) {
    StringArray words;
    tokenize(query, words);
    return words.size();
}

void InvertedIndex::matchAll(const std::string& query, IntArray& out) const {
    out.clear();
    const IntArray** lists = new const IntArray*[countWords(query) + 1];
    int n = 0;
    bool allFound = queryLists(*this, query, lists, n);
    if (allFound && n > 0) {
        for (int i = 1; i < n; i++) {
            const IntArray* item = lists[i];
            int j = i;
            while (j > 0 && lists[j - 1]->size() > item->size()) {
                lists[j] = lists[j - 1];
                j--;
            }
            lists[j] = item;
        }
        out = *lists[0];
        IntArray next;
        for (int i = 1; i < n && out.size() > 0; i++) {
            if (lists[i]->size() > 8 * out.size())
                intersectGallop(out, *lists[i], next);
            else
                intersectMerge(out, *lists[i], next);
            out = next;
        }
    }
    delete[] lists;
}

void InvertedIndex::matchAny(const std::string& query, IntArray& out) const {
    out.clear();
    const IntArray** lists = new const IntArray*[countWords(query) + 1];
    int n = 0;
    queryLists(*this, query, lists, n);
    IntArray merged;
    for (int i = 0; i < n; i++) {
        unionMerge(out, *lists[i], merged);
        out = merged;
    }
    delete[] lists;
}

int InvertedIndex::vocabularySize() const { return terms.size(); }
long long InvertedIndex::postingCount() const { return totalPostings; }
const HashTable& InvertedIndex::table() const { return terms; }
