// A small JSON reader (declared in data.h), written by recursive descent: one function per
// kind of value, and parseValue() calls itself for the values inside objects and arrays.
//
// Every node is linked into the tree *before* its contents are parsed, so when the input
// turns out to be malformed, freeing the root frees everything allocated so far.
#include "data.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

static const int MAX_DEPTH = 100;  // deeper nesting is rejected instead of overflowing the stack

static JsonNode* newNode() {
    JsonNode* node = new JsonNode;
    node->type = JsonNode::NULL_VALUE;
    node->number = 0.0;
    node->flag = false;
    node->child = nullptr;
    node->next = nullptr;
    return node;
}

// Frees a node, its siblings after it, and all their children. Siblings are walked with a
// loop (an array can have 30,000 of them); only the nesting depth uses recursion.
static void freeNodes(JsonNode* node) {
    while (node != nullptr) {
        JsonNode* following = node->next;
        freeNodes(node->child);
        delete node;
        node = following;
    }
}

class JsonParser {
public:
    JsonParser(const std::string& input, const std::string& fileName) : text(input), file(fileName) {
        pos = 0;
    }

    void parseDocument(JsonNode* root) {
        parseValue(root, 0);
        skipSpaces();
        if (pos != text.size()) fail("unexpected text after the end of the document");
    }

private:
    const std::string& text;
    std::string file;
    size_t pos;

    void fail(const std::string& what) {
        throw Error("malformed JSON in " + file + " at byte " + std::to_string(pos) + ": " + what);
    }

    void skipSpaces() {
        while (pos < text.size() &&
               (text[pos] == ' ' || text[pos] == '\n' || text[pos] == '\r' || text[pos] == '\t'))
            pos++;
    }

    char peek() {
        skipSpaces();
        if (pos >= text.size()) fail("unexpected end of file");
        return text[pos];
    }

    void expect(char c) {
        if (peek() != c) fail(std::string("expected '") + c + "'");
        pos++;
    }

    void expectWord(const std::string& word) {
        if (text.compare(pos, word.size(), word) != 0) fail("expected '" + word + "'");
        pos += word.size();
    }

    void parseValue(JsonNode* node, int depth) {
        if (depth > MAX_DEPTH) fail("nested too deeply");
        char c = peek();
        if (c == '{') {
            parseObject(node, depth);
        } else if (c == '[') {
            parseArray(node, depth);
        } else if (c == '"') {
            node->type = JsonNode::STRING;
            node->text = parseString();
        } else if (c == 't') {
            expectWord("true");
            node->type = JsonNode::BOOLEAN;
            node->flag = true;
        } else if (c == 'f') {
            expectWord("false");
            node->type = JsonNode::BOOLEAN;
            node->flag = false;
        } else if (c == 'n') {
            expectWord("null");
            node->type = JsonNode::NULL_VALUE;
        } else {
            node->type = JsonNode::NUMBER;
            node->number = parseNumber();
        }
    }

    void parseObject(JsonNode* node, int depth) {
        node->type = JsonNode::OBJECT;
        expect('{');
        if (peek() == '}') {
            pos++;
            return;
        }
        JsonNode* last = nullptr;
        while (true) {
            if (peek() != '"') fail("expected a member name");
            JsonNode* member = newNode();
            if (last == nullptr)
                node->child = member;
            else
                last->next = member;
            last = member;
            member->key = parseString();
            expect(':');
            parseValue(member, depth + 1);
            char c = peek();
            pos++;
            if (c == '}') return;
            if (c != ',') fail("expected ',' or '}'");
        }
    }

    void parseArray(JsonNode* node, int depth) {
        node->type = JsonNode::ARRAY;
        expect('[');
        if (peek() == ']') {
            pos++;
            return;
        }
        JsonNode* last = nullptr;
        while (true) {
            JsonNode* item = newNode();
            if (last == nullptr)
                node->child = item;
            else
                last->next = item;
            last = item;
            parseValue(item, depth + 1);
            char c = peek();
            pos++;
            if (c == ']') return;
            if (c != ',') fail("expected ',' or ']'");
        }
    }

    int hexDigit(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        fail("bad hex digit in \\u escape");
        return 0;
    }

    unsigned int parseHex4() {
        if (pos + 4 > text.size()) fail("unexpected end of file in \\u escape");
        unsigned int value = 0;
        for (int i = 0; i < 4; i++) value = value * 16 + (unsigned int)hexDigit(text[pos + i]);
        pos += 4;
        return value;
    }

    // Appends code point `cp` to `out` as UTF-8 (1 to 4 bytes).
    static void appendUtf8(std::string& out, unsigned int cp) {
        if (cp < 0x80) {
            out += (char)cp;
        } else if (cp < 0x800) {
            out += (char)(0xC0 | (cp >> 6));
            out += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += (char)(0xE0 | (cp >> 12));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        } else {
            out += (char)(0xF0 | (cp >> 18));
            out += (char)(0x80 | ((cp >> 12) & 0x3F));
            out += (char)(0x80 | ((cp >> 6) & 0x3F));
            out += (char)(0x80 | (cp & 0x3F));
        }
    }

    std::string parseString() {
        expect('"');
        std::string out;
        while (true) {
            if (pos >= text.size()) fail("unterminated string");
            char c = text[pos];
            pos++;
            if (c == '"') return out;
            if ((unsigned char)c < 0x20) fail("control character inside a string");
            if (c != '\\') {
                out += c;
                continue;
            }
            if (pos >= text.size()) fail("unterminated escape");
            char e = text[pos];
            pos++;
            if (e == '"' || e == '\\' || e == '/') out += e;
            else if (e == 'b') out += '\b';
            else if (e == 'f') out += '\f';
            else if (e == 'n') out += '\n';
            else if (e == 'r') out += '\r';
            else if (e == 't') out += '\t';
            else if (e == 'u') {
                unsigned int cp = parseHex4();
                // Characters above U+FFFF are written as two escapes (a surrogate pair).
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    if (text.compare(pos, 2, "\\u") != 0) fail("unpaired surrogate");
                    pos += 2;
                    unsigned int low = parseHex4();
                    if (low < 0xDC00 || low > 0xDFFF) fail("bad low surrogate");
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                    fail("unpaired surrogate");
                }
                appendUtf8(out, cp);
            } else {
                fail("unknown escape");
            }
        }
    }

    double parseNumber() {
        size_t start = pos;
        while (pos < text.size()) {
            char c = text[pos];
            if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E')
                pos++;
            else
                break;
        }
        if (pos == start) fail("unexpected character");
        std::string digits = text.substr(start, pos - start);
        char* end = nullptr;
        double value = std::strtod(digits.c_str(), &end);
        if (*end != '\0') fail("bad number '" + digits + "'");
        return value;
    }
};

JsonDocument::JsonDocument(const std::string& path) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) throw Error("cannot open " + path);
    std::stringstream contents;
    contents << in.rdbuf();
    std::string text = contents.str();

    top = newNode();
    try {
        JsonParser parser(text, path);
        parser.parseDocument(top);
    } catch (...) {
        freeNodes(top);
        throw;
    }
}

JsonDocument::~JsonDocument() {
    freeNodes(top);
}

const JsonNode* JsonDocument::root() const {
    return top;
}

const JsonNode* jsonFind(const JsonNode* object, const std::string& key) {
    if (object == nullptr || object->type != JsonNode::OBJECT) return nullptr;
    for (const JsonNode* member = object->child; member != nullptr; member = member->next)
        if (member->key == key) return member;
    return nullptr;
}

const JsonNode* jsonGet(const JsonNode* object, const std::string& key) {
    const JsonNode* member = jsonFind(object, key);
    if (member == nullptr) throw Error("missing key '" + key + "'");
    return member;
}

std::string jsonString(const JsonNode* object, const std::string& key) {
    const JsonNode* member = jsonGet(object, key);
    if (member->type != JsonNode::STRING) throw Error("key '" + key + "' is not a string");
    return member->text;
}

double jsonNumber(const JsonNode* object, const std::string& key) {
    const JsonNode* member = jsonGet(object, key);
    if (member->type != JsonNode::NUMBER) throw Error("key '" + key + "' is not a number");
    return member->number;
}

int jsonInt(const JsonNode* object, const std::string& key) {
    double value = jsonNumber(object, key);
    if (value != (double)(int)value) throw Error("key '" + key + "' is not an integer");
    return (int)value;
}

bool jsonBool(const JsonNode* object, const std::string& key) {
    const JsonNode* member = jsonGet(object, key);
    if (member->type != JsonNode::BOOLEAN) throw Error("key '" + key + "' is not true/false");
    return member->flag;
}
