// SHA-256 (FIPS 180-4), declared in storage.h.
// The message is processed in 64-byte blocks. Each block is expanded into 64 words and
// mixed into the eight 32-bit state words h[0..7] over 64 rounds. The end of the message
// is padded with 0x80, zeros, and the message length in bits.
#include "storage.h"

#include <cstring>
#include <fstream>

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

// Rotate right by n bits.
static uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

Sha256::Sha256() {
    h[0] = 0x6a09e667;
    h[1] = 0xbb67ae85;
    h[2] = 0x3c6ef372;
    h[3] = 0xa54ff53a;
    h[4] = 0x510e527f;
    h[5] = 0x9b05688c;
    h[6] = 0x1f83d9ab;
    h[7] = 0x5be0cd19;
    bufferLength = 0;
    totalBytes = 0;
}

void Sha256::block(const unsigned char* p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)p[4 * i] << 24) | ((uint32_t)p[4 * i + 1] << 16) |
               ((uint32_t)p[4 * i + 2] << 8) | (uint32_t)p[4 * i + 3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = hh + S1 + ch + K[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
}

void Sha256::update(const void* data, long long n) {
    const unsigned char* p = (const unsigned char*)data;
    totalBytes += (uint64_t)n;
    if (bufferLength > 0) {  // first top up a partly filled block
        long long take = 64 - bufferLength;
        if (take > n) take = n;
        memcpy(buffer + bufferLength, p, take);
        bufferLength += (int)take;
        p += take;
        n -= take;
        if (bufferLength == 64) {
            block(buffer);
            bufferLength = 0;
        }
    }
    while (n >= 64) {
        block(p);
        p += 64;
        n -= 64;
    }
    if (n > 0) {
        memcpy(buffer, p, n);
        bufferLength = (int)n;
    }
}

void Sha256::finish(unsigned char digest[32]) {
    uint64_t bits = totalBytes * 8;
    unsigned char pad = 0x80;
    update(&pad, 1);
    unsigned char zero = 0;
    while (bufferLength != 56) update(&zero, 1);
    unsigned char length[8];
    for (int i = 0; i < 8; i++) length[i] = (unsigned char)(bits >> (56 - 8 * i));
    update(length, 8);
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++) digest[4 * i + j] = (unsigned char)(h[i] >> (24 - 8 * j));
}

std::string toHex(const unsigned char* bytes, int n) {
    const char* digits = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < n; i++) {
        s += digits[bytes[i] >> 4];
        s += digits[bytes[i] & 15];
    }
    return s;
}

std::string sha256Hex(const void* data, long long n) {
    Sha256 hasher;
    hasher.update(data, n);
    unsigned char digest[32];
    hasher.finish(digest);
    return toHex(digest, 32);
}

std::string sha256File(const std::string& path) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) throw PersistenceError("cannot open for hashing: " + path);
    Sha256 hasher;
    const int CHUNK = 1 << 20;
    char* buf = new char[CHUNK];
    while (in) {
        in.read(buf, CHUNK);
        hasher.update(buf, in.gcount());
    }
    delete[] buf;
    unsigned char digest[32];
    hasher.finish(digest);
    return toHex(digest, 32);
}
