// The Python CLIP text-encoder helper process (storage.h).
//
// The helper's stdin and stdout are both one end of a Unix socket pair; we keep the other
// end. Messages are length-prefixed: a 4-byte little-endian length, then that many bytes.
#include "storage.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

static const uint32_t ERROR_BIT = 0x80000000u;  // set in a reply header when it is an error

TextEncoderProcess::TextEncoderProcess(const std::string& python, const std::string& script,
                                       const std::string& config, int expectedDim,
                                       const std::string& logFile) {
    pid = -1;
    sock = -1;
    dim = expectedDim;
    if (!fileExists(python)) throw Error("text encoder: Python interpreter not found: " + python);
    if (!fileExists(script)) throw Error("text encoder: helper script not found: " + script);

    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sv) != 0)
        throw Error(std::string("text encoder: socketpair failed: ") + strerror(errno));

    // The child gets sv[1] as stdin and stdout, and stderr goes to the log file. dup2
    // clears close-on-exec on the copies, so only fds 0, 1 and 2 survive into Python.
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, sv[1], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&actions, sv[1], STDOUT_FILENO);
    posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, logFile.c_str(),
                                     O_WRONLY | O_CREAT | O_APPEND, 0644);
    char* argv[4];
    argv[0] = (char*)python.c_str();
    argv[1] = (char*)script.c_str();
    argv[2] = (char*)config.c_str();
    argv[3] = nullptr;
    pid_t child = -1;
    int rc = posix_spawn(&child, python.c_str(), &actions, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(sv[1]);
    if (rc != 0) {
        close(sv[0]);
        throw Error(std::string("text encoder: posix_spawn failed: ") + strerror(rc));
    }
    pid = (int)child;
    sock = sv[0];

    // If the constructor throws, the destructor does not run, so close the socket here;
    // the helper then reads end-of-file and exits.
    uint32_t hello = 0;
    try {
        receiveAll(&hello, 4);  // the helper sends its dimension once the model is loaded
    } catch (const Error&) {
        close(sock);
        throw Error("text encoder failed to start (see " + logFile +
                    "; pass --encoder-log <file> to capture its output)");
    }
    if ((int)hello != dim) {
        close(sock);
        throw Error("text encoder dimension " + std::to_string(hello) + " != config dimension " +
                    std::to_string(dim));
    }
}

TextEncoderProcess::~TextEncoderProcess() {
    if (sock >= 0) {
        uint32_t bye = 0;  // a zero-length request asks the helper to exit
        send(sock, &bye, 4, MSG_NOSIGNAL);
        shutdown(sock, SHUT_WR);
    }
    if (pid > 0) {
        // Wait up to 3 s, then SIGTERM; if it still runs 3 s later, SIGKILL.
        double deadline = nowMs() + 3000.0;
        int status = 0;
        bool signalled = false;
        while (waitpid(pid, &status, WNOHANG) == 0) {
            if (nowMs() > deadline) {
                if (signalled) {
                    kill(pid, SIGKILL);
                    waitpid(pid, &status, 0);
                    break;
                }
                kill(pid, SIGTERM);
                signalled = true;
                deadline = nowMs() + 3000.0;
            }
            usleep(10000);
        }
    }
    if (sock >= 0) close(sock);
}

void TextEncoderProcess::sendAll(const void* data, int n) {
    const char* p = (const char*)data;
    while (n > 0) {
        ssize_t written = send(sock, p, n, MSG_NOSIGNAL);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) throw Error("text encoder: helper process is not accepting input");
        p += written;
        n -= (int)written;
    }
}

void TextEncoderProcess::receiveAll(void* data, int n) {
    char* p = (char*)data;
    while (n > 0) {
        ssize_t got = recv(sock, p, n, 0);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) throw Error("text encoder: helper process exited unexpectedly");
        p += got;
        n -= (int)got;
    }
}

void TextEncoderProcess::encode(const std::string& text, FloatArray& out) {
    if (text.empty()) throw QueryError("query is empty");
    uint32_t length = (uint32_t)text.size();
    sendAll(&length, 4);
    sendAll(text.data(), (int)text.size());
    uint32_t header = 0;
    receiveAll(&header, 4);
    if (header & ERROR_BIT) {
        std::string message(header & ~ERROR_BIT, '\0');
        receiveAll(&message[0], (int)message.size());
        throw Error("text encoder error: " + message);
    }
    if ((int)header != dim * 4)
        throw Error("text encoder returned " + std::to_string(header) + " bytes, expected " +
                    std::to_string(dim * 4));
    out = FloatArray(dim, 0.0f);
    receiveAll(out.data(), (int)header);
}
