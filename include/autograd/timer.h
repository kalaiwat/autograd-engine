#ifndef AUTOGRAD_ENGINE_TIMER_H
#define AUTOGRAD_ENGINE_TIMER_H

#include <chrono>

class Timer {
    // steady_clock: monotonic, so an NTP step mid-run can't skew the duration.
    using clock = std::chrono::steady_clock;

    clock::time_point started_at;
    clock::duration accumulated;
    bool running;

public:
    Timer() : started_at(clock::now()), accumulated(clock::duration::zero()), running(true) {}

    void start() {
        if (!running) {
            started_at = clock::now();
            running = true;
        }
    }

    void stop() {
        if (running) {
            accumulated += clock::now() - started_at;
            running = false;
        }
    }

    void reset() {
        accumulated = clock::duration::zero();
        started_at = clock::now();
        running = true;
    }

    bool is_running() const { return running; }

    double elapsed_seconds() const {
        clock::duration total = accumulated;
        if (running) {
            total += clock::now() - started_at;
        }
        return std::chrono::duration<double>(total).count();
    }

    double elapsed_ms() const { return elapsed_seconds() * 1e3; }
};

#endif //AUTOGRAD_ENGINE_TIMER_H
