#ifndef AUTOGRAD_ENGINE_TIMER_H
#define AUTOGRAD_ENGINE_TIMER_H

#include <chrono>

// Generic wall-clock timer for benchmarks and for timing gradient-check sweeps.
// Deliberately knows nothing about autograd: it just accumulates elapsed time.
//
// Starts running on construction, so the common case is a bare scope:
//
//     Timer t;
//     do_work();
//     double ms = t.elapsed_ms();
//
// start()/stop() allow accumulating across several regions, skipping setup:
//
//     Timer t;
//     t.stop();
//     for (int i = 0; i < n; ++i) {
//         setup(i);
//         t.start();
//         do_work(i);
//         t.stop();
//     }
class Timer {
    // steady_clock, not system_clock: monotonic, so an NTP step mid-run cannot
    // produce a negative or wildly inflated duration.
    using clock = std::chrono::steady_clock;

    clock::time_point started_at;
    clock::duration accumulated;
    bool running;

public:
    Timer() : started_at(clock::now()), accumulated(clock::duration::zero()), running(true) {}

    // Resume accumulating. No-op while already running.
    void start() {
        if (!running) {
            started_at = clock::now();
            running = true;
        }
    }

    // Pause accumulating, banking the time since the last start(). No-op while
    // already stopped.
    void stop() {
        if (running) {
            accumulated += clock::now() - started_at;
            running = false;
        }
    }

    // Drop the accumulated time and start over, running.
    void reset() {
        accumulated = clock::duration::zero();
        started_at = clock::now();
        running = true;
    }

    bool is_running() const { return running; }

    // Readable while running or stopped; does not disturb the accumulation.
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
