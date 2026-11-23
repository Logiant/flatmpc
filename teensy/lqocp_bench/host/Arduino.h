// Host stand-in for <Arduino.h>: runs teensy/lqocp_bench/src/main.cpp on x86 (sanity check, x86 timings).
// The "cycle counter" counts nanoseconds at a nominal 1 GHz, so cyc_us() reports microseconds.
#pragma once
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdarg>
#include <cstdlib>

static inline uint64_t host_ns() {
    static const auto t0 = std::chrono::steady_clock::now();
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
}
static inline uint32_t micros() { return (uint32_t)(host_ns() / 1000); }
static inline uint32_t millis() { return (uint32_t)(host_ns() / 1000000); }
static inline void delay(uint32_t) {}
#define ARM_DWT_CYCCNT ((uint32_t)host_ns())
#define F_CPU_ACTUAL 1000000000u
static uint32_t ARM_DEMCR, ARM_DWT_CTRL;
#define ARM_DEMCR_TRCENA 0
#define ARM_DWT_CTRL_CYCCNTENA 0
#define PROGMEM
struct HostSerial {
    void begin(int) {}
    explicit operator bool() const { return true; }
    int printf(const char* f, ...) {
        va_list a; va_start(a, f); int r = std::vprintf(f, a); va_end(a); std::fflush(stdout); return r;
    }
};
static HostSerial Serial;
void setup();
void loop();
int main() { setup(); loop(); return 0; }
