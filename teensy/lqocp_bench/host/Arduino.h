// x86 stand-in for the Teensy Arduino API used by ../src/main.cpp (host timing of the same bench).
#pragma once
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <ctime>
#define PROGMEM
#define F_CPU_ACTUAL 1000000000u
static inline uint64_t host_ns() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000000000ull + t.tv_nsec; }
static inline uint32_t micros() { return (uint32_t)(host_ns() / 1000); }
static inline uint32_t millis() { return (uint32_t)(host_ns() / 1000000); }
static inline void delay(int) {}
#define ARM_DWT_CYCCNT ((uint32_t)host_ns())   // "cycles" = ns at 1 GHz
static uint32_t ARM_DEMCR, ARM_DWT_CTRL;
#define ARM_DEMCR_TRCENA 0
#define ARM_DWT_CTRL_CYCCNTENA 0
struct HostSerial {
    void begin(int) {}
    explicit operator bool() const { return true; }
    template <class... A> void printf(const char* f, A... a) { std::printf(f, a...); }
};
static HostSerial Serial;
