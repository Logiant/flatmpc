// lqocp bench on Teensy 4.1: solves TINY (general cost) or A and B (energy cost) and prints
// cost, iterations and solve time (setup + solve) over serial. See platformio.ini.
#include <Arduino.h>
#if defined(BENCH_TINY) || defined(BENCH_ENERGY)
#include "bench_data.h"
#endif

#ifdef BENCH_TINY
static lqocp::GSolver gs;
static lqocp::GResult gr;
static const lqocp::SetBasis* lookup(void*, uint16_t S) {
    for (int i = 0; i < TINY_NBASES; ++i)
        if (TINY_BASES[i].S == S) return &TINY_BASES[i];
    return nullptr;
}
#endif
#ifdef BENCH_ENERGY
static lqocp::Solver es;
static lqocp::Result er;
#endif

static void report(const char* name, double J, int its, uint32_t us, int conv, int nsw) {
    Serial.printf("%s J=%.12f its=%d t=%lu us conv=%d switches=%d\n", name, J, its, (unsigned long)us, conv, nsw);
}

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {}
}

void loop() {
#ifdef BENCH_TINY
    {
        const uint32_t t0 = micros();
        bool ok = gs.setup(BENCH_PROB[2], lookup, nullptr) && gs.solve(gr);
        const uint32_t t1 = micros();
        report("TINY", gr.cost, gr.iterations, t1 - t0, ok, gr.narcs - 1);
        if (!ok) Serial.printf("  failure: %s\n", gr.failure ? gr.failure : "setup");
    }
#elif defined(BENCH_ENERGY)
    for (int k = 0; k < 2; ++k) {
        const uint32_t t0 = micros();
        bool ok = es.setup(BENCH_PROB[k]) && es.solve(er, lqocp::METHOD_NEWTON);
        const uint32_t t1 = micros();
        report(k ? "B" : "A", er.cost, er.iterations, t1 - t0, ok, er.narcs - 1);
    }
#else
    const uint32_t t0 = micros();
    const uint32_t t1 = micros();
    report("none", 0.0, 0, t1 - t0, 0, 0);
#endif
    delay(2000);
}
