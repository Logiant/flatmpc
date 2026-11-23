// lqocp bench on Teensy 4.1: solves TINY (general cost) or A and B (energy cost) and prints
// cost, iterations and solve time (setup + solve) over serial. With BENCH_TINY it also runs the
// closed loop of teensy/tinympc_bench: 100 MPC steps at dt = 0.05 on the exact ZOH model, warm
// started (solve_warm), applying the mean of the planned input over the step. With BENCH_ENERGY
// the same cold solve and loop run on TINY with Q = 0 (energy path: terminal cost Qf, TINY's R).
// See platformio.ini.
#include <Arduino.h>
#include <algorithm>
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
#include "qf_q0.h"                                    // TQ0: TINY with Q = 0, in flash
#endif

#if defined(BENCH_TINY) || defined(BENCH_ENERGY)
#ifdef BENCH_TINY
static const lqocp::Problem& loop_prob() { return BENCH_PROB[2]; }
static bool loop_setup() { return gs.setup(BENCH_PROB[2], lookup, nullptr); }
static void loop_x0(const double* x) { gs.set_x0(x); }
static bool loop_solve(int k, double dt, int cap) { return (k == 0) ? gs.solve(gr) : gs.solve_warm(gr, dt, cap); }
static int loop_u(double dt, double* u) {
    gs.mean_u(gr, 0.0, dt, u);                        // saturated to the bounds; false: holds u
    return gr.iterations;
}
#else
static const lqocp::Problem& loop_prob() { return TQ0; }
static bool loop_setup() { return es.setup(TQ0); }
static void loop_x0(const double* x) { es.set_x0(x); }
static bool loop_solve(int k, double, int cap) { return (k == 0) ? es.solve(er) : es.solve_warm(er, cap); }
static int loop_u(double dt, double* u) {
    es.mean_u(er, 0.0, dt, u);                        // v = Proj(w) is feasible for any iterate
    return er.iterations;
}
#endif

static float cyc_us(uint32_t c) { return c / (F_CPU_ACTUAL / 1e6f); }

// x+ = Ad x + Bd u: exact zero-order hold of the integrator chains (dt^k / k!)
static void zoh_step(const lqocp::Problem& p, double dt, const double* u, double* x) {
    double y[lqocp::MAXN], f[5] = {1, dt, dt * dt / 2, dt * dt * dt / 6, dt * dt * dt * dt / 24};
    int off = 0;
    for (int j = 0; j < p.m; ++j) {
        const int k = p.orders[j];
        for (int r = 0; r < k; ++r) {
            double s = f[k - r] * u[j];
            for (int c = r; c < k; ++c) s += f[c - r] * x[off + c];
            y[off + r] = s;
        }
        off += k;
    }
    for (int i = 0; i < off; ++i) x[i] = y[i];
}

static void closed_loop(int cap) {
    const int STEPS = 100;
    const double dt = 0.05;
    const lqocp::Problem& p = loop_prob();
    double x[lqocp::MAXN], u[lqocp::MAXM] = {0.0, 0.0, 0.0, 0.0}, J = 0.0;
    float t[STEPS];
    int itot = 0, itmax = 0, nconv = 0;
    for (int i = 0; i < 12; ++i) x[i] = p.x0[i];
    loop_setup();
    for (int k = 0; k < STEPS; ++k) {
        loop_x0(x);
        const uint32_t c0 = ARM_DWT_CYCCNT, m0 = micros();
        const bool ok = loop_solve(k, dt, cap);
        const uint32_t c1 = ARM_DWT_CYCCNT, m1 = micros();
        t[k] = (m1 - m0 > 1000000u) ? (float)(m1 - m0) : cyc_us(c1 - c0);   // the cycle counter wraps after 7.2 s
        const int its = loop_u(dt, u);
        itot += its;
        itmax = std::max(itmax, its);
        nconv += ok;
        double sq = 0.0, ur = 0.0;                    // stage cost 1/2 (x'Qx + u'Ru) dt
        for (int i = 0; i < 12; ++i)
            for (int j = 0; j < 12; ++j) sq += x[i] * p.Q[i][j] * x[j];
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) ur += u[i] * p.R[i][j] * u[j];
        J += 0.5 * (sq + ur) * dt;
        zoh_step(p, dt, u, x);
    }
    double xn = 0.0, tsum = 0.0;
    for (int i = 0; i < 12; ++i) xn += x[i] * x[i];
    const float cold = t[0];
    std::sort(t + 1, t + STEPS);                       // warm steps 1..99
    for (int k = 1; k < STEPS; ++k) tsum += t[k];
    Serial.printf("loop cap=%d cold_us=%.1f warm(99) mean=%.1f median=%.1f max=%.1f min=%.1f us "
                  "total_iter=%d max_iter=%d converged=%d/%d J_cl=%.9f |x_end|=%.3g\n",
                  cap, cold, tsum / (STEPS - 1), t[50], t[STEPS - 1], t[1], itot, itmax, nconv, STEPS, J,
                  sqrt(xn));
}
#endif

static void report(const char* name, double J, int its, uint32_t us, int conv, int nsw) {
    Serial.printf("%s J=%.12f its=%d t=%lu us conv=%d switches=%d\n", name, J, its, (unsigned long)us, conv, nsw);
}

void setup() {
    Serial.begin(115200);
    ARM_DEMCR |= ARM_DEMCR_TRCENA;                     // DWT cycle counter
    ARM_DWT_CTRL |= ARM_DWT_CTRL_CYCCNTENA;
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
        closed_loop(100);
        closed_loop(5);
    }
#elif defined(BENCH_ENERGY)
    for (int k = 0; k < 2; ++k) {
        const uint32_t t0 = micros();
        bool ok = es.setup(BENCH_PROB[k]) && es.solve(er, lqocp::METHOD_NEWTON);
        const uint32_t t1 = micros();
        report(k ? "B" : "A", er.cost, er.iterations, t1 - t0, ok, er.narcs - 1);
    }
    {
        const uint32_t t0 = micros();
        bool ok = es.setup(TQ0) && es.solve(er, lqocp::METHOD_NEWTON);   // TinyMPC's own Q=0 terminal weight
        const uint32_t t1 = micros();
        report("TINY_Q0", er.cost, er.iterations, t1 - t0, ok, er.narcs - 1);
        closed_loop(100);
        closed_loop(5);
    }
#else
    const uint32_t t0 = micros();
    const uint32_t t1 = micros();
    report("none", 0.0, 0, t1 - t0, 0, 0);
#endif
    delay(2000);
}
