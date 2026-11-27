// TinyMPC 0.0.8 solve time on Teensy 4.1: quad hover (quad.h), single-rho cache (rho fix, as Crazyflie / PR #84).
#include "quad.h"  // before Arduino.h: its F() macro breaks Eigen
#include <Arduino.h>
#include <algorithm>
#include <malloc.h>
#ifdef EXACT_ZOH
#include "exact_zoh.h"
#endif

// measured peak stack (DTCM painted from above the 32-byte MPU guard at _ebss to the stack top) and heap (bytes malloc got from sbrk)
extern "C" char _ebss, _estack;
static void stack_paint() {
    char *sp = (char *)__builtin_frame_address(0) - 256;
    for (uint32_t *p = (uint32_t *)((uintptr_t)&_ebss + 32); (char *)p < sp; ++p) *p = 0xA5A5A5A5u;
}
static void mem_report() {
    const uint32_t *p = (const uint32_t *)((uintptr_t)&_ebss + 32);
    while (*p == 0xA5A5A5A5u) ++p;
    Serial.printf("mem stack_peak=%u heap_arena=%u heap_inuse=%u\n", (unsigned)(&_estack - (const char *)p),
                  (unsigned)mallinfo().arena, (unsigned)mallinfo().uordblks);
}

#ifdef SAT_U
#define SAT_TAG " sat"
#else
#define SAT_TAG ""
#endif
static TinySolver *make(int max_iter) {
    TinySolver *s = quad_setup();
    // rho fix: rebuild the cache from the ORIGINAL Q, R (tiny_setup passes Q+rho, R+rho, so rho is added twice).
    tinyMatrix Q = Map<const tinyVector>(Q_stage, NX).asDiagonal();
    tinyMatrix R = tinytype(4) * tinyMatrix::Identity(NU, NU);
    tiny_precompute_and_set_cache(s->cache, s->work->Adyn, s->work->Bdyn, s->work->fdyn, Q, R, NX, NU, 5.0, 0);
    s->settings->max_iter = max_iter;
    return s;
}
static void destroy(TinySolver *s) {
    delete s->solution; delete s->cache; delete s->settings; delete s->work; delete s;
}
static float us(uint32_t cyc) { return cyc / (F_CPU_ACTUAL / 1e6f); }

static void cold(int max_iter) {
    for (int rep = 0; rep < 3; rep++) {
        TinySolver *s = make(max_iter);
        quad_set_x0(s);
        uint32_t c0 = ARM_DWT_CYCCNT;
        tiny_solve(s);
        uint32_t c = ARM_DWT_CYCCNT - c0;
        int it = s->solution->iter;
#ifdef TRACE   // plan (unprojected work->u, 9 columns) and the cached Pinf, for the host judge (t3_tinympc_matched/judge.py)
        if (rep == 0) {
            Serial.printf("TC");
            for (int k = 0; k < NH - 1; k++) for (int i = 0; i < NU; i++) Serial.printf(" %.17g", (double)s->work->u(i, k));
            Serial.printf("\nTP");
            for (int i = 0; i < NX; i++) for (int j = 0; j < NX; j++) Serial.printf(" %.17g", (double)s->cache->Pinf(i, j));
            Serial.printf("\n");
        }
#endif
        Serial.printf("cold maxit=%d rep=%d iter=%d solved=%d t_us=%.1f us_per_iter=%.2f u0=[%.6f %.6f %.6f %.6f]\n",
                      max_iter, rep, it, s->solution->solved, us(c), us(c) / it,
                      (double)s->solution->u(0, 0), (double)s->solution->u(1, 0),
                      (double)s->solution->u(2, 0), (double)s->solution->u(3, 0));
        destroy(s);
    }
}

// As upstream examples/quadrotor_hovering.cpp: warm start, x+ = A x + B u0 (unprojected work->u).
static void closed_loop(int max_iter) {
    const int STEPS = 100;
    TinySolver *s = make(max_iter);
    tinyVector x = Map<const tinyVector>(X0, NX);
#ifdef EXACT_ZOH   // plant state in double, propagated by the exact ZOH of lqocp_bench's loop
    double xd[NX];
    for (int i = 0; i < NX; i++) xd[i] = X0[i];
#define XK(i) xd[i]
#else
#define XK(i) (double)x(i)
#endif
    double J = 0, Jq = 0, umax = 0;  // J_cl = sum 1/2 (x'Qx + u'Ru) dt with the stage Q, and with TINY's Q; max |u| applied
    float t[STEPS]; long total = 0; int nconv = 0, itmax = 0; double tsum = 0, itus = 0;
    for (int k = 0; k < STEPS; k++) {
        tiny_set_x0(s, x);
        uint32_t c0 = ARM_DWT_CYCCNT;
        tiny_solve(s);
        t[k] = us(ARM_DWT_CYCCNT - c0);
        int it = s->solution->iter;
        total += it; nconv += s->solution->solved; itmax = std::max(itmax, it); tsum += t[k];
        if (it) itus += t[k] / it;
#ifdef TRACE   // step, iter, solved, t_us, plant state, plan (unprojected work->u)
        Serial.printf("TS %d %d %d %.1f", k, it, s->solution->solved, (double)t[k]);
        for (int i = 0; i < NX; i++) Serial.printf(" %.17g", XK(i));
        for (int j = 0; j < NH - 1; j++) for (int i = 0; i < NU; i++) Serial.printf(" %.17g", (double)s->work->u(i, j));
        Serial.printf("\n");
#endif
        tinyVector u0 = s->work->u.col(0);
#ifdef SAT_U   // host variant (t3_tinympc_matched): apply col 0 clipped to the box, as ours (MpcPlan::mean_u saturates)
        u0 = u0.cwiseMax(tinytype(-0.5)).cwiseMin(tinytype(0.5));
#endif
        double ur = 4.0 * (double)u0.squaredNorm(), sq = 0, sqt = 0;
        for (int i = 0; i < NX; i++) { sq += Q_stage[i] * XK(i) * XK(i); sqt += Q_d[i] * XK(i) * XK(i); }
        J += 0.5 * (sq + ur) * 0.05; Jq += 0.5 * (sqt + ur) * 0.05;
        umax = std::max(umax, (double)u0.cwiseAbs().maxCoeff());
#ifdef EXACT_ZOH
        double y[NX];
        for (int i = 0; i < NX; i++) {
            y[i] = 0;
            for (int j = 0; j < NX; j++) y[i] += AZ[i][j] * xd[j];
            for (int j = 0; j < NU; j++) y[i] += BZ[i][j] * (double)u0(j);
        }
        for (int i = 0; i < NX; i++) x(i) = (tinytype)(xd[i] = y[i]);
#else
        x = s->work->Adyn * x + s->work->Bdyn * s->work->u.col(0);
#endif
    }
#undef XK
    std::sort(t, t + STEPS);
    Serial.printf("loop maxit=%d total_iter=%ld mean_iter=%.2f max_iter_seen=%d converged=%d/%d "
                  "t_us mean=%.1f median=%.1f max=%.1f min=%.1f total_ms=%.2f us_per_iter=%.2f final|x|=%.3g J_cl=%.9f J_cl_tinyQ=%.9f max|u|=%.6f\n",
                  max_iter, total, total / (double)STEPS, itmax, nconv, STEPS,
                  tsum / STEPS, (t[49] + t[50]) / 2, t[STEPS - 1], t[0], tsum / 1000, tsum / total, (double)x.norm(), J, Jq, umax);
    destroy(s);
}

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 5000) {}
    delay(500);
    stack_paint();
}

void loop() {
    Serial.printf("=== TinyMPC 0.0.8 %s Q%s" SAT_TAG ", F_CPU=%lu ===\n", sizeof(tinytype) == 8 ? "double" : "float", Q_stage == Q_d ? "=TINY" : "=0", (unsigned long)F_CPU_ACTUAL);
    cold(10000);
    closed_loop(1000);
    closed_loop(100);
    closed_loop(5);
    mem_report();
    Serial.println("=== done ===");
    delay(3000);
}
