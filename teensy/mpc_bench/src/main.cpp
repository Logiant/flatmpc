// Ideal-model closed-loop MPC bench: ours (tier 1 Solver, tier 2 StateSolver) vs TinyMPC 0.0.8 on the exact
// continuous plant (flat integrator chains, exact flow of every held input, no noise), plus TinyMPC's knot sweep.
// The same file builds on x86 (host/Makefile) and on the Teensy 4.1 (platformio.ini; envs ours, tiny, both).
// Output: CSV lines (header lines start with '#'); summarise with teensy/mpc_bench/summarise.py.
//
// Problem (gen_data.py): Q = 0, R = 4I on the thrusts, |u| <= 0.5, T = 0.45, Qf(N) = dt (DARE(Ad, Bd, rho I, R + rho I) - rho I),
// tier 2 adds |x_phys| <= 5 on all 12 states. Control period 0.05 s, 100 steps.
// Applied input: ours, the mean of its plan over the period (ZOH); TinyMPC, its plan's held inputs over the period
// (u0 for N = 10; for N > 10 the inputs of every knot interval inside [0, 0.05], the last one cut at 0.05),
// unprojected work->u as upstream (clamp = 0) or clamped to the box (clamp = 1).
// The state box is checked in continuous time on every held piece: each state is a polynomial of degree <= 4 in t,
// its max |.| is taken at the ends and at the real roots of its derivative (monotone pieces + bisection, no grid).
#ifdef BENCH_TINY
#include "tiny_api.hpp"                               // before Arduino.h: its F() macro breaks Eigen
#endif
#ifdef ARDUINO
#include <Arduino.h>
#define DATA_FLASH PROGMEM
#define OUT Serial.printf
typedef uint32_t Stamp;
static Stamp tic() { return ARM_DWT_CYCCNT; }
static double toc(Stamp s) { return (uint32_t)(ARM_DWT_CYCCNT - s) / (F_CPU_ACTUAL / 1e6); }   // wraps after 7.2 s
#ifndef BENCH_NX0
#define BENCH_NX0 1                                   // TINY's x0 only on the board (11 x0 take minutes)
#endif
#else
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#define OUT printf
typedef std::chrono::steady_clock::time_point Stamp;
static Stamp tic() { return std::chrono::steady_clock::now(); }
static double toc(Stamp s) { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - s).count(); }
#endif
#include <algorithm>
#include <cmath>
#include "mpc_data.h"
#ifdef ARDUINO
#define DBG(...) ((void)0)
#else
#define DBG(...) (getenv("MPCDBG") ? (void)fprintf(stderr, __VA_ARGS__) : (void)0)   // per-step failure reasons
#endif
#ifndef BENCH_NX0
#define BENCH_NX0 NX0
#endif
#ifdef BENCH_OURS
#include "lqocp_state.hpp"
#endif

static const int ORD[4] = {4, 4, 2, 2}, STEPS = 100, KP = 96;   // KP: max held pieces per plan (N - 1 <= 89)
static const double XB = 5.0, UB = 0.5, DT = 0.05, TH = 0.45;

// ---------- exact plant and continuous-time bound check ----------
static double peval(const double* c, int d, double t) {
    double s = c[d];
    for (int i = d - 1; i >= 0; --i) s = s * t + c[i];
    return s;
}
// Candidates for the extrema of p (degree d <= 4) on (0, h): the real roots of p' (bisection on the monotone pieces
// of p', split at the roots of p'' recursively) and those split points. Returns their count (<= 7).
static int crit(const double* c, int d, double h, double* out) {
    if (d < 2) return 0;
    double dc[4], br[8];
    for (int i = 1; i <= d; ++i) dc[i - 1] = i * c[i];
    int nb = 0, n = 0;
    br[nb++] = 0.0;
    nb += crit(dc, d - 1, h, br + nb);
    std::sort(br + 1, br + nb);
    br[nb++] = h;
    for (int k = 0; k + 1 < nb; ++k) {
        double a = br[k], b = br[k + 1], fa = peval(dc, d - 1, a), fb = peval(dc, d - 1, b);
        if (fa == 0.0 || fb == 0.0 || (fa < 0) == (fb < 0)) continue;   // zeros at split points are candidates below
        for (int it = 0; it < 200 && b - a > 1e-16 * h; ++it) {
            const double m = 0.5 * (a + b), fm = peval(dc, d - 1, m);
            if ((fm < 0) == (fa < 0)) a = m, fa = fm; else b = m;
        }
        out[n++] = 0.5 * (a + b);
    }
    for (int k = 1; k + 1 < nb; ++k) out[n++] = br[k];
    return n;
}
static double maxabs(const double* c, int d, double h) {
    double t[8], m = std::max(std::fabs(c[0]), std::fabs(peval(c, d, h)));
    const int n = crit(c, d, h, t);
    for (int i = 0; i < n; ++i) m = std::max(m, std::fabs(peval(c, d, t[i])));
    return m;
}
static double PSA[12];                                // |physical / flat| scale per flat state
// Exact flow of the flat chains over h with v held; returns max over [0, h] of |x_phys| - XB.
static double piece(double* s, const double* v, double h) {
    static const double FACT[5] = {1, 1, 2, 6, 24};
    double y[12], viol = -HUGE_VAL, c[5];
    for (int j = 0, off = 0; j < 4; off += ORD[j++]) {
        const int k = ORD[j];
        for (int r = 0; r < k; ++r) {
            for (int i = 0; r + i < k; ++i) c[i] = s[off + r + i] / FACT[i];
            c[k - r] = v[j] / FACT[k - r];
            viol = std::max(viol, PSA[off + r] * maxabs(c, k - r, h) - XB);
            y[off + r] = peval(c, k - r, h);
        }
    }
    std::copy(y, y + 12, s);
    return viol;
}
static void selfcheck() {                             // p = (t - 0.3)^2 (t - 1) on [0, 1]: max |p| = 0.09 at t = 0
    const double c[4] = {-0.09, 0.69, -1.6, 1.0}, c2[3] = {0.0, 1.0, -1.0};   // t - t^2: max 0.25 at 0.5
    if (std::fabs(maxabs(c, 3, 1.0) - 0.09) > 1e-15 || std::fabs(maxabs(c2, 2, 1.0) - 0.25) > 1e-15) OUT("# SELFCHECK FAILED\n");
}

// ---------- closed loop ----------
enum Who { OURS, TINY };
struct Res { double Ju, Jcl, xv, uv, xend, tcold, tmed, tmean, tmax; int conv; long its; int itmax; };
struct Plan { int n; double v[KP][4], h[KP], u[KP][4]; };

static void mulv(const double* M, const double* a, double* b) {   // b = M a, 4x4 row-major
    for (int i = 0; i < 4; ++i) { b[i] = 0; for (int j = 0; j < 4; ++j) b[i] += M[4 * i + j] * a[j]; }
}
static double qf_cost(int sw, const double* s) {
    double J = 0;
    for (int i = 0; i < 12; ++i) for (int j = 0; j < 12; ++j) J += 0.5 * s[i] * SW_QF[sw][12 * i + j] * s[j];
    return J;
}

#ifdef BENCH_OURS
static lqocp::Problem P;
static lqocp::Solver es;
static lqocp::Result er;
static lqocp::StateSolver ss;
static lqocp::StateResult sr;
static double SLO[12], SHI[12];
static void make_problem(int sw, const double* s0) {
    P.m = 4;
    for (int j = 0; j < 4; ++j) P.orders[j] = ORD[j];
    P.T = TH;
    for (int i = 0; i < 12; ++i) P.x0[i] = s0[i], P.xf[i] = 0.0;
    P.has_bounds = 1;
    for (int i = 0; i < 4; ++i) {
        P.lo[i] = -UB, P.hi[i] = UB;
        for (int j = 0; j < 4; ++j) P.M[i][j] = UMAP[4 * i + j], P.R[i][j] = RFLAT[4 * i + j];
    }
    P.has_cost = 1;                                   // Q = 0: energy path with terminal weight
    P.has_Qf = 1;
    for (int i = 0; i < 12; ++i) for (int j = 0; j < 12; ++j) P.Q[i][j] = 0.0, P.Qf[i][j] = SW_QF[sw][12 * i + j];
    for (int i = 0; i < 12; ++i) SLO[PF[i]] = -XB / std::fabs(PS[i]), SHI[PF[i]] = XB / std::fabs(PS[i]);
}
static bool ours_setup(int tier, int sw, const double* s0) {
    make_problem(sw, s0);
    return tier == 1 ? es.setup(P) : ss.setup(P, SLO, SHI);
}
static bool ours_solve(int tier, int k, int cap, int* its) {
    bool ok;
    if (tier == 1) {
        ok = (k == 0) ? es.solve(er) : es.solve_warm(er, cap);
        *its = er.iterations;
    } else {
        ok = (k == 0) ? ss.solve(sr) : ss.solve_warm(sr, cap, DT);
        if (!ok && k > 0 && !(sr.failure && strstr(sr.failure, "iteration cap"))) ok = ss.solve(sr);   // cold fallback
        *its = sr.iterations;
    }
    if (!ok) DBG("tier %d step %d: %s\n", tier, k, tier == 1 ? er.failure : sr.failure);
    return ok;
}
#endif

#ifdef BENCH_TINY
static TinySolver* tiny_make(int sw, int tier, int max_iter) {
    typedef Matrix<tinytype, Dynamic, Dynamic, RowMajor> RM;
    const int N = SW_N[sw];
    tinyMatrix A = Map<const RM>(SW_AD[sw], 12, 12), B = Map<const RM>(SW_BD[sw], 12, 4), f = tinyMatrix::Zero(12, 1);
    tinyMatrix Q = tinyMatrix::Zero(12, 12), R = tinytype(4) * tinyMatrix::Identity(4, 4);
    TinySolver* s = nullptr;
    if (tiny_setup(&s, A, B, f, Q, R, 5.0, 12, 4, N, 0)) return nullptr;
    // single-rho cache (as teensy/tinympc_bench): tiny_setup passes Q+rho, R+rho, so rho would be added twice
    tiny_precompute_and_set_cache(s->cache, A, B, f, Q, R, 12, 4, 5.0, 0);
    const double xb = tier == 2 ? XB : 1e17;
    tiny_set_bound_constraints(s, tinyMatrix::Constant(12, N, -xb), tinyMatrix::Constant(12, N, xb),
                               tinyMatrix::Constant(4, N - 1, -UB), tinyMatrix::Constant(4, N - 1, UB));
    s->settings->en_state_bound = tier == 2;
    s->settings->max_iter = max_iter;
    return s;
}
static void tiny_free(TinySolver* s) { delete s->solution; delete s->cache; delete s->settings; delete s->work; delete s; }
static void tiny_x0(TinySolver* s, const double* sf) {
    tinyVector x(12);
    for (int i = 0; i < 12; ++i) x(i) = PS[i] * sf[PF[i]];
    tiny_set_x0(s, x);
}
// The plan's held inputs over [0, span]: knot interval j is [j h, (j+1) h]; U = work->u or solution->u.
static void tiny_plan(const tinyMatrix& U, int sw, double span, int clamp, Plan& pl) {
    const double h = TH / (SW_N[sw] - 1);
    pl.n = 0;
    for (int j = 0; j < U.cols() && j * h < span * (1 - 1e-12) && pl.n < KP; ++j, ++pl.n) {
        pl.h[pl.n] = std::min(h, span - j * h);
        for (int i = 0; i < 4; ++i) pl.u[pl.n][i] = clamp ? std::max(-UB, std::min(UB, (double)U(i, j))) : (double)U(i, j);
        mulv(UMINV, pl.u[pl.n], pl.v[pl.n]);
    }
}
#endif

// Applies a plan to the plant state s; accumulates J_u and the violations.
static void apply(const Plan& pl, double* s, Res& r) {
    for (int p = 0; p < pl.n; ++p) {
        double uu = 0;
        for (int i = 0; i < 4; ++i) uu += pl.u[p][i] * pl.u[p][i], r.uv = std::max(r.uv, std::fabs(pl.u[p][i]) - UB);
        r.Ju += 0.5 * 4.0 * uu * pl.h[p];
        r.xv = std::max(r.xv, piece(s, pl.v[p], pl.h[p]));
    }
}

static Res loop(Who who, int tier, int cap, int clamp, int sw, const double* s0) {
    Res r = {0, 0, -HUGE_VAL, -HUGE_VAL, 0, 0, 0, 0, 0, 0, 0, 0};
    double s[12], t[STEPS];
    std::copy(s0, s0 + 12, s);
    Plan pl;
#ifdef BENCH_OURS
    if (who == OURS && !ours_setup(tier, sw, s0)) { OUT("# ours setup failed\n"); return r; }
#endif
#ifdef BENCH_TINY
    TinySolver* ts = who == TINY ? tiny_make(sw, tier, cap) : nullptr;
    if (who == TINY && !ts) { OUT("# tiny setup failed\n"); return r; }
#endif
    for (int k = 0; k < STEPS; ++k) {
        int its = 0;
        bool ok = false;
#ifdef BENCH_OURS
        if (who == OURS) {
            tier == 1 ? es.set_x0(s) : ss.set_x0(s);
            const Stamp c0 = tic();
            ok = ours_solve(tier, k, cap, &its);
            t[k] = toc(c0);
            pl.n = 1, pl.h[0] = DT;
            tier == 1 ? es.mean_u(er, 0.0, DT, pl.v[0]) : ss.mean_u(sr, 0.0, DT, pl.v[0]);
            mulv(UMAP, pl.v[0], pl.u[0]);
        }
#endif
#ifdef BENCH_TINY
        if (who == TINY) {
            tiny_x0(ts, s);
            const Stamp c0 = tic();
            tiny_solve(ts);
            t[k] = toc(c0);
            ok = ts->solution->solved, its = ts->solution->iter;
            tiny_plan(ts->work->u, sw, DT, clamp, pl);
        }
#endif
        r.conv += ok, r.its += its, r.itmax = std::max(r.itmax, its);
        apply(pl, s, r);
    }
#ifdef BENCH_TINY
    if (ts) tiny_free(ts);
#endif
    for (int i = 0; i < 12; ++i) r.xend += s[i] * s[i];
    r.xend = std::sqrt(r.xend);
    r.Jcl = r.Ju + qf_cost(0, s);
    r.tcold = t[0];
    std::sort(t + 1, t + STEPS);                      // warm steps 1..99
    for (int k = 1; k < STEPS; ++k) r.tmean += t[k] / (STEPS - 1);
    r.tmed = t[STEPS / 2], r.tmax = t[STEPS - 1];
    return r;
}
static void print_loop(const char* tag, const char* who, int tier, int x0, int N, int cap, int clamp, const Res& r) {
    OUT("%s,%s,%d,%d,%d,%d,%d,%d,%.10g,%.10g,%.4g,%.6g,%.6g,%.1f,%.2f,%.2f,%.1f,%ld,%d\n", tag, who, tier, x0, N, cap, clamp,
        r.conv, r.Ju, r.Jcl, r.xend, r.xv, r.uv, r.tcold, r.tmed, r.tmean, r.tmax, r.its, r.itmax);
}
static void loop_header() {
    OUT("#tag,solver,tier,x0,N,cap,clamp,conv,J_u,J_cl,x_end,max_xviol,max_uviol,t_cold_us,t_med_us,t_mean_us,t_max_us,its_total,its_max\n");
}

// ---------- parts ----------
static void closed_loops() {
    loop_header();
    for (int tier = 1; tier <= 2; ++tier)
        for (int x = 0; x < BENCH_NX0; ++x) {
#ifdef BENCH_OURS
            for (int cap : {100, 5}) DBG("== tier %d x0 %d cap %d\n", tier, x, cap), print_loop("cl", "ours", tier, x, 10, cap, 0, loop(OURS, tier, cap, 0, 0, X0F[x]));
#endif
#ifdef BENCH_TINY
            for (int mi : {1000, 100, 5})
                for (int cl = 0; cl < 2; ++cl) print_loop("cl", "tiny", tier, x, 10, mi, cl, loop(TINY, tier, mi, cl, 0, X0F[x]));
#endif
        }
}

static void sweep() {
    OUT("#tag,N,x0,solver,its,conv,t_us,J,max_xviol,max_uviol\n");
    for (int sw = 0; sw < NSW; ++sw)
        for (int x = 0; x < BENCH_NX0; ++x) {
        const int N = SW_N[sw];
#ifdef BENCH_OURS
        {
            ours_setup(2, sw, X0F[x]);
            const Stamp c0 = tic();
            const bool ok = ss.solve(sr);
            const double tu = toc(c0);
            OUT("os,%d,%d,ours,%d,%d,%.1f,%.15g,0,0\n", N, x, sr.iterations, (int)ok, tu, sr.cost);   // continuous box: no violation
        }
#endif
#ifdef BENCH_TINY
        {
            TinySolver* ts = tiny_make(sw, 2, 100000);
            tiny_x0(ts, X0F[x]);
            const Stamp c0 = tic();
            tiny_solve(ts);
            const double tu = toc(c0);
            Plan pl;
            Res r = {0, 0, -HUGE_VAL, -HUGE_VAL, 0, 0, 0, 0, 0, 0, 0, 0};
            double s[12];
            std::copy(X0F[x], X0F[x] + 12, s);
            tiny_plan(ts->solution->u, sw, TH, 0, pl);    // its whole (projected) plan over the horizon
            apply(pl, s, r);
            OUT("os,%d,%d,tiny,%d,%d,%.1f,%.15g,%.6g,%.6g\n", N, x, ts->solution->iter, ts->solution->solved, tu, r.Ju + qf_cost(sw, s), r.xv, r.uv);
            tiny_free(ts);
        }
#endif
    }
    loop_header();                                    // closed loop per N; ours with the same Qf(N) as reference
    for (int sw = 0; sw < NSW; ++sw)
        for (int x = 0; x < BENCH_NX0; ++x) {
#ifdef BENCH_OURS
            print_loop("sw", "ours", 2, x, SW_N[sw], 100, 0, loop(OURS, 2, 100, 0, sw, X0F[x]));
#endif
#ifdef BENCH_TINY
            for (int mi : {1000, 100})
                for (int cl = 0; cl < 2; ++cl) print_loop("sw", "tiny", 2, x, SW_N[sw], mi, cl, loop(TINY, 2, mi, cl, sw, X0F[x]));
#endif
        }
}

static void run_all(bool cl, bool sw) {
    for (int i = 0; i < 12; ++i) PSA[PF[i]] = std::fabs(PS[i]);
    selfcheck();
    if (cl) closed_loops();
    if (sw) sweep();
}

#ifdef ARDUINO
void setup() {
    Serial.begin(115200);
    ARM_DEMCR |= ARM_DEMCR_TRCENA;                    // DWT cycle counter
    ARM_DWT_CTRL |= ARM_DWT_CTRL_CYCCNTENA;
    while (!Serial && millis() < 3000) {}
}
void loop() {
    OUT("=== mpc_bench F_CPU=%lu ===\n", (unsigned long)F_CPU_ACTUAL);
    run_all(true, true);
    OUT("=== done ===\n");
    delay(10000);
}
#else
int main(int argc, char** argv) {                     // mpc_bench [cl|sweep]
    std::cout.setstate(std::ios_base::badbit);        // TinyMPC prints "Solver converged" via std::cout
    const char* a = argc > 1 ? argv[1] : "";
    run_all(!strcmp(a, "") || !strcmp(a, "cl"), !strcmp(a, "") || !strcmp(a, "sweep"));
    return 0;
}
#endif
