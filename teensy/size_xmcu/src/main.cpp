// Size-only firmware (no hardware): setup + one cold solve + one warm closed-loop step, result to a
// volatile sink so the linker keeps the solver. PROG_BASE: no solver (reference for the increments).
// PROG_OURS: energy path (TINY with Q = 0, TinyMPC's Qf). PROG_TINY: TinyMPC 0.0.8 on the matched
// problem (Q = 0, input bounds only, N = 10), set up as teensy/tinympc_bench env double_q0_nox.
#if defined(PROG_TINY)
#include "quad.h"
#elif defined(PROG_OURS)
#include "lqocp.hpp"
#include "qf_q0.h"
#endif

volatile double sink;

#if defined(PROG_OURS)
static lqocp::Solver es;
static lqocp::Result er;

int main() {
    double u[4], x[12];
    bool ok = es.setup(TQ0) && es.solve(er) && es.mean_u(er, 0.0, 0.05, u);
    for (int i = 0; i < 12; ++i) x[i] = 0.9 * TQ0.x0[i];   // stand-in for the plant step
    ok = ok && es.set_x0(x);
    ok = ok && es.solve_warm(er, 5, 0.05) && es.mean_u(er, 0.0, 0.05, u);
    sink = ok + er.cost + u[0] + u[1] + u[2] + u[3];
    for (;;) {}
}
#elif defined(PROG_TINY)
int main() {
    TinySolver *s = quad_setup();
    // rho fix, as the bench: rebuild the cache from the original Q, R
    tinyMatrix Q = Map<const tinyVector>(Q_stage, NX).asDiagonal();
    tinyMatrix R = tinytype(4) * tinyMatrix::Identity(NU, NU);
    tiny_precompute_and_set_cache(s->cache, s->work->Adyn, s->work->Bdyn, s->work->fdyn, Q, R, NX, NU, 5.0, 0);
    s->settings->max_iter = 1000;
    tinyVector x = Map<const tinyVector>(X0, NX);
    tiny_set_x0(s, x);
    tiny_solve(s);
    x = s->work->Adyn * x + s->work->Bdyn * s->work->u.col(0);
    tiny_set_x0(s, x);
    tiny_solve(s);
    sink = s->solution->iter + (double)s->work->u(0, 0) + (double)s->work->u(1, 0) + (double)s->work->u(2, 0) +
           (double)s->work->u(3, 0);
    for (;;) {}
}
#else
int main() {
    sink = 1.0;
    for (;;) {}
}
#endif
