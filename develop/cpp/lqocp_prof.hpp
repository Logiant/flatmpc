// Profiling scopes (measurement only, -DLQOCP_PROF). Without the flag every macro is empty and the object code is
// unchanged. Clock: ARM DWT cycle counter on the Teensy (extended to 64 bits; each scope reads it, so the gap between
// reads stays far below its 7.2 s wrap), steady_clock ns on the host. Per id: calls, inclusive (outermost activation
// only, so recursion is not double counted) and exclusive time; per (parent, id) pair: calls and inclusive time.
#ifndef LQOCP_PROF_HPP
#define LQOCP_PROF_HPP

#ifdef LQOCP_PROF
#include <stdint.h>

#define LQOCP_PROF_IDS(X)                                                                                              \
    X(G_SETUP) X(G_COLD) X(G_WARM) X(G_SHIFT) X(G_CORRECT) X(G_NEWTON) X(N_SENSQR) X(N_QR) X(N_CHORD) X(N_SOLVE)        \
    X(N_COLLAPSE) X(N_LSEARCH) X(G_BVP) X(G_FORWARD) X(G_FSTEP) X(G_VSTEP) X(G_ORTHO) X(G_TERMINAL) X(G_LOAD)         \
    X(G_RECOVER) X(G_ARC) X(G_CEXP) X(G_ZAT) X(G_SENS) X(G_JUMP) X(G_RESID) X(G_JROWS) X(G_EXPL) X(G_REFINE)            \
    X(G_DIVERGED) X(G_INFEAS) X(G_PROPOSE) X(G_GWALK) X(G_CHAN) X(G_EVENTS) X(G_LEAD) X(G_GLEX) X(G_BB) X(G_PBB)        \
    X(G_NODE) X(G_FROOT) X(G_FEVAL) X(G_MTAB) X(G_FINISH) X(G_GTERM) X(G_JNOISE) X(G_GVERIFY) X(G_GCOST) X(G_GENDS)    \
    X(G_GMAXR) X(G_GSCALE) X(G_MEANU) X(G_CLAMPU) X(M_KEEP) X(M_MEANU)                                                 \
    X(E_SETUP) X(E_SOLVE) X(E_WARM) X(E_SHIFT) X(E_REDUCED) X(E_NEWTON) X(E_EVAL) X(E_ASSEMBLE) X(E_SOLVEC) X(E_WALK)  \
    X(E_WALKBOX) X(E_DUAL) X(E_VERIFY) X(E_COST) X(E_INFEAS) X(E_SWJAC) X(E_GAPK) X(E_MEANU)                          \
    X(S_SETUP) X(S_SOLVE) X(S_WARM) X(S_RUN) X(S_INNER) X(S_EVAL) X(S_KKT) X(S_WALK) X(S_GRES) X(S_TANGENT)            \
    X(S_JUNCTIONS) X(S_CHECK) X(S_SCAN) X(S_FILL) X(S_MEANU) X(S_INTEG)

namespace lqocp {
namespace prof {
#define LQOCP_PROF_ENUM(x) x,
enum Id { NONE = 0, LQOCP_PROF_IDS(LQOCP_PROF_ENUM) NID };
#undef LQOCP_PROF_ENUM
extern const char* const name[NID];
extern uint64_t incl[NID], excl[NID], pincl[NID][NID];
extern uint32_t calls[NID], pcalls[NID][NID], depth[NID];
extern int cur;                                    // innermost open scope (NONE at top level)
extern uint64_t* cur_child;                        // child time of the innermost open scope

#if defined(__arm__)
extern uint32_t lo_, hi_;
inline uint64_t now() {
    const uint32_t c = *(volatile uint32_t*)0xE0001004;   // ARM_DWT_CYCCNT
    if (c < lo_) ++hi_;
    lo_ = c;
    return ((uint64_t)hi_ << 32) | c;
}
#else
uint64_t now();
#endif

struct Scope {
    int id, parent;
    uint64_t t0, child = 0, *pchild;
    bool open = true;
    explicit Scope(int i) : id(i), parent(cur), pchild(cur_child) {
        cur = i;
        cur_child = &child;
        ++depth[i];
        t0 = now();
    }
    void stop() {
        if (!open) return;
        const uint64_t dt = now() - t0;
        open = false;
        excl[id] += dt - child;
        if (depth[id] == 1) incl[id] += dt;
        --depth[id];
        ++calls[id];
        ++pcalls[parent][id];
        if (parent != id) pincl[parent][id] += dt;
        if (pchild) *pchild += dt;
        cur = parent;
        cur_child = pchild;
    }
    ~Scope() { stop(); }
};
void reset();
}  // namespace prof
}  // namespace lqocp

#define LQOCP_PROF_SCOPE(id) lqocp::prof::Scope lqocp_prof_##id(lqocp::prof::id)
#define LQOCP_PROF_END(id) lqocp_prof_##id.stop()
#else
#define LQOCP_PROF_SCOPE(id)
#define LQOCP_PROF_END(id)
#endif

#endif
