// Tier 2: energy cost (Q = 0, optional Qf), thrust box AND a box on first-order (relative degree 1)
// states: the top state of each chain (x/y jerk = pitch/roll rate, z speed, yaw rate). NOTES.md 14.
//
// A boundary arc [t1, t2] of chain j pins v_j = 0 (the top state stays at the bound b). Junction
// conditions on states and controls: D_top_j(t1) = b, v_j(t1-) = 0, v_j(t2+) = 0 (control
// continuous). For fixed junction times the problem is tier 1 plus the pin and the interior point
// condition D_top_j(t1) = b; its concave dual adds one unknown per arc (the step Delta of w_j
// before t1). The junction times come from Newton on the two control-continuity conditions.
// Same embedded rules as lqocp.hpp: fixed sizes, no heap, no exceptions.
#ifndef LQOCP_STATE_HPP
#define LQOCP_STATE_HPP

#include <math.h>

#include "lqocp.hpp"

namespace lqocp {

#ifndef LQOCP_MAXSA
#define LQOCP_MAXSA 4
#endif
constexpr int MAXSA = LQOCP_MAXSA;   // max boundary (state) arcs
constexpr int NY = MAXN + MAXSA;     // dual unknowns: nu (or mu with Qf), then one Delta per arc

struct StateArc {
    int8_t chain, side;              // side +1: upper bound, -1: lower bound
    bool fix1, fix2;                 // t1 = 0 (x0 already at the bound) / t2 = T (rides to the end)
                                     // fix2 with t1 = T: terminal touch, D_top(T) = b only
    double b, t1, t2;
};

struct StateResult {
    int narcs;                       // plan: arc a is [tau[a-1], tau[a]], set S[a]; bits 8+j: chain j pinned
    uint16_t S[MAXARC_E];
    double tau[MAXJ_E];
    double y[NY];                    // dual iterate (nu or mu, then Delta)
    int nsa;
    StateArc sa[MAXSA];
    double cost;
    int iterations;                  // inner Newton iterations (all solves)
    int outer;                       // junction-time Newton iterations
    int inserted;                    // boundary arcs inserted
    int steps;                       // homotopy steps (solve: 1 = the direct solve at the target bounds)
    double residual, state_violation, pmp_violation, sign_violation;
    bool converged;
    const char* failure;
};

class StateSolver {
public:
    // slo/shi: bounds on the flat states (+-HUGE_VAL: none). Only the top state of a chain
    // (relative degree 1) is enforced; bounds on lower states are verified exactly on the result,
    // which fails with "unsupported constraint order" if one is violated. p must stay alive.
    bool setup(const Problem& p, const double* slo, const double* shi);
    const char* setup_failure() const { return sfail_; }
    // cold solve by continuation in the state bounds (NOTES.md 14.6)
    bool solve(StateResult& r);
    // MPC: from the previous arcs and dual iterate, at most max_its inner Newton iterations per solve.
    // shift: time the horizon moved since r was computed (arcs move left by it; an arc that now starts
    // at or before 0 starts at 0, one that ended is dropped). Update x0 with set_x0 first.
    bool solve_warm(StateResult& r, int max_its, double shift = 0.0);
    bool set_x0(const double* x0) { return base_.set_x0(x0); }
    bool mean_u(const StateResult& r, double t0, double t1, double u[MAXM]) const;
    void eval_u(const StateResult& r, double t, double u[MAXM]) const;
    void eval_x(const StateResult& r, double t, double x[MAXN]) const;
    // observed arc sequence: per arc the active thrust rows ("2+": row 2 at its upper bound) and
    // pinned chains ("|y+": y's top state at its upper bound); "f": free
    const char* sequence(const uint16_t* S, int narcs, const StateArc* sa, int nsa, const double* tau,
                         char* buf, int len) const;

private:
    struct Ev {
        int narcs;
        uint16_t S[MAXARC_E];
        double tau[MAXJ_E];
        double F[NY];
        double scale, D;
    };
    Solver base_;                    // tier 1 and the setup data
    const char* sfail_ = nullptr;
    double lo_[MAXM], hi_[MAXM];     // bound on the top state of chain j at s = 0 (start; target until the first solve)
    double xlo_[MAXN], xhi_[MAXN];   // all flat-state bounds (relative degree > 1: verified only)
    mutable double hiord_;
    mutable int hiidx_, release_;    // release_: fix2 arc whose terminal multiplier has the wrong sign
    double hs_ = 1.0;                // homotopy parameter s: bound(s) = s target + (1 - s) lo_ / hi_
    int nsa_;                        // working arcs (copied to / from StateResult)
    StateArc sa_[MAXSA];
    int ny_;
    int its_, cap_;
    Ev ev_[2];
    // Newton matrix (-Hess D) and Z = H^-1 dF/dth (junction Jacobian; in inner: a packed copy of H) live
    // in the tier-1 solve scratch, which is dead once the tier-1 solve is done
    double* const H_ = base_.xs_;
    double* const Z_ = base_.xs_ + NY * NY;
    static_assert(sizeof(Solver::xs_) >= (NY * NY + (NY * (NY + 1) / 2 > NY * 2 * MAXSA ? NY * (NY + 1) / 2 : NY * 2 * MAXSA)) * sizeof(double), "tier-1 scratch too small");
    char why_[256];
    int collapsed_;

    bool run(StateResult& r, int warm_its);
    bool cold(StateResult& r);
    void wpoly(const double* c, const double* y, const StateArc* sa, int nsa, double tm, double Wc[MAXM][4]) const;
    bool kkt(uint16_t S, const double Wc[MAXM][4], double Pv[MAXM][MAXM], double Vv[MAXM][4],
             double phi[2 * MAXM][4], bool has[2 * MAXM], double sig[MAXM][4]) const;
    bool lex(const double Wc[MAXM][4], uint16_t pin, double t, double L, uint16_t guess, uint16_t* out,
             double Pv[MAXM][MAXM], double Vv[MAXM][4], double phi[2 * MAXM][4], bool has[2 * MAXM],
             double sig[MAXM][4]) const;
    bool walk(const double* c, const double* y, Ev& e) const;
    void cvec(const double* y, double* c) const;
    bool evaluate(const double* y, Ev& e, bool wantH);
    bool integrate(const double* c, const double* y, Ev& e, bool wantH);
    bool inner(double* y, int max_its);
    double wbasis(int i, double t, double tm) const;
    bool gres(const double* y, double* g, int* ng, double* vs, double* J = nullptr, const double* zs = nullptr,
              double* dgs = nullptr);
    bool tangent(const double* y, double* dth, double* dy);
    bool touch(const StateArc& s) const { return s.fix2 && s.t1 >= base_.p_->T; }
    double tgt(int j, int side) const { const int i = base_.off_[j] + base_.p_->orders[j] - 1; return side > 0 ? xhi_[i] : xlo_[i]; }
    double bnd(int j, int side) const {             // bound at the current s
        const double t = tgt(j, side);
        return (hs_ >= 1.0 || !isfinite(t)) ? t : t + (1.0 - hs_) * ((side > 0 ? hi_[j] : lo_[j]) - t);
    }
    double dbds(const StateArc& s) const { return tgt(s.chain, s.side) - (s.side > 0 ? hi_[s.chain] : lo_[s.chain]); }
    void start(const double* y);
    void predict(StateResult& r, double h);
    void stalled(StateResult& r, double s);
    void drop(int q, double* y, unsigned* gone);
    bool junctions(double* y, int max_its, int* outer);
    double scan(const double* c, const double* y, int j, int side, double tw, double* twout, double* ta, double* tb) const;
    double check(const double* y, int* wc, int* ws, double* ta, double* tb, double* tw, double* psign, double* ppmp) const;
    void fill(StateResult& r) const;
    void vpoly(const uint16_t* S, const double* tau, int n, int a, const double* c, const double* y,
               const StateArc* sa, int nsa, double V[MAXM][4]) const;
};

}  // namespace lqocp

#endif
