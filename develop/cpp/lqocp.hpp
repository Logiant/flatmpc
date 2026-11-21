// Indirect motion-primitive solver for fixed-time transfers of integrator chains with a
// polytopic input constraint lo <= M v <= hi. Port of develop/python/lqocp.py.
//
// Costs:
//   energy  : J = 1/2 int |v|^2, fixed endpoint (Q = 0, R = I, no Qf)
//   general : J = 1/2 int (s'Qs + v'Rv) + 1/2 (s(T)-xf)'Qf(s(T)-xf)  (or fixed endpoint)
// Methods:
//   METHOD_NEWTON (default): energy -> Newton on the 12 polynomial coefficients, exact cubic
//                  events, line search on the concave dual; general -> semismooth Newton
//                  (primal-dual active set) with a Riccati-sweep linear BVP solve and grid-free
//                  Taylor/Bernstein event detection on exponential-polynomial arcs.
//   METHOD_TAU    : junction-time Gauss-Newton, sequence read off a time grid (energy only).
//
// Embedded-friendly core: fixed-size arrays only, no heap allocation, no exceptions, no STL.
// Sizes are reported in NOTES.md. Declare Solver/Result static/global on a microcontroller.
#ifndef LQOCP_HPP
#define LQOCP_HPP

#include <stdint.h>

namespace lqocp {

// Storage precision of the large arrays (bases, carried relations, substep states).
// Arithmetic is always double. Build with -DLQOCP_FLOAT for float storage.
#ifdef LQOCP_FLOAT
typedef float real;
#else
typedef double real;
#endif

constexpr int MAXM = 4;              // inputs / chains
constexpr int MAXN = 12;             // states = sum of chain orders
constexpr int MAXK = 4;              // max chain order (4-pt Gauss-Legendre exact for 2K-2 <= 7)
constexpr int MAXARC = 64;           // max arcs of any trajectory
constexpr int MAXJ = MAXARC - 1;     // max junctions (switch times)
constexpr int MAXARC_E = 32;         // energy-cost buffers (tau / energy newton)
constexpr int MAXJ_E = MAXARC_E - 1;
constexpr int MAXJ_TAU = 16;         // max junctions handled by the tau method
constexpr int MAXRES = 2 * MAXJ_TAU;
constexpr int MAXTOUCH = 16;         // recorded tangential touch points (energy newton)
constexpr int NGRID = 2000;          // grid used by the tau method only
constexpr int MAXSETS = 81;          // 3^MAXM candidate active sets
constexpr int MAX_OUTER = 50;
constexpr int MAX_NEWTON = 50;
constexpr int MAX_PDAS = 30;         // PDAS iterations per continuation step
constexpr int MAX_CONT = 400;        // continuation steps
// general-cost mode
constexpr int GZ = 2 * MAXN + 1;     // augmented Hamiltonian state (s, lambda, 1)
constexpr int GN = 24;               // Taylor order per substep
constexpr int MAXSUB = 192;          // max substeps over [0, T] (Taylor models)
constexpr int MAXGSET = 16;          // cached Taylor data per active set
constexpr int N2 = 2 * MAXN;         // Hamiltonian state (s, lambda)
constexpr int NC_ROWS = MAXN;        // rows of the carried relation
constexpr int MAXSB = 8;             // cached stable-basis primitives (LRU)
constexpr int MAXARC_G = 24;         // max arcs of a general-cost trajectory

enum Method { METHOD_TAU = 0, METHOD_NEWTON = 1 };

struct Problem {
    int m;                           // number of chains / inputs
    int orders[MAXM];                // chain orders (contiguous Brunovsky ordering)
    double T;
    double x0[MAXN], xf[MAXN];
    int has_bounds;                  // 0: unconstrained
    double M[MAXM][MAXM];            // u = M v
    double lo[MAXM], hi[MAXM];       // +-HUGE_VAL disables a side
    int has_cost;                    // 0: energy cost (Q = 0, R = I, fixed endpoint)
    double Q[MAXN][MAXN];
    double R[MAXM][MAXM];
    int has_Qf;                      // 0: fixed endpoint s(T) = xf
    double Qf[MAXN][MAXN];
};

struct Arc {
    double t0, t1;
    uint16_t S;                      // active signed rows: bit 2r = upper of row r, 2r+1 = lower
    double P[MAXM][MAXM];            // energy: v = P w + q;  general: vt = P wt + q (R-scaled)
    double q[MAXM];
};

struct Result {
    int method;
    bool general;                    // general-cost path used
    int narcs;
    Arc arcs[MAXARC];
    double c[MAXN];                  // energy: coefficients of the switching polynomial w(t)
    double cost;
    int iterations;                  // tau: BC solves + Newton its; newton: Newton iterations
    int newton_iterations;
    int outer_iterations;
    double residual;                 // tau: junction residual; newton: |F|_inf / terminal residual
    double pmp_violation;            // max violation of the optimality conditions found
    int backtracks;
    int degenerate_events;
    int ntouch;
    double touch[MAXTOUCH];
    bool box_fast_path;
    bool converged;
    // general mode: substeps (arc a is split into nsub[a] equal substeps); z0 = (s, lambda)
    int gnsub;
    int nsub[MAXARC];
    real gz0[MAXSUB][2 * MAXN];
};

class Solver {
public:
    bool setup(const Problem& p);
    // Returns r.converged. METHOD_TAU with a general cost returns false (unsupported).
    bool solve(Result& r, int method = METHOD_NEWTON);
    void eval_u(const Result& r, double t, double u[MAXM]) const;
    void eval_x(const Result& r, double t, double x[MAXN]) const;
    int switch_times(const Result& r, double* out, int maxout) const;
    bool is_general() const { return general_; }

private:
    struct Eval {                    // energy newton: evaluation of one iterate c
        int narcs;
        Arc arcs[MAXARC_E];
        double G[MAXN * MAXN];
        double F[MAXN];
        double scale, D;
        int ndeg, ntouch;
        double touch[MAXTOUCH];
    };
    struct EnergyWork {
        int nseq;
        uint16_t seq[MAXARC_E];
        double tau[MAXJ_E];
        Arc arcs[MAXARC_E];
        int nseq2;
        uint16_t seq2[MAXARC_E];
        double tau2[MAXJ_E];
        Arc trial[MAXARC_E];
        Eval ev[2];
        uint16_t wseq[MAXARC_E];
        double wtau[MAXJ_E];
    };
    struct GSet {
        uint16_t S;
        double P[MAXM][MAXM], q[MAXM];
        double nrmN, h;
    };
    // Stable-basis primitive of one active set (bound-independent part cacheable offline):
    // H_S = W blockdiag(Ts, Tc, Tu) W^-1, columns of W = [stable | centre | unstable] bases.
    struct SBasis {
        uint16_t S;
        int16_t ns, nc, nu;
        uint32_t stamp;              // LRU
        int part_ok;                 // particular part valid for the current bound scale
        real W[N2 * N2];
        real Tp[N2 * N2];            // packed blocks Ts (ns^2), Tc (nc^2), Tu (nu^2)
        double yps[N2], ypu[N2], gc[N2];   // particular solution (depends on the bounds)
    };
    struct GenWork {
        GSet sets[MAXGSET];          // Taylor data (P, q, ||H^24||, substep h) per active set
        int nsets;
        SBasis sb[MAXSB];
        int nsb;
        uint32_t clock;
        double Z[GZ][GN];            // Taylor coefficients of one substep
        double M1[GZ * GZ], M2[GZ * GZ], M3[GZ * GZ], M4[GZ * GZ], M5[GZ * GZ];  // scratch
        double Ms[GZ * GZ], Me[GZ * GZ];   // maps of the current arc (also expm scratch)
        double ps[N2], pe[N2];
        real C[MAXARC_G + 1][NC_ROWS * N2];   // carried left relation C_k z(tau_k) = d_k
        real d[MAXARC_G + 1][NC_ROWS];
        real dd[MAXARC_G + 1][NC_ROWS][MAXARC_G];   // sensitivity right-hand sides
        double Dl[N2][MAXARC_G];     // vector-field jumps at the junctions
        int jq[2 * MAXARC_G];        // junction residual rows
        double jG[2 * MAXARC_G][MAXM], jg0[2 * MAXARC_G];
        double J[2 * MAXARC_G * MAXARC_G];
        double x[MAXARC_G][N2];      // arc coefficients (a, c, b)
        double zj[MAXARC_G + 1][N2]; // z at the junctions (index k = tau_k, k = N: T)
        int nseq;
        uint16_t seq[MAXARC];
        double tau[MAXJ];
        int an;                      // last accepted arc structure (continuation)
        uint16_t aseq[MAXARC];
        double atau[MAXJ];
    };

    Problem p_;
    int n_, m_, off_[MAXM];
    double rows_[2 * MAXM][MAXM], rhs_[2 * MAXM], rownorm2_[2 * MAXM];
    bool finite_[2 * MAXM];
    uint16_t sets_[MAXSETS];
    int nsets_;
    double free_[MAXN];
    double rhs_scale_;
    double rhs0_[2 * MAXM];          // general: unscaled bounds (continuation)
    double R_;
    bool identity_;
    bool general_;
    double Lm_[MAXN * MAXN], Linv_[MAXN * MAXN];   // energy: c = Lm nu
    double Lc_[MAXM][MAXM];          // general: R = Lc Lc^T (Cholesky)
    double Bt_[MAXN][MAXM];          // general: B Lc^{-T}
    union {
        EnergyWork ew_;
        GenWork gw_;
    };

    // shared / energy
    void w_of(const double* c, double t, double* w) const;
    void phib(double dt, double K[MAXN][MAXM]) const;
    void phi_x0(double dt, double* s) const;
    bool affine(uint16_t S, double P[MAXM][MAXM], double q[MAXM]) const;
    bool project(const double* w, uint16_t guess, double* v, uint16_t* Sout) const;
    bool try_set(uint16_t S, const double* w, double* v) const;
    bool build_arcs(const uint16_t* seq, int nseq, const double* tau, Arc* arcs) const;
    void set_times(Arc* arcs, int nseq, const double* tau) const;
    void assemble(const Arc* arcs, int narcs, double* G, double* h) const;
    bool solve_c(const Arc* arcs, int narcs, double* c) const;
    bool detect(const double* c, uint16_t* seq, int* nseq, double* tau) const;
    bool residual(Arc* arcs, int nseq, const double* tau, double* res, int* nres, double* c) const;
    bool newton(Arc* arcs, int nseq, double* tau, double* c, int* its, double* resmax) const;
    double pmp_violation(const Arc* arcs, int narcs, const double* c) const;
    double cost(const Arc* arcs, int narcs, const double* c) const;
    void v_on(const Arc& a, const double* c, double t, double* v) const;
    bool solve_tau(Result& r);
    bool detect_exact(const double* c);
    bool solve_newton(Result& r);
    void row_polys(uint16_t S, const double* c, double phi[2 * MAXM][4], bool has[2 * MAXM]) const;
    bool lex_ok(uint16_t S, const double* c, double t) const;
    bool lex_set(const double* c, double t, uint16_t guess, uint16_t* out) const;
    bool valid_set(uint16_t S) const;
    bool walk(const double* c, Eval& e);
    bool walk_box(const double* c, Eval& e);
    bool evaluate(const double* c, Eval& e);
    double dual(const Arc* arcs, int narcs, const double* c) const;
    double verify(const Arc* arcs, int narcs, const double* c) const;

    // general cost (stable-basis motion primitives)
    bool gsetup(double Mrow[MAXM][MAXM]);
    int gset(uint16_t S);
    void hmul(const GSet& g, const double* z, double* out) const;
    void taylor(const GSet& g, const double* z0, double Z[GZ][GN]) const;
    double eps_sub(const GSet& g, double h, double Z[GZ][GN]) const;
    int gphi(uint16_t S, int keys[2 * MAXM], double G[2 * MAXM][MAXM], double g0[2 * MAXM]) const;
    bool glex_ok(uint16_t S, const double W[MAXM][GN], double sig, double h) const;
    bool glex_set(const double W[MAXM][GN], double sig, double h, uint16_t guess, uint16_t* out) const;
    bool gwalk(const Result& R, int* ndeg);
    double gverify(const Result& R);
    double gcost(const Result& R, const double* zT);
    void gz_at(const Result& R, double t, double z[GZ], int* arc) const;
    void gscale(double kap);
    double gmax_ratio(const Result& R);
    int sbasis(uint16_t S);
    bool sparticular(SBasis& b, uint16_t S);
    bool arc_maps(int bi, double L, bool deriv, double* Ds, double* De, double* dpe);
    bool bvp(const uint16_t* seq, int N, const double* tau, double* zT);
    bool sens(const uint16_t* seq, int N, const double* tau, int nr, double* J);
    int jrows(const uint16_t* seq, int N, int* jq, double G[][MAXM], double* g0) const;
    bool newton_tau(const uint16_t* seq, int N, double* tau, int* its, double* zT);
    bool make_subs(Result& R, const uint16_t* seq, int N, const double* tau);
    int sloop(Result& R, double* zT, int maxit, int* its, int* outer, int* ndeg);
    bool solve_general(Result& R);
};

}  // namespace lqocp

#endif
