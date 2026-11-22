// Indirect motion-primitive solver for fixed-time transfers of integrator chains with a
// polytopic input constraint lo <= M v <= hi. Port of develop/python/lqocp.py.
//
// Costs:
//   energy  : J = 1/2 int |v|^2, fixed endpoint (Q = 0, R = I, no Qf)
//   general : J = 1/2 int (s'Qs + v'Rv) + 1/2 (s(T)-xf)'Qf(s(T)-xf)  (or fixed endpoint)
// Methods:
//   METHOD_NEWTON (default): energy -> Newton on the 12 polynomial coefficients, exact cubic
//                  events, line search on the concave dual; general -> stable-basis motion
//                  primitives, Newton on the junction times with exact derivatives and exact
//                  collapse events, event-driven bound-scale homotopy (NOTES.md section 9).
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
// Storage precision of the junction Jacobian and sensitivities of the general-cost solver
// (-DLQOCP_FLOAT: float). Arithmetic, relations and arc coefficients are always double: TINY's
// R (cond 2e6) amplifies float round-off in the costate beyond the junction tolerance.
#ifdef LQOCP_FLOAT
typedef float real;
#else
typedef double real;
#endif

constexpr int MAXM = 4;              // inputs / chains
constexpr int MAXN = 12;             // states = sum of chain orders
constexpr int MAXK = 4;              // max chain order (4-pt Gauss-Legendre exact for 2K-2 <= 7)
constexpr int MAXARC = 64;           // max switch times printed by the CLI
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
constexpr int MAX_TAU_TOTAL = 500;   // total junction-time Newton iterations (tau method)
constexpr int MAX_PDAS = 30;         // PDAS iterations per continuation step
constexpr int MAX_CONT = 400;        // continuation steps
constexpr int N2 = 2 * MAXN;         // Hamiltonian state (s, lambda)
constexpr int GN = 24;               // Taylor order per substep (event detection, general cost)
constexpr int MAX_TAU_NEWTON = 8;    // Newton iterations on the junction times per corrector
constexpr double KAP_PLAIN = 3.0;    // violation ratio up to which kappa0 -> 1 is tried in one step
constexpr int MAX_GEN_ITS = 1000;    // total Newton iterations, general cost (all homotopy steps)
constexpr double DL_MIN = 1e-3;
constexpr int NPOLY = 32;            // max terms of an arc's centre polynomial (general cost)      // homotopy stall (feasible runs use steps >= 0.03 in log kappa)

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
    int narcs;
    Arc arcs[MAXARC_E];
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
    Problem p_;
    int n_, m_, off_[MAXM];
    double rows_[2 * MAXM][MAXM], rhs_[2 * MAXM], rownorm2_[2 * MAXM];
    bool finite_[2 * MAXM];
    uint16_t sets_[MAXSETS];
    int nsets_;
    double free_[MAXN];
    double rhs_scale_;
    double R_;
    bool identity_;
    bool general_;
    double Lm_[MAXN * MAXN], Linv_[MAXN * MAXN];   // energy: c = Lm nu
    EnergyWork ew_;

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
};

// =====================================================================================
// General LQ cost (NOTES.md sections 9 and 10): stable-basis motion primitives, Newton on the
// junction times, bound-scale homotopy. Memory is sized by compile-time constants:
//   LQOCP_NA  max arcs; LQOCP_NCK stored forward relations (fewer -> recomputed, O(N^2) time);
//   LQOCP_NB  junction-sensitivity columns per sweep (fewer -> more sweeps).
// The per-active-set bases come from a BasisFn: a flash table made offline (embedded) or
// BasisBuilder (lqocp_basis.cpp, computes them on demand; host).
// =====================================================================================
#ifndef LQOCP_NA
#define LQOCP_NA 48
#endif
#ifndef LQOCP_NCK
#define LQOCP_NCK (LQOCP_NA + 1)
#endif
#ifndef LQOCP_NB
#define LQOCP_NB (LQOCP_NA - 1)
#endif
constexpr int NA = LQOCP_NA;         // max arcs of a general-cost trajectory
constexpr int NJ = NA - 1;
constexpr int NR = NA;               // max junction residual rows
constexpr int NCK = LQOCP_NCK;
constexpr int NB = LQOCP_NB;
constexpr int NCOL = N2 + 1 + NB;    // relation row: z | rhs | sensitivities

// Bound-independent primitive of one active set S: H_S = W diag(Ls, Tc, Lu) W^-1, where the
// stable/unstable parts are modal (1x1 real or 2x2 [[re, im], [-im, re]] blocks) and Tc is the
// centre block (|Re eig| <= 1/T, nearly nilpotent). H_S depends only on WHICH rows are
// active, not on the side, so one basis serves S and its mirror: key basis_key(S). Arrays may
// live in flash.
inline uint16_t basis_key(uint16_t S) { return (uint16_t)((S | (S >> 1)) & 0x5555u); }
struct SetBasis {
    uint16_t S;                      // basis_key of the sets it serves
    uint8_t ns, nc;                  // nu = 2n - ns - nc
    double tcn;                      // ||Tc||_inf
    double tcr;                      // ||Tc^24||_inf^(1/24) (Taylor radius of e^{Tc L})
    const double* W;                 // 2n x 2n, row-major
    const double* Wi;                // W^-1
    const double* re;                // 2n modal real parts (stable / unstable columns)
    const double* im;                // +b, -b on the two columns of a complex pair, else 0
    const double* Tc;                // nc x nc
    const double* cs;                // column norms of W (GResult.x stores x * cs)
};
typedef const SetBasis* (*BasisFn)(void* ctx, uint16_t S);

struct GResult {
    int narcs;
    uint16_t S[NA];                  // active set of each arc
    double t1[NA];                   // arc end times (t1[narcs-1] = T)
    double x[NA][N2];                // arc coefficients (a, c, b) in the set's basis, times cs
    double cost;
    double residual;                 // terminal condition residual
    double pmp_violation;
    int iterations;                  // Newton iterations on the junction times (all homotopy steps)
    int outer_iterations;            // homotopy steps
    int degenerate_events;
    bool converged;
    const char* failure;             // short reason when not converged (else nullptr)
};

class GSolver {
public:
    // p must stay alive (it is referenced, not copied). R = Lc Lc^T must be SPD.
    bool setup(const Problem& p, BasisFn fn, void* ctx);
    bool solve(GResult& r);
    // MPC warm start from the previous solution in r (x0 updated in the Problem; junction times
    // shifted by -dt); at most max_its Newton iterations (then r is the current iterate)
    bool solve_warm(GResult& r, double dt, int max_its);
    void set_x0(const double* x0) { for (int i = 0; i < n_; ++i) x0_[i] = x0[i]; }   // (setup copies p.x0)
    void eval_u(const GResult& r, double t, double u[MAXM]) const;
    void eval_x(const GResult& r, double t, double x[MAXN]) const;
    void mean_u(const GResult& r, double t0, double t1, double u[MAXM]) const;   // mean u on [t0, t1]
    int switch_times(const GResult& r, double* out, int maxout) const;
    // for basis builders
    int n2() const { return 2 * n_; }
    int nsets() const { return nsets_; }
    uint16_t set(int i) const;
    bool hmat(uint16_t S, double* H) const;   // homogeneous H_S (2n x 2n, row-major)
    double horizon() const { return p_->T; }

private:
    struct Arc {                     // per-arc data derived on the fly
        const SetBasis* b;
        uint16_t S;
        double t0, L;
        double P[MAXM][MAXM], q[MAXM];
        double yp[N2];               // particular part (modal); centre entries: forcing
        const double* x;
    };
    struct ArcFn {                   // wt on one arc as exponential-polynomials (NOTES.md 11)
        const SetBasis* b;
        double L;
        int np;                      // centre polynomial terms
        double amp[MAXM][N2];        // modal amplitudes (a pair: P, Q of A = P - iQ)
        double poly[MAXM][NPOLY];
        double c[MAXM];
    };
    struct RowFn {                   // one switching row g0 + G wt on the arc of F
        const ArcFn* F;
        double G[MAXM], c0, scale;
        double u(int i) const { double s = 0.0; for (int j = 0; j < MAXM; ++j) s += G[j] * F->amp[j][i]; return s; }
        double p(int k) const { double s = 0.0; for (int j = 0; j < MAXM; ++j) s += G[j] * F->poly[j][k]; return s; }
    };
    const Problem* p_;
    BasisFn bfn_;
    void* bctx_;
    int n_, m_, off_[MAXM], end_[MAXM];
    double x0_[MAXN];
    double rows_[2 * MAXM][MAXM], rhs_[2 * MAXM], rhs0_[2 * MAXM], rownorm2_[2 * MAXM];
    bool finite_[2 * MAXM];
    int nsets_;
    double R_;
    double LiT_[MAXM][MAXM];         // Lc^-T: B~ = B Lc^-T has row end_[j] = LiT_[j]
    double Lc_[MAXM][MAXM];
    bool capfail_;
    int budget_;                     // Newton iterations left (iteration cap)
    bool cert_;                      // the BVP solution in place was certified by the walk
    double wthr_;                    // largest switching-row level the walk certified                   // an arc/row capacity (NA, NR) was exceeded
    GResult* R_out;
    // homotopy / structure
    int an_, hn_, wn_;
    uint16_t aseq_[NA], hseq_[NA], wseq_[NA];
    double atau_[NJ], htau_[NJ], wtau_[NA];
    double hkap_;
    double zT_[N2];
    int events_;
    // junction residual rows
    int nr_;
    int8_t jq_[NR], jk_[NR];
    // linear algebra work
    double rel_[MAXN][NCOL];         // carried relation (12 rows)
    // stored relations at checkpoints 1..NCK-1 (checkpoint 0 is the initial relation s = x0)
    double ck_[NCK > 1 ? NCK - 1 : 1][NCK > 1 ? MAXN : 1][NCK > 1 ? N2 + 1 + NB : 1];
    real Z_[N2][NB];                 // backward sensitivities
    double Dj_[NB][MAXM];            // jumps of the sensitivity block (chain-end entries)
    mutable Arc ac_;                 // scratch arc (sweep step, substep loop, zstart)
    real J_[NR][NJ + 1];             // junction Jacobian | residual

    void wt_of(const double* lam, double* wt) const;
    void hmul(const double P[MAXM][MAXM], const double* q, const double* z, double* out) const;
    bool affine(uint16_t S, double P[MAXM][MAXM], double q[MAXM]) const;
    bool valid_set(uint16_t S) const;
    template <class F> bool for_sets(F f) const;
    int gphi(uint16_t S, int keys[2 * MAXM], double G[2 * MAXM][MAXM], double g0[2 * MAXM]) const;
    bool rowphi(const uint16_t* seq, int q, double G[MAXM], double* g0) const;
    bool arc(int k, const uint16_t* seq, int N, const double* tau, Arc& a) const;
    __attribute__((noinline)) void zat(const Arc& a, double sig, double* z) const;
    void zstart(const uint16_t* seq, int N, const double* tau, int k, double* z) const;
    void cexp(const Arc& a, double L, double* y, bool row, bool aff) const;
    bool chan(const Arc& a, ArcFn& F) const;
    void rowfn(const ArcFn& F, const double* G, double g0, RowFn& f) const;
    void feval(const RowFn& f, double sig, int nd, double* d) const;
    __attribute__((noinline)) void ptaylor(const RowFn& f, double sig, int nd, double* d) const;
    double node(const RowFn& f, double lo, double hi, double* d, double* M2) const;
    double froot(const RowFn& f, double lo, double hi) const;
    double bb(const RowFn& f, double a, double b, double thr, double atol, bool first, double* root,
              bool* simple) const;
    double hloc(const SetBasis& b) const;
    void wint(const ArcFn& F, double sa, double sb, double* I) const;
    void rtaylor(const ArcFn& F, const double* G, double g0, double sig, double* cs) const;
    void gscale(double kap);
    bool glex_ok(uint16_t S, const ArcFn& F, double sig, double h, double* tol) const;
    bool glex_set(const ArcFn& F, double sig, double h, uint16_t guess, uint16_t* out, double* tol) const;
    __attribute__((noinline)) double lead(const ArcFn& F, const double* G, double g0, double sig, double h,
                                          double* tol, int* ilead) const;
    __attribute__((noinline)) int events(const ArcFn& F, uint16_t S, double sig, double h, double* rootk,
                                         bool* simplek, int* keyk, double* s1, const double* lexed);
    bool gwalk(const uint16_t* seq, int N, const double* tau, int* ndeg);
    double gverify(const uint16_t* seq, int N, const double* tau);
    __attribute__((noinline)) double gends(const uint16_t* seq, int N, const double* tau) const;
    double gcost(const uint16_t* seq, int N, const double* tau, double ends);
    double gmax_ratio(const uint16_t* seq, int N, const double* tau);
    void init_rel(int nb);
    bool fstep(const Arc& a, int nb);
    void jump(const uint16_t* seq, int N, const double* tau, int j, double* D) const;
    void inject(int k, int b0, int nb);
    bool advance(const uint16_t* seq, int N, const double* tau, int k, int b0, int nb);
    bool forward(const uint16_t* seq, int N, const double* tau, int b0, int nb, int* stride);
    bool load(const uint16_t* seq, int N, const double* tau, int k, int stride, int b0, int nb);
    __attribute__((noinline)) bool terminal(int nb, double* zT);
    __attribute__((noinline)) bool recover(const Arc& a, int nb, bool aff, double* xout, double* zb);
    bool bvp(const uint16_t* seq, int N, const double* tau);
    int jrows(const uint16_t* seq, int N);
    double resid(const uint16_t* seq, int N, const double* tau, double* r) const;
    bool sens(const uint16_t* seq, int N, const double* tau);
    __attribute__((noinline)) double explicit_term(const uint16_t* seq, int N, const double* tau, int q) const;
    void collapse(uint16_t* seq, int* N, double* tau, int j) const;
    bool newton_tau(uint16_t* seq, int* N, double* tau, int* its);
    bool propose(const uint16_t* seq, int N, const double* tau, int* ndeg);
    bool correct(int* its, int* outer, int* ndeg, int rounds);
    void begin(GResult& R, int max_its);
    bool cold(GResult& R, int max_its);
    bool finish(GResult& R, bool conv, const char* why, int its, int nsteps, int ndeg);
    double zmax(const uint16_t* seq, int N, const double* tau) const;
};

// Host helper: computes SetBasis data on demand (sign-function dichotomy + modal blocks).
class BasisBuilder {
public:
    void attach(const GSolver* g) { g_ = g; n_ = 0; }
    static const SetBasis* fn(void* ctx, uint16_t S) { return static_cast<BasisBuilder*>(ctx)->get(S); }
    const SetBasis* get(uint16_t S);
    int count() const { return n_; }
    const SetBasis& at(int i) const { return sb_[i]; }

private:
    const GSolver* g_ = nullptr;
    int n_ = 0;
    SetBasis sb_[MAXSETS];
    double W_[MAXSETS][N2 * N2], Wi_[MAXSETS][N2 * N2], re_[MAXSETS][N2], im_[MAXSETS][N2],
        Tc_[MAXSETS][N2 * N2], cs_[MAXSETS][N2];
    bool build(uint16_t S, int slot);
};

}  // namespace lqocp

#endif
