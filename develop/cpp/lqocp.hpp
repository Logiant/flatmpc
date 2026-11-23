// Indirect motion-primitive solver for fixed-time transfers of integrator chains with a
// polytopic input constraint lo <= M v <= hi. Port of develop/python/lqocp.py.
//
// Costs:
//   energy  : Q = 0: J = 1/2 int v'Rv [+ 1/2 (s(T)-xf)'Qf(s(T)-xf)], else fixed endpoint
//             (polynomial costate, NOTES.md sections 7 and 13)
//   general : J = 1/2 int (s'Qs + v'Rv) + 1/2 (s(T)-xf)'Qf(s(T)-xf)  (or fixed endpoint)
// Methods:
//   METHOD_NEWTON (default): energy -> Newton on the 12 polynomial coefficients, exact cubic
//                  events, line search on the concave dual; general -> stable-basis motion
//                  primitives, Newton on the junction times with exact derivatives and exact
//                  collapse events, event-driven bound-scale homotopy (NOTES.md section 9);
//                  closed-form walk/cost and the MPC warm start GSolver::solve_warm (section 11).
//   METHOD_TAU    : junction-time Gauss-Newton, sequence from the exact event walk (energy only).
//
// Embedded-friendly core: fixed-size arrays only, no heap allocation, no exceptions, no STL.
// Sizes are reported in NOTES.md. Declare Solver/Result static/global on a microcontroller.
#ifndef LQOCP_HPP
#define LQOCP_HPP

#include <stdint.h>

namespace lqocp {

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
constexpr int MAXK = 4;              // max chain order (energy cost: 4-pt Gauss-Legendre is exact, |v|^2 has degree 2K-2)
constexpr int MAXARC = 64;           // max switch times printed by the CLI
constexpr int MAXJ = MAXARC - 1;     // max junctions (switch times)
#ifndef LQOCP_MAXARC_E
#define LQOCP_MAXARC_E 32
#endif
constexpr int MAXARC_E = LQOCP_MAXARC_E;   // energy cost: max arcs (overflow -> Result.failure)
constexpr int MAXJ_E = MAXARC_E - 1;
static_assert(MAXARC_E >= 2, "LQOCP_MAXARC_E must be >= 2");
constexpr int MAXJ_TAU = 16;         // max junctions handled by the tau method
constexpr int MAXRES = 2 * MAXJ_TAU;
constexpr int MAXSETS = 81;          // 3^MAXM candidate active sets
constexpr int MAX_OUTER = 50;
constexpr int MAX_NEWTON = 50;
constexpr int MAX_TAU_TOTAL = 500;   // total junction-time Newton iterations (tau method)
constexpr int N2 = 2 * MAXN;         // Hamiltonian state (s, lambda)
constexpr int GN = 24;               // Taylor order of the sign tests at an event (general cost)
constexpr int MAX_TAU_NEWTON = 8;    // Newton iterations on the junction times per corrector
constexpr double KAP_PLAIN = 3.0;    // violation ratio up to which kappa0 -> 1 is tried in one step
constexpr int MAX_GEN_ITS = 1000;    // total Newton iterations, general cost (all homotopy steps)
constexpr double DL_MIN = 1e-3;      // homotopy stall (feasible runs use steps >= 0.03 in log kappa)
constexpr int NPOLY = 2 * MAXK + 1;  // terms of an arc's centre polynomial: nilpotency index + 1
#ifndef LQOCP_NSD
#define LQOCP_NSD 8
#endif
constexpr int NSD = LQOCP_NSD;       // max slow modal dimensions coupled into the centre (general cost;
                                     // TINY at T = 0.45 needs 4, R1 8; 2 x 4 x 10 doubles of stack each)

enum Method { METHOD_TAU = 0, METHOD_NEWTON = 1 };

struct Problem {
    int m;                           // number of chains / inputs
    int orders[MAXM];                // chain orders (contiguous Brunovsky ordering)
    double T;
    double x0[MAXN], xf[MAXN];
    int has_bounds;                  // 0: unconstrained
    double M[MAXM][MAXM];            // u = M v
    double lo[MAXM], hi[MAXM];       // +-HUGE_VAL disables a side
    int has_cost;                    // 0: energy cost (Q = 0, R = I, fixed endpoint); Q = 0 is always energy
    double Q[MAXN][MAXN];
    double R[MAXM][MAXM];
    int has_Qf;                      // 0: fixed endpoint s(T) = xf
    double Qf[MAXN][MAXN];
};

// Energy-cost plan: v(t) on arc a is the projection of w(t) (from c) on the face of set S[a]
// (v = P w + q with P, q = affine(S), recomputed when needed).
struct Result {
    int method;
    int narcs;
    uint16_t S[MAXARC_E];            // active signed rows: bit 2r = upper of row r, 2r+1 = lower
    double tau[MAXJ_E];              // switch times: arc a is [tau[a-1], tau[a]] (0 and T at the ends)
    double c[MAXN];                  // energy: coefficients of the switching polynomial w(t)
    double mu[MAXN];                 // energy with Qf (newton): dual iterate, nu = Cf^T mu, c = Lm nu
    double cost;
    int iterations;                  // tau: BC solves + Newton its; newton: Newton iterations
    int newton_iterations;
    int outer_iterations;
    double residual;                 // tau: junction residual; newton: |F|_inf / terminal residual
    double pmp_violation;            // max violation of the optimality conditions found
    int backtracks;
    int degenerate_events;
    bool box_fast_path;
    bool converged;
    const char* failure;             // short reason when not converged (else nullptr)
};

class Solver {
public:
    // p must stay alive (it is referenced, not copied; it may live in flash). Problem.x0 is copied.
    bool setup(const Problem& p);
    // Returns r.converged. A general cost returns false (use GSolver).
    bool solve(Result& r, int method = METHOD_NEWTON);
    // MPC warm start (newton): from the previous r.c / r.mu, at most max_its Newton iterations
    // (then r is the current iterate; its v = Proj(w) is feasible). Update x0 with set_x0 first.
    bool solve_warm(Result& r, int max_its);
    bool set_x0(const double* x0);   // false if x0 is not finite
    // mean u on [t0, t1] (what a zero-order hold should apply); false: no plan
    bool mean_u(const Result& r, double t0, double t1, double u[MAXM]) const;
    void eval_u(const Result& r, double t, double u[MAXM]) const;
    void eval_x(const Result& r, double t, double x[MAXN]) const;
    int switch_times(const Result& r, double* out, int maxout) const;
    bool is_general() const { return general_; }

private:
    struct Eval {                    // energy newton: evaluation of one iterate c (its ds(T)/dc is in G_)
        int narcs;
        uint16_t S[MAXARC_E];
        double tau[MAXJ_E];
        double F[MAXN];
        double scale, D;
        int ndeg;
    };
    struct EnergyWork {
#ifndef LQOCP_NO_TAU
        int nseq;
        uint16_t seq[MAXARC_E];
        double tau[MAXJ_E];
        int nseq2;
        uint16_t seq2[MAXARC_E];
        double tau2[MAXJ_E];
#endif
        Eval ev[2];
    };
    const Problem* p_;
    double x0_[MAXN];
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
    bool rI_, qf_;                   // R = I; terminal cost (else fixed endpoint)
    double Lc_[MAXM][MAXM], Lci_[MAXM][MAXM];       // R = Lc Lc^T; projection in vt = Lc^T v
    double Lm_[MAXN * MAXN], Linv_[MAXN * MAXN];   // energy: c = Lm nu
    double Cf_[MAXN * MAXN], g_[MAXN];             // Qf = Cf^T Cf; g = Cf (xf - e^{AT} x0)
    EnergyWork ew_;
    mutable double G_[MAXN * MAXN], W_[MAXN * MAXN];  // workspace: ds(T)/dc of ev[cur]; Newton matrix
    bool capfail_;                   // MAXARC_E was exceeded during this solve

    // shared / energy (an arc sequence is S[0..n-1] with switch times tau[0..n-2])
    void w_of(const double* c, double t, double* w) const;
    void phib(double dt, double K[MAXN][MAXM]) const;
    void phi_x0(double dt, double* s) const;
    bool affine(uint16_t S, double P[MAXM][MAXM], double q[MAXM]) const;
    bool pq(uint16_t S, double P[MAXM][MAXM], double q[MAXM]) const;   // v = P w + q on set S
    void span(const double* tau, int n, int a, double* t0, double* t1) const {
        *t0 = a ? tau[a - 1] : 0.0;
        *t1 = (a + 1 < n) ? tau[a] : p_->T;
    }
    bool sets_ok(const uint16_t* S, int n) const;
    void assemble(const uint16_t* S, const double* tau, int n, double* G, double* h) const;
    bool solve_c(const uint16_t* S, const double* tau, int n, double* c) const;
    double cost(const uint16_t* S, const double* tau, int n, const double* c) const;
    double vR(int j, const double* v) const {        // (R v)_j
        double s = 0.0;
        for (int k = 0; k < m_; ++k) s += p_->R[j][k] * v[k];
        return s;
    }
    void v_on(const double P[MAXM][MAXM], const double* q, const double* c, double t, double* v) const;
#ifndef LQOCP_NO_TAU
    bool residual(const uint16_t* S, int nseq, const double* tau, double* res, int* nres, double* c) const;
    bool newton(const uint16_t* S, int nseq, double* tau, double* c, int* its, double* resmax) const;
    bool solve_tau(Result& r);
    bool detect_exact(const double* c);
#endif
    bool start(Result& r, int method, int warm_its);
    bool solve_newton(Result& r, int warm_its);
    void row_polys(uint16_t S, const double* c, double phi[2 * MAXM][4], bool has[2 * MAXM]) const;
    bool lex_ok(uint16_t S, const double* c, double t) const;
    bool lex_set(const double* c, double t, uint16_t guess, uint16_t* out) const;
    bool valid_set(uint16_t S) const;
    bool walk(const double* c, Eval& e);
    bool walk_box(const double* c, Eval& e);
    bool evaluate(const double* c, const double* mu, Eval& e);
    double dual(const uint16_t* S, const double* tau, int n, const double* c, const double* mu) const;
    double verify(const uint16_t* S, const double* tau, int n, const double* c) const;
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
static_assert(NA >= 1 && NA <= 127, "LQOCP_NA: 1..127 (junction indices are int8_t)");
static_assert(NB >= 1 && NCK >= 1, "LQOCP_NB and LQOCP_NCK must be >= 1");

// Bound-independent primitive of one active set S: H_S = W diag(Ls, Tc, Lu) W^-1, where the
// stable/unstable parts are modal (1x1 real or 2x2 [[re, im], [-im, re]] blocks) and Tc is the
// centre block Tc = [[N, X], [0, Ls]]: N (nz x nz) holds the zero eigenvalues of the saturated
// channels, exactly nilpotent (staircase form, strictly block upper triangular, N^ni = 0); Ls holds
// the slow modal blocks whose eigenvectors are too close to N's subspace to split off (re/im of
// their columns are set), coupled to N by X. e^{Tc L} is then exact in closed form: a polynomial
// in N plus the phi-functions of the slow eigenvalues (NOTES.md section 12). H_S depends only on WHICH rows are
// active, not on the side, so one basis serves S and its mirror: key basis_key(S). Arrays may
// live in flash.
inline uint16_t basis_key(uint16_t S) { return (uint16_t)((S | (S >> 1)) & 0x5555u); }
struct SetBasis {
    uint16_t S;                      // basis_key of the sets it serves
    uint8_t ns, nc, nz, ni;          // nu = 2n - ns - nc; nz: size of N; ni: index of N (<= 2 MAXK)
    double tcn;                      // ||Tc||_inf
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
    // (setup copies p.x0) false if x0 is not finite: the next solve then fails
    bool set_x0(const double* x0);
    // eval_u and mean_u (mean u on [t0, t1], what a zero-order hold should apply) saturate u to the
    // bounds lo <= M u <= hi (a capped warm step's iterate can leave them); false: no plan (hold u)
    bool eval_u(const GResult& r, double t, double u[MAXM]) const;
    void eval_x(const GResult& r, double t, double x[MAXN]) const;
    bool mean_u(const GResult& r, double t0, double t1, double u[MAXM]) const;
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
        int m;                       // channels (rows j >= m are not filled)
        int np;                      // centre polynomial terms
        int nsb, sk, nz;             // slow blocks, phi terms per block (ni + 2), size of N
        int sc[NSD];                 // first centre column of each slow block
        double amp[MAXM][N2];        // modal amplitudes (a pair: P, Q of A = P - iQ)
        double poly[MAXM][NPOLY];
        double sl[MAXM][NSD][NPOLY + 1];      // slow block q: sum_k Re[A_k g_k(sig)], A_k = (P, -Q)
                                              // in the slots of its columns sc[q] - nz (+1: pair)
        double c[MAXM];
    };
    struct RowFn {                   // one switching row g0 + G wt on the arc of F
        const ArcFn* F;
        double G[MAXM], c0, scale;
        double u(int i) const { double s = 0.0; for (int j = 0; j < F->m; ++j) s += G[j] * F->amp[j][i]; return s; }
        double p(int k) const { double s = 0.0; for (int j = 0; j < F->m; ++j) s += G[j] * F->poly[j][k]; return s; }
        double s(int q, int k, int c) const {
            if (c && F->b->im[F->b->ns + F->sc[q]] == 0.0) return 0.0;
            double t = 0.0;
            for (int j = 0; j < F->m; ++j) t += G[j] * F->sl[j][F->sc[q] - F->nz + c][k];
            return t;
        }
    };
    const Problem* p_;
    BasisFn bfn_;
    void* bctx_;
    int n_, m_, off_[MAXM], end_[MAXM];
    double x0_[MAXN];
    bool x0ok_;
    double rows_[2 * MAXM][MAXM], rhs_[2 * MAXM], rhs0_[2 * MAXM], rownorm2_[2 * MAXM];
    bool finite_[2 * MAXM];
    int nsets_;
    double R_;
    double LiT_[MAXM][MAXM];         // Lc^-T: B~ = B Lc^-T has row end_[j] = LiT_[j]
    double Lc_[MAXM][MAXM];
    bool capfail_;                   // an arc/row capacity (NA, NR) was exceeded
    int budget_;                     // Newton iterations left (iteration cap)
    bool cert_;                      // the BVP solution in place was certified by the walk
    double wthr_;                    // largest switching-row level the walk certified
    mutable uint16_t bkey_;          // basis key whose BasisFn lookup failed (0xFFFF: none)
    char why_[40];                   // failure text of bfail()
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
    mutable Arc ac_;                 // scratch arc (sweep step, walk, zstart)
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
    void scoef(const RowFn& f, int q, int nder, double (*A)[2]) const;
    double sbound(const RowFn& f, int nder, double t, bool full) const;
    void sadd(const RowFn& f, double sig, int nd, double* d) const;
    __attribute__((noinline)) void ptaylor(const RowFn& f, double sig, int nd, double* d) const;
    double node(const RowFn& f, double lo, double hi, double* d, double* M2) const;
    double froot(const RowFn& f, double lo, double hi) const;
    double pbb(const RowFn& f, double a, double b, double thr, bool first, double* root, bool* simple) const;
    double bb(const RowFn& f, double a, double b, double thr, double atol, bool first, double* root,
              bool* simple) const;
    double hloc(const SetBasis& b) const;
    void wint(const ArcFn& F, double sa, double sb, double* I) const;
    void rtaylor(const ArcFn& F, const double* G, double g0, double sig, double* cs) const;
    void gscale(double kap);
    void clampu(double* u) const;
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
    void init_rel();
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
    const char* bfail();
    bool cold(GResult& R, int max_its);
    bool finish(GResult& R, bool conv, const char* why, int its, int nsteps, int ndeg);
    double zmax(const uint16_t* seq, int N, const double* tau) const;
};

// Host helper: computes SetBasis data on demand (staircase + sign function + modal blocks).
// A key whose build failed is remembered and not rebuilt.
class BasisBuilder {
public:
    void attach(const GSolver* g) { g_ = g; n_ = nbad_ = 0; }
    static const SetBasis* fn(void* ctx, uint16_t S) { return static_cast<BasisBuilder*>(ctx)->get(S); }
    const SetBasis* get(uint16_t S);
    int count() const { return n_; }
    const SetBasis& at(int i) const { return sb_[i]; }

private:
    const GSolver* g_ = nullptr;
    int n_ = 0, nbad_ = 0;
    uint16_t bad_[MAXSETS];
    SetBasis sb_[MAXSETS];
    double W_[MAXSETS][N2 * N2], Wi_[MAXSETS][N2 * N2], re_[MAXSETS][N2], im_[MAXSETS][N2],
        Tc_[MAXSETS][N2 * N2], cs_[MAXSETS][N2];
    bool build(uint16_t S, int slot);
};

}  // namespace lqocp

#endif
