// Tier 2: thrust box + first-order state box, energy cost. See lqocp_state.hpp and NOTES.md 14.
#include "lqocp_state.hpp"

#include <math.h>
#include <stdio.h>
#include <string.h>

namespace lqocp {

namespace pd {   // helpers defined in lqocp.cpp
double fact(int k);
double ipow(double x, int k);
template <class T> bool lu_solve(T* A, int n, int ld, T* B, int nrhs, int ldb);
double pval(const double* b, int len, double s);
void shift(const double* a, int len, double t0, double* b);
double pscale(const double* b, int len, double L);
int strip(const double* b, int len, double L, double tol, double* red, int* kout);
int roots_le2(const double* d0, int len, double* out);
double bisect(const double* r, int len, double a, double b);
struct Cross { int n; double root[4]; bool simple[4]; };
void crossings(const double* red, int len, int k, double L, double tol, Cross& o);
}  // namespace pd
using namespace pd;

namespace {
const double SGX[4] = {-0.86113631159405257522, -0.33998104358485626480,
                       0.33998104358485626480, 0.86113631159405257522};
const double SGW[4] = {0.34785484513745385737, 0.65214515486254614263,
                       0.65214515486254614263, 0.34785484513745385737};
const double EPS_ = 2.220446049250313e-16;
const double TOL_X = 1e-9;           // state violation, relative to 1 + |b|
const double TOL_SIGN = 1e-7;        // eta >= 0 on a boundary arc, relative
const char CH[4] = {'x', 'y', 'z', 'p'};

void span_(const double* tau, int n, int a, double T, double* t0, double* t1) {
    *t0 = a ? tau[a - 1] : 0.0;
    *t1 = (a + 1 < n) ? tau[a] : T;
}
double peval(const double* V, double t) { return ((V[3] * t + V[2]) * t + V[1]) * t + V[0]; }

// real roots of a polynomial (len <= 8) in (0, L), ascending: monotone pieces between the roots of
// the derivative (recursively), one bisection per sign change
int proots(const double* a, int len, double L, double* out) {
    while (len > 1 && a[len - 1] == 0.0) --len;
    if (len <= 1) return 0;
    double d[8], pts[9];
    for (int i = 1; i < len; ++i) d[i - 1] = i * a[i];
    int np = 0;
    pts[np++] = 0.0;
    np += proots(d, len - 1, L, pts + 1);
    pts[np++] = L;
    int nr = 0;
    for (int i = 0; i + 1 < np; ++i) {
        const double fa = pval(a, len, pts[i]), fb = pval(a, len, pts[i + 1]);
        if ((fa < 0 && fb > 0) || (fa > 0 && fb < 0)) out[nr++] = bisect(a, len, pts[i], pts[i + 1]);
        else if (fb == 0.0 && i + 2 < np) out[nr++] = pts[i + 1];
    }
    return nr;
}
}  // namespace

bool StateSolver::setup(const Problem& p, const double* slo, const double* shi) {
    nsa_ = 0;
    if (!base_.setup(p)) { sfail_ = base_.setup_failure(); return false; }
    if (base_.general_) { sfail_ = "tier 2 needs Q = 0 (energy cost)"; return false; }
    sfail_ = nullptr;
    for (int j = 0; j < base_.m_; ++j) {
        const int o = base_.off_[j], k = p.orders[j];
        for (int r = 0; r < k; ++r) {
            const double l = slo[o + r], h = shi[o + r];
            if (isnan(l) || isnan(h) || l == HUGE_VAL || h == -HUGE_VAL) { sfail_ = "state bound NaN or infinite on the wrong side"; return false; }
            if (l > h) { sfail_ = "state bound lo > hi"; return false; }
            xlo_[o + r] = l;
            xhi_[o + r] = h;
        }
        lo_[j] = slo[o + k - 1];
        hi_[j] = shi[o + k - 1];
    }
    return true;
}

void StateSolver::cvec(const double* y, double* c) const {
    const Solver& B = base_;
    const int n = B.n_;
    double nu[MAXN];
    for (int i = 0; i < n; ++i) {
        double s = y[i];
        if (B.qf_) { s = 0.0; for (int j = 0; j <= i; ++j) s += B.Cf_[j * MAXN + i] * y[j]; }
        nu[i] = s;
    }
    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) s += B.Lm_[i * MAXN + j] * nu[j];
        c[i] = s;
    }
}

// w~ = Lc^-1 w on the piece containing tm: w_j = poly(c) + sum of Delta_q (chain j, tm < t1_q)
void StateSolver::wpoly(const double* c, const double* y, const StateArc* sa, int nsa, double tm,
                        double Wc[MAXM][4]) const {
    const Solver& B = base_;
    double wp[MAXM][4];
    for (int l = 0; l < B.m_; ++l)
        for (int p = 0; p < 4; ++p) wp[l][p] = (p < B.p_->orders[l]) ? c[B.off_[l] + p] : 0.0;
    for (int q = 0; q < nsa; ++q)
        if (!sa[q].fix1 && tm < sa[q].t1) wp[sa[q].chain][0] += y[B.n_ + q];
    for (int j = 0; j < B.m_; ++j)
        for (int p = 0; p < 4; ++p) {
            double s = 0.0;
            for (int k = 0; k <= j; ++k) s += B.Lci_[j][k] * wp[k][p];
            Wc[j][p] = s;
        }
}

// Pointwise KKT on thrust set S (bits 0..7) with chains pinned (bits 8+j: v_j = 0):
// v = Pv w + q, Vv = v's coefficients, phi/has the row functions of Solver::row_polys,
// sig[j] the pin multiplier of chain j in w units (R v = w - sig_j e_j - ...). false: singular.
bool StateSolver::kkt(uint16_t S, const double Wc[MAXM][4], double Pv[MAXM][MAXM], double Vv[MAXM][4],
                      double phi[2 * MAXM][4], bool has[2 * MAXM], double sig[MAXM][4]) const {
    const Solver& B = base_;
    const int m = B.m_;
    double rw[MAXM][MAXM], rh[MAXM];
    int key[MAXM], na = 0;
    for (int k = 0; k < 2 * m; ++k)
        if (S & (1u << k)) {
            if (na >= m) return false;
            for (int i = 0; i < m; ++i) rw[na][i] = B.rows_[k][i];
            rh[na] = B.rhs_[k]; key[na++] = k;
        }
    for (int j = 0; j < m; ++j)
        if (S & (1u << (8 + j))) {
            if (na >= m) return false;
            for (int i = 0; i < m; ++i) rw[na][i] = B.Lci_[i][j];
            rh[na] = 0.0; key[na++] = -1 - j;
        }
    const int ld = MAXM + 4;
    double G[MAXM * MAXM], X[MAXM * (MAXM + 4)];
    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < na; ++b) {
            double s = 0.0;
            for (int i = 0; i < m; ++i) s += rw[a][i] * rw[b][i];
            G[a * MAXM + b] = s;
        }
        for (int i = 0; i < m; ++i) X[a * ld + i] = rw[a][i];
        for (int p = 0; p < 4; ++p) {
            double s = (p == 0) ? -rh[a] : 0.0;
            for (int i = 0; i < m; ++i) s += rw[a][i] * Wc[i][p];
            X[a * ld + m + p] = s;
        }
    }
    if (na > 0 && !lu_solve(G, na, MAXM, X, m + 4, ld)) return false;
    double Pt[MAXM][MAXM], Vt[MAXM][4];
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            double s = (i == j) ? 1.0 : 0.0;
            for (int a = 0; a < na; ++a) s -= rw[a][i] * X[a * ld + j];
            Pt[i][j] = s;
        }
        for (int p = 0; p < 4; ++p) {
            double s = Wc[i][p];
            for (int a = 0; a < na; ++a) s -= rw[a][i] * X[a * ld + m + p];
            Vt[i][p] = s;
        }
    }
    for (int i = 0; i < m; ++i) {                    // v = Lc^-T vt, Pv = Lci^T Pt Lci
        for (int p = 0; p < 4; ++p) {
            double s = 0.0;
            for (int k = 0; k < m; ++k) s += B.Lci_[k][i] * Vt[k][p];
            Vv[i][p] = s;
        }
        for (int j = 0; j < m; ++j) {
            double s = 0.0;
            for (int k = 0; k < m; ++k)
                for (int l = 0; l < m; ++l) s += B.Lci_[k][i] * Pt[k][l] * B.Lci_[l][j];
            Pv[i][j] = s;
        }
        for (int p = 0; p < 4; ++p) sig[i][p] = 0.0;
    }
    for (int k = 0; k < 2 * m; ++k) {
        has[k] = B.finite_[k];
        if (!has[k] || (S & (1u << k))) continue;
        for (int p = 0; p < 4; ++p) {
            double s = (p == 0) ? -B.rhs_[k] : 0.0;
            for (int i = 0; i < m; ++i) s += B.rows_[k][i] * Vt[i][p];
            phi[k][p] = s;
        }
    }
    for (int a = 0; a < na; ++a)
        for (int p = 0; p < 4; ++p) {
            if (key[a] >= 0) phi[key[a]][p] = -X[a * ld + m + p] * B.rownorm2_[key[a]];
            else sig[-1 - key[a]][p] = X[a * ld + m + p];
        }
    return true;
}

bool StateSolver::lex(const double Wc[MAXM][4], uint16_t pin, double t, double T, uint16_t guess, uint16_t* out) const {
    const Solver& B = base_;
    double Pv[MAXM][MAXM], Vv[MAXM][4], phi[2 * MAXM][4], sig[MAXM][4], b[4];
    bool has[2 * MAXM];
    for (int i = -1; i < B.nsets_; ++i) {
        const uint16_t cand = (i < 0) ? guess : B.sets_[i];
        if (i < 0 && !B.valid_set(cand)) continue;
        const uint16_t S = (uint16_t)(cand | (pin << 8));
        if (!kkt(S, Wc, Pv, Vv, phi, has, sig)) continue;
        bool ok = true;
        for (int key = 0; key < 2 * B.m_ && ok; ++key) {
            if (!has[key]) continue;
            shift(phi[key], 4, t, b);
            const double tol = 1e-12 * (pscale(b, 4, T) + B.R_);
            for (int q = 0; q < 4; ++q)
                if (fabs(b[q]) * ipow(T, q) > tol) { if (b[q] > 0) ok = false; break; }
        }
        if (ok) { *out = S; return true; }
    }
    return false;
}

// Event walk (Solver::walk) on each piece between boundary-arc junctions
bool StateSolver::walk(const double* c, const double* y, Ev& e) const {
    const Solver& B = base_;
    const double T = B.p_->T;
    double bp[2 * MAXSA + 2];
    int nb = 0;
    bp[nb++] = 0.0;
    for (int q = 0; q < nsa_; ++q) {
        if (!sa_[q].fix1) bp[nb++] = sa_[q].t1;
        if (!sa_[q].fix2) bp[nb++] = sa_[q].t2;
    }
    bp[nb++] = T;
    for (int i = 1; i < nb; ++i)
        for (int k = i; k > 0 && bp[k - 1] > bp[k]; --k) { double x = bp[k]; bp[k] = bp[k - 1]; bp[k - 1] = x; }
    int ns = 0;
    uint16_t S = 0;
    double Wc[MAXM][4], Pv[MAXM][MAXM], Vv[MAXM][4], phi[2 * MAXM][4], sig[MAXM][4];
    bool has[2 * MAXM];
    for (int k = 0; k + 1 < nb; ++k) {
        const double ta = bp[k], tb = bp[k + 1];
        if (!(tb > ta)) continue;
        const double tm = 0.5 * (ta + tb);
        uint16_t pin = 0;
        for (int q = 0; q < nsa_; ++q)
            if (sa_[q].t1 <= tm && tm < sa_[q].t2) pin |= (uint16_t)(1u << sa_[q].chain);
        wpoly(c, y, sa_, nsa_, tm, Wc);
        if (!lex(Wc, pin, ta, tb - ta, (uint16_t)(S & 0xFF), &S)) return false;
        if (ns >= MAXARC_E) return false;
        if (ns) e.tau[ns - 1] = ta;
        e.S[ns++] = S;
        double t = ta;
        int stall = 0;
        for (;;) {
            const double L = tb - t;
            if (!kkt(S, Wc, Pv, Vv, phi, has, sig)) return false;
            double rootk[2 * MAXM], s1 = HUGE_VAL;
            bool simplek[2 * MAXM], evk[2 * MAXM];
            for (int key = 0; key < 2 * B.m_; ++key) {
                evk[key] = false;
                if (!has[key]) continue;
                double b[4], red[4];
                int kk;
                shift(phi[key], 4, t, b);
                const double tol = 1e-12 * (pscale(b, 4, L) + B.R_);
                const int rl = strip(b, 4, L, tol, red, &kk);
                if (rl == 0) continue;
                if (red[0] > 0) { evk[key] = true; rootk[key] = 0.0; simplek[key] = false; s1 = 0.0; continue; }
                Cross cr;
                crossings(red, rl, kk, L, tol, cr);
                if (cr.n) {
                    evk[key] = true; rootk[key] = cr.root[0]; simplek[key] = cr.simple[0];
                    if (cr.root[0] < s1) s1 = cr.root[0];
                }
            }
            if (s1 == HUGE_VAL || t + s1 >= tb - 1e-14 * T) break;
            int ngroup = 0, gkey = -1;
            bool gsimple = false;
            uint16_t gmask = 0;
            for (int key = 0; key < 2 * B.m_; ++key)
                if (evk[key] && rootk[key] <= s1 + 1e-12 * T) {
                    ++ngroup; gkey = key; gsimple = simplek[key]; gmask |= (uint16_t)(1u << key);
                }
            t += s1;
            uint16_t Snew = 0xFFFFu;
            if (ngroup == 1 && gsimple) {
                const uint16_t cand = (uint16_t)(S ^ (1u << gkey));
                if (B.valid_set(cand) && kkt(cand, Wc, Pv, Vv, phi, has, sig)) Snew = cand;
            }
            if (Snew == 0xFFFFu) {
                const uint16_t guess = (uint16_t)((S ^ gmask) & 0xFF);
                if (!lex(Wc, pin, t, tb - t, B.valid_set(guess) ? guess : 0xFFFFu, &Snew)) return false;
            }
            if (Snew != S) {
                if (ns >= MAXARC_E) return false;
                e.tau[ns - 1] = t;
                e.S[ns++] = Snew;
                S = Snew;
                stall = 0;
            } else if (++stall > 3) {
                return false;
            }
        }
    }
    e.narcs = ns;
    return true;
}

void StateSolver::vpoly(const uint16_t* S, const double* tau, int n, int a, const double* c, const double* y,
                        const StateArc* sa, int nsa, double V[MAXM][4]) const {
    double t0, t1, Wc[MAXM][4], Pv[MAXM][MAXM], phi[2 * MAXM][4], sig[MAXM][4];
    bool has[2 * MAXM];
    span_(tau, n, a, base_.p_->T, &t0, &t1);
    wpoly(c, y, sa, nsa, 0.5 * (t0 + t1), Wc);
    if (!kkt(S[a], Wc, Pv, V, phi, has, sig))
        for (int j = 0; j < MAXM; ++j) for (int p = 0; p < 4; ++p) V[j][p] = 0.0;
}

// F = -grad D (terminal residual, then D_top(t1) - b per arc); H = -Hess D in H_ when wantH
bool StateSolver::evaluate(const double* y, Ev& e, bool wantH) {
    const Solver& B = base_;
    const Problem& p = *B.p_;
    const int n = B.n_, ny = n + nsa_;
    const double T = p.T;
    double c[MAXN];
    cvec(y, c);
    if (!walk(c, y, e)) return false;
    double ab[NY], ci[NY], v[MAXM], w[MAXM];
    int li[NY];
    for (int i = 0; i < ny; ++i) { e.F[i] = 0.0; ab[i] = 0.0; }
    if (wantH) for (int i = 0; i < NY * NY; ++i) H_[i] = 0.0;
    for (int l = 0; l < B.m_; ++l)
        for (int r = 0; r < p.orders[l]; ++r) li[B.off_[l] + r] = l;
    for (int q = 0; q < nsa_; ++q) li[n + q] = sa_[q].chain;
    double D = 0.0;
    for (int a = 0; a < e.narcs; ++a) {
        double t0, t1, Wc[MAXM][4], Pv[MAXM][MAXM], Vv[MAXM][4], phi[2 * MAXM][4], sig[MAXM][4];
        bool has[2 * MAXM];
        span_(e.tau, e.narcs, a, T, &t0, &t1);
        if (!(t1 > t0)) continue;
        const double tm = 0.5 * (t0 + t1);
        wpoly(c, y, sa_, nsa_, tm, Wc);
        if (!kkt(e.S[a], Wc, Pv, Vv, phi, has, sig)) return false;
        const double hw = 0.5 * (t1 - t0);
        for (int g = 0; g < 4; ++g) {
            const double t = t0 + hw * (SGX[g] + 1.0), wg = SGW[g] * hw;
            for (int j = 0; j < B.m_; ++j) {
                v[j] = peval(Vv[j], t);
                double s = 0.0, tp = 1.0;                 // raw w_j (Wc is Lc^-1 w)
                for (int k = 0; k < p.orders[j]; ++k) { s += c[B.off_[j] + k] * tp; tp *= t; }
                for (int q = 0; q < nsa_; ++q)
                    if (!sa_[q].fix1 && tm < sa_[q].t1 && sa_[q].chain == j) s += y[n + q];
                w[j] = s;
            }
            for (int l = 0; l < B.m_; ++l)
                for (int r = 0; r < p.orders[l]; ++r)
                    ci[B.off_[l] + r] = ipow(T - t, p.orders[l] - 1 - r) / fact(p.orders[l] - 1 - r);
            for (int q = 0; q < nsa_; ++q) ci[n + q] = (!sa_[q].fix1 && tm < sa_[q].t1) ? 1.0 : 0.0;
            for (int i = 0; i < ny; ++i) {
                const double f = wg * ci[i] * v[li[i]];
                e.F[i] += f;
                ab[i] += fabs(f);
            }
            if (wantH)
                for (int i = 0; i < ny; ++i) {
                    if (ci[i] == 0.0) continue;
                    for (int k = 0; k < ny; ++k) H_[i * NY + k] += wg * ci[i] * ci[k] * Pv[li[i]][li[k]];
                }
            double s = 0.0;
            for (int j = 0; j < B.m_; ++j) s += 0.5 * v[j] * (B.rI_ ? v[j] : B.vR(j, v)) - w[j] * v[j];
            D += wg * s;
        }
    }
    double sc = 1.0;
    for (int i = 0; i < n; ++i) {
        e.F[i] += B.free_[i] - p.xf[i];
        sc = fmax(sc, fmax(fabs(p.xf[i]), fabs(B.free_[i])));
        sc = fmax(sc, ab[i] + fabs(B.free_[i]));
    }
    for (int q = 0; q < nsa_; ++q) {
        const int j = sa_[q].chain;
        const double x0t = B.x0_[B.off_[j] + p.orders[j] - 1];
        if (sa_[q].fix1) {                               // no entry: Delta_q = 0
            e.F[n + q] = y[n + q];
            D -= 0.5 * y[n + q] * y[n + q];
            if (wantH) H_[(n + q) * NY + n + q] = 1.0;
        } else {
            e.F[n + q] += x0t - sa_[q].b;
            D += y[n + q] * (sa_[q].b - x0t);
            sc = fmax(sc, fmax(fabs(sa_[q].b), ab[n + q] + fabs(x0t)));
        }
    }
    if (B.qf_) {   // mu: F = Cf (s(T) - xf) + mu, -Hess = C H C^T + diag(I, 0), C = diag(Cf, I)
        double r[MAXN], sq = 0.0;
        for (int i = 0; i < n; ++i) r[i] = e.F[i];
        for (int i = 0; i < n; ++i) {
            double f = y[i], a = fabs(y[i]);
            for (int j = i; j < n; ++j) { f += B.Cf_[i * MAXN + j] * r[j]; a += fabs(B.Cf_[i * MAXN + j]) * sc; }
            e.F[i] = f;
            sq = fmax(sq, a);
            D += y[i] * (B.g_[i] - 0.5 * y[i]);
        }
        sc = fmax(sc, sq);
        if (wantH) {
            double* X = A_;                              // X = C H (rows 0..n-1 change)
            for (int i = 0; i < ny; ++i)
                for (int k = 0; k < ny; ++k) {
                    double s = H_[i * NY + k];
                    if (i < n) { s = 0.0; for (int j = i; j < n; ++j) s += B.Cf_[i * MAXN + j] * H_[j * NY + k]; }
                    X[i * NY + k] = s;
                }
            for (int i = 0; i < ny; ++i)                 // H = X C^T + diag(I, 0)
                for (int k = 0; k < ny; ++k) {
                    double s = X[i * NY + k];
                    if (k < n) { s = 0.0; for (int j = k; j < n; ++j) s += X[i * NY + j] * B.Cf_[k * MAXN + j]; }
                    H_[i * NY + k] = s + ((i == k && i < n) ? 1.0 : 0.0);
                }
        }
    } else {
        for (int i = 0; i < n; ++i) D += y[i] * (p.xf[i] - B.free_[i]);
    }
    e.scale = sc;
    e.D = D;
    return true;
}

// Newton on the concave dual for fixed junction times (NOTES.md 7.2 / 13.1, plus the arcs' Delta)
bool StateSolver::inner(double* y, int max_its) {
    const int ny = base_.n_ + nsa_;
    int cur = 0;
    if (!evaluate(y, ev_[cur], true)) { strcpy(why_, "event walk failed"); return false; }
    for (int it = 0;; ++it) {
        const Ev& E = ev_[cur];
        double Fn = 0.0;
        for (int i = 0; i < ny; ++i) Fn = fmax(Fn, fabs(E.F[i]));
        if (Fn <= 1e-12 * E.scale) { if (cur) ev_[0] = ev_[1]; return true; }
        if (it >= max_its) { if (cur) ev_[0] = ev_[1]; strcpy(why_, "iteration cap"); return false; }
        ++its_;
        double d[NY], yn[NY], hmax = 0.0;
        for (int i = 0; i < NY * NY; ++i) hmax = fmax(hmax, fabs(H_[i]));
        bool solved = false;
        double rho = 0.0;
        for (int att = 0; att < 20 && !solved; ++att) {
            for (int i = 0; i < NY * NY; ++i) A_[i] = H_[i];
            for (int i = 0; i < ny; ++i) { A_[i * NY + i] += rho; d[i] = -E.F[i]; }
            solved = lu_solve(A_, ny, NY, d, 1, 1);
            rho = (rho == 0.0) ? 1e-12 * hmax : 10.0 * rho;
        }
        if (!solved) { strcpy(why_, "singular Newton matrix"); return false; }
        double slope = 0.0, F2 = 0.0;
        for (int i = 0; i < ny; ++i) { slope -= E.F[i] * d[i]; F2 += E.F[i] * E.F[i]; }
        const int nxt = 1 - cur;
        bool acc = false;
        for (double al = 1.0; al >= 1e-10; al *= 0.5) {
            for (int i = 0; i < ny; ++i) yn[i] = y[i] + al * d[i];
            if (!evaluate(yn, ev_[nxt], true)) continue;
            const Ev& N = ev_[nxt];
            if (N.D >= E.D + 1e-4 * al * slope) { acc = true; break; }
            double nF2 = 0.0;
            for (int i = 0; i < ny; ++i) nF2 += N.F[i] * N.F[i];
            if (al * slope <= 1e3 * EPS_ * (1.0 + fabs(E.D)) && nF2 < F2) { acc = true; break; }
        }
        if (!acc) { strcpy(why_, "line search failed"); if (cur) ev_[0] = ev_[1]; return false; }
        for (int i = 0; i < ny; ++i) y[i] = yn[i];
        cur = nxt;
    }
}

// junction residuals v_j(t1-) and v_j(t2+) of the plan in ev_[0]
bool StateSolver::gres(const double* y, double* g, int* ng, double* vs) {
    const Ev& e = ev_[0];
    const double T = base_.p_->T;
    double c[MAXN], V[MAXM][4];
    cvec(y, c);
    int k = 0;
    *vs = 1.0;
    for (int a = 0; a < e.narcs; ++a) {
        double t0, t1;
        span_(e.tau, e.narcs, a, T, &t0, &t1);
        vpoly(e.S, e.tau, e.narcs, a, c, y, sa_, nsa_, V);
        for (int j = 0; j < base_.m_; ++j) *vs = fmax(*vs, 1.0 + fabs(peval(V[j], t0)));
    }
    for (int q = 0; q < nsa_; ++q) {
        const int j = sa_[q].chain;
        for (int side = 0; side < 2; ++side) {
            if (side == 0 ? sa_[q].fix1 : sa_[q].fix2) continue;
            const double tj = side == 0 ? sa_[q].t1 : sa_[q].t2;
            int ab = -1;                                 // arc ending at tj (ab + 1 starts there)
            double bd = HUGE_VAL;
            for (int a = 0; a + 1 < e.narcs; ++a) {
                const double dd = fabs(e.tau[a] - tj);
                if (dd < bd) { bd = dd; ab = a; }
            }
            if (ab < 0 || bd > 1e-12 * T) return false;
            // control continuity: v_j(t1-) = 0, v_j(t2+) = 0 (v_j = 0 on the arc). This is dJ/dtj = 0
            // when the thrust rows active at the junction do not change (NOTES.md 14.2).
            vpoly(e.S, e.tau, e.narcs, side == 0 ? ab : ab + 1, c, y, sa_, nsa_, V);
            g[k++] = peval(V[j], tj);
        }
    }
    *ng = k;
    return true;
}

// Newton (finite-difference Jacobian) on the free junction times: dJ/dt = 0 at each (see gres).
// false with collapsed_ = q: arc q collapsed (t2 <= t1).
bool StateSolver::junctions(double* y, int max_its, int* outer) {
    const double T = base_.p_->T;
    const int ny = base_.n_ + nsa_;
    for (int it = 0; it < 40; ++it) {
        double* tv[2 * MAXSA];
        int nu = 0;
        for (int q = 0; q < nsa_; ++q) {
            if (!sa_[q].fix1) tv[nu++] = &sa_[q].t1;
            if (!sa_[q].fix2) tv[nu++] = &sa_[q].t2;
        }
        double g[2 * MAXSA], vs;
        int ng;
        if (!gres(y, g, &ng, &vs)) { strcpy(why_, "junction not found in the walk"); return false; }
        double gn = 0.0;
        for (int i = 0; i < ng; ++i) gn = fmax(gn, fabs(g[i]));
        if (gn <= 1e-10 * vs) return true;
        ++*outer;
        double J[4 * MAXSA * MAXSA], y0[NY], tau0[2 * MAXSA], d[2 * MAXSA];
        for (int i = 0; i < ny; ++i) y0[i] = y[i];
        for (int k = 0; k < nu; ++k) tau0[k] = *tv[k];
        for (int k = 0; k < nu; ++k) {
            const double h = 1e-7 * T;
            *tv[k] = tau0[k] + h;
            for (int i = 0; i < ny; ++i) y[i] = y0[i];
            double gk[2 * MAXSA], vk;
            int nk;
            if (!inner(y, max_its) || !gres(y, gk, &nk, &vk)) { strcpy(why_, "junction Jacobian failed"); return false; }
            for (int i = 0; i < ng; ++i) J[i * nu + k] = (gk[i] - g[i]) / h;
            *tv[k] = tau0[k];
        }
        // (J^T J + lam I) d = -J^T g with a tiny lam: Newton when J is regular; a junction that does
        // not move its residual (e.g. entry inside a fully saturated arc, where v_j = 0 anyway) stays put
        double N[4 * MAXSA * MAXSA], dm = 0.0;
        for (int a = 0; a < nu; ++a) {
            double s = 0.0;
            for (int i = 0; i < ng; ++i) s -= J[i * nu + a] * g[i];
            d[a] = s;
            for (int b = 0; b < nu; ++b) {
                double t = 0.0;
                for (int i = 0; i < ng; ++i) t += J[i * nu + a] * J[i * nu + b];
                N[a * nu + b] = t;
            }
            dm = fmax(dm, N[a * nu + a]);
        }
        for (int a = 0; a < nu; ++a) N[a * nu + a] += 1e-10 * dm;
        if (!lu_solve(N, nu, nu, d, 1, 1)) { strcpy(why_, "singular junction Jacobian"); return false; }
        bool acc = false;
        for (double al = 1.0; al >= 1.0 / 256; al *= 0.5) {
            for (int k = 0; k < nu; ++k) *tv[k] = tau0[k] + al * d[k];
            bool bad = false;
            for (int q = 0; q < nsa_; ++q) {
                StateArc& s = sa_[q];
                if (!s.fix2 && s.t2 >= T * (1 - 1e-9)) { s.t2 = T; s.fix2 = true; }
                if (!s.fix1 && s.t1 <= 1e-9 * T) bad = true;
                if (s.t2 - s.t1 <= 1e-9 * T) {           // collapse only if no shorter step avoids it
                    if (al <= 1.0 / 256) { collapsed_ = q; return false; }
                    bad = true;
                }
            }
            if (bad) continue;
            for (int i = 0; i < ny; ++i) y[i] = y0[i];
            double gt[2 * MAXSA], vt;
            int nt;
            if (!inner(y, max_its) || !gres(y, gt, &nt, &vt)) continue;
            double tn = 0.0;
            for (int i = 0; i < nt; ++i) tn = fmax(tn, fabs(gt[i]));
            if (nt != ng || tn < gn) { acc = true; break; }
            for (int q = 0; q < nsa_; ++q) { sa_[q].fix2 = false; }   // undo a tentative fix2
            for (int q = 0; q < nsa_; ++q) if (sa_[q].t2 >= T) sa_[q].fix2 = true;
        }
        if (!acc) { strcpy(why_, "junction Newton stalled"); return false; }
    }
    strcpy(why_, "junction Newton iteration cap");
    return false;
}

// Worst violation s*(D_top_j - b) on the non-pinned arcs of ev_[0] (exact: endpoints and the roots
// of v_j). With tw >= 0: the crossings of the bound around tw (entry ta, exit tb).
double StateSolver::scan(const double* c, const double* y, int j, int side, double tw, double* twout,
                         double* ta, double* tb) const {
    const Solver& B = base_;
    const Ev& e = ev_[0];
    const double T = B.p_->T, b = side > 0 ? hi_[j] : lo_[j];
    double top = B.x0_[B.off_[j] + B.p_->orders[j] - 1], worst = -HUGE_VAL, V[MAXM][4];
    *ta = 0.0; *tb = T;
    bool pend = tw >= 0.0;
    for (int a = 0; a < e.narcs; ++a) {
        double t0, t1, av[4], fp[5], red[4], pts[6];
        span_(e.tau, e.narcs, a, T, &t0, &t1);
        const double L = t1 - t0;
        if (!(L > 0)) continue;
        vpoly(e.S, e.tau, e.narcs, a, c, y, sa_, nsa_, V);
        shift(V[j], 4, t0, av);
        fp[0] = side * (top - b);
        for (int p = 0; p < 4; ++p) fp[p + 1] = side * av[p] / (p + 1);
        const double topn = side * pval(fp, 5, L) + b;   // top(t1)
        if (e.S[a] & (1u << (8 + j))) { top = topn; continue; }   // pinned: on the bound
        int np = 0, k;
        pts[np++] = 0.0;
        const double tol = 1e-12 * (pscale(av, 4, L) + 1.0);
        const int rl = strip(av, 4, L, tol, red, &k);
        if (rl > 0) {
            Cross cr;
            crossings(red, rl, k, L, tol, cr);
            for (int i = 0; i < cr.n; ++i) if (cr.root[i] > 0 && cr.root[i] < L) pts[np++] = cr.root[i];
        }
        pts[np++] = L;
        double fprev = pval(fp, 5, 0.0);
        for (int i = 0; i < np; ++i) {
            const double f = pval(fp, 5, pts[i]);
            if (f > worst) { worst = f; *twout = t0 + pts[i]; }
            if (i > 0 && (fprev > 0) != (f > 0)) {           // monotone piece: one crossing
                const double x = t0 + bisect(fp, 5, pts[i - 1], pts[i]);
                if (f > 0 && x <= tw) *ta = x;
                if (f <= 0 && pend && x >= tw) { *tb = x; pend = false; }
            }
            fprev = f;
        }
        top = topn;
    }
    return worst;
}

double StateSolver::check(const double* y, int* wc, int* ws, double* ta, double* tb, double* twout,
                          double* psign, double* ppmp) const {
    const Solver& B = base_;
    const Ev& e = ev_[0];
    const double T = B.p_->T;
    double c[MAXN], worst = 0.0, tw = 0.0, xx, xa, xb;
    cvec(y, c);
    *wc = -1;
    hiord_ = 0.0;
    hiidx_ = -1;
    for (int j = 0; j < B.m_; ++j)
        for (int side = -1; side <= 1; side += 2) {
            const double b = side > 0 ? hi_[j] : lo_[j];
            if (!isfinite(b)) continue;
            const double v = scan(c, y, j, side, -1.0, &xx, &xa, &xb) / (1.0 + fabs(b));
            if (v > worst) { worst = v; *wc = j; *ws = side; tw = xx; }
        }
    if (*wc >= 0) scan(c, y, *wc, *ws, tw, &xx, ta, tb);
    *twout = tw;
    // eta = -side * sig' >= 0 on boundary arcs; thrust rows feasible with mu >= 0 (Solver::verify)
    double ps = 0.0, pp = 0.0;
    for (int a = 0; a < e.narcs; ++a) {
        double t0, t1, Wc[MAXM][4], Pv[MAXM][MAXM], Vv[MAXM][4], phi[2 * MAXM][4], sig[MAXM][4], bb[4], d[3], cr[2];
        bool has[2 * MAXM];
        span_(e.tau, e.narcs, a, T, &t0, &t1);
        const double L = t1 - t0, tm = 0.5 * (t0 + t1);
        if (!(L > 0)) continue;
        wpoly(c, y, sa_, nsa_, tm, Wc);
        if (!kkt(e.S[a], Wc, Pv, Vv, phi, has, sig)) { pp = HUGE_VAL; continue; }
        for (int key = 0; key < 2 * B.m_; ++key) {
            if (!has[key]) continue;
            shift(phi[key], 4, t0, bb);
            pp = fmax(pp, fmax(bb[0], pval(bb, 4, L)));
            for (int i = 1; i < 4; ++i) d[i - 1] = i * bb[i];
            const int nc = roots_le2(d, 3, cr);
            for (int i = 0; i < nc; ++i) if (cr[i] > 0.0 && cr[i] < L) pp = fmax(pp, pval(bb, 4, cr[i]));
        }
        for (int q = 0; q < nsa_; ++q) {
            const int j = sa_[q].chain;
            if (!(e.S[a] & (1u << (8 + j))) || !(sa_[q].t1 <= tm && tm < sa_[q].t2)) continue;
            shift(sig[j], 4, t0, bb);
            const double nrm = (1.0 + fmax(fabs(bb[0]), fabs(pval(bb, 4, L)))) / T;
            const double dd[3] = {bb[1], 2 * bb[2], 3 * bb[3]};
            double mx = fmax(pval(dd, 3, 0.0), pval(dd, 3, L)), mn = fmin(pval(dd, 3, 0.0), pval(dd, 3, L));
            if (dd[2] != 0.0) {
                const double xv = -dd[1] / (2 * dd[2]);
                if (xv > 0 && xv < L) { mx = fmax(mx, pval(dd, 3, xv)); mn = fmin(mn, pval(dd, 3, xv)); }
            }
            ps = fmax(ps, (sa_[q].side > 0 ? mx : -mn) / nrm);
        }
    }
    *psign = ps;
    *ppmp = pp / B.R_;
    // states of relative degree > 1 are not enforced, only verified (exact extrema per arc)
    double x[MAXN], V[MAXM][4], av[4];
    for (int i = 0; i < B.n_; ++i) x[i] = B.x0_[i];
    for (int a = 0; a < e.narcs; ++a) {
        double t0, t1;
        span_(e.tau, e.narcs, a, T, &t0, &t1);
        const double h = t1 - t0;
        vpoly(e.S, e.tau, e.narcs, a, c, y, sa_, nsa_, V);
        for (int j = 0; j < B.m_; ++j) {
            const int k = B.p_->orders[j], o = B.off_[j];
            shift(V[j], 4, t0, av);
            double sn[MAXK];
            for (int rr = 0; rr < k; ++rr) {
                double P[8] = {0, 0, 0, 0, 0, 0, 0, 0};     // s_rr(t0 + h) as a polynomial in h
                const int mm = k - 1 - rr;
                for (int l = rr; l < k; ++l) P[l - rr] += x[o + l] / fact(l - rr);
                for (int i = 0; i < 4; ++i) P[mm + i + 1] += av[i] * fact(i) / fact(mm + i + 1);
                sn[rr] = pval(P, 8, h);
                if (rr == k - 1 || !(h > 0) || (!isfinite(xlo_[o + rr]) && !isfinite(xhi_[o + rr]))) continue;
                double d[8], rt[8];
                for (int i = 1; i < 8; ++i) d[i - 1] = i * P[i];
                const int nr = proots(d, 7, h, rt);
                double mx = fmax(P[0], sn[rr]), mn = fmin(P[0], sn[rr]);
                for (int i = 0; i < nr; ++i) { const double f = pval(P, 8, rt[i]); mx = fmax(mx, f); mn = fmin(mn, f); }
                const double bm = fmax(isfinite(xlo_[o + rr]) ? fabs(xlo_[o + rr]) : 0.0, isfinite(xhi_[o + rr]) ? fabs(xhi_[o + rr]) : 0.0);
                const double vv = fmax(mx - xhi_[o + rr], xlo_[o + rr] - mn) / (1.0 + bm);
                if (vv > TOL_X && vv > hiord_) { hiord_ = vv; hiidx_ = o + rr; }
            }
            for (int rr = 0; rr < k; ++rr) x[o + rr] = sn[rr];
        }
    }
    return worst;
}

const char* StateSolver::sequence(const uint16_t* S, int narcs, const StateArc* sa, int nsa, const double* tau,
                                  char* buf, int len) const {
    int k = 0;
    buf[0] = 0;
    uint16_t prev = 0xFFFFu;
    for (int a = 0; a < narcs && k < len - 24; ++a) {
        if (S[a] == prev) continue;
        prev = S[a];
        if (k) buf[k++] = ' ';
        bool any = false;
        for (int r = 0; r < base_.m_; ++r)
            for (int s = 0; s < 2; ++s)
                if (S[a] & (1u << (2 * r + s))) { k += snprintf(buf + k, len - k, "%d%c", r, s ? '-' : '+'); any = true; }
        const double tm = (a ? tau[a - 1] : 0.0);
        for (int j = 0; j < base_.m_; ++j)
            if (S[a] & (1u << (8 + j))) {
                int side = 1;
                for (int q = 0; q < nsa; ++q)
                    if (sa[q].chain == j && sa[q].t1 <= tm + 1e-12 && tm < sa[q].t2) side = sa[q].side;
                k += snprintf(buf + k, len - k, "|%c%c", CH[j], side > 0 ? '+' : '-');
                any = true;
            }
        if (!any) buf[k++] = 'f';
        buf[k] = 0;
    }
    return buf;
}

void StateSolver::fill(StateResult& r) const {
    const Ev& e = ev_[0];
    r.narcs = e.narcs;
    for (int a = 0; a < e.narcs; ++a) r.S[a] = e.S[a];
    for (int a = 0; a + 1 < e.narcs; ++a) r.tau[a] = e.tau[a];
    r.nsa = nsa_;
    for (int q = 0; q < nsa_; ++q) r.sa[q] = sa_[q];
    double Fn = 0.0;
    for (int i = 0; i < base_.n_ + nsa_; ++i) Fn = fmax(Fn, fabs(e.F[i]));
    r.residual = Fn;
}

bool StateSolver::run(StateResult& r, int warm) {
    const Solver& B = base_;
    const int n = B.n_;
    const double T = B.p_->T;
    const int cap = warm < 0 ? MAX_NEWTON : warm;
    double* const y = r.y;
    its_ = 0;
    r.outer = 0;
    r.converged = false;
    r.cost = HUGE_VAL;
    r.state_violation = r.pmp_violation = r.sign_violation = HUGE_VAL;
    if (warm < 0) {
        r.inserted = 0;
        Result r1;
        if (!base_.solve(r1)) {
            snprintf(why_, sizeof why_, "tier 1: %s", r1.failure ? r1.failure : "not converged");
            r.failure = why_;
            r.iterations = r1.iterations;
            r.narcs = 0;
            return false;
        }
        nsa_ = 0;
        for (int i = 0; i < n; ++i) {
            double s = r1.mu[i];
            if (!B.qf_) { s = 0.0; for (int j = 0; j < n; ++j) s += B.Linv_[i * MAXN + j] * r1.c[j]; }
            y[i] = s;
        }
        its_ = r1.iterations;
    } else {
        nsa_ = (r.nsa >= 0 && r.nsa <= MAXSA) ? r.nsa : 0;
        for (int q = 0; q < nsa_; ++q) sa_[q] = r.sa[q];
    }
    const char* why = nullptr;
    bool conv = false;
    char seq[160], det[64];
    unsigned gone = 0;                                       // chains whose boundary arc collapsed
    seq[0] = 0;
    if (!inner(y, cap)) why = why_;
    for (int pass = 0; !why && pass < 3 * MAXSA + 3; ++pass) {
        if (nsa_) {
            collapsed_ = -1;
            if (!junctions(y, cap, &r.outer)) {
                if (collapsed_ >= 0) {                       // collapsed arc: q = 1 touch = drop it
                    sequence(ev_[0].S, ev_[0].narcs, sa_, nsa_, ev_[0].tau, seq, sizeof seq);
                    gone |= 1u << sa_[collapsed_].chain;
                    for (int q = collapsed_; q + 1 < nsa_; ++q) { sa_[q] = sa_[q + 1]; y[n + q] = y[n + q + 1]; }
                    --nsa_;
                    if (!inner(y, cap)) {
                        snprintf(why_, sizeof why_, "arc order rejected: %s (arc collapsed, re-solve failed)", seq);
                        why = why_;
                    }
                    continue;
                }
                strncpy(det, why_, sizeof det - 1); det[sizeof det - 1] = 0;
                sequence(ev_[0].S, ev_[0].narcs, sa_, nsa_, ev_[0].tau, seq, sizeof seq);
                snprintf(why_, sizeof why_, "arc order rejected: %s (%s)", seq, det);
                why = why_;
                break;
            }
        }
        int wc, ws;
        double ta, tb, tw, ps, pp;
        const double viol = check(y, &wc, &ws, &ta, &tb, &tw, &ps, &pp);
        r.state_violation = viol; r.sign_violation = ps; r.pmp_violation = pp;
        if (ps > TOL_SIGN) {
            sequence(ev_[0].S, ev_[0].narcs, sa_, nsa_, ev_[0].tau, seq, sizeof seq);
            snprintf(why_, sizeof why_, "arc order rejected: %s (sign of the state multiplier)", seq);
            why = why_;
            break;
        }
        if (viol > TOL_X) {
            if (nsa_ >= MAXSA) { why = "state arc capacity LQOCP_MAXSA exceeded"; break; }
            if (gone & (1u << wc)) {                         // the arc collapsed but the violation is back
                snprintf(why_, sizeof why_, "arc order rejected: %s (boundary arc on %c collapsed, violation remains)", seq, CH[wc]);
                why = why_;
                break;
            }
            for (int q = 0; q < nsa_; ++q)
                if (sa_[q].chain == wc && ta < sa_[q].t2 && tb > sa_[q].t1) why = "overlapping boundary arcs";
            if (why) break;
            // initial boundary arc: the violation interval [ta, tb]; if the pinned problem has no
            // solution from there, a shorter piece around the worst point
            StateArc& s = sa_[nsa_];
            s.chain = (int8_t)wc; s.side = (int8_t)ws; s.b = ws > 0 ? hi_[wc] : lo_[wc];
            s.fix1 = ta <= 0.0; s.fix2 = tb >= T;
            ++nsa_;
            ++r.inserted;
            double ysave[NY];
            for (int i = 0; i < n + nsa_ - 1; ++i) ysave[i] = y[i];
            bool ok = false;
            for (double f = 1.0; f > 0.05 && !ok; f *= 0.5) {
                s.t1 = s.fix1 ? 0.0 : tw - f * (tw - ta);
                s.t2 = s.fix2 ? T : tw + f * (tb - tw);
                for (int i = 0; i < n + nsa_ - 1; ++i) y[i] = ysave[i];
                y[n + nsa_ - 1] = 0.0;
                ok = inner(y, cap);
            }
            if (!ok) {
                strncpy(det, why_, sizeof det - 1); det[sizeof det - 1] = 0;
                snprintf(why_, sizeof why_, "arc order rejected: insertion on %c (%s)", CH[wc], det);
                why = why_;
            }
            continue;
        }
        if (pp > 1e-9) { why = "PMP verification failed"; break; }
        if (hiidx_ >= 0) {
            snprintf(why_, sizeof why_, "unsupported constraint order: flat state %d violates its bound (relative degree > 1)", hiidx_);
            why = why_;
            break;
        }
        conv = true;
        break;
    }
    if (!conv && !why) why = "outer iteration cap";
    fill(r);
    r.iterations = its_;
    // cost: 1/2 int v'Rv (degree 6: 4-point Gauss is exact) + terminal
    double J = 0.0, c[MAXN], V[MAXM][4], v[MAXM];
    cvec(y, c);
    for (int a = 0; a < r.narcs; ++a) {
        double t0, t1;
        span_(r.tau, r.narcs, a, T, &t0, &t1);
        vpoly(r.S, r.tau, r.narcs, a, c, y, r.sa, r.nsa, V);
        const double hw = 0.5 * (t1 - t0);
        for (int g = 0; g < 4; ++g) {
            const double t = t0 + hw * (SGX[g] + 1.0);
            for (int j = 0; j < B.m_; ++j) v[j] = peval(V[j], t);
            double s = 0.0;
            for (int j = 0; j < B.m_; ++j) s += v[j] * (B.rI_ ? v[j] : B.vR(j, v));
            J += 0.5 * SGW[g] * hw * s;
        }
    }
    if (B.qf_) {
        double x[MAXN];
        eval_x(r, T, x);
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < n; ++k) J += 0.5 * (x[i] - B.p_->xf[i]) * B.p_->Qf[i][k] * (x[k] - B.p_->xf[k]);
    }
    r.cost = J;
    r.converged = conv;
    r.failure = conv ? nullptr : why;
    return conv;
}

bool StateSolver::solve_warm(StateResult& r, int max_its, double shift) {
    const double T = base_.p_->T;
    const int n = base_.n_;
    int k = 0;
    for (int q = 0; q < r.nsa && q < MAXSA; ++q) {
        StateArc a = r.sa[q];
        if (!a.fix2) a.t2 -= shift;
        if (a.t2 <= 1e-9 * T) continue;                  // ended: drop
        a.t1 -= shift;
        double d = r.y[n + q];
        if (a.fix1 || a.t1 <= 1e-9 * T) { a.t1 = 0.0; a.fix1 = true; d = 0.0; }
        r.sa[k] = a;
        r.y[n + k++] = d;
    }
    r.nsa = k;
    return run(r, max_its);
}

void StateSolver::eval_u(const StateResult& r, double t, double u[MAXM]) const {
    const double T = base_.p_->T;
    if (t < 0.0) t = 0.0;
    if (t > T) t = T;
    int a = 0;
    while (a < r.narcs - 1 && t >= r.tau[a]) ++a;
    double c[MAXN], V[MAXM][4];
    cvec(r.y, c);
    vpoly(r.S, r.tau, r.narcs, a, c, r.y, r.sa, r.nsa, V);
    for (int j = 0; j < base_.m_; ++j) u[j] = peval(V[j], t);
}

// exact: per arc, the chain states are Taylor-shifted and the cubic v integrated in closed form
void StateSolver::eval_x(const StateResult& r, double t, double x[MAXN]) const {
    const Solver& B = base_;
    const double T = B.p_->T;
    if (t < 0.0) t = 0.0;
    if (t > T) t = T;
    for (int i = 0; i < B.n_; ++i) x[i] = B.x0_[i];
    double c[MAXN], V[MAXM][4], av[4], s[MAXK];
    cvec(r.y, c);
    for (int a = 0; a < r.narcs; ++a) {
        double t0, t1;
        span_(r.tau, r.narcs, a, T, &t0, &t1);
        if (t0 >= t) break;
        const double h = fmin(t1, t) - t0;
        vpoly(r.S, r.tau, r.narcs, a, c, r.y, r.sa, r.nsa, V);
        for (int j = 0; j < B.m_; ++j) {
            const int k = B.p_->orders[j], o = B.off_[j];
            shift(V[j], 4, t0, av);
            for (int rr = 0; rr < k; ++rr) {
                double acc = 0.0;
                for (int l = rr; l < k; ++l) acc += x[o + l] * ipow(h, l - rr) / fact(l - rr);
                const int mm = k - 1 - rr;                   // int (t-s)^mm/mm! (s-t0)^i ds = i! h^(mm+i+1)/(mm+i+1)!
                for (int i = 0; i < 4; ++i) acc += av[i] * fact(i) * ipow(h, mm + i + 1) / fact(mm + i + 1);
                s[rr] = acc;
            }
            for (int rr = 0; rr < k; ++rr) x[o + rr] = s[rr];
        }
    }
}

bool StateSolver::mean_u(const StateResult& r, double t0, double t1, double u[MAXM]) const {
    if (r.narcs < 1 || !(t1 > t0)) return false;
    const double T = base_.p_->T;
    double c[MAXN], V[MAXM][4], av[4], acc[MAXM] = {0.0, 0.0, 0.0, 0.0};
    cvec(r.y, c);
    for (int a = 0; a < r.narcs; ++a) {
        double a0, a1;
        span_(r.tau, r.narcs, a, T, &a0, &a1);
        const double lo = fmax(t0, a0), hi = fmin(t1, a1);
        if (!(hi > lo)) continue;
        vpoly(r.S, r.tau, r.narcs, a, c, r.y, r.sa, r.nsa, V);
        const double h = hi - lo;
        for (int j = 0; j < base_.m_; ++j) {
            shift(V[j], 4, lo, av);
            acc[j] += (((av[3] * h / 4 + av[2] / 3) * h + av[1] / 2) * h + av[0]) * h;
        }
    }
    for (int j = 0; j < base_.m_; ++j) u[j] = acc[j] / (t1 - t0);
    return true;
}

}  // namespace lqocp
