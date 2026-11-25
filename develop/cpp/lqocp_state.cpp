// Tier 2: thrust box + first-order state box, energy cost. See lqocp_state.hpp and NOTES.md 14.
#include "lqocp_state.hpp"

#include "lqocp_cont.hpp"

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
const double RND_F = 30.0 * EPS_ / 1e-12;    // inner tolerance is 1e-12 scale: scale gets 30x the rounding of F / 1e-12
const char CH[4] = {'x', 'y', 'z', 'p'};
const int ZC = 2 * MAXSA;            // columns of Z_: one per free junction time

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
    hs_ = 1.0;
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
                      double phi[2 * MAXM][4], bool has[2 * MAXM], double sig[MAXM][4], double* amp) const {
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
    if (amp)                                         // |d mult / d v-space error|: large for nearly dependent rows
        for (int a = 0; a < na; ++a)
            if (key[a] >= 0) {
                double s = 0.0;
                for (int i = 0; i < m; ++i) s += X[a * ld + i] * X[a * ld + i];
                amp[key[a]] = sqrt(s) * B.rownorm2_[key[a]];
            }
    return true;
}

// Pv..sig: the caller's kkt buffers (clobbered), so they are not on the stack twice
bool StateSolver::lex(const double Wc[MAXM][4], uint16_t pin, double t, double T, uint16_t guess, uint16_t* out,
                      double Pv[MAXM][MAXM], double Vv[MAXM][4], double phi[2 * MAXM][4], bool has[2 * MAXM],
                      double sig[MAXM][4], uint16_t ev) const {
    const Solver& B = base_;
    double b[4];
    for (int i = -1; i < B.nsets_; ++i) {
        const uint16_t cand = (i < 0) ? guess : B.sets_[i];
        if (i < 0 && !B.valid_set(cand)) continue;
        const uint16_t S = (uint16_t)(cand | (pin << 8));
        double amp[2 * MAXM];
        if (!kkt(S, Wc, Pv, Vv, phi, has, sig, amp)) continue;
        rtol(S, Wc, phi, t, amp);
        bool ok = true;
        for (int key = 0; key < 2 * B.m_ && ok; ++key) {
            if (!has[key]) continue;
            shift(phi[key], 4, t, b);
            const double tol = 1e-12 * (pscale(b, 4, T) + B.R_) + amp[key];
            // ev: rows at an event of the walk, zero at t by construction: their sign is that of the derivatives,
            // unless the value exceeds what the root's time error (1e-9 T) explains
            const int q0 = ((ev >> key) & 1u) && fabs(b[0]) <= 1e-9 * T * fabs(b[1]);
            for (int q = q0; q < 4; ++q)
                if (fabs(b[q]) * ipow(T, q) > tol) { if (b[q] > 0) ok = false; break; }
        }
        if (ok) { *out = S; return true; }
    }
    return false;
}

// Rounding of the row functions at t (amp in: kkt's amplification per active row; out: absolute tolerance per row).
// kkt cancels terms of size cs (|Lc^-1 w| plus the active rows times their multipliers); nearly dependent active
// rows amplify that into their multipliers (B: the motor pairs nearly span the z direction).
void StateSolver::rtol(uint16_t S, const double Wc[MAXM][4], const double phi[2 * MAXM][4], double t, double* amp) const {
    const Solver& B = base_;
    double cs = 0.0;
    for (int i = 0; i < B.m_; ++i) cs += fabs(peval(Wc[i], t));
    for (int k = 0; k < 2 * B.m_; ++k) if (S & (1u << k)) cs += fabs(peval(phi[k], t)) / sqrt(B.rownorm2_[k]);
    for (int k = 0; k < 2 * B.m_; ++k) amp[k] = 100.0 * EPS_ * cs * ((S & (1u << k)) ? amp[k] : sqrt(B.rownorm2_[k]));
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
        if (!lex(Wc, pin, ta, tb - ta, (uint16_t)(S & 0xFF), &S, Pv, Vv, phi, has, sig)) return false;
        if (ns >= MAXARC_E) return false;
        if (ns) e.tau[ns - 1] = ta;
        e.S[ns++] = S;
        double t = ta;
        int stall = 0;
        uint16_t zev = 0;                                // rows of the last event: zero at t by construction
        for (;;) {
            const double L = tb - t;
            double amp[2 * MAXM];
            if (!kkt(S, Wc, Pv, Vv, phi, has, sig, amp)) return false;
            rtol(S, Wc, phi, t, amp);
            double rootk[2 * MAXM], s1 = HUGE_VAL;
            bool simplek[2 * MAXM], evk[2 * MAXM];
            for (int key = 0; key < 2 * B.m_; ++key) {
                evk[key] = false;
                if (!has[key]) continue;
                double b[4], red[4];
                int kk;
                shift(phi[key], 4, t, b);
                if ((zev >> key) & 1u) b[0] = 0.0;      // else a new multiplier's root error, amplified by nearly dependent rows, stalls
                const double tol = 1e-12 * (pscale(b, 4, T) + B.R_) + amp[key];   // as the tier-1 walk (energy-float 71b3166) + rounding
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
            zev = gmask;
            uint16_t Snew = 0xFFFFu;
            if (ngroup == 1 && gsimple) {
                const uint16_t cand = (uint16_t)(S ^ (1u << gkey));
                if (B.valid_set(cand) && kkt(cand, Wc, Pv, Vv, phi, has, sig)) Snew = cand;
            }
            if (Snew == 0xFFFFu) {
                const uint16_t guess = (uint16_t)((S ^ gmask) & 0xFF);
                if (!lex(Wc, pin, t, tb - t, B.valid_set(guess) ? guess : 0xFFFFu, &Snew, Pv, Vv, phi, has, sig, gmask)) return false;
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
    double c[MAXN];
    cvec(y, c);
    return walk(c, y, e) && integrate(c, y, e, wantH);
}

// the integrals of evaluate (own frame: its buffers are not on the stack under walk)
__attribute__((noinline)) bool StateSolver::integrate(const double* c, const double* y, Ev& e, bool wantH) {
    const Solver& B = base_;
    const Problem& p = *B.p_;
    const int n = B.n_, ny = n + nsa_;
    const double T = p.T;
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
            // v = Lc^-T (Lc^-1 w - active rows x multipliers): the size of the terms that cancel, for the rounding of F
            double rw[MAXM], rv[MAXM];
            for (int i = 0; i < B.m_; ++i) {
                rw[i] = fabs(peval(Wc[i], t));
                for (int k = 0; k < 2 * B.m_; ++k)
                    if (e.S[a] & (1u << k)) rw[i] += fabs(B.rows_[k][i] * peval(phi[k], t)) / B.rownorm2_[k];
                for (int l = 0; l < B.m_; ++l)
                    if (e.S[a] & (1u << (8 + l))) rw[i] += fabs(B.Lci_[i][l] * peval(sig[l], t));
            }
            for (int j = 0; j < B.m_; ++j) { rv[j] = 0.0; for (int i = j; i < B.m_; ++i) rv[j] += fabs(B.Lci_[i][j]) * rw[i]; }
            for (int i = 0; i < ny; ++i) {
                const double f = wg * ci[i] * v[li[i]];
                e.F[i] += f;
                ab[i] += fabs(f) + RND_F * wg * fabs(ci[i]) * rv[li[i]];   // tolerance >= 30x the rounding of F
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
            // in place (Cf upper triangular: entry i reads entries >= i): H = C H, then H = H C^T + diag(I, 0)
            for (int i = 0; i < n; ++i)
                for (int k = 0; k < ny; ++k) {
                    double s = 0.0;
                    for (int j = i; j < n; ++j) s += B.Cf_[i * MAXN + j] * H_[j * NY + k];
                    H_[i * NY + k] = s;
                }
            for (int i = 0; i < ny; ++i)
                for (int k = 0; k < n; ++k) {
                    double s = 0.0;
                    for (int j = k; j < n; ++j) s += H_[i * NY + j] * B.Cf_[k * MAXN + j];
                    H_[i * NY + k] = s + ((i == k) ? 1.0 : 0.0);
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
        if (it >= max_its || its_ >= cap_) { if (cur) ev_[0] = ev_[1]; strcpy(why_, "iteration cap"); return false; }
        ++its_;
        double d[NY], yn[NY], hmax = 0.0;
        for (int i = 0; i < NY * NY; ++i) hmax = fmax(hmax, fabs(H_[i]));
        bool solved = false;
        double rho = 0.0;
        double* const Hs = Z_;                           // upper triangle of H_ (symmetric): H_ is factored in place
        for (int i = 0, q = 0; i < ny; ++i) for (int k = i; k < ny; ++k) Hs[q++] = H_[i * NY + k];
        for (int att = 0; att < 20 && !solved; ++att) {
            if (att) for (int i = 0, q = 0; i < ny; ++i) for (int k = i; k < ny; ++k, ++q) H_[i * NY + k] = H_[k * NY + i] = Hs[q];
            for (int i = 0; i < ny; ++i) { H_[i * NY + i] += rho; d[i] = -E.F[i]; }
            solved = lu_solve(H_, ny, NY, d, 1, 1);
            rho = (rho == 0.0) ? 1e-12 * hmax : 10.0 * rho;
        }
        if (!solved) { strcpy(why_, "singular Newton matrix"); return false; }
        double slope = 0.0, F2 = 0.0;
        for (int i = 0; i < ny; ++i) { slope -= E.F[i] * d[i]; F2 += E.F[i] * E.F[i]; }
        const int nxt = 1 - cur;
        bool acc = false;
        for (double al = 1.0; al >= 1e-3; al *= 0.5) {
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

// dw_l/dy_i at time t (l the chain of unknown i), raw nu coordinates; tm: a time on the same piece
double StateSolver::wbasis(int i, double t, double tm) const {
    const Solver& B = base_;
    if (i >= B.n_) { const StateArc& s = sa_[i - B.n_]; return (!s.fix1 && tm < s.t1) ? 1.0 : 0.0; }
    int l = 0;
    while (l + 1 < B.m_ && B.off_[l + 1] <= i) ++l;
    const int e = B.p_->orders[l] - 1 - (i - B.off_[l]);
    return ipow(B.p_->T - t, e) / fact(e);
}

// junction residuals v_j(t1-) and v_j(t2+) of the plan in ev_[0] (y converged, H_ its Newton matrix).
// With J: their exact Jacobian in the free junction times (row-major, ng x ng), by the implicit function
// theorem through the inner KKT: dg/dth = g_th - g_y H^-1 F_th. For fixed y every piece is a fixed
// polynomial, so F_th is the jump of the integrand at th (thrust events inside a piece keep v continuous)
// and g_th = v_j'(th). J may point into H_ (written only after H_ is factored).
bool StateSolver::gres(const double* y, double* g, int* ng, double* vs, double* J, const double* zs, double* dgs) {
    const Solver& B = base_;
    const Problem& p = *B.p_;
    const Ev& e = ev_[0];
    const double T = p.T;
    const int n = B.n_, ny = n + nsa_;
    double c[MAXN], t0, t1, Wc[MAXM][4], Pv[MAXM][MAXM], Vv[MAXM][4], phi[2 * MAXM][4], sig[MAXM][4];
    bool has[2 * MAXM];
    cvec(y, c);
    auto arc = [&](int a) {                              // Vv, Pv of arc a (as vpoly)
        span_(e.tau, e.narcs, a, T, &t0, &t1);
        wpoly(c, y, sa_, nsa_, 0.5 * (t0 + t1), Wc);
        return kkt(e.S[a], Wc, Pv, Vv, phi, has, sig);
    };
    int k = 0;
    *vs = 1.0;
    for (int a = 0; a < e.narcs; ++a) {
        if (!arc(a)) for (int j = 0; j < MAXM; ++j) for (int p = 0; p < 4; ++p) Vv[j][p] = 0.0;
        for (int j = 0; j < B.m_; ++j) *vs = fmax(*vs, 1.0 + fabs(peval(Vv[j], t0)));
    }
    double* const Fth = Z_;                      // becomes Z = H^-1 F_th (dy/dth = -Z)
    double pg[2 * MAXSA][MAXM], tjk[2 * MAXSA], tmk[2 * MAXSA], gtk[2 * MAXSA];   // g_y row k, compact
    int li[NY];
    for (int l = 0; l < B.m_; ++l)
        for (int r = 0; r < p.orders[l]; ++r) li[B.off_[l] + r] = l;
    for (int q = 0; q < nsa_; ++q) li[n + q] = sa_[q].chain;
    for (int q = 0; q < nsa_; ++q) {
        const int j = sa_[q].chain;
        for (int side = 0; side < 2; ++side) {
            if (side == 0 ? sa_[q].fix1 || touch(sa_[q]) : sa_[q].fix2) continue;
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
            if (!J) {
                g[k++] = arc(side == 0 ? ab : ab + 1) ? peval(Vv[j], tj) : 0.0;
                continue;
            }
            double vv[2][MAXM], tm[2];
            for (int s = 0; s < 2; ++s) {                // s = 0: arc left of tj, 1: right
                if (!arc(ab + s)) return false;
                tm[s] = 0.5 * (t0 + t1);
                for (int l = 0; l < B.m_; ++l) vv[s][l] = peval(Vv[l], tj);
                if (s == side) {
                    for (int l = 0; l < B.m_; ++l) pg[k][l] = Pv[j][l];
                    gtk[k] = (3 * Vv[j][3] * tj + 2 * Vv[j][2]) * tj + Vv[j][1];
                }
            }
            g[k] = vv[side][j];
            tjk[k] = tj;
            tmk[k] = tm[side];
            for (int i = 0; i < ny; ++i)                 // F_th (column k), raw nu coordinates
                Fth[i * ZC + k] = wbasis(i, tj, tm[0]) * vv[0][li[i]] - wbasis(i, tj, tm[1]) * vv[1][li[i]];
            ++k;
        }
    }
    if (J && k) {
        if (B.qf_)                                       // y = mu: F_i = mu_i + (Cf F_raw)_i, nu = Cf^T mu
            for (int i = 0; i < n; ++i)                  // row i only reads rows >= i: in place, ascending
                for (int a = 0; a < k; ++a) {
                    double f = 0.0;
                    for (int jj = i; jj < n; ++jj) f += B.Cf_[i * MAXN + jj] * Fth[jj * ZC + a];
                    Fth[i * ZC + a] = f;
                }
        if (!lu_solve(H_, ny, NY, Fth, k, ZC)) return false;   // H_ is rebuilt by the next evaluate
        for (int a = 0; a < k; ++a) {
            double gy[NY];                               // dv_j/dy at the junction: Pv[j][l] times the w basis
            for (int i = 0; i < ny; ++i) gy[i] = pg[a][li[i]] * wbasis(i, tjk[a], tmk[a]);
            if (B.qf_)
                for (int i = 0; i < n; ++i) {
                    double h = 0.0;
                    for (int jj = i; jj < n; ++jj) h += B.Cf_[i * MAXN + jj] * gy[jj];
                    gy[i] = h;
                }
            for (int b = 0; b < k; ++b) {
                double s = (a == b) ? gtk[a] : 0.0;
                for (int i = 0; i < ny; ++i) s -= gy[i] * Fth[i * ZC + b];
                J[a * k + b] = s;
            }
            if (dgs) {                                   // dg/ds at fixed junction times: -g_y zs
                double s = 0.0;
                for (int i = 0; i < ny; ++i) s -= gy[i] * zs[i];
                dgs[a] = s;
            }
        }
    }
    *ng = k;
    return true;
}

// (J^T J + lam I) d = -J^T g with a tiny lam: Newton when J (ng x nu, row-major) is regular; a junction
// that does not move its residual (e.g. entry inside a fully saturated arc, where v_j = 0 anyway) stays put
static bool lsq(const double* J, const double* g, int ng, int nu, double* N, double* d) {
    double dm = 0.0;
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
    return lu_solve(N, nu, nu, d, 1, 1);
}

// Path tangent of the continuation at a converged point (y, junction times) for the current bounds b(s):
// dth = d(free junction times)/ds and dy = dy/ds by the implicit function theorem through the inner KKT
// (F(y, th, s) = 0) and the junction conditions (g(y, th) = 0): J dth = -dg/ds|th, dy = -H^-1 (F_s + F_th dth).
bool StateSolver::tangent(const double* y, double* dth, double* dy) {
    const int n = base_.n_, ny = n + nsa_;
    double g[2 * MAXSA], dgs[2 * MAXSA], vs;
    int ng;
    // dy (first) = zs = H^-1 F_s; F_s: only the rows D_top(t1) - b(s) depend on s. gres factors H_ in place,
    // so H_ is built twice
    for (int i = 0; i < ny; ++i) dy[i] = (i >= n && !sa_[i - n].fix1) ? -dbds(sa_[i - n]) : 0.0;
    if (!evaluate(y, ev_[0], true) || !lu_solve(H_, ny, NY, dy, 1, 1)) return false;
    if (!evaluate(y, ev_[0], true) || !gres(y, g, &ng, &vs, H_, dy, dgs)) return false;
    if (ng && !lsq(H_, dgs, ng, ng, H_ + 4 * MAXSA * MAXSA, dth)) return false;
    for (int i = 0; i < ny; ++i) {
        double s = dy[i];
        for (int k = 0; k < ng; ++k) s += Z_[i * ZC + k] * dth[k];
        dy[i] = -s;
    }
    return true;
}

// Newton (exact Jacobian, gres) on the free junction times: dJ/dt = 0 at each (see gres).
// false with collapsed_ = q: arc q collapsed (t2 <= t1).
bool StateSolver::junctions(double* y, int max_its, int* outer) {
    const double T = base_.p_->T;
    const int ny = base_.n_ + nsa_;
    for (int it = 0; it < 40; ++it) {
        double* tv[2 * MAXSA];
        int nu = 0;
        for (int q = 0; q < nsa_; ++q) {
            if (!sa_[q].fix1 && !touch(sa_[q])) tv[nu++] = &sa_[q].t1;
            if (!sa_[q].fix2) tv[nu++] = &sa_[q].t2;
        }
        double* const J = H_;                            // J and N in H_ (dead until the next inner solve)
        double* const N = H_ + 4 * MAXSA * MAXSA;
        double g[2 * MAXSA], vs, y0[NY], tau0[2 * MAXSA], d[2 * MAXSA];
        int ng;
        if (!gres(y, g, &ng, &vs, J)) { strcpy(why_, "junction not found in the walk"); return false; }
        double gn = 0.0;
        for (int i = 0; i < ng; ++i) gn = fmax(gn, fabs(g[i]));
        if (gn <= 1e-10 * vs) return true;
        ++*outer;
        for (int i = 0; i < ny; ++i) y0[i] = y[i];
        for (int k = 0; k < nu; ++k) tau0[k] = *tv[k];
        if (!lsq(J, g, ng, nu, N, d)) { strcpy(why_, "singular junction Jacobian"); return false; }
        double zd[NY];                                   // Z d (Z_ is overwritten by the inner solves)
        for (int i = 0; i < ny; ++i) {
            double s = 0.0;
            for (int k = 0; k < nu; ++k) s += Z_[i * ZC + k] * d[k];
            zd[i] = s;
        }
        bool acc = false, fix2s[MAXSA];
        for (int q = 0; q < nsa_; ++q) fix2s[q] = sa_[q].fix2;
        for (double al = 1.0; al >= 1.0 / 256; al *= 0.5) {
            for (int q = 0; q < nsa_; ++q) sa_[q].fix2 = fix2s[q];
            for (int k = 0; k < nu; ++k) *tv[k] = tau0[k] + al * d[k];
            bool bad = false;
            for (int q = 0; q < nsa_; ++q) {
                StateArc& s = sa_[q];
                if (!s.fix2 && s.t2 >= T * (1 - 1e-9)) { s.t2 = T; s.fix2 = true; }
                if (s.fix2 && !s.fix1 && s.t1 >= T * (1 - 1e-9)) s.t1 = T;   // entry at T: terminal touch
                if (!s.fix1 && s.t1 <= 1e-9 * T) bad = true;
                for (int o = 0; o < q; ++o)
                    if (sa_[o].chain == s.chain && s.t1 < sa_[o].t2 && sa_[o].t1 < s.t2) bad = true;
                if (!touch(s) && s.t2 - s.t1 <= 1e-9 * T) {   // collapse only if no shorter step avoids it
                    if (al <= 1.0 / 256) { collapsed_ = q; return false; }
                    bad = true;
                }
            }
            if (bad) continue;
            for (int i = 0; i < ny; ++i) y[i] = y0[i] - al * zd[i];   // first-order predictor dy = -Z dth
            double gt[2 * MAXSA], vt;
            int nt;
            if (!inner(y, max_its) || !gres(y, gt, &nt, &vt)) break;   // shorter steps rarely help: fail fast
            double tn = 0.0;
            for (int i = 0; i < nt; ++i) tn = fmax(tn, fabs(gt[i]));
            if (nt != ng || tn < gn) { acc = true; break; }
        }
        if (!acc) {                                     // back to the last iterate, so the plan left behind is its plan
            for (int q = 0; q < nsa_; ++q) sa_[q].fix2 = fix2s[q];
            for (int k = 0; k < nu; ++k) *tv[k] = tau0[k];
            for (int i = 0; i < ny; ++i) y[i] = y0[i];
            inner(y, max_its);
            strcpy(why_, "junction Newton stalled");
            return false;
        }
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
    const double T = B.p_->T, b = bnd(j, side);
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
    release_ = -1;
    for (int j = 0; j < B.m_; ++j)
        for (int side = -1; side <= 1; side += 2) {
            const double b = bnd(j, side);
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
        double am[2 * MAXM], a1[2 * MAXM];
        if (!kkt(e.S[a], Wc, Pv, Vv, phi, has, sig, am)) { pp = HUGE_VAL; continue; }
        for (int k = 0; k < 2 * B.m_; ++k) a1[k] = am[k];
        rtol(e.S[a], Wc, phi, t0, am);                   // rounding of the row functions (as the walk)
        rtol(e.S[a], Wc, phi, t1, a1);
        for (int key = 0; key < 2 * B.m_; ++key) {
            if (!has[key]) continue;
            shift(phi[key], 4, t0, bb);
            bb[0] -= fmax(am[key], a1[key]);
            if (a && ((e.S[a] ^ e.S[a - 1]) >> key & 1u)) bb[0] = fmin(bb[0], 0.0);   // row of the event at t0: zero there
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
            // arc riding to T: sigma_j(T) is the terminal atom of the multiplier, side * sigma_j(T) >= 0;
            // otherwise the arc must leave the bound before T
            if (sa_[q].fix2 && t1 >= T && sa_[q].side * pval(bb, 4, L) < -TOL_SIGN * nrm * T) release_ = q;
        }
    }
    // terminal touch: its multiplier is the whole jump Delta, side * Delta <= 0 (as on every entry)
    for (int q = 0; q < nsa_; ++q)
        if (touch(sa_[q]) && sa_[q].side * y[B.n_ + q] > 1e-12) release_ = q;
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

void StateSolver::drop(int q, double* y, unsigned* gone) {
    *gone |= 1u << sa_[q].chain;
    for (; q + 1 < nsa_; ++q) { sa_[q] = sa_[q + 1]; y[base_.n_ + q] = y[base_.n_ + q + 1]; }
    --nsa_;
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

// cold start: tier-1 solve, its dual as the first iterate (its own function: the tier-1 Result is not
// on the stack during the junction iterations)
__attribute__((noinline)) bool StateSolver::cold(StateResult& r) {
    const Solver& B = base_;
    const int n = B.n_;
    double* const y = r.y;
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
    return true;
}

// Junction solver from the arcs and dual iterate in r (the corrector of the continuation). Iterations count
// into its_ / r.outer; the inner solves stop at the call's total budget cap_.
bool StateSolver::run(StateResult& r) {
    const Solver& B = base_;
    const int n = B.n_;
    const double T = B.p_->T;
    const int cap = MAX_NEWTON;
    double* const y = r.y;
    r.converged = false;
    r.cost = HUGE_VAL;
    r.state_violation = r.pmp_violation = r.sign_violation = HUGE_VAL;
    nsa_ = (r.nsa >= 0 && r.nsa <= MAXSA) ? r.nsa : 0;
    for (int q = 0; q < nsa_; ++q) sa_[q] = r.sa[q];
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
                    if (sa_[collapsed_].fix2) sa_[collapsed_].t1 = T;   // rode to T: becomes the terminal touch
                    else drop(collapsed_, y, &gone);
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
        if (release_ >= 0 && touch(sa_[release_])) {         // touch with the wrong sign: the bound is not active
            drop(release_, y, &gone);
            if (!inner(y, cap)) why = why_;
            continue;
        }
        if (release_ >= 0) {                                 // exit before T: free t2 just below T
            StateArc& s = sa_[release_];
            s.fix2 = false;
            s.t2 = T - 1e-2 * (T - s.t1);
            if (!inner(y, cap)) {
                strncpy(det, why_, sizeof det - 1); det[sizeof det - 1] = 0;
                snprintf(why_, sizeof why_, "arc order rejected: release before T on %c (%s)", CH[s.chain], det);
                why = why_;
            }
            continue;
        }
        if (ps > TOL_SIGN) {
            sequence(ev_[0].S, ev_[0].narcs, sa_, nsa_, ev_[0].tau, seq, sizeof seq);
            snprintf(why_, sizeof why_, "arc order rejected: %s (sign of the state multiplier)", seq);
            why = why_;
            break;
        }
        if (viol > TOL_X) {
            // a terminal touch on this chain grows into the arc; a violation next to an arc on the same bound
            // (gap shorter than the arc) extends that arc: the free arc between them dies
            int k = nsa_;
            for (int q = 0; q < nsa_; ++q) {
                const StateArc& a = sa_[q];
                if (a.chain != wc) continue;
                if (touch(a) && tb >= T) k = q;
                else if (a.side == ws && !touch(a) && fmax(a.t1 - tb, ta - a.t2) < a.t2 - a.t1) {
                    k = q;
                    ta = fmin(ta, a.t1);
                    tb = fmax(tb, a.t2);
                }
            }
            if (k == MAXSA) { why = "state arc capacity LQOCP_MAXSA exceeded"; break; }
            if (gone & (1u << wc)) {                         // the arc collapsed but the violation is back
                snprintf(why_, sizeof why_, "arc order rejected: %s (boundary arc on %c collapsed, violation remains)", seq, CH[wc]);
                why = why_;
                break;
            }
            for (int q = 0; q < nsa_; ++q)
                if (q != k && sa_[q].chain == wc && ta < sa_[q].t2 && tb > sa_[q].t1) why = "overlapping boundary arcs";
            if (why) break;
            // initial boundary arc: the violation interval [ta, tb]
            StateArc& s = sa_[k];
            s.chain = (int8_t)wc; s.side = (int8_t)ws; s.b = bnd(wc, ws);
            s.fix1 = ta <= 0.0; s.fix2 = tb >= T;
            s.t1 = s.fix1 ? 0.0 : ta;
            s.t2 = s.fix2 ? T : tb;
            if (k == nsa_) y[n + nsa_++] = 0.0;
            ++r.inserted;
            const bool ok = inner(y, cap);
            if (!ok) {
                strncpy(det, why_, sizeof det - 1); det[sizeof det - 1] = 0;
                snprintf(why_, sizeof why_, "arc order rejected: insertion on %c (%s)", CH[wc], det);
                why = why_;
            }
            continue;
        }
        if (pp > 1e-9) { why = "PMP verification failed"; break; }
        if (hiidx_ >= 0 && hs_ >= 1.0) {                    // homotopy: only the target problem must hold
            snprintf(why_, sizeof why_, "unsupported constraint order: flat state %d violates its bound (relative degree > 1)", hiidx_);
            why = why_;
            break;
        }
        conv = true;
        break;
    }
    if (!conv && !why) why = "outer iteration cap";
    if (!conv && its_ >= cap_) why = "iteration cap (total)";
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
    its_ = 0;
    cap_ = max_its;
    hs_ = 1.0;
    r.outer = 0;
    r.steps = 0;
    int k = 0;
    for (int q = 0; q < r.nsa && q < MAXSA; ++q) {
        StateArc a = r.sa[q];
        if (!a.fix2) a.t2 -= shift;
        if (a.t2 <= 1e-9 * T) continue;                  // ended: drop
        if (!touch(a)) a.t1 -= shift;
        a.b = bnd(a.chain, a.side);                      // the target (a stalled continuation leaves b(s < 1))
        double d = r.y[n + q];
        if (a.fix1 || a.t1 <= 1e-9 * T) {
            // starts at 0: only valid with x0 on the bound (no entry condition); otherwise drop it and
            // let the violation check re-insert it with a free entry
            const double top = base_.x0_[base_.off_[a.chain] + base_.p_->orders[a.chain] - 1];
            if (fabs(top - a.b) > TOL_X * (1.0 + fabs(a.b))) continue;
            a.t1 = 0.0; a.fix1 = true; d = 0.0;
        }
        r.sa[k] = a;
        r.y[n + k++] = d;
    }
    r.nsa = k;
    if (run(r) || its_ >= cap_) return r.converged;
    // the previous structure failed: continuation from the arc-free problem, inside the same budget
    r.nsa = nsa_ = 0;
    if (!inner(r.y, MAX_NEWTON)) {
        fill(r);
        r.iterations = its_;
        r.failure = its_ >= cap_ ? "iteration cap (total)" : why_;
        return false;
    }
    return homotopy(r);
}

// Excess of the tier-1 plan (dual y, no arcs) over each target bound: lo_ / hi_ become the bounds at s = 0
__attribute__((noinline)) void StateSolver::start(const double* y) {
    double c[MAXN], tw, ta, tb;
    nsa_ = 0;
    const bool ok = evaluate(y, ev_[0], false);
    cvec(y, c);
    for (int j = 0; j < base_.m_; ++j)
        for (int side = -1; side <= 1; side += 2) {
            double& b0 = side > 0 ? hi_[j] : lo_[j];
            b0 = tgt(j, side);
            if (ok && isfinite(b0)) b0 += side * fmax(0.0, scan(c, y, j, side, -1.0, &tw, &ta, &tb));
        }
}

// IFT predictor from the accepted point in r (current s) to s + h. Leaves r unchanged if the predicted
// junction times leave the valid region (arc shorter than 0, overlap, entry at 0).
__attribute__((noinline)) void StateSolver::predict(StateResult& r, double h) {
    const double T = base_.p_->T;
    const int n = base_.n_;
    double dth[2 * MAXSA], dy[NY];
    StateArc a[MAXSA];
    nsa_ = r.nsa;
    for (int q = 0; q < nsa_; ++q) sa_[q] = a[q] = r.sa[q];
    if (!nsa_ || !tangent(r.y, dth, dy)) return;
    for (int q = 0, k = 0; q < nsa_; ++q) {
        if (!a[q].fix1 && !touch(a[q])) a[q].t1 += dth[k++] * h;
        if (!a[q].fix2) a[q].t2 += dth[k++] * h;
        if (a[q].fix2 && a[q].t1 >= T * (1 - 1e-9)) a[q].t1 = T;   // entry reached T: terminal touch
        if ((!a[q].fix1 && a[q].t1 <= 1e-9 * T) || (!touch(a[q]) && a[q].t2 - a[q].t1 <= 1e-9 * T) ||
            (!a[q].fix2 && a[q].t2 >= T * (1 - 1e-9))) return;
        for (int o = 0; o < q; ++o)
            if (a[o].chain == a[q].chain && a[q].t1 < a[o].t2 && a[o].t1 < a[q].t2) return;
    }
    for (int q = 0; q < nsa_; ++q) r.sa[q] = a[q];
    for (int i = 0; i < n + nsa_; ++i) r.y[i] += dy[i] * h;
}

__attribute__((noinline)) void StateSolver::stalled(StateResult& r, double s) {   // own frame: det
    char det[200];
    strncpy(det, r.failure ? r.failure : "?", sizeof det - 1); det[sizeof det - 1] = 0;
    snprintf(why_, sizeof why_, "homotopy stalled at s = %.4g: %s", s, det);
    r.failure = why_;
}

// Continuation client (lqocp_cont.hpp) for the bound homotopy (NOTES.md 14.6): s = 0 is the arc-free (tier-1)
// solution in r.y, each violated bound starts at that plan's peak (start), b(s) = s target + (1 - s) peak.
// correct = the junction solver (run: Newton corrector, arc insertion at new violations = births, collapse /
// wrong sign = deaths) at b(s1); predict = the exact IFT tangent.
struct StateSolver::Hom {
    StateSolver& S;
    StateResult& r;
    int nb;                                          // last accepted point: nb arcs sb, dual yb
    StateArc sb[MAXSA];
    double yb[NY];
    void back() {
        r.nsa = nb;
        for (int q = 0; q < nb; ++q) r.sa[q] = sb[q];
        for (int i = 0; i < S.base_.n_ + nb; ++i) r.y[i] = yb[i];
    }
    void predict(double s, double s1) { S.predict(r, s1 - s); }
    bool correct(double s1) {
        S.hs_ = s1;
        for (int q = 0; q < r.nsa; ++q) r.sa[q].b = S.bnd(r.sa[q].chain, r.sa[q].side);
        ++r.steps;
        return S.run(r);
    }
    void accept() {
        nb = r.nsa;
        for (int q = 0; q < nb; ++q) sb[q] = r.sa[q];
        for (int i = 0; i < S.base_.n_ + nb; ++i) yb[i] = r.y[i];
    }
    bool spent() const { return S.its_ >= S.cap_; }
    void restart() { S.start(yb); }                 // the direct solve failed: bounds at s = 0
};

// From the arc-free solution in r.y (its_, cap_ set by the caller). On a stall r holds the last accepted
// point (optimal for the bounds b(s), s < 1), not converged.
__attribute__((noinline)) bool StateSolver::homotopy(StateResult& r) {
    Hom h{*this, r, 0};
    for (int i = 0; i < base_.n_; ++i) h.yb[i] = r.y[i];
    const double s = continuation(h);
    if (s < 1.0) {
        stalled(r, s);
        h.back();
        hs_ = s;
        nsa_ = r.nsa;
        for (int q = 0; q < nsa_; ++q) sa_[q] = r.sa[q];
        if (evaluate(r.y, ev_[0], false)) fill(r);
        r.converged = false;
    }
    hs_ = 1.0;
    r.iterations = its_;
    return r.converged;
}

// Cold solve: tier 1, then the bound continuation. The first step is s = 0 -> 1, i.e. the direct solve.
bool StateSolver::solve(StateResult& r) {
    hs_ = 1.0;
    its_ = 0;
    cap_ = 1 << 30;
    r.steps = 0;
    r.outer = 0;
    r.converged = false;
    r.cost = HUGE_VAL;
    if (!cold(r)) return false;
    return homotopy(r);
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
