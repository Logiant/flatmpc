// Solver core. See lqocp.hpp and develop/NOTES.md. No heap, no exceptions, no STL.
#include "lqocp.hpp"

#include <float.h>
#include <math.h>
#include <string.h>

namespace lqocp {

namespace {

// Round-off tolerances below are written for double as k * DBL_EPSILON; TS rescales them to
// the scalar type (1 in double, 2^29 in float). see NOTES 13.6.
const real EPS = sizeof(real) == 4 ? FLT_EPSILON : DBL_EPSILON;
#ifndef LQOCP_FTOL
#define LQOCP_FTOL 32
#endif
const real TS = sizeof(real) == 4 ? LQOCP_FTOL * FLT_EPSILON / 1e-12 : 1.0;

const real GL_X[4] = {-0.86113631159405257522, -0.33998104358485626480,
                        0.33998104358485626480, 0.86113631159405257522};
const real GL_W[4] = {0.34785484513745385737, 0.65214515486254614263,
                        0.65214515486254614263, 0.34785484513745385737};

real fact(int k) {
    real f = 1.0;
    for (int i = 2; i <= k; ++i) f *= i;
    return f;
}

real ipow(real x, int k) {
    real r = 1.0;
    for (int i = 0; i < k; ++i) r *= x;
    return r;
}

// In-place LU solve of A x = b (A is n x n stored with row stride ld). b overwritten by x.
// nrhs right-hand sides stored column-wise in B (row stride ldb). Returns false if singular.
template <class T>
bool lu_solve(T* A, int n, int ld, T* B, int nrhs, int ldb) {
    T amax = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) amax = fmax(amax, fabs(A[i * ld + j]));
    if (amax == T(0)) return false;
    for (int k = 0; k < n; ++k) {
        int piv = k;
        T best = fabs(A[k * ld + k]);
        for (int i = k + 1; i < n; ++i)
            if (fabs(A[i * ld + k]) > best) { best = fabs(A[i * ld + k]); piv = i; }
        if (best <= T(1e-300) || best < T(1e-15) * amax * T(1e-3)) return false;
        if (piv != k) {
            for (int j = 0; j < n; ++j) { T t = A[k * ld + j]; A[k * ld + j] = A[piv * ld + j]; A[piv * ld + j] = t; }
            for (int j = 0; j < nrhs; ++j) { T t = B[k * ldb + j]; B[k * ldb + j] = B[piv * ldb + j]; B[piv * ldb + j] = t; }
        }
        for (int i = k + 1; i < n; ++i) {
            T f = A[i * ld + k] / A[k * ld + k];
            if (f == T(0)) continue;
            for (int j = k + 1; j < n; ++j) A[i * ld + j] -= f * A[k * ld + j];
            for (int j = 0; j < nrhs; ++j) B[i * ldb + j] -= f * B[k * ldb + j];
        }
    }
    for (int j = 0; j < nrhs; ++j) {
        for (int i = n - 1; i >= 0; --i) {
            T s = B[i * ldb + j];
            for (int k = i + 1; k < n; ++k) s -= A[i * ld + k] * B[k * ldb + j];
            B[i * ldb + j] = s / A[i * ld + i];
        }
    }
    return true;
}

}  // namespace

bool Solver::setup(const Problem& p) {
    if (p.m < 1 || p.m > MAXM) return false;
    p_ = &p;
    m_ = p.m;
    n_ = 0;
    for (int j = 0; j < m_; ++j) {
        if (p.orders[j] < 1 || p.orders[j] > MAXK) return false;
        off_[j] = n_;
        n_ += p.orders[j];
    }
    if (n_ > MAXN || !(p.T > 0.0)) return false;
    // Q = 0: energy path (polynomial costate; any R, optional Qf); else general LQ cost
    general_ = false;
    rI_ = true;
    qf_ = p.has_cost && p.has_Qf;
    if (p.has_cost) {
        for (int i = 0; i < n_; ++i)
            for (int j = 0; j < n_; ++j)
                if (p.Q[i][j] != 0.0) general_ = true;
        for (int i = 0; i < m_; ++i)
            for (int j = 0; j < m_; ++j)
                if (p.R[i][j] != ((i == j) ? 1.0 : 0.0)) rI_ = false;
    }
    if (general_) return true;                      // general cost: use GSolver
    // R = Lc Lc^T: with vt = Lc^T v the cost is |vt|^2 and the rows act on vt (M Lc^-T)
    for (int i = 0; i < m_; ++i)
        for (int j = 0; j < m_; ++j) { Lc_[i][j] = 0.0; Lci_[i][j] = (i == j) ? 1.0 : 0.0; }
    if (!rI_) {
        for (int j = 0; j < m_; ++j) {
            real d = p.R[j][j];
            for (int k = 0; k < j; ++k) d -= Lc_[j][k] * Lc_[j][k];
            if (!(d > 0.0)) return false;           // R not SPD
            Lc_[j][j] = sqrt(d);
            for (int i = j + 1; i < m_; ++i) {
                real s = p.R[i][j];
                for (int k = 0; k < j; ++k) s -= Lc_[i][k] * Lc_[j][k];
                Lc_[i][j] = s / Lc_[j][j];
            }
        }
        for (int j = 0; j < m_; ++j)                 // Lci = Lc^-1, column j by forward substitution
            for (int i = 0; i < m_; ++i) {
                real s = (i == j) ? 1.0 : 0.0;
                for (int k = 0; k < i; ++k) s -= Lc_[i][k] * Lci_[k][j];
                Lci_[i][j] = s / Lc_[i][i];
            }
    }
    real Mrow[MAXM][MAXM];
    for (int r = 0; r < m_; ++r)
        for (int j = 0; j < m_; ++j) {
            real s = 0.0;
            for (int k = 0; k < m_; ++k) s += p.M[r][k] * Lci_[j][k];
            Mrow[r][j] = rI_ ? p.M[r][j] : s;
        }
    // Qf = Cf^T Cf (Cf = L^T, semidefinite Cholesky: a zero pivot leaves a zero row)
    for (int i = 0; i < MAXN * MAXN; ++i) Cf_[i] = 0.0;
    if (qf_) {
        real qmax = 0.0;
        for (int i = 0; i < n_; ++i) qmax = fmax(qmax, fabs(p.Qf[i][i]));
        for (int j = 0; j < n_; ++j) {
            real d = p.Qf[j][j];
            for (int k = 0; k < j; ++k) d -= Cf_[k * MAXN + j] * Cf_[k * MAXN + j];
            if (d < -1e-12 * TS * qmax) return false;    // Qf not PSD
            if (d <= 1e-14 * TS * qmax) continue;
            Cf_[j * MAXN + j] = sqrt(d);
            for (int i = j + 1; i < n_; ++i) {
                real s = p.Qf[i][j];
                for (int k = 0; k < j; ++k) s -= Cf_[k * MAXN + i] * Cf_[k * MAXN + j];
                Cf_[j * MAXN + i] = s / Cf_[j * MAXN + j];
            }
        }
    }
    rhs_scale_ = 1.0;
    for (int r = 0; r < m_; ++r) {
        for (int j = 0; j < m_; ++j) {
            rows_[2 * r][j] = Mrow[r][j];
            rows_[2 * r + 1][j] = -Mrow[r][j];
        }
        rhs_[2 * r] = p.hi[r];
        rhs_[2 * r + 1] = -p.lo[r];
        finite_[2 * r] = p.has_bounds && isfinite(p.hi[r]);
        finite_[2 * r + 1] = p.has_bounds && isfinite(p.lo[r]);
        if (finite_[2 * r]) rhs_scale_ = fmax(rhs_scale_, 1.0 + fabs(p.hi[r]));
        if (finite_[2 * r + 1]) rhs_scale_ = fmax(rhs_scale_, 1.0 + fabs(p.lo[r]));
    }
    // candidate active sets, sorted by size
    nsets_ = 0;
    int total = 1;
    for (int r = 0; r < m_; ++r) total *= 3;
    for (int sz = 0; sz <= m_; ++sz) {
        for (int code = 0; code < total; ++code) {
            uint16_t S = 0;
            int cc = code, cnt = 0;
            bool ok = true;
            for (int r = 0; r < m_; ++r) {
                int d = cc % 3;
                cc /= 3;
                if (d == 1) { S |= (uint16_t)(1u << (2 * r)); ++cnt; if (!finite_[2 * r]) ok = false; }
                if (d == 2) { S |= (uint16_t)(1u << (2 * r + 1)); ++cnt; if (!finite_[2 * r + 1]) ok = false; }
            }
            if (ok && cnt == sz) sets_[nsets_++] = S;
        }
    }
    // newton-mode data
    R_ = 1.0;
    real mx = 0.0;
    for (int k = 0; k < 2 * m_; ++k) {
        real s = 0.0;
        for (int j = 0; j < m_; ++j) s += rows_[k][j] * rows_[k][j];
        rownorm2_[k] = s;
        if (finite_[k]) mx = fmax(mx, fabs(rhs_[k]));
    }
    R_ = 1.0 + mx;
    identity_ = p.has_bounds != 0 && rI_;
    for (int r = 0; r < m_; ++r)
        for (int j = 0; j < m_; ++j)
            if (p.M[r][j] != ((r == j) ? 1.0 : 0.0)) identity_ = false;
    // c = Lm nu with w(t) = (e^{A(T-t)} B)^T nu
    for (int i = 0; i < MAXN * MAXN; ++i) { Lm_[i] = 0.0; Linv_[i] = 0.0; }
    for (int j = 0; j < m_; ++j) {
        const int k = p.orders[j], o = off_[j];
        for (int r = 0; r < k; ++r) {
            const int mm = k - 1 - r;
            for (int q = 0; q <= mm; ++q) {
                const real binom = fact(mm) / (fact(q) * fact(mm - q));
                Lm_[(o + q) * MAXN + (o + r)] =
                    binom * ipow(p.T, mm - q) * ((q % 2) ? -1.0 : 1.0) / fact(mm);
            }
        }
    }
    {
        real A[MAXN * MAXN];
        for (int i = 0; i < MAXN * MAXN; ++i) A[i] = Lm_[i];
        for (int i = 0; i < n_; ++i) Linv_[i * MAXN + i] = 1.0;
        if (!lu_solve(A, n_, MAXN, Linv_, n_, MAXN)) return false;
    }
    return set_x0(p.x0);
}

bool Solver::set_x0(const double* x0) {
    for (int i = 0; i < n_; ++i) {
        if (!isfinite(x0[i])) return false;
        x0_[i] = x0[i];
    }
    phi_x0((real)p_->T, free_);
    for (int i = 0; i < n_; ++i) {
        real s = 0.0;
        for (int j = i; j < n_; ++j) s += Cf_[i * MAXN + j] * ((real)p_->xf[j] - free_[j]);
        g_[i] = s;
    }
    return true;
}

void Solver::w_of(const real* c, real t, real* w) const {
    for (int j = 0; j < m_; ++j) {
        real s = 0.0, tp = 1.0;
        for (int k = 0; k < p_->orders[j]; ++k) { s += c[off_[j] + k] * tp; tp *= t; }
        w[j] = s;
    }
}

void Solver::phib(real dt, real K[MAXN][MAXM]) const {
    for (int i = 0; i < n_; ++i)
        for (int j = 0; j < m_; ++j) K[i][j] = 0.0;
    for (int j = 0; j < m_; ++j) {
        int k = p_->orders[j];
        for (int r = 0; r < k; ++r) K[off_[j] + r][j] = ipow(dt, k - 1 - r) / fact(k - 1 - r);
    }
}

void Solver::phi_x0(real dt, real* s) const {
    for (int j = 0; j < m_; ++j) {
        int k = p_->orders[j], o = off_[j];
        for (int r = 0; r < k; ++r) {
            real acc = 0.0;
            for (int l = r; l < k; ++l) acc += ipow(dt, l - r) / fact(l - r) * x0_[o + l];
            s[o + r] = acc;
        }
    }
}

bool Solver::affine(uint16_t S, real P[MAXM][MAXM], real q[MAXM]) const {
    int idx[MAXM], na = 0;
    for (int k = 0; k < 2 * m_; ++k)
        if (S & (1u << k)) { if (na >= MAXM) return false; idx[na++] = k; }
    for (int i = 0; i < m_; ++i) {
        q[i] = 0.0;
        for (int j = 0; j < m_; ++j) P[i][j] = (i == j) ? 1.0 : 0.0;
    }
    if (na == 0) return true;
    real G[MAXM * MAXM], X[MAXM * (MAXM + 1)];
    const int ldx = MAXM + 1;
    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < na; ++b) {
            real s = 0.0;
            for (int j = 0; j < m_; ++j) s += rows_[idx[a]][j] * rows_[idx[b]][j];
            G[a * MAXM + b] = s;
        }
        for (int j = 0; j < m_; ++j) X[a * ldx + j] = rows_[idx[a]][j];
        X[a * ldx + m_] = rhs_[idx[a]];
    }
    if (!lu_solve(G, na, MAXM, X, m_ + 1, ldx)) return false;
    for (int i = 0; i < m_; ++i) {
        for (int a = 0; a < na; ++a) {
            real ai = rows_[idx[a]][i];
            for (int j = 0; j < m_; ++j) P[i][j] -= ai * X[a * ldx + j];
            q[i] += ai * X[a * ldx + m_];
        }
    }
    return true;
}

bool Solver::pq(uint16_t S, real P[MAXM][MAXM], real q[MAXM]) const {
    if (!affine(S, P, q)) return false;
    if (rI_) return true;
    real Pt[MAXM][MAXM], qt[MAXM];               // back to v = Lc^-T vt
    for (int a = 0; a < m_; ++a) {
        qt[a] = q[a];
        for (int b = 0; b < m_; ++b) {
            real t = 0.0;
            for (int k = 0; k < m_; ++k) t += P[a][k] * Lci_[k][b];
            Pt[a][b] = t;
        }
    }
    for (int a = 0; a < m_; ++a) {
        real t = 0.0;
        for (int k = 0; k < m_; ++k) t += Lci_[k][a] * qt[k];
        q[a] = t;
        for (int b = 0; b < m_; ++b) {
            t = 0.0;
            for (int k = 0; k < m_; ++k) t += Lci_[k][a] * Pt[k][b];
            P[a][b] = t;
        }
    }
    return true;
}

bool Solver::sets_ok(const uint16_t* S, int n) const {
    real P[MAXM][MAXM], q[MAXM];
    for (int i = 0; i < n; ++i)
        if (!affine(S[i], P, q)) return false;
    return true;
}

void Solver::v_on(const real P[MAXM][MAXM], const real* q, const real* c, real t, real* v) const {
    real w[MAXM];
    w_of(c, t, w);
    for (int i = 0; i < m_; ++i) {
        real s = q[i];
        for (int j = 0; j < m_; ++j) s += P[i][j] * w[j];
        v[i] = s;
    }
}

void Solver::assemble(const uint16_t* S, const real* tau, int n, real* G, real* h) const {
    // s(T) = G c + h ; G is also the exact Jacobian ds(T)/dc. Row i of e^{A(T-t)} B has one
    // nonzero, (T-t)^(k-1-r)/(k-1-r)! in column l (state i = entry r of chain l, order k).
    real P[MAXM][MAXM], q[MAXM], KP[MAXM], tp[MAXK];
    memset(G, 0, sizeof(real) * MAXN * MAXN);
    for (int i = 0; i < n_; ++i) h[i] = free_[i];
    for (int a = 0; a < n; ++a) {
        real t0, t1;
        span(tau, n, a, &t0, &t1);
        if (!(t1 > t0)) continue;
        pq(S[a], P, q);
        const real hw = 0.5 * (t1 - t0);
        for (int g = 0; g < 4; ++g) {
            const real t = t0 + hw * (GL_X[g] + 1.0);
            const real wg = GL_W[g] * hw;
            tp[0] = 1.0;
            for (int k = 1; k < MAXK; ++k) tp[k] = tp[k - 1] * t;
            for (int l = 0; l < m_; ++l) {
                const int k = p_->orders[l];
                for (int r = 0; r < k; ++r) {
                    const int i = off_[l] + r;
                    const real K = ipow((real)p_->T - t, k - 1 - r) / fact(k - 1 - r);
                    for (int j = 0; j < m_; ++j) KP[j] = wg * (K * P[l][j]);
                    h[i] += wg * (K * q[l]);
                    for (int j = 0; j < m_; ++j)
                        for (int kk = 0; kk < p_->orders[j]; ++kk) G[i * MAXN + off_[j] + kk] += KP[j] * tp[kk];
                }
            }
        }
    }
}

// uses G_ and W_
bool Solver::solve_c(const uint16_t* S, const real* tau, int n, real* c) const {
    real h[MAXN];
    real* M = G_;
    assemble(S, tau, n, G_, h);
    for (int i = 0; i < n_; ++i) h[i] = (real)p_->xf[i] - h[i];
    if (qf_) {
        real b[MAXN];
        for (int i = 0; i < n_; ++i) {
            b[i] = 0.0;
            for (int k = 0; k < n_; ++k) b[i] += (real)p_->Qf[i][k] * h[k];
            for (int j = 0; j < n_; ++j) {
                real t = Linv_[i * MAXN + j];
                for (int k = 0; k < n_; ++k) t += (real)p_->Qf[i][k] * G_[k * MAXN + j];
                W_[i * MAXN + j] = t;
            }
        }
        for (int i = 0; i < n_; ++i) h[i] = b[i];
        M = W_;
    }
    if (!lu_solve(M, n_, MAXN, h, 1, 1)) return false;
    for (int i = 0; i < n_; ++i) c[i] = h[i];
    return true;
}

real Solver::cost(const uint16_t* S, const real* tau, int n, const real* c) const {
    real J = 0.0, v[MAXM], P[MAXM][MAXM], q[MAXM];
    for (int a = 0; a < n; ++a) {
        real t0, t1;
        span(tau, n, a, &t0, &t1);
        pq(S[a], P, q);
        const real hw = 0.5 * (t1 - t0);
        for (int g = 0; g < 4; ++g) {
            v_on(P, q, c, t0 + hw * (GL_X[g] + 1.0), v);
            real s = 0.0;
            for (int j = 0; j < m_; ++j) s += rI_ ? v[j] * v[j] : v[j] * vR(j, v);
            J += 0.5 * GL_W[g] * hw * s;
        }
    }
    if (qf_) {
        real h[MAXN], e[MAXN];
        assemble(S, tau, n, G_, h);
        for (int i = 0; i < n_; ++i) {
            e[i] = h[i] - (real)p_->xf[i];
            for (int j = 0; j < n_; ++j) e[i] += G_[i * MAXN + j] * c[j];
        }
        for (int i = 0; i < n_; ++i)
            for (int j = 0; j < n_; ++j) J += 0.5 * e[i] * (real)p_->Qf[i][j] * e[j];
    }
    return J;
}

#ifndef LQOCP_NO_TAU
bool Solver::residual(const uint16_t* S, int nseq, const real* tau, real* res, int* nres,
                      real* c) const {
    if (!solve_c(S, tau, nseq, c)) return false;
    int nr = 0;
    real v[MAXM], P[MAXM][MAXM], q[MAXM];
    for (int k = 0; k + 1 < nseq; ++k) {
        const uint16_t L = S[k], R = S[k + 1], X = (uint16_t)(L ^ R);
        for (int key = 0; key < 2 * m_; ++key) {
            if (!(X & (1u << key))) continue;
            pq((R & (1u << key)) ? L : R, P, q);   // the side where key is inactive
            v_on(P, q, c, tau[k], v);
            real g = -rhs_[key];
            for (int j = 0; j < m_; ++j) {
                real vt = rI_ ? v[j] : 0.0;
                if (!rI_) for (int i = j; i < m_; ++i) vt += Lc_[i][j] * v[i];
                g += rows_[key][j] * vt;
            }
            if (nr >= MAXRES) return false;
            res[nr++] = g;
        }
    }
    *nres = nr;
    return true;
}

bool Solver::newton(const uint16_t* S, int nseq, real* tau, real* c, int* its, real* resmax) const {
    const int nj = nseq - 1;
    const real tol = 1e-12;
    const real T = (real)p_->T;
    real r[MAXRES], rp[MAXRES], rm[MAXRES], rn[MAXRES];
    real J[MAXRES][MAXJ_TAU];
    real tn[MAXJ_TAU], tt[MAXJ_TAU], cn[MAXN], ctmp[MAXN];
    int nr, nr2;
    *its = 0;
    if (!residual(S, nseq, tau, r, &nr, c)) return false;
    real rmax = 0.0;
    for (int i = 0; i < nr; ++i) rmax = fmax(rmax, fabs(r[i]));
    *resmax = rmax;
    if (nj == 0) return true;
    const real gap = 1e-9 * T;
    for (int it = 1; it <= MAX_NEWTON; ++it) {
        if (rmax < tol) return true;
        *its = it;
        const real hstep = 1e-7 * fmax(1.0, T);
        for (int i = 0; i < nj; ++i) {
            for (int q = 0; q < nj; ++q) tt[q] = tau[q];
            tt[i] = tau[i] + hstep;
            if (!residual(S, nseq, tt, rp, &nr2, ctmp) || nr2 != nr) return false;
            tt[i] = tau[i] - hstep;
            if (!residual(S, nseq, tt, rm, &nr2, ctmp) || nr2 != nr) return false;
            for (int q = 0; q < nr; ++q) J[q][i] = (rp[q] - rm[q]) / (2.0 * hstep);
        }
        // Gauss-Newton step from the normal equations (J^T J) d = -J^T r
        real N[MAXJ_TAU * MAXJ_TAU], d[MAXJ_TAU];
        for (int a = 0; a < nj; ++a) {
            real s = 0.0;
            for (int q = 0; q < nr; ++q) s += J[q][a] * r[q];
            d[a] = -s;
            for (int b = 0; b < nj; ++b) {
                real g = 0.0;
                for (int q = 0; q < nr; ++q) g += J[q][a] * J[q][b];
                N[a * MAXJ_TAU + b] = g;
            }
        }
        if (!lu_solve(N, nj, MAXJ_TAU, d, 1, 1)) return false;
        real nr0 = 0.0;
        for (int q = 0; q < nr; ++q) nr0 += r[q] * r[q];
        nr0 = sqrt(nr0);
        real alpha = 1.0;
        bool accepted = false;
        while (alpha > 1e-6) {
            bool ordered = true;
            for (int a = 0; a < nj; ++a) tn[a] = tau[a] + alpha * d[a];
            real prev = 0.0;
            for (int a = 0; a <= nj; ++a) {
                const real cur = (a < nj) ? tn[a] : T;
                if (!(cur - prev > gap)) ordered = false;
                prev = cur;
            }
            if (ordered && residual(S, nseq, tn, rn, &nr2, cn) && nr2 == nr) {
                real nn = 0.0;
                for (int q = 0; q < nr; ++q) nn += rn[q] * rn[q];
                nn = sqrt(nn);
                if (nn < (1.0 - 1e-4 * alpha) * nr0 || nn < tol) { accepted = true; break; }
            }
            alpha *= 0.5;
        }
        if (!accepted) return false;
        rmax = 0.0;
        for (int a = 0; a < nj; ++a) tau[a] = tn[a];
        for (int q = 0; q < nr; ++q) { r[q] = rn[q]; rmax = fmax(rmax, fabs(rn[q])); }
        for (int i = 0; i < n_; ++i) c[i] = cn[i];
        *resmax = rmax;
    }
    return rmax < tol;
}
#endif

static const char* const CAPFAIL = "arc capacity LQOCP_MAXARC_E exceeded";

bool Solver::solve(Result& R, int method) { return start(R, method, -1); }
bool Solver::solve_warm(Result& R, int max_its) { return start(R, METHOD_NEWTON, max_its); }

bool Solver::start(Result& R, int method, int warm_its) {
    R.method = method;
    R.converged = false;
    R.failure = "general cost: use GSolver";
    if (general_) return false;                       // general cost: use GSolver
    R.backtracks = 0;
    R.degenerate_events = 0;
    R.box_fast_path = false;
    capfail_ = false;
    if (method == METHOD_NEWTON) return solve_newton(R, warm_its);
#ifdef LQOCP_NO_TAU
    R.failure = "tau method not built (LQOCP_NO_TAU)";
    return false;
#else
    return solve_tau(R);
#endif
}

#ifndef LQOCP_NO_TAU
// Active-set sequence for the tau method: the exact event walk of the newton method (per arc and
// row, the cubic gap's roots in closed form; section 7.3). Fails if the walk fails or needs more
// junctions than the tau method supports.
bool Solver::detect_exact(const real* c) {
    const Eval& e = ew_.ev[0];
    if (!walk(c, ew_.ev[0]) || e.narcs > MAXJ_TAU + 1) return false;
    ew_.nseq2 = e.narcs;
    for (int i = 0; i < ew_.nseq2; ++i) ew_.seq2[i] = e.S[i];
    for (int i = 0; i + 1 < ew_.nseq2; ++i) ew_.tau2[i] = e.tau[i];
    return true;
}

bool Solver::solve_tau(Result& R) {
    R.iterations = 0;
    R.newton_iterations = 0;
    R.outer_iterations = 0;
    R.residual = 0.0;
    R.pmp_violation = 0.0;
    R.narcs = 0;
    R.failure = "tau: setup";
    // 1) unconstrained primitive
    ew_.nseq = 1;
    ew_.seq[0] = 0;
    if (!solve_c(ew_.seq, ew_.tau, ew_.nseq, R.c)) return false;
    R.iterations = 1;
    bool conv = !p_->has_bounds;
    if (p_->has_bounds) {
        real c_try[MAXN];
        real c0n = 1.0;
        for (int i = 0; i < n_; ++i) c0n = fmax(c0n, 1.0 + fabs(R.c[i]));
        for (int outer = 1; outer <= MAX_OUTER; ++outer) {
            R.outer_iterations = outer;
            // early exits: iteration cap, diverging w (dual unbounded: infeasible)
            if (R.newton_iterations > MAX_TAU_TOTAL) break;
            real cn = 0.0;
            for (int i = 0; i < n_; ++i) cn = fmax(cn, fabs(R.c[i]));
            if (cn > 1e6 * c0n) break;
            // 2) active-set sequence from the pointwise H-minimiser of the current w(t)
            if (!detect_exact(R.c)) break;
            if (outer == 1 && ew_.nseq2 == 1 && ew_.seq2[0] == 0) { conv = true; break; }
            // 3) junction-time Newton with inner linear BC solve
            int its = 0;
            real rmax = 0.0;
            bool ok = newton(ew_.seq2, ew_.nseq2, ew_.tau2, c_try, &its, &rmax);
            R.newton_iterations += its;
            R.iterations += 1 + its;
            R.residual = rmax;
            if (ok) {
                ew_.nseq = ew_.nseq2;
                for (int i = 0; i < ew_.nseq; ++i) ew_.seq[i] = ew_.seq2[i];
                for (int i = 0; i + 1 < ew_.nseq; ++i) ew_.tau[i] = ew_.tau2[i];
                for (int i = 0; i < n_; ++i) R.c[i] = c_try[i];
                R.pmp_violation = verify(ew_.seq, ew_.tau, ew_.nseq, R.c);
                if (R.pmp_violation <= 1e-9 * R_) { conv = true; break; }
            } else {
                // fixed-point step: fix detected junctions, solve c, re-detect next round
                if (!detect_exact(R.c)) break;
                if (!solve_c(ew_.seq2, ew_.tau2, ew_.nseq2, c_try)) break;
                ew_.nseq = ew_.nseq2;
                for (int i = 0; i < ew_.nseq; ++i) ew_.seq[i] = ew_.seq2[i];
                for (int i = 0; i + 1 < ew_.nseq; ++i) ew_.tau[i] = ew_.tau2[i];
                for (int i = 0; i < n_; ++i) R.c[i] = c_try[i];
            }
        }
    }
    R.narcs = ew_.nseq;
    for (int i = 0; i < ew_.nseq; ++i) R.S[i] = ew_.seq[i];
    for (int i = 0; i + 1 < ew_.nseq; ++i) R.tau[i] = ew_.tau[i];
    R.cost = cost(R.S, R.tau, R.narcs, R.c);
    R.converged = conv;
    R.failure = conv ? nullptr : capfail_ ? CAPFAIL : "tau method did not converge";
    return conv;
}
#endif

void Solver::eval_u(const Result& R, double t, double u[MAXM]) const {
    if (t < 0.0) t = 0.0;
    if (t > (real)p_->T) t = (real)p_->T;
    int a = 0;
    while (a < R.narcs - 1 && t >= R.tau[a]) ++a;
    real P[MAXM][MAXM], q[MAXM];
    pq(R.S[a], P, q);
    real v[MAXM];
    v_on(P, q, R.c, (real)t, v);
    for (int j = 0; j < m_; ++j) u[j] = v[j];
}

void Solver::eval_x(const Result& R, double td, double xd[MAXN]) const {
    real t = (real)td, x[MAXN];
    if (t < 0.0) t = 0.0;
    if (t > (real)p_->T) t = (real)p_->T;
    phi_x0(t, x);
    real K[MAXN][MAXM], v[MAXM], P[MAXM][MAXM], q[MAXM];
    for (int a = 0; a < R.narcs; ++a) {
        real t0, t1;
        span(R.tau, R.narcs, a, &t0, &t1);
        if (t0 >= t) break;
        if (t1 > t) t1 = t;
        pq(R.S[a], P, q);
        const real hw = 0.5 * (t1 - t0);
        for (int g = 0; g < 4; ++g) {
            const real tt = t0 + hw * (GL_X[g] + 1.0);
            v_on(P, q, R.c, tt, v);
            phib(t - tt, K);
            for (int i = 0; i < n_; ++i) {
                real s = 0.0;
                for (int j = 0; j < m_; ++j) s += K[i][j] * v[j];
                x[i] += GL_W[g] * hw * s;
            }
        }
    }
    for (int i = 0; i < n_; ++i) xd[i] = x[i];
}

bool Solver::mean_u(const Result& R, double t0d, double t1d, double u[MAXM]) const {
    const real t0 = (real)t0d, t1 = (real)t1d;
    if (R.narcs < 1 || !(t1 > t0)) return false;
    real acc[MAXM] = {0.0, 0.0, 0.0, 0.0}, v[MAXM], P[MAXM][MAXM], q[MAXM];
    for (int a = 0; a < R.narcs; ++a) {             // v is a cubic per arc: 4-point Gauss is exact
        real a0, a1;
        span(R.tau, R.narcs, a, &a0, &a1);
        const real lo = fmax(t0, a0), hi = fmin(t1, a1);
        if (!(hi > lo)) continue;
        pq(R.S[a], P, q);
        const real hw = 0.5 * (hi - lo);
        for (int g = 0; g < 4; ++g) {
            v_on(P, q, R.c, lo + hw * (GL_X[g] + 1.0), v);
            for (int j = 0; j < m_; ++j) acc[j] += GL_W[g] * hw * v[j];
        }
    }
    for (int j = 0; j < m_; ++j) u[j] = acc[j] / (t1 - t0);
    return true;
}

int Solver::switch_times(const Result& R, double* out, int maxout) const {
    int k = 0;
    for (int a = 0; a + 1 < R.narcs && k < maxout; ++a) out[k++] = R.tau[a];
    return k;
}

// =====================================================================================
// METHOD_NEWTON: Newton on c with exact (grid-free) event detection. Mirrors
// develop/python/lqocp.py (_NewtonCtx / _solve_newton).
// =====================================================================================
namespace {

real pval(const real* b, int len, real s) {
    real r = 0.0;
    for (int i = len - 1; i >= 0; --i) r = r * s + b[i];
    return r;
}

// coefficients of p(t0 + s) from those of p(t) (len <= 4)
void shift(const real* a, int len, real t0, real* b) {
    for (int i = 0; i < len; ++i) {
        real s = 0.0;
        for (int p = i; p < len; ++p) s += a[p] * (fact(p) / (fact(i) * fact(p - i))) * ipow(t0, p - i);
        b[i] = s;
    }
}

real pscale(const real* b, int len, real L) {
    real s = 0.0;
    for (int i = 0; i < len; ++i) s += fabs(b[i]) * ipow(L, i);
    return s;
}

// Divide out the root at s=0 (drop leading coefficients with |b_i| L^i <= tol).
// Returns reduced length (0 if ~identically zero); k = multiplicity removed.
int strip(const real* b, int len, real L, real tol, real* red, int* kout) {
    int k = 0;
    while (k < len && fabs(b[k]) * ipow(L, k) <= tol) ++k;
    *kout = k;
    if (k == len) return 0;
    int rl = len - k;
    for (int i = 0; i < rl; ++i) red[i] = b[k + i];
    while (rl > 1 && fabs(red[rl - 1]) * ipow(L, rl - 1 + k) <= tol * EPS) --rl;
    return rl;
}

// real roots of a degree <= 2 polynomial (ascending), sorted; returns count
int roots_le2(const real* d0, int len, real* out) {
    real d[3] = {0, 0, 0};
    for (int i = 0; i < len && i < 3; ++i) d[i] = d0[i];
    while (len > 0 && d[len - 1] == 0.0) --len;
    if (len <= 1) return 0;
    if (len == 2) { out[0] = -d[0] / d[1]; return 1; }
    const real c0 = d[0], b1 = d[1], a2 = d[2];
    const real disc = b1 * b1 - 4 * a2 * c0;
    if (disc < 0) {
        if (disc > -1e-14 * TS * (b1 * b1 + fabs(4 * a2 * c0))) { out[0] = -b1 / (2 * a2); return 1; }
        return 0;
    }
    const real sq = sqrt(disc);
    const real qq = -0.5 * (b1 + (b1 >= 0 ? sq : -sq));
    if (qq == 0.0) { out[0] = 0.0; return 1; }
    real r0 = qq / a2, r1 = c0 / qq;
    if (r0 > r1) { real t = r0; r0 = r1; r1 = t; }
    out[0] = r0; out[1] = r1;
    return 2;
}

real bisect(const real* r, int len, real a, real b) {
    real fa = pval(r, len, a);
    for (int it = 0; it < 200; ++it) {
        const real mid = 0.5 * (a + b);
        if (mid <= a || mid >= b) break;
        const real fm = pval(r, len, mid);
        if (((fm > 0) == (fa > 0)) && fm != 0.0) { a = mid; fa = fm; }
        else b = mid;
    }
    return 0.5 * (a + b);
}

struct Cross {
    int n;
    real root[4];
    bool simple[4];
};

// Sign changes of phi(s) = s^k red(s) on (0, L] (tangential touches are not events).
void crossings(const real* red, int len, int k, real L, real tol, Cross& o) {
    o.n = 0;
    if (len <= 1) return;
    real d[3];
    for (int i = 1; i < len; ++i) d[i - 1] = i * red[i];
    real cr[2];
    const int nc = roots_le2(d, len - 1, cr);
    real pts[4], vals[4];
    int np = 0;
    pts[np++] = 0.0;
    for (int i = 0; i < nc; ++i)
        if (cr[i] > 0.0 && cr[i] < L && (np == 1 || cr[i] != pts[np - 1])) pts[np++] = cr[i];
    pts[np++] = L;
    vals[0] = red[0];
    for (int i = 1; i < np; ++i) vals[i] = pval(red, len, pts[i]);
    int zeros[4], nz = 0;
    real sig = vals[0] > 0 ? 1.0 : -1.0;
    for (int i = 1; i < np; ++i) {
        const real v = vals[i], x = pts[i];
        if (fabs(v) * ipow(x, k) <= tol) {
            if (i < np - 1) zeros[nz++] = i;
            continue;
        }
        const real sg = v > 0 ? 1.0 : -1.0;
        if (sg != sig) {
            if (vals[i - 1] * v <= 0) {
                o.root[o.n] = bisect(red, len, pts[i - 1], x);
                o.simple[o.n] = (nz == 0);
            } else {
                o.root[o.n] = pts[zeros[nz - 1]];          // odd root at a flat point
                o.simple[o.n] = false;
            }
            ++o.n;
            sig = sg;
        }
        nz = 0;
    }
}

}  // namespace

bool Solver::valid_set(uint16_t S) const {
    int cnt = 0;
    for (int r = 0; r < m_; ++r) {
        const bool up = S & (1u << (2 * r)), lo = S & (1u << (2 * r + 1));
        if (up && lo) return false;
        if (up && !finite_[2 * r]) return false;
        if (lo && !finite_[2 * r + 1]) return false;
        cnt += up || lo;
    }
    return cnt <= m_;
}

void Solver::row_polys(uint16_t S, const real* c, real phi[2 * MAXM][4],
                       bool has[2 * MAXM]) const {
    real Wc[MAXM][4];
    for (int j = 0; j < m_; ++j)
        for (int p = 0; p < 4; ++p) {
            real s = 0.0;
            for (int k = 0; k <= j; ++k) s += Lci_[j][k] * ((p < p_->orders[k]) ? c[off_[k] + p] : 0.0);
            Wc[j][p] = s;
        }
    int idx[MAXM], na = 0;
    {                                                 // own scope: stack shared with the block below
        real P[MAXM][MAXM], q[MAXM], V[MAXM][4];
        affine(S, P, q);
        for (int i = 0; i < m_; ++i)
            for (int p = 0; p < 4; ++p) {
                real s = (p == 0) ? q[i] : 0.0;
                for (int j = 0; j < m_; ++j) s += P[i][j] * Wc[j][p];
                V[i][p] = s;
            }
        for (int key = 0; key < 2 * m_; ++key) {
            has[key] = false;
            if (!finite_[key]) continue;
            has[key] = true;
            if (S & (1u << key)) { idx[na++] = key; continue; }
            for (int p = 0; p < 4; ++p) {
                real s = 0.0;
                for (int j = 0; j < m_; ++j) s += rows_[key][j] * V[j][p];
                phi[key][p] = s - ((p == 0) ? rhs_[key] : 0.0);
            }
        }
    }
    if (na > 0) {
        real G[MAXM * MAXM], X[MAXM * 4];
        for (int a = 0; a < na; ++a) {
            for (int b = 0; b < na; ++b) {
                real g = 0.0;
                for (int j = 0; j < m_; ++j) g += rows_[idx[a]][j] * rows_[idx[b]][j];
                G[a * MAXM + b] = g;
            }
            for (int p = 0; p < 4; ++p) {
                real s = 0.0;
                for (int j = 0; j < m_; ++j) s += rows_[idx[a]][j] * Wc[j][p];
                X[a * 4 + p] = s - ((p == 0) ? rhs_[idx[a]] : 0.0);
            }
        }
        if (!lu_solve(G, na, MAXM, X, 4, 4)) {
            for (int a = 0; a < na; ++a) has[idx[a]] = false;
            return;
        }
        for (int a = 0; a < na; ++a)
            for (int p = 0; p < 4; ++p) phi[idx[a]][p] = -X[a * 4 + p] * rownorm2_[idx[a]];
    }
}

bool Solver::lex_ok(uint16_t S, const real* c, real t, real phi[2 * MAXM][4], bool has[2 * MAXM]) const {
    real b[4];
    row_polys(S, c, phi, has);
    const real T = (real)p_->T;
    for (int key = 0; key < 2 * m_; ++key) {
        if (!has[key]) continue;
        shift(phi[key], 4, t, b);
        const real tol = 1e-12 * TS * (pscale(b, 4, T) + R_);
        for (int i = 0; i < 4; ++i) {
            if (fabs(b[i]) * ipow(T, i) > tol) {
                if (b[i] > 0) return false;
                break;
            }
        }
    }
    return true;
}

bool Solver::lex_set(const real* c, real t, uint16_t guess, uint16_t* out, real phi[2 * MAXM][4],
                     bool has[2 * MAXM]) const {
    if (guess != 0xFFFFu && valid_set(guess) && lex_ok(guess, c, t, phi, has)) { *out = guess; return true; }
    for (int i = 0; i < nsets_; ++i)
        if (lex_ok(sets_[i], c, t, phi, has)) { *out = sets_[i]; return true; }
    return false;
}

bool Solver::walk(const real* c, Eval& e) {
    e.ndeg = 0;
    if (identity_) return walk_box(c, e);
    const real T = (real)p_->T;
    // G_ is free during the walk (evaluate refills it after); phi is also lex_set's scratch (dead there)
    real (*const phi)[4] = reinterpret_cast<real (*)[4]>(G_);
    bool has[2 * MAXM];
    uint16_t S;
    if (!lex_set(c, 0.0, 0, &S, phi, has)) return false;   // (a) initial set: one-sided KKT at 0+
    int ns = 0, stall = 0;
    e.S[ns++] = S;
    real t = 0.0;
    for (;;) {
        const real L = T - t;
        row_polys(S, c, phi, has);
        real rootk[2 * MAXM];
        bool simplek[2 * MAXM], evk[2 * MAXM];
        real s1 = HUGE_VAL;
        for (int key = 0; key < 2 * m_; ++key) {
            evk[key] = false;
            if (!has[key]) continue;
            real b[4], red[4];
            int k;
            shift(phi[key], 4, t, b);
            const real tol = 1e-12 * TS * (pscale(b, 4, T) + R_);   // as lex_ok: no stall on a tiny last arc
            const int rl = strip(b, 4, L, tol, red, &k);
            if (rl == 0) continue;
            if (red[0] > 0) {                            // inconsistent at t+: degenerate event
                evk[key] = true; rootk[key] = 0.0; simplek[key] = false;
                s1 = 0.0;
                continue;
            }
            Cross cr;
            crossings(red, rl, k, L, tol, cr);
            if (cr.n) {
                evk[key] = true; rootk[key] = cr.root[0]; simplek[key] = cr.simple[0];
                if (cr.root[0] < s1) s1 = cr.root[0];
            }
        }
        if (s1 == (real)HUGE_VAL || t + s1 >= T * (1 - 1e-14 * TS)) break;
        int ngroup = 0, gkey = -1;
        bool gsimple = false;
        uint16_t gmask = 0;
        for (int key = 0; key < 2 * m_; ++key)
            if (evk[key] && rootk[key] <= s1 + 1e-12 * TS * T) {
                ++ngroup; gkey = key; gsimple = simplek[key]; gmask |= (uint16_t)(1u << key);
            }
        t += s1;
        uint16_t Snew = 0xFFFFu;
        if (ngroup == 1 && gsimple) {
            // control continuity: add the row whose gap hit 0 / drop the row whose mu hit 0
            const uint16_t cand = (uint16_t)(S ^ (1u << gkey));
            if (valid_set(cand)) Snew = cand;
        }
        if (Snew == 0xFFFFu) {                           // (b) degenerate event
            ++e.ndeg;
            const uint16_t guess = (uint16_t)(S ^ gmask);
            if (!lex_set(c, t, valid_set(guess) ? guess : 0xFFFFu, &Snew, phi, has)) return false;
        }
        if (Snew != S) {
            if (ns >= MAXARC_E) { capfail_ = true; return false; }
            e.tau[ns - 1] = t;
            e.S[ns++] = Snew;
            S = Snew;
            stall = 0;
        } else if (++stall > 3) {
            return false;
        }
    }
    e.narcs = ns;
    return sets_ok(e.S, ns);
}

bool Solver::walk_box(const real* c, Eval& e) {
    // Umap = identity: v_j = clip(w_j, lo_j, hi_j) per chain, events = roots of w_j - bound.
    const real T = (real)p_->T;
    real times[MAXARC_E];
    int nt = 0;
    for (int j = 0; j < m_; ++j) {
        for (int side = 0; side < 2; ++side) {
            const int key = 2 * j + side;
            if (!finite_[key]) continue;
            const real sgn = side ? -1.0 : 1.0;
            real b[4], red[4];
            for (int p = 0; p < 4; ++p) b[p] = (p < p_->orders[j]) ? sgn * c[off_[j] + p] : 0.0;
            b[0] -= rhs_[key];                       // rhs_ = hi (upper) or -lo (lower)
            const real tol = 1e-12 * TS * (pscale(b, 4, T) + R_);
            int k;
            const int rl = strip(b, 4, T, tol, red, &k);
            if (rl == 0) continue;
            Cross cr;
            crossings(red, rl, k, T, tol, cr);
            for (int i = 0; i < cr.n; ++i) {
                if (cr.root[i] >= T * (1 - 1e-14 * TS)) continue;
                bool dup = false;
                for (int q = 0; q < nt; ++q) if (times[q] == cr.root[i]) dup = true;
                if (!dup) { if (nt >= MAXARC_E) { capfail_ = true; return false; } times[nt++] = cr.root[i]; }
            }
        }
    }
    for (int i = 1; i < nt; ++i) {                   // insertion sort
        real x = times[i];
        int q = i - 1;
        while (q >= 0 && times[q] > x) { times[q + 1] = times[q]; --q; }
        times[q + 1] = x;
    }
    int ns = 0;
    real w[MAXM];
    for (int i = 0; i <= nt; ++i) {
        const real a = (i == 0) ? 0.0 : times[i - 1], b = (i == nt) ? T : times[i];
        w_of(c, 0.5 * (a + b), w);                   // no root inside: any interior point
        uint16_t S = 0;
        for (int j = 0; j < m_; ++j) {
            if (finite_[2 * j] && w[j] > (real)p_->hi[j]) S |= (uint16_t)(1u << (2 * j));
            else if (finite_[2 * j + 1] && w[j] < (real)p_->lo[j]) S |= (uint16_t)(1u << (2 * j + 1));
        }
        if (ns == 0 || S != e.S[ns - 1]) {
            if (ns >= MAXARC_E) { capfail_ = true; return false; }
            if (ns > 0) e.tau[ns - 1] = a;
            e.S[ns++] = S;
        }
    }
    e.narcs = ns;
    return sets_ok(e.S, ns);
}

real Solver::dual(const uint16_t* S, const real* tau, int n, const real* c, const real* mu) const {
    real val = 0.0;
    for (int i = 0; i < n_ && qf_; ++i) val += mu[i] * (g_[i] - 0.5 * mu[i]);
    for (int i = 0; i < n_ && !qf_; ++i) {
        real nu = 0.0;
        for (int j = 0; j < n_; ++j) nu += Linv_[i * MAXN + j] * c[j];
        val += nu * ((real)p_->xf[i] - free_[i]);
    }
    real w[MAXM], v[MAXM], P[MAXM][MAXM], q[MAXM];
    for (int a = 0; a < n; ++a) {
        real t0, t1;
        span(tau, n, a, &t0, &t1);
        pq(S[a], P, q);
        const real hw = 0.5 * (t1 - t0);
        for (int g = 0; g < 4; ++g) {
            const real t = t0 + hw * (GL_X[g] + 1.0);
            w_of(c, t, w);
            v_on(P, q, c, t, v);
            real s = 0.0;
            for (int j = 0; j < m_; ++j) s += 0.5 * v[j] * (rI_ ? v[j] : vR(j, v)) - w[j] * v[j];
            val += GL_W[g] * hw * s;
        }
    }
    return val;
}

bool Solver::evaluate(const real* c, const real* mu, Eval& e) {
    if (!walk(c, e)) return false;
    real h[MAXN];
    assemble(e.S, e.tau, e.narcs, G_, h);
    real sc = 1.0;
    for (int i = 0; i < n_; ++i) {
        real gc = 0.0, s = 0.0;
        for (int j = 0; j < n_; ++j) { gc += G_[i * MAXN + j] * c[j]; s += fabs(G_[i * MAXN + j] * c[j]); }
        e.F[i] = gc + h[i] - (real)p_->xf[i];
        sc = fmax(sc, fabs((real)p_->xf[i]));
        sc = fmax(sc, fabs(free_[i]));
        sc = fmax(sc, s + fabs(h[i] - free_[i]));
    }
    if (qf_) {                                       // transversality: F = Cf (s(T) - xf) + mu
        real r[MAXN], sq = 0.0;
        for (int i = 0; i < n_; ++i) r[i] = e.F[i];
        for (int i = 0; i < n_; ++i) {
            real f = mu[i], a = fabs(mu[i]);
            for (int j = i; j < n_; ++j) { f += Cf_[i * MAXN + j] * r[j]; a += fabs(Cf_[i * MAXN + j]) * sc; }
            e.F[i] = f;
            sq = fmax(sq, a);
        }
        sc = fmax(1.0, sq);
    }
    e.scale = sc;
    e.D = dual(e.S, e.tau, e.narcs, c, mu);
    return true;
}

real Solver::verify(const uint16_t* S, const real* tau, int n, const real* c, real* amp) const {
    real worst = 0.0, a_max = 0.0, phi[2 * MAXM][4], b[4], d[3], cr[2];
    bool has[2 * MAXM];
    for (int a = 0; a < n; ++a) {
        real t0, t1;
        span(tau, n, a, &t0, &t1);
        const real L = t1 - t0;
        row_polys(S[a], c, phi, has);
        for (int key = 0; key < 2 * m_; ++key) {
            if (!has[key]) continue;
            a_max = fmax(a_max, pscale(phi[key], 4, (real)p_->T));
            shift(phi[key], 4, t0, b);
            worst = fmax(worst, fmax(b[0], pval(b, 4, L)));
            for (int i = 1; i < 4; ++i) d[i - 1] = i * b[i];
            const int nc = roots_le2(d, 3, cr);
            for (int i = 0; i < nc; ++i)
                if (cr[i] > 0.0 && cr[i] < L) worst = fmax(worst, pval(b, 4, cr[i]));
        }
    }
    if (amp) *amp = a_max;
    return worst;
}

// warm_its < 0: cold start (unconstrained solution; mu = 0 with Qf), else from R.c / R.mu.
bool Solver::solve_newton(Result& R, int warm_its) {
    R.iterations = R.newton_iterations = R.outer_iterations = 0;
    R.residual = HUGE_VAL;
    R.pmp_violation = HUGE_VAL;
    R.box_fast_path = identity_;
    R.narcs = 0;
    R.failure = "unconstrained start failed";
    real* const c = R.c;                          // the iterate lives in R
    real* const mu = R.mu;
    const int max_its = warm_its < 0 ? MAX_NEWTON : warm_its;
    if (!(warm_its >= 0 && qf_))
        for (int i = 0; i < n_; ++i) mu[i] = 0.0;
    if (qf_) {                                      // c = Lm Cf^T mu
        real* const nu = W_;                      // workspace (free here)
        for (int i = 0; i < n_; ++i) {
            real s = 0.0;
            for (int j = 0; j <= i; ++j) s += Cf_[j * MAXN + i] * mu[j];
            nu[i] = s;
        }
        for (int i = 0; i < n_; ++i) {
            real s = 0.0;
            for (int j = 0; j < n_; ++j) s += Lm_[i * MAXN + j] * nu[j];
            c[i] = s;
        }
    } else if (warm_its >= 0) {
        // warm: c = R.c
    } else {
        const uint16_t S0 = 0;                      // unconstrained start
        if (!solve_c(&S0, nullptr, 1, c)) return false;
    }
    real c0 = 1.0;
    for (int i = 0; i < n_; ++i) c0 = fmax(c0, 1.0 + fabs(c[i]));
    const char* why = "event walk failed";
    int cur = 0;
    bool ok = evaluate(c, mu, ew_.ev[cur]);
    bool conv = false;
    // float: stop at the round-off level of F's terms (scale); double keeps 1e-12
    const real rtol = sizeof(real) == 4 ? EPS : 1e-12;
    int its = 0;
    while (ok) {
        Eval& E = ew_.ev[cur];
        real Fn = 0.0, F2 = 0.0;
        for (int i = 0; i < n_; ++i) { Fn = fmax(Fn, fabs(E.F[i])); F2 += E.F[i] * E.F[i]; }
        if (Fn <= rtol * E.scale) { conv = true; break; }
        if (its >= max_its) { why = "iteration cap"; break; }
        if (!qf_) {   // dual unbounded (infeasible problem): the iterates c diverge
            real cm = 0.0;
            for (int i = 0; i < n_; ++i) cm = fmax(cm, fabs(c[i]));
            if (cm > 1e6 * c0) { why = "iterates diverge (infeasible?)"; break; }
        }
        ++its;
        // (J Lm) d_nu = -F, J Lm = int K P K^T dt symmetric PSD; d_c = Lm d_nu
        // workspace: E's ds(T)/dc in G_ -> H in W_; X and A reuse G_ (evaluate refills it)
        real* const H = W_;
        real dnu[MAXN];
        for (int i = 0; i < n_; ++i)
            for (int j = 0; j < n_; ++j) {
                real s = 0.0;
                for (int k = 0; k < n_; ++k) s += G_[i * MAXN + k] * Lm_[k * MAXN + j];
                H[i * MAXN + j] = s;
            }
        for (int i = 0; i < n_; ++i)
            for (int j = i + 1; j < n_; ++j) {
                const real s = 0.5 * (H[i * MAXN + j] + H[j * MAXN + i]);
                H[i * MAXN + j] = H[j * MAXN + i] = s;
            }
        if (qf_) {   // -Hess D(mu) = Cf H Cf^T + I (uniformly positive definite)
            real* const X = G_;
            for (int i = 0; i < n_; ++i)
                for (int j = 0; j < n_; ++j) {
                    real s = 0.0;
                    for (int k = i; k < n_; ++k) s += Cf_[i * MAXN + k] * H[k * MAXN + j];
                    X[i * MAXN + j] = s;
                }
            for (int i = 0; i < n_; ++i)
                for (int j = i; j < n_; ++j) {
                    real s = (i == j) ? 1.0 : 0.0;
                    for (int k = j; k < n_; ++k) s += X[i * MAXN + k] * Cf_[j * MAXN + k];
                    H[i * MAXN + j] = H[j * MAXN + i] = s;
                }
        }
        real hmax = 0.0;
        for (int i = 0; i < n_ * MAXN; ++i) hmax = fmax(hmax, fabs(H[i]));
        real rho = 0.0;
        bool solved = false;
        for (int attempt = 0; attempt < 20 && !solved; ++attempt) {
            real* const A = G_;
            for (int i = 0; i < MAXN * MAXN; ++i) A[i] = H[i];
            for (int i = 0; i < n_; ++i) { A[i * MAXN + i] += rho; dnu[i] = -E.F[i]; }
            solved = lu_solve(A, n_, MAXN, dnu, 1, 1);
            rho = (rho == 0.0) ? 1e-12 * TS * hmax : 10.0 * rho;
        }
        if (!solved) { why = "singular Newton matrix"; break; }
        // H is dead: the step and the trial point live in W_
        real* const dmu = W_, * const d = W_ + MAXN, * const cn = W_ + 2 * MAXN, * const mun = W_ + 3 * MAXN;
        real slope = 0.0;
        for (int i = 0; i < n_; ++i) slope -= E.F[i] * dnu[i];
        if (qf_) {                                  // solved for d_mu; d_nu = Cf^T d_mu
            for (int i = 0; i < n_; ++i) dmu[i] = dnu[i];
            for (int i = 0; i < n_; ++i) {
                real s = 0.0;
                for (int j = 0; j <= i; ++j) s += Cf_[j * MAXN + i] * dmu[j];
                dnu[i] = s;
            }
        }
        for (int i = 0; i < n_; ++i) {
            real s = 0.0;
            for (int j = 0; j < n_; ++j) s += Lm_[i * MAXN + j] * dnu[j];
            d[i] = s;
        }
        real alpha = 1.0;
        bool accepted = false;
        const int nxt = 1 - cur;
        while (alpha >= 1e-10) {
            for (int i = 0; i < n_; ++i) cn[i] = c[i] + alpha * d[i];
            for (int i = 0; i < n_ && qf_; ++i) mun[i] = mu[i] + alpha * dmu[i];
            if (evaluate(cn, mun, ew_.ev[nxt])) {
                const Eval& N = ew_.ev[nxt];
                if (N.D >= E.D + 1e-4 * alpha * slope) { accepted = true; break; }
                real nF2 = 0.0;
                for (int i = 0; i < n_; ++i) nF2 += N.F[i] * N.F[i];
                if (alpha * slope <= 1e3 * EPS * (1.0 + fabs(E.D)) && nF2 < F2) { accepted = true; break; }
            }
            alpha *= 0.5;
            ++R.backtracks;
        }
        if (!accepted) { why = "line search failed"; break; }
        for (int i = 0; i < n_; ++i) { c[i] = cn[i]; if (qf_) mu[i] = mun[i]; }
        cur = nxt;
    }
    R.iterations = R.newton_iterations = its;
    if (!ok) { R.failure = capfail_ ? CAPFAIL : why; return false; }
    const Eval& E = ew_.ev[cur];
    R.narcs = E.narcs;
    for (int i = 0; i < E.narcs; ++i) R.S[i] = E.S[i];
    for (int i = 0; i + 1 < E.narcs; ++i) R.tau[i] = E.tau[i];
    real Fn = 0.0;
    for (int i = 0; i < n_; ++i) Fn = fmax(Fn, fabs(E.F[i]));
    R.residual = Fn;
    R.degenerate_events = E.ndeg;
    real amp;
    R.pmp_violation = verify(R.S, R.tau, R.narcs, R.c, &amp);
    // float: the walk's own event tolerance (round-off of the monomial row polynomials)
    const real vtol = sizeof(real) == 4 ? 1e-12 * TS * (amp + R_) : 1e-9 * R_;
    if (conv && R.pmp_violation > vtol) { conv = false; why = "PMP verification failed"; }
    R.cost = cost(R.S, R.tau, R.narcs, R.c);
    R.converged = conv;
    R.failure = conv ? nullptr : capfail_ ? CAPFAIL : why;
    return conv;
}


// =====================================================================================
// General LQ cost: GSolver (NOTES.md sections 9 and 10). R-scaled inputs vt = Lc^T v,
// Hamiltonian state z = (s, lambda). Arc solution in the set's basis W (modal stable/unstable
// blocks, centre block Tc): z(t) = W y(t),
//   y_s = e^{Ls (t - t0)} a + yp_s,  [y_c; 1] = e^{[Tc yp_c; 0 0](t - t0)} [c; 1],
//   y_u = e^{Lu (t - t1)} b + yp_u.
// The linear BVP for fixed junction times is a forward sweep of a 12-row relation
// C z(tau_k) = d_k plus a backward recovery of the arc coefficients (a, c, b); the relations
// are stored at NCK checkpoints and recomputed in between.
// =====================================================================================
namespace {

const double GTOL = 1e-12;           // junction residual tolerance (relative to R)
const double STOL = 1e-9;            // walked vs Newton junction times (relative to T)
const double VTOL = 1e-9;            // PMP verification tolerance (relative to R)

// exp and log without libm's lookup tables (a Teensy keeps those in RAM)
double xexp(double x) {
    if (!(x == x)) return x;                           // NaN
    if (x < -745.0) return 0.0;
    if (x > 709.0) return HUGE_VAL;
    const double k = floor(x * 1.4426950408889634 + 0.5);
    const double r = (x - k * 6.93147180369123816490e-01) - k * 1.90821492927058770002e-10;
    static const double inv[14] = {0, 1.0, 1.0 / 2, 1.0 / 3, 1.0 / 4, 1.0 / 5, 1.0 / 6, 1.0 / 7,
                                   1.0 / 8, 1.0 / 9, 1.0 / 10, 1.0 / 11, 1.0 / 12, 1.0 / 13};
    double p = 1.0;                                    // (no divisions: slow on a Cortex-M7)
    for (int i = 13; i >= 1; --i) p = 1.0 + p * r * inv[i];
    return ldexp(p, (int)k);
}

double xlog(double x) {
    if (!(x > 0.0) || x == HUGE_VAL) return (x == 0.0) ? -HUGE_VAL : (x > 0.0) ? x : NAN;
    int e;
    double m = frexp(x, &e);                       // x = m 2^e, m in [0.5, 1)
    if (m < 0.70710678118654752) { m *= 2.0; --e; }
    const double s = (m - 1.0) / (m + 1.0), s2 = s * s;
    double p = 0.0;
    for (int i = 23; i >= 1; i -= 2) p = p * s2 + 1.0 / i;
    return 2.0 * s * p + e * 0.69314718055994530942;
}

// exponential of the modal block at column i over L: 1x1 or 2x2 (row-major), returns the size
inline int mblk(const double* re, const double* im, int i, double L, double E[4]) {
    const double e = xexp(re[i] * L);
    if (im[i] == 0.0) { E[0] = e; return 1; }
    const double c = cos(im[i] * L) * e, s = sin(im[i] * L) * e;
    E[0] = c; E[1] = s; E[2] = -s; E[3] = c;
    return 2;
}

// g[j] = t^j phi_j(lam t), j = 0..K, complex lam = (lr, li): phi_0 = e^z, phi_j(z) = sum_i z^i/(i+j)!,
// the divided differences of e^{z} at (0, .., 0, z) that couple a slow centre mode to the nilpotent
// block (g_j' = g_{j-1}, g_0' = lam g_0). |z| <= 4: phi_K by 40 terms of its series, then the stable
// downward recurrence phi_{j-1} = z phi_j + 1/(j-1)!; |z| > 4: upward, phi_j = (phi_{j-1} - 1/(j-1)!)/z
// from e^z. Relative error <= 1e-13 for K <= 10 (checked against 40-digit values): a fixed
// evaluation of entire functions, like xexp.
void gfun(double lr, double li, double t, int K, double (*g)[2]) {
    const double zr = lr * t, zi = li * t, z2 = zr * zr + zi * zi;
    if (z2 > 16.0) {
        const double e = xexp(zr);
        g[0][0] = e * cos(zi); g[0][1] = e * sin(zi);
        double fk = 1.0;                               // 1/(j-1)!
        for (int j = 1; j <= K; ++j) {
            const double ar = g[j - 1][0] - fk, ai = g[j - 1][1];
            g[j][0] = (ar * zr + ai * zi) / z2;
            g[j][1] = (ai * zr - ar * zi) / z2;
            fk /= j;
        }
    } else {
        double fk = 1.0;                               // 1/K!
        for (int j = 2; j <= K; ++j) fk /= j;
        static const double inv[2 * MAXK + 43] = {      // 1/n (no divisions: slow on a Cortex-M7)
            0, 1.0, 1.0 / 2, 1.0 / 3, 1.0 / 4, 1.0 / 5, 1.0 / 6, 1.0 / 7, 1.0 / 8, 1.0 / 9, 1.0 / 10,
            1.0 / 11, 1.0 / 12, 1.0 / 13, 1.0 / 14, 1.0 / 15, 1.0 / 16, 1.0 / 17, 1.0 / 18, 1.0 / 19,
            1.0 / 20, 1.0 / 21, 1.0 / 22, 1.0 / 23, 1.0 / 24, 1.0 / 25, 1.0 / 26, 1.0 / 27, 1.0 / 28,
            1.0 / 29, 1.0 / 30, 1.0 / 31, 1.0 / 32, 1.0 / 33, 1.0 / 34, 1.0 / 35, 1.0 / 36, 1.0 / 37,
            1.0 / 38, 1.0 / 39, 1.0 / 40, 1.0 / 41, 1.0 / 42, 1.0 / 43, 1.0 / 44, 1.0 / 45, 1.0 / 46,
            1.0 / 47, 1.0 / 48, 1.0 / 49, 1.0 / 50};
        double sr = 0.0, si = 0.0, tr = fk, ti = 0.0;
        for (int i = 1; i <= 40; ++i) {                // (i + K <= 50: K <= ni + 2 <= 2 MAXK + 2)
            sr += tr; si += ti;
            const double nr = (tr * zr - ti * zi) * inv[i + K], ni = (tr * zi + ti * zr) * inv[i + K];
            tr = nr; ti = ni;
        }
        g[K][0] = sr; g[K][1] = si;
        for (int j = K; j >= 1; --j) {
            fk *= j;                                   // 1/(j-1)!
            g[j - 1][0] = zr * g[j][0] - zi * g[j][1] + fk;
            g[j - 1][1] = zr * g[j][1] + zi * g[j][0];
        }
    }
    double tj = 1.0;
    for (int j = 0; j <= K; ++j) { g[j][0] *= tj; g[j][1] *= tj; tj *= t; }
}

// action of a block function G = [[gr, gi], [-gi, gr]] (q = (gr, gi); 1x1 block: gr) of width w:
// o += s G v (column), o += r G (row); dot: A = <c, v> = P - iQ (P = c0 v0 + c1 v1, Q = c0 v1 - c1 v0)
__attribute__((noinline)) void col(const double* q, const double* v, int w, double* o, double s) {
    if (w == 1) { o[0] += s * q[0] * v[0]; return; }
    o[0] += s * (q[0] * v[0] + q[1] * v[1]);
    o[1] += s * (q[0] * v[1] - q[1] * v[0]);
}
__attribute__((noinline)) void rowa(const double* q, const double* r, int w, double* o) {
    if (w == 1) { o[0] += r[0] * q[0]; return; }
    o[0] += r[0] * q[0] - r[1] * q[1];
    o[1] += r[0] * q[1] + r[1] * q[0];
}
__attribute__((noinline)) void dot(const double* c, const double* v, int w, double* A0, double* A1) {
    *A0 += c[0] * v[0] + ((w == 2) ? c[1] * v[1] : 0.0);
    if (w == 2) *A1 -= c[0] * v[1] - c[1] * v[0];
}

// polynomial sum_k P[k] t^k of np terms; root of it on [lo, hi] where it is monotone (safeguarded
// Newton; D: its derivative's coefficients)
__attribute__((noinline)) double horner(const double* P, int np, double t) {
    double s = 0.0;
    for (int k = np - 1; k >= 0; --k) s = s * t + P[k];
    return s;
}
__attribute__((noinline)) double pzero(const double* P, const double* D, int np, double lo, double hi) {
    const double flo = horner(P, np, lo);
    double t = 0.5 * (lo + hi);
    for (int it = 0; it < 200; ++it) {
        const double ft = horner(P, np, t), dt = horner(D, np - 1, t);
        if ((ft > 0) == (flo > 0)) lo = t; else hi = t;
        double tn = (dt != 0.0) ? t - ft / dt : 0.5 * (lo + hi);
        if (!(tn > lo && tn < hi)) tn = 0.5 * (lo + hi);
        if (tn == t || !(hi > lo)) break;
        t = tn;
    }
    return t;
}

// Householder reflection H with H x = alpha e_p for the column x = R[.][col] over the rows
// R[0..nr-1] (eliminates that unknown; row p is its pivot). Column col is then replaced by eta,
// the coefficients of the unknown that takes its place, and H is applied to columns [0, c1)
// and to the optional vector extra.
void reflect(double* const* R, int nr, int p, int col, int c1, const double* eta, double* extra) {
    double v[MAXN + 2], sg = 0.0;
    for (int i = 0; i < nr; ++i) {
        v[i] = R[i][col];
        sg += v[i] * v[i];
        R[i][col] = eta[i];
    }
    sg = sqrt(sg);
    double vv = 0.0;
    if (sg > 0.0) {
        v[p] -= (v[p] > 0) ? -sg : sg;
        for (int i = 0; i < nr; ++i) vv += v[i] * v[i];
    }
    if (vv == 0.0) return;
    for (int c = 0; c <= c1; ++c) {
        double* col_c = nullptr;
        double s = 0.0;
        if (c == c1) {
            if (!extra) break;
            col_c = extra;
            for (int i = 0; i < nr; ++i) s += v[i] * extra[i];
        } else {
            for (int i = 0; i < nr; ++i) s += v[i] * R[i][c];
        }
        if (s == 0.0) continue;
        s *= 2.0 / vv;
        for (int i = 0; i < nr; ++i) {
            if (col_c) col_c[i] -= s * v[i];
            else R[i][c] -= s * v[i];
        }
    }
}

// t = row M (M row-major n x n with row stride ld). M is streamed row by row: it may live in
// flash, where sequential reads are much cheaper than strided ones (Teensy 4.1).
void rowmul(const double* row, const double* M, int n, int ld, double* t) {
    for (int j = 0; j < n; ++j) t[j] = 0.0;
    for (int i = 0; i < n; ++i) {
        const double ri = row[i];
        if (ri == 0.0) continue;
        const double* Mi = M + i * ld;
        for (int j = 0; j < n; ++j) t[j] += ri * Mi[j];
    }
}

// t = M v (M row-major n x n, row stride ld), two rows at a time
void matvec(const double* M, int n, int ld, const double* v, double* t) {
    int i = 0;
    for (; i + 2 <= n; i += 2) {
        const double *m0 = M + i * ld, *m1 = m0 + ld;
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        int j = 0;
        for (; j + 2 <= n; j += 2) {
            s0 += m0[j] * v[j]; s1 += m0[j + 1] * v[j + 1];
            s2 += m1[j] * v[j]; s3 += m1[j + 1] * v[j + 1];
        }
        if (j < n) { s0 += m0[j] * v[j]; s2 += m1[j] * v[j]; }
        t[i] = s0 + s1;
        t[i + 1] = s2 + s3;
    }
    if (i < n) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) s += M[i * ld + j] * v[j];
        t[i] = s;
    }
}

}  // namespace

bool GSolver::setup(const Problem& p, BasisFn fn, void* ctx) {
    if (p.m < 1 || p.m > MAXM || !fn) return false;
    p_ = &p;
    bfn_ = fn;
    bctx_ = ctx;
    m_ = p.m;
    n_ = 0;
    for (int j = 0; j < m_; ++j) {
        if (p.orders[j] < 1 || p.orders[j] > MAXK) return false;
        off_[j] = n_;
        n_ += p.orders[j];
        end_[j] = n_ - 1;
    }
    if (n_ > MAXN || !(p.T > 0.0)) return false;
    set_x0(p.x0);
    // Cholesky R = Lc Lc^T (symmetrised)
    for (int i = 0; i < m_; ++i)
        for (int j = 0; j < m_; ++j) Lc_[i][j] = 0.0;
    for (int j = 0; j < m_; ++j) {
        double d = p.R[j][j];
        for (int k = 0; k < j; ++k) d -= Lc_[j][k] * Lc_[j][k];
        if (!(d > 0.0)) return false;
        Lc_[j][j] = sqrt(d);
        for (int i = j + 1; i < m_; ++i) {
            double s = 0.5 * (p.R[i][j] + p.R[j][i]);
            for (int k = 0; k < j; ++k) s -= Lc_[i][k] * Lc_[j][k];
            Lc_[i][j] = s / Lc_[j][j];
        }
    }
    for (int c = 0; c < m_; ++c)                       // LiT = Lc^{-T}: Lc^T X = I
        for (int i = m_ - 1; i >= 0; --i) {
            double s = (i == c) ? 1.0 : 0.0;
            for (int k = i + 1; k < m_; ++k) s -= Lc_[k][i] * LiT_[k][c];
            LiT_[i][c] = s / Lc_[i][i];
        }
    double mx = 0.0;
    for (int r = 0; r < m_; ++r) {
        for (int k = 0; k < m_; ++k) {
            double s = 0.0;
            for (int j = 0; j < m_; ++j) s += p.M[r][j] * LiT_[j][k];
            rows_[2 * r][k] = s;
            rows_[2 * r + 1][k] = -s;
        }
        rhs_[2 * r] = p.hi[r];
        rhs_[2 * r + 1] = -p.lo[r];
        finite_[2 * r] = p.has_bounds && isfinite(p.hi[r]);
        finite_[2 * r + 1] = p.has_bounds && isfinite(p.lo[r]);
    }
    for (int k = 0; k < 2 * m_; ++k) {
        double s = 0.0;
        for (int j = 0; j < m_; ++j) s += rows_[k][j] * rows_[k][j];
        rownorm2_[k] = s;
        if (finite_[k]) mx = fmax(mx, fabs(rhs_[k]));
        rhs0_[k] = rhs_[k];
    }
    R_ = 1.0 + mx;
    nsets_ = 0;
    for_sets([&](uint16_t) { ++nsets_; return false; });
    return true;
}

bool GSolver::set_x0(const double* x0) {
    x0ok_ = true;
    for (int i = 0; i < n_; ++i) { x0_[i] = x0[i]; x0ok_ = x0ok_ && isfinite(x0[i]); }
    return x0ok_;
}

// Calls f(S) for the candidate active sets, by increasing size, until f returns true.
template <class F>
bool GSolver::for_sets(F f) const {
    int total = 1;
    for (int r = 0; r < m_; ++r) total *= 3;
    for (int sz = 0; sz <= m_; ++sz)
        for (int code = 0; code < total; ++code) {
            uint16_t S = 0;
            int cc = code, cnt = 0;
            bool ok = true;
            for (int r = 0; r < m_; ++r) {
                const int d = cc % 3;
                cc /= 3;
                if (d == 1) { S |= (uint16_t)(1u << (2 * r)); ++cnt; if (!finite_[2 * r]) ok = false; }
                if (d == 2) { S |= (uint16_t)(1u << (2 * r + 1)); ++cnt; if (!finite_[2 * r + 1]) ok = false; }
            }
            if (ok && cnt == sz && f(S)) return true;
        }
    return false;
}

uint16_t GSolver::set(int i) const {
    uint16_t out = 0;
    for_sets([&](uint16_t S) { if (i-- == 0) { out = S; return true; } return false; });
    return out;
}

bool GSolver::valid_set(uint16_t S) const {
    int cnt = 0;
    for (int r = 0; r < m_; ++r) {
        const bool up = S & (1u << (2 * r)), lo = S & (1u << (2 * r + 1));
        if (up && lo) return false;
        if (up && !finite_[2 * r]) return false;
        if (lo && !finite_[2 * r + 1]) return false;
        cnt += up || lo;
    }
    return cnt <= m_;
}

bool GSolver::affine(uint16_t S, double P[MAXM][MAXM], double q[MAXM]) const {
    int idx[MAXM], na = 0;
    for (int k = 0; k < 2 * m_; ++k)
        if (S & (1u << k)) { if (na >= MAXM) return false; idx[na++] = k; }
    for (int i = 0; i < m_; ++i) {
        q[i] = 0.0;
        for (int j = 0; j < m_; ++j) P[i][j] = (i == j) ? 1.0 : 0.0;
    }
    if (na == 0) return true;
    double G[MAXM * MAXM], X[MAXM * (MAXM + 1)];
    const int ldx = MAXM + 1;
    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < na; ++b) {
            double s = 0.0;
            for (int j = 0; j < m_; ++j) s += rows_[idx[a]][j] * rows_[idx[b]][j];
            G[a * MAXM + b] = s;
        }
        for (int j = 0; j < m_; ++j) X[a * ldx + j] = rows_[idx[a]][j];
        X[a * ldx + m_] = rhs_[idx[a]];
    }
    if (!lu_solve(G, na, MAXM, X, m_ + 1, ldx)) return false;
    for (int i = 0; i < m_; ++i)                       // (every channel fixed: P = 0 exactly)
        for (int a = 0; a < na; ++a) {
            const double ai = rows_[idx[a]][i];
            for (int j = 0; j < m_; ++j) P[i][j] = (na == m_) ? 0.0 : P[i][j] - ai * X[a * ldx + j];
            q[i] += ai * X[a * ldx + m_];
        }
    return true;
}

// wt = -Bt^T lambda (Bt = B Lc^-T has the rows LiT_[j] at the chain ends)
void GSolver::wt_of(const double* lam, double* wt) const {
    for (int k = 0; k < m_; ++k) {
        double s = 0.0;
        for (int j = 0; j < m_; ++j) s -= LiT_[j][k] * lam[end_[j]];
        wt[k] = s;
    }
}

// out = H_S z for the augmented z = (s, lambda, 1)
void GSolver::hmul(const double P[MAXM][MAXM], const double* q, const double* z, double* out) const {
    const int n = n_;
    const double* s = z;
    const double* lam = z + n;
    double wt[MAXM], vt[MAXM];
    wt_of(lam, wt);
    for (int i = 0; i < m_; ++i) {
        double a = q[i] * z[2 * n];
        for (int j = 0; j < m_; ++j) a += P[i][j] * wt[j];
        vt[i] = a;
    }
    for (int i = 0; i < n; ++i) {
        out[i] = 0.0;
        double b = 0.0;
        for (int j = 0; j < n; ++j) b -= 0.5 * (p_->Q[i][j] + p_->Q[j][i]) * s[j];
        out[n + i] = b;
    }
    for (int j = 0; j < m_; ++j) {
        double a = 0.0;
        for (int k = 0; k < m_; ++k) a += LiT_[j][k] * vt[k];
        out[end_[j]] = a;
    }
    for (int j = 0; j < m_; ++j)
        for (int r = 0; r + 1 < p_->orders[j]; ++r) {
            const int i = off_[j] + r;
            out[i] += s[i + 1];
            out[n + i + 1] -= lam[i];
        }
    out[2 * n] = 0.0;
}

bool GSolver::hmat(uint16_t S, double* H) const {
    double P[MAXM][MAXM], q[MAXM], e[N2 + 1], o[N2 + 1];
    if (!affine(S, P, q)) return false;
    const int n2 = 2 * n_;
    for (int i = 0; i < m_; ++i) q[i] = 0.0;       // homogeneous (q may be inf for a basis key)
    for (int c = 0; c < n2; ++c) {
        for (int i = 0; i <= n2; ++i) e[i] = (i == c) ? 1.0 : 0.0;
        hmul(P, q, e, o);
        for (int i = 0; i < n2; ++i) H[i * n2 + c] = o[i];
    }
    return true;
}

// arc k of the structure (seq, tau): basis, projection, particular part
bool GSolver::arc(int k, const uint16_t* seq, int N, const double* tau, Arc& a) const {
    a.t0 = (k == 0) ? 0.0 : tau[k - 1];
    a.L = ((k == N - 1) ? p_->T : tau[k]) - a.t0;
    a.S = seq[k];
    a.b = bfn_(bctx_, basis_key(seq[k]));
    a.x = nullptr;
    if (!a.b || a.b->nc - a.b->nz > NSD) { bkey_ = basis_key(seq[k]); return false; }   // (NSD: a table
                                                                                        // built larger)
    if (!affine(seq[k], a.P, a.q)) return false;
    const SetBasis& b = *a.b;
    const int n2 = 2 * n_, s1 = b.ns + b.nc;
    double f[MAXM];
    for (int j = 0; j < m_; ++j) {
        double s = 0.0;
        for (int k2 = 0; k2 < m_; ++k2) s += LiT_[j][k2] * a.q[k2];
        f[j] = s;                                      // (Bt q) at row end_[j]
    }
    double phi[N2];
    for (int i = 0; i < n2; ++i) {
        double s = 0.0;
        for (int j = 0; j < m_; ++j) s += b.Wi[i * n2 + end_[j]] * f[j];
        phi[i] = s;
    }
    for (int i = 0; i < n2;) {
        if (i >= b.ns && i < s1) { a.yp[i] = phi[i]; ++i; continue; }   // centre: forcing
        const double al = b.re[i], be = b.im[i];
        if (be == 0.0) { a.yp[i] = -phi[i] / al; ++i; continue; }
        const double det = al * al + be * be;
        a.yp[i] = -(al * phi[i] - be * phi[i + 1]) / det;
        a.yp[i + 1] = -(be * phi[i] + al * phi[i + 1]) / det;
        i += 2;
    }
    return true;
}

// y <- e^{M L} y (column) or y <- y e^{M L} (row), M = [[Tc, aff ? yp_c : 0], [0, 0]], y has nc + 1
// entries; Tc = [[N, X], [0, Ls]] (SetBasis). Exact closed form (N^ni = 0):
//   y_s(L) = G0 y_s + G1 g_s,  y_N(L) = sum_{k<ni} N^k [y_N L^k/k! + g_N L^(k+1)/(k+1)! + X (G(k+1) y_s + G(k+2) g_s)]
// with the slow blocks' G(j) = L^j phi_j(Ls L) (gfun), the sum by Horner in N; the row form is its
// transpose. No truncation, no substeps.
void GSolver::cexp(const Arc& a, double L, double* y, bool row, bool aff) const {
    const SetBasis& b = *a.b;
    const int nc = b.nc, nz = b.nz, ni = b.ni;
    if (nc == 0 || L == 0.0) return;
    const double* T = b.Tc;                            // row stride nc
    const double* g = a.yp + b.ns;
    const double* bre = b.re + b.ns;
    const double* bim = b.im + b.ns;
    const double af = aff ? (row ? 1.0 : y[nc]) : 0.0;
    double G[NSD][NPOLY + 2][2], fk[NPOLY + 2];
    int nb = 0, cb[NSD + 1];
    for (int c = nz; c < nc; c += (bim[c] != 0.0) ? 2 : 1) {
        gfun(bre[c], bim[c], L, ni + 1, G[nb]);
        cb[nb++] = c;
    }
    cb[nb] = nc;
    fk[0] = 1.0;
    for (int k = 1; k <= ni + 1; ++k) fk[k] = fk[k - 1] * L / k;
    if (!row) {
        double acc[N2], v[N2], xs[N2];
        for (int k = ni - 1; k >= 0; --k) {
            for (int l = nz; l < nc; ++l) xs[l] = 0.0;
            for (int q = 0; q < nb; ++q) {
                const int c = cb[q], w = cb[q + 1] - c;
                col(G[q][k + 1], y + c, w, xs + c, 1.0);
                col(G[q][k + 2], g + c, w, xs + c, af);
            }
            for (int i = 0; i < nz; ++i) {
                double s = y[i] * fk[k] + af * g[i] * fk[k + 1];
                for (int l = nz; l < nc; ++l) s += T[i * nc + l] * xs[l];
                if (k < ni - 1)
                    for (int l = i + 1; l < nz; ++l) s += T[i * nc + l] * acc[l];
                v[i] = s;
            }
            for (int i = 0; i < nz; ++i) acc[i] = v[i];
        }
        for (int l = nz; l < nc; ++l) xs[l] = 0.0;
        for (int q = 0; q < nb; ++q) {
            const int c = cb[q], w = cb[q + 1] - c;
            col(G[q][0], y + c, w, xs + c, 1.0);
            col(G[q][1], g + c, w, xs + c, af);
        }
        for (int i = 0; i < nz; ++i) y[i] = acc[i];
        for (int l = nz; l < nc; ++l) y[l] = xs[l];
        return;
    }
    // row: w_k = r_N N^k; out_N = sum w_k L^k/k!, out_s = r_s G0 + sum (w_k X) G(k+1),
    // affine slot += sum (w_k g_N) L^(k+1)/(k+1)! + r_s G1 g_s + sum (w_k X) G(k+2) g_s
    double w[N2], wn[N2], out[N2 + 1], c[N2], t2[2];
    for (int i = 0; i <= nc; ++i) out[i] = 0.0;
    out[nc] = y[nc];
    for (int q = 0; q < nb; ++q) {
        const int c0 = cb[q], ww = cb[q + 1] - c0;
        rowa(G[q][0], y + c0, ww, out + c0);
        t2[0] = t2[1] = 0.0;
        rowa(G[q][1], y + c0, ww, t2);
        for (int l = 0; l < ww; ++l) out[nc] += af * t2[l] * g[c0 + l];
    }
    for (int i = 0; i < nz; ++i) w[i] = y[i];
    for (int k = 0; k < ni; ++k) {
        double wg = 0.0;
        for (int i = 0; i < nz; ++i) { out[i] += w[i] * fk[k]; wg += w[i] * g[i]; }
        out[nc] += af * wg * fk[k + 1];
        for (int l = nz; l < nc; ++l) {
            double s = 0.0;
            for (int i = 0; i < nz; ++i) s += w[i] * T[i * nc + l];
            c[l] = s;
        }
        for (int q = 0; q < nb; ++q) {
            const int c0 = cb[q], ww = cb[q + 1] - c0;
            rowa(G[q][k + 1], c + c0, ww, out + c0);
            t2[0] = t2[1] = 0.0;
            rowa(G[q][k + 2], c + c0, ww, t2);
            for (int l = 0; l < ww; ++l) out[nc] += af * t2[l] * g[c0 + l];
        }
        if (k + 1 < ni) {
            for (int j = 0; j < nz; ++j) {
                double s = 0.0;
                for (int i = 0; i < j; ++i) s += w[i] * T[i * nc + j];
                wn[j] = s;
            }
            for (int i = 0; i < nz; ++i) w[i] = wn[i];
        }
    }
    for (int i = 0; i <= nc; ++i) y[i] = out[i];
}

// z (augmented, z[2n] = 1) at t0 + sig on arc a
void GSolver::zat(const Arc& a, double sig, double* z) const {
    const SetBasis& b = *a.b;
    const int n2 = 2 * n_, ns = b.ns, nc = b.nc, s1 = ns + nc;
    const double* x = a.x;
    const double* cs = b.cs;
    double y[N2 + 1], E[4];
    for (int i = 0; i < ns;) {
        const int sz = mblk(b.re, b.im, i, sig, E);
        if (sz == 1) y[i] = E[0] * x[i] / cs[i] + a.yp[i];
        else {
            const double x0 = x[i] / cs[i], x1 = x[i + 1] / cs[i + 1];
            y[i] = E[0] * x0 + E[1] * x1 + a.yp[i];
            y[i + 1] = E[2] * x0 + E[3] * x1 + a.yp[i + 1];
        }
        i += sz;
    }
    if (nc) {                                          // [y_c; 1] in y[ns .. s1]
        for (int i = ns; i < s1; ++i) y[i] = x[i] / cs[i];
        y[s1] = 1.0;
        cexp(a, sig, y + ns, false, true);
    }
    for (int i = s1; i < n2;) {
        const int sz = mblk(b.re, b.im, i, sig - a.L, E);
        if (sz == 1) y[i] = E[0] * x[i] / cs[i] + a.yp[i];
        else {
            const double x0 = x[i] / cs[i], x1 = x[i + 1] / cs[i + 1];
            y[i] = E[0] * x0 + E[1] * x1 + a.yp[i];
            y[i + 1] = E[2] * x0 + E[3] * x1 + a.yp[i + 1];
        }
        i += sz;
    }
    matvec(b.W, n2, n2, y, z);
    z[n2] = 1.0;
}

void GSolver::zstart(const uint16_t* seq, int N, const double* tau, int k, double* z) const {
    if (!arc(k, seq, N, tau, ac_)) { for (int i = 0; i <= 2 * n_; ++i) z[i] = 0.0; return; }
    ac_.x = R_out->x[k];
    zat(ac_, 0.0, z);
}

// Switching rows of set S (one solve for the projection and the multipliers): inactive rows ->
// gap a_r vt - b_r = G wt + g0; active rows -> -mu_r |a_r|^2. Both must stay <= 0.
int GSolver::gphi(uint16_t S, int keys[2 * MAXM], double G[2 * MAXM][MAXM], double g0[2 * MAXM]) const {
    int idx[MAXM], na = 0;
    for (int key = 0; key < 2 * m_; ++key)
        if (S & (1u << key)) { if (na >= MAXM) return 0; idx[na++] = key; }
    double X[MAXM * (MAXM + 1)], P[MAXM][MAXM], q[MAXM];
    const int ldx = MAXM + 1;
    bool ok = true;
    if (na > 0) {
        double A[MAXM * MAXM];
        for (int a = 0; a < na; ++a) {
            for (int b = 0; b < na; ++b) {
                double s = 0.0;
                for (int j = 0; j < m_; ++j) s += rows_[idx[a]][j] * rows_[idx[b]][j];
                A[a * MAXM + b] = s;
            }
            for (int j = 0; j < m_; ++j) X[a * ldx + j] = rows_[idx[a]][j];
            X[a * ldx + m_] = rhs_[idx[a]];
        }
        ok = lu_solve(A, na, MAXM, X, m_ + 1, ldx);
    }
    for (int i = 0; i < m_; ++i) {                     // P, q as in affine()
        q[i] = 0.0;
        for (int j = 0; j < m_; ++j) P[i][j] = (i == j) ? 1.0 : 0.0;
        if (ok)
            for (int a = 0; a < na; ++a) {
                const double ai = rows_[idx[a]][i];
                for (int j = 0; j < m_; ++j) P[i][j] = (na == m_) ? 0.0 : P[i][j] - ai * X[a * ldx + j];
                q[i] += ai * X[a * ldx + m_];
            }
    }
    int nk = 0;
    for (int key = 0; key < 2 * m_; ++key) {
        if (!finite_[key] || (S & (1u << key))) continue;
        keys[nk] = key;
        double c0 = -rhs_[key];
        for (int j = 0; j < m_; ++j) {
            double a = 0.0;
            for (int i = 0; i < m_; ++i) a += rows_[key][i] * P[i][j];
            G[nk][j] = a;
            c0 += rows_[key][j] * q[j];
        }
        g0[nk++] = c0;
    }
    if (ok)
        for (int a = 0; a < na; ++a) {
            keys[nk] = idx[a];
            for (int j = 0; j < m_; ++j) G[nk][j] = -X[a * ldx + j] * rownorm2_[idx[a]];
            g0[nk++] = X[a * ldx + m_] * rownorm2_[idx[a]];
        }
    return nk;
}

// ---- wt on an arc as exponential-polynomials (NOTES.md section 11) ----
// Channel j: wt_j(sig) = c_j + sum_k p_jk sig^k + sum_i Re[A_ji e^{lam_i (sig - anc_i)}] with the
// modal eigenvalues lam_i (stable: anchor 0, unstable: anchor L; a complex pair stores P, Q of
// A = P - iQ in its two slots) and the centre part e^{M sig} [c; 1], M = [[Tc, yp_c], [0, 0]],
// a polynomial of ni + 1 terms (M^(ni+1) = 0).
bool GSolver::chan(const Arc& a, ArcFn& F) const {
    const SetBasis& b = *a.b;
    const int n = n_, n2 = 2 * n, ns = b.ns, nc = b.nc, s1 = ns + nc;
    double (*Bw)[N2] = F.amp;                          // wt = Bw y (y: modal coordinates); the
    for (int j = 0; j < m_; ++j) {                     // centre columns stay Bw, the rest become
        for (int i = 0; i < n2; ++i) Bw[j][i] = 0.0;   // amplitudes in place
        for (int l = 0; l < m_; ++l) {
            const double f = -LiT_[l][j];
            const double* Wr = b.W + (n + end_[l]) * n2;
            for (int i = 0; i < n2; ++i) Bw[j][i] += f * Wr[i];
        }
        F.c[j] = 0.0;
    }
    F.b = a.b;
    F.L = a.L;
    F.m = m_;
    F.np = F.nsb = 0;
    for (int i = 0; i < n2;) {
        if (i == ns && nc) { i = s1; continue; }
        const double x0 = a.x[i] / b.cs[i];
        if (b.im[i] == 0.0) {
            for (int j = 0; j < m_; ++j) { F.c[j] += Bw[j][i] * a.yp[i]; F.amp[j][i] = Bw[j][i] * x0; }
            ++i;
        } else {
            const double x1 = a.x[i + 1] / b.cs[i + 1];
            for (int j = 0; j < m_; ++j) {
                const double b0 = Bw[j][i], b1 = Bw[j][i + 1];
                F.c[j] += b0 * a.yp[i] + b1 * a.yp[i + 1];
                F.amp[j][i] = b0 * x0 + b1 * x1;
                F.amp[j][i + 1] = b0 * x1 - b1 * x0;
            }
            i += 2;
        }
    }
    if (!nc) return true;
    const double* g = a.yp + ns;
    const int nz = b.nz, ni = b.ni;
    double v[N2 + 1], u[N2 + 1], y0[N2];
    for (int i = 0; i < nc; ++i) y0[i] = v[i] = a.x[ns + i] / b.cs[ns + i];
    v[nz] = 1.0;
    if (ni >= NPOLY) return false;
    F.np = ni + 1;
    for (int k = 0; k <= ni; ++k) {                    // N part: v = M^k [c_N; 1] / k!, M = [[N, g_N], [0, 0]]
        for (int j = 0; j < m_; ++j) {
            double s = 0.0;
            for (int i = 0; i < nz; ++i) s += Bw[j][ns + i] * v[i];
            F.poly[j][k] = s;
        }
        matvec(b.Tc, nz, nc, v, u);
        for (int i = 0; i < nz; ++i) v[i] = (u[i] + g[i] * v[nz]) / (k + 1);
        v[nz] = 0.0;
    }
    // slow blocks: sum_k Re[A_k g_k(sig)] with A_0 = <Bw_s, y_s>, A_1 = <Bw_s, g_s> + <Bw_N X, y_s>,
    // A_(k+1) += <Bw_N N^k X, y_s>, A_(k+2) += <Bw_N N^k X, g_s>; <c, v> of a pair = P - iQ,
    // P = c0 v0 + c1 v1, Q = c0 v1 - c1 v0 (Re[(P - iQ) g] = c G v)
    F.sk = ni + 2;
    F.nz = nz;
    for (int c = nz; c < nc; c += (b.im[ns + c] != 0.0) ? 2 : 1) F.sc[F.nsb++] = c;
    if (!F.nsb) return true;
    for (int j = 0; j < m_; ++j) {
        double r[N2], rn[N2], cx[N2];
        for (int q = 0; q < nc - nz; ++q)
            for (int k = 0; k < F.sk; ++k) F.sl[j][q][k] = 0.0;
        for (int i = 0; i < nz; ++i) r[i] = Bw[j][ns + i];
        for (int k = 0; k < ni; ++k) {
            for (int l = nz; l < nc; ++l) {
                double s = 0.0;
                for (int i = 0; i < nz; ++i) s += r[i] * b.Tc[i * nc + l];
                cx[l] = s;
            }
            for (int q = 0; q < F.nsb; ++q) {
                const int c = F.sc[q], w = ((q + 1 < F.nsb) ? F.sc[q + 1] : nc) - c;
                double* s0 = F.sl[j][c - nz];
                double* s1 = F.sl[j][c - nz + w - 1];  // (= s0 for a real block, unused)
                if (k == 0) {
                    dot(Bw[j] + ns + c, y0 + c, w, s0, s1);
                    dot(Bw[j] + ns + c, g + c, w, s0 + 1, s1 + 1);
                }
                dot(cx + c, y0 + c, w, s0 + k + 1, s1 + k + 1);
                dot(cx + c, g + c, w, s0 + k + 2, s1 + k + 2);
            }
            for (int jj = 0; jj < nz; ++jj) {
                double s = 0.0;
                for (int i = 0; i < jj; ++i) s += r[i] * b.Tc[i * nc + jj];
                rn[jj] = s;
            }
            for (int i = 0; i < nz; ++i) r[i] = rn[i];
        }
    }
    return true;
}

// row function g0 + G wt on the arc of F
void GSolver::rowfn(const ArcFn& F, const double* G, double g0, RowFn& f) const {
    const SetBasis& b = *F.b;
    const int n2 = 2 * n_, ns = b.ns, s1 = ns + b.nc;
    f.F = &F;
    for (int j = 0; j < MAXM; ++j) f.G[j] = (j < m_) ? G[j] : 0.0;
    double c0 = g0, sc = 0.0, Lk = 1.0;
    for (int j = 0; j < m_; ++j) c0 += G[j] * F.c[j];
    for (int i = 0; i < n2; ++i)
        if (i < ns || i >= s1) sc += fabs(f.u(i));
    for (int k = 0; k < F.np; ++k) { sc += fabs(f.p(k)) * Lk; Lk *= F.L; }
    f.c0 = c0;
    f.scale = sc + fabs(c0) + (F.nsb ? sbound(f, 0, F.L, false) : 0.0);
}

// Slow blocks of the row f (F.sl): coefficients A (complex, F.sk terms) of block q, differentiated
// nder times (d/dsig sum A_k g_k = sum A'_k g_k with A'_0 = lam A_0 + A_1, A'_k = A_(k+1))
void GSolver::scoef(const RowFn& f, int q, int nder, double (*A)[2]) const {
    const ArcFn& F = *f.F;
    const double lr = F.b->re[F.b->ns + F.sc[q]], li = F.b->im[F.b->ns + F.sc[q]];
    for (int k = 0; k < F.sk; ++k) { A[k][0] = f.s(q, k, 0); A[k][1] = f.s(q, k, 1); }
    for (int d = 0; d < nder; ++d) {
        const double r0 = lr * A[0][0] - li * A[0][1] + A[1][0], i0 = lr * A[0][1] + li * A[0][0] + A[1][1];
        A[0][0] = r0; A[0][1] = i0;
        for (int k = 1; k < F.sk; ++k) {
            A[k][0] = (k + 1 < F.sk) ? A[k + 1][0] : 0.0;
            A[k][1] = (k + 1 < F.sk) ? A[k + 1][1] : 0.0;
        }
    }
}

// bound of |slow part^(nder)| on [0, t] (t >= 0; |g_k(t)| <= t^k/k! e^{max(0, Re lam t)}), or with
// full = true the Taylor majorant at radius t (e^{|lam| t})
double GSolver::sbound(const RowFn& f, int nder, double t, bool full) const {
    const ArcFn& F = *f.F;
    double B = 0.0, A[NPOLY + 1][2];
    for (int q = 0; q < F.nsb; ++q) {
        const double lr = F.b->re[F.b->ns + F.sc[q]], li = F.b->im[F.b->ns + F.sc[q]];
        scoef(f, q, nder, A);
        double s = 0.0, tk = 1.0;
        for (int k = 0; k < F.sk; ++k) { s += sqrt(A[k][0] * A[k][0] + A[k][1] * A[k][1]) * tk; tk *= t / (k + 1); }
        B += s * xexp(full ? sqrt(lr * lr + li * li) * t : fmax(0.0, lr * t));
    }
    return B;
}

// derivatives d[0..nd] of f at sig (closed form)
void GSolver::feval(const RowFn& f, double sig, int nd, double* d) const {
    const SetBasis& b = *f.F->b;
    const double L = f.F->L;
    const int n2 = 2 * n_, ns = b.ns, s1 = ns + b.nc;
    for (int k = 0; k <= nd; ++k) d[k] = 0.0;
    d[0] = f.c0;
    for (int i = 0; i < n2;) {
        if (i == ns && s1 > ns) { i = s1; continue; }
        const double re = b.re[i], t = sig - ((i < ns) ? 0.0 : L), e = xexp(re * t);
        if (b.im[i] == 0.0) {
            double v = f.u(i) * e;
            for (int k = 0; k <= nd; ++k) { d[k] += v; v *= re; }
            ++i;
        } else {
            const double om = b.im[i], c = cos(om * t), sn = sin(om * t), P = f.u(i), Q = f.u(i + 1);
            double x = e * (P * c + Q * sn), y = e * (P * sn - Q * c);
            for (int k = 0; k <= nd; ++k) {
                d[k] += x;
                const double xn = re * x - om * y;
                y = re * y + om * x;
                x = xn;
            }
            i += 2;
        }
    }
    ptaylor(f, sig, nd, d);
    sadd(f, sig, nd, d);
}

// adds the derivatives 0..nd at sig of the slow blocks (sum_k Re[A_k g_k], closed form)
void GSolver::sadd(const RowFn& f, double sig, int nd, double* d) const {
    const ArcFn& F = *f.F;
    for (int q = 0; q < F.nsb; ++q) {
        double A[NPOLY + 1][2], gk[NPOLY + 2][2];
        const double lr = F.b->re[F.b->ns + F.sc[q]], li = F.b->im[F.b->ns + F.sc[q]];
        gfun(lr, li, sig, F.sk - 1, gk);
        for (int n = 0; n <= nd; ++n) {
            scoef(f, q, n, A);
            for (int k = 0; k < F.sk; ++k) d[n] += A[k][0] * gk[k][0] - A[k][1] * gk[k][1];
        }
    }
}

// adds the derivatives 0..nd at sig of the centre polynomial: k! sum_j C(j, k) p_j sig^(j - k)
void GSolver::ptaylor(const RowFn& f, double sig, int nd, double* d) const {
    const int np = f.F->np;
    double fk = 1.0;
    for (int k = 0; k <= nd && k < np; ++k) {
        double s = 0.0, c = 1.0, sp = 1.0;
        for (int j = k; j < np; ++j) {
            s += f.p(j) * c * sp;
            c = c * (j + 1) / (j + 1 - k);
            sp *= sig;
        }
        d[k] += fk * s;
        fk *= k + 1;
    }
}

// One node [lo, hi] of bb(): d = (f, f') at the midpoint m, *M2 >= |f''| on the node (modal
// envelopes, largest at the end nearer the anchor, plus the centre polynomial's second derivative
// bounded by its Taylor coefficients at m). Returns an upper bound of f on the node: the smaller of
// f(m) + |f'(m)| h + M2 h^2 / 2 and the same bound for the slow modes plus the envelopes
// |A| e^{Re lam (t - anc)} of the fast ones (|lam| h > 1), which only decay away from their anchor.
double GSolver::node(const RowFn& f, double lo, double hi, double* d, double* M2p) const {
    const SetBasis& b = *f.F->b;
    const int n2 = 2 * n_, ns = b.ns, s1 = ns + b.nc, np = f.F->np;
    const double m = 0.5 * (lo + hi), h = 0.5 * (hi - lo), L = f.F->L;
    double d0 = f.c0, d1 = 0.0, M2 = 0.0, v = f.c0, v1 = 0.0, M2s = 0.0, E = 0.0;
    for (int i = 0; i < n2;) {
        if (i == ns && s1 > ns) { i = s1; continue; }
        const double re = b.re[i], anc = (i < ns) ? 0.0 : L, om = b.im[i];
        const double e = xexp(re * (m - anc)), ee = xexp(re * (((re > 0) ? hi : lo) - anc));
        double x, x1, A;
        if (om == 0.0) {
            const double u = f.u(i);
            x = u * e; x1 = re * x; A = fabs(u);
            ++i;
        } else {
            const double P = f.u(i), Q = f.u(i + 1), c = cos(om * (m - anc)), sn = sin(om * (m - anc));
            x = e * (P * c + Q * sn);
            const double y = e * (P * sn - Q * c);
            x1 = re * x - om * y;
            A = sqrt(P * P + Q * Q);
            i += 2;
        }
        const double l2 = re * re + om * om;
        d0 += x; d1 += x1; M2 += A * l2 * ee;
        if (l2 * h * h > 1.0) E += A * ee;
        else { v += x; v1 += x1; M2s += A * l2 * ee; }
    }
    if (np > 0) {                                      // p''(m + t) = sum_k p^(k+2)(m) t^k / k!
        double pd[NPOLY + 1] = {0.0}, s = 0.0, hk = 1.0;
        ptaylor(f, m, (np > 1) ? np - 1 : 1, pd);
        for (int k = 0; k + 2 < np; ++k) { s += fabs(pd[k + 2]) * hk; hk *= h / (k + 1); }
        d0 += pd[0]; d1 += pd[1]; M2 += s;
        v += pd[0]; v1 += pd[1]; M2s += s;
    }
    if (f.F->nsb) {                                    // slow blocks: exact f, f'; |f''| bound on [0, hi]
        double sd[2] = {0.0, 0.0};
        sadd(f, m, 1, sd);
        const double s2 = sbound(f, 2, hi, false);
        d0 += sd[0]; d1 += sd[1]; M2 += s2;
        v += sd[0]; v1 += sd[1]; M2s += s2;
    }
    d[0] = d0;
    d[1] = d1;
    *M2p = M2;
    return fmin(d0 + fabs(d1) * h + 0.5 * M2 * h * h, v + fabs(v1) * h + 0.5 * M2s * h * h + E);
}

// root of f in [lo, hi] with f(lo) <= 0 < f(hi): safeguarded Newton
double GSolver::froot(const RowFn& f, double lo, double hi) const {
    double x = 0.5 * (lo + hi), d[2];
    for (int it = 0; it < 200; ++it) {
        feval(f, x, 1, d);
        if (d[0] <= 0.0) lo = x;
        else hi = x;
        double xn = (d[1] != 0.0) ? x - d[0] / d[1] : 0.5 * (lo + hi);
        if (!(xn > lo && xn < hi)) xn = 0.5 * (lo + hi);
        if (xn <= lo || xn >= hi || xn == x) break;
        x = xn;
    }
    return 0.5 * (lo + hi);
}

// Branch and bound over the dyadic subintervals of [a, b], depth-first, left to right (no stack).
// A node [lo, hi] (midpoint m, half-width h) is cleared when its upper bound (node()) is at most
// the level, or when f is monotone on it (|f'(m)| >= M2 h; its maximum is then an end).
// first: the first point where f > thr; *root is the crossing of 0 that brackets it, *simple when
// f is increasing there. Otherwise: max f to within atol above max(level, floor), returned.
// NaN when f is not finite.
double GSolver::bb(const RowFn& f, double a, double b, double thr, double atol, bool first,
                   double* root, bool* simple) const {
    const SetBasis& sb = *f.F->b;
    if (sb.nc == 2 * n_ && !f.F->nsb) return pbb(f, a, b, thr, first, root, simple);
    const double minw = 1e-14 * p_->T;
    double best = -HUGE_VAL;
    int level = 0;
    uint64_t k = 0;
    for (;;) {
        const double w = ldexp(b - a, -level), lo = a + (double)k * w;
        const double hi = (k + 1 == ((uint64_t)1 << level)) ? b : lo + w;
        const double m = 0.5 * (lo + hi), h = 0.5 * (hi - lo);
        double d[2], e[1], M2;
        const double ub = node(f, lo, hi, d, &M2);
        if (!(ub < HUGE_VAL) || !isfinite(d[0])) return NAN;   // non-finite data: every test fails
        const double lvl = first ? thr : fmax(fmax(best, d[0]), thr) + atol;
        best = fmax(best, d[0]);
        const bool inc = d[1] - M2 * h >= 0.0, dec = d[1] + M2 * h <= 0.0;
        bool leaf = true;
        if (ub <= lvl) {
        } else if (dec) {                              // max at lo
            if (!first) { feval(f, lo, 0, e); best = fmax(best, e[0]); }
        } else if (inc || h <= minw || level >= 60) {
            feval(f, hi, 0, e);
            best = fmax(best, e[0]);
            if (first && fmax(d[0], e[0]) > thr) {
                const double x1 = (!inc && d[0] > e[0]) ? m : hi;
                feval(f, lo, 0, e);
                if (e[0] <= 0.0) { *root = froot(f, lo, x1); *simple = inc; }
                else { *root = lo; *simple = false; }
                return 1.0;
            }
        } else {
            leaf = false;
        }
        if (!leaf) { ++level; k *= 2; continue; }      // descend to the left child
        while (level > 0 && (k & 1)) { --level; k >>= 1; }   // up while a right child
        if (level == 0) return first ? 0.0 : best;
        ++k;                                           // right sibling
    }
}

// bb() for a pure polynomial row (a vertex arc: H nilpotent, no modes), exactly: the critical points
// are the roots of p', isolated from the roots of p'', ... (each derivative is monotone between the
// roots of the next one; one root per sign change, safeguarded Newton). Same contract as bb().
double GSolver::pbb(const RowFn& f, double a, double b, double thr, bool first, double* root,
                    bool* simple) const {
    const int np = f.F->np;
    double P[NPOLY] = {0.0}, Pl[NPOLY], Pd[NPOLY], x[NPOLY + 1], y[NPOLY + 1];
    for (int k = 0; k < np; ++k) P[k] = f.p(k) + ((k == 0) ? f.c0 : 0.0);
    if (!isfinite(P[0])) return NAN;
    int lv = 0;                                        // Pl = p^(lv), Pd = p^(lv+1)
    auto level = [&](int l) {
        for (int k = 0; k < np; ++k) Pl[k] = P[k];
        for (int j = 0; j <= l; ++j) {
            for (int k = 0; k + j + 1 < np; ++k) Pd[k] = Pl[k + 1] * (k + 1);
            if (j < l) for (int k = 0; k + j + 1 < np; ++k) Pl[k] = Pd[k];
        }
        lv = l;
    };
    auto val = [&](int l, double t) { if (l != lv) level(l); return horner(Pl, np - l, t); };
    auto zero = [&](int l, double lo, double hi) { if (l != lv) level(l); return pzero(Pl, Pd, np - l, lo, hi); };
    level(0);
    int nx = 0;                                        // roots of p^(l) in (a, b), sorted, l = np-2 .. 1
    for (int l = np - 2; l >= 1; --l) {
        int ny = 0;
        double lo = a, vlo = val(l, a);
        for (int i = 0; i <= nx; ++i) {
            const double hi = (i < nx) ? x[i] : b, vhi = val(l, hi);
            if ((vlo < 0 && vhi > 0) || (vlo > 0 && vhi < 0)) y[ny++] = zero(l, lo, hi);
            lo = hi; vlo = vhi;
        }
        for (int i = 0; i < ny; ++i) x[i] = y[i];
        nx = ny;
    }
    // p is monotone between a, x[0..nx), b
    double best = -HUGE_VAL, lo = a, vlo = val(0, a);
    for (int i = 0; i <= nx; ++i) {
        const double hi = (i < nx) ? x[i] : b, vhi = val(0, hi);
        best = fmax(best, fmax(vlo, vhi));
        if (first && fmax(vlo, vhi) > thr) {
            if (vlo <= 0.0 && vhi > 0.0) { *root = zero(0, lo, hi); *simple = true; }
            else { *root = lo; *simple = false; }
            return 1.0;
        }
        lo = hi; vlo = vhi;
    }
    return first ? 0.0 : best;
}

// local scale of the Taylor sign tests: |lam| h <= 2 for the fastest mode
double GSolver::hloc(const SetBasis& b) const {
    double r = b.tcn;
    for (int i = 0; i < 2 * n_; ++i) r = fmax(r, sqrt(b.re[i] * b.re[i] + b.im[i] * b.im[i]));
    return fmin(p_->T, 2.0 / fmax(r, 1e-300));
}

// Taylor coefficients cs[0..GN) at sig of the row g0 + G wt
void GSolver::rtaylor(const ArcFn& F, const double* G, double g0, double sig, double* cs) const {
    RowFn f;
    rowfn(F, G, g0, f);
    feval(f, sig, GN - 1, cs);
    double fk = 1.0;
    for (int k = 1; k < GN; ++k) { fk *= k; cs[k] /= fk; }
}

bool GSolver::glex_ok(uint16_t S, const ArcFn& F, double sig, double h, double* tol) const {
    int keys[2 * MAXM], il;
    double G[2 * MAXM][MAXM], g0[2 * MAXM];
    const int nk = gphi(S, keys, G, g0);
    for (int r = 0; r < nk; ++r)
        if (lead(F, G[r], g0[r], sig, h, &tol[r], &il) > 0) return false;
    return true;
}

// the minimiser's set just right of sig; tol[r] gets the lead() tolerances of its rows
bool GSolver::glex_set(const ArcFn& F, double sig, double h, uint16_t guess, uint16_t* out, double* tol) const {
    if (guess != 0xFFFFu && valid_set(guess) && glex_ok(guess, F, sig, h, tol)) { *out = guess; return true; }
    return for_sets([&](uint16_t S) {
        if (!glex_ok(S, F, sig, h, tol)) return false;
        *out = S;
        return true;
    });
}

// Leading Taylor coefficient at sig of the row g0 + G wt (coefficients that matter at scale h;
// 0 if none), its order and the tolerance used
double GSolver::lead(const ArcFn& F, const double* G, double g0, double sig, double h, double* tolp,
                     int* ilead) const {
    {   // cheap case: f(sig) above a bound of the tolerance (sum_k |c_k| h^k <= this sum)
        RowFn f;
        rowfn(F, G, g0, f);
        const SetBasis& b = *F.b;
        const int n2 = 2 * n_, ns = b.ns, s1 = ns + b.nc;
        double B = fabs(f.c0) + R_, d0, x = 1.0;
        for (int i = 0; i < n2;) {
            if (i == ns && s1 > ns) { i = s1; continue; }
            const double re = b.re[i], om = b.im[i], t = sig - ((i < ns) ? 0.0 : F.L);
            const double A = (om == 0.0) ? fabs(f.u(i)) : sqrt(f.u(i) * f.u(i) + f.u(i + 1) * f.u(i + 1));
            B += A * xexp(re * t + sqrt(re * re + om * om) * h);
            i += (om == 0.0) ? 1 : 2;
        }
        for (int k = 0; k < F.np; ++k) { B += fabs(f.p(k)) * x; x *= fabs(sig) + h; }
        if (F.nsb) B += sbound(f, 0, fabs(sig) + h, true);
        feval(f, sig, 0, &d0);
        if (fabs(d0) > 1e-12 * B) { *tolp = 1e-12 * B; *ilead = 0; return d0; }
    }
    double cs[GN], tol = 0.0, hk = 1.0;
    rtaylor(F, G, g0, sig, cs);
    for (int k = 0; k < GN; ++k) { tol += fabs(cs[k]) * hk; hk *= h; }
    tol = 1e-12 * (tol + R_);
    *tolp = tol;
    *ilead = 0;
    hk = 1.0;
    for (int k = 0; k < GN; ++k, hk *= h)
        if (fabs(cs[k]) * hk > tol) { *ilead = k; return cs[k]; }
    return 0.0;
}

// Events of set S's switching rows on (sig, L] of the arc of F (a row whose leading Taylor
// coefficient at sig is positive switches at sig; when glex_ok(S) at sig just passed, lexed holds
// its row tolerances and none does): first root per row, *s1 the earliest.
int GSolver::events(const ArcFn& F, uint16_t S, double sig, double h, double* rootk, bool* simplek,
                    int* keyk, double* s1p, const double* lexed) {
    int keys[2 * MAXM], nc = 0;
    double G[2 * MAXM][MAXM], g0[2 * MAXM], s1 = HUGE_VAL;
    const int nk = gphi(S, keys, G, g0);
    for (int r = 0; r < nk; ++r) {
        double tol = lexed ? lexed[r] : 0.0;           // lexed: glex_ok(S) at sig passed, no lead > 0
        int ilead;
        if (!lexed && lead(F, G[r], g0[r], sig, h, &tol, &ilead) > 0) {
            rootk[nc] = sig; simplek[nc] = (ilead == 1); keyk[nc++] = keys[r];
            if (sig < s1) s1 = sig;
            continue;
        }
        RowFn f;
        rowfn(F, G[r], g0[r], f);
        double root;
        bool simple;
        const double thr = fmax(tol, 1e-12 * (f.scale + R_));
        wthr_ = fmax(wthr_, thr);
        const double fb = (sig < F.L) ? bb(f, sig, F.L, thr, 0.0, true, &root, &simple) : 0.0;
        if (fb != fb) return -1;
        if (fb > 0) {
            rootk[nc] = root; simplek[nc] = simple; keyk[nc++] = keys[r];
            if (root < s1) s1 = root;
        }
    }
    *s1p = s1;
    return nc;
}

// Exact event walk of the pointwise H-minimiser along the current BVP solution -> wseq_/wtau_.
// Per arc of the iterate, the switching rows are exponential-polynomials: the first crossing is
// found by bb() (certified, no grid), the set after it by the Taylor sign tests at the event.
bool GSolver::gwalk(const uint16_t* seq, int N, const double* tau, int* ndeg) {
    const double T = p_->T;
    *ndeg = 0;
    int ns = 0;
    if (!p_->has_bounds) {
        wseq_[0] = 0;
        wn_ = 1;
        return true;
    }
    uint16_t S = 0xFFFFu;
    ArcFn F;
    wthr_ = 0.0;
    for (int ka = 0; ka < N; ++ka) {
        Arc& a = ac_;
        if (!arc(ka, seq, N, tau, a)) return false;
        a.x = R_out->x[ka];
        if (!chan(a, F)) return false;
        const double h = hloc(*a.b), t0 = a.t0;
        double ltol[2 * MAXM];
        bool lexed = false;
        if (S == 0xFFFFu) {
            if (!glex_set(F, 0.0, h, 0, &S, ltol)) return false;    // (a) initial set
            wseq_[ns++] = S;
            lexed = true;
        }
        double sig = 0.0;
        int stall = 0;
        for (;;) {
            double rootk[2 * MAXM], s1 = HUGE_VAL;
            bool simplek[2 * MAXM];
            int keyk[2 * MAXM];
            const int nc = events(F, S, sig, h, rootk, simplek, keyk, &s1, lexed ? ltol : nullptr);
            lexed = false;
            if (nc < 0) return false;
            if (nc == 0) break;
            if (t0 + s1 >= T * (1 - 1e-14)) break;
            int ngroup = 0, gkey = -1;
            bool gsimple = false;
            uint16_t gmask = 0;
            for (int q = 0; q < nc; ++q)
                if (rootk[q] <= s1 + 1e-12 * T) {
                    ++ngroup; gkey = keyk[q]; gsimple = simplek[q];
                    gmask |= (uint16_t)(1u << keyk[q]);
                }
            uint16_t Snew = 0xFFFFu;
            if (ngroup == 1 && gsimple) {
                const uint16_t cand = (uint16_t)(S ^ (1u << gkey));
                if (valid_set(cand)) Snew = cand;
            }
            if (Snew == 0xFFFFu) {
                ++*ndeg;
                const uint16_t guess = (uint16_t)(S ^ gmask);
                if (!glex_set(F, s1, h, valid_set(guess) ? guess : 0xFFFFu, &Snew, ltol)) return false;
                lexed = true;
            }
            if (Snew != S) {
                if (ns >= NA) { capfail_ = true; return false; }
                wtau_[ns - 1] = t0 + s1;
                wseq_[ns++] = Snew;
                S = Snew;
                stall = 0;
            } else if (++stall > 3) {
                return false;
            }
            sig = s1;
        }
    }
    wn_ = ns;
    return true;
}

// max over the trajectory of the switching rows of each arc's own set (<= 0 at a PMP point)
double GSolver::gverify(const uint16_t* seq, int N, const double* tau) {
    double worst = 0.0;
    const double lv = 0.1 * VTOL * R_;
    ArcFn F;
    RowFn f;
    for (int ka = 0; ka < N; ++ka) {
        Arc& a = ac_;
        if (!arc(ka, seq, N, tau, a)) return HUGE_VAL;
        a.x = R_out->x[ka];
        int keys[2 * MAXM];
        double G[2 * MAXM][MAXM], g0[2 * MAXM];
        const int nk = gphi(a.S, keys, G, g0);
        if (nk == 0) continue;
        if (!chan(a, F)) return HUGE_VAL;
        for (int r = 0; r < nk; ++r) {
            rowfn(F, G[r], g0[r], f);
            const double v = bb(f, 0.0, a.L, lv, lv, false, nullptr, nullptr);
            if (v != v) return HUGE_VAL;
            worst = fmax(worst, v);
        }
    }
    return worst;
}

namespace {
// phi1(z) = (e^z - 1) / z for complex z = (zr, zi) with zr <= ~0
void cphi1(double zr, double zi, double* or_, double* oi) {
    if (zr * zr + zi * zi < 0.25) {                    // series: sum z^k / (k+1)!
        double tr = 1.0, ti = 0.0, sr = 1.0, si = 0.0;
        for (int k = 1; k < 20; ++k) {
            const double nr = (tr * zr - ti * zi) / (k + 1), ni = (tr * zi + ti * zr) / (k + 1);
            tr = nr; ti = ni;
            sr += tr; si += ti;
        }
        *or_ = sr; *oi = si;
        return;
    }
    const double e = xexp(zr), nr = e * cos(zi) - 1.0, ni = e * sin(zi), d = zr * zr + zi * zi;
    *or_ = (nr * zr + ni * zi) / d;
    *oi = (ni * zr - nr * zi) / d;
}
}  // namespace

// I_j = int_sa^sb wt_j on the arc of F (closed form): modes e^{lam (sa - anc)} len phi1(lam len)
// (stable) or e^{lam (sb - anc)} len phi1(-lam len) (unstable), plus the integrated polynomial
void GSolver::wint(const ArcFn& F, double sa, double sb, double* I) const {
    const SetBasis& b = *F.b;
    const int n2 = 2 * n_, ns = b.ns, s1 = ns + b.nc;
    const double len = sb - sa;
    for (int j = 0; j < m_; ++j) {
        double s = F.c[j] * len, ka = sa, kb = sb;
        for (int k = 0; k < F.np; ++k) { s += F.poly[j][k] * (kb - ka) / (k + 1); ka *= sa; kb *= sb; }
        for (int q = 0; q < F.nsb; ++q) {              // slow: int g_k = g_(k+1)
            double ga[NPOLY + 2][2], gb[NPOLY + 2][2];
            const double lr = b.re[ns + F.sc[q]], li = b.im[ns + F.sc[q]];
            gfun(lr, li, sa, F.sk, ga);
            gfun(lr, li, sb, F.sk, gb);
            const double* s0 = F.sl[j][F.sc[q] - F.nz];
            const double* s1 = F.sl[j][F.sc[q] - F.nz + ((li != 0.0) ? 1 : 0)];
            for (int k = 0; k < F.sk; ++k)
                s += s0[k] * (gb[k + 1][0] - ga[k + 1][0]) - ((li != 0.0) ? s1[k] : 0.0) * (gb[k + 1][1] - ga[k + 1][1]);
        }
        I[j] = s;
    }
    for (int i = 0; i < n2;) {
        if (i == ns && s1 > ns) { i = s1; continue; }
        const bool pair = b.im[i] != 0.0, st = i < ns;
        const double re = b.re[i], om = pair ? b.im[i] : 0.0, sg = st ? len : -len;
        const double t = st ? sa : sb - F.L, e = xexp(re * t), c = cos(om * t), sn = sin(om * t);
        double pr, pi;
        cphi1(re * sg, om * sg, &pr, &pi);
        const double gr = len * e * (c * pr - sn * pi), gi = len * e * (c * pi + sn * pr);   // e^{lam t} len phi1
        for (int j = 0; j < m_; ++j) {
            const double P = F.amp[j][i], Q = pair ? F.amp[j][i + 1] : 0.0;
            I[j] += P * gr + Q * gi;                   // Re[(P - iQ)(gr + i gi)]
        }
        i += pair ? 2 : 1;
    }
}

// lambda(0)'x0 - lambda(T)'s(T)
double GSolver::gends(const uint16_t* seq, int N, const double* tau) const {
    double z0[N2 + 1], J = 0.0;
    zstart(seq, N, tau, 0, z0);
    for (int i = 0; i < n_; ++i) J += z0[n_ + i] * x0_[i] - zT_[n_ + i] * zT_[i];
    return J;
}

// J in closed form (ends = gends()). Along an extremal d/dt (lambda's) = -s'Qs - |vt|^2 - mu'b (mu' (Mt vt - b) = 0),
// so J = 1/2 [lambda(0)'x0 - lambda(T)'s(T) - int mu'b dt] + terminal cost; mu is linear in wt
// on a saturated arc, and int wt = sum of modal phi1 terms + the integrated polynomial.
double GSolver::gcost(const uint16_t* seq, int N, const double* tau, double ends) {
    const int n = n_;
    double J = ends, mub = 0.0;
    ArcFn F;
    for (int ka = 0; ka < N; ++ka) {
        if (!seq[ka]) continue;
        Arc& a = ac_;
        if (!arc(ka, seq, N, tau, a)) return HUGE_VAL;
        a.x = R_out->x[ka];
        if (!chan(a, F)) return HUGE_VAL;
        double I[MAXM];                                // int_0^L wt_j
        wint(F, 0.0, a.L, I);
        int keys[2 * MAXM];
        double G[2 * MAXM][MAXM], g0[2 * MAXM];
        const int nk = gphi(seq[ka], keys, G, g0);
        for (int r = 0; r < nk; ++r) {
            if (!(seq[ka] & (1u << keys[r]))) continue;   // active row: -mu |a|^2 = G wt + g0
            double v = g0[r] * a.L;
            for (int j = 0; j < m_; ++j) v += G[r][j] * I[j];
            mub -= rhs_[keys[r]] * v / rownorm2_[keys[r]];
        }
    }
    J = 0.5 * (J - mub);
    if (p_->has_Qf)
        for (int i = 0; i < n; ++i)
            for (int k = 0; k < n; ++k)
                J += 0.5 * (zT_[i] - p_->xf[i]) * p_->Qf[i][k] * (zT_[k] - p_->xf[k]);
    return J;
}

void GSolver::gscale(double kap) {
    for (int k = 0; k < 2 * m_; ++k) rhs_[k] = rhs0_[k] * kap;
}

// max_t of (row r of the unscaled bounds) / bound along the current trajectory (exact max)
double GSolver::gmax_ratio(const uint16_t* seq, int N, const double* tau) {
    double r = 0.0;
    ArcFn F;
    RowFn f;
    for (int ka = 0; ka < N; ++ka) {
        Arc& a = ac_;
        if (!arc(ka, seq, N, tau, a)) return HUGE_VAL;
        a.x = R_out->x[ka];
        if (!chan(a, F)) return HUGE_VAL;
        for (int k = 0; k < 2 * m_; ++k) {
            if (!finite_[k] || !(rhs0_[k] > 0)) continue;
            double G[MAXM], g0 = 0.0;                  // rows_[k] (P wt + q) / rhs0
            for (int j = 0; j < m_; ++j) {
                double s = 0.0;
                for (int i = 0; i < m_; ++i) s += rows_[k][i] * a.P[i][j];
                G[j] = s / rhs0_[k];
                g0 += rows_[k][j] * a.q[j] / rhs0_[k];
            }
            rowfn(F, G, g0, f);
            const double v = bb(f, 0.0, a.L, -HUGE_VAL, 1e-6, false, nullptr, nullptr);
            if (v != v) return HUGE_VAL;
            r = fmax(r, v);
        }
    }
    return r;
}

// ---- linear BVP for fixed junction times ----
// relation rows rel_[r] (r < n): [z coefficients (2n) | rhs (col N2) | nb sensitivity columns]
void GSolver::init_rel() {
    for (int r = 0; r < n_; ++r) {
        for (int c = 0; c < NCOL; ++c) rel_[r][c] = 0;
        rel_[r][r] = 1;
        rel_[r][N2] = x0_[r];
    }
}

// Advance the relation from the start of arc a to its end (z-space at both ends).
bool GSolver::fstep(const Arc& a, int nb) {
    const SetBasis& b = *a.b;
    const int n = n_, n2 = 2 * n, ns = b.ns, nc = b.nc, s1 = ns + nc;
    const int c1 = N2 + 1 + nb;
    double t[N2 + 1], E[4], eu[N2];
    for (int i = s1; i < n2;) {                        // e^{-Lu L}: (e) or (e cos, e sin) per block
        const int sz = mblk(b.re, b.im, i, -a.L, E);
        eu[i] = E[0];
        if (sz == 2) eu[i + 1] = E[1];
        i += sz;
    }
    for (int r = 0; r < n; ++r) {
        double* row = rel_[r];
        rowmul(row, b.W, n2, n2, t);                        // modal coordinates: row W
        double d = row[N2];
        for (int j = 0; j < ns; ++j) d -= t[j] * a.yp[j];
        for (int j = s1; j < n2; ++j) d -= t[j] * a.yp[j];
        for (int i = s1; i < n2;) {                    // unstable part anchored at the arc end
            const int sz = (b.im[i] == 0.0) ? 1 : 2;
            E[0] = E[3] = eu[i];
            if (sz == 2) { E[1] = eu[i + 1]; E[2] = -eu[i + 1]; }
            if (sz == 1) t[i] *= E[0];
            else {
                const double u0 = t[i] * E[0] + t[i + 1] * E[2], u1 = t[i] * E[1] + t[i + 1] * E[3];
                t[i] = u0;
                t[i + 1] = u1;
            }
            i += sz;
        }
        for (int j = s1; j < n2; ++j) d += t[j] * a.yp[j];
        if (nc) {                                      // centre: [c; 1] = e^{-M L} [eta_c; 1]
            const double tu = t[s1];                   // t[s1] is the row's affine slot meanwhile
            t[s1] = 0.0;
            cexp(a, -a.L, t + ns, true, true);
            d -= t[s1];
            t[s1] = tu;
        }
        for (int j = 0; j < n2; ++j) row[j] = t[j];
        row[N2] = d;
    }
    // eliminate the stable coefficients a with E_s a - eta_s = -yp_s (lower rows); the column of
    // a_i then holds the coefficients of eta_i
    double low[2][NCOL], *R[MAXN + 2], e1[MAXN + 2], e2[MAXN + 2];
    for (int r = 0; r < n; ++r) R[r] = rel_[r];
    R[n] = low[0];
    R[n + 1] = low[1];
    for (int i = 0; i < ns;) {
        const int sz = mblk(b.re, b.im, i, a.L, E);
        for (int l = 0; l < sz; ++l) {
            for (int c = 0; c < c1; ++c) low[l][c] = 0;
            low[l][i] = E[2 * l];
            if (sz == 2) low[l][i + 1] = E[2 * l + 1];
            low[l][N2] = -a.yp[i + l];
        }
        for (int r = 0; r < n + 2; ++r) e1[r] = e2[r] = 0.0;
        e1[n] = -1.0;
        if (sz == 1) {
            reflect(R, n + 1, n, i, c1, e1, nullptr);
        } else {
            e2[n + 1] = -1.0;
            reflect(R, n + 2, n, i, c1, e1, e2);
            R[n] = low[1];                             // low[0] is the pivot row of a_i
            e2[n] = e2[n + 1];
            reflect(R, n + 1, n, i + 1, c1, e2, nullptr);
            R[n] = low[0];
        }
        i += sz;
    }
    for (int r = 0; r < n; ++r) {                      // back to z: eta = Wi z
        double* row = rel_[r];
        rowmul(row, b.Wi, n2, n2, t);
        for (int j = 0; j < n2; ++j) row[j] = t[j];
    }
    // re-orthonormalise the rows (modified Gram-Schmidt, twice), same ops on rhs/sensitivities
    for (int i = 0; i < n; ++i) {
        for (int pass = 0; pass < 2; ++pass)
            for (int k = 0; k < i; ++k) {
                double s = 0.0;
                for (int j = 0; j < n2; ++j) s += rel_[i][j] * rel_[k][j];
                for (int j = 0; j < n2; ++j) rel_[i][j] = (rel_[i][j] - s * rel_[k][j]);
                for (int j = N2; j < c1; ++j) rel_[i][j] = (rel_[i][j] - s * rel_[k][j]);
            }
        double nn = 0.0;
        for (int j = 0; j < n2; ++j) nn += rel_[i][j] * rel_[i][j];
        nn = sqrt(nn);
        if (nn == 0.0) return false;
        for (int j = 0; j < n2; ++j) rel_[i][j] = (rel_[i][j] / nn);
        for (int j = N2; j < c1; ++j) rel_[i][j] = (rel_[i][j] / nn);
    }
    return true;
}

// Vector-field jump (f_{j-1} - f_j)(z(tau_j)) = (Bt (vt_{j-1} - vt_j), 0): only the chain-end
// entries end_[i] are nonzero; D[i] holds them.
void GSolver::jump(const uint16_t* seq, int N, const double* tau, int j, double* D) const {
    double z[N2 + 1], wt[MAXM], P[MAXM][MAXM], q[MAXM], dv[MAXM];
    zstart(seq, N, tau, j, z);
    wt_of(z + n_, wt);
    for (int side = 0; side < 2; ++side) {
        affine(seq[j - 1 + side], P, q);
        for (int i = 0; i < m_; ++i) {
            double s = q[i];
            for (int k = 0; k < m_; ++k) s += P[i][k] * wt[k];
            dv[i] = side ? dv[i] - s : s;
        }
    }
    for (int jj = 0; jj < m_; ++jj) {
        double s = 0.0;
        for (int k = 0; k < m_; ++k) s += LiT_[jj][k] * dv[k];
        D[jj] = s;
    }
}

// sensitivity column of junction k enters the relation at tau_k (jumps of the block in Dj_)
void GSolver::inject(int k, int b0, int nb) {
    if (!nb || k < 1 || k - 1 < b0 || k - 1 >= b0 + nb) return;
    const int col = N2 + 1 + (k - 1 - b0);
    for (int r = 0; r < n_; ++r) {
        double s = 0.0;
        for (int i = 0; i < m_; ++i) s += rel_[r][end_[i]] * Dj_[k - 1 - b0][i];
        rel_[r][col] += s;
    }
}

bool GSolver::advance(const uint16_t* seq, int N, const double* tau, int k, int b0, int nb) {
    inject(k, b0, nb);
    return arc(k, seq, N, tau, ac_) && fstep(ac_, nb);
}

bool GSolver::forward(const uint16_t* seq, int N, const double* tau, int b0, int nb, int* stride) {
    const int st = (N + NCK) / NCK;                    // ceil((N + 1) / NCK)
    const int w = N2 + 1 + nb;
    init_rel();
    for (int k = 0; k <= N; ++k) {
        if (k > 0 && k % st == 0)
            for (int r = 0; r < n_; ++r)
                for (int c = 0; c < w; ++c) ck_[k / st - 1][r][c] = rel_[r][c];
        if (k < N && !advance(seq, N, tau, k, b0, nb)) return false;
    }
    *stride = st;
    return true;
}

// relation at tau_k^+ (start of arc k) from the nearest stored one
bool GSolver::load(const uint16_t* seq, int N, const double* tau, int k, int stride, int b0, int nb) {
    const int c = k / stride, w = N2 + 1 + nb;
    if (c == 0) init_rel();
    else
        for (int r = 0; r < n_; ++r) {
            for (int j = 0; j < w; ++j) rel_[r][j] = ck_[c - 1][r][j];
        }
    for (int j = c * stride; j < k; ++j)
        if (!advance(seq, N, tau, j, b0, nb)) return false;
    inject(k, b0, nb);                                 // relation at tau_k^+
    return true;
}

// Terminal conditions with the relation at T: s(T) = xf, or lambda(T) = Qf (s(T) - xf).
// Column 0 (rhs) -> zT (if given); sensitivity columns (homogeneous) -> Z_.
bool GSolver::terminal(int nb, double* zT) {
    const int n = n_;
    const bool qf = p_->has_Qf;
    double cs[MAXN], cl[MAXN];
    for (int r = 0; r < n; ++r) {
        double* row = rel_[r];
        for (int j = 0; j < n; ++j) { cs[j] = row[j]; cl[j] = row[n + j]; }
        double d = row[N2];
        for (int j = 0; j < n; ++j) {
            if (qf) {
                double s = cs[j], u = 0.0;
                for (int i = 0; i < n; ++i) {
                    const double q = 0.5 * (p_->Qf[i][j] + p_->Qf[j][i]);
                    s += cl[i] * q;
                    u += q * p_->xf[i];
                }
                row[j] = s;
                d += cl[j] * u;
            } else {
                row[j] = cl[j];
                d -= cs[j] * p_->xf[j];
            }
        }
        row[N2] = d;
    }
    if (!lu_solve(&rel_[0][0], n, NCOL, &rel_[0][N2], 1 + nb, NCOL)) return false;
    for (int c = 0; c <= nb; ++c) {
        double z[N2];
        for (int i = 0; i < n; ++i) {
            const double v = rel_[i][N2 + c];
            if (qf) z[i] = v;
            else { z[i] = (c == 0) ? p_->xf[i] : 0.0; z[n + i] = v; }
        }
        if (qf)
            for (int i = 0; i < n; ++i) {
                double s = 0.0;
                for (int j = 0; j < n; ++j)
                    s += 0.5 * (p_->Qf[i][j] + p_->Qf[j][i]) * (z[j] - ((c == 0) ? p_->xf[j] : 0.0));
                z[n + i] = s;
            }
        for (int i = 0; i < 2 * n; ++i) {
            if (c > 0) Z_[i][c - 1] = (real)z[i];
            else if (zT) zT[i] = z[i];
        }
    }
    return true;
}

// Backward step over arc a: given z(t1) (zb for the BVP column, Z_ for nb homogeneous
// sensitivity columns) and the relation at t0 in rel_, solve for the arc coefficients
// (-> xout for the BVP column) and replace the columns by z(t0).
bool GSolver::recover(const Arc& a, int nb, bool aff, double* xout, double* zb) {
    const SetBasis& b = *a.b;
    const int n = n_, n2 = 2 * n, ns = b.ns, nc = b.nc, s1 = ns + nc;
    const int ncols = aff ? 1 : nb, c0 = aff ? N2 : N2 + 1;
    if (ns > n) return false;
    double E[4], t[N2];
    // per column: eta = Wi z(t1) -> (eta_s | c | u0), u0 = y_u(t0)
    for (int c = 0; c < ncols; ++c) {
        double y[N2], e[N2 + 1];
        for (int i = 0; i < n2; ++i) y[i] = aff ? zb[i] : (double)Z_[i][c];
        matvec(b.Wi, n2, n2, y, e);
        if (nc) {
            double v[N2 + 1];
            for (int i = 0; i < nc; ++i) v[i] = e[ns + i];
            v[nc] = aff ? 1.0 : 0.0;
            cexp(a, -a.L, v, false, aff);
            for (int i = 0; i < nc; ++i) e[ns + i] = v[i];
        }
        if (aff) for (int i = s1; i < n2; ++i) e[i] -= a.yp[i];    // b
        if (aff && xout) for (int i = ns; i < n2; ++i) xout[i] = (e[i] * b.cs[i]);
        for (int i = s1; i < n2;) {
            const int sz = mblk(b.re, b.im, i, -a.L, E);
            if (sz == 1) e[i] = E[0] * e[i];
            else {
                const double u0 = E[0] * e[i] + E[1] * e[i + 1], u1 = E[2] * e[i] + E[3] * e[i + 1];
                e[i] = u0;
                e[i + 1] = u1;
            }
            i += sz;
        }
        if (aff) for (int i = s1; i < n2; ++i) e[i] += a.yp[i];
        for (int i = 0; i < n2; ++i) {
            if (aff) zb[i] = e[i];
            else Z_[i][c] = (real)e[i];
        }
    }
    // upper rows: (C W)_s a = rhs - (C W)_s yp_s - (C W)_c c - (C W)_u u0
    for (int r = 0; r < n; ++r) {
        double* row = rel_[r];
        rowmul(row, b.W, n2, n2, t);
        for (int c = 0; c < ncols; ++c) {
            double s = row[c0 + c];
            if (aff) for (int j = 0; j < ns; ++j) s -= t[j] * a.yp[j];
            for (int j = ns; j < n2; ++j) s -= t[j] * (aff ? zb[j] : (double)Z_[j][c]);
            row[c0 + c] = s;
        }
        for (int j = 0; j < ns; ++j) row[j] = t[j];
    }
    // QR of the upper a-part (Householder), then absorb the lower rows E_s a = eta_s - yp_s
    for (int k = 0; k < ns; ++k) {
        double v[MAXN], sg = 0.0;
        for (int r = k; r < n; ++r) { v[r] = rel_[r][k]; sg += v[r] * v[r]; }
        sg = sqrt(sg);
        if (sg == 0.0) continue;
        const double alpha = (v[k] > 0) ? -sg : sg;
        v[k] -= alpha;
        double vv = 0.0;
        for (int r = k; r < n; ++r) vv += v[r] * v[r];
        if (vv == 0.0) continue;
        for (int c = k; c < c0 + ncols; ++c) {
            if (c >= ns && c < c0) continue;
            double s = 0.0;
            for (int r = k; r < n; ++r) s += v[r] * rel_[r][c];
            s *= 2.0 / vv;
            for (int r = k; r < n; ++r) rel_[r][c] = (rel_[r][c] - s * v[r]);
        }
    }
    for (int i = 0; i < ns;) {
        const int sz = mblk(b.re, b.im, i, a.L, E);
        for (int l = 0; l < sz; ++l) {
            double lr[N2], lc[NB + 1];
            for (int j = 0; j < ns; ++j) lr[j] = 0.0;
            lr[i] = E[2 * l];
            if (sz == 2) lr[i + 1] = E[2 * l + 1];
            for (int c = 0; c < ncols; ++c)
                lc[c] = aff ? zb[i + l] - a.yp[i + l] : (double)Z_[i + l][c];
            for (int j = i; j < ns; ++j) {            // Givens: rotate the row into R row j
                if (lr[j] == 0.0) continue;
                const double rj = rel_[j][j], hh = sqrt(rj * rj + lr[j] * lr[j]);
                const double cg = rj / hh, sg = lr[j] / hh;
                for (int k = j; k < ns; ++k) {
                    const double u = rel_[j][k], w = lr[k];
                    rel_[j][k] = (cg * u + sg * w);
                    lr[k] = -sg * u + cg * w;
                }
                for (int c = 0; c < ncols; ++c) {
                    const double u = rel_[j][c0 + c], w = lc[c];
                    rel_[j][c0 + c] = (cg * u + sg * w);
                    lc[c] = -sg * u + cg * w;
                }
            }
        }
        i += sz;
    }
    for (int c = 0; c < ncols; ++c)                    // back substitution -> a in rel_ col c0+c
        for (int i = ns - 1; i >= 0; --i) {
            double s = rel_[i][c0 + c];
            for (int k = i + 1; k < ns; ++k) s -= rel_[i][k] * rel_[k][c0 + c];
            if (rel_[i][i] == 0) return false;
            rel_[i][c0 + c] = (s / rel_[i][i]);
        }
    // z(t0) = W (a + yp_s, c, u0)
    for (int c = 0; c < ncols; ++c) {
        double y[N2];
        for (int i = 0; i < ns; ++i) {
            y[i] = rel_[i][c0 + c] + (aff ? a.yp[i] : 0.0);
            if (aff && xout) xout[i] = (rel_[i][c0 + c] * b.cs[i]);
        }
        for (int i = ns; i < n2; ++i) y[i] = aff ? zb[i] : (double)Z_[i][c];
        matvec(b.W, n2, n2, y, t);
        for (int i = 0; i < n2; ++i) {
            if (aff) zb[i] = t[i];
            else Z_[i][c] = (real)t[i];
        }
    }
    return true;
}

// Linear BVP for fixed junction times -> arc coefficients R_out->x, zT_
bool GSolver::bvp(const uint16_t* seq, int N, const double* tau) {
    if (N > NA) { capfail_ = true; return false; }
    int stride;
    if (!forward(seq, N, tau, 0, 0, &stride) || !terminal(0, zT_)) return false;
    double zb[N2];
    for (int i = 0; i < 2 * n_; ++i) zb[i] = zT_[i];
    for (int k = N - 1; k >= 0; --k) {
        if (!load(seq, N, tau, k, stride, 0, 0) || !arc(k, seq, N, tau, ac_)) return false;
        if (!recover(ac_, 0, true, R_out->x[k], zb)) return false;
        for (int i = 0; i < 2 * n_; ++i)
            if (!isfinite(R_out->x[k][i])) return false;
    }
    for (int i = 0; i < 2 * n_; ++i)
        if (!isfinite(zT_[i])) return false;
    return true;
}

// Residual rows of the junction conditions: one per switching row, evaluated on the side where
// the row is inactive (control continuity). Returns -1 if there are more than NR.
int GSolver::jrows(const uint16_t* seq, int N) {
    int nr = 0;
    for (int j = 1; j < N; ++j) {
        const uint16_t L = seq[j - 1], Rr = seq[j], X = (uint16_t)(L ^ Rr);
        for (int key = 0; key < 2 * m_; ++key) {
            if (!(X & (1u << key))) continue;
            const uint16_t side = (Rr & (1u << key)) ? L : Rr;
            int keys[2 * MAXM];
            double Gs[2 * MAXM][MAXM], gs[2 * MAXM];
            const int nk = gphi(side, keys, Gs, gs);
            for (int q = 0; q < nk; ++q)
                if (keys[q] == key) {
                    if (nr >= NR) { capfail_ = true; return -1; }
                    jq_[nr] = (int8_t)j;
                    jk_[nr] = (int8_t)key;
                    ++nr;
                }
        }
    }
    nr_ = nr;
    return nr;
}

bool GSolver::rowphi(const uint16_t* seq, int q, double G[MAXM], double* g0) const {
    const int j = jq_[q], key = jk_[q];
    const uint16_t side = (seq[j] & (1u << key)) ? seq[j - 1] : seq[j];
    int keys[2 * MAXM];
    double Gs[2 * MAXM][MAXM], gs[2 * MAXM];
    const int nk = gphi(side, keys, Gs, gs);
    for (int r = 0; r < nk; ++r)
        if (keys[r] == key) {
            for (int c = 0; c < m_; ++c) G[c] = Gs[r][c];
            *g0 = gs[r];
            return true;
        }
    for (int c = 0; c < m_; ++c) G[c] = 0.0;
    *g0 = 0.0;
    return false;
}

double GSolver::resid(const uint16_t* seq, int N, const double* tau, double* rv) const {
    double mx = 0.0;
    for (int q = 0; q < nr_; ++q) {
        double z[N2 + 1], wt[MAXM], G[MAXM], g0 = 0.0;
        rowphi(seq, q, G, &g0);
        zstart(seq, N, tau, jq_[q], z);
        wt_of(z + n_, wt);
        double v = g0;
        for (int c = 0; c < m_; ++c) v += G[c] * wt[c];
        rv[q] = v;
        mx = fmax(mx, fabs(v));
    }
    return mx;
}

// Jacobian of the junction residuals w.r.t. the junction times (implicit part): moving tau_j
// injects the jump D_j into the homogeneous BVP. Columns in blocks of NB -> J_.
bool GSolver::sens(const uint16_t* seq, int N, const double* tau) {
    const int nj = N - 1;
    for (int b0 = 0; b0 < nj; b0 += NB) {
        const int nb = (nj - b0 < NB) ? nj - b0 : NB;
        int stride;
        for (int c = 0; c < nb; ++c) jump(seq, N, tau, b0 + c + 1, Dj_[c]);
        if (!forward(seq, N, tau, b0, nb, &stride) || !terminal(nb, nullptr)) return false;
        for (int k = N - 1; k >= 0; --k) {
            if (k + 1 <= N - 1)                        // Z_ = dz(tau_{k+1}^-): rows of junction k+1
                for (int q = 0; q < nr_; ++q) {
                    if (jq_[q] != k + 1) continue;
                    double G[MAXM], g0;
                    rowphi(seq, q, G, &g0);
                    for (int c = 0; c < nb; ++c) {
                        double lam[MAXN], wt[MAXM], v = 0.0;
                        for (int i = 0; i < n_; ++i) lam[i] = Z_[n_ + i][c];
                        wt_of(lam, wt);
                        for (int cc = 0; cc < m_; ++cc) v += G[cc] * wt[cc];
                        J_[q][b0 + c] = (real)v;
                    }
                }
            if (!load(seq, N, tau, k, stride, b0, nb) || !arc(k, seq, N, tau, ac_)) return false;
            if (!recover(ac_, nb, false, nullptr, nullptr)) return false;
            if (k >= 1 && k - 1 >= b0 && k - 1 < b0 + nb)    // left limit at tau_k
                for (int i = 0; i < m_; ++i)
                    Z_[end_[i]][k - 1 - b0] = (real)(Z_[end_[i]][k - 1 - b0] - Dj_[k - 1 - b0][i]);
        }
    }
    return true;
}

// Exact event: arc j shrinks to zero length along a Newton step. Equal neighbours -> merge
// them; neighbours that differ in two rows -> the two switches swap order; else drop the arc.
void GSolver::collapse(uint16_t* seq, int* N, double* tau, int j) const {
    const int n = *N;
    if (j == 0 || j == n - 1) {
        if (j == 0) {
            for (int k = 0; k + 1 < n; ++k) seq[k] = seq[k + 1];
            for (int k = 0; k + 2 < n; ++k) tau[k] = tau[k + 1];
        }
        *N = n - 1;
        return;
    }
    const uint16_t A = seq[j - 1], Mid = seq[j], B = seq[j + 1], D = (uint16_t)(A ^ B);
    const double mid = 0.5 * (tau[j - 1] + tau[j]);
    if (D == 0) {
        for (int k = j; k + 2 < n; ++k) seq[k] = seq[k + 2];
        for (int k = j - 1; k + 2 < n - 1; ++k) tau[k] = tau[k + 2];
        *N = n - 2;
        return;
    }
    const uint16_t first = (uint16_t)(A ^ Mid);
    const bool one = first && !(first & (first - 1));
    int nd = 0;
    for (uint16_t x = D; x; x &= (uint16_t)(x - 1)) ++nd;
    if (nd == 2 && one && (first & D) == first) {
        const uint16_t Mn = (uint16_t)(A ^ (D & ~first));
        if (valid_set(Mn)) {
            seq[j] = Mn;
            tau[j - 1] = mid;
            tau[j] = mid + 1e-9 * p_->T;
            return;
        }
    }
    for (int k = j; k + 1 < n; ++k) seq[k] = seq[k + 1];
    tau[j - 1] = mid;
    for (int k = j; k + 2 < n; ++k) tau[k] = tau[k + 1];
    *N = n - 1;
}

// explicit part of d r_q / d tau_k: the evaluation point moves, z(tau_k^-)' = zdot, whose lambda
// part -Q s - A^T lambda is the same on both sides
double GSolver::explicit_term(const uint16_t* seq, int N, const double* tau, int q) const {
    const int k = jq_[q];
    double z[N2 + 1], ld[MAXN], wt[MAXM], G[MAXM], g0, v = 0.0;
    zstart(seq, N, tau, k, z);
    for (int i = 0; i < n_; ++i) {
        double s = 0.0;
        for (int j = 0; j < n_; ++j) s -= 0.5 * (p_->Q[i][j] + p_->Q[j][i]) * z[j];
        ld[i] = s;
    }
    for (int j = 0; j < m_; ++j)
        for (int rr = 0; rr + 1 < p_->orders[j]; ++rr) ld[off_[j] + rr + 1] -= z[n_ + off_[j] + rr];
    wt_of(ld, wt);
    rowphi(seq, q, G, &g0);
    for (int c = 0; c < m_; ++c) v += G[c] * wt[c];
    return v;
}

// Newton on the junction times with exact derivatives. Arc collapse inside a step is an exact
// event. Fails fast when the residual does not contract. seq/N/tau are updated in place.
bool GSolver::newton_tau(uint16_t* seq, int* Np, double* tau, int* its) {
    *its = 0;
    int N = *Np;
    if (!bvp(seq, N, tau)) return false;
    int nr = jrows(seq, N);
    if (nr < 0) return false;
    if (nr == 0) return true;
    double r[NR], rn[NR], tn[NA], d[NR];
    double rmax = resid(seq, N, tau, r);
    const double tol = GTOL * R_, gap = 1e-12 * p_->T;
    double prev = HUGE_VAL;
    int bad = 0;
    while (*its < MAX_TAU_NEWTON && budget_ > 0) {
        const int nj = N - 1;
        if (rmax <= tol) { *Np = N; return true; }
        if (rmax > 0.5 * prev && ++bad > 2) { *Np = N; return rmax <= 1e3 * tol; }
        prev = rmax;
        ++*its;
        --budget_;
        if (nr < nj || !sens(seq, N, tau)) { *Np = N; return false; }
        for (int q = 0; q < nr; ++q) {
            const int k = jq_[q];
            J_[q][k - 1] = (real)(J_[q][k - 1] + explicit_term(seq, N, tau, q));
            J_[q][nj] = (real)(-r[q]);
        }
        // Gauss-Newton step: least squares J d = -r by Householder QR (in place)
        double jmax = 0.0;
        for (int q = 0; q < nr; ++q)
            for (int c = 0; c < nj; ++c) jmax = fmax(jmax, fabs(J_[q][c]));
        bool sing = !(jmax > 0.0);
        for (int k = 0; k < nj && !sing; ++k) {
            double sg = 0.0, *v = rn;                  // rn is free until the line search
            for (int q = k; q < nr; ++q) { v[q] = J_[q][k]; sg += v[q] * v[q]; }
            sg = sqrt(sg);
            if (sg <= 1e-15 * jmax) { sing = true; break; }
            const double alpha = (v[k] > 0) ? -sg : sg;
            v[k] -= alpha;
            double vv = 0.0;
            for (int q = k; q < nr; ++q) vv += v[q] * v[q];
            for (int c = k; c <= nj; ++c) {
                double s = 0.0;
                for (int q = k; q < nr; ++q) s += v[q] * J_[q][c];
                s *= 2.0 / vv;
                for (int q = k; q < nr; ++q) J_[q][c] = (real)(J_[q][c] - s * v[q]);
            }
        }
        if (sing) { *Np = N; return false; }
        for (int i = nj - 1; i >= 0; --i) {
            double s = J_[i][nj];
            for (int k = i + 1; k < nj; ++k) s -= J_[i][k] * d[k];
            d[i] = s / J_[i][i];
        }
        // exact arc-collapse event along the full step
        double amax = 1.0;
        int jc = -1;
        for (int j = 0; j <= nj; ++j) {
            const double L = ((j == nj) ? p_->T : tau[j]) - ((j == 0) ? 0.0 : tau[j - 1]);
            const double dL = ((j == nj) ? 0.0 : d[j]) - ((j == 0) ? 0.0 : d[j - 1]);
            if (dL < 0 && L + dL <= gap && L / -dL < amax) { amax = L / -dL; jc = j; }
        }
        if (jc >= 0) {
            uint16_t sq[NA];                          // undo copy (tn is free here)
            const int N0 = N;
            for (int j = 0; j < N; ++j) sq[j] = seq[j];
            for (int j = 0; j < nj; ++j) { tn[j] = tau[j]; tau[j] += amax * d[j]; }
            collapse(seq, &N, tau, jc);
            double pv = 0.0;
            bool ok = true;
            for (int j = 0; j + 1 < N; ++j) { ok = ok && tau[j] > pv; pv = tau[j]; }
            ok = ok && p_->T > pv && bvp(seq, N, tau) && (nr = jrows(seq, N)) >= 0;
            if (!ok) {                                // restore the iterate (callers walk R_out->x)
                N = N0;
                for (int j = 0; j < N; ++j) seq[j] = sq[j];
                for (int j = 0; j < nj; ++j) tau[j] = tn[j];
                bvp(seq, N, tau);
                jrows(seq, N);
                *Np = N;
                return false;
            }
            ++events_;
            if (nr == 0) { *Np = N; return true; }
            rmax = resid(seq, N, tau, r);
            prev = HUGE_VAL;
            bad = 0;
            continue;
        }
        double nr0 = 0.0;
        for (int q = 0; q < nr; ++q) nr0 += r[q] * r[q];
        nr0 = sqrt(nr0);
        double alpha = 1.0;
        bool acc = false;
        while (alpha > 1e-3) {
            for (int j = 0; j < nj; ++j) tn[j] = tau[j] + alpha * d[j];
            if (bvp(seq, N, tn)) {
                const double mx = resid(seq, N, tn, rn);
                double nn = 0.0;
                for (int q = 0; q < nr; ++q) nn += rn[q] * rn[q];
                if (sqrt(nn) <= (1 - 1e-4 * alpha) * nr0 || mx <= tol) { acc = true; rmax = mx; break; }
            }
            alpha *= 0.5;
        }
        if (!acc) {
            bvp(seq, N, tau);                         // restore the state of tau
            *Np = N;
            return rmax <= 1e3 * tol;                 // round-off stall
        }
        for (int j = 0; j < nj; ++j) tau[j] = tn[j];
        for (int q = 0; q < nr; ++q) r[q] = rn[q];
    }
    *Np = N;
    return rmax <= tol;
}

// Exact event walk along the current BVP solution -> wseq_/wtau_ (round-off chatter removed)
bool GSolver::propose(const uint16_t* seq, int N, const double* tau, int* ndeg) {
    if (!gwalk(seq, N, tau, ndeg)) return false;
    for (int i = 1; i + 1 < wn_;) {
        const double len = wtau_[i] - wtau_[i - 1];
        if (len < 1e-10 * p_->T && wseq_[i - 1] == wseq_[i + 1]) {
            for (int k = i; k + 2 < wn_; ++k) wseq_[k] = wseq_[k + 2];
            for (int k = i - 1; k + 2 < wn_ - 1; ++k) wtau_[k] = wtau_[k + 2];
            wn_ -= 2;
        } else {
            ++i;
        }
    }
    return true;
}

// Corrector at a fixed bound scale from the predicted structure aseq_/atau_: Newton on tau, then
// the exact walk; repeat with the walked structure (at most `rounds` times). On failure aseq_/atau_
// hold the last iterate's structure.
bool GSolver::correct(int* its, int* outer, int* ndeg, int rounds) {
    uint16_t seq[NA];
    double tau[NA];
    int N = an_, retry = (rounds > 3) ? rounds : 1;   // walks after a failed Newton (warm: every round)
    for (int i = 0; i < N; ++i) seq[i] = aseq_[i];
    for (int i = 0; i + 1 < N; ++i) tau[i] = atau_[i];
    bool ok = true;
    for (int o = 0; o < rounds && ok; ++o) {
        int k = 0;
        ok = newton_tau(seq, &N, tau, &k);
        *its += k;
        if (!ok && (budget_ <= 0 || retry-- == 0 || (o > 0 && rounds <= 3))) break;
        ok = propose(seq, N, tau, ndeg);
        if (!ok) break;
        bool same = (wn_ == N);
        for (int i = 0; same && i < N; ++i) {
            if (wseq_[i] != seq[i]) same = false;
            if (i > 0 && fabs(wtau_[i - 1] - tau[i - 1]) > STOL * p_->T) same = false;
        }
        if (same) {
            cert_ = true;
            an_ = N;
            for (int i = 0; i < N; ++i) aseq_[i] = seq[i];
            for (int i = 0; i + 1 < N; ++i) atau_[i] = tau[i];
            return true;
        }
        if (budget_ <= 0) { ok = false; break; }        // cap: return the Newton iterate
        N = wn_;
        for (int i = 0; i < N; ++i) seq[i] = wseq_[i];
        for (int i = 0; i + 1 < N; ++i) tau[i] = wtau_[i];
        ++*outer;
    }
    an_ = N;
    for (int i = 0; i < N; ++i) aseq_[i] = seq[i];
    for (int i = 0; i + 1 < N; ++i) atau_[i] = tau[i];
    return false;
}

double GSolver::zmax(const uint16_t* seq, int N, const double* tau) const {
    double m = 0.0, z[N2 + 1];
    for (int k = 0; k < N; ++k) {
        zstart(seq, N, tau, k, z);
        for (int i = 0; i < 2 * n_; ++i) m = fmax(m, fabs(z[i]));
    }
    for (int i = 0; i < 2 * n_; ++i) m = fmax(m, fabs(zT_[i]));
    return m;
}

void GSolver::begin(GResult& R, int max_its) {
    R_out = &R;
    R.iterations = R.outer_iterations = R.degenerate_events = 0;
    R.cost = R.residual = R.pmp_violation = HUGE_VAL;
    R.converged = false;
    R.failure = nullptr;
    for (int k = 0; k < 2 * m_; ++k) rhs_[k] = rhs0_[k];
    events_ = 0;
    capfail_ = false;
    budget_ = max_its;
    cert_ = false;
    wthr_ = 0.0;
    bkey_ = 0xFFFFu;
}

// "basis/dichotomy failed for key k" when a basis was missing (BasisFn returned null), else null
const char* GSolver::bfail() {
    if (bkey_ == 0xFFFFu) return nullptr;
    static const char pre[] = "basis/dichotomy failed for key ";
    int i = 0, k = bkey_;
    for (; pre[i]; ++i) why_[i] = pre[i];
    char d[6];
    int nd = 0;
    do { d[nd++] = (char)('0' + k % 10); k /= 10; } while (k);
    while (nd) why_[i++] = d[--nd];
    why_[i] = 0;
    return why_;
}

bool GSolver::solve(GResult& R) { return cold(R, MAX_GEN_ITS); }

bool GSolver::cold(GResult& R, int max_its) {
    begin(R, max_its);
    R.narcs = 0;
    if (!x0ok_) { R.failure = "non-finite x0"; return false; }
    int its = 0, outer = 0, ndeg = 0, nsteps = 0;
    an_ = 1;
    aseq_[0] = 0;
    bool conv = false;
    const char* why = nullptr;
    if (!bvp(aseq_, 1, atau_)) {
        R.failure = bfail() ? bfail() : "linear BVP singular (infeasible endpoint?)";
        return false;
    }
    if (!p_->has_bounds) {
        cert_ = conv = true;
    } else {
        const double ratio = gmax_ratio(aseq_, 1, atau_);
        if (!(ratio < HUGE_VAL)) { R.failure = "non-finite trajectory"; return false; }
        if (ratio <= 1.0 && propose(aseq_, 1, atau_, &ndeg))
            cert_ = conv = (wn_ == 1 && wseq_[0] == 0);
        if (!conv) {
            // homotopy in the bound scale kappa (NOTES.md 9.4): exact collapse events inside
            // the tau-Newton, secant predictor, step halving on failure
            const double z0 = zmax(aseq_, 1, atau_);
            double kap = 1.05 * fmax(ratio, 1.0);
            double dl = xlog(kap) * ((ratio <= KAP_PLAIN) ? 1.0 : 1.0 / 3.0);
            gscale(kap);
            if (!bvp(aseq_, 1, atau_)) { R.failure = "linear BVP failed"; return false; }
            hn_ = 0;
            uint16_t sseq[NA];
            double stau[NA];
            for (;;) {
                if (kap == 1.0) { conv = true; break; }
                if (budget_ <= 0) { why = "iteration cap reached"; break; }
                if (dl < DL_MIN) { why = "homotopy stalled at bound scale > 1 (bounds infeasible?)"; break; }
                if (zmax(aseq_, an_, atau_) > 1e8 * (1.0 + z0)) { why = "costate diverged (bounds infeasible?)"; break; }
                const int sn = an_;
                for (int i = 0; i < sn; ++i) sseq[i] = aseq_[i];
                for (int i = 0; i + 1 < sn; ++i) stau[i] = atau_[i];
                const double kt = fmax(1.0, kap * xexp(-dl));
                gscale(kt);
                ++nsteps;
                bool same = (hn_ == sn);
                for (int i = 0; same && i < sn; ++i) same = (hseq_[i] == sseq[i]);
                if (same) {                                  // secant predictor
                    double pv = 0.0;
                    bool ord = true;
                    for (int i = 0; i + 1 < sn; ++i) {
                        const double t = stau[i] + (stau[i] - htau_[i]) / (kap - hkap_) * (kt - kap);
                        atau_[i] = t;
                        if (!(t > pv)) ord = false;
                        pv = t;
                    }
                    if (!ord || !(p_->T > pv))
                        for (int i = 0; i + 1 < sn; ++i) atau_[i] = stau[i];
                }
                if (correct(&its, &outer, &ndeg, 3)) {
                    hn_ = sn;
                    for (int i = 0; i < sn; ++i) hseq_[i] = sseq[i];
                    for (int i = 0; i + 1 < sn; ++i) htau_[i] = stau[i];
                    hkap_ = kap;
                    kap = kt;
                    dl *= 1.5;
                } else {
                    if (capfail_) { why = "arc capacity NA exceeded"; break; }
                    dl *= 0.5;
                    gscale(kap);
                    an_ = sn;
                    for (int i = 0; i < sn; ++i) aseq_[i] = sseq[i];
                    for (int i = 0; i + 1 < sn; ++i) atau_[i] = stau[i];
                    if (!bvp(aseq_, an_, atau_)) { why = "linear BVP failed (bounds infeasible?)"; break; }
                }
            }
        }
    }
    return finish(R, conv, why, its, nsteps, ndeg);
}

// MPC warm start (NOTES.md 11.4): R holds the previous solution of this problem; the Problem's x0
// has been updated by the caller. The previous arc structure, with its junction times moved by
// -dt (arcs that ended are dropped, the last arc is stretched to T), is corrected by Newton on
// tau and the exact walk at the true bounds: no homotopy. If that fails with iterations left, the
// cold solve (homotopy) takes over. At most max_its Newton iterations in all; when the cap is hit,
// R is the current iterate (it satisfies the dynamics and both boundary conditions) with
// converged = false.
bool GSolver::solve_warm(GResult& R, double dt, int max_its) {
    if (R.narcs < 1 || R.narcs > NA || !x0ok_) return cold(R, max_its);
    const double T = p_->T;
    an_ = 0;
    for (int k = 0; k < R.narcs; ++k) {
        const bool last = (k == R.narcs - 1);
        const double t1 = last ? T : R.t1[k] - dt;
        if (!last && t1 <= 1e-9 * T) continue;         // this arc has ended
        aseq_[an_] = R.S[k];
        if (!last) atau_[an_] = t1;
        ++an_;
    }
    begin(R, max_its);
    int its = 0, outer = 0, ndeg = 0;
    if (correct(&its, &outer, &ndeg, 12)) return finish(R, true, nullptr, its, 0, ndeg);
    if (budget_ > 0 && !capfail_) {                    // cold fallback with the iterations left
        const bool ok = cold(R, budget_);
        R.iterations += its;
        return ok;
    }
    return finish(R, false, capfail_ ? "arc capacity NA exceeded" : "iteration cap reached", its, 0, ndeg);
}

bool GSolver::finish(GResult& R, bool conv, const char* why, int its, int nsteps, int ndeg) {
    R.iterations = its;
    R.outer_iterations = nsteps;
    R.degenerate_events = ndeg;
    bool loose = false;                                // stopped at a bound scale > 1: back to the true
    for (int k = 0; k < 2 * m_; ++k) loose = loose || rhs_[k] != rhs0_[k];   // bounds (eval_u, mean_u)
    gscale(1.0);
    // a converged corrector or walk left the BVP solution of aseq_/atau_ in place and certified it
    const bool cert = conv && cert_ && !loose;
    if (!cert && !bvp(aseq_, an_, atau_)) {
        R.narcs = 0;                                   // R.x is not a plan
        R.failure = why ? why : "linear BVP failed";
        return false;
    }
    R.narcs = an_;
    for (int k = 0; k < an_; ++k) {
        R.S[k] = aseq_[k];
        R.t1[k] = (k == an_ - 1) ? p_->T : atau_[k];
    }
    double rmax = 0.0;
    for (int i = 0; i < n_; ++i) {
        double v;
        if (p_->has_Qf) {
            v = zT_[n_ + i];
            for (int k = 0; k < n_; ++k) v -= p_->Qf[i][k] * (zT_[k] - p_->xf[k]);
        } else {
            v = zT_[i] - p_->xf[i];
        }
        rmax = fmax(rmax, fabs(v));
    }
    R.residual = rmax;
    // the walk's certificate suffices when its levels are below the verification tolerance
    R.pmp_violation = (cert && wthr_ <= VTOL * R_) ? wthr_ : gverify(R.S, R.narcs, R.t1);
    if (conv && !(R.pmp_violation <= VTOL * R_)) { conv = false; why = "optimality check failed"; }
    R.cost = loose ? HUGE_VAL : gcost(R.S, R.narcs, R.t1, gends(R.S, R.narcs, R.t1));
    R.converged = conv;
    R.failure = conv ? nullptr : bfail() ? bfail() : (why ? why : "not converged");
    return conv;
}

void GSolver::eval_x(const GResult& R, double t, double x[MAXN]) const {
    double z[N2 + 1];
    for (int i = 0; i < n_; ++i) x[i] = 0.0;
    if (R.narcs < 1) return;
    if (t < 0.0) t = 0.0;
    if (t > p_->T) t = p_->T;
    int k = 0;
    while (k < R.narcs - 1 && t >= R.t1[k]) ++k;
    Arc a;
    if (!arc(k, R.S, R.narcs, R.t1, a)) return;
    a.x = R.x[k];
    zat(a, t - a.t0, z);
    for (int i = 0; i < n_; ++i) x[i] = z[i];
}

// u = Lc^-T vt (R = Lc Lc^T), saturated to lo <= M u <= hi
void GSolver::clampu(double* u) const {
    for (int i = m_ - 1; i >= 0; --i) {
        double s = u[i];
        for (int k = i + 1; k < m_; ++k) s -= Lc_[k][i] * u[k];
        u[i] = s / Lc_[i][i];
    }
    if (!p_->has_bounds) return;
    double w[MAXM], A[MAXM * MAXM];
    bool out = false;
    for (int r = 0; r < m_; ++r) {
        double s = 0.0;
        for (int j = 0; j < m_; ++j) { s += p_->M[r][j] * u[j]; A[r * MAXM + j] = p_->M[r][j]; }
        w[r] = fmin(fmax(s, p_->lo[r]), p_->hi[r]);
        out = out || w[r] != s;
    }
    if (out && lu_solve(A, m_, MAXM, w, 1, 1))
        for (int j = 0; j < m_; ++j) u[j] = w[j];
}

bool GSolver::eval_u(const GResult& R, double t, double u[MAXM]) const {
    double z[N2 + 1], wt[MAXM], v[MAXM];
    if (R.narcs < 1) return false;
    if (t < 0.0) t = 0.0;
    if (t > p_->T) t = p_->T;
    int k = 0;
    while (k < R.narcs - 1 && t >= R.t1[k]) ++k;
    Arc a;
    if (!arc(k, R.S, R.narcs, R.t1, a)) return false;
    a.x = R.x[k];
    zat(a, t - a.t0, z);
    wt_of(z + n_, wt);
    for (int i = 0; i < m_; ++i) {
        double s = a.q[i];
        for (int j = 0; j < m_; ++j) s += a.P[i][j] * wt[j];
        v[i] = s;
    }
    clampu(v);
    for (int i = 0; i < m_; ++i) u[i] = v[i];
    return true;
}

// mean input over [t0, t1] (closed form); u is left unchanged on failure
bool GSolver::mean_u(const GResult& R, double t0, double t1, double u[MAXM]) const {
    double v[MAXM] = {0.0, 0.0, 0.0, 0.0};
    if (R.narcs < 1 || !(t1 > t0)) return false;
    Arc a;
    ArcFn F;
    for (int k = 0; k < R.narcs; ++k) {
        if (!arc(k, R.S, R.narcs, R.t1, a)) return false;
        const double sa = fmax(t0, a.t0) - a.t0, sb = fmin(t1, a.t0 + a.L) - a.t0;
        if (!(sb > sa)) continue;
        a.x = R.x[k];
        double I[MAXM];
        if (!chan(a, F)) return false;
        wint(F, sa, sb, I);
        for (int i = 0; i < m_; ++i) {
            double s = a.q[i] * (sb - sa);
            for (int j = 0; j < m_; ++j) s += a.P[i][j] * I[j];
            v[i] += s / (t1 - t0);
        }
    }
    clampu(v);
    for (int i = 0; i < m_; ++i) u[i] = v[i];
    return true;
}

int GSolver::switch_times(const GResult& R, double* out, int maxout) const {
    int k = 0;
    for (int a = 0; a + 1 < R.narcs && k < maxout; ++a) out[k++] = R.t1[a];
    return k;
}

}  // namespace lqocp
