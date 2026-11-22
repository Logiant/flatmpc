// Solver core. See lqocp.hpp and develop/NOTES.md. No heap, no exceptions, no STL.
#include "lqocp.hpp"

#include <math.h>
#include <string.h>

namespace lqocp {

namespace {

const double GL_X[4] = {-0.86113631159405257522, -0.33998104358485626480,
                        0.33998104358485626480, 0.86113631159405257522};
const double GL_W[4] = {0.34785484513745385737, 0.65214515486254614263,
                        0.65214515486254614263, 0.34785484513745385737};

double fact(int k) {
    double f = 1.0;
    for (int i = 2; i <= k; ++i) f *= i;
    return f;
}

double ipow(double x, int k) {
    double r = 1.0;
    for (int i = 0; i < k; ++i) r *= x;
    return r;
}

// In-place LU solve of A x = b (A is n x n stored with row stride ld). b overwritten by x.
// nrhs right-hand sides stored column-wise in B (row stride ldb). Returns false if singular.
bool lu_solve(double* A, int n, int ld, double* B, int nrhs, int ldb) {
    double amax = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) amax = fmax(amax, fabs(A[i * ld + j]));
    if (amax == 0.0) return false;
    for (int k = 0; k < n; ++k) {
        int piv = k;
        double best = fabs(A[k * ld + k]);
        for (int i = k + 1; i < n; ++i)
            if (fabs(A[i * ld + k]) > best) { best = fabs(A[i * ld + k]); piv = i; }
        if (best <= 1e-300 || best < 1e-15 * amax * 1e-3) return false;
        if (piv != k) {
            for (int j = 0; j < n; ++j) { double t = A[k * ld + j]; A[k * ld + j] = A[piv * ld + j]; A[piv * ld + j] = t; }
            for (int j = 0; j < nrhs; ++j) { double t = B[k * ldb + j]; B[k * ldb + j] = B[piv * ldb + j]; B[piv * ldb + j] = t; }
        }
        for (int i = k + 1; i < n; ++i) {
            double f = A[i * ld + k] / A[k * ld + k];
            if (f == 0.0) continue;
            for (int j = k + 1; j < n; ++j) A[i * ld + j] -= f * A[k * ld + j];
            for (int j = 0; j < nrhs; ++j) B[i * ldb + j] -= f * B[k * ldb + j];
        }
    }
    for (int j = 0; j < nrhs; ++j) {
        for (int i = n - 1; i >= 0; --i) {
            double s = B[i * ldb + j];
            for (int k = i + 1; k < n; ++k) s -= A[i * ld + k] * B[k * ldb + j];
            B[i * ldb + j] = s / A[i * ld + i];
        }
    }
    return true;
}

}  // namespace

bool Solver::setup(const Problem& p) {
    if (p.m < 1 || p.m > MAXM) return false;
    p_ = p;
    m_ = p.m;
    n_ = 0;
    for (int j = 0; j < m_; ++j) {
        if (p.orders[j] < 1 || p.orders[j] > MAXK) return false;
        off_[j] = n_;
        n_ += p.orders[j];
    }
    if (n_ > MAXN || !(p.T > 0.0)) return false;
    // energy cost (Q = 0, R = I, fixed endpoint) or general LQ cost
    general_ = false;
    if (p.has_cost) {
        bool energy = !p.has_Qf;
        for (int i = 0; i < n_; ++i)
            for (int j = 0; j < n_; ++j)
                if (p.Q[i][j] != 0.0) energy = false;
        for (int i = 0; i < m_; ++i)
            for (int j = 0; j < m_; ++j)
                if (p.R[i][j] != ((i == j) ? 1.0 : 0.0)) energy = false;
        general_ = !energy;
    }
    double Mrow[MAXM][MAXM];
    for (int r = 0; r < m_; ++r)
        for (int j = 0; j < m_; ++j) Mrow[r][j] = p.M[r][j];
    if (general_ && !gsetup(Mrow)) return false;   // rows become M Lc^{-T} (R-scaled inputs)
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
    phi_x0(p.T, free_);
    // newton-mode data
    R_ = 1.0;
    double mx = 0.0;
    for (int k = 0; k < 2 * m_; ++k) {
        double s = 0.0;
        for (int j = 0; j < m_; ++j) s += rows_[k][j] * rows_[k][j];
        rownorm2_[k] = s;
        if (finite_[k]) mx = fmax(mx, fabs(rhs_[k]));
    }
    R_ = 1.0 + mx;
    identity_ = p.has_bounds != 0 && !general_;
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
                const double binom = fact(mm) / (fact(q) * fact(mm - q));
                Lm_[(o + q) * MAXN + (o + r)] =
                    binom * ipow(p.T, mm - q) * ((q % 2) ? -1.0 : 1.0) / fact(mm);
            }
        }
    }
    {
        double A[MAXN * MAXN];
        for (int i = 0; i < MAXN * MAXN; ++i) A[i] = Lm_[i];
        for (int i = 0; i < n_; ++i) Linv_[i * MAXN + i] = 1.0;
        if (!lu_solve(A, n_, MAXN, Linv_, n_, MAXN)) return false;
    }
    return true;
}

void Solver::w_of(const double* c, double t, double* w) const {
    for (int j = 0; j < m_; ++j) {
        double s = 0.0, tp = 1.0;
        for (int k = 0; k < p_.orders[j]; ++k) { s += c[off_[j] + k] * tp; tp *= t; }
        w[j] = s;
    }
}

void Solver::phib(double dt, double K[MAXN][MAXM]) const {
    for (int i = 0; i < n_; ++i)
        for (int j = 0; j < m_; ++j) K[i][j] = 0.0;
    for (int j = 0; j < m_; ++j) {
        int k = p_.orders[j];
        for (int r = 0; r < k; ++r) K[off_[j] + r][j] = ipow(dt, k - 1 - r) / fact(k - 1 - r);
    }
}

void Solver::phi_x0(double dt, double* s) const {
    for (int j = 0; j < m_; ++j) {
        int k = p_.orders[j], o = off_[j];
        for (int r = 0; r < k; ++r) {
            double acc = 0.0;
            for (int l = r; l < k; ++l) acc += ipow(dt, l - r) / fact(l - r) * p_.x0[o + l];
            s[o + r] = acc;
        }
    }
}

bool Solver::affine(uint16_t S, double P[MAXM][MAXM], double q[MAXM]) const {
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
    for (int i = 0; i < m_; ++i) {
        for (int a = 0; a < na; ++a) {
            double ai = rows_[idx[a]][i];
            for (int j = 0; j < m_; ++j) P[i][j] -= ai * X[a * ldx + j];
            q[i] += ai * X[a * ldx + m_];
        }
    }
    return true;
}

bool Solver::try_set(uint16_t S, const double* w, double* v) const {
    int idx[MAXM], na = 0;
    for (int k = 0; k < 2 * m_; ++k)
        if (S & (1u << k)) { if (na >= MAXM || !finite_[k]) return false; idx[na++] = k; }
    for (int j = 0; j < m_; ++j) v[j] = w[j];
    if (na > 0) {
        double G[MAXM * MAXM], mu[MAXM];
        for (int a = 0; a < na; ++a) {
            double s = 0.0;
            for (int j = 0; j < m_; ++j) s += rows_[idx[a]][j] * w[j];
            mu[a] = s - rhs_[idx[a]];
            for (int b = 0; b < na; ++b) {
                double g = 0.0;
                for (int j = 0; j < m_; ++j) g += rows_[idx[a]][j] * rows_[idx[b]][j];
                G[a * MAXM + b] = g;
            }
        }
        if (!lu_solve(G, na, MAXM, mu, 1, 1)) return false;
        double mmax = 0.0;
        for (int a = 0; a < na; ++a) mmax = fmax(mmax, fabs(mu[a]));
        for (int a = 0; a < na; ++a)
            if (mu[a] < -1e-12 * (1.0 + mmax)) return false;
        for (int a = 0; a < na; ++a)
            for (int j = 0; j < m_; ++j) v[j] -= rows_[idx[a]][j] * mu[a];
    }
    const double tol = 1e-10 * rhs_scale_;
    for (int k = 0; k < 2 * m_; ++k) {
        if ((S & (1u << k)) || !finite_[k]) continue;
        double g = -rhs_[k];
        for (int j = 0; j < m_; ++j) g += rows_[k][j] * v[j];
        if (g > tol) return false;
    }
    return true;
}

bool Solver::project(const double* w, uint16_t guess, double* v, uint16_t* Sout) const {
    if (guess != 0xFFFFu && try_set(guess, w, v)) { *Sout = guess; return true; }
    for (int i = 0; i < nsets_; ++i) {
        if (try_set(sets_[i], w, v)) { *Sout = sets_[i]; return true; }
    }
    for (int j = 0; j < m_; ++j) v[j] = w[j];
    *Sout = 0;
    return false;
}

void Solver::set_times(Arc* arcs, int nseq, const double* tau) const {
    for (int i = 0; i < nseq; ++i) {
        arcs[i].t0 = (i == 0) ? 0.0 : tau[i - 1];
        arcs[i].t1 = (i == nseq - 1) ? p_.T : tau[i];
    }
}

bool Solver::build_arcs(const uint16_t* seq, int nseq, const double* tau, Arc* arcs) const {
    for (int i = 0; i < nseq; ++i) {
        arcs[i].S = seq[i];
        if (!affine(seq[i], arcs[i].P, arcs[i].q)) return false;
    }
    set_times(arcs, nseq, tau);
    return true;
}

void Solver::v_on(const Arc& a, const double* c, double t, double* v) const {
    double w[MAXM];
    w_of(c, t, w);
    for (int i = 0; i < m_; ++i) {
        double s = a.q[i];
        for (int j = 0; j < m_; ++j) s += a.P[i][j] * w[j];
        v[i] = s;
    }
}

void Solver::assemble(const Arc* arcs, int narcs, double* G, double* h) const {
    // s(T) = G c + h ; G is also the exact Jacobian ds(T)/dc.
    double K[MAXN][MAXM], KP[MAXN][MAXM];
    memset(G, 0, sizeof(double) * MAXN * MAXN);
    for (int i = 0; i < n_; ++i) h[i] = free_[i];
    for (int a = 0; a < narcs; ++a) {
        const Arc& A = arcs[a];
        if (!(A.t1 > A.t0)) continue;
        const double hw = 0.5 * (A.t1 - A.t0);
        for (int g = 0; g < 4; ++g) {
            const double t = A.t0 + hw * (GL_X[g] + 1.0);
            const double wg = GL_W[g] * hw;
            phib(p_.T - t, K);
            for (int i = 0; i < n_; ++i) {
                double kq = 0.0;
                for (int j = 0; j < m_; ++j) {
                    double s = 0.0;
                    for (int l = 0; l < m_; ++l) s += K[i][l] * A.P[l][j];
                    KP[i][j] = wg * s;
                    kq += K[i][j] * A.q[j];
                }
                h[i] += wg * kq;
            }
            for (int j = 0; j < m_; ++j) {
                double tp = 1.0;
                for (int k = 0; k < p_.orders[j]; ++k) {
                    const int col = off_[j] + k;
                    for (int i = 0; i < n_; ++i) G[i * MAXN + col] += KP[i][j] * tp;
                    tp *= t;
                }
            }
        }
    }
}

bool Solver::solve_c(const Arc* arcs, int narcs, double* c) const {
    double G[MAXN * MAXN], h[MAXN];
    assemble(arcs, narcs, G, h);
    for (int i = 0; i < n_; ++i) h[i] = p_.xf[i] - h[i];
    if (!lu_solve(G, n_, MAXN, h, 1, 1)) return false;
    for (int i = 0; i < n_; ++i) c[i] = h[i];
    return true;
}

bool Solver::detect(const double* c, uint16_t* seq, int* nseq, double* tau) const {
    uint16_t S = 0xFFFFu;
    int ns = 0;
    const double dt = p_.T / NGRID;
    double w[MAXM], v[MAXM];
    for (int g = 0; g <= NGRID; ++g) {
        const double t = g * dt;
        w_of(c, t, w);
        uint16_t Sn;
        project(w, S, v, &Sn);
        if (g == 0) {
            seq[ns++] = Sn;
        } else if (Sn != S) {
            if (ns >= MAXJ_TAU + 1) return false;
            tau[ns - 1] = t - 0.5 * dt;
            seq[ns++] = Sn;
        }
        S = Sn;
    }
    *nseq = ns;
    return true;
}

bool Solver::residual(Arc* arcs, int nseq, const double* tau, double* res, int* nres,
                      double* c) const {
    set_times(arcs, nseq, tau);
    if (!solve_c(arcs, nseq, c)) return false;
    int nr = 0;
    double v[MAXM];
    for (int k = 0; k + 1 < nseq; ++k) {
        const uint16_t L = arcs[k].S, R = arcs[k + 1].S, X = (uint16_t)(L ^ R);
        for (int key = 0; key < 2 * m_; ++key) {
            if (!(X & (1u << key))) continue;
            const Arc& side = (R & (1u << key)) ? arcs[k] : arcs[k + 1];  // key inactive here
            v_on(side, c, tau[k], v);
            double g = -rhs_[key];
            for (int j = 0; j < m_; ++j) g += rows_[key][j] * v[j];
            if (nr >= MAXRES) return false;
            res[nr++] = g;
        }
    }
    *nres = nr;
    return true;
}

bool Solver::newton(Arc* arcs, int nseq, double* tau, double* c, int* its, double* resmax) const {
    const int nj = nseq - 1;
    const double tol = 1e-12;
    const double T = p_.T;
    double r[MAXRES], rp[MAXRES], rm[MAXRES], rn[MAXRES];
    double J[MAXRES][MAXJ_TAU];
    double tn[MAXJ_TAU], tt[MAXJ_TAU], cn[MAXN], ctmp[MAXN];
    int nr, nr2;
    *its = 0;
    if (!residual(arcs, nseq, tau, r, &nr, c)) return false;
    double rmax = 0.0;
    for (int i = 0; i < nr; ++i) rmax = fmax(rmax, fabs(r[i]));
    *resmax = rmax;
    if (nj == 0) return true;
    const double gap = 1e-9 * T;
    for (int it = 1; it <= MAX_NEWTON; ++it) {
        if (rmax < tol) return true;
        *its = it;
        const double hstep = 1e-7 * fmax(1.0, T);
        for (int i = 0; i < nj; ++i) {
            for (int q = 0; q < nj; ++q) tt[q] = tau[q];
            tt[i] = tau[i] + hstep;
            if (!residual(arcs, nseq, tt, rp, &nr2, ctmp) || nr2 != nr) return false;
            tt[i] = tau[i] - hstep;
            if (!residual(arcs, nseq, tt, rm, &nr2, ctmp) || nr2 != nr) return false;
            for (int q = 0; q < nr; ++q) J[q][i] = (rp[q] - rm[q]) / (2.0 * hstep);
        }
        // Gauss-Newton step from the normal equations (J^T J) d = -J^T r
        double N[MAXJ_TAU * MAXJ_TAU], d[MAXJ_TAU];
        for (int a = 0; a < nj; ++a) {
            double s = 0.0;
            for (int q = 0; q < nr; ++q) s += J[q][a] * r[q];
            d[a] = -s;
            for (int b = 0; b < nj; ++b) {
                double g = 0.0;
                for (int q = 0; q < nr; ++q) g += J[q][a] * J[q][b];
                N[a * MAXJ_TAU + b] = g;
            }
        }
        if (!lu_solve(N, nj, MAXJ_TAU, d, 1, 1)) return false;
        double nr0 = 0.0;
        for (int q = 0; q < nr; ++q) nr0 += r[q] * r[q];
        nr0 = sqrt(nr0);
        double alpha = 1.0;
        bool accepted = false;
        while (alpha > 1e-6) {
            bool ordered = true;
            for (int a = 0; a < nj; ++a) tn[a] = tau[a] + alpha * d[a];
            double prev = 0.0;
            for (int a = 0; a <= nj; ++a) {
                const double cur = (a < nj) ? tn[a] : T;
                if (!(cur - prev > gap)) ordered = false;
                prev = cur;
            }
            if (ordered && residual(arcs, nseq, tn, rn, &nr2, cn) && nr2 == nr) {
                double nn = 0.0;
                for (int q = 0; q < nr; ++q) nn += rn[q] * rn[q];
                nn = sqrt(nn);
                if (nn < (1.0 - 1e-4 * alpha) * nr0 || nn < tol) { accepted = true; break; }
            }
            alpha *= 0.5;
        }
        if (!accepted) { set_times(arcs, nseq, tau); return false; }
        rmax = 0.0;
        for (int a = 0; a < nj; ++a) tau[a] = tn[a];
        for (int q = 0; q < nr; ++q) { r[q] = rn[q]; rmax = fmax(rmax, fabs(rn[q])); }
        for (int i = 0; i < n_; ++i) c[i] = cn[i];
        *resmax = rmax;
    }
    set_times(arcs, nseq, tau);
    return rmax < tol;
}

double Solver::pmp_violation(const Arc* arcs, int narcs, const double* c) const {
    double err = 0.0, w[MAXM], va[MAXM], vo[MAXM];
    int ai = 0;
    for (int g = 0; g <= NGRID; ++g) {
        const double t = p_.T * g / NGRID;
        while (ai < narcs - 1 && t > arcs[ai].t1) ++ai;
        v_on(arcs[ai], c, t, va);
        w_of(c, t, w);
        uint16_t S;
        project(w, arcs[ai].S, vo, &S);
        for (int j = 0; j < m_; ++j) err = fmax(err, fabs(va[j] - vo[j]));
    }
    return err;
}

double Solver::cost(const Arc* arcs, int narcs, const double* c) const {
    double J = 0.0, v[MAXM];
    for (int a = 0; a < narcs; ++a) {
        const double hw = 0.5 * (arcs[a].t1 - arcs[a].t0);
        for (int g = 0; g < 4; ++g) {
            v_on(arcs[a], c, arcs[a].t0 + hw * (GL_X[g] + 1.0), v);
            double s = 0.0;
            for (int j = 0; j < m_; ++j) s += v[j] * v[j];
            J += 0.5 * GL_W[g] * hw * s;
        }
    }
    return J;
}

bool Solver::solve(Result& R, int method) {
    R.method = method;
    R.general = general_;
    R.gnsub = 0;
    if (general_) {
        R.converged = false;
        if (method != METHOD_NEWTON) return false;     // tau: energy cost only
        return solve_general(R);
    }
    R.converged = false;
    R.backtracks = 0;
    R.degenerate_events = 0;
    R.ntouch = 0;
    R.box_fast_path = false;
    if (method == METHOD_NEWTON) return solve_newton(R);
    return solve_tau(R);
}

// Active-set sequence for the tau method: exact event walk of the newton method (grid only
// as a fallback, e.g. if the walk needs more junctions than the tau method supports).
bool Solver::detect_exact(const double* c) {
    if (walk(c, ew_.ev[0]) && ew_.ev[0].narcs <= MAXJ_TAU + 1) {
        ew_.nseq2 = ew_.ev[0].narcs;
        for (int i = 0; i < ew_.nseq2; ++i) ew_.seq2[i] = ew_.wseq[i];
        for (int i = 0; i + 1 < ew_.nseq2; ++i) ew_.tau2[i] = ew_.wtau[i];
        return true;
    }
    return detect(c, ew_.seq2, &ew_.nseq2, ew_.tau2);
}

bool Solver::solve_tau(Result& R) {
    R.iterations = 0;
    R.newton_iterations = 0;
    R.outer_iterations = 0;
    R.residual = 0.0;
    R.pmp_violation = 0.0;
    // 1) unconstrained primitive
    ew_.nseq = 1;
    ew_.seq[0] = 0;
    if (!build_arcs(ew_.seq, ew_.nseq, ew_.tau, ew_.arcs)) return false;
    if (!solve_c(ew_.arcs, ew_.nseq, R.c)) return false;
    R.iterations = 1;
    bool conv = !p_.has_bounds;
    if (p_.has_bounds) {
        double c_try[MAXN];
        double c0n = 1.0;
        for (int i = 0; i < n_; ++i) c0n = fmax(c0n, 1.0 + fabs(R.c[i]));
        for (int outer = 1; outer <= MAX_OUTER; ++outer) {
            R.outer_iterations = outer;
            // early exits: iteration cap, diverging w (dual unbounded: infeasible)
            if (R.newton_iterations > MAX_TAU_TOTAL) break;
            double cn = 0.0;
            for (int i = 0; i < n_; ++i) cn = fmax(cn, fabs(R.c[i]));
            if (cn > 1e6 * c0n) break;
            // 2) active-set sequence from the pointwise H-minimiser of the current w(t)
            if (!detect_exact(R.c)) break;
            if (outer == 1 && ew_.nseq2 == 1 && ew_.seq2[0] == 0) { conv = true; break; }
            Arc* trial = ew_.trial;
            if (!build_arcs(ew_.seq2, ew_.nseq2, ew_.tau2, trial)) break;
            // 3) junction-time Newton with inner linear BC solve
            int its = 0;
            double rmax = 0.0;
            bool ok = newton(trial, ew_.nseq2, ew_.tau2, c_try, &its, &rmax);
            R.newton_iterations += its;
            R.iterations += 1 + its;
            R.residual = rmax;
            if (ok) {
                ew_.nseq = ew_.nseq2;
                for (int i = 0; i < ew_.nseq; ++i) { ew_.seq[i] = ew_.seq2[i]; ew_.arcs[i] = trial[i]; }
                for (int i = 0; i + 1 < ew_.nseq; ++i) ew_.tau[i] = ew_.tau2[i];
                for (int i = 0; i < n_; ++i) R.c[i] = c_try[i];
                R.pmp_violation = verify(ew_.arcs, ew_.nseq, R.c);
                if (R.pmp_violation <= 1e-9 * R_) { conv = true; break; }
            } else {
                // fixed-point step: fix detected junctions, solve c, re-detect next round
                if (!detect_exact(R.c)) break;
                if (!build_arcs(ew_.seq2, ew_.nseq2, ew_.tau2, trial)) break;
                if (!solve_c(trial, ew_.nseq2, c_try)) break;
                ew_.nseq = ew_.nseq2;
                for (int i = 0; i < ew_.nseq; ++i) { ew_.seq[i] = ew_.seq2[i]; ew_.arcs[i] = trial[i]; }
                for (int i = 0; i + 1 < ew_.nseq; ++i) ew_.tau[i] = ew_.tau2[i];
                for (int i = 0; i < n_; ++i) R.c[i] = c_try[i];
            }
        }
    }
    R.narcs = ew_.nseq;
    for (int i = 0; i < ew_.nseq; ++i) R.arcs[i] = ew_.arcs[i];
    R.cost = cost(R.arcs, R.narcs, R.c);
    R.converged = conv;
    return conv;
}

void Solver::eval_u(const Result& R, double t, double u[MAXM]) const {
    if (R.general) {
        double z[GZ];
        int a;
        gz_at(R, t, z, &a);
        double wt[MAXM], vt[MAXM];
        for (int j = 0; j < m_; ++j) {
            double sum = 0.0;
            for (int i = 0; i < n_; ++i) sum -= Bt_[i][j] * z[n_ + i];
            wt[j] = sum;
        }
        for (int i = 0; i < m_; ++i) {
            double sum = R.arcs[a].q[i];
            for (int j = 0; j < m_; ++j) sum += R.arcs[a].P[i][j] * wt[j];
            vt[i] = sum;
        }
        for (int i = m_ - 1; i >= 0; --i) {            // Lc^T u = vt
            double sum = vt[i];
            for (int k = i + 1; k < m_; ++k) sum -= Lc_[k][i] * u[k];
            u[i] = sum / Lc_[i][i];
        }
        return;
    }
    if (t < 0.0) t = 0.0;
    if (t > p_.T) t = p_.T;
    int a = 0;
    while (a < R.narcs - 1 && t >= R.arcs[a].t1) ++a;
    v_on(R.arcs[a], R.c, t, u);
}

void Solver::eval_x(const Result& R, double t, double x[MAXN]) const {
    if (R.general) {
        double z[GZ];
        int a;
        gz_at(R, t, z, &a);
        for (int i = 0; i < n_; ++i) x[i] = z[i];
        return;
    }
    if (t < 0.0) t = 0.0;
    if (t > p_.T) t = p_.T;
    phi_x0(t, x);
    double K[MAXN][MAXM], v[MAXM];
    for (int a = 0; a < R.narcs; ++a) {
        const double t0 = R.arcs[a].t0;
        if (t0 >= t) break;
        const double t1 = (R.arcs[a].t1 < t) ? R.arcs[a].t1 : t;
        const double hw = 0.5 * (t1 - t0);
        for (int g = 0; g < 4; ++g) {
            const double tt = t0 + hw * (GL_X[g] + 1.0);
            v_on(R.arcs[a], R.c, tt, v);
            phib(t - tt, K);
            for (int i = 0; i < n_; ++i) {
                double s = 0.0;
                for (int j = 0; j < m_; ++j) s += K[i][j] * v[j];
                x[i] += GL_W[g] * hw * s;
            }
        }
    }
}

int Solver::switch_times(const Result& R, double* out, int maxout) const {
    int k = 0;
    for (int a = 0; a + 1 < R.narcs && k < maxout; ++a) out[k++] = R.arcs[a].t1;
    return k;
}

// =====================================================================================
// METHOD_NEWTON: Newton on c with exact (grid-free) event detection. Mirrors
// develop/python/lqocp.py (_NewtonCtx / _solve_newton).
// =====================================================================================
namespace {

const double EPS = 2.220446049250313e-16;

double pval(const double* b, int len, double s) {
    double r = 0.0;
    for (int i = len - 1; i >= 0; --i) r = r * s + b[i];
    return r;
}

// coefficients of p(t0 + s) from those of p(t) (len <= 4)
void shift(const double* a, int len, double t0, double* b) {
    for (int i = 0; i < len; ++i) {
        double s = 0.0;
        for (int p = i; p < len; ++p) s += a[p] * (fact(p) / (fact(i) * fact(p - i))) * ipow(t0, p - i);
        b[i] = s;
    }
}

double pscale(const double* b, int len, double L) {
    double s = 0.0;
    for (int i = 0; i < len; ++i) s += fabs(b[i]) * ipow(L, i);
    return s;
}

// Divide out the root at s=0 (drop leading coefficients with |b_i| L^i <= tol).
// Returns reduced length (0 if ~identically zero); k = multiplicity removed.
int strip(const double* b, int len, double L, double tol, double* red, int* kout) {
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
int roots_le2(const double* d0, int len, double* out) {
    double d[3] = {0, 0, 0};
    for (int i = 0; i < len && i < 3; ++i) d[i] = d0[i];
    while (len > 0 && d[len - 1] == 0.0) --len;
    if (len <= 1) return 0;
    if (len == 2) { out[0] = -d[0] / d[1]; return 1; }
    const double c0 = d[0], b1 = d[1], a2 = d[2];
    const double disc = b1 * b1 - 4 * a2 * c0;
    if (disc < 0) {
        if (disc > -1e-14 * (b1 * b1 + fabs(4 * a2 * c0))) { out[0] = -b1 / (2 * a2); return 1; }
        return 0;
    }
    const double sq = sqrt(disc);
    const double qq = -0.5 * (b1 + (b1 >= 0 ? sq : -sq));
    if (qq == 0.0) { out[0] = 0.0; return 1; }
    double r0 = qq / a2, r1 = c0 / qq;
    if (r0 > r1) { double t = r0; r0 = r1; r1 = t; }
    out[0] = r0; out[1] = r1;
    return 2;
}

double bisect(const double* r, int len, double a, double b) {
    double fa = pval(r, len, a);
    for (int it = 0; it < 200; ++it) {
        const double mid = 0.5 * (a + b);
        if (mid <= a || mid >= b) break;
        const double fm = pval(r, len, mid);
        if (((fm > 0) == (fa > 0)) && fm != 0.0) { a = mid; fa = fm; }
        else b = mid;
    }
    return 0.5 * (a + b);
}

struct Cross {
    int n;
    double root[4];
    bool simple[4];
    int nt;
    double touch[4];
};

// Sign changes of phi(s) = s^k red(s) on (0, L]; tangential touches recorded separately.
void crossings(const double* red, int len, int k, double L, double tol, Cross& o) {
    o.n = 0;
    o.nt = 0;
    if (len <= 1) return;
    double d[3];
    for (int i = 1; i < len; ++i) d[i - 1] = i * red[i];
    double cr[2];
    const int nc = roots_le2(d, len - 1, cr);
    double pts[4], vals[4];
    int np = 0;
    pts[np++] = 0.0;
    for (int i = 0; i < nc; ++i)
        if (cr[i] > 0.0 && cr[i] < L && (np == 1 || cr[i] != pts[np - 1])) pts[np++] = cr[i];
    pts[np++] = L;
    vals[0] = red[0];
    for (int i = 1; i < np; ++i) vals[i] = pval(red, len, pts[i]);
    int zeros[4], nz = 0;
    double sig = vals[0] > 0 ? 1.0 : -1.0;
    for (int i = 1; i < np; ++i) {
        const double v = vals[i], x = pts[i];
        if (fabs(v) * ipow(x, k) <= tol) {
            if (i < np - 1) zeros[nz++] = i;
            continue;
        }
        const double sg = v > 0 ? 1.0 : -1.0;
        if (sg != sig) {
            if (vals[i - 1] * v <= 0) {
                o.root[o.n] = bisect(red, len, pts[i - 1], x);
                o.simple[o.n] = (nz == 0);
            } else {
                o.root[o.n] = pts[zeros[nz - 1]];          // odd root at a flat point
                o.simple[o.n] = false;
            }
            ++o.n;
            for (int z = 0; z + 1 < nz; ++z) o.touch[o.nt++] = pts[zeros[z]];
            sig = sg;
        } else {
            for (int z = 0; z < nz; ++z) o.touch[o.nt++] = pts[zeros[z]];
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

void Solver::row_polys(uint16_t S, const double* c, double phi[2 * MAXM][4],
                       bool has[2 * MAXM]) const {
    double Wc[MAXM][4], P[MAXM][MAXM], q[MAXM], V[MAXM][4];
    for (int j = 0; j < m_; ++j)
        for (int p = 0; p < 4; ++p) Wc[j][p] = (p < p_.orders[j]) ? c[off_[j] + p] : 0.0;
    affine(S, P, q);
    for (int i = 0; i < m_; ++i)
        for (int p = 0; p < 4; ++p) {
            double s = (p == 0) ? q[i] : 0.0;
            for (int j = 0; j < m_; ++j) s += P[i][j] * Wc[j][p];
            V[i][p] = s;
        }
    int idx[MAXM], na = 0;
    for (int key = 0; key < 2 * m_; ++key) {
        has[key] = false;
        if (!finite_[key]) continue;
        has[key] = true;
        if (S & (1u << key)) { idx[na++] = key; continue; }
        for (int p = 0; p < 4; ++p) {
            double s = 0.0;
            for (int j = 0; j < m_; ++j) s += rows_[key][j] * V[j][p];
            phi[key][p] = s - ((p == 0) ? rhs_[key] : 0.0);
        }
    }
    if (na > 0) {
        double G[MAXM * MAXM], X[MAXM * 4];
        for (int a = 0; a < na; ++a) {
            for (int b = 0; b < na; ++b) {
                double g = 0.0;
                for (int j = 0; j < m_; ++j) g += rows_[idx[a]][j] * rows_[idx[b]][j];
                G[a * MAXM + b] = g;
            }
            for (int p = 0; p < 4; ++p) {
                double s = 0.0;
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

bool Solver::lex_ok(uint16_t S, const double* c, double t) const {
    double phi[2 * MAXM][4], b[4];
    bool has[2 * MAXM];
    row_polys(S, c, phi, has);
    const double T = p_.T;
    for (int key = 0; key < 2 * m_; ++key) {
        if (!has[key]) continue;
        shift(phi[key], 4, t, b);
        const double tol = 1e-12 * (pscale(b, 4, T) + R_);
        for (int i = 0; i < 4; ++i) {
            if (fabs(b[i]) * ipow(T, i) > tol) {
                if (b[i] > 0) return false;
                break;
            }
        }
    }
    return true;
}

bool Solver::lex_set(const double* c, double t, uint16_t guess, uint16_t* out) const {
    if (guess != 0xFFFFu && valid_set(guess) && lex_ok(guess, c, t)) { *out = guess; return true; }
    for (int i = 0; i < nsets_; ++i)
        if (lex_ok(sets_[i], c, t)) { *out = sets_[i]; return true; }
    return false;
}

bool Solver::walk(const double* c, Eval& e) {
    e.ndeg = 0;
    e.ntouch = 0;
    if (identity_) return walk_box(c, e);
    const double T = p_.T;
    uint16_t S;
    if (!lex_set(c, 0.0, 0, &S)) return false;      // (a) initial set: one-sided KKT at 0+
    int ns = 0, stall = 0;
    ew_.wseq[ns++] = S;
    double t = 0.0;
    double phi[2 * MAXM][4];
    bool has[2 * MAXM];
    for (;;) {
        const double L = T - t;
        row_polys(S, c, phi, has);
        double rootk[2 * MAXM];
        bool simplek[2 * MAXM], evk[2 * MAXM];
        double s1 = HUGE_VAL;
        double tch[2 * MAXM * 4];
        int ntch = 0;
        for (int key = 0; key < 2 * m_; ++key) {
            evk[key] = false;
            if (!has[key]) continue;
            double b[4], red[4];
            int k;
            shift(phi[key], 4, t, b);
            const double tol = 1e-12 * (pscale(b, 4, L) + R_);
            const int rl = strip(b, 4, L, tol, red, &k);
            if (rl == 0) continue;
            if (red[0] > 0) {                            // inconsistent at t+: degenerate event
                evk[key] = true; rootk[key] = 0.0; simplek[key] = false;
                s1 = 0.0;
                continue;
            }
            Cross cr;
            crossings(red, rl, k, L, tol, cr);
            const double lim = cr.n ? cr.root[0] : HUGE_VAL;
            for (int z = 0; z < cr.nt; ++z)
                if (cr.touch[z] < lim) tch[ntch++] = cr.touch[z];
            if (cr.n) {
                evk[key] = true; rootk[key] = cr.root[0]; simplek[key] = cr.simple[0];
                if (cr.root[0] < s1) s1 = cr.root[0];
            }
        }
        for (int z = 0; z < ntch; ++z)
            if (tch[z] < s1 && e.ntouch < MAXTOUCH) e.touch[e.ntouch++] = t + tch[z];
        if (s1 == HUGE_VAL || t + s1 >= T * (1 - 1e-14)) break;
        int ngroup = 0, gkey = -1;
        bool gsimple = false;
        uint16_t gmask = 0;
        for (int key = 0; key < 2 * m_; ++key)
            if (evk[key] && rootk[key] <= s1 + 1e-12 * T) {
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
            if (!lex_set(c, t, valid_set(guess) ? guess : 0xFFFFu, &Snew)) return false;
        }
        if (Snew != S) {
            if (ns >= MAXARC_E) return false;
            ew_.wtau[ns - 1] = t;
            ew_.wseq[ns++] = Snew;
            S = Snew;
            stall = 0;
        } else if (++stall > 3) {
            return false;
        }
    }
    e.narcs = ns;
    return build_arcs(ew_.wseq, ns, ew_.wtau, e.arcs);
}

bool Solver::walk_box(const double* c, Eval& e) {
    // Umap = identity: v_j = clip(w_j, lo_j, hi_j) per chain, events = roots of w_j - bound.
    const double T = p_.T;
    double times[MAXARC_E];
    int nt = 0;
    for (int j = 0; j < m_; ++j) {
        for (int side = 0; side < 2; ++side) {
            const int key = 2 * j + side;
            if (!finite_[key]) continue;
            const double sgn = side ? -1.0 : 1.0;
            double b[4], red[4];
            for (int p = 0; p < 4; ++p) b[p] = (p < p_.orders[j]) ? sgn * c[off_[j] + p] : 0.0;
            b[0] -= rhs_[key];                       // rhs_ = hi (upper) or -lo (lower)
            const double tol = 1e-12 * (pscale(b, 4, T) + R_);
            int k;
            const int rl = strip(b, 4, T, tol, red, &k);
            if (rl == 0) continue;
            Cross cr;
            crossings(red, rl, k, T, tol, cr);
            for (int i = 0; i < cr.n; ++i) {
                if (cr.root[i] >= T * (1 - 1e-14)) continue;
                bool dup = false;
                for (int q = 0; q < nt; ++q) if (times[q] == cr.root[i]) dup = true;
                if (!dup) { if (nt >= MAXARC_E) return false; times[nt++] = cr.root[i]; }
            }
            for (int z = 0; z < cr.nt; ++z)
                if (e.ntouch < MAXTOUCH) e.touch[e.ntouch++] = cr.touch[z];
        }
    }
    for (int i = 1; i < nt; ++i) {                   // insertion sort
        double x = times[i];
        int q = i - 1;
        while (q >= 0 && times[q] > x) { times[q + 1] = times[q]; --q; }
        times[q + 1] = x;
    }
    int ns = 0;
    double w[MAXM];
    for (int i = 0; i <= nt; ++i) {
        const double a = (i == 0) ? 0.0 : times[i - 1], b = (i == nt) ? T : times[i];
        w_of(c, 0.5 * (a + b), w);                   // no root inside: any interior point
        uint16_t S = 0;
        for (int j = 0; j < m_; ++j) {
            if (finite_[2 * j] && w[j] > p_.hi[j]) S |= (uint16_t)(1u << (2 * j));
            else if (finite_[2 * j + 1] && w[j] < p_.lo[j]) S |= (uint16_t)(1u << (2 * j + 1));
        }
        if (ns == 0 || S != ew_.wseq[ns - 1]) {
            if (ns >= MAXARC_E) return false;
            if (ns > 0) ew_.wtau[ns - 1] = a;
            ew_.wseq[ns++] = S;
        }
    }
    e.narcs = ns;
    return build_arcs(ew_.wseq, ns, ew_.wtau, e.arcs);
}

double Solver::dual(const Arc* arcs, int narcs, const double* c) const {
    double val = 0.0;
    for (int i = 0; i < n_; ++i) {
        double nu = 0.0;
        for (int j = 0; j < n_; ++j) nu += Linv_[i * MAXN + j] * c[j];
        val += nu * (p_.xf[i] - free_[i]);
    }
    double w[MAXM], v[MAXM];
    for (int a = 0; a < narcs; ++a) {
        const double hw = 0.5 * (arcs[a].t1 - arcs[a].t0);
        for (int g = 0; g < 4; ++g) {
            const double t = arcs[a].t0 + hw * (GL_X[g] + 1.0);
            w_of(c, t, w);
            v_on(arcs[a], c, t, v);
            double s = 0.0;
            for (int j = 0; j < m_; ++j) s += 0.5 * v[j] * v[j] - w[j] * v[j];
            val += GL_W[g] * hw * s;
        }
    }
    return val;
}

bool Solver::evaluate(const double* c, Eval& e) {
    if (!walk(c, e)) return false;
    double h[MAXN];
    assemble(e.arcs, e.narcs, e.G, h);
    double sc = 1.0;
    for (int i = 0; i < n_; ++i) {
        double gc = 0.0, s = 0.0;
        for (int j = 0; j < n_; ++j) { gc += e.G[i * MAXN + j] * c[j]; s += fabs(e.G[i * MAXN + j] * c[j]); }
        e.F[i] = gc + h[i] - p_.xf[i];
        sc = fmax(sc, fabs(p_.xf[i]));
        sc = fmax(sc, fabs(free_[i]));
        sc = fmax(sc, s + fabs(h[i] - free_[i]));
    }
    e.scale = sc;
    e.D = dual(e.arcs, e.narcs, c);
    return true;
}

double Solver::verify(const Arc* arcs, int narcs, const double* c) const {
    double worst = 0.0, phi[2 * MAXM][4], b[4], d[3], cr[2];
    bool has[2 * MAXM];
    for (int a = 0; a < narcs; ++a) {
        const double L = arcs[a].t1 - arcs[a].t0;
        row_polys(arcs[a].S, c, phi, has);
        for (int key = 0; key < 2 * m_; ++key) {
            if (!has[key]) continue;
            shift(phi[key], 4, arcs[a].t0, b);
            worst = fmax(worst, fmax(b[0], pval(b, 4, L)));
            for (int i = 1; i < 4; ++i) d[i - 1] = i * b[i];
            const int nc = roots_le2(d, 3, cr);
            for (int i = 0; i < nc; ++i)
                if (cr[i] > 0.0 && cr[i] < L) worst = fmax(worst, pval(b, 4, cr[i]));
        }
    }
    return worst;
}

bool Solver::solve_newton(Result& R) {
    R.iterations = R.newton_iterations = R.outer_iterations = 0;
    R.residual = HUGE_VAL;
    R.pmp_violation = HUGE_VAL;
    R.box_fast_path = identity_;
    R.narcs = 0;
    // unconstrained start
    ew_.nseq = 1;
    ew_.seq[0] = 0;
    if (!build_arcs(ew_.seq, 1, ew_.tau, ew_.arcs)) return false;
    double c[MAXN], cn[MAXN];
    if (!solve_c(ew_.arcs, 1, c)) return false;
    double c_start[MAXN];
    for (int i = 0; i < n_; ++i) c_start[i] = c[i];
    int cur = 0;
    bool ok = evaluate(c, ew_.ev[cur]);
    bool conv = false;
    const double rtol = 1e-12;
    int its = 0;
    while (ok) {
        Eval& E = ew_.ev[cur];
        double Fn = 0.0, F2 = 0.0;
        for (int i = 0; i < n_; ++i) { Fn = fmax(Fn, fabs(E.F[i])); F2 += E.F[i] * E.F[i]; }
        if (Fn <= rtol * E.scale) { conv = true; break; }
        if (its >= MAX_NEWTON) break;
        {   // dual unbounded (infeasible problem): the iterates c diverge
            double cn = 0.0, c0 = 1.0;
            for (int i = 0; i < n_; ++i) { cn = fmax(cn, fabs(c[i])); c0 = fmax(c0, 1.0 + fabs(c_start[i])); }
            if (cn > 1e6 * c0) break;
        }
        ++its;
        // (J Lm) d_nu = -F, J Lm = int K P K^T dt symmetric PSD; d_c = Lm d_nu
        double H[MAXN * MAXN], dnu[MAXN], d[MAXN];
        for (int i = 0; i < n_; ++i)
            for (int j = 0; j < n_; ++j) {
                double s = 0.0;
                for (int k = 0; k < n_; ++k) s += E.G[i * MAXN + k] * Lm_[k * MAXN + j];
                H[i * MAXN + j] = s;
            }
        for (int i = 0; i < n_; ++i)
            for (int j = i + 1; j < n_; ++j) {
                const double s = 0.5 * (H[i * MAXN + j] + H[j * MAXN + i]);
                H[i * MAXN + j] = H[j * MAXN + i] = s;
            }
        double hmax = 0.0;
        for (int i = 0; i < n_ * MAXN; ++i) hmax = fmax(hmax, fabs(H[i]));
        double rho = 0.0;
        bool solved = false;
        for (int attempt = 0; attempt < 20 && !solved; ++attempt) {
            double A[MAXN * MAXN];
            for (int i = 0; i < MAXN * MAXN; ++i) A[i] = H[i];
            for (int i = 0; i < n_; ++i) { A[i * MAXN + i] += rho; dnu[i] = -E.F[i]; }
            solved = lu_solve(A, n_, MAXN, dnu, 1, 1);
            rho = (rho == 0.0) ? 1e-12 * hmax : 10.0 * rho;
        }
        if (!solved) break;
        double slope = 0.0;
        for (int i = 0; i < n_; ++i) {
            double s = 0.0;
            for (int j = 0; j < n_; ++j) s += Lm_[i * MAXN + j] * dnu[j];
            d[i] = s;
            slope -= E.F[i] * dnu[i];
        }
        double alpha = 1.0;
        bool accepted = false;
        const int nxt = 1 - cur;
        while (alpha >= 1e-10) {
            for (int i = 0; i < n_; ++i) cn[i] = c[i] + alpha * d[i];
            if (evaluate(cn, ew_.ev[nxt])) {
                const Eval& N = ew_.ev[nxt];
                if (N.D >= E.D + 1e-4 * alpha * slope) { accepted = true; break; }
                double nF2 = 0.0;
                for (int i = 0; i < n_; ++i) nF2 += N.F[i] * N.F[i];
                if (alpha * slope <= 1e3 * EPS * (1.0 + fabs(E.D)) && nF2 < F2) { accepted = true; break; }
            }
            alpha *= 0.5;
            ++R.backtracks;
        }
        if (!accepted) break;
        for (int i = 0; i < n_; ++i) c[i] = cn[i];
        cur = nxt;
    }
    R.iterations = R.newton_iterations = its;
    if (!ok) return false;
    const Eval& E = ew_.ev[cur];
    R.narcs = E.narcs;
    for (int i = 0; i < E.narcs; ++i) R.arcs[i] = E.arcs[i];
    for (int i = 0; i < n_; ++i) R.c[i] = c[i];
    double Fn = 0.0;
    for (int i = 0; i < n_; ++i) Fn = fmax(Fn, fabs(E.F[i]));
    R.residual = Fn;
    R.degenerate_events = E.ndeg;
    R.ntouch = E.ntouch;
    for (int i = 0; i < E.ntouch; ++i) R.touch[i] = E.touch[i];
    R.pmp_violation = verify(R.arcs, R.narcs, R.c);
    if (R.pmp_violation > 1e-9 * R_) conv = false;
    R.cost = cost(R.arcs, R.narcs, R.c);
    R.converged = conv;
    return conv;
}

// =====================================================================================
// General LQ cost (Revision 3). Mirrors develop/python/lqocp.py (_GenCtx / _solve_general).
// R-scaled inputs vt = Lc^T v (R = Lc Lc^T), augmented Hamiltonian state z = (s, lambda, 1).
// =====================================================================================
namespace {

const double GL10_X[10] = {-0.97390652851717172008, -0.86506336668898451073,
                           -0.67940956829902440623, -0.43339539412924719080,
                           -0.14887433898163121088, 0.14887433898163121088,
                           0.43339539412924719080, 0.67940956829902440623,
                           0.86506336668898451073, 0.97390652851717172008};
const double GL10_W[10] = {0.066671344308688069, 0.14945134915058036, 0.21908636251598201, 0.26926671930999652, 0.29552422471475298, 0.29552422471475298, 0.26926671930999652, 0.21908636251598201, 0.14945134915058036, 0.066671344308688069};
const double SUB_TOL = 1e-16;

double g_binom[GN][GN];
bool g_binom_ready = false;

void init_binom() {
    if (g_binom_ready) return;
    for (int k = 0; k < GN; ++k)
        for (int j = 0; j < GN; ++j)
            g_binom[k][j] = (j == 0 || j == k) ? 1.0
                            : (j > k ? 0.0 : g_binom[k - 1][j - 1] + g_binom[k - 1][j]);
    g_binom_ready = true;
}

// coefficients of p(a + w x) from those of p(sigma)
void tshift(const double* c, double a, double w, double* q) {
    double ap[GN];
    ap[0] = 1.0;
    for (int i = 1; i < GN; ++i) ap[i] = ap[i - 1] * a;
    double wp = 1.0;
    for (int j = 0; j < GN; ++j) {
        double s = 0.0;
        for (int k = j; k < GN; ++k) s += c[k] * g_binom[k][j] * ap[k - j];
        q[j] = s * wp;
        wp *= w;
    }
}

double bern_max(const double* q, bool* mono) {
    const int n = GN - 1;
    double mx = -HUGE_VAL, prev = -HUGE_VAL;
    bool mon = true;
    for (int i = 0; i <= n; ++i) {
        double s = 0.0;
        for (int j = 0; j <= i; ++j) s += g_binom[i][j] / g_binom[n][j] * q[j];
        if (s > mx) mx = s;
        if (i > 0 && s < prev) mon = false;
        prev = s;
    }
    if (mono) *mono = mon;
    return mx;
}

// First up-crossing of p on (a, h] that exceeds tol (Bernstein enclosures + remainder eps).
bool first_exceed(const double* c, double a, double h, double tol, double eps, double minw,
                  double* root, bool* simple) {
    double stlo[64], sthi[64], q[GN];
    int sp = 0;
    stlo[sp] = a; sthi[sp] = h; ++sp;
    double last_nonpos = a;
    // event threshold tol + eps: pruned intervals satisfy phi <= tol + 2 eps (certified),
    // an event needs the polynomial above tol + eps
    tol += eps;
    while (sp > 0) {
        --sp;
        const double lo = stlo[sp], hi = sthi[sp];
        tshift(c, lo, hi - lo, q);
        bool mono;
        const double bmax = bern_max(q, &mono);
        if (bmax <= tol) {
            if (pval(c, GN, hi) <= 0) last_nonpos = hi;
            continue;
        }
        if (mono || hi - lo <= minw || sp >= 62) {
            const double phi = pval(c, GN, hi);
            if (phi > tol) {
                if (pval(c, GN, last_nonpos) <= 0) {
                    *root = bisect(c, GN, last_nonpos, hi);
                    *simple = mono && last_nonpos == lo;
                } else {
                    *root = last_nonpos;
                    *simple = false;
                }
                return true;
            }
            if (phi <= 0) last_nonpos = hi;
            continue;
        }
        const double mid = 0.5 * (lo + hi);
        stlo[sp] = mid; sthi[sp] = hi; ++sp;
        stlo[sp] = lo; sthi[sp] = mid; ++sp;
    }
    return false;
}

}  // namespace

bool Solver::gsetup(double Mrow[MAXM][MAXM]) {
    init_binom();
    // Cholesky R = Lc Lc^T (symmetrised)
    for (int i = 0; i < m_; ++i)
        for (int j = 0; j < m_; ++j) Lc_[i][j] = 0.0;
    for (int j = 0; j < m_; ++j) {
        double d = 0.5 * (p_.R[j][j] + p_.R[j][j]);
        for (int k = 0; k < j; ++k) d -= Lc_[j][k] * Lc_[j][k];
        if (!(d > 0.0)) return false;
        Lc_[j][j] = sqrt(d);
        for (int i = j + 1; i < m_; ++i) {
            double s = 0.5 * (p_.R[i][j] + p_.R[j][i]);
            for (int k = 0; k < j; ++k) s -= Lc_[i][k] * Lc_[j][k];
            Lc_[i][j] = s / Lc_[j][j];
        }
    }
    // LiT = Lc^{-T} (upper triangular): solve Lc^T X = I
    double LiT[MAXM][MAXM];
    for (int c = 0; c < m_; ++c)
        for (int i = m_ - 1; i >= 0; --i) {
            double s = (i == c) ? 1.0 : 0.0;
            for (int k = i + 1; k < m_; ++k) s -= Lc_[k][i] * LiT[k][c];
            LiT[i][c] = s / Lc_[i][i];
        }
    for (int i = 0; i < n_; ++i)
        for (int j = 0; j < m_; ++j) Bt_[i][j] = 0.0;
    for (int j = 0; j < m_; ++j) {
        const int e = off_[j] + p_.orders[j] - 1;   // B has a 1 at the end of chain j
        for (int k = 0; k < m_; ++k) Bt_[e][k] = LiT[j][k];
    }
    double Mt[MAXM][MAXM];
    for (int r = 0; r < m_; ++r)
        for (int k = 0; k < m_; ++k) {
            double s = 0.0;
            for (int j = 0; j < m_; ++j) s += Mrow[r][j] * LiT[j][k];
            Mt[r][k] = s;
        }
    for (int r = 0; r < m_; ++r)
        for (int k = 0; k < m_; ++k) Mrow[r][k] = Mt[r][k];
    // symmetrise Q, Qf in place
    for (int i = 0; i < n_; ++i)
        for (int j = i + 1; j < n_; ++j) {
            double a = 0.5 * (p_.Q[i][j] + p_.Q[j][i]);
            p_.Q[i][j] = p_.Q[j][i] = a;
            double b = 0.5 * (p_.Qf[i][j] + p_.Qf[j][i]);
            p_.Qf[i][j] = p_.Qf[j][i] = b;
        }
    return true;
}

void Solver::hmul(const GSet& g, const double* z, double* out) const {
    const int n = n_;
    const double* s = z;
    const double* lam = z + n;
    double wt[MAXM], vt[MAXM];
    for (int j = 0; j < m_; ++j) {
        double a = 0.0;
        for (int i = 0; i < n; ++i) a -= Bt_[i][j] * lam[i];
        wt[j] = a;
    }
    for (int i = 0; i < m_; ++i) {
        double a = g.q[i] * z[2 * n];
        for (int j = 0; j < m_; ++j) a += g.P[i][j] * wt[j];
        vt[i] = a;
    }
    for (int i = 0; i < n; ++i) {
        double a = 0.0;
        for (int k = 0; k < m_; ++k) a += Bt_[i][k] * vt[k];
        out[i] = a;
        double b = 0.0;
        for (int j = 0; j < n; ++j) b -= p_.Q[i][j] * s[j];
        out[n + i] = b;
    }
    for (int j = 0; j < m_; ++j)
        for (int r = 0; r + 1 < p_.orders[j]; ++r) {
            const int i = off_[j] + r;
            out[i] += s[i + 1];
            out[n + i + 1] -= lam[i];
        }
    out[2 * n] = 0.0;
}

int Solver::gset(uint16_t S) {
    for (int i = 0; i < gw_.nsets; ++i)
        if (gw_.sets[i].S == S) return i;
    if (gw_.nsets >= MAXGSET) gw_.nsets = 0;      // cache full: restart (cheap to rebuild)
    GSet& g = gw_.sets[gw_.nsets];
    g.S = S;
    if (!affine(S, g.P, g.q)) return -1;
    // ||H^GN||_inf, column by column with matrix-free products (no dense powers stored)
    double rowsum[GZ], a[GZ], b[GZ];
    for (int i = 0; i < GZ; ++i) rowsum[i] = 0.0;
    for (int c = 0; c < GZ; ++c) {
        for (int i = 0; i < GZ; ++i) a[i] = (i == c) ? 1.0 : 0.0;
        for (int k = 0; k < GN; ++k) {
            hmul(g, a, b);
            for (int i = 0; i < GZ; ++i) a[i] = b[i];
        }
        for (int i = 0; i < GZ; ++i) rowsum[i] += fabs(a[i]);
    }
    double nrm = 0.0;
    for (int i = 0; i < GZ; ++i) nrm = fmax(nrm, rowsum[i]);
    g.nrmN = nrm;
    g.h = (nrm == 0.0) ? p_.T : fmin(p_.T, pow(fact(GN) * SUB_TOL / nrm, 1.0 / GN));
    return gw_.nsets++;
}

void Solver::taylor(const GSet& g, const double* z0, double Z[GZ][GN]) const {
    double cur[GZ], nxt[GZ];
    for (int i = 0; i < GZ; ++i) { cur[i] = z0[i]; Z[i][0] = z0[i]; }
    for (int k = 1; k < GN; ++k) {
        hmul(g, cur, nxt);
        for (int i = 0; i < GZ; ++i) { cur[i] = nxt[i] / k; Z[i][k] = cur[i]; }
    }
}

double Solver::eps_sub(const GSet& g, double h, double Z[GZ][GN]) const {
    const double c = g.nrmN * ipow(h, GN) / fact(GN);
    double Zb = 0.0, hp = 1.0;
    for (int k = 0; k < GN; ++k) {
        double s = 0.0;
        for (int i = 0; i < GZ; ++i) s += fabs(Z[i][k]);
        Zb += s * hp;
        hp *= h;
    }
    double bmax = 0.0;
    for (int j = 0; j < m_; ++j) {
        double s = 0.0;
        for (int i = 0; i < n_; ++i) s += fabs(Bt_[i][j]);
        bmax = fmax(bmax, s);
    }
    return c * Zb / fmax(1e-300, 1.0 - c) * bmax;
}

int Solver::gphi(uint16_t S, int keys[2 * MAXM], double G[2 * MAXM][MAXM],
                 double g0[2 * MAXM]) const {
    double P[MAXM][MAXM], q[MAXM];
    affine(S, P, q);
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
    int idx[MAXM], na = 0;
    for (int key = 0; key < 2 * m_; ++key)
        if (S & (1u << key)) idx[na++] = key;
    if (na > 0) {
        double A[MAXM * MAXM], X[MAXM * (MAXM + 1)];
        const int ldx = MAXM + 1;
        for (int a = 0; a < na; ++a) {
            for (int b = 0; b < na; ++b) {
                double s = 0.0;
                for (int j = 0; j < m_; ++j) s += rows_[idx[a]][j] * rows_[idx[b]][j];
                A[a * MAXM + b] = s;
            }
            for (int j = 0; j < m_; ++j) X[a * ldx + j] = rows_[idx[a]][j];
            X[a * ldx + m_] = rhs_[idx[a]];
        }
        if (!lu_solve(A, na, MAXM, X, m_ + 1, ldx)) return nk;
        for (int a = 0; a < na; ++a) {
            keys[nk] = idx[a];
            for (int j = 0; j < m_; ++j) G[nk][j] = -X[a * ldx + j] * rownorm2_[idx[a]];
            g0[nk++] = X[a * ldx + m_] * rownorm2_[idx[a]];
        }
    }
    return nk;
}

bool Solver::glex_ok(uint16_t S, const double W[MAXM][GN], double sig, double h) const {
    int keys[2 * MAXM];
    double G[2 * MAXM][MAXM], g0[2 * MAXM], c[GN], cs[GN];
    const int nk = gphi(S, keys, G, g0);
    for (int r = 0; r < nk; ++r) {
        for (int k = 0; k < GN; ++k) {
            double a = (k == 0) ? g0[r] : 0.0;
            for (int j = 0; j < m_; ++j) a += G[r][j] * W[j][k];
            c[k] = a;
        }
        tshift(c, sig, 1.0, cs);
        double tol = 0.0, hp = 1.0;
        for (int k = 0; k < GN; ++k) { tol += fabs(cs[k]) * hp; hp *= h; }
        tol = 1e-12 * (tol + R_);
        hp = 1.0;
        for (int k = 0; k < GN; ++k) {
            if (fabs(cs[k]) * hp > tol) {
                if (cs[k] > 0) return false;
                break;
            }
            hp *= h;
        }
    }
    return true;
}

bool Solver::glex_set(const double W[MAXM][GN], double sig, double h, uint16_t guess,
                      uint16_t* out) const {
    if (guess != 0xFFFFu && valid_set(guess) && glex_ok(guess, W, sig, h)) { *out = guess; return true; }
    for (int i = 0; i < nsets_; ++i)
        if (glex_ok(sets_[i], W, sig, h)) { *out = sets_[i]; return true; }
    return false;
}

void Solver::gz_at(const Result& R, double t, double z[GZ], int* arc) const {
    if (t < 0.0) t = 0.0;
    if (t > p_.T) t = p_.T;
    int a = 0, gi = 0;
    while (a < R.narcs - 1 && t >= R.arcs[a].t1) { gi += R.nsub[a]; ++a; }
    const double h = (R.arcs[a].t1 - R.arcs[a].t0) / R.nsub[a];
    int j = (int)((t - R.arcs[a].t0) / h);
    if (j >= R.nsub[a]) j = R.nsub[a] - 1;
    if (j < 0) j = 0;
    const double sig = t - (R.arcs[a].t0 + j * h);
    GSet g;
    g.S = R.arcs[a].S;
    for (int i = 0; i < m_; ++i) {
        g.q[i] = R.arcs[a].q[i];
        for (int k = 0; k < m_; ++k) g.P[i][k] = R.arcs[a].P[i][k];
    }
    double cur[GZ], nxt[GZ];
    for (int i = 0; i < 2 * n_; ++i) cur[i] = R.gz0[gi + j][i];
    cur[2 * n_] = 1.0;
    for (int i = 0; i < GZ; ++i) z[i] = cur[i];
    double sp = 1.0;
    for (int k = 1; k < GN; ++k) {
        hmul(g, cur, nxt);
        sp *= sig;
        for (int i = 0; i < GZ; ++i) { cur[i] = nxt[i] / k; z[i] += cur[i] * sp; }
    }
    *arc = a;
}

bool Solver::gwalk(const Result& R, int* ndeg) {
    const double T = p_.T;
    *ndeg = 0;
    int ns = 0;
    if (!p_.has_bounds) {
        gw_.seq[0] = 0;
        gw_.nseq = 1;
        return true;
    }
    uint16_t S = 0xFFFFu;
    double (&Z)[GZ][GN] = gw_.Z;
    double W[MAXM][GN];
    int gi = 0;
    for (int a = 0; a < R.narcs; ++a) {
        const int si = gset(R.arcs[a].S);
        if (si < 0) return false;
        const GSet& g = gw_.sets[si];
        const double h = (R.arcs[a].t1 - R.arcs[a].t0) / R.nsub[a];
        for (int j = 0; j < R.nsub[a]; ++j, ++gi) {
            const double t0 = R.arcs[a].t0 + j * h;
            double z0[GZ];
            for (int i = 0; i < 2 * n_; ++i) z0[i] = R.gz0[gi][i];
            z0[2 * n_] = 1.0;
            taylor(g, z0, Z);
            for (int jj = 0; jj < m_; ++jj)
                for (int k = 0; k < GN; ++k) {
                    double s = 0.0;
                    for (int i = 0; i < n_; ++i) s -= Bt_[i][jj] * Z[n_ + i][k];
                    W[jj][k] = s;
                }
            const double eps0 = eps_sub(g, h, Z);
            if (S == 0xFFFFu) {
                if (!glex_set(W, 0.0, h, 0, &S)) return false;    // (a) initial set
                gw_.seq[ns++] = S;
            }
            double sig = 0.0;
            int stall = 0;
            for (;;) {
                int keys[2 * MAXM];
                double G[2 * MAXM][MAXM], g0[2 * MAXM];
                const int nk = gphi(S, keys, G, g0);
                if (nk == 0) break;
                double hp[GN];
                hp[0] = 1.0;
                for (int k = 1; k < GN; ++k) hp[k] = hp[k - 1] * h;
                double rootk[2 * MAXM];
                bool simplek[2 * MAXM];
                int keyk[2 * MAXM], nc = 0;
                double s1 = HUGE_VAL;
                for (int r = 0; r < nk; ++r) {
                    double c[GN], cs[GN];
                    double gn = 0.0;
                    for (int jj = 0; jj < m_; ++jj) gn += fabs(G[r][jj]);
                    for (int k = 0; k < GN; ++k) {
                        double v = (k == 0) ? g0[r] : 0.0;
                        for (int jj = 0; jj < m_; ++jj) v += G[r][jj] * W[jj][k];
                        c[k] = v;
                    }
                    tshift(c, sig, 1.0, cs);
                    double tol = 0.0;
                    for (int k = 0; k < GN; ++k) tol += fabs(cs[k]) * hp[k];
                    tol = 1e-12 * (tol + R_);
                    double lead = 0.0;
                    int ilead = 0;
                    for (int k = 0; k < GN; ++k)
                        if (fabs(cs[k]) * hp[k] > tol) { lead = cs[k]; ilead = k; break; }
                    if (lead > 0) {
                        rootk[nc] = sig; simplek[nc] = (ilead == 1); keyk[nc++] = keys[r];
                        if (sig < s1) s1 = sig;
                        continue;
                    }
                    double root;
                    bool simple;
                    if (first_exceed(c, sig, h, tol, eps0 * gn, 1e-14 * T, &root, &simple)) {
                        rootk[nc] = root; simplek[nc] = simple; keyk[nc++] = keys[r];
                        if (root < s1) s1 = root;
                    }
                }
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
                    if (!glex_set(W, s1, h, valid_set(guess) ? guess : 0xFFFFu, &Snew)) return false;
                }
                if (Snew != S) {
                    if (ns >= MAXARC) return false;
                    gw_.tau[ns - 1] = t0 + s1;
                    gw_.seq[ns++] = Snew;
                    S = Snew;
                    stall = 0;
                } else if (++stall > 3) {
                    return false;
                }
                sig = s1;
            }
        }
    }
    gw_.nseq = ns;
    return true;
}

double Solver::gverify(const Result& R) {
    double (&Z)[GZ][GN] = gw_.Z;
    double W[MAXM][GN], worst = 0.0;
    int gi = 0;
    for (int a = 0; a < R.narcs; ++a) {
        const int si = gset(R.arcs[a].S);
        if (si < 0) return HUGE_VAL;
        const GSet& g = gw_.sets[si];
        const double h = (R.arcs[a].t1 - R.arcs[a].t0) / R.nsub[a];
        int keys[2 * MAXM];
        double G[2 * MAXM][MAXM], g0[2 * MAXM];
        const int nk = gphi(R.arcs[a].S, keys, G, g0);
        for (int j = 0; j < R.nsub[a]; ++j, ++gi) {
            if (nk == 0) continue;
            double z0[GZ];
            for (int i = 0; i < 2 * n_; ++i) z0[i] = R.gz0[gi][i];
            z0[2 * n_] = 1.0;
            taylor(g, z0, Z);
            for (int jj = 0; jj < m_; ++jj)
                for (int k = 0; k < GN; ++k) {
                    double s = 0.0;
                    for (int i = 0; i < n_; ++i) s -= Bt_[i][jj] * Z[n_ + i][k];
                    W[jj][k] = s;
                }
            const double eps0 = eps_sub(g, h, Z);
            for (int r = 0; r < nk; ++r) {
                double c[GN], q[GN], gn = 0.0;
                for (int jj = 0; jj < m_; ++jj) gn += fabs(G[r][jj]);
                for (int k = 0; k < GN; ++k) {
                    double v = (k == 0) ? g0[r] : 0.0;
                    for (int jj = 0; jj < m_; ++jj) v += G[r][jj] * W[jj][k];
                    c[k] = v;
                }
                tshift(c, 0.0, h, q);
                worst = fmax(worst, bern_max(q, nullptr) + eps0 * gn);
            }
        }
    }
    return worst;
}

double Solver::gcost(const Result& R, const double* zT) {
    double J = 0.0;
    int gi = 0;
    double (&Z)[GZ][GN] = gw_.Z;
    for (int a = 0; a < R.narcs; ++a) {
        const int si = gset(R.arcs[a].S);
        if (si < 0) return HUGE_VAL;
        const GSet& g = gw_.sets[si];
        const double h = (R.arcs[a].t1 - R.arcs[a].t0) / R.nsub[a];
        for (int j = 0; j < R.nsub[a]; ++j, ++gi) {
            double z0[GZ];
            for (int i = 0; i < 2 * n_; ++i) z0[i] = R.gz0[gi][i];
            z0[2 * n_] = 1.0;
            taylor(g, z0, Z);
            for (int q = 0; q < 10; ++q) {
                const double sig = 0.5 * h * (GL10_X[q] + 1.0);
                double z[GZ];
                for (int i = 0; i < GZ; ++i) z[i] = pval(Z[i], GN, sig);
                double wt[MAXM], val = 0.0;
                for (int jj = 0; jj < m_; ++jj) {
                    double s = 0.0;
                    for (int i = 0; i < n_; ++i) s -= Bt_[i][jj] * z[n_ + i];
                    wt[jj] = s;
                }
                for (int i = 0; i < m_; ++i) {
                    double vt = g.q[i];
                    for (int k = 0; k < m_; ++k) vt += g.P[i][k] * wt[k];
                    val += vt * vt;
                }
                for (int i = 0; i < n_; ++i) {
                    double qs = 0.0;
                    for (int k = 0; k < n_; ++k) qs += p_.Q[i][k] * z[k];
                    val += z[i] * qs;
                }
                J += 0.25 * h * GL10_W[q] * val;
            }
        }
    }
    if (p_.has_Qf) {
        for (int i = 0; i < n_; ++i)
            for (int k = 0; k < n_; ++k)
                J += 0.5 * (zT[i] - p_.xf[i]) * p_.Qf[i][k] * (zT[k] - p_.xf[k]);
    }
    return J;
}

void Solver::gscale(double kap) {
    for (int k = 0; k < 2 * m_; ++k) rhs_[k] = rhs0_[k] * kap;
    gw_.nsets = 0;                            // Taylor set data (q, H) depend on the bounds
    for (int i = 0; i < gw_.nsb; ++i) gw_.sb[i].part_ok = 0;   // bases W, T do not
    ++gw_.gen;                                // invalidates cached arc exponentials
}

double Solver::gmax_ratio(const Result& R) {
    double r = 0.0;
    int gi = 0;
    double (&Z)[GZ][GN] = gw_.Z;
    for (int a = 0; a < R.narcs; ++a) {
        const int si = gset(R.arcs[a].S);
        if (si < 0) return HUGE_VAL;
        const GSet& g = gw_.sets[si];
        const double h = (R.arcs[a].t1 - R.arcs[a].t0) / R.nsub[a];
        for (int j = 0; j < R.nsub[a]; ++j, ++gi) {
            double z0[GZ];
            for (int i = 0; i < 2 * n_; ++i) z0[i] = R.gz0[gi][i];
            z0[2 * n_] = 1.0;
            taylor(g, z0, Z);
            for (int q = 0; q < 10; ++q) {
                const double sig = 0.5 * h * (GL10_X[q] + 1.0);
                double wt[MAXM], vt[MAXM];
                for (int jj = 0; jj < m_; ++jj) {
                    double sum = 0.0;
                    for (int i = 0; i < n_; ++i) sum -= Bt_[i][jj] * pval(Z[n_ + i], GN, sig);
                    wt[jj] = sum;
                }
                for (int i = 0; i < m_; ++i) {
                    double sum = g.q[i];
                    for (int k = 0; k < m_; ++k) sum += g.P[i][k] * wt[k];
                    vt[i] = sum;
                }
                for (int k = 0; k < 2 * m_; ++k) {
                    if (!finite_[k] || !(rhs0_[k] > 0)) continue;
                    double gk = 0.0;
                    for (int i = 0; i < m_; ++i) gk += rows_[k][i] * vt[i];
                    r = fmax(r, gk / rhs0_[k]);
                }
            }
        }
    }
    return r;
}

// =====================================================================================
// Stable-basis motion primitives (general LQ cost). Mirrors _StabCtx / _solve_stable in
// develop/python/lqocp.py; the boundary-value solve is a carried-relation sweep here (see
// NOTES.md section 9) instead of Python's dense solve.
// =====================================================================================
namespace {

const double DELTA_FAC[3] = {1.0, 0.5, 2.0};

// C(n x m) = A(n x k) B(k x m), leading dimensions given
void mmul(const double* A, int lda, const double* B, int ldb, double* C, int ldc, int n, int k,
          int m) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) {
            double s = 0.0;
            for (int l = 0; l < k; ++l) s += A[i * lda + l] * B[l * ldb + j];
            C[i * ldc + j] = s;
        }
}

// E = expm(t A) for n <= GZ (row stride ld); scaling and squaring with Taylor terms.
void expm_small(const double* A, int n, int ld, double t, double* E, double* X, double* Y,
                double* Z) {
    double nrm = 0.0;
    for (int j = 0; j < n; ++j) {
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += fabs(A[i * ld + j]);
        nrm = fmax(nrm, s);
    }
    nrm *= fabs(t);
    int sq = 0;
    while (nrm > 0.5 && sq < 60) { nrm *= 0.5; ++sq; }
    const double sc = t / ldexp(1.0, sq);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            X[i * n + j] = A[i * ld + j] * sc;
            E[i * n + j] = Y[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    for (int k = 1; k <= 20; ++k) {
        mmul(Y, n, X, n, Z, n, n, n, n);
        double mx = 0.0;
        for (int i = 0; i < n * n; ++i) { Y[i] = Z[i] / k; E[i] += Y[i]; mx = fmax(mx, fabs(Y[i])); }
        if (mx < 1e-18) break;
    }
    for (int s = 0; s < sq; ++s) {
        mmul(E, n, E, n, Z, n, n, n, n);
        for (int i = 0; i < n * n; ++i) E[i] = Z[i];
    }
}

// Householder QR of A (m x n, stride lda, m >= n); Q^T applied to B (m x nb, stride ldb).
void house_qr(double* A, int m, int n, int lda, double* B, int nb, int ldb) {
    for (int k = 0; k < n; ++k) {
        double nrm = 0.0;
        for (int i = k; i < m; ++i) nrm += A[i * lda + k] * A[i * lda + k];
        nrm = sqrt(nrm);
        if (nrm == 0.0) continue;
        const double alpha = (A[k * lda + k] > 0) ? -nrm : nrm;
        double v[64];
        for (int i = k; i < m; ++i) v[i] = A[i * lda + k];
        v[k] -= alpha;
        double vn = 0.0;
        for (int i = k; i < m; ++i) vn += v[i] * v[i];
        if (vn == 0.0) continue;
        for (int j = k; j < n; ++j) {
            double s = 0.0;
            for (int i = k; i < m; ++i) s += v[i] * A[i * lda + j];
            s = 2.0 * s / vn;
            for (int i = k; i < m; ++i) A[i * lda + j] -= s * v[i];
        }
        for (int j = 0; j < nb; ++j) {
            double s = 0.0;
            for (int i = k; i < m; ++i) s += v[i] * B[i * ldb + j];
            s = 2.0 * s / vn;
            for (int i = k; i < m; ++i) B[i * ldb + j] -= s * v[i];
        }
    }
}

// R x = b for the upper triangle of A (n x n, stride lda); B has nb columns (stride ldb)
bool back_solve(const double* A, int n, int lda, double* B, int nb, int ldb) {
    for (int c = 0; c < nb; ++c)
        for (int i = n - 1; i >= 0; --i) {
            double s = B[i * ldb + c];
            for (int k = i + 1; k < n; ++k) s -= A[i * lda + k] * B[k * ldb + c];
            if (A[i * lda + i] == 0.0) return false;
            B[i * ldb + c] = s / A[i * lda + i];
        }
    return true;
}

// Orthonormalise the rows of C (r x n) by modified Gram-Schmidt, same row operations on D
// (r x nd). Deterministic, so a repeated call reproduces the transformation exactly.
void mgs_rows(double* C, int r, int n, int ldc, double* D, int nd, int ldd) {
    for (int i = 0; i < r; ++i) {
        for (int pass = 0; pass < 2; ++pass)
            for (int k = 0; k < i; ++k) {
                double s = 0.0;
                for (int j = 0; j < n; ++j) s += C[i * ldc + j] * C[k * ldc + j];
                for (int j = 0; j < n; ++j) C[i * ldc + j] -= s * C[k * ldc + j];
                for (int j = 0; j < nd; ++j) D[i * ldd + j] -= s * D[k * ldd + j];
            }
        double nn = 0.0;
        for (int j = 0; j < n; ++j) nn += C[i * ldc + j] * C[i * ldc + j];
        nn = sqrt(nn);
        if (nn == 0.0) continue;
        for (int j = 0; j < n; ++j) C[i * ldc + j] /= nn;
        for (int j = 0; j < nd; ++j) D[i * ldd + j] /= nn;
    }
}

// matrix sign function (scaled Newton); S in/out (n x n), W1, W2 scratch
bool sign_fn(double* S, int n, double* W1, double* W2) {
    for (int it = 0; it < 60; ++it) {
        for (int i = 0; i < n * n; ++i) W1[i] = S[i];
        for (int i = 0; i < n * n; ++i) W2[i] = 0.0;
        for (int i = 0; i < n; ++i) W2[i * n + i] = 1.0;
        if (!lu_solve(W1, n, n, W2, n, n)) return false;        // W2 = S^-1
        double ns = 0.0, ni = 0.0;
        for (int j = 0; j < n; ++j) {
            double a = 0.0, b = 0.0;
            for (int i = 0; i < n; ++i) { a += fabs(S[i * n + j]); b += fabs(W2[i * n + j]); }
            ns = fmax(ns, a);
            ni = fmax(ni, b);
        }
        const double mu = (it < 20) ? sqrt(ni / ns) : 1.0;
        double dn = 0.0, sn = 0.0;
        for (int j = 0; j < n; ++j) {
            double a = 0.0, b = 0.0;
            for (int i = 0; i < n; ++i) {
                const double v = 0.5 * (mu * S[i * n + j] + W2[i * n + j] / mu);
                a += fabs(v - S[i * n + j]);
                b += fabs(v);
                S[i * n + j] = v;
            }
            dn = fmax(dn, a);
            sn = fmax(sn, b);
        }
        if (dn <= 1e-14 * sn) return true;
    }
    return false;
}

// r columns spanning range(P) (n x n) by Gram-Schmidt with column pivoting -> out cols c0..
void range_basis(const double* P, int n, int r, double* Wb, int c0, double* Wk) {
    for (int i = 0; i < n * n; ++i) Wk[i] = P[i];
    for (int k = 0; k < r; ++k) {
        int best = -1;
        double bn = -1.0;
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int i = 0; i < n; ++i) s += Wk[i * n + j] * Wk[i * n + j];
            if (s > bn) { bn = s; best = j; }
        }
        bn = sqrt(bn);
        double q[N2];
        for (int i = 0; i < n; ++i) q[i] = (bn > 0) ? Wk[i * n + best] / bn : 0.0;
        for (int pass = 0; pass < 2; ++pass)
            for (int kk = 0; kk < k; ++kk) {
                double s = 0.0;
                for (int i = 0; i < n; ++i) s += q[i] * Wb[i * n + c0 + kk];
                for (int i = 0; i < n; ++i) q[i] -= s * Wb[i * n + c0 + kk];
            }
        double qn = 0.0;
        for (int i = 0; i < n; ++i) qn += q[i] * q[i];
        qn = sqrt(qn);
        for (int i = 0; i < n; ++i) { q[i] /= qn; Wb[i * n + c0 + k] = q[i]; }
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int i = 0; i < n; ++i) s += q[i] * Wk[i * n + j];
            for (int i = 0; i < n; ++i) Wk[i * n + j] -= s * q[i];
        }
    }
}

}  // namespace

// ---- per-active-set stable-basis primitive (LRU cache) ----
int Solver::sbasis(uint16_t S) {
    ++gw_.clock;
    for (int i = 0; i < gw_.nsb; ++i)
        if (gw_.sb[i].S == S) {
            gw_.sb[i].stamp = gw_.clock;
            if (!gw_.sb[i].part_ok && !sparticular(gw_.sb[i], S)) return -1;
            return i;
        }
    int slot = gw_.nsb;
    if (slot >= MAXSB) {
        slot = 0;
        for (int i = 1; i < MAXSB; ++i)
            if (gw_.sb[i].stamp < gw_.sb[slot].stamp) slot = i;
    } else {
        ++gw_.nsb;
    }
    SBasis& b = gw_.sb[slot];
    b.S = 0xFFFFu;                            // invalid until complete
    ++gw_.gen;                                // a basis changes: drop cached exponentials
    const int gi = gset(S);
    if (gi < 0) return -1;
    const GSet& g = gw_.sets[gi];
    const int n = N2;
    double* H = gw_.M1;                       // H (24 x 24) by columns via hmul
    {
        double e[GZ], o[GZ];
        for (int c = 0; c < n; ++c) {
            for (int i = 0; i < GZ; ++i) e[i] = (i == c) ? 1.0 : 0.0;
            e[2 * n_] = 0.0;
            hmul(g, e, o);
            for (int i = 0; i < n; ++i) H[i * n + c] = o[i];
        }
    }
    // balancing: Hb = D^-1 H D (powers of two)
    double dsc[N2];
    for (int i = 0; i < n; ++i) dsc[i] = 1.0;
    for (int sweep = 0; sweep < 30; ++sweep) {
        bool changed = false;
        for (int i = 0; i < n; ++i) {
            double c = 0.0, r = 0.0;
            for (int k = 0; k < n; ++k)
                if (k != i) { c += fabs(H[k * n + i]); r += fabs(H[i * n + k]); }
            if (c == 0.0 || r == 0.0) continue;
            double f = 1.0;
            const double s0 = c + r;
            while (c < r / 2) { c *= 2; r /= 2; f *= 2; }
            while (c >= r * 2) { c /= 2; r *= 2; f /= 2; }
            if (f != 1.0 && c + r < 0.95 * s0) {
                dsc[i] *= f;
                for (int k = 0; k < n; ++k) { H[k * n + i] *= f; H[i * n + k] /= f; }
                changed = true;
            }
        }
        if (!changed) break;
    }
    double hmax = 0.0;
    for (int i = 0; i < n * n; ++i) hmax = fmax(hmax, fabs(H[i]));
    // dichotomy by the matrix sign function, centre half-width delta in {1, 0.5, 2} / T
    double* Pu = gw_.M2;
    double* Ps = gw_.M3;
    int ns = -1, nu = -1;
    for (int fi = 0; fi < 3 && ns < 0; ++fi) {
        const double de = DELTA_FAC[fi] / p_.T;
        for (int i = 0; i < n * n; ++i) { Pu[i] = H[i]; Ps[i] = H[i]; }
        for (int i = 0; i < n; ++i) { Pu[i * n + i] -= de; Ps[i * n + i] += de; }
        if (!sign_fn(Pu, n, gw_.M4, gw_.M5) || !sign_fn(Ps, n, gw_.M4, gw_.M5)) continue;
        double tu = 0.0, ts = 0.0;
        for (int i = 0; i < n * n; ++i) { Pu[i] *= 0.5; Ps[i] *= -0.5; }
        for (int i = 0; i < n; ++i) { Pu[i * n + i] += 0.5; Ps[i * n + i] += 0.5; tu += Pu[i * n + i]; ts += Ps[i * n + i]; }
        if (fabs(tu - floor(tu + 0.5)) > 1e-6 || fabs(ts - floor(ts + 0.5)) > 1e-6) continue;
        // invariance residual (I - P) H P
        bool ok = true;
        for (int w = 0; w < 2 && ok; ++w) {
            const double* P = w ? Ps : Pu;
            mmul(H, n, P, n, gw_.M4, n, n, n, n);
            mmul(P, n, gw_.M4, n, gw_.M5, n, n, n, n);
            double res = 0.0;
            for (int i = 0; i < n * n; ++i) res = fmax(res, fabs(gw_.M4[i] - gw_.M5[i]));
            if (res > 1e-10 * hmax && hmax > 0) ok = false;
        }
        if (!ok) continue;
        nu = (int)floor(tu + 0.5);
        ns = (int)floor(ts + 0.5);
    }
    if (ns < 0) return -1;
    const int nc = n - ns - nu;
    double* Pc = gw_.M4;
    for (int i = 0; i < n * n; ++i) Pc[i] = -Pu[i] - Ps[i];
    for (int i = 0; i < n; ++i) Pc[i * n + i] += 1.0;
    double* Wb = gw_.M5;
    range_basis(Ps, n, ns, Wb, 0, gw_.Ms);
    range_basis(Pc, n, nc, Wb, ns, gw_.Ms);
    range_basis(Pu, n, nu, Wb, ns + nc, gw_.Ms);
    // T = Wb^-1 Hb Wb ; W = D Wb
    mmul(H, n, Wb, n, gw_.M2, n, n, n, n);
    for (int i = 0; i < n * n; ++i) gw_.M3[i] = Wb[i];
    if (!lu_solve(gw_.M3, n, n, gw_.M2, n, n)) return -1;
    int k = 0;
    const int off[3] = {0, ns, ns + nc}, sz[3] = {ns, nc, nu};
    for (int blk = 0; blk < 3; ++blk)
        for (int i = 0; i < sz[blk]; ++i)
            for (int j = 0; j < sz[blk]; ++j) b.Tp[k++] = (real)gw_.M2[(off[blk] + i) * n + off[blk] + j];
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) b.W[i * n + j] = (real)(dsc[i] * Wb[i * n + j]);
    b.ns = (int16_t)ns;
    b.nc = (int16_t)nc;
    b.nu = (int16_t)nu;
    b.stamp = gw_.clock;
    if (!sparticular(b, S)) return -1;
    b.S = S;
    return slot;
}

bool Solver::sparticular(SBasis& b, uint16_t S) {
    const int gi = gset(S);
    if (gi < 0) return false;
    const GSet& g = gw_.sets[gi];
    const int n = N2;
    double f[N2];
    for (int i = 0; i < n_; ++i) {
        double s = 0.0;
        for (int k = 0; k < m_; ++k) s += Bt_[i][k] * g.q[k];
        f[i] = s;
        f[n_ + i] = 0.0;
    }
    for (int i = 0; i < n * n; ++i) gw_.M1[i] = b.W[i];
    if (!lu_solve(gw_.M1, n, n, f, 1, 1)) return false;       // f <- W^-1 f
    const int ns = b.ns, nc = b.nc, nu = b.nu;
    const real* Ts = b.Tp;
    const real* Tu = b.Tp + ns * ns + nc * nc;
    for (int i = 0; i < ns; ++i) {
        b.yps[i] = -f[i];
        for (int j = 0; j < ns; ++j) gw_.M1[i * ns + j] = Ts[i * ns + j];
    }
    if (ns && !lu_solve(gw_.M1, ns, ns, b.yps, 1, 1)) return false;
    for (int i = 0; i < nu; ++i) {
        b.ypu[i] = -f[ns + nc + i];
        for (int j = 0; j < nu; ++j) gw_.M1[i * nu + j] = Tu[i * nu + j];
    }
    if (nu && !lu_solve(gw_.M1, nu, nu, b.ypu, 1, 1)) return false;
    for (int i = 0; i < nc; ++i) b.gc[i] = f[ns + i];
    b.part_ok = 1;
    return true;
}

// Ms, ps (z at the arc start) and Me, pe (z at the arc end) for an arc of length L.
bool Solver::arc_maps(int bi, double L, bool, double*, double*, double*) {
    const SBasis& b = gw_.sb[bi];
    const int n = N2, ns = b.ns, nc = b.nc, nu = b.nu;
    double* Es = gw_.M1;      // ns x ns
    double* Ec = gw_.M2;      // (nc+1)^2
    double* Eu = gw_.M3;      // nu x nu
    double* A = gw_.M4;
    const real* Ts = b.Tp;
    const real* Tc = b.Tp + ns * ns;
    const real* Tu = Tc + nc * nc;
    const int m = nc + 1;
    // per-arc exponential cache (slot = arc position), keyed by (set, L, generation)
    const int need = ns * ns + m * m + nu * nu;
    int slot = -1;
    for (int i = 0; i < MAXARC_G; ++i)
        if (gw_.ex_S[i] == b.S && gw_.ex_L[i] == L && gw_.ex_gen[i] == gw_.gen) { slot = i; break; }
    if (slot >= 0) {
        const real* e = gw_.ex[slot];
        for (int i = 0; i < ns * ns; ++i) Es[i] = e[i];
        for (int i = 0; i < m * m; ++i) Ec[i] = e[ns * ns + i];
        for (int i = 0; i < nu * nu; ++i) Eu[i] = e[ns * ns + m * m + i];
    } else {
        if (ns) {
            for (int i = 0; i < ns * ns; ++i) A[i] = Ts[i];
            expm_small(A, ns, ns, L, Es, gw_.M5, gw_.Ms, gw_.Me);
        }
        for (int i = 0; i < m * m; ++i) A[i] = 0.0;
        for (int i = 0; i < nc; ++i) {
            for (int j = 0; j < nc; ++j) A[i * m + j] = Tc[i * nc + j];
            A[i * m + nc] = b.gc[i];
        }
        expm_small(A, m, m, L, Ec, gw_.M5, gw_.Ms, gw_.Me);
        if (nu) {
            for (int i = 0; i < nu * nu; ++i) A[i] = Tu[i];
            expm_small(A, nu, nu, -L, Eu, gw_.M5, gw_.Ms, gw_.Me);
        }
        if (need <= EXPCAP) {
            const int i0 = gw_.ex_next;
            gw_.ex_next = (gw_.ex_next + 1) % MAXARC_G;
            real* e = gw_.ex[i0];
            for (int i = 0; i < ns * ns; ++i) e[i] = (real)Es[i];
            for (int i = 0; i < m * m; ++i) e[ns * ns + i] = (real)Ec[i];
            for (int i = 0; i < nu * nu; ++i) e[ns * ns + m * m + i] = (real)Eu[i];
            gw_.ex_S[i0] = b.S;
            gw_.ex_L[i0] = L;
            gw_.ex_gen[i0] = gw_.gen;
        }
    }
    for (int i = 0; i < n; ++i) {
        const real* Wr = b.W + i * n;
        double ps = 0.0, pe = 0.0;
        for (int j = 0; j < ns; ++j) {
            gw_.Ms[i * n + j] = Wr[j];
            double s = 0.0;
            for (int l = 0; l < ns; ++l) s += Wr[l] * Es[l * ns + j];
            gw_.Me[i * n + j] = s;
            ps += Wr[j] * b.yps[j];
        }
        pe = ps;
        for (int j = 0; j < nc; ++j) {
            gw_.Ms[i * n + ns + j] = Wr[ns + j];
            double s = 0.0;
            for (int l = 0; l < nc; ++l) s += Wr[ns + l] * Ec[l * m + j];
            gw_.Me[i * n + ns + j] = s;
            pe += Wr[ns + j] * Ec[j * m + nc];
        }
        for (int j = 0; j < nu; ++j) {
            double s = 0.0;
            for (int l = 0; l < nu; ++l) s += Wr[ns + nc + l] * Eu[l * nu + j];
            gw_.Ms[i * n + ns + nc + j] = s;
            gw_.Me[i * n + ns + nc + j] = Wr[ns + nc + j];
            ps += Wr[ns + nc + j] * b.ypu[j];
            pe += Wr[ns + nc + j] * b.ypu[j];
        }
        gw_.ps[i] = ps;
        gw_.pe[i] = pe;
    }
    return true;
}

// Linear BVP for fixed junction times: forward sweep of the carried relation
// C_k z(tau_k) = d_k (12 orthonormal rows), terminal solve, backward recovery of the arc
// coefficients. Every operation is orthogonal or uses only bounded exponentials.
bool Solver::bvp(const uint16_t* seq, int N, const double* tau, double* zT) {
    const int n = N2, r = n_;
    if (N > MAXARC_G) return false;
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < n; ++j) gw_.C[0][i * n + j] = (i == j) ? 1.0 : 0.0;
        gw_.d[0][i] = p_.x0[i];
    }
    double A[(N2 + MAXN) * N2], B[(N2 + MAXN) * (N2 + 1)];
    const int ldb = n + 1;
    for (int k = 0; k < N; ++k) {
        const double t0 = (k == 0) ? 0.0 : tau[k - 1], t1 = (k == N - 1) ? p_.T : tau[k];
        const int bi = sbasis(seq[k]);
        if (bi < 0 || !arc_maps(bi, t1 - t0, false, nullptr, nullptr, nullptr)) return false;
        for (int i = 0; i < r; ++i) {
            double dv = gw_.d[k][i];
            for (int j = 0; j < n; ++j) {
                double s = 0.0;
                for (int l = 0; l < n; ++l) s += gw_.C[k][i * n + l] * gw_.Ms[l * n + j];
                A[i * n + j] = s;
                dv -= gw_.C[k][i * n + j] * gw_.ps[j];
                B[i * ldb + j] = 0.0;
            }
            B[i * ldb + n] = dv;
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A[(r + i) * n + j] = gw_.Me[i * n + j];
                B[(r + i) * ldb + j] = (i == j) ? -1.0 : 0.0;
            }
            B[(r + i) * ldb + n] = -gw_.pe[i];
        }
        house_qr(A, r + n, n, n, B, n + 1, ldb);
        double Cn[MAXN * N2], dn[MAXN];
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < n; ++j) Cn[i * n + j] = B[(n + i) * ldb + j];
            dn[i] = B[(n + i) * ldb + n];
        }
        mgs_rows(Cn, r, n, n, dn, 1, 1);
        for (int i = 0; i < r * n; ++i) gw_.C[k + 1][i] = (real)Cn[i];
        for (int i = 0; i < r; ++i) gw_.d[k + 1][i] = (real)dn[i];
    }
    // terminal: [C_N; Cm] z_T = [d_N; dv]
    double Mt[N2 * N2];
    double rhs[N2];
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < n; ++j) Mt[i * n + j] = gw_.C[N][i * n + j];
        rhs[i] = gw_.d[N][i];
        for (int j = 0; j < n; ++j) Mt[(r + i) * n + j] = 0.0;
        if (p_.has_Qf) {
            for (int j = 0; j < r; ++j) Mt[(r + i) * n + j] = -p_.Qf[i][j];
            Mt[(r + i) * n + r + i] = 1.0;
            double s = 0.0;
            for (int j = 0; j < r; ++j) s -= p_.Qf[i][j] * p_.xf[j];
            rhs[r + i] = s;
        } else {
            Mt[(r + i) * n + i] = 1.0;
            rhs[r + i] = p_.xf[i];
        }
    }
    if (!lu_solve(Mt, n, n, rhs, 1, 1)) return false;     // singular: infeasible endpoint
    for (int i = 0; i < n; ++i) { gw_.zj[N][i] = rhs[i]; zT[i] = rhs[i]; }
    // backward
    for (int k = N - 1; k >= 0; --k) {
        const double t0 = (k == 0) ? 0.0 : tau[k - 1], t1 = (k == N - 1) ? p_.T : tau[k];
        const int bi = sbasis(seq[k]);
        if (bi < 0 || !arc_maps(bi, t1 - t0, false, nullptr, nullptr, nullptr)) return false;
        double y[N2 + MAXN];
        for (int i = 0; i < r; ++i) {
            double dv = gw_.d[k][i];
            for (int j = 0; j < n; ++j) {
                double s = 0.0;
                for (int l = 0; l < n; ++l) s += gw_.C[k][i * n + l] * gw_.Ms[l * n + j];
                A[i * n + j] = s;
                dv -= gw_.C[k][i * n + j] * gw_.ps[j];
            }
            y[i] = dv;
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) A[(r + i) * n + j] = gw_.Me[i * n + j];
            y[r + i] = gw_.zj[k + 1][i] - gw_.pe[i];
        }
        house_qr(A, r + n, n, n, y, 1, 1);
        if (!back_solve(A, n, n, y, 1, 1)) return false;
        for (int i = 0; i < n; ++i) {
            gw_.x[k][i] = y[i];
            double s = gw_.ps[i];
            for (int j = 0; j < n; ++j) s += gw_.Ms[i * n + j] * y[j];
            gw_.zj[k][i] = s;
        }
    }
    return true;
}

// Residual rows of the junction conditions: for each switching row, its gap on the side
// where it is inactive (G, g0 act on wt = -Bt^T lambda); jq = junction index.
int Solver::jrows(const uint16_t* seq, int N, int* jq, double G[][MAXM], double* g0) const {
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
                    jq[nr] = j;
                    for (int c = 0; c < m_; ++c) G[nr][c] = Gs[q][c];
                    g0[nr] = gs[q];
                    ++nr;
                }
        }
    }
    return nr;
}

// Sensitivities of z(tau_q^-) to all junction times: moving tau_j adds the jump
// (f_{j-1} - f_j)(z(tau_j)) to the homogeneous BVP. One forward and one backward sweep with
// N-1 right-hand sides. dz_left[(q * N2 + i) * MAXJ_G + j] = d z_i(tau_q^-) / d tau_j.
bool Solver::sens(const uint16_t* seq, int N, const double* tau, int nr, double* Jout) {
    const int n = N2, r = n_, nj = N - 1;
    if (nj <= 0) return true;
    double (&Dl)[N2][MAXARC_G] = gw_.Dl;            // jumps Delta_j
    for (int j = 1; j < N; ++j) {
        double z[GZ], f1[GZ], f2[GZ];
        for (int i = 0; i < n; ++i) z[i] = gw_.zj[j][i];
        z[n] = 1.0;
        const int g1 = gset(seq[j - 1]);
        if (g1 < 0) return false;
        hmul(gw_.sets[g1], z, f1);
        const int g2 = gset(seq[j]);
        if (g2 < 0) return false;
        hmul(gw_.sets[g2], z, f2);
        for (int i = 0; i < n; ++i) Dl[i][j - 1] = f1[i] - f2[i];
    }
    // forward: dd_k (12 x nj) for the relation at tau_k
    double A[(N2 + MAXN) * N2], B[(N2 + MAXN) * (N2 + MAXARC_G)];
    const int ldb = n + nj;
    real (&dd)[MAXARC_G + 1][MAXN][MAXARC_G] = gw_.dd;
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < nj; ++j) dd[0][i][j] = 0.0;
    for (int k = 0; k < N; ++k) {
        if (k >= 1) {                                  // jump of junction k enters here
            for (int i = 0; i < r; ++i) {
                double s = 0.0;
                for (int l = 0; l < n; ++l) s += gw_.C[k][i * n + l] * Dl[l][k - 1];
                dd[k][i][k - 1] += s;
            }
        }
        const double t0 = (k == 0) ? 0.0 : tau[k - 1], t1 = (k == N - 1) ? p_.T : tau[k];
        const int bi = sbasis(seq[k]);
        if (bi < 0 || !arc_maps(bi, t1 - t0, false, nullptr, nullptr, nullptr)) return false;
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < n; ++j) {
                double s = 0.0;
                for (int l = 0; l < n; ++l) s += gw_.C[k][i * n + l] * gw_.Ms[l * n + j];
                A[i * n + j] = s;
                B[i * ldb + j] = 0.0;
            }
            for (int j = 0; j < nj; ++j) B[i * ldb + n + j] = dd[k][i][j];
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                A[(r + i) * n + j] = gw_.Me[i * n + j];
                B[(r + i) * ldb + j] = (i == j) ? -1.0 : 0.0;
            }
            for (int j = 0; j < nj; ++j) B[(r + i) * ldb + n + j] = 0.0;
        }
        house_qr(A, r + n, n, n, B, ldb, ldb);
        double Cn[MAXN * N2], Dn[MAXN * MAXARC_G];
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < n; ++j) Cn[i * n + j] = B[(n + i) * ldb + j];
            for (int j = 0; j < nj; ++j) Dn[i * nj + j] = B[(n + i) * ldb + n + j];
        }
        mgs_rows(Cn, r, n, n, Dn, nj, nj);
        for (int i = 0; i < r; ++i)
            for (int j = 0; j < nj; ++j) dd[k + 1][i][j] = (real)Dn[i * nj + j];
    }
    // terminal: [C_N; Cm] dz_T = [dd_N; 0]
    double Mt[N2 * N2], Z[N2 * MAXARC_G];
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < n; ++j) Mt[i * n + j] = gw_.C[N][i * n + j];
        for (int j = 0; j < nj; ++j) { Z[i * nj + j] = dd[N][i][j]; Z[(r + i) * nj + j] = 0.0; }
        for (int j = 0; j < n; ++j) Mt[(r + i) * n + j] = 0.0;
        if (p_.has_Qf) {
            for (int j = 0; j < r; ++j) Mt[(r + i) * n + j] = -p_.Qf[i][j];
            Mt[(r + i) * n + r + i] = 1.0;
        } else {
            Mt[(r + i) * n + i] = 1.0;
        }
    }
    if (!lu_solve(Mt, n, n, Z, nj, nj)) return false;
    // backward
    for (int k = N - 1; k >= 0; --k) {
        // Z holds dz(tau_{k+1}^-) as seen from the end of arc k: Jacobian rows of junction k+1
        if (k + 1 <= N - 1)
            for (int q = 0; q < nr; ++q) {
                if (gw_.jq[q] != k + 1) continue;
                for (int j = 0; j < nj; ++j) {
                    double v = 0.0;
                    for (int c = 0; c < m_; ++c) {
                        double w = 0.0;
                        for (int i = 0; i < n_; ++i) w -= Bt_[i][c] * Z[(n_ + i) * nj + j];
                        v += gw_.jG[q][c] * w;
                    }
                    Jout[q * MAXARC_G + j] = v;
                }
            }
        const double t0 = (k == 0) ? 0.0 : tau[k - 1], t1 = (k == N - 1) ? p_.T : tau[k];
        const int bi = sbasis(seq[k]);
        if (bi < 0 || !arc_maps(bi, t1 - t0, false, nullptr, nullptr, nullptr)) return false;
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < n; ++j) {
                double s = 0.0;
                for (int l = 0; l < n; ++l) s += gw_.C[k][i * n + l] * gw_.Ms[l * n + j];
                A[i * n + j] = s;
            }
            for (int j = 0; j < nj; ++j) B[i * nj + j] = dd[k][i][j];
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) A[(r + i) * n + j] = gw_.Me[i * n + j];
            for (int j = 0; j < nj; ++j) B[(r + i) * nj + j] = Z[i * nj + j];
        }
        house_qr(A, r + n, n, n, B, nj, nj);
        if (!back_solve(A, n, n, B, nj, nj)) return false;
        // dz(tau_k^+) = Ms x'; left limit at tau_k subtracts the jump of junction k
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < nj; ++j) {
                double s = 0.0;
                for (int l = 0; l < n; ++l) s += gw_.Ms[i * n + l] * B[l * nj + j];
                Z[i * nj + j] = s;
            }
        if (k >= 1)
            for (int i = 0; i < n; ++i) Z[i * nj + (k - 1)] -= Dl[i][k - 1];
    }
    return true;
}

// Exact event: arc j shrinks to zero length along a Newton step. Equal neighbours -> merge
// them; neighbours that differ in two rows -> the two switches swap order; else drop the arc.
void Solver::collapse(uint16_t* seq, int* N, double* tau, int j) const {
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
    if (D == 0) {                                   // merge: drop arcs j, j+1 and two junctions
        for (int k = j; k + 2 < n; ++k) seq[k] = seq[k + 2];
        for (int k = j - 1; k + 2 < n - 1; ++k) tau[k] = tau[k + 2];
        *N = n - 2;
        return;
    }
    const uint16_t first = (uint16_t)(A ^ Mid);
    const bool one = first && !(first & (first - 1));
    int nd = 0;
    for (uint16_t x = D; x; x &= (uint16_t)(x - 1)) ++nd;
    if (nd == 2 && one && (first & D) == first) {   // swap the order of the two switches
        const uint16_t Mn = (uint16_t)(A ^ (D & ~first));
        if (valid_set(Mn)) {
            seq[j] = Mn;
            tau[j - 1] = mid;
            tau[j] = mid + 1e-9 * p_.T;
            return;
        }
    }
    for (int k = j; k + 1 < n; ++k) seq[k] = seq[k + 1];          // drop arc j
    tau[j - 1] = mid;
    for (int k = j; k + 2 < n; ++k) tau[k] = tau[k + 1];
    *N = n - 1;
}

// Newton on the junction times with exact derivatives (sens). Arc collapse inside a step is an
// exact event (collapse). Fails fast when the residual does not contract. seq/N/tau are
// updated in place (events may change the structure).
bool Solver::newton_tau(uint16_t* seq, int* Np, double* tau, int* its, double* zT) {
    *its = 0;
    int N = *Np;
    if (!bvp(seq, N, tau, zT)) return false;
    int* jq = gw_.jq;
    double (*G)[MAXM] = gw_.jG;
    double* g0 = gw_.jg0;
    double* J = gw_.J;
    int nr = jrows(seq, N, jq, G, g0);
    if (nr == 0) return true;
    auto resid = [&](double* rv) {
        double mx = 0.0;
        for (int q = 0; q < nr; ++q) {
            double v = g0[q];
            for (int c = 0; c < m_; ++c) {
                double w = 0.0;
                for (int i = 0; i < n_; ++i) w -= Bt_[i][c] * gw_.zj[jq[q]][n_ + i];
                v += G[q][c] * w;
            }
            rv[q] = v;
            mx = fmax(mx, fabs(v));
        }
        return mx;
    };
    double r[2 * MAXARC_G], rn[2 * MAXARC_G], tn[MAXARC_G], d[MAXARC_G];
    double rmax = resid(r);
    const double tol = 1e-12 * R_, gap = 1e-12 * p_.T;
    double prev = HUGE_VAL;
    int bad = 0;
    while (*its < MAX_TAU_NEWTON) {
        const int nj = N - 1;
        if (rmax <= tol) { *Np = N; return true; }
        if (rmax > 0.5 * prev && ++bad > 2) { *Np = N; return rmax <= 1e3 * tol; }   // stall: round-off or wrong structure
        prev = rmax;
        ++*its;
        if (!sens(seq, N, tau, nr, J)) { *Np = N; return false; }
        for (int q = 0; q < nr; ++q) {
            // explicit part: d z(tau_k^-)/d tau_k includes zdot of arc k-1
            const int k = jq[q];
            double z[GZ], f[GZ];
            for (int i = 0; i < N2; ++i) z[i] = gw_.zj[k][i];
            z[N2] = 1.0;
            const int gi = gset(seq[k - 1]);
            if (gi < 0) return false;
            hmul(gw_.sets[gi], z, f);
            double v = 0.0;
            for (int c = 0; c < m_; ++c) {
                double w = 0.0;
                for (int i = 0; i < n_; ++i) w -= Bt_[i][c] * f[n_ + i];
                v += G[q][c] * w;
            }
            J[q * MAXARC_G + (k - 1)] += v;
        }
        // Gauss-Newton step (normal equations)
        double Nm[MAXARC_G * MAXARC_G];
        for (int a = 0; a < nj; ++a) {
            double s = 0.0;
            for (int q = 0; q < nr; ++q) s += J[q * MAXARC_G + a] * r[q];
            d[a] = -s;
            for (int b2 = 0; b2 < nj; ++b2) {
                double g = 0.0;
                for (int q = 0; q < nr; ++q) g += J[q * MAXARC_G + a] * J[q * MAXARC_G + b2];
                Nm[a * nj + b2] = g;
            }
        }
        if (!lu_solve(Nm, nj, nj, d, 1, 1)) { *Np = N; return false; }
        // exact arc-collapse event along the full step
        double amax = 1.0;
        int jc = -1;
        for (int j = 0; j <= nj; ++j) {
            const double L = ((j == nj) ? p_.T : tau[j]) - ((j == 0) ? 0.0 : tau[j - 1]);
            const double dL = ((j == nj) ? 0.0 : d[j]) - ((j == 0) ? 0.0 : d[j - 1]);
            if (dL < 0 && L + dL <= gap && L / -dL < amax) { amax = L / -dL; jc = j; }
        }
        if (jc >= 0) {
            for (int j = 0; j < nj; ++j) tau[j] += amax * d[j];
            collapse(seq, &N, tau, jc);
            double pv = 0.0;
            for (int j = 0; j + 1 < N; ++j) { if (!(tau[j] > pv)) { *Np = N; return false; } pv = tau[j]; }
            if (!(p_.T > pv)) { *Np = N; return false; }
            ++gw_.events;
            if (!bvp(seq, N, tau, zT)) { *Np = N; return false; }
            nr = jrows(seq, N, jq, G, g0);
            if (nr == 0) { *Np = N; return true; }
            rmax = resid(r);
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
            if (bvp(seq, N, tn, zT)) {
                const double mx = resid(rn);
                double nn = 0.0;
                for (int q = 0; q < nr; ++q) nn += rn[q] * rn[q];
                if (sqrt(nn) <= (1 - 1e-4 * alpha) * nr0 || mx <= tol) { acc = true; rmax = mx; break; }
            }
            alpha *= 0.5;
        }
        if (!acc) {
            bvp(seq, N, tau, zT);                     // restore the state of tau
            *Np = N;
            return rmax <= 1e3 * tol;                 // round-off stall
        }
        for (int j = 0; j < nj; ++j) tau[j] = tn[j];
        for (int q = 0; q < nr; ++q) r[q] = rn[q];
    }
    *Np = N;
    return rmax <= tol;
}

// Fill R (arcs, substep states) from the current BVP solution (gw_.x, sb) for the walk,
// the cost and the evaluation; Taylor substeps as in section 8.
bool Solver::make_subs(Result& R, const uint16_t* seq, int N, const double* tau) {
    if (N > MAXARC) return false;
    if (!build_arcs(seq, N, tau, R.arcs)) return false;
    R.narcs = N;
    int gi = 0;
    const int n = N2;
    for (int k = 0; k < N; ++k) {
        const double L = R.arcs[k].t1 - R.arcs[k].t0;
        const int si = gset(seq[k]);
        if (si < 0) return false;
        int K = (int)ceil(L / gw_.sets[si].h * (1 - 1e-12));
        if (K < 1) K = 1;
        if (gi + K > MAXSUB) return false;
        R.nsub[k] = K;
        const double h = L / K;
        const int bi = sbasis(seq[k]);
        if (bi < 0) return false;
        const SBasis& b = gw_.sb[bi];
        const int ns = b.ns, nc = b.nc, nu = b.nu, m = nc + 1;
        double* Es = gw_.M1;
        double* Ec = gw_.M2;
        double* Eu = gw_.M3;
        double* A = gw_.M4;
        const real* Ts = b.Tp;
        const real* Tc = b.Tp + ns * ns;
        const real* Tu = Tc + nc * nc;
        if (ns) { for (int i = 0; i < ns * ns; ++i) A[i] = Ts[i]; expm_small(A, ns, ns, h, Es, gw_.M5, gw_.Ms, gw_.Me); }
        for (int i = 0; i < m * m; ++i) A[i] = 0.0;
        for (int i = 0; i < nc; ++i) { for (int j = 0; j < nc; ++j) A[i * m + j] = Tc[i * nc + j]; A[i * m + nc] = b.gc[i]; }
        expm_small(A, m, m, h, Ec, gw_.M5, gw_.Ms, gw_.Me);
        if (nu) { for (int i = 0; i < nu * nu; ++i) A[i] = Tu[i]; expm_small(A, nu, nu, -h, Eu, gw_.M5, gw_.Ms, gw_.Me); }
        double y[N2 + 1], t2[N2 + 1];
        // unstable part, anchored at the arc end, propagated backwards
        for (int i = 0; i < nu; ++i) y[i] = gw_.x[k][ns + nc + i];
        for (int s = K - 1; s >= 0; --s) {
            for (int i = 0; i < nu; ++i) {
                double v = 0.0;
                for (int j = 0; j < nu; ++j) v += Eu[i * nu + j] * y[j];
                t2[i] = v;
            }
            for (int i = 0; i < nu; ++i) y[i] = t2[i];
            for (int i = 0; i < n; ++i) {
                double v = 0.0;
                for (int j = 0; j < nu; ++j) v += (double)b.W[i * n + ns + nc + j] * (y[j] + b.ypu[j]);
                R.gz0[gi + s][i] = (real)v;
            }
        }
        // stable and centre parts, anchored at the arc start, propagated forwards
        double ys[N2], yc[N2 + 1];
        for (int i = 0; i < ns; ++i) ys[i] = gw_.x[k][i];
        for (int i = 0; i < nc; ++i) yc[i] = gw_.x[k][ns + i];
        yc[nc] = 1.0;
        for (int s = 0; s < K; ++s) {
            for (int i = 0; i < n; ++i) {
                double v = R.gz0[gi + s][i];
                for (int j = 0; j < ns; ++j) v += (double)b.W[i * n + j] * (ys[j] + b.yps[j]);
                for (int j = 0; j < nc; ++j) v += (double)b.W[i * n + ns + j] * yc[j];
                R.gz0[gi + s][i] = (real)v;
            }
            for (int i = 0; i < ns; ++i) {
                double v = 0.0;
                for (int j = 0; j < ns; ++j) v += Es[i * ns + j] * ys[j];
                t2[i] = v;
            }
            for (int i = 0; i < ns; ++i) ys[i] = t2[i];
            for (int i = 0; i < m; ++i) {
                double v = 0.0;
                for (int j = 0; j < m; ++j) v += Ec[i * m + j] * yc[j];
                t2[i] = v;
            }
            for (int i = 0; i < m; ++i) yc[i] = t2[i];
        }
        gi += K;
    }
    R.gnsub = gi;
    return true;
}

// Exact event walk of the pointwise H-minimiser along the current BVP solution -> gw_.seq/tau
// (round-off chatter arcs removed).
bool Solver::propose(Result& R, const uint16_t* seq, int N, const double* tau, int* ndeg) {
    if (!make_subs(R, seq, N, tau) || !gwalk(R, ndeg)) return false;
    for (int i = 1; i + 1 < gw_.nseq;) {
        const double len = gw_.tau[i] - gw_.tau[i - 1];
        if (len < 1e-10 * p_.T && gw_.seq[i - 1] == gw_.seq[i + 1]) {
            for (int k = i; k + 2 < gw_.nseq; ++k) gw_.seq[k] = gw_.seq[k + 2];
            for (int k = i - 1; k + 2 < gw_.nseq - 1; ++k) gw_.tau[k] = gw_.tau[k + 2];
            gw_.nseq -= 2;
        } else {
            ++i;
        }
    }
    return true;
}

// Corrector at a fixed bound scale, starting from the predicted structure in gw_.aseq/atau/an:
// Newton on tau, then the exact walk; repeat with the walked structure (at most 3 times).
// On success the accepted, self-consistent structure is in gw_.aseq/atau/an.
bool Solver::correct(Result& R, double* zT, int* its, int* outer, int* ndeg) {
    uint16_t seq[MAXARC];
    double tau[MAXJ];
    int N = gw_.an, retry = 1;
    for (int i = 0; i < N; ++i) seq[i] = gw_.aseq[i];
    for (int i = 0; i + 1 < N; ++i) tau[i] = gw_.atau[i];
    for (int o = 0; o < 3; ++o) {
        int k = 0;
        const bool ok = newton_tau(seq, &N, tau, &k, zT);
        *its += k;
        if (!ok && (o > 0 || retry-- == 0)) return false;
        if (!propose(R, seq, N, tau, ndeg)) return false;
        bool same = (gw_.nseq == N);
        for (int i = 0; same && i < N; ++i) {
            if (gw_.seq[i] != seq[i]) same = false;
            if (i > 0 && fabs(gw_.tau[i - 1] - tau[i - 1]) > 1e-9 * p_.T) same = false;
        }
        if (same) {
            gw_.an = N;
            for (int i = 0; i < N; ++i) gw_.aseq[i] = seq[i];
            for (int i = 0; i + 1 < N; ++i) gw_.atau[i] = tau[i];
            return true;
        }
        if (gw_.nseq > MAXARC_G) return false;
        N = gw_.nseq;
        for (int i = 0; i < N; ++i) seq[i] = gw_.seq[i];
        for (int i = 0; i + 1 < N; ++i) tau[i] = gw_.tau[i];
        ++*outer;
    }
    return false;
}

static double zmax_of(const double (*zj)[N2], int N) {
    double m = 0.0;
    for (int k = 0; k <= N; ++k)
        for (int i = 0; i < N2; ++i) m = fmax(m, fabs(zj[k][i]));
    return m;
}

bool Solver::solve_general(Result& R) {
    R.iterations = R.newton_iterations = R.outer_iterations = 0;
    R.backtracks = 0;
    R.box_fast_path = false;
    R.ntouch = 0;
    R.cost = 0.0;
    for (int k = 0; k < 2 * m_; ++k) rhs0_[k] = rhs_[k];
    gw_.nsets = 0;
    gw_.nsb = 0;
    gw_.clock = 0;
    gw_.gen = 1;
    gw_.ex_next = 0;
    gw_.events = 0;
    for (int i = 0; i < MAXARC_G; ++i) gw_.ex_gen[i] = 0;
    for (int i = 0; i < n_; ++i) R.c[i] = 0.0;
    double zT[N2 + 1];
    int its = 0, outer = 0, ndeg = 0, nsteps = 0;
    gw_.an = 1;
    gw_.aseq[0] = 0;
    bool conv = false, infeasible = false;
    if (!bvp(gw_.aseq, 1, gw_.atau, zT)) {
        infeasible = true;
    } else if (!p_.has_bounds) {
        conv = true;
    } else {
        if (!make_subs(R, gw_.aseq, 1, gw_.atau)) return false;
        const double ratio = gmax_ratio(R);
        if (ratio <= 1.0 && propose(R, gw_.aseq, 1, gw_.atau, &ndeg))
            conv = (gw_.nseq == 1 && gw_.seq[0] == 0);
        if (!conv) {
            // homotopy in the bound scale kappa (NOTES.md 9.5): exact collapse events inside
            // the tau-Newton, secant predictor, step halving on failure
            const double z0 = zmax_of(gw_.zj, 1);
            double kap = 1.05 * fmax(ratio, 1.0);
            double dl = log(kap) * ((ratio <= KAP_PLAIN) ? 1.0 : 1.0 / 3.0);
            gscale(kap);
            if (!bvp(gw_.aseq, 1, gw_.atau, zT)) return false;
            gw_.hn = 0;
            uint16_t sseq[MAXARC];
            double stau[MAXJ];
            for (;;) {
                if (kap == 1.0) { conv = true; break; }
                if (its > MAX_GEN_ITS || dl < DL_MIN) break;
                if (zmax_of(gw_.zj, gw_.an) > 1e8 * (1.0 + z0)) { infeasible = true; break; }
                const int sn = gw_.an;
                for (int i = 0; i < sn; ++i) sseq[i] = gw_.aseq[i];
                for (int i = 0; i + 1 < sn; ++i) stau[i] = gw_.atau[i];
                const double kt = fmax(1.0, kap * exp(-dl));
                gscale(kt);
                ++nsteps;
                bool same = (gw_.hn == sn);
                for (int i = 0; same && i < sn; ++i) same = (gw_.hseq[i] == sseq[i]);
                if (same) {                                  // secant predictor
                    double pv = 0.0;
                    bool ord = true;
                    for (int i = 0; i + 1 < sn; ++i) {
                        const double t = stau[i] + (stau[i] - gw_.htau[i]) / (kap - gw_.hkap) * (kt - kap);
                        gw_.atau[i] = t;
                        if (!(t > pv)) ord = false;
                        pv = t;
                    }
                    if (!ord || !(p_.T > pv))
                        for (int i = 0; i + 1 < sn; ++i) gw_.atau[i] = stau[i];
                }
                if (correct(R, zT, &its, &outer, &ndeg)) {
                    gw_.hn = sn;
                    for (int i = 0; i < sn; ++i) gw_.hseq[i] = sseq[i];
                    for (int i = 0; i + 1 < sn; ++i) gw_.htau[i] = stau[i];
                    gw_.hkap = kap;
                    kap = kt;
                    dl *= 1.5;
                } else {
                    dl *= 0.5;
                    gscale(kap);
                    gw_.an = sn;
                    for (int i = 0; i < sn; ++i) gw_.aseq[i] = sseq[i];
                    for (int i = 0; i + 1 < sn; ++i) gw_.atau[i] = stau[i];
                    if (!bvp(gw_.aseq, gw_.an, gw_.atau, zT)) { infeasible = true; break; }
                }
            }
        }
    }
    R.iterations = R.newton_iterations = its;
    R.outer_iterations = nsteps;
    R.degenerate_events = ndeg;
    if (infeasible) {
        R.narcs = 0;
        R.gnsub = 0;
        R.residual = HUGE_VAL;
        R.pmp_violation = HUGE_VAL;
        R.cost = HUGE_VAL;
        R.converged = false;
        return false;
    }
    if (!bvp(gw_.aseq, gw_.an, gw_.atau, zT) || !make_subs(R, gw_.aseq, gw_.an, gw_.atau)) {
        R.cost = HUGE_VAL;
        R.converged = false;
        return false;
    }
    double rmax = 0.0;
    for (int i = 0; i < n_; ++i) {
        double v;
        if (p_.has_Qf) {
            v = gw_.zj[gw_.an][n_ + i];
            for (int k = 0; k < n_; ++k) v -= p_.Qf[i][k] * (gw_.zj[gw_.an][k] - p_.xf[k]);
        } else {
            v = gw_.zj[gw_.an][i] - p_.xf[i];
        }
        rmax = fmax(rmax, fabs(v));
    }
    R.residual = rmax;
    R.pmp_violation = gverify(R);
    if (conv && R.pmp_violation > 1e-9 * R_) conv = false;
    for (int i = 0; i < N2; ++i) zT[i] = gw_.zj[gw_.an][i];
    zT[N2] = 1.0;
    R.cost = gcost(R, zT);
    R.converged = conv;
    return conv;
}
}  // namespace lqocp
