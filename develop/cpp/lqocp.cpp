// Solver core. See lqocp.hpp. No heap, no exceptions, no STL.
#include "lqocp.hpp"
#include "lqocp_pd.hpp"

#include <math.h>
#include <string.h>

namespace lqocp {

namespace pd {   // polynomial/linear-algebra helpers (lqocp_pd.hpp)

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
template bool lu_solve<double>(double*, int, int, double*, int, int);

}  // namespace pd
using namespace pd;

namespace pd {   // setup shared by tier 1 (Solver) and tier 3 (GSolver)

// TS of the scalar type T (a double GSolver in a float build keeps the double tolerances)
template <class T> constexpr double ts_of() { return sizeof(T) == sizeof(real) ? (double)TS : 1.0; }

// S = C^T C (C upper triangular, row stride MAXN): semidefinite Cholesky of the unit-diagonal Ss = D S D, D = diag(S)^-1/2
// (pivots are scale-free; a zero pivot leaves a zero row and needs a zero column, else S is indefinite). false: not PSD
template <class T>
bool psd_factor(const double (*S)[MAXN], int n, T* C) {
    for (int i = 0; i < MAXN * MAXN; ++i) C[i] = 0.0;
    T D[MAXN];
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k)          // a zero diagonal needs a zero row (else indefinite)
            if (S[i][i] < 0.0 || (S[i][i] == 0.0 && S[i][k] != 0.0)) return false;
        D[i] = S[i][i] > 0.0 ? 1.0 / sqrt((T)S[i][i]) : 0.0;
    }
    const T tol = 1e-14 * ts_of<T>();
    for (int j = 0; j < n; ++j) {
        T d = D[j] > 0.0 ? 1.0 : 0.0;
        for (int k = 0; k < j; ++k) d -= C[k * MAXN + j] * C[k * MAXN + j];
        if (d < -1e2 * tol) return false;
        const bool zero = d <= tol;
        if (!zero) C[j * MAXN + j] = sqrt(d);
        for (int i = j + 1; i < n; ++i) {
            T s = S[i][j] * D[i] * D[j];
            for (int k = 0; k < j; ++k) s -= C[k * MAXN + i] * C[k * MAXN + j];
            if (zero) {   // PSD: s^2 <= d * Ss_ii <= tol
                if (s * s > tol) return false;
            } else {
                C[j * MAXN + i] = s / C[j * MAXN + j];
            }
        }
    }
    for (int j = 0; j < n; ++j)                  // undo the scaling: C = Cs D^-1 (columns)
        for (int i = 0; i <= j; ++i) C[i * MAXN + j] = D[j] > 0.0 ? C[i * MAXN + j] / D[j] : 0.0;
    return true;
}

// own frame: the factor's scratch lives only during the check
__attribute__((noinline)) bool psd_ok(const double (*S)[MAXN], int n) {
    double C[MAXN * MAXN];
    return psd_factor(S, n, C);
}

bool symmetric(const double (*S)[MAXN], int n) {   // relative to |S_ij| <= sqrt(S_ii S_jj) of a PSD S
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (fabs(S[i][j] - S[j][i]) > 1e-12 * (fabs(S[i][j]) + fabs(S[j][i]) + sqrt(fabs(S[i][i] * S[j][j]))))
                return false;
    return true;
}

// input checks at the trust boundary (tiers 1 and 3); nullptr if fine
const char* check_problem(const Problem& p) {
    if (p.m < 1 || p.m > MAXM) return LQOCP_STR("unsupported number of chains");
    int n = 0;
    for (int j = 0; j < p.m; ++j) {
        if (p.orders[j] < 1 || p.orders[j] > MAXK) return LQOCP_STR("unsupported chain order");
        n += p.orders[j];
    }
    if (n > MAXN) return LQOCP_STR("too many states");
    if (!(p.T > 0.0) || !isfinite(p.T)) return LQOCP_STR("T must be finite and > 0");
    for (int i = 0; i < n; ++i)
        if (!isfinite(p.x0[i])) return LQOCP_STR("x0 not finite");
    for (int i = 0; i < n; ++i)
        if (!isfinite(p.xf[i])) return LQOCP_STR("xf not finite");
    if (p.has_bounds) {
        for (int r = 0; r < p.m; ++r) {
            for (int j = 0; j < p.m; ++j)
                if (!isfinite(p.M[r][j])) return LQOCP_STR("Umap not finite");
            // -inf lo / +inf hi disable a side (documented); NaN, +inf lo, -inf hi are errors
            if (isnan(p.lo[r]) || isnan(p.hi[r]) || p.lo[r] == HUGE_VAL || p.hi[r] == -HUGE_VAL)
                return LQOCP_STR("bounds NaN or infinite on the wrong side");
            if (p.lo[r] > p.hi[r]) return LQOCP_STR("bounds: lo > hi");
        }
    }
    if (p.has_cost) {
        for (int i = 0; i < p.m; ++i)
            for (int j = 0; j < p.m; ++j) {
                if (!isfinite(p.R[i][j])) return LQOCP_STR("R not finite");
                if (fabs(p.R[i][j] - p.R[j][i]) > 1e-12 * (fabs(p.R[i][j]) + fabs(p.R[j][i])))
                    return LQOCP_STR("R not symmetric");       // (rounding-level asymmetry is accepted)
            }
        bool q0 = true;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                if (!isfinite(p.Q[i][j])) return LQOCP_STR("Q not finite");
                q0 = q0 && p.Q[i][j] == 0.0;
            }
        if (!q0 && !symmetric(p.Q, n)) return LQOCP_STR("Q not symmetric");
        if (!q0 && !psd_ok(p.Q, n)) return LQOCP_STR("Q not PSD");   // (Qf: by each solver's own factor)
        if (p.has_Qf) {
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    if (!isfinite(p.Qf[i][j])) return LQOCP_STR("Qf not finite");
            if (!symmetric(p.Qf, n)) return LQOCP_STR("Qf not symmetric");
        }
    }
    return nullptr;
}

// Box rows on vt (Mrow = M Lc^-T): rows[2r] = Mrow[r] <= hi_r, rows[2r+1] = -Mrow[r] <= -lo_r. Parallel rows
// (duplicate, opposing, multiple: row r = a row q) make the active-row Gram singular: row r's bounds tighten
// row q's, row r is dropped. rownorm2 = |rows|^2; R = 1 + the largest finite bound. nullptr if fine
template <class T>
const char* build_rows(const Problem& p, const T (*Mrow)[MAXM], T (*rows)[MAXM], T* rhs, bool* finite, T* rownorm2,
                       T* R) {
    const int m = p.m;
    for (int r = 0; r < m; ++r) {
        for (int j = 0; j < m; ++j) {
            rows[2 * r][j] = Mrow[r][j];
            rows[2 * r + 1][j] = -Mrow[r][j];
        }
        rhs[2 * r] = p.hi[r];
        rhs[2 * r + 1] = -p.lo[r];
        finite[2 * r] = p.has_bounds && isfinite(p.hi[r]);
        finite[2 * r + 1] = p.has_bounds && isfinite(p.lo[r]);
    }
    for (int r = 1; r < m; ++r)
        for (int q = 0; q < r && (finite[2 * r] || finite[2 * r + 1]); ++q) {
            T dot = 0.0, n2 = 0.0, dev = 0.0, rm = 0.0;
            for (int j = 0; j < m; ++j) { dot += Mrow[q][j] * Mrow[r][j]; n2 += Mrow[q][j] * Mrow[q][j]; }
            if (!(n2 > 0.0)) continue;
            const T a = dot / n2;
            for (int j = 0; j < m; ++j) { dev = fmax(dev, fabs(Mrow[r][j] - a * Mrow[q][j])); rm = fmax(rm, fabs(Mrow[r][j])); }
            if (!(dev <= 1e-12 * ts_of<T>() * rm) || a == 0.0) continue;
            const T hi = finite[a > 0.0 ? 2 * r : 2 * r + 1] ? (a > 0.0 ? rhs[2 * r] : -rhs[2 * r + 1]) / a : HUGE_VAL;
            const T lo = finite[a > 0.0 ? 2 * r + 1 : 2 * r] ? (a > 0.0 ? -rhs[2 * r + 1] : rhs[2 * r]) / a : -HUGE_VAL;
            if (isfinite(hi) && (!finite[2 * q] || hi < rhs[2 * q])) { rhs[2 * q] = hi; finite[2 * q] = true; }
            if (isfinite(lo) && (!finite[2 * q + 1] || -lo < rhs[2 * q + 1])) { rhs[2 * q + 1] = -lo; finite[2 * q + 1] = true; }
            if (finite[2 * q] && finite[2 * q + 1] && rhs[2 * q] < -rhs[2 * q + 1]) return LQOCP_STR("bounds: lo > hi");
            finite[2 * r] = finite[2 * r + 1] = false;
        }
    T mx = 0.0;
    for (int k = 0; k < 2 * m; ++k) {
        T s = 0.0;
        for (int j = 0; j < m; ++j) s += rows[k][j] * rows[k][j];
        rownorm2[k] = s;
        if (finite[k]) mx = fmax(mx, fabs(rhs[k]));
    }
    *R = 1.0 + mx;
    return nullptr;
}

// Vertices v = M^-1 u of the box {lo <= M v <= hi} on the merged rows: column k < 2^m of V (row stride 2^m + m;
// the last m columns get M^-1). false: unbounded (a side disabled, or a row dropped as parallel: the box is open
// along M's null direction), or M singular to rounding (kappa_inf(M) m EPS >= 1, LU's relative forward-error bound
// reaches 1: the vertices are undetermined, counted as unbounded like an exactly singular M)
template <class T>
bool box_vertices(const Problem& p, const T* rhs, const bool* finite, T* V) {
    const int m = p.m;
    for (int k = 0; k < 2 * m; ++k)
        if (!finite[k]) return false;
    T A[MAXM * MAXM];
    const int nv = 1 << m, ld = nv + m;
    T mn = 0.0, mi = 0.0;
    for (int r = 0; r < m; ++r) {
        T s = 0.0;
        for (int j = 0; j < m; ++j) {
            A[r * MAXM + j] = p.M[r][j];
            s += fabs(A[r * MAXM + j]);
            V[r * ld + nv + j] = r == j ? 1.0 : 0.0;
        }
        mn = fmax(mn, s);
        for (int k = 0; k < nv; ++k) V[r * ld + k] = (k >> r & 1) ? rhs[2 * r] : -rhs[2 * r + 1];
    }
    if (!lu_solve(A, m, MAXM, V, ld, ld)) return false;
    for (int r = 0; r < m; ++r) {
        T s = 0.0;
        for (int j = 0; j < m; ++j) s += fabs(V[r * ld + nv + j]);
        mi = fmax(mi, s);
    }
    return mn * mi * m * (sizeof(T) == sizeof(real) ? EPS : DBL_EPSILON) < 1.0;
}

// sig_i = max(sig_i, reach of state i under the box: umax_j T^e / e!, e = order - derivative index), V from box_vertices
template <class T>
void box_reach(const Problem& p, const T* V, T* sig) {
    const int nv = 1 << p.m, ld = nv + p.m;
    for (int j = 0, o = 0; j < p.m; o += p.orders[j++]) {
        T um = 0.0;
        for (int k = 0; k < nv; ++k) um = fmax(um, fabs(V[j * ld + k]));
        for (int q = 0; q < p.orders[j]; ++q) {
            const int e = p.orders[j] - q;
            sig[o + q] = fmax(sig[o + q], um * ipow(p.T, e) / fact(e));
        }
    }
}

// Projection on the face of the active rows S (m inputs): P = I - A^T G^-1 A, q = A^T G^-1 b with G = A A^T;
// X (row stride MAXM + 1) = G^-1 [A | b] for the multipliers, idx/na the active rows. zfull: P = 0 exactly when
// every input is fixed (na == m). false (P = I, q = 0) on more than MAXM rows or a singular Gram
template <class T>
bool project(const T (*rows)[MAXM], const T* rhs, int m, uint16_t S, bool zfull, T P[MAXM][MAXM], T q[MAXM],
             T* X, int* idx, int* nap) {
    int na = 0;
    *nap = 0;
    for (int i = 0; i < m; ++i) {
        q[i] = 0.0;
        for (int j = 0; j < m; ++j) P[i][j] = (i == j) ? 1.0 : 0.0;
    }
    for (int k = 0; k < 2 * m; ++k)
        if (S & (1u << k)) { if (na >= MAXM) return false; idx[na++] = k; }
    if (na == 0) return true;
    T G[MAXM * MAXM];
    const int ldx = MAXM + 1;
    for (int a = 0; a < na; ++a) {
        for (int b = 0; b < na; ++b) {
            T s = 0.0;
            for (int j = 0; j < m; ++j) s += rows[idx[a]][j] * rows[idx[b]][j];
            G[a * MAXM + b] = s;
        }
        for (int j = 0; j < m; ++j) X[a * ldx + j] = rows[idx[a]][j];
        X[a * ldx + m] = rhs[idx[a]];
    }
    if (!lu_solve(G, na, MAXM, X, m + 1, ldx)) return false;
    for (int i = 0; i < m; ++i)
        for (int a = 0; a < na; ++a) {
            const T ai = rows[idx[a]][i];
            for (int j = 0; j < m; ++j) P[i][j] = (zfull && na == m) ? 0.0 : P[i][j] - ai * X[a * ldx + j];
            q[i] += ai * X[a * ldx + m];
        }
    *nap = na;
    return true;
}
}  // namespace pd

bool Solver::setup(const Problem& p) {
    sfail_ = pd::check_problem(p);
    return !sfail_ && setup_(p);
}

bool Solver::setup_(const Problem& p) {
    p_ = &p;
    m_ = p.m;
    n_ = 0;
    for (int j = 0; j < m_; ++j) {
        off_[j] = n_;
        n_ += p.orders[j];
    }
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
            if (!(d > 0.0)) { sfail_ = LQOCP_STR("R not SPD"); return false; }
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
    // Qf = Cf^T Cf (scale-free semidefinite factor, psd_factor)
    for (int i = 0; i < MAXN * MAXN; ++i) Cf_[i] = 0.0;
    if (qf_ && !psd_factor(p.Qf, n_, Cf_)) { sfail_ = LQOCP_STR("Qf not PSD"); return false; }
    if ((sfail_ = build_rows(p, Mrow, rows_, rhs_, finite_, rownorm2_, &R_))) return false;
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
    // jmax_: 1/2 v^T R v is convex, so its max over the box {lo <= M v <= hi} is at a vertex v = M^-1 u
    jmax_ = HUGE_VAL;
    for (int i = 0; i < n_; ++i) sig_[i] = fabs((real)p.xf[i]);
    real V[MAXM * ((1 << MAXM) + MAXM)];
    if (box_vertices(p, rhs_, finite_, V)) {
        const int nv = 1 << m_, ld = nv + m_;
        jmax_ = 0.0;
        for (int k = 0; k < nv; ++k) {
            real v[MAXM], q = 0.0;
            for (int j = 0; j < m_; ++j) v[j] = V[j * ld + k];
            for (int j = 0; j < m_; ++j) q += 0.5 * v[j] * (rI_ ? v[j] : vR(j, v));
            jmax_ = fmax(jmax_, q);
        }
        box_reach(p, V, sig_);
    }
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
        if (!lu_solve(A, n_, MAXN, Linv_, n_, MAXN)) { sfail_ = LQOCP_STR("internal: Lm singular"); return false; }
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
    int idx[MAXM], na;
    real X[MAXM * (MAXM + 1)];
    return project(rows_, rhs_, m_, S, false, P, q, X, idx, &na);
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
        for (int i = 0; i < n_; ++i) {   // 1/2 |Cf e|^2: e^T Qf e cancels to below 0 for an ill-conditioned Qf
            real s = 0.0;
            for (int j = i; j < n_; ++j) s += Cf_[i * MAXN + j] * e[j];
            J += 0.5 * s * s;
        }
    }
    return J;
}

static const char* const CAPFAIL = "arc capacity LQOCP_MAXARC_E exceeded";
static const char* const INFEAS = "infeasible endpoint (no feasible input)";

bool Solver::solve(Result& R) { return start(R, -1); }
bool Solver::solve_warm(Result& R, int max_its, double dt) {
    if (!isfinite(dt)) { R.converged = false; R.failure = LQOCP_STR("dt not finite"); return false; }
    if (dt != 0.0 && !general_) shift_warm(R, (real)dt);
    return start(R, max_its);
}

bool Solver::start(Result& R, int warm_its) {
    R.converged = false;
    R.failure = LQOCP_STR("general cost: use GSolver");
    if (general_) return false;                       // general cost: use GSolver
    R.backtracks = 0;
    R.degenerate_events = 0;
    R.box_fast_path = false;
    R.reduced = false;
    capfail_ = false;
    return solve_newton(R, warm_its);
}

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
// Energy cost: Newton on c with exact (grid-free) event detection. Mirrors
// develop/python/lqocp.py (_NewtonCtx / _solve_newton).
// =====================================================================================
namespace pd {

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

}  // namespace pd

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

// ev: rows of a walk event, zero at t by construction (as in the tier-2 walk): their sign is that of the derivatives
bool Solver::lex_ok(uint16_t S, const real* c, real t, uint16_t ev, real phi[2 * MAXM][4], bool has[2 * MAXM]) const {
    real b[4];
    row_polys(S, c, phi, has);
    const real T = (real)p_->T;
    for (int key = 0; key < 2 * m_; ++key) {
        if (!has[key]) continue;
        shift(phi[key], 4, t, b);
        if ((ev >> key) & 1u) b[0] = 0.0;
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

bool Solver::lex_set(const real* c, real t, uint16_t ev, uint16_t guess, uint16_t* out, real phi[2 * MAXM][4],
                     bool has[2 * MAXM]) const {
    if (guess != 0xFFFFu && valid_set(guess) && lex_ok(guess, c, t, ev, phi, has)) { *out = guess; return true; }
    for (int i = 0; i < nsets_; ++i)
        if (lex_ok(sets_[i], c, t, ev, phi, has)) { *out = sets_[i]; return true; }
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
    if (!lex_set(c, 0.0, 0, 0, &S, phi, has)) return false;   // (a) initial set: one-sided KKT at 0+
    int ns = 0, stall = 0;
    e.S[ns++] = S;
    real t = 0.0;
    uint16_t zev = 0;                                // rows of the last event: zero at t by construction
    for (;;) {
        const real L = T - t;
        row_polys(S, c, phi, has);
        real rootk[2 * MAXM];
        bool simplek[2 * MAXM], evk[2 * MAXM];
        real s1 = HUGE_VAL;
        uint16_t imm = 0;                            // rows already beyond tolerance at t: not zero there
        for (int key = 0; key < 2 * m_; ++key) {
            evk[key] = false;
            if (!has[key]) continue;
            real b[4], red[4];
            int k;
            shift(phi[key], 4, t, b);
            // else a new multiplier's root error, amplified by nearly dependent rows (B near T*), stalls the walk
            if ((zev >> key) & 1u) b[0] = 0.0;
            const real tol = 1e-12 * TS * (pscale(b, 4, T) + R_);   // as lex_ok: no stall on a tiny last arc
            const int rl = strip(b, 4, L, tol, red, &k);
            if (rl == 0) continue;
            if (red[0] > 0) {                            // inconsistent at t+: degenerate event
                evk[key] = true; rootk[key] = 0.0; simplek[key] = false;
                s1 = 0.0;
                if (!k) imm |= (uint16_t)(1u << key);
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
        t += s1;
        if (!walk_next(
                2 * m_,
                [&](int q, int* key, real* root, bool* simple) {
                    if (!evk[q]) return false;
                    *key = q; *root = rootk[q]; *simple = simplek[q];
                    return true;
                },
                s1, 1e-12 * TS * T, imm, [&](uint16_t cand) { return valid_set(cand); },
                [&](uint16_t guess, uint16_t zv, uint16_t* out) {   // (b) degenerate event
                    ++e.ndeg;
                    return lex_set(c, t, zv, valid_set(guess) ? guess : 0xFFFFu, out, phi, has);
                },
                &zev, S, e.S, e.tau, t, ns, MAXARC_E, &capfail_, stall))
            return false;
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

real Solver::dual(const uint16_t* S, const real* tau, int n, const real* c, const real* mu, real* dabs) const {
    real val = 0.0, ab = 0.0;
    for (int i = 0; i < n_ && qf_; ++i) val += mu[i] * (g_[i] - 0.5 * mu[i]);
    for (int i = 0; i < n_ && !qf_; ++i) {
        real nu = 0.0, na = 0.0;
        for (int j = 0; j < n_; ++j) { nu += Linv_[i * MAXN + j] * c[j]; na += fabs(Linv_[i * MAXN + j] * c[j]); }
        val += nu * ((real)p_->xf[i] - free_[i]);
        ab += na * fabs((real)p_->xf[i] - free_[i]);
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
            real s = 0.0, sa = 0.0;
            for (int j = 0; j < m_; ++j) {
                const real e = 0.5 * v[j] * (rI_ ? v[j] : vR(j, v));
                s += e - w[j] * v[j];
                sa += fabs(e) + fabs(w[j] * v[j]);
            }
            val += GL_W[g] * hw * s;
            ab += GL_W[g] * hw * sa;
        }
    }
    if (dabs) *dabs = ab;
    return val;
}

// Weak duality: a feasible problem has D <= J* <= T jmax_. D above that by more than its rounding (a sum of
// n_ + 4 narcs (2 m_ + 1) terms of total size Dabs) proves that no feasible input exists (exact, no threshold).
bool Solver::infeasible(real D, real Dabs, int narcs) const {
    const real J = (real)p_->T * jmax_;
    return D - J > (n_ + 4 * narcs * (2 * m_ + 1) + 2) * EPS * (Dabs + J);
}

bool Solver::evaluate(const real* c, const real* mu, Eval& e) {
    if (!walk(c, e)) return false;
    real h[MAXN];
    assemble(e.S, e.tau, e.narcs, G_, h);
    real sc = 1.0, si[MAXN];
    for (int i = 0; i < n_; ++i) {
        real gc = 0.0, s = 0.0;
        for (int j = 0; j < n_; ++j) { gc += G_[i * MAXN + j] * c[j]; s += fabs(G_[i * MAXN + j] * c[j]); }
        e.F[i] = gc + h[i] - (real)p_->xf[i];
        si[i] = fmax(fabs((real)p_->xf[i]), fmax(fabs(free_[i]), s + fabs(h[i] - free_[i])));
        sc = fmax(sc, si[i]);
    }
    if (qf_) {                                       // transversality: F = Cf (s(T) - xf) + mu
        // stop per row, |F_i| <= rtol a_i with a_i its own terms (state i's scale): a global max would let rows in
        // small units (per-chain scaling) stop early. e.scale = Fn / max_i |F_i| / a_i makes solve_newton's test that
        real r[MAXN], Fn = 0.0, rel = 0.0;
        for (int i = 0; i < n_; ++i) r[i] = e.F[i];
        for (int i = 0; i < n_; ++i) {
            real f = mu[i], a = fabs(mu[i]);
            for (int j = i; j < n_; ++j) { f += Cf_[i * MAXN + j] * r[j]; a += fabs(Cf_[i * MAXN + j]) * si[j]; }
            e.F[i] = f;
            Fn = fmax(Fn, fabs(f));
            if (a > 0.0) rel = fmax(rel, fabs(f) / a);
        }
        sc = rel > 0.0 ? Fn / rel : 1.0;
    }
    e.scale = sc;
    e.D = dual(e.S, e.tau, e.narcs, c, mu, &e.Dabs);
    return true;
}

real Solver::verify(const uint16_t* S, const real* tau, int n, const real* c, real* amp) const {
    real worst = 0.0, box = 0.0, a_max = 0.0, phi[2 * MAXM][4], b[4], d[3], cr[2];
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
            // row of the event at t0: zero there for the PMP test (its value is the root's rounding, taken off the
            // arc), but a box row's value is the plan's input: the box limit sees it back (off)
            const real off = a && ((S[a] ^ S[a - 1]) >> key & 1u) ? fmax(b[0], (real)0.0) : 0.0;
            b[0] -= off;
            real w = fmax(b[0], pval(b, 4, L));
            for (int i = 1; i < 4; ++i) d[i - 1] = i * b[i];
            const int nc = roots_le2(d, 3, cr);
            for (int i = 0; i < nc; ++i)
                if (cr[i] > 0.0 && cr[i] < L) w = fmax(w, pval(b, 4, cr[i]));
            worst = fmax(worst, w);
            if (!(S[a] & (1u << key))) box = fmax(box, w + off);
        }
    }
    if (amp) *amp = a_max;
    // a box row's value is in input units: its violation may not grow with amp (cancelling huge polynomial terms,
    // e.g. tiny R: 1e-5 accepted); above 1e-7 R_ (float 1e-4 R_) the plan is rejected whatever vtol(amp) is.
    // Double: the plan's input is known only to EPS amp (one rounding of its largest term), charged to the box too;
    // this bounds amp by the box acceptance. Float: not charged, its 1e-4 R_ is below
    // FLT_EPS amp for amp > 840 R_ (would reject 27% of the fscan float accepts)
    const real rnd = sizeof(real) == 4 ? 0.0 : EPS * a_max;
    return box + rnd > BOXTOL * R_ ? HUGE_VAL : worst;
}

// c(t) -> c(t + dt) per chain (exact Taylor shift); y = Lm^-1 c (nu), with Qf the mu of nu = Cf^T mu (Cf upper
// triangular; a zero pivot of a semidefinite Qf leaves that mu at 0: only a starting guess)
void Solver::shift_dual(real* c, real* y, real dt) const {
    real b[MAXK], nu[MAXN];
    for (int j = 0; j < m_; ++j) {
        shift(c + off_[j], p_->orders[j], dt, b);
        for (int k = 0; k < p_->orders[j]; ++k) c[off_[j] + k] = b[k];
    }
    for (int i = 0; i < n_; ++i) {
        real s = 0.0;
        for (int j = 0; j < n_; ++j) s += Linv_[i * MAXN + j] * c[j];
        nu[i] = s;
    }
    for (int i = 0; i < n_; ++i) {
        if (!qf_) { y[i] = nu[i]; continue; }
        real s = nu[i];
        for (int j = 0; j < i; ++j) s -= Cf_[j * MAXN + i] * y[j];
        y[i] = Cf_[i * MAXN + i] > 0.0 ? s / Cf_[i * MAXN + i] : 0.0;
    }
}

// Time shift of a warm start (solve_warm): the pattern loses the arcs that ended by dt, its switch times move
// by -dt (the last arc reaches T), and c / mu are Taylor-shifted (natural extension of w on [T - dt, T]).
void Solver::shift_warm(Result& R, real dt) {
    real nu[MAXN];
    shift_dual(R.c, qf_ ? R.mu : nu, dt);
    int k0 = 0;
    while (k0 + 1 < R.narcs && R.tau[k0] - dt <= 1e-9 * (real)p_->T) ++k0;
    R.narcs -= k0;
    for (int a = 0; a < R.narcs; ++a) R.S[a] = R.S[a + k0];
    for (int a = 0; a + 1 < R.narcs; ++a) R.tau[a] = R.tau[a + k0] - dt;
}

// warm_its < 0: cold start (unconstrained solution; mu = 0 with Qf), else from R.c / R.mu.
bool Solver::solve_newton(Result& R, int warm_its) {
    R.iterations = 0;                             // warm: first the previous pattern
    if (warm_its >= 0 && try_reduced(R, warm_its < 4 ? warm_its : 4)) return true;
    R.newton_iterations = R.outer_iterations = 0;
    R.residual = HUGE_VAL;
    R.pmp_violation = HUGE_VAL;
    R.box_fast_path = identity_;
    R.narcs = 0;
    R.failure = LQOCP_STR("unconstrained start failed");
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
    const char* why = LQOCP_STR("event walk failed");
    int cur = 0;
    bool ok = evaluate(c, mu, ew_.ev[cur]);
    bool conv = false;
    // float: stop at the round-off level of F's terms (scale); double: 1e-12 of it
    const real rtol = sizeof(real) == 4 ? EPS : 1e-12;
    int its = 0;
    real pv = HUGE_VAL, amp = 0.0, Fst = HUGE_VAL;
    while (ok) {
        Eval& E = ew_.ev[cur];
        real Fn = 0.0, F2 = 0.0;
        for (int i = 0; i < n_; ++i) { Fn = fmax(Fn, fabs(E.F[i])); F2 += E.F[i] * E.F[i]; }
        // dual unbounded (infeasible problem): the iterates c diverge. Tested before the stop: E.scale
        // grows with |c|, so a diverging iterate can meet the relative stop with an O(1) endpoint miss.
        if (!qf_ && infeasible(E.D, E.Dabs, E.narcs)) { why = INFEAS; break; }
        bool st = true;
        if (sizeof(real) == 4 || qf_) {
            st = Fn <= rtol * E.scale;
        } else {   // double, fixed endpoint: per state, relative to its scale sigma_i or its rounding floor
            for (int i = 0; i < n_ && st; ++i) {
                real gc = 0.0, a = 0.0;   // a = sum |terms of F_i| (G c, the constant part, xf): F_i's rounding is (n+2) EPS a
                for (int j = 0; j < n_; ++j) { gc += G_[i * MAXN + j] * c[j]; a += fabs(G_[i * MAXN + j] * c[j]); }
                a += fabs(E.F[i] - gc + (real)p_->xf[i]) + fabs((real)p_->xf[i]);
                st = fabs(E.F[i]) <= fmax(rtol * fmax(sig_[i], fabs(free_[i])), (n_ + 2) * EPS * a);
            }
        }
        if (st) {   // verify's box limit can need a plan beyond the stop: Newton goes on while |F| still falls
            pv = verify(E.S, E.tau, E.narcs, c, &amp);
            if (pv <= vtol(amp) || !(Fn < Fst) || its >= max_its) { conv = true; break; }
            Fst = Fn;
        }
        if (its >= max_its) { why = LQOCP_STR("iteration cap"); break; }
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
        if (!solved) { why = LQOCP_STR("singular Newton matrix"); break; }
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
        if (!accepted) { why = LQOCP_STR("line search failed"); break; }
        for (int i = 0; i < n_; ++i) { c[i] = cn[i]; if (qf_) mu[i] = mun[i]; }
        cur = nxt;
    }
    R.newton_iterations = its;
    R.iterations += its;                          // plus a failed reduced attempt's iterations
    if (!conv && !qf_ && !capfail_ && ok && infeasible(ew_.ev[cur].D, ew_.ev[cur].Dabs, ew_.ev[cur].narcs))
        why = INFEAS;                                 // walk / line search failed on a diverging dual
    if (!ok) { R.failure = capfail_ ? CAPFAIL : why; return false; }
    const Eval& E = ew_.ev[cur];
    R.narcs = E.narcs;
    for (int i = 0; i < E.narcs; ++i) R.S[i] = E.S[i];
    for (int i = 0; i + 1 < E.narcs; ++i) R.tau[i] = E.tau[i];
    real Fn = 0.0;
    for (int i = 0; i < n_; ++i) Fn = fmax(Fn, fabs(E.F[i]));
    R.residual = Fn;
    R.degenerate_events = E.ndeg;
    R.pmp_violation = conv ? pv : verify(R.S, R.tau, R.narcs, R.c, &amp);   // conv: verified at the stop
    if (conv && !(R.pmp_violation <= vtol(amp))) { conv = false; why = LQOCP_STR("PMP verification failed"); }   // NaN fails
    R.cost = cost(R.S, R.tau, R.narcs, R.c);
    R.converged = conv;
    R.failure = conv ? nullptr : capfail_ ? CAPFAIL : why;
    return conv;
}

// the walk's own event tolerance (the Newton stop's relative accuracy times the row amplitude): the walk ignores
// sign changes below it, so it is the accuracy the plan's rows have. Box rows are also held to 1e-7 R_ in verify.
real Solver::vtol(real amp) const { return 1e-12 * TS * (amp + R_); }

// Switch between arcs SL and SR (one row toggles): the row's gap on the side where it is
// inactive, g(t) = e^T w(t) + g0. Control continuity at the switch <=> g = 0. gd = dg/dt,
// gs = magnitude of g's terms (tolerance scale).
real Solver::gapk(uint16_t SL, uint16_t SR, real t, const real* c, real* e, real* gd, real* gs) const {
    const uint16_t X = (uint16_t)(SL ^ SR);
    int key = 0;
    while (!(X & (1u << key))) ++key;
    real P[MAXM][MAXM], q[MAXM], rho[MAXM], w[MAXM];
    pq((SR & X) ? SL : SR, P, q);
    for (int i = 0; i < m_; ++i) {                  // rows act on vt = Lc^T v
        real s = rI_ ? rows_[key][i] : 0.0;
        if (!rI_) for (int j = 0; j <= i; ++j) s += Lc_[i][j] * rows_[key][j];
        rho[i] = s;
    }
    w_of(c, t, w);
    real g = -rhs_[key], a = fabs(rhs_[key]), d = 0.0;
    for (int i = 0; i < m_; ++i) { g += rho[i] * q[i]; a += fabs(rho[i] * q[i]); }
    for (int j = 0; j < m_; ++j) {
        real ej = 0.0, wd = 0.0, tp = 1.0;
        for (int i = 0; i < m_; ++i) ej += rho[i] * P[i][j];
        for (int p = 1; p < p_->orders[j]; ++p) { wd += p * c[off_[j] + p] * tp; tp *= t; }
        e[j] = ej;
        g += ej * w[j];
        a += fabs(ej * w[j]);
        d += ej * wd;
    }
    *gd = d;
    *gs = a;
    return g;
}

// Warm step on the previous pattern: accepted if the reduced Newton converges and verify passes
// (then R is the solution); else R.iterations = the reduced iterations spent, R otherwise unchanged.
bool Solver::try_reduced(Result& R, int max_its) {
    if (!solve_reduced(R, max_its, &R.iterations)) return false;
    real amp;
    const real viol = verify(R.S, ew_.ev[1].tau, R.narcs, ew_.ev[1].F, &amp);
    if (!(viol <= vtol(amp))) return false;
    if (!qf_) {                                   // verified: the arcs are v(c), so D is the dual value (as solve_newton)
        real Dabs;
        const real D = dual(R.S, ew_.ev[1].tau, R.narcs, ew_.ev[1].F, nullptr, &Dabs);
        if (infeasible(D, Dabs, R.narcs)) return false;
    }
    for (int i = 0; i < n_; ++i) { R.c[i] = ew_.ev[1].F[i]; if (qf_) R.mu[i] = ew_.ev[0].F[i]; }
    for (int k = 0; k + 1 < R.narcs; ++k) R.tau[k] = ew_.ev[1].tau[k];
    R.newton_iterations = R.outer_iterations = R.degenerate_events = 0;
    R.box_fast_path = identity_;
    R.residual = ew_.ev[1].D;
    R.pmp_violation = viol;
    R.cost = cost(R.S, R.tau, R.narcs, R.c);
    R.converged = R.reduced = true;
    R.failure = nullptr;
    return true;
}

// Switch at t between arcs SL and SR: its tau-elimination term in the reduced Newton system,
// W_ -= qk a^T and d += qk g, with qk = Qh ds(T)/dtau / gd, ds(T)/dtau = K(T - t)(v_SL - v_SR) and
// a = dg/dc (entries e_j t^p).
void Solver::swjac(uint16_t SL, uint16_t SR, real t, const real* c, real g, real gd, const real* e,
                   real* d) const {
    real dv[MAXM];
    {                                                 // own scope: stack shared with qk
        real P[MAXM][MAXM], q[MAXM], vr[MAXM];
        pq(SL, P, q);
        v_on(P, q, c, t, dv);
        pq(SR, P, q);
        v_on(P, q, c, t, vr);
        for (int l = 0; l < m_; ++l) dv[l] -= vr[l];
    }
    const real T = (real)p_->T;
    real qk[MAXN];
    for (int i = 0; i < n_; ++i) qk[i] = 0.0;
    for (int l = 0; l < m_; ++l) {
        const int ko = p_->orders[l];
        for (int r = 0; r < ko; ++r) {
            const real ks = ipow(T - t, ko - 1 - r) / fact(ko - 1 - r) * dv[l];
            const int col = off_[l] + r;
            if (!qf_) qk[col] += ks;
            else for (int i = 0; i < n_; ++i) qk[i] += (real)p_->Qf[i][col] * ks;
        }
    }
    for (int i = 0; i < n_; ++i) { qk[i] /= gd; d[i] += qk[i] * g; }
    for (int j = 0; j < m_; ++j) {
        real tp = 1.0;
        for (int p = 0; p < p_->orders[j]; ++p, tp *= t)
            for (int i = 0; i < n_; ++i) W_[i * MAXN + off_[j] + p] -= qk[i] * e[j] * tp;
    }
}

// MPC warm step on the previous plan's pattern R.S (arXiv 2211.05251 style): the unknowns are the
// switch times; the conditions are the terminal conditions (linear in c) and control continuity
// at each switch (gap of the toggled row = 0). Newton on (c, tau) with tau eliminated per switch:
//   [L0 + Qh G - sum_k Qh ks_k a_k^T / gd_k] dc = -E1 + sum_k Qh ks_k g_k / gd_k,
//   dtau_k = -(g_k + a_k^T dc) / gd_k,
// with E1 = L0 c + Qh (s(T) - xf) (L0 = Lm^-1, Qh = Qf; fixed endpoint: L0 = 0, Qh = I),
// ks_k = ds(T)/dtau_k, a_k = dg_k/dc. No walk. On convergence returns true with the candidate in
// scratch (c: ev[1].F, tau: ev[1].tau, mu: ev[0].F, residual: ev[1].D); the caller accepts it only
// if verify passes (exact row maxima per arc: gaps <= 0, multipliers >= 0). R is not modified.
bool Solver::solve_reduced(Result& R, int max_its, int* its) {
    const int n = R.narcs, nj = n - 1;
    *its = 0;
    if (n < 1 || max_its < 1) return false;
    for (int k = 0; k < nj; ++k) {
        const uint16_t X = (uint16_t)(R.S[k] ^ R.S[k + 1]);
        if (!X || (X & (X - 1))) return false;         // a simultaneous switch: walk it
    }
    const real T = (real)p_->T, gap = 1e-9 * T;
    const real rtol = sizeof(real) == 4 ? EPS : 1e-12;
    // scratch: trial c and tau in ev[1]; h in ev[0].F; per switch g_k/gd_k in ev[0].tau and
    // e_k/gd_k in G_ (free once the Newton matrix is built)
    real* const tau = ew_.ev[1].tau, * const c = ew_.ev[1].F, * const h = ew_.ev[0].F;
    real* const sk = ew_.ev[0].tau, * const ek = G_;
    for (int k = 0; k < nj; ++k) tau[k] = R.tau[k];
    for (int i = 0; i < n_; ++i) c[i] = R.c[i];
    real d[MAXN], E1[MAXN], e[MAXM], gd, gs;
    bool fresh = false;                               // G_ and h hold ds(T)/dc and s(T) - G c at tau
    for (;;) {
        if (!fresh) assemble(R.S, tau, n, G_, h);
        fresh = nj == 0;                              // tau fixed and G_ kept only without switches
        real sc = 1.0, sc1 = 1.0, En = 0.0, Gn = 0.0;
        for (int i = 0; i < n_; ++i) {                // d = xf - s(T)
            real gc = 0.0, a = 0.0;
            for (int j = 0; j < n_; ++j) { gc += G_[i * MAXN + j] * c[j]; a += fabs(G_[i * MAXN + j] * c[j]); }
            d[i] = (real)p_->xf[i] - gc - h[i];
            sc = fmax(sc, fmax(fabs((real)p_->xf[i]), fabs(free_[i])));
            sc = fmax(sc, a + fabs(h[i] - free_[i]));
        }
        for (int i = 0; i < n_; ++i) {
            real f = qf_ ? 0.0 : -d[i], a = qf_ ? 0.0 : sc;
            for (int j = 0; j < n_ && qf_; ++j) {
                f += Linv_[i * MAXN + j] * c[j] - (real)p_->Qf[i][j] * d[j];
                a += fabs(Linv_[i * MAXN + j] * c[j]) + fabs((real)p_->Qf[i][j]) * sc;
            }
            E1[i] = f;
            En = fmax(En, fabs(f));
            sc1 = fmax(sc1, a);
        }
        bool conv = En <= rtol * sc1;
        for (int k = 0; k < nj && conv; ++k) {
            const real g = gapk(R.S[k], R.S[k + 1], tau[k], c, e, &gd, &gs);
            Gn = fmax(Gn, fabs(g));
            conv = fabs(g) <= 1e-12 * TS * (gs + R_);
        }
        if (conv) {
            for (int i = 0; i < n_ && qf_; ++i) {       // dual iterate mu = Cf (xf - s(T)) (h is dead)
                real m = 0.0;
                for (int j = i; j < n_; ++j) m += Cf_[i * MAXN + j] * d[j];
                h[i] = m;
            }
            ew_.ev[1].D = fmax(En, Gn);
            return true;
        }
        if (*its >= max_its) return false;
        ++*its;
        for (int i = 0; i < n_; ++i) {                // Newton matrix L0 + Qh G in W_ (then G_ is free)
            for (int j = 0; j < n_; ++j) {
                real t = qf_ ? Linv_[i * MAXN + j] : G_[i * MAXN + j];
                for (int k = 0; k < n_ && qf_; ++k) t += (real)p_->Qf[i][k] * G_[k * MAXN + j];
                W_[i * MAXN + j] = t;
            }
            d[i] = -E1[i];
        }
        for (int k = 0; k < nj; ++k) {
            const real g = gapk(R.S[k], R.S[k + 1], tau[k], c, e, &gd, &gs);
            if (!(fabs(gd) * T > 1e-12 * TS * (gs + R_))) return false;   // tangential switch
            swjac(R.S[k], R.S[k + 1], tau[k], c, g, gd, e, d);
            for (int j = 0; j < m_; ++j) ek[k * MAXM + j] = e[j] / gd;
            sk[k] = g / gd;
        }
        if (!lu_solve(W_, n_, MAXN, d, 1, 1)) return false;
        for (int k = 0; k < nj; ++k) {
            real ad = 0.0;
            for (int j = 0; j < m_; ++j) {
                real tp = 1.0, wj = 0.0;
                for (int p = 0; p < p_->orders[j]; ++p, tp *= tau[k]) wj += d[off_[j] + p] * tp;
                ad += ek[k * MAXM + j] * wj;
            }
            tau[k] -= sk[k] + ad;
        }
        for (int k = 0; k <= nj; ++k) {                // arcs keep a positive length: else the pattern changed
            const real lo = k ? tau[k - 1] : 0.0, hi = k < nj ? tau[k] : T;
            if (!(hi - lo > gap)) return false;
        }
        for (int i = 0; i < n_; ++i) c[i] += d[i];
    }
}

// =====================================================================================
// General LQ cost: GSolver. R-scaled inputs vt = Lc^T v,
// Hamiltonian state z = (s, lambda). Arc solution in the set's basis W (modal stable/unstable
// blocks, centre block Tc): z(t) = W y(t),
//   y_s = e^{Ls (t - t0)} a + yp_s,  [y_c; 1] = e^{[Tc yp_c; 0 0](t - t0)} [c; 1],
//   y_u = e^{Lu (t - t1)} b + yp_u.
// The linear BVP for fixed junction times is a forward sweep of a 12-row relation
// C z(tau_k) = d_k plus a backward recovery of the arc coefficients (a, c, b); the relations
// are stored at NCK checkpoints and recomputed in between.
// =====================================================================================
namespace {

const double CHORD = 0.1;           // Newton: reuse the factored J while the residual contracts by this
const double GTOL = 1e-12;           // junction residual tolerance (relative to R)
const double STOL = 1e-9;            // walked vs Newton junction times (relative to T)
const double WRESET = 0.3;          // warm start: a capped plan violating PMP by > WRESET*R restarts cold

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

// rows[0..nr) <- rows M (M row-major n x n), in place. Four rows per pass over M: each element of M
// is read once per four rows (M may live in flash, slow on a Teensy 4.1; read row by row).
void relmul(double (*rows)[NCOL], int nr, const double* M, int n) {
    for (int r0 = 0; r0 < nr; r0 += 4) {
        double t[4][N2];
        const int k = (nr - r0 < 4) ? nr - r0 : 4;
        for (int q = 0; q < 4; ++q)
            for (int j = 0; j < n; ++j) t[q][j] = 0.0;
        for (int i = 0; i < n; ++i) {
            const double* Mi = M + i * n;
            const double a0 = rows[r0][i], a1 = (k > 1) ? rows[r0 + 1][i] : 0.0,
                         a2 = (k > 2) ? rows[r0 + 2][i] : 0.0, a3 = (k > 3) ? rows[r0 + 3][i] : 0.0;
            for (int j = 0; j < n; ++j) {
                const double m = Mi[j];
                t[0][j] += a0 * m; t[1][j] += a1 * m; t[2][j] += a2 * m; t[3][j] += a3 * m;
            }
        }
        for (int q = 0; q < k; ++q)
            for (int j = 0; j < n; ++j) rows[r0 + q][j] = t[q][j];
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
    mt_.b = nullptr;
    setup_failure = fn ? check_problem(p) : LQOCP_STR("no BasisFn");
    if (setup_failure) return false;
    p_ = &p;
    bfn_ = fn;
    bctx_ = ctx;
    m_ = p.m;
    n_ = 0;
    for (int j = 0; j < m_; ++j) {
        off_[j] = n_;
        n_ += p.orders[j];
        end_[j] = n_ - 1;
    }
    setup_failure = LQOCP_STR("non-finite problem data");        // (check_problem skips R, Q, Qf with has_cost = 0, M without bounds)
    bool fin = true;
    for (int i = 0; i < n_; ++i)
        for (int j = 0; j < n_; ++j) fin = fin && isfinite(p.Q[i][j]) && (!p.has_Qf || isfinite(p.Qf[i][j]));
    for (int r = 0; r < m_; ++r)
        for (int j = 0; j < m_; ++j) fin = fin && isfinite(p.R[r][j]) && isfinite(p.M[r][j]);
    if (!fin) return false;
    setup_failure = LQOCP_STR("Qf not PSD");
    if (p.has_Qf && !psd_ok(p.Qf, n_)) return false;
    setup_failure = LQOCP_STR("R not SPD");
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
    double Mrow[MAXM][MAXM];                           // rows on vt: M Lc^-T
    for (int r = 0; r < m_; ++r)
        for (int k = 0; k < m_; ++k) {
            double s = 0.0;
            for (int j = 0; j < m_; ++j) s += p.M[r][j] * LiT_[j][k];
            Mrow[r][k] = s;
        }
    if ((setup_failure = build_rows(p, Mrow, rows_, rhs_, finite_, rownorm2_, &R_))) return false;
    for (int k = 0; k < 2 * m_; ++k) rhs0_[k] = rhs_[k];
    for (int i = 0; i < n_; ++i) sig_[i] = fabs(p.xf[i]);
    double V[MAXM * ((1 << MAXM) + MAXM)];
    if (box_vertices(p, rhs_, finite_, V)) box_reach(p, V, sig_);
    // clampu inverts M only when it is regular to rounding: kappa_inf(M) m EPS < 1 (box_vertices' rule)
    double A[MAXM * MAXM], Mi[MAXM * MAXM], mn = 0.0, mi = 0.0;
    for (int r = 0; r < m_; ++r) {
        double s = 0.0;
        for (int j = 0; j < m_; ++j) { A[r * MAXM + j] = p.M[r][j]; Mi[r * MAXM + j] = r == j; s += fabs(p.M[r][j]); }
        mn = fmax(mn, s);
    }
    mreg_ = p.has_bounds && lu_solve(A, m_, MAXM, Mi, m_, MAXM);
    for (int r = 0; mreg_ && r < m_; ++r) {
        double s = 0.0;
        for (int j = 0; j < m_; ++j) s += fabs(Mi[r * MAXM + j]);
        mi = fmax(mi, s);
    }
    mreg_ = mreg_ && mn * mi * m_ * DBL_EPSILON < 1.0;
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
    int idx[MAXM], na;
    double X[MAXM * (MAXM + 1)];
    return project(rows_, rhs_, m_, S, true, P, q, X, idx, &na);
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
    int lo = 0;                                        // w[0..lo) = 0 (w N^k moves down the staircase)
    for (int k = 0; k < ni; ++k) {
        while (lo < nz && w[lo] == 0.0) ++lo;
        double wg = 0.0;
        for (int i = lo; i < nz; ++i) { out[i] += w[i] * fk[k]; wg += w[i] * g[i]; }
        out[nc] += af * wg * fk[k + 1];
        for (int l = nz; l < nc; ++l) {
            double s = 0.0;
            for (int i = lo; i < nz; ++i) s += w[i] * T[i * nc + l];
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
            for (int j = lo; j < nz; ++j) {
                double s = 0.0;
                for (int i = lo; i < j; ++i) s += w[i] * T[i * nc + j];
                wn[j] = s;
            }
            for (int i = lo; i < nz; ++i) w[i] = wn[i];
        }
    }
    for (int i = 0; i <= nc; ++i) y[i] = out[i];
}

// z (augmented, z[2n] = 1) at t0 + sig on arc a
// za (optional): za_i = sum_j |W_ij y_j|, the terms of z_i (its scale and rounding)
void GSolver::zat(const Arc& a, double sig, double* z, double* za) const {
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
    for (int i = 0; i < n2 && za; ++i) {
        double s = 0.0;
        for (int j = 0; j < n2; ++j) s += fabs(b.W[i * n2 + j] * y[j]);
        za[i] = s;
    }
}

void GSolver::zstart(const uint16_t* seq, int N, const double* tau, int k, double* z) const {
    if (!arc(k, seq, N, tau, ac_)) { for (int i = 0; i <= 2 * n_; ++i) z[i] = 0.0; return; }
    ac_.x = R_out->x[k];
    zat(ac_, 0.0, z);
}

// Switching rows of set S (one solve for the projection and the multipliers): inactive rows ->
// gap a_r vt - b_r = G wt + g0; active rows -> -mu_r |a_r|^2. Both must stay <= 0. -1: singular Gram (no face)
int GSolver::gphi(uint16_t S, int keys[2 * MAXM], double G[2 * MAXM][MAXM], double g0[2 * MAXM]) const {
    int idx[MAXM], na;
    double X[MAXM * (MAXM + 1)], P[MAXM][MAXM], q[MAXM];
    const int ldx = MAXM + 1;
    if (!project(rows_, rhs_, m_, S, true, P, q, X, idx, &na)) return -1;
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
    for (int a = 0; a < na; ++a) {
        keys[nk] = idx[a];
        for (int j = 0; j < m_; ++j) G[nk][j] = -X[a * ldx + j] * rownorm2_[idx[a]];
        g0[nk++] = X[a * ldx + m_] * rownorm2_[idx[a]];
    }
    return nk;
}

// ---- wt on an arc as exponential-polynomials ----
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

// fills mt_ for the arc of F at sig (and its bound factors of kind 1 or 2 at half-width h; kind 2:
// node [lo, hi] with midpoint sig)
void GSolver::mtab(const ArcFn& F, double sig, double h, int kind, double lo, double hi) const {
    const SetBasis& b = *F.b;
    const int n2 = 2 * n_, ns = b.ns, s1 = ns + b.nc;
    const bool at = (mt_.b == F.b && mt_.L == F.L && mt_.sig == sig);
    if (at && (kind == 0 || (mt_.kind == kind && mt_.h == h))) return;
    for (int i = 0; i < n2; ++i) {
        if (i == ns && s1 > ns) { i = s1 - 1; continue; }
        const double re = b.re[i], om = b.im[i], anc = (i < ns) ? 0.0 : F.L, t = sig - anc;
        if (!at) {
            mt_.e[i] = xexp(re * t);
            if (om != 0.0) { mt_.e[i + 1] = mt_.e[i]; mt_.cs[i] = cos(om * t); mt_.cs[i + 1] = sin(om * t); }
        }
        if (kind == 1) mt_.bnd[i] = xexp(re * t + sqrt(re * re + om * om) * h);
        if (kind == 2) mt_.bnd[i] = xexp(re * (((re > 0) ? hi : lo) - anc));
        if (om != 0.0) { mt_.bnd[i + 1] = mt_.bnd[i]; ++i; }
    }
    mt_.b = F.b; mt_.L = F.L; mt_.sig = sig;
    if (kind) { mt_.kind = kind; mt_.h = h; }
    else if (!at) mt_.kind = 0;
}

// derivatives d[0..nd] of f at sig (closed form)
void GSolver::feval(const RowFn& f, double sig, int nd, double* d) const {
    const SetBasis& b = *f.F->b;
    const int n2 = 2 * n_, ns = b.ns, s1 = ns + b.nc;
    for (int k = 0; k <= nd; ++k) d[k] = 0.0;
    d[0] = f.c0;
    mtab(*f.F, sig, 0.0, 0);
    for (int i = 0; i < n2;) {
        if (i == ns && s1 > ns) { i = s1; continue; }
        const double re = b.re[i], e = mt_.e[i];
        if (b.im[i] == 0.0) {
            double v = f.u(i) * e;
            for (int k = 0; k <= nd; ++k) { d[k] += v; v *= re; }
            ++i;
        } else {
            const double om = b.im[i], c = mt_.cs[i], sn = mt_.cs[i + 1], P = f.u(i), Q = f.u(i + 1);
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
    mtab(*f.F, m, h, 2, lo, hi);
    for (int i = 0; i < n2;) {
        if (i == ns && s1 > ns) { i = s1; continue; }
        const double re = b.re[i], om = b.im[i], e = mt_.e[i], ee = mt_.bnd[i];
        double x, x1, A;
        if (om == 0.0) {
            const double u = f.u(i);
            x = u * e; x1 = re * x; A = fabs(u);
            ++i;
        } else {
            const double P = f.u(i), Q = f.u(i + 1), c = mt_.cs[i], sn = mt_.cs[i + 1];
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

// ev: rows of a walk event, zero at sig by construction (as in tiers 1 and 2): their sign is that of the derivatives
bool GSolver::glex_ok(uint16_t S, const ArcFn& F, double sig, double h, double* tol, uint16_t ev) const {
    int keys[2 * MAXM], il;
    double G[2 * MAXM][MAXM], g0[2 * MAXM];
    const int nk = gphi(S, keys, G, g0);
    if (nk < 0) return false;
    for (int r = 0; r < nk; ++r)
        if (lead(F, G[r], g0[r], sig, h, &tol[r], &il, (ev >> keys[r]) & 1u) > 0) return false;
    return true;
}

// the minimiser's set just right of sig; tol[r] gets the lead() tolerances of its rows
bool GSolver::glex_set(const ArcFn& F, double sig, double h, uint16_t guess, uint16_t* out, double* tol,
                       uint16_t ev) const {
    if (guess != 0xFFFFu && valid_set(guess) && glex_ok(guess, F, sig, h, tol, ev)) { *out = guess; return true; }
    return for_sets([&](uint16_t S) {
        if (!glex_ok(S, F, sig, h, tol, ev)) return false;
        *out = S;
        return true;
    });
}

// Leading Taylor coefficient at sig of the row g0 + G wt (coefficients that matter at scale h;
// 0 if none), its order and the tolerance used; ev: the row is zero at sig by construction (a walk event), its
// value there (a root's rounding, amplified by nearly dependent rows) is not used
double GSolver::lead(const ArcFn& F, const double* G, double g0, double sig, double h, double* tolp,
                     int* ilead, bool ev) const {
    if (!ev) {   // cheap case: f(sig) above a bound of the tolerance (sum_k |c_k| h^k <= this sum)
        RowFn f;
        rowfn(F, G, g0, f);
        const SetBasis& b = *F.b;
        const int n2 = 2 * n_, ns = b.ns, s1 = ns + b.nc;
        double B = fabs(f.c0) + R_, d0, x = 1.0;
        mtab(F, sig, h, 1);
        for (int i = 0; i < n2;) {
            if (i == ns && s1 > ns) { i = s1; continue; }
            const double om = b.im[i];
            const double A = (om == 0.0) ? fabs(f.u(i)) : sqrt(f.u(i) * f.u(i) + f.u(i + 1) * f.u(i + 1));
            B += A * mt_.bnd[i];
            i += (om == 0.0) ? 1 : 2;
        }
        for (int k = 0; k < F.np; ++k) { B += fabs(f.p(k)) * x; x *= fabs(sig) + h; }
        if (F.nsb) B += sbound(f, 0, fabs(sig) + h, true);
        feval(f, sig, 0, &d0);
        if (fabs(d0) > 1e-12 * B) { *tolp = 1e-12 * B; *ilead = 0; return d0; }
    }
    double cs[GN], tol = 0.0, hk = 1.0;
    rtaylor(F, G, g0, sig, cs);
    if (ev) cs[0] = 0.0;
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
// its row tolerances and none does): first root per row, *s1 the earliest. ev: the rows of the event at sig
// (lead()); *imm gets the rows already beyond tolerance at sig (order 0: not zero there).
int GSolver::events(const ArcFn& F, uint16_t S, double sig, double h, double* rootk, bool* simplek,
                    int* keyk, double* s1p, const double* lexed, uint16_t ev, uint16_t* imm) {
    int keys[2 * MAXM], nc = 0;
    double G[2 * MAXM][MAXM], g0[2 * MAXM], s1 = HUGE_VAL;
    const int nk = gphi(S, keys, G, g0);
    for (int r = 0; r < nk; ++r) {
        double tol = lexed ? lexed[r] : 0.0;           // lexed: glex_ok(S) at sig passed, no lead > 0
        int ilead;
        if (!lexed && lead(F, G[r], g0[r], sig, h, &tol, &ilead, (ev >> keys[r]) & 1u) > 0) {
            rootk[nc] = sig; simplek[nc] = (ilead == 1); keyk[nc++] = keys[r];
            if (sig < s1) s1 = sig;
            if (ilead == 0) *imm |= (uint16_t)(1u << keys[r]);
            continue;
        }
        RowFn f;
        rowfn(F, G[r], g0[r], f);
        double root;
        bool simple;
        const double vt = 1e-12 * (f.scale + R_), thr = fmax(tol, vt);   // (vt: gverify's tolerance without noise)
        wthr_ = fmax(wthr_, thr);
        wrat_ = fmax(wrat_, thr / vt);
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
    uint16_t S = 0xFFFFu, zev = 0;                   // zev: rows of the last event (at zt), zero there by construction
    double zt = -1.0;
    ArcFn F;
    wthr_ = wrat_ = 0.0;
    for (int ka = 0; ka < N; ++ka) {
        Arc& a = ac_;
        if (!arc(ka, seq, N, tau, a)) return false;
        a.x = R_out->x[ka];
        if (!chan(a, F)) return false;
        const double h = hloc(*a.b), t0 = a.t0;
        double ltol[2 * MAXM];
        bool lexed = false;
        if (S == 0xFFFFu) {
            if (!glex_set(F, 0.0, h, 0, &S, ltol, 0)) return false;    // (a) initial set
            wseq_[ns++] = S;
            lexed = true;
        }
        double sig = 0.0;
        int stall = 0;
        if (zt != t0) zev = 0;                         // (an event at the junction keeps its rows)
        for (;;) {
            double rootk[2 * MAXM], s1 = HUGE_VAL;
            bool simplek[2 * MAXM];
            int keyk[2 * MAXM];
            uint16_t imm = 0;
            const int nc = events(F, S, sig, h, rootk, simplek, keyk, &s1, lexed ? ltol : nullptr, zev, &imm);
            lexed = false;
            if (nc < 0) return false;
            if (nc == 0) break;
            if (t0 + s1 >= T * (1 - 1e-14)) break;
            if (!walk_next(
                    nc,
                    [&](int q, int* key, double* root, bool* simple) {
                        *key = keyk[q]; *root = rootk[q]; *simple = simplek[q];
                        return true;
                    },
                    s1, 1e-12 * T, imm, [&](uint16_t cand) { return valid_set(cand); },
                    [&](uint16_t guess, uint16_t zv, uint16_t* out) {
                        ++*ndeg;
                        return lexed = glex_set(F, s1, h, valid_set(guess) ? guess : 0xFFFFu, out, ltol, zv);
                    },
                    &zev, S, wseq_, wtau_, t0 + s1, ns, NA, &capfail_, stall))
                return false;
            zt = t0 + s1;
            sig = s1;
        }
    }
    wn_ = ns;
    return true;
}

// How well the plan's switching rows are determined, in row units: the larger of (a) the BVP noise D = max over the
// junctions of |G (wt(lambda_L) - wt(lambda_R))| for the rows of both neighbouring sets (lambda is continuous: the two
// arcs differ only by the BVP solution's error) and (b) the junction residuals the tau stop left (newton_tau's resid:
// a row's value at its own junction, zero at the exact junction time; a stall leaves it at the noise, t3_port3.md).
double GSolver::jnoise(const uint16_t* seq, int N, const double* tau) const {
    double D = 0.0, zl[N2 + 1], zr[N2 + 1], wl[MAXM], wr[MAXM], G[2 * MAXM][MAXM], g0[2 * MAXM];
    Arc al, ar;
    for (int k = 1; k < N; ++k) {
        if (!arc(k - 1, seq, N, tau, al) || !arc(k, seq, N, tau, ar)) return HUGE_VAL;
        al.x = R_out->x[k - 1];
        ar.x = R_out->x[k];
        zat(al, al.L, zl);
        zat(ar, 0.0, zr);
        wt_of(zl + n_, wl);
        wt_of(zr + n_, wr);
        const uint16_t X = (uint16_t)(seq[k - 1] ^ seq[k]);
        for (int sd = 0; sd < 2; ++sd) {
            int keys[2 * MAXM];
            const int nk = gphi(seq[k - sd], keys, G, g0);
            for (int r = 0; r < nk; ++r) {
                double v = 0.0, j = g0[r];
                for (int c = 0; c < m_; ++c) { v += G[r][c] * (wl[c] - wr[c]); j += G[r][c] * wr[c]; }
                if (v != v || j != j) return NAN;
                D = fmax(D, fabs(v));
                if ((X >> keys[r] & 1u) && !(seq[k - sd] >> keys[r] & 1u)) D = fmax(D, fabs(j));   // (b), as resid
            }
        }
    }
    return D;
}

// max over the trajectory of the switching rows of each arc's own set (<= 0 at a PMP point). Returns the largest ratio
// of a row's max to its tolerance vt = 1e-12 (amplitude + R) + D (tier 1's vtol(amp), the walk's event tolerance,
// plus D = jnoise: a row is not determined better than the two sides of a junction agree, nor than the tau stop left
// it at its junction); a box row
// (inactive: a gap in input units) is also held to BOXTOL R - D, so its true value is within the box acceptance.
// bb's max is within its level of the true one, a tenth of vt. *worst = the largest row max
double GSolver::gverify(const uint16_t* seq, int N, const double* tau, double D, double* worst) {
    double q = 0.0;
    ArcFn F;
    RowFn f;
    *worst = HUGE_VAL;
    double w = 0.0;
    for (int ka = 0; ka < N; ++ka) {
        Arc& a = ac_;
        if (!arc(ka, seq, N, tau, a)) return HUGE_VAL;
        a.x = R_out->x[ka];
        int keys[2 * MAXM];
        double G[2 * MAXM][MAXM], g0[2 * MAXM];
        const int nk = gphi(a.S, keys, G, g0);
        if (nk < 0) return HUGE_VAL;
        if (nk == 0) continue;
        if (!chan(a, F)) return HUGE_VAL;
        for (int r = 0; r < nk; ++r) {
            rowfn(F, G[r], g0[r], f);
            const double vt = 1e-12 * (f.scale + R_) + D;
            const double v = bb(f, 0.0, a.L, 0.1 * vt, 0.1 * vt, false, nullptr, nullptr);
            if (v != v) return HUGE_VAL;
            w = fmax(w, v);
            q = fmax(q, v / vt);
            if (!(a.S >> keys[r] & 1u)) q = fmax(q, (v + D) / (BOXTOL * R_));
        }
    }
    *worst = w;
    return (D <= BOXTOL * R_) ? q : HUGE_VAL;          // NaN fails
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

// Running cost in closed form (ends = gends()). Along an extremal d/dt (lambda's) = -s'Qs - |vt|^2 - mu'b (mu' (Mt vt - b) = 0),
// so int = 1/2 [lambda(0)'x0 - lambda(T)'s(T) - int mu'b dt] (the terminal cost: gterm); mu is linear in wt
// on a saturated arc, and int wt = sum of modal phi1 terms + the integrated polynomial.
double GSolver::gcost(const uint16_t* seq, int N, const double* tau, double ends) {
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
        if (nk < 0) return HUGE_VAL;
        for (int r = 0; r < nk; ++r) {
            if (!(seq[ka] & (1u << keys[r]))) continue;   // active row: -mu |a|^2 = G wt + g0
            double v = g0[r] * a.L;
            for (int j = 0; j < m_; ++j) v += G[r][j] * I[j];
            mub -= rhs_[keys[r]] * v / rownorm2_[keys[r]];
        }
    }
    return 0.5 * (J - mub);
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
        rel_[r][N2] = hom_ ? 0.0 : x0_[r];             // (refine: x0's defect enters at junction 0)
    }
}

// Advance the relation from the start of arc a to its end (z-space at both ends).
bool GSolver::fstep(const Arc& a, int nb) {
    const SetBasis& b = *a.b;
    const int n = n_, n2 = 2 * n, ns = b.ns, nc = b.nc, s1 = ns + nc;
    const int c1 = N2 + 1 + nb;
    if (nc == n2) { vstep(a); return ortho(c1); }
    double E[4], eu[N2];
    for (int i = s1; i < n2;) {                        // e^{-Lu L}: (e) or (e cos, e sin) per block
        const int sz = mblk(b.re, b.im, i, -a.L, E);
        eu[i] = E[0];
        if (sz == 2) eu[i + 1] = E[1];
        i += sz;
    }
    relmul(rel_, n, b.W, n2);                          // modal coordinates: rows W
    for (int r = 0; r < n; ++r) {
        double* t = rel_[r];
        double d = t[N2];
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
        t[N2] = d;
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
    relmul(rel_, n, b.Wi, n2);                         // back to z: eta = Wi z
    return ortho(c1);
}

// Vertex arc (every channel fixed: P = 0, H = [[A, 0], [-Q, -A^T]] nilpotent, forcing f = (Bt q, 0)),
// in z directly: C' = C e^{-HL} = sum_k w_k, w_k = C (-HL)^k / k!, d' = d + sum_k (w_k f) L / (k + 1)
// (= C int_0^L e^{-Hs} f ds). Finite sums: w_k = 0 for k >= the nilpotency index (<= 2 n).
void GSolver::vstep(const Arc& a) {
    const int n = n_, n2 = 2 * n;
    double f[MAXM];
    for (int j = 0; j < m_; ++j) {
        double s = 0.0;
        for (int k = 0; k < m_; ++k) s += LiT_[j][k] * a.q[k];
        f[j] = s;                                      // (Bt q) at row end_[j]
    }
    for (int r = 0; r < n; ++r) {
        double* row = rel_[r];
        double w[N2], v[N2], d = row[N2];
        for (int j = 0; j < n2; ++j) w[j] = row[j];
        for (int k = 0; k <= n2; ++k) {
            double wf = 0.0, nz = 0.0;
            for (int j = 0; j < m_; ++j) wf += w[end_[j]] * f[j];
            d += wf * a.L / (k + 1);
            if (k) for (int j = 0; j < n2; ++j) row[j] += w[j];
            for (int j = 0; j < n2; ++j) v[j] = 0.0;   // v = w H: s-part w_s A - w_l Q, l-part -w_l A^T
            for (int i = 0; i < n; ++i)
                if (w[n + i] != 0.0)                   // (w_l vanishes after the longest chain's steps)
                    for (int j = 0; j < n; ++j) v[j] -= w[n + i] * 0.5 * (p_->Q[i][j] + p_->Q[j][i]);
            for (int j = 0; j < m_; ++j)
                for (int q = 0; q + 1 < p_->orders[j]; ++q) {
                    const int i = off_[j] + q;
                    v[i + 1] += w[i];
                    v[n + i] -= w[n + i + 1];
                }
            const double c = -a.L / (k + 1);
            for (int j = 0; j < n2; ++j) { w[j] = c * v[j]; nz += fabs(w[j]); }
            if (nz == 0.0) break;
        }
        row[N2] = d;
    }
}

// re-orthonormalise the relation rows (modified Gram-Schmidt, twice), same ops on rhs/sensitivities
bool GSolver::ortho(int c1) {
    const int n = n_, n2 = 2 * n;
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
    if (!hjump(seq, N, tau, k) || !arc(k, seq, N, tau, ac_)) return false;
    if (hom_) unforce(ac_);
    return fstep(ac_, nb);
}

// plan defect at junction k of R_out->x on (seq, tau): z_k(tau_k^+) - z_{k-1}(tau_k^-); k = 0: (s_0(0) - x0, 0)
bool GSolver::defect(const uint16_t* seq, int N, const double* tau, int k, double* d) const {
    Arc a;
    double z[N2 + 1], w[N2 + 1];
    if (!arc(k, seq, N, tau, a)) return false;
    a.x = R_out->x[k];
    zat(a, 0.0, z);
    if (k == 0)
        for (int i = 0; i < n_; ++i) { w[i] = x0_[i]; w[n_ + i] = z[n_ + i]; }
    else {
        if (!arc(k - 1, seq, N, tau, a)) return false;
        a.x = R_out->x[k - 1];
        zat(a, a.L, w);
    }
    for (int i = 0; i < 2 * n_; ++i) d[i] = z[i] - w[i];
    return true;
}

// correction sweep (hom_): the relation at tau_k^+ takes the plan's defect, delta(tau_k^+) = delta(tau_k^-) - D_k
bool GSolver::hjump(const uint16_t* seq, int N, const double* tau, int k) {
    if (!hom_) return true;
    double D[N2];
    if (!defect(seq, N, tau, k, D)) return false;
    for (int r = 0; r < n_; ++r) {
        double s = 0.0;
        for (int j = 0; j < 2 * n_; ++j) s += rel_[r][j] * D[j];
        rel_[r][N2] -= s;
    }
    return true;
}

// the correction is homogeneous: no particular part, no forcing
void GSolver::unforce(Arc& a) const {
    for (int i = 0; i < 2 * n_; ++i) a.yp[i] = 0.0;
    for (int i = 0; i < m_; ++i) a.q[i] = 0.0;
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
    return hjump(seq, N, tau, k);
}

// Terminal conditions with the relation at T: s(T) = xf, or lambda(T) = Qf (s(T) - xf).
// Column 0 (rhs) -> zT (if given); sensitivity columns (homogeneous) -> Z_.
// zp (refine): the plan's z(T); column 0 is then the correction delta with zp + delta meeting the condition.
bool GSolver::terminal(int nb, double* zT, const double* zp) {
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
                    u += q * (zp ? zp[i] - p_->xf[i] : p_->xf[i]);
                }
                row[j] = s;
                if (zp) d -= cl[j] * (u - zp[n + j]);  // delta_l = Qf delta_s + Qf (s_p - xf) - lambda_p
                else d += cl[j] * u;
            } else {
                row[j] = cl[j];
                d -= cs[j] * (zp ? p_->xf[j] - zp[j] : p_->xf[j]);
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
            else { z[i] = (c > 0) ? 0.0 : zp ? p_->xf[i] - zp[i] : p_->xf[i]; z[n + i] = v; }
        }
        if (qf)
            for (int i = 0; i < n; ++i) {
                double s = 0.0;
                for (int j = 0; j < n; ++j)
                    s += 0.5 * (p_->Qf[i][j] + p_->Qf[j][i]) * (z[j] - ((c > 0) ? 0.0 : zp ? p_->xf[j] - zp[j] : p_->xf[j]));
                z[n + i] = (zp && c == 0) ? s - zp[n + i] : s;
            }
        for (int i = 0; i < 2 * n; ++i) {
            if (c > 0) Z_[i][c - 1] = (real)z[i];
            else if (zT) zT[i] = z[i];
        }
    }
    return true;
}

// recover()'s end rows for refine(), in z units: W_s (E_s a - eta_s) = 0 for the rhs column (the defects it
// corrects are z-space misfits; the modal rows weight them by W's column norms, up to 3e3 in TINY)
void GSolver::zrows(const Arc& a, const double* eta) {
    const SetBasis& b = *a.b;
    const int n2 = 2 * n_, ns = b.ns;
    double E[4], lr[N2];
    for (int r = 0; r < n2; ++r) {
        for (int i = 0; i < ns;) {
            const int sz = mblk(b.re, b.im, i, a.L, E);
            for (int m = 0; m < sz; ++m) {
                double s = 0.0;
                for (int l = 0; l < sz; ++l) s += b.W[r * n2 + i + l] * E[2 * l + m];
                lr[i + m] = s;
            }
            i += sz;
        }
        double lc = 0.0;
        for (int i = 0; i < ns; ++i) lc += b.W[r * n2 + i] * (eta[i] - a.yp[i]);
        for (int j = 0; j < ns; ++j) {                // Givens: rotate the row into R row j
            if (lr[j] == 0.0) continue;
            const double rj = rel_[j][j], hh = sqrt(rj * rj + lr[j] * lr[j]);
            const double cg = rj / hh, sg = lr[j] / hh;
            for (int k = j; k < ns; ++k) {
                const double u = rel_[j][k], w = lr[k];
                rel_[j][k] = (cg * u + sg * w);
                lr[k] = -sg * u + cg * w;
            }
            const double u = rel_[j][N2];
            rel_[j][N2] = (cg * u + sg * lc);
            lc = -sg * u + cg * lc;
        }
    }
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
    relmul(rel_, n, b.W, n2);
    for (int r = 0; r < n; ++r) {
        double* row = rel_[r];
        for (int c = 0; c < ncols; ++c) {
            double s = row[c0 + c];
            if (aff) for (int j = 0; j < ns; ++j) s -= row[j] * a.yp[j];
            for (int j = ns; j < n2; ++j) s -= row[j] * (aff ? zb[j] : (double)Z_[j][c]);
            row[c0 + c] = s;
        }
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
    if (hom_) zrows(a, zb);
    else
        for (int i = 0; i < ns;) {
            const int sz = mblk(b.re, b.im, i, a.L, E);
            for (int l = 0; l < sz; ++l) {
                double lr[N2], lc[NB + 1];
                for (int j = 0; j < ns; ++j) lr[j] = 0.0;
                lr[i] = E[2 * l];
                if (sz == 2) lr[i + 1] = E[2 * l + 1];
                for (int c = 0; c < ncols; ++c)
                    lc[c] = aff ? zb[i + l] - a.yp[i + l] : (double)Z_[i + l][c];
                for (int j = i; j < ns; ++j) {        // Givens: rotate the row into R row j
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

// One defect correction (iterative refinement) of the BVP solution R_out->x of (seq, tau): the backward recovery
// fits each arc's stable part in least squares to its start relation and its end value, so the plan the arcs
// represent misses x0, continuity and the terminal condition by the BVP's conditioning times rounding. Those
// defects drive the same sweep without particular parts; the correction is added to the arcs, zT_ follows.
bool GSolver::refine(const uint16_t* seq, int N, const double* tau) {
    double zp[N2 + 1], zb[N2], dT[N2], dx[N2], D[N2];
    Arc a;
    if (!arc(N - 1, seq, N, tau, a)) return false;
    a.x = R_out->x[N - 1];
    zat(a, a.L, zp);
    hom_ = true;
    int stride;
    bool ok = forward(seq, N, tau, 0, 0, &stride) && terminal(0, zb, zp);
    for (int i = 0; i < 2 * n_; ++i) dT[i] = zb[i];
    for (int k = N - 1; ok && k >= 0; --k) {
        ok = load(seq, N, tau, k, stride, 0, 0) && arc(k, seq, N, tau, ac_);
        if (ok) unforce(ac_);
        ok = ok && recover(ac_, 0, true, dx, zb) && (k == 0 || defect(seq, N, tau, k, D));
        for (int i = 0; ok && i < 2 * n_; ++i) {
            R_out->x[k][i] += dx[i];                   // (D_k is taken before arc k changes)
            if (k) zb[i] += D[i];
            ok = isfinite(R_out->x[k][i]);
        }
    }
    hom_ = false;
    for (int i = 0; ok && i < 2 * n_; ++i) zT_[i] = zp[i] + dT[i];
    return ok;
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
    double prev = HUGE_VAL, hv[NJ], hvv[NJ];
    int bad = 0;
    bool fact = false;                                // J_ holds the QR of the current structure's J
    int nfresh = 0, nchord = 0;
    while (nfresh < MAX_TAU_NEWTON && nchord < 2 * MAX_TAU_NEWTON && budget_ > 0) {
        const int nj = N - 1;
        if (rmax <= tol) { *Np = N; return true; }
        if (rmax > 0.5 * prev && ++bad > 2) { *Np = N; return rmax <= BOXTOL * R_; }   // stall: see below
        // chord step (reuse the factored J) while the last step contracted well
        const bool chord = fact && rmax <= CHORD * prev;
        prev = rmax;
        ++*its;
        --budget_;
        bool sing = false;
        if (chord) ++nchord;
        else ++nfresh;
        if (chord) {
            for (int q = 0; q < nr; ++q) J_[q][nj] = (real)(-r[q]);
            for (int k = 0; k < nj; ++k) {              // stored reflections (v below the diagonal)
                double s = hv[k] * J_[k][nj];
                for (int q = k + 1; q < nr; ++q) s += J_[q][k] * J_[q][nj];
                s *= 2.0 / hvv[k];
                J_[k][nj] = (real)(J_[k][nj] - s * hv[k]);
                for (int q = k + 1; q < nr; ++q) J_[q][nj] = (real)(J_[q][nj] - s * J_[q][k]);
            }
        } else {
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
        sing = !(jmax > 0.0);
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
            hv[k] = v[k];
            hvv[k] = vv;
            for (int q = k + 1; q < nr; ++q) J_[q][k] = (real)v[q];
        }
        fact = !sing;
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
            fact = false;
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
        if (!acc && chord && bvp(seq, N, tau)) { fact = false; continue; }   // retry with a fresh J
        if (!acc) {
            bvp(seq, N, tau);                         // restore the state of tau
            *Np = N;
            // Stall: the residual is at the BVP's noise (t3_port3.md: stalls at r ~ the rows' junction mismatch jnoise).
            // A junction residual is the plan's switching-row value at the junction, so the plan can pass gverify only if
            // it is within the loosest row tolerance (BOXTOL R): the walk and gverify decide, Newton cannot do better.
            return rmax <= BOXTOL * R_;
        }
        for (int j = 0; j < nj; ++j) tau[j] = tn[j];
        for (int q = 0; q < nr; ++q) r[q] = rn[q];
        if (diverged(false)) { *Np = N; return false; }
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
        if (!ok && (infeas_ || budget_ <= 0 || retry-- == 0 || (o > 0 && rounds <= 3))) break;
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
    cert_ = infeas_ = false;
    wthr_ = wrat_ = 0.0;
    free_bound();
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
    if (!x0ok_) { R.failure = LQOCP_STR("non-finite x0"); return false; }
    int its = 0, outer = 0, ndeg = 0, nsteps = 0;
    an_ = 1;
    aseq_[0] = 0;
    bool conv = false;
    const char* why = nullptr;
    if (!bvp(aseq_, 1, atau_)) {
        R.failure = bfail() ? bfail() : p_->has_Qf ? LQOCP_STR("linear BVP singular (ill-conditioned data?)")
                                                   : LQOCP_STR("linear BVP singular (infeasible endpoint?)");
        return false;
    }
    if (!p_->has_bounds) {
        cert_ = conv = true;
    } else {
        const double ratio = gmax_ratio(aseq_, 1, atau_);
        if (!(ratio < HUGE_VAL)) { R.failure = LQOCP_STR("non-finite trajectory"); return false; }
        if (ratio <= 1.0 && propose(aseq_, 1, atau_, &ndeg))
            cert_ = conv = (wn_ == 1 && wseq_[0] == 0);
        if (!conv && diverged(true))                 // (the plain solution is the PMP point of its own bound scale)
            why = infeas_ ? LQOCP_STR("infeasible endpoint (no feasible input)") : LQOCP_STR("costate diverged (ill-conditioned: |x0| too large?)");
        if (!conv && !why) {
            // homotopy in the bound scale kappa: exact collapse events inside
            // the tau-Newton, secant predictor, step halving on failure
            double kap = 1.05 * fmax(ratio, 1.0);
            double dl = xlog(kap) * ((ratio <= KAP_PLAIN) ? 1.0 : 1.0 / 3.0);
            gscale(kap);
            if (!bvp(aseq_, 1, atau_)) { R.failure = LQOCP_STR("linear BVP failed"); return false; }
            hn_ = 0;
            uint16_t sseq[NA];
            double stau[NA];
            for (;;) {
                if (kap == 1.0) { conv = true; break; }
                if (infeas_ || diverged(true)) { why = infeas_ ? LQOCP_STR("infeasible endpoint (no feasible input)") : LQOCP_STR("costate diverged (ill-conditioned: |x0| too large?)"); break; }
                if (budget_ <= 0) { why = LQOCP_STR("iteration cap reached"); break; }
                if (dl < DL_MIN) { why = p_->has_Qf ? LQOCP_STR("homotopy stalled at bound scale > 1 (ill-conditioned: |x0| too large?)")
                                                  : LQOCP_STR("homotopy stalled at bound scale > 1 (bounds infeasible?)"); break; }
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
                    if (capfail_) { why = LQOCP_STR("arc capacity NA exceeded"); break; }
                    dl *= 0.5;
                    gscale(kap);
                    an_ = sn;
                    for (int i = 0; i < sn; ++i) aseq_[i] = sseq[i];
                    for (int i = 0; i + 1 < sn; ++i) atau_[i] = stau[i];
                    if (!bvp(aseq_, an_, atau_)) { why = LQOCP_STR("linear BVP failed (bounds infeasible?)"); break; }
                }
            }
        }
    }
    return finish(R, conv, why, its, nsteps, ndeg);
}

// MPC warm start: R holds the previous solution of this problem; the Problem's x0
// has been updated by the caller. The previous arc structure, with its junction times moved by
// -dt (arcs that ended are dropped, the last arc is stretched to T), is corrected by Newton on
// tau and the exact walk at the true bounds: no homotopy. If that fails with iterations left, the
// cold solve (homotopy) takes over. At most max_its Newton iterations in all; when the cap is hit,
// R is the current iterate (it satisfies the dynamics and both boundary conditions) with
// converged = false. A previous plan that is not converged and violates PMP by more than WRESET*R
// (a stale active set asking for inputs far outside the box) is not reused: cold restart.
bool GSolver::solve_warm(GResult& R, int max_its, double dt) {
    if (!isfinite(dt)) { R.converged = false; R.failure = LQOCP_STR("dt not finite"); return false; }
    if (R.narcs < 1 || R.narcs > NA || !x0ok_ || (!R.converged && !(R.pmp_violation <= WRESET * R_)))
        return cold(R, max_its);
    const double T = p_->T;
    an_ = 0;
    for (int k = 0; k < R.narcs; ++k) {
        const bool last = (k == R.narcs - 1);
        const double t1 = last ? T : R.t1[k] - dt;
        if (!last && t1 <= STOL * T) continue;         // this arc has ended (to the junction-time tolerance)
        aseq_[an_] = R.S[k];
        if (!last) atau_[an_] = t1;
        ++an_;
    }
    begin(R, max_its);
    int its = 0, outer = 0, ndeg = 0;
    if (correct(&its, &outer, &ndeg, 12)) return finish(R, true, nullptr, its, 0, ndeg);
    if (infeas_) return finish(R, false, LQOCP_STR("infeasible endpoint (no feasible input)"), its, 0, ndeg);
    if (budget_ > 0 && !capfail_) {                    // cold fallback with the iterations left
        const bool ok = cold(R, budget_);
        R.iterations += its;
        return ok;
    }
    return finish(R, false, capfail_ ? LQOCP_STR("arc capacity NA exceeded") : LQOCP_STR("iteration cap reached"), its, 0, ndeg);
}

// Terminal condition of the plan the arcs represent: z(T) of the last arc. (The BVP recovers the arcs backward from
// zT_, which meets the condition by construction; an ill-conditioned BVP shows only in the arcs.) Per state, as tier 1's
// relative stop: fixed endpoint F_i = s_i(T) - xf_i, terms a_i = sig_i; Qf = Cf^T Cf: F = Cf e - mu with
// Cf^T mu = lambda(T) (tier 1's transversality form, Cf units), terms a_i = |mu_i| + sum_j |Cf_ij| sig_j; a zero
// pivot of the semidefinite Cf leaves lambda_i - (Cf^T mu)_i (lambda units), terms zl_i + sum_k |Cf_ki| a_k.
// sig_i = max(sig_, |free_i|, zs_i) (free = e^{AT} x0; zs, zl: the modal terms of s_i(T), lambda_i(T), as tier 1's
// terms of s(T)). Returns max_i |F_i| / a_i; *jf = 1/2 |Cf (s(T) - xf)|^2 at
// zT_ (factored: e'Qf e cancels below 0 for an ill-conditioned Qf).
double GSolver::gterm(double* jf) const {
    const GResult& R = *R_out;
    const int n = n_;
    double z[N2 + 1], za[N2], sg[MAXN], r = 0.0;
    Arc a;
    if (R.narcs < 1 || !arc(R.narcs - 1, R.S, R.narcs, R.t1, a)) return HUGE_VAL;
    a.x = R.x[R.narcs - 1];
    zat(a, a.L, z, za);
    for (int i = 0; i < 2 * n; ++i)
        if (!isfinite(za[i])) return HUGE_VAL;
    for (int j = 0; j < m_; ++j)
        for (int i = off_[j]; i <= end_[j]; ++i) {   // e^{AT} x0 on chain j
            double f = 0.0;
            for (int l = i; l <= end_[j]; ++l) f += ipow(p_->T, l - i) / fact(l - i) * x0_[l];
            sg[i] = fmax(sig_[i], fmax(fabs(f), za[i]));
        }
    *jf = 0.0;
    if (!p_->has_Qf) {
        for (int i = 0; i < n; ++i) {
            const double d = fabs(z[i] - p_->xf[i]);
            if (d > 0.0) r = fmax(r, d / sg[i]);       // (d > 0: sg_i >= |s_i| or |xf_i| > 0)
        }
        return r;
    }
    double Cf[MAXN * MAXN], mu[MAXN], am[MAXN];
    if (!psd_factor(p_->Qf, n, Cf)) return HUGE_VAL;
    for (int i = 0; i < n; ++i) {                     // Cf^T mu = lambda (lower triangular), mu_i = 0 on a zero pivot
        double s = z[n + i], b = za[n + i];
        for (int k = 0; k < i; ++k) { s -= Cf[k * MAXN + i] * mu[k]; b += fabs(Cf[k * MAXN + i]) * am[k]; }
        double ce = 0.0, ct = 0.0, cm = 0.0;
        for (int j = i; j < n; ++j) {
            ce += Cf[i * MAXN + j] * (z[j] - p_->xf[j]);
            ct += Cf[i * MAXN + j] * (zT_[j] - p_->xf[j]);
            cm += fabs(Cf[i * MAXN + j]) * sg[j];
        }
        *jf += 0.5 * ct * ct;
        if (Cf[i * MAXN + i] > 0.0) {
            mu[i] = s / Cf[i * MAXN + i];
            am[i] = fabs(mu[i]) + cm;
            const double d = fabs(ce - mu[i]);
            if (d > 0.0) r = fmax(r, d / am[i]);
        } else {
            mu[i] = am[i] = 0.0;
            if (s != 0.0) r = fmax(r, fabs(s) / b);   // (b >= |lambda_i|)
        }
    }
    return r;
}

// Weak-duality infeasibility certificate for a fixed endpoint: tier 1's, on the same chains with Q ignored (the input
// constraint alone decides feasibility). Any input with lo <= M u <= hi that reaches xf has
// nu'(xf - e^{AT} x0) = int w'u dt <= int sum_r max(g_r hi_r, g_r lo_r) dt, w_j = sum_i nu_i (T - t)^(end_j - i) / (end_j - i)!
// on chain j and g = M^-T w (polynomials of degree < MAXK, integrated exactly between their roots). A nu breaking this by
// more than its rounding proves that no such input exists: exact, no threshold. nu = -+lambda(T) of the BVP solution in
// place (an infeasible problem's costate runs off along a separating nu; any nu is valid, so both signs are tried).
bool GSolver::infeasible() const {
    static_assert(MAXK <= 4, "g' has degree <= 2");
    if (p_->has_Qf || !p_->has_bounds || !mreg_) return false;
    auto pv = [](const double* c, double x) { double v = 0.0; for (int k = MAXK - 1; k >= 0; --k) v = v * x + c[k]; return v; };
    const int n = n_, m = m_, ldb = MAXK + MAXM;
    const double T = p_->T;
    double d[MAXN];
    for (int j = 0; j < m; ++j)
        for (int i = off_[j]; i <= end_[j]; ++i) {
            double f = 0.0;
            for (int l = i; l <= end_[j]; ++l) f += ipow(T, l - i) / fact(l - i) * x0_[l];
            d[i] = p_->xf[i] - f;
        }
    for (int sg = -1; sg <= 1; sg += 2) {
        double A[MAXM * MAXM], B[MAXM * (MAXK + MAXM)], val = 0.0, ab = 0.0, I = 0.0, abI = 0.0, mn = 0.0, mi = 0.0;
        for (int i = 0; i < n; ++i) { const double t = sg * zT_[n + i] * d[i]; val += t; ab += fabs(t); }
        for (int j = 0; j < m; ++j) {                  // A = M^T, B = [w_j coefficients in s = T - t | I]
            for (int i = 0; i < m; ++i) A[j * MAXM + i] = p_->M[i][j];
            for (int k = 0; k < ldb; ++k) B[j * ldb + k] = 0.0;
            for (int k = 0; k <= end_[j] - off_[j]; ++k) B[j * ldb + k] = sg * zT_[n + end_[j] - k] / fact(k);
            B[j * ldb + MAXK + j] = 1.0;
        }
        if (!lu_solve(A, m, MAXM, B, ldb, ldb)) return false;
        for (int i = 0; i < m; ++i) {                  // kappa_inf(M^T): the solve's relative error is below kappa m EPS
            double a = 0.0, b = 0.0;
            for (int j = 0; j < m; ++j) { a += fabs(p_->M[j][i]); b += fabs(B[i * ldb + MAXK + j]); }
            mn = fmax(mn, a);
            mi = fmax(mi, b);
        }
        bool ok = true;
        for (int r = 0; r < m && ok; ++r) {
            const double* g = B + r * ldb;
            double cr[2] = {0.0, 0.0}, pts[4];
            int np = 0, nc = 0;
            const double a2 = 3.0 * g[3], b1 = 2.0 * g[2], c0 = g[1], disc = b1 * b1 - 4.0 * a2 * c0;   // g' = 0
            if (a2 != 0.0 && disc > 0.0) {
                const double qq = -0.5 * (b1 + (b1 >= 0.0 ? sqrt(disc) : -sqrt(disc)));
                cr[0] = qq / a2; cr[1] = (qq != 0.0) ? c0 / qq : 0.0; nc = 2;
                if (cr[0] > cr[1]) { const double t = cr[0]; cr[0] = cr[1]; cr[1] = t; }
            } else if (a2 == 0.0 && b1 != 0.0) {
                cr[0] = -c0 / b1; nc = 1;
            }
            pts[np++] = 0.0;
            for (int q = 0; q < nc; ++q) if (cr[q] > pts[np - 1] && cr[q] < T) pts[np++] = cr[q];
            pts[np++] = T;
            double a = 0.0;                            // pieces between sign changes (g monotone between pts)
            for (int q = 1; q < np && ok; ++q) {
                const double ga = pv(g, pts[q - 1]), gb = pv(g, pts[q]);
                double z = pts[q];
                if (ga * gb < 0.0)                     // the root of the monotone piece, bisected to rounding
                    for (double lo = pts[q - 1], hi = pts[q];;) {
                        const double mid = 0.5 * (lo + hi);
                        if (!(mid > lo && mid < hi)) { z = mid; break; }
                        if ((pv(g, mid) > 0.0) == (ga > 0.0)) lo = mid; else hi = mid;
                    }
                for (int h = 0; h < 2; ++h) {
                    const double b = h ? pts[q] : z;
                    if (!(b > a)) continue;
                    const double gm = pv(g, 0.5 * (a + b));
                    const double bnd = gm > 0.0 ? p_->hi[r] : p_->lo[r];
                    double P = 0.0, Pa = 0.0;
                    for (int k = 0; k < MAXK; ++k) {
                        const double c = g[k] / (k + 1), e = ipow(b, k + 1) - ipow(a, k + 1);
                        P += c * e;
                        Pa += fabs(c) * e;
                    }
                    if (gm != 0.0 && !isfinite(bnd)) { ok = false; break; }   // the box is open along g: no bound
                    if (gm != 0.0) { I += bnd * P; abI += fabs(bnd) * Pa; }
                    a = b;
                }
            }
        }
        if (ok && val - I > DBL_EPSILON * ((n + 4 * m * (MAXK + 2) + 2) * (ab + abI) + m * mn * mi * abI)) return true;
    }
    return false;
}

// Divergence test of the BVP solution in place (cold homotopy, newton_tau, warm path). Fixed endpoint: the certificate
// (sets infeas_: a proof, valid at any iterate). Qf, at a PMP point (pmp: the homotopy's accepted points) of any bound
// scale >= 1: lambda(T) = Qf e and |Qf e|^2 <= tr(Qf) e'Qf e <= 2 tr(Qf) J* <= 2 tr(Qf) J0 (J0 = cost of u = 0, feasible
// when 0 is in the box, so in every scaled box): a costate beyond that is not a PMP point, the BVP has diverged. Not a
// test for Newton iterates (measured: they pass the bound by up to 190x on their way to an optimum; t3_port3.md).
// shortcut: no test without a certificate (fixed endpoint, M singular) or with 0 outside the box (Qf, jbnd_ < 0): the
// homotopy's step floor or the iteration cap ends a diverging run there; a feasible point's cost would be the upgrade.
bool GSolver::diverged(bool pmp) {
    if (!p_->has_Qf) return infeas_ = infeasible();
    if (!pmp || !(jbnd_ >= 0.0)) return false;
    double l2 = 0.0;
    for (int i = 0; i < n_; ++i) l2 += zT_[n_ + i] * zT_[n_ + i];
    return !(l2 <= jbnd_);                             // NaN: diverged
}

// jbnd_ = 2 tr(Qf) J0 (+ rounding), J0 = 1/2 int x'Qx + 1/2 e'Qf e of the free response x_i = sum_k x0_(i+k) t^k / k!
void GSolver::free_bound() {
    jbnd_ = -1.0;
    for (int k = 0; k < 2 * m_; ++k)
        if (finite_[k] && rhs0_[k] < 0.0) return;      // u = 0 infeasible
    if (!p_->has_Qf) return;
    const double T = p_->T;
    double c[MAXN][MAXK], e[MAXN], J = 0.0, Ja = 0.0, tr = 0.0;
    for (int j = 0; j < m_; ++j)
        for (int i = off_[j]; i <= end_[j]; ++i) {
            double f = 0.0;
            for (int k = 0; k < MAXK; ++k) {
                c[i][k] = (i + k <= end_[j]) ? x0_[i + k] / fact(k) : 0.0;
                f += c[i][k] * ipow(T, k);
            }
            e[i] = f - p_->xf[i];
        }
    for (int i = 0; i < n_; ++i) {
        tr += p_->Qf[i][i];
        for (int j = 0; j < n_; ++j) {
            double I = 0.0;
            for (int k = 0; k < MAXK; ++k)
                for (int l = 0; l < MAXK; ++l) I += c[i][k] * c[j][l] * ipow(T, k + l + 1) / (k + l + 1);
            const double t = 0.5 * (p_->Q[i][j] * I + p_->Qf[i][j] * e[i] * e[j]);
            J += t;
            Ja += fabs(t);
        }
    }
    jbnd_ = 2.0 * tr * (J + (n_ * n_ * MAXK * MAXK + 2) * DBL_EPSILON * Ja);
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
        R.failure = why ? why : LQOCP_STR("linear BVP failed");
        return false;
    }
    R.narcs = an_;
    for (int k = 0; k < an_; ++k) {
        R.S[k] = aseq_[k];
        R.t1[k] = (k == an_ - 1) ? p_->T : atau_[k];
    }
    double jf = 0.0;
    R.residual = gterm(&jf);
    const double rtol = fmax(GTOL, (n_ + 2) * DBL_EPSILON);
    // the BVP's rounding shows in the arcs: defect corrections while they lower the residual (each one a BVP, from
    // the iteration budget)
    const bool ref = conv && !(R.residual <= rtol);
    for (double r0 = HUGE_VAL; conv && budget_ > 0 && !(R.residual <= rtol) && R.residual < r0; --budget_) {
        r0 = R.residual;
        if (!refine(R.S, R.narcs, R.t1)) {
            if (!bvp(aseq_, an_, atau_)) { R.narcs = 0; R.failure = LQOCP_STR("linear BVP failed"); return false; }
            R.residual = gterm(&jf);
            break;
        }
        R.residual = gterm(&jf);
    }
    // the walk's certificate suffices when each row's level was its verification tolerance without noise
    double rat = wrat_;
    R.pmp_violation = wthr_;
    if (!(cert && !ref && wrat_ <= 1.0 && wthr_ <= BOXTOL * R_))
        rat = gverify(R.S, R.narcs, R.t1, jnoise(R.S, R.narcs, R.t1), &R.pmp_violation);
    if (conv && !(rat <= 1.0)) { conv = false; why = LQOCP_STR("optimality check failed"); }   // NaN fails
    if (conv && !(R.residual <= rtol)) { conv = false; why = LQOCP_STR("terminal condition check failed"); }
    R.cost = loose ? HUGE_VAL : gcost(R.S, R.narcs, R.t1, gends(R.S, R.narcs, R.t1)) + jf;
    if (conv && !isfinite(R.cost)) { conv = false; why = LQOCP_STR("non-finite cost"); }
    R.converged = conv;
    R.failure = conv ? nullptr : bfail() ? bfail() : (why ? why : LQOCP_STR("not converged"));
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
    if (out && mreg_ && lu_solve(A, m_, MAXM, w, 1, 1))   // (M singular to rounding: u as projected)
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
