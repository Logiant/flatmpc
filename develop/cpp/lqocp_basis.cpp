// BasisBuilder: per-active-set stable-basis primitives for GSolver, computed on demand (host,
// or offline to generate a flash table, see gen_basis_table.cpp). NOTES.md sections 9.1, 10.
// H_S = W diag(Ls, Tc, Lu) W^-1: sign-function dichotomy (stable | centre | unstable), then
// the stable and unstable blocks are diagonalised (real 2x2 blocks for complex pairs).
#include "lqocp.hpp"

#include <complex>
#include <math.h>

namespace lqocp {

namespace {

const double YMAX = 8.0;    // lift |y| / |u| above which a slow mode stays coupled in the centre
const double ZMAX = 12.0;   // ... if |Re lambda| T <= ZMAX (anchored at the arc start: growth <= e^ZMAX)

bool lu(double* A, int n, double* B, int nrhs) {       // A (n x n), B (n x nrhs), in place
    for (int k = 0; k < n; ++k) {
        int piv = k;
        for (int i = k + 1; i < n; ++i)
            if (fabs(A[i * n + k]) > fabs(A[piv * n + k])) piv = i;
        if (A[piv * n + k] == 0.0) return false;
        if (piv != k) {
            for (int j = 0; j < n; ++j) { double t = A[k * n + j]; A[k * n + j] = A[piv * n + j]; A[piv * n + j] = t; }
            for (int j = 0; j < nrhs; ++j) { double t = B[k * nrhs + j]; B[k * nrhs + j] = B[piv * nrhs + j]; B[piv * nrhs + j] = t; }
        }
        for (int i = k + 1; i < n; ++i) {
            const double f = A[i * n + k] / A[k * n + k];
            if (f == 0.0) continue;
            for (int j = k + 1; j < n; ++j) A[i * n + j] -= f * A[k * n + j];
            for (int j = 0; j < nrhs; ++j) B[i * nrhs + j] -= f * B[k * nrhs + j];
        }
    }
    for (int j = 0; j < nrhs; ++j)
        for (int i = n - 1; i >= 0; --i) {
            double s = B[i * nrhs + j];
            for (int k = i + 1; k < n; ++k) s -= A[i * n + k] * B[k * nrhs + j];
            B[i * nrhs + j] = s / A[i * n + i];
        }
    return true;
}

void mmul(const double* A, const double* B, double* C, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int l = 0; l < n; ++l) s += A[i * n + l] * B[l * n + j];
            C[i * n + j] = s;
        }
}

// matrix sign function (scaled Newton); S in/out (n x n), W1, W2 scratch. Converged at a relative
// step <= 1e-14, or at the round-off floor (a step <= 1e-10 that no longer halves: ill-conditioned)
bool sign_fn(double* S, int n, double* W1, double* W2) {
    double prev = HUGE_VAL;
    for (int it = 0; it < 60; ++it) {
        for (int i = 0; i < n * n; ++i) W1[i] = S[i];
        for (int i = 0; i < n * n; ++i) W2[i] = 0.0;
        for (int i = 0; i < n; ++i) W2[i * n + i] = 1.0;
        if (!lu(W1, n, W2, n)) return false;           // W2 = S^-1
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
        if (dn <= 1e-14 * sn || (it >= 20 && dn <= 1e-10 * sn && dn > 0.5 * prev)) return true;
        prev = dn;
    }
    return false;
}

// Orthonormal basis of null(A) (A: k x k, row stride lda) by QR with column pivoting of A^T:
// A^T P = Qk R, columns r.. of Qk span null(A) (pivot norms <= tol from column r on). Returns r.
int null_basis(const double* A, int k, int lda, double tol, double* Qk, double* B) {
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) { B[i * k + j] = A[j * lda + i]; Qk[i * k + j] = (i == j) ? 1.0 : 0.0; }
    int r = 0;
    for (; r < k; ++r) {
        int piv = r;
        double bn = -1.0;
        for (int j = r; j < k; ++j) {
            double t = 0.0;
            for (int i = r; i < k; ++i) t += B[i * k + j] * B[i * k + j];
            if (t > bn) { bn = t; piv = j; }
        }
        if (sqrt(bn) <= tol) break;
        for (int i = 0; i < k; ++i) { const double t = B[i * k + r]; B[i * k + r] = B[i * k + piv]; B[i * k + piv] = t; }
        double v[N2], vv = 0.0;
        const double sg = sqrt(bn);
        for (int i = r; i < k; ++i) v[i] = B[i * k + r];
        v[r] -= (v[r] > 0) ? -sg : sg;
        for (int i = r; i < k; ++i) vv += v[i] * v[i];
        if (vv == 0.0) continue;
        for (int j = r; j < k; ++j) {
            double t = 0.0;
            for (int i = r; i < k; ++i) t += v[i] * B[i * k + j];
            t *= 2.0 / vv;
            for (int i = r; i < k; ++i) B[i * k + j] -= t * v[i];
        }
        for (int i = 0; i < k; ++i) {
            double t = 0.0;
            for (int l = r; l < k; ++l) t += Qk[i * k + l] * v[l];
            t *= 2.0 / vv;
            for (int l = r; l < k; ++l) Qk[i * k + l] -= t * v[l];
        }
    }
    return r;
}

// r columns spanning range(P) (n x n) by Gram-Schmidt with column pivoting -> Wb cols c0..
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

typedef std::complex<double> cd;

// Real modal form of A (k x k): A V = V L, L block diagonal with 1x1 blocks re[i] or 2x2 blocks
// [[re, im], [-im, re]] (im[i] = b, im[i+1] = -b). Complex Schur form by shifted QR on the
// Hessenberg matrix, eigenvectors by back substitution.
bool modal(const double* A, int k, double* V, double* re, double* im) {
    if (k == 0) return true;
    cd H[MAXN * MAXN], Q[MAXN * MAXN], y[MAXN], cs[MAXN], sn[MAXN];
    for (int i = 0; i < k * k; ++i) H[i] = A[i];
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) Q[i * k + j] = (i == j) ? 1.0 : 0.0;
    for (int c = 0; c + 2 < k; ++c) {                  // Householder Hessenberg reduction
        double nrm = 0.0;
        for (int r = c + 1; r < k; ++r) nrm += std::norm(H[r * k + c]);
        nrm = sqrt(nrm);
        if (nrm == 0.0) continue;
        cd v[MAXN];
        const cd x0 = H[(c + 1) * k + c];
        const cd alpha = -(std::abs(x0) > 0 ? x0 / std::abs(x0) : cd(1.0)) * nrm;
        double vv = 0.0;
        for (int r = c + 1; r < k; ++r) v[r] = H[r * k + c];
        v[c + 1] -= alpha;
        for (int r = c + 1; r < k; ++r) vv += std::norm(v[r]);
        if (vv == 0.0) continue;
        for (int j = 0; j < k; ++j) {                  // H = P H
            cd s = 0.0;
            for (int r = c + 1; r < k; ++r) s += std::conj(v[r]) * H[r * k + j];
            s *= 2.0 / vv;
            for (int r = c + 1; r < k; ++r) H[r * k + j] -= v[r] * s;
        }
        for (int i = 0; i < k; ++i) {                  // H = H P, Q = Q P
            cd s = 0.0, t = 0.0;
            for (int r = c + 1; r < k; ++r) { s += H[i * k + r] * v[r]; t += Q[i * k + r] * v[r]; }
            s *= 2.0 / vv;
            t *= 2.0 / vv;
            for (int r = c + 1; r < k; ++r) {
                H[i * k + r] -= s * std::conj(v[r]);
                Q[i * k + r] -= t * std::conj(v[r]);
            }
        }
    }
    int hi = k - 1, iter = 0;
    while (hi > 0) {
        int l = hi;
        while (l > 0 && std::abs(H[l * k + l - 1]) >
                            1e-16 * (std::abs(H[(l - 1) * k + l - 1]) + std::abs(H[l * k + l])))
            --l;
        if (l > 0) H[l * k + l - 1] = 0.0;
        if (l == hi) { --hi; iter = 0; continue; }
        if (++iter > 500) return false;
        const cd a = H[(hi - 1) * k + hi - 1], b = H[(hi - 1) * k + hi], c = H[hi * k + hi - 1],
                 d = H[hi * k + hi];
        const cd tr = a + d, disc = std::sqrt(tr * tr / 4.0 - (a * d - b * c));
        const cd m1 = tr / 2.0 + disc, m2 = tr / 2.0 - disc;
        cd mu = (std::abs(m1 - d) < std::abs(m2 - d)) ? m1 : m2;
        if (iter % 11 == 10) mu = d + 0.75 * std::abs(c);   // exceptional shift
        for (int i = l; i <= hi; ++i) H[i * k + i] -= mu;
        for (int j = l; j < hi; ++j) {                 // H - mu I = Q R (Givens)
            const cd x = H[j * k + j], z = H[(j + 1) * k + j];
            const double r = sqrt(std::norm(x) + std::norm(z));
            const cd cc = (r == 0) ? cd(1.0) : x / r, ss = (r == 0) ? cd(0.0) : z / r;
            for (int col = j; col < k; ++col) {
                const cd u = H[j * k + col], w = H[(j + 1) * k + col];
                H[j * k + col] = std::conj(cc) * u + std::conj(ss) * w;
                H[(j + 1) * k + col] = -ss * u + cc * w;
            }
            cs[j] = cc;
            sn[j] = ss;
        }
        for (int j = l; j < hi; ++j) {                 // R Q + mu I
            const cd cc = cs[j], ss = sn[j];
            const int rmax = (j + 2 < hi) ? j + 2 : hi;
            for (int row = 0; row <= rmax; ++row) {
                const cd u = H[row * k + j], w = H[row * k + j + 1];
                H[row * k + j] = u * cc + w * ss;
                H[row * k + j + 1] = -u * std::conj(ss) + w * std::conj(cc);
            }
            for (int row = 0; row < k; ++row) {
                const cd u = Q[row * k + j], w = Q[row * k + j + 1];
                Q[row * k + j] = u * cc + w * ss;
                Q[row * k + j + 1] = -u * std::conj(ss) + w * std::conj(cc);
            }
        }
        for (int i = l; i <= hi; ++i) H[i * k + i] += mu;
    }
    double sc = 0.0;
    for (int i = 0; i < k; ++i) sc = fmax(sc, std::abs(H[i * k + i]));
    int col = 0;
    for (int e = 0; e < k; ++e) {
        const cd lam = H[e * k + e];
        const double thr = 1e-9 * fmax(1.0, std::abs(lam));
        if (lam.imag() < -thr) continue;               // the conjugate's partner is used
        for (int i = 0; i < k; ++i) y[i] = 0.0;
        y[e] = 1.0;
        for (int i = e - 1; i >= 0; --i) {
            cd s = 0.0;
            for (int j = i + 1; j <= e; ++j) s += H[i * k + j] * y[j];
            cd den = H[i * k + i] - lam;
            if (std::abs(den) < 1e-14 * sc) den = 1e-14 * sc;
            y[i] = -s / den;
        }
        cd v[MAXN], s2 = 0.0;
        double vn = 0.0;
        for (int i = 0; i < k; ++i) {
            v[i] = 0.0;
            for (int j = 0; j <= e; ++j) v[i] += Q[i * k + j] * y[j];
            vn += std::norm(v[i]);
        }
        vn = sqrt(vn);
        for (int i = 0; i < k; ++i) { v[i] /= vn; s2 += v[i] * v[i]; }
        if (lam.imag() <= thr) {                       // real: rotate onto the real axis
            int im0 = 0;
            for (int i = 1; i < k; ++i) if (std::abs(v[i]) > std::abs(v[im0])) im0 = i;
            const cd ph = std::conj(v[im0]) / std::abs(v[im0]);
            if (col >= k) return false;
            for (int i = 0; i < k; ++i) V[i * k + col] = (v[i] * ph).real();
            re[col] = lam.real();
            im[col] = 0.0;
            ++col;
        } else {                                       // pair: x, y orthogonal (sum v_i^2 real)
            const cd ph = (std::abs(s2) > 0) ? std::conj(std::sqrt(s2)) / std::abs(std::sqrt(s2)) : cd(1.0);
            if (col + 1 >= k) return false;
            for (int i = 0; i < k; ++i) {
                const cd w = v[i] * ph;
                V[i * k + col] = w.real();
                V[i * k + col + 1] = w.imag();
            }
            re[col] = re[col + 1] = lam.real();
            im[col] = lam.imag();
            im[col + 1] = -lam.imag();
            col += 2;
        }
    }
    if (col != k) return false;
    // check A V = V L
    double res = 0.0, an = 0.0;
    for (int i = 0; i < k * k; ++i) an = fmax(an, fabs(A[i]));
    for (int i = 0; i < k; ++i)
        for (int c = 0; c < k;) {
            const bool pair = im[c] != 0.0;
            for (int q = 0; q < (pair ? 2 : 1); ++q) {
                double av = 0.0;
                for (int j = 0; j < k; ++j) av += A[i * k + j] * V[j * k + c + q];
                double vl;
                if (!pair) vl = V[i * k + c] * re[c];
                else if (q == 0) vl = V[i * k + c] * re[c] - V[i * k + c + 1] * im[c];
                else vl = V[i * k + c] * im[c] + V[i * k + c + 1] * re[c];
                res = fmax(res, fabs(av - vl));
            }
            c += pair ? 2 : 1;
        }
    return res <= 1e-9 * fmax(an, 1e-300);
}

}  // namespace

const SetBasis* BasisBuilder::get(uint16_t S) {
    S = basis_key(S);
    for (int i = 0; i < n_; ++i)
        if (sb_[i].S == S) return &sb_[i];
    for (int i = 0; i < nbad_; ++i)                    // failed builds are not retried
        if (bad_[i] == S) return nullptr;
    if (g_ && n_ < MAXSETS && build(S, n_)) return &sb_[n_++];
    if (nbad_ < MAXSETS) bad_[nbad_++] = S;
    return nullptr;
}

// H_S (balanced) = Q [[N, X], [0, H22]] Q^T: the staircase (repeated null spaces) splits off the
// exactly nilpotent part N of the zero eigenvalues (saturated channels: index <= 2 MAXK). H22 has no
// eigenvalue on the imaginary axis: the sign function splits it into stable and unstable parts,
// which are diagonalised (real 2x2 blocks for pairs). Each modal vector u (H22 u = u Lam) is lifted
// to [y; u] with N y - y Lam = -X u, y = sum_j N^j X u Lam^-(j+1) (a finite sum: N^ni = 0).
bool BasisBuilder::build(uint16_t S, int slot) {
    static double H[N2 * N2], A[N2 * N2], Q[N2 * N2], Qk[N2 * N2], Pu[N2 * N2], Ps[N2 * N2],
        M4[N2 * N2], M5[N2 * N2], Wb[N2 * N2], Wk[N2 * N2], V[N2 * N2], H2[N2 * N2];
    const int n = g_->n2();
    if (!g_->hmat(S, H)) return false;
    double dsc[N2];                                    // balancing: Hb = D^-1 H D
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
    double hn = 0.0;                                   // ||Hb||_1
    for (int j = 0; j < n; ++j) {
        double c = 0.0;
        for (int i = 0; i < n; ++i) c += fabs(H[i * n + j]);
        hn = fmax(hn, c);
    }
    // staircase: A = Q^T Hb Q, block b (columns blk[b] .. blk[b+1]) = null space of the trailing block
    for (int i = 0; i < n * n; ++i) { A[i] = H[i]; Q[i] = 0.0; }
    for (int i = 0; i < n; ++i) Q[i * n + i] = 1.0;
    int p = 0, ni = 0, blk[N2 + 1];
    blk[0] = 0;
    while (p < n) {
        const int k = n - p, r = null_basis(A + p * n + p, k, n, 1e-11 * hn, Qk, Wk);
        if (r == k) break;
        for (int i = 0; i < k; ++i)                    // G: null vectors first
            for (int j = 0; j < k; ++j) M4[i * k + j] = Qk[i * k + ((j < k - r) ? r + j : j - (k - r))];
        for (int i = 0; i < n; ++i) {                  // A <- A G, Q <- Q G (columns p..)
            double ta[N2], tq[N2];
            for (int j = 0; j < k; ++j) {
                double sa = 0.0, sq = 0.0;
                for (int l = 0; l < k; ++l) {
                    sa += A[i * n + p + l] * M4[l * k + j];
                    sq += Q[i * n + p + l] * M4[l * k + j];
                }
                ta[j] = sa;
                tq[j] = sq;
            }
            for (int j = 0; j < k; ++j) { A[i * n + p + j] = ta[j]; Q[i * n + p + j] = tq[j]; }
        }
        for (int j = 0; j < n; ++j) {                  // A <- G^T A (rows p..)
            double t[N2];
            for (int i = 0; i < k; ++i) {
                double s = 0.0;
                for (int l = 0; l < k; ++l) s += M4[l * k + i] * A[(p + l) * n + j];
                t[i] = s;
            }
            for (int i = 0; i < k; ++i) A[(p + i) * n + j] = t[i];
        }
        p += k - r;
        blk[++ni] = p;
    }
    if (ni + 1 > NPOLY) return false;
    double err = 0.0;                                  // exact structure: zero what the staircase made
    for (int b = 0; b < ni; ++b)                       // round-off (block b's own and later rows)
        for (int j = blk[b]; j < blk[b + 1]; ++j)
            for (int i = blk[b]; i < n; ++i) { err = fmax(err, fabs(A[i * n + j])); A[i * n + j] = 0.0; }
    if (err > 1e-9 * hn) return false;
    // H22 = A[p.., p..]: stable / unstable by the sign function
    const int k = n - p;
    for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) H2[i * k + j] = A[(p + i) * n + p + j];
    int ns = 0, nu = 0;
    if (k > 0) {
        for (int i = 0; i < k * k; ++i) Pu[i] = H2[i];
        if (!sign_fn(Pu, k, M4, M5)) return false;
        double tu = 0.0;
        for (int i = 0; i < k * k; ++i) { Pu[i] *= 0.5; Ps[i] = -Pu[i]; }
        for (int i = 0; i < k; ++i) { Pu[i * k + i] += 0.5; Ps[i * k + i] += 0.5; tu += Pu[i * k + i]; }
        if (fabs(tu - floor(tu + 0.5)) > 1e-6) return false;
        double h2 = 0.0;
        for (int i = 0; i < k * k; ++i) h2 = fmax(h2, fabs(H2[i]));
        for (int w = 0; w < 2; ++w) {                  // invariance residual (I - P) H P
            const double* P = w ? Ps : Pu;
            mmul(H2, P, M4, k);
            mmul(P, M4, M5, k);
            double res = 0.0, pn = 1.0;
            for (int i = 0; i < k * k; ++i) res = fmax(res, fabs(M4[i] - M5[i]));
            for (int i = 0; i < k; ++i) {
                double r = 0.0;
                for (int j = 0; j < k; ++j) r += fabs(P[i * k + j]);
                pn = fmax(pn, r);
            }
            // round-off in P grows with its norm (nearly parallel subspaces): ||P||^2 eps ||H||
            if (res > 1e-10 * h2 * pn * pn) return false;
        }
        nu = (int)floor(tu + 0.5);
        ns = k - nu;
        range_basis(Ps, k, ns, Wb, 0, Wk);
        range_basis(Pu, k, nu, Wb, ns, Wk);
        mmul(H2, Wb, M5, k);                           // T2 = Wb^-1 H22 Wb
        for (int i = 0; i < k * k; ++i) M4[i] = Wb[i];
        if (!lu(M4, k, M5, k)) return false;
    }
    // modal blocks of H22: columns of U = Wb diag(Vs, Vu) (k x k), eigenvalues rk/ik
    double rk[N2], ik[N2];
    for (int i = 0; i < k * k; ++i) V[i] = 0.0;
    const int off[2] = {0, ns}, sz[2] = {ns, nu};
    for (int b = 0; b < 2; ++b) {
        const int kb = sz[b], o = off[b];
        double Ab[MAXN * MAXN], Vb[MAXN * MAXN];
        if (kb > MAXN) return false;
        for (int i = 0; i < kb; ++i)
            for (int j = 0; j < kb; ++j) Ab[i * kb + j] = M5[(o + i) * k + o + j];
        if (!modal(Ab, kb, Vb, rk + o, ik + o)) return false;
        for (int i = 0; i < kb; ++i)
            for (int j = 0; j < kb; ++j) V[(o + i) * k + o + j] = Vb[i * kb + j];
    }
    mmul(Wb, V, M4, k);                                // U (k x k)
    // per modal block: X u (XU) and the lift y (Y, p x k): t = (N^it X u) Lam^-(it+1), y = sum t
    static double XU[N2 * N2], Y[N2 * N2];
    double lift[N2];
    for (int c = 0; c < k;) {
        const int w = (ik[c] != 0.0) ? 2 : 1;
        double t[2][N2], u2[2][N2];
        const double a = rk[c], bi = ik[c], d = a * a + bi * bi;
        for (int q = 0; q < w; ++q)
            for (int i = 0; i < p; ++i) {
                double s = 0.0;
                for (int l = 0; l < k; ++l) s += A[i * n + p + l] * M4[l * k + c + q];
                XU[i * k + c + q] = u2[q][i] = s;
                Y[i * k + c + q] = 0.0;
            }
        for (int it = 0; it < ni; ++it) {
            if (it > 0)
                for (int q = 0; q < w; ++q)
                    for (int i = 0; i < p; ++i) {
                        double s = 0.0;
                        for (int l = 0; l < p; ++l) s += A[i * n + l] * t[q][l];
                        u2[q][i] = s;
                    }
            for (int i = 0; i < p; ++i) {
                if (w == 1) t[0][i] = u2[0][i] / a;
                else {                                 // [t0 t1] Lam^-1, Lam = [[a, b], [-b, a]]
                    t[0][i] = (u2[0][i] * a + u2[1][i] * bi) / d;
                    t[1][i] = (u2[1][i] * a - u2[0][i] * bi) / d;
                }
                for (int q = 0; q < w; ++q) Y[i * k + c + q] += t[q][i];
            }
        }
        double yn = 0.0, un = 0.0;
        for (int q = 0; q < w; ++q) {
            for (int i = 0; i < p; ++i) yn = fmax(yn, fabs(Y[i * k + c + q]));
            for (int i = 0; i < k; ++i) un = fmax(un, fabs(M4[i * k + c + q]));
        }
        // a large lift (slow mode close to the zero block: ill-conditioned eigenvectors) keeps the
        // mode coupled to N in the centre, if its exponential stays moderate over the horizon
        lift[c] = (yn > YMAX * un && fabs(a) * g_->horizon() <= ZMAX) ? yn / un : 0.0;
        if (w == 2) lift[c + 1] = -1.0;
        c += w;
    }
    int r = 0;                                         // up to NSD slow dimensions, largest lifts first
    for (;;) {
        int best = -1;
        for (int c = 0; c < k; ++c)
            if (lift[c] > 0.0 && (best < 0 || lift[c] > lift[best])) best = c;
        if (best < 0) break;
        const int w = (ik[best] != 0.0) ? 2 : 1;
        if (r + w > NSD) { lift[best] = 0.0; continue; }
        lift[best] = -2.0;                             // -2: slow (both columns of a pair)
        if (w == 2) lift[best + 1] = -2.0;
        r += w;
    }
    // final columns: stable decoupled | N (p) | slow (r) | unstable decoupled
    const int nc = p + r;
    int col[N2], fc = 0;                               // col[c]: final column of H22 column c
    for (int pass = 0; pass < 3; ++pass) {
        if (pass == 1) fc += p;
        for (int c = 0; c < k;) {
            const int w = (ik[c] != 0.0) ? 2 : 1;
            const bool slow = lift[c] == -2.0, stable = c < ns;
            if ((pass == 0 && !slow && stable) || (pass == 1 && slow) || (pass == 2 && !slow && !stable))
                for (int q = 0; q < w; ++q) col[c + q] = fc++;
            c += w;
        }
    }
    int nsd = 0;
    for (int c = 0; c < ns; ++c) nsd += lift[c] != -2.0;
    double* re = re_[slot];
    double* im = im_[slot];
    double* Tc = Tc_[slot];
    for (int i = 0; i < n; ++i) re[i] = im[i] = 0.0;
    for (int i = 0; i < n * n; ++i) { Wk[i] = 0.0; V[i] = 0.0; }   // Wa (A coordinates), T (n x n)
    for (int j = 0; j < p; ++j) {
        Wk[j * n + nsd + j] = 1.0;
        for (int l = 0; l < p; ++l) V[(nsd + l) * n + nsd + j] = A[l * n + j];
    }
    for (int c = 0; c < k; ++c) {
        const int f = col[c];
        const bool slow = lift[c] == -2.0;
        re[f] = rk[c];
        im[f] = ik[c];
        for (int i = 0; i < k; ++i) Wk[(p + i) * n + f] = M4[i * k + c];
        if (slow)
            for (int i = 0; i < p; ++i) V[(nsd + i) * n + f] = XU[i * k + c];   // coupling X u
        else
            for (int i = 0; i < p; ++i) Wk[i * n + f] = Y[i * k + c];          // lift
    }
    for (int c = 0; c < k;) {                          // modal blocks of T
        const int f = col[c];
        V[f * n + f] = rk[c];
        if (ik[c] == 0.0) { ++c; continue; }
        V[f * n + f + 1] = ik[c];
        V[(f + 1) * n + f] = -ik[c];
        V[(f + 1) * n + f + 1] = rk[c];
        c += 2;
    }
    double* W = W_[slot];
    double* Wi = Wi_[slot];
    mmul(Q, Wk, Wb, n);                                // W = D Q Wa
    {   // check Hb (Q Wa) = (Q Wa) T
        mmul(H, Wb, M5, n);
        mmul(Wb, V, M4, n);
        double res = 0.0, wn = 0.0;
        for (int i = 0; i < n * n; ++i) { res = fmax(res, fabs(M5[i] - M4[i])); wn = fmax(wn, fabs(Wb[i])); }
        if (res > 1e-9 * hn * wn) return false;
    }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) W[i * n + j] = dsc[i] * Wb[i * n + j];
    double* cs = cs_[slot];                            // column norms: stored x is scaled by them
    for (int j = 0; j < n; ++j) {
        double q = 0.0;
        for (int i = 0; i < n; ++i) q += W[i * n + j] * W[i * n + j];
        cs[j] = sqrt(q);
    }
    for (int i = 0; i < n * n; ++i) { M4[i] = W[i]; Wi[i] = 0.0; }
    for (int i = 0; i < n; ++i) Wi[i * n + i] = 1.0;
    if (!lu(M4, n, Wi, n)) return false;
    double tcn = 0.0;
    for (int i = 0; i < nc; ++i) {
        double s = 0.0;
        for (int j = 0; j < nc; ++j) {
            Tc[i * nc + j] = V[(nsd + i) * n + nsd + j];
            s += fabs(Tc[i * nc + j]);
        }
        tcn = fmax(tcn, s);
    }
    SetBasis& b = sb_[slot];
    b.S = S;
    b.ns = (uint8_t)nsd;
    b.nc = (uint8_t)nc;
    b.nz = (uint8_t)p;
    b.ni = (uint8_t)ni;
    b.tcn = tcn;
    b.W = W;
    b.Wi = Wi;
    b.re = re;
    b.im = im;
    b.Tc = Tc;
    b.cs = cs;
    return true;
}

}  // namespace lqocp
