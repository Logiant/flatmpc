"""Independent reference solutions for the shared problem (validator side).

Built without the developer's (indirect / PMP) method:
  * Case A: closed-form minimum-energy solution. For each integrator chain of order k the flat
    output is the unique polynomial of degree 2k-1 that satisfies the 2k boundary conditions
    (Hermite interpolation; equivalent to the controllability-Gramian solution). It is solved in
    exact rational arithmetic (fractions.Fraction), so it is exact up to final float rounding.
    A scipy Gramian computation is used as a cross-check.
  * Case B: direct transcription as a convex QP. The input v is continuous piecewise linear on a
    uniform grid of N intervals (first-order hold), the dynamics are discretized EXACTLY for that
    input, the cost of the piecewise-linear input is integrated EXACTLY, and the polytope bounds
    are imposed at the nodes (which by convexity makes them hold for all t). Each grid solution
    is therefore a genuinely feasible trajectory of the continuous problem and its cost is an
    upper bound on the true optimum J*. Solved with Clarabel via cvxpy; refined over N and
    Richardson-extrapolated to estimate J* and the discretization error.
"""
import os
import sys
from fractions import Fraction
from math import factorial

import numpy as np
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "problem"))
import quad_problem as qp  # noqa: E402

ORDERS = qp.CHAIN_ORDERS
# Extra contract case (validator-defined, not in quad_problem): Umap=None, i.e. box bounds directly
# on v, on the case-A boundary conditions. Bounds chosen below the unconstrained peaks
# (6, 65.2, 3, 0) for the first three channels so that they activate.
_UB_C = np.array([5.0, 50.0, 2.5, 1.0])
CASE_C = dict(x0=qp.S0, xf=qp.SF, T=2.0, Umap=None, u_min=-_UB_C, u_max=_UB_C)
OFFS = np.cumsum((0,) + ORDERS)[:-1]
N_X, N_V = 12, 4
CACHE = os.path.join(HERE, "ref_cache.npz")


# ----------------------------------------------------------------------------------------------
# Case A: exact polynomial (Hermite) solution
# ----------------------------------------------------------------------------------------------
def _hermite_chain(a0, aT, k, T):
    """Coefficients c[0..2k-1] (Fractions) of y(t)=sum c_j t^j with y^(r)(0)=a0[r], y^(r)(T)=aT[r]."""
    n = 2 * k
    T = Fraction(T)
    rows, rhs = [], []
    for r in range(k):
        rows.append([Fraction(factorial(j), factorial(j - r)) * (0 if j != r else 1) if j >= r else Fraction(0)
                     for j in range(n)])
        rhs.append(Fraction(a0[r]))
    for r in range(k):
        rows.append([Fraction(factorial(j), factorial(j - r)) * T ** (j - r) if j >= r else Fraction(0)
                     for j in range(n)])
        rhs.append(Fraction(aT[r]))
    # Gaussian elimination in exact arithmetic
    M = [row[:] + [b] for row, b in zip(rows, rhs)]
    for c in range(n):
        p = next(i for i in range(c, n) if M[i][c] != 0)
        M[c], M[p] = M[p], M[c]
        for i in range(n):
            if i != c and M[i][c] != 0:
                f = M[i][c] / M[c][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


class PolyRef:
    """Exact case-A reference. Flat outputs are polynomials; v_i = y_i^(k_i)."""

    def __init__(self, x0, xf, T):
        self.T = T
        self.coefs = []  # list of Fraction coefficient lists (ascending powers)
        for j, k in enumerate(ORDERS):
            o = OFFS[j]
            self.coefs.append(_hermite_chain(x0[o:o + k], xf[o:o + k], k, T))
        # cost exactly: 1/2 sum_i int_0^T (y_i^(k))^2 dt
        J = Fraction(0)
        for j, k in enumerate(ORDERS):
            d = self._deriv(self.coefs[j], k)
            sq = [Fraction(0)] * (2 * len(d))
            for a, ca in enumerate(d):
                for b, cb in enumerate(d):
                    sq[a + b] += ca * cb
            J += sum(c * Fraction(self.T) ** (p + 1) / (p + 1) for p, c in enumerate(sq))
        self.cost_exact = J / 2
        self.cost = float(self.cost_exact)
        self.fcoefs = [[float(c) for c in cs] for cs in self.coefs]

    @staticmethod
    def _deriv(c, r):
        out = list(c)
        for _ in range(r):
            out = [out[j] * j for j in range(1, len(out))]
        return out

    def _evald(self, j, r, t):
        c = self.fcoefs[j]
        d = list(c)
        for _ in range(r):
            d = [d[i] * i for i in range(1, len(d))]
        return np.polyval(d[::-1], t) if d else np.zeros_like(t)

    def x(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        out = np.zeros((t.size, N_X))
        for j, k in enumerate(ORDERS):
            for r in range(k):
                out[:, OFFS[j] + r] = self._evald(j, r, t)
        return out

    def u(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        out = np.zeros((t.size, N_V))
        for j, k in enumerate(ORDERS):
            out[:, j] = self._evald(j, k, t)
        return out


def gramian_cost(x0, xf, T):
    """Numerical (scipy) Gramian minimum-energy cost, as a cross-check of PolyRef."""
    A, B = qp.Af, qp.Bf
    n, m = A.shape[0], B.shape[1]
    # Van Loan: expm([[-A, BB^T],[0, A^T]] T) -> Gramian
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A
    M[:n, n:] = B @ B.T
    M[n:, n:] = A.T
    E = expm(M * T)
    Phi = E[n:, n:].T  # e^{AT}
    W = Phi @ E[:n, n:]
    d = xf - Phi @ x0
    return 0.5 * d @ np.linalg.solve(W, d), W


# ----------------------------------------------------------------------------------------------
# Exact first-order-hold discretization and simulation helpers
# ----------------------------------------------------------------------------------------------
def foh_matrices(h, A=None, B=None):
    """x+ = Phi x + G0 a + G1 b for input linear from a (t=0) to b (t=h), exact."""
    A = qp.Af if A is None else A
    B = qp.Bf if B is None else B
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + 2 * m, n + 2 * m))
    M[:n, :n] = A
    M[:n, n:n + m] = B
    M[n:n + m, n + m:] = np.eye(m)
    E = expm(M * h)
    Phi, E1, E2 = E[:n, :n], E[:n, n:n + m], E[:n, n + m:]
    # input w(t) = a + (b-a) t/h  ->  x(h) = Phi x + E1 a + E2 (b-a)/h
    return Phi, E1 - E2 / h, E2 / h


def simulate_foh(t, U, x0):
    """Integrate s' = Af s + Bf v exactly for v linearly interpolated between samples (t_i, U_i).
    Works for nonuniform t. Returns X at the sample times."""
    X = np.zeros((len(t), N_X))
    X[0] = x0
    cache = {}
    for i in range(len(t) - 1):
        h = float(t[i + 1] - t[i])
        key = round(h, 15)
        if key not in cache:
            cache[key] = foh_matrices(h)
        Phi, G0, G1 = cache[key]
        X[i + 1] = Phi @ X[i] + G0 @ U[i] + G1 @ U[i + 1]
    return X


def simulate_gauss(ufun, x0, T, t_out, nsub=2000, order=8, breaks=()):
    """Integrate s(t) = e^{At} x0 + int_0^t e^{A(t-tau)} B v(tau) dtau with composite Gauss-Legendre
    quadrature (order points per subinterval), evaluating ufun (vectorized, (K,)->(K,4)) at the
    nodes. Because Af is nilpotent, e^{A s} is a finite polynomial and is computed exactly.
    Returns the states at t_out (each t_out must be on the subinterval grid or the quadrature is
    split there automatically)."""
    gx, gw = np.polynomial.legendre.leggauss(order)
    edges = np.union1d(np.linspace(0, T, nsub + 1), np.asarray(t_out, float))
    br = np.asarray([b for b in breaks if 0 < b < T], float)
    edges = np.union1d(edges, br)
    a, b = edges[:-1], edges[1:]
    mid, half = (a + b) / 2, (b - a) / 2
    tau = (mid[:, None] + half[:, None] * gx[None, :]).ravel()
    w = (half[:, None] * gw[None, :]).ravel()
    V = np.asarray(ufun(tau))
    A, B = qp.Af, qp.Bf
    out = []
    for to in np.atleast_1d(t_out):
        sel = tau < to
        x = expm(A * to) @ x0
        if np.any(sel):
            # e^{A(t-tau)} B v(tau): for chain j, component r gets (t-tau)^(k-1-r)/(k-1-r)! v_j
            dt = to - tau[sel]
            for j, k in enumerate(ORDERS):
                for r in range(k):
                    p = k - 1 - r
                    x[OFFS[j] + r] += np.sum(w[sel] * dt ** p / factorial(p) * V[sel, j])
        out.append(x)
    return np.array(out)


# ----------------------------------------------------------------------------------------------
# Case B: direct transcription QP
# ----------------------------------------------------------------------------------------------
def solve_qp(case, N, solver="CLARABEL", verbose=False, vscale=100.0):
    """Condensed transcription: variables are the node inputs only (W = V / vscale); the exact FOH
    terminal map gives 12 (row-normalized) equality constraints; node bounds on Umap v."""
    import cvxpy as cp
    c = qp.CASES[case] if isinstance(case, str) else case
    T, x0, xf = c["T"], c["x0"], c["xf"]
    h = T / N
    Phi, G0, G1 = foh_matrices(h)
    # terminal map: x_N = Phi^N x0 + sum_i Phi^(N-1-i) (G0 V_i + G1 V_{i+1})
    E = np.zeros((N_X, (N + 1) * N_V))
    P = np.eye(N_X)
    for i in range(N - 1, -1, -1):         # P = Phi^(N-1-i)
        E[:, i * N_V:(i + 1) * N_V] += P @ G0
        E[:, (i + 1) * N_V:(i + 2) * N_V] += P @ G1
        P = Phi @ P
    rhs = xf - P @ x0                       # P = Phi^N now
    E = E * vscale
    rn = np.linalg.norm(E, axis=1)
    E, rhs = E / rn[:, None], rhs / rn
    W = cp.Variable((N + 1, N_V))
    cons = [E @ cp.vec(W.T, order='F') == rhs]
    if c["u_max"] is not None:
        Um = c["Umap"] if c["Umap"] is not None else np.eye(N_V)
        un = np.linalg.norm(Um, axis=1)
        Ums = Um / un[:, None] * vscale
        U = W @ Ums.T
        cons += [U <= np.tile(c["u_max"] / un, (N + 1, 1)), U >= np.tile(c["u_min"] / un, (N + 1, 1))]
    # exact integral of |v|^2 for piecewise-linear v: h/3 (|a|^2 + a.b + |b|^2)
    #  = h/6 (|a|^2 + |b|^2 + |a+b|^2)
    cost = (h / 6) * (cp.sum_squares(W[:-1]) + cp.sum_squares(W[1:]) + cp.sum_squares(W[:-1] + W[1:]))
    prob = cp.Problem(cp.Minimize(0.5 * cost), cons)
    kw = {}
    if solver == "CLARABEL":
        kw = dict(tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12, max_iter=500)
    prob.solve(solver=solver, verbose=verbose, **kw)
    t = np.linspace(0, T, N + 1)
    Vv = W.value * vscale
    Xv = simulate_foh(t, Vv, x0)  # exact simulation of the piecewise-linear input
    Ja = foh_cost(h, Vv)
    mu = None
    if c["u_max"] is not None:
        # node multipliers of the motor bounds, per unit time (divide by h), signed: + upper, - lower
        mu = (cons[1].dual_value - cons[2].dual_value) / h
    return dict(t=t, V=Vv, X=Xv, cost=Ja, status=prob.status, N=N, mu=mu,
                bc_err=float(np.max(np.abs(Xv[-1] - xf))))


def gauss_energy(ufun, T, nsub=8000, order=8, breaks=()):
    """1/2 int_0^T |v|^2 dt by composite Gauss-Legendre (breakpoints optional)."""
    gx, gw = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(0, T, nsub + 1)
    br = np.asarray([b for b in breaks if 0 < b < T], float)
    edges = np.union1d(edges, br)
    a, b = edges[:-1], edges[1:]
    mid, half = (a + b) / 2, (b - a) / 2
    tau = (mid[:, None] + half[:, None] * gx[None, :]).ravel()
    w = (half[:, None] * gw[None, :]).ravel()
    V = np.asarray(ufun(tau))
    return 0.5 * np.sum(w * np.sum(V * V, axis=1))


def simpson_energy(t, U):
    """1/2 int |v|^2 by composite Simpson on a uniform grid with an even number of intervals."""
    from scipy.integrate import simpson
    return 0.5 * simpson(np.sum(U * U, axis=1), x=t)


def foh_cost(h, V):
    a, b = V[:-1], V[1:]
    return 0.5 * (h / 3) * np.sum(a * a + a * b + b * b)


def motor(V, case="B"):
    c = qp.CASES[case]
    Um = c["Umap"] if c["Umap"] is not None else np.eye(N_V)
    return V @ Um.T


def active_mask(V, case="B", tol=1e-7):
    c = qp.CASES[case]
    U = motor(V, case)
    return (U >= c["u_max"] - tol) | (U <= c["u_min"] + tol)


# ----------------------------------------------------------------------------------------------
# Pointwise projection onto the polytope {v : u_min <= Umap v <= u_max} (exact, by active-set
# enumeration; 4 dims, 81 sign patterns). Used for PMP checks and switch-time extraction.
# ----------------------------------------------------------------------------------------------
_PATTERNS = None


def _patterns():
    global _PATTERNS
    if _PATTERNS is None:
        import itertools
        pats = []
        for signs in itertools.product((0, 1, -1), repeat=4):
            pats.append(np.array(signs))
        pats.sort(key=lambda s: np.count_nonzero(s))
        _PATTERNS = pats
    return _PATTERNS


def project(p, Um, umin, umax, tol=1e-10):
    """argmin_v 1/2|v-p|^2 s.t. umin <= Um v <= umax. Returns (v, active-sign pattern).
    Exact active-set enumeration (81 patterns); returns the first pattern (fewest active) whose KKT
    conditions hold to a scaled tolerance, else the pattern with the smallest KKT violation."""
    up = Um @ p
    sc_mu = 1.0 + np.max(np.abs(up))
    sc_u = max(np.max(np.abs(umax)), np.max(np.abs(umin)))
    best = (np.inf, None, None)
    for s in _patterns():
        idx = np.nonzero(s)[0]
        if idx.size == 0:
            v = p.copy()
            viol_mu = 0.0
        else:
            G = Um[idx]                       # active rows
            g = np.where(s[idx] > 0, umax[idx], umin[idx])
            # v = p - G^T mu,  G v = g  ->  (G G^T) mu = G p - g
            mu = np.linalg.solve(G @ G.T, G @ p - g)
            v = p - G.T @ mu
            # multiplier sign: upper bound -> mu >= 0 ; lower bound -> mu <= 0
            # mu * |G_row|^2 is in thrust units; compare with the size of Um p
            viol_mu = max(0.0, np.max(-(mu * s[idx]) * np.sum(G * G, axis=1))) / sc_mu
        u = Um @ v
        viol_u = max(0.0, np.max(u - umax), np.max(umin - u)) / sc_u
        score = max(viol_mu, viol_u)
        if score <= tol:
            return v, s
        if score < best[0]:
            best = (score, v, s)
    return best[1], best[2]


def fit_costate_poly(t, V, free_mask):
    """On samples where no bound is active, v_j(t) = p_j(t) with p_j a polynomial of degree k_j-1
    (PMP for this problem: p = -B^T lambda, lambda polynomial). Fit p by least squares."""
    P = []
    for j, k in enumerate(ORDERS):
        P.append(np.polynomial.polynomial.polyfit(t[free_mask], V[free_mask, j], k - 1))
    return P


def eval_p(P, t):
    t = np.atleast_1d(t)
    return np.stack([np.polynomial.polynomial.polyval(t, c) for c in P], axis=1)


def switch_times_from_p(P, case, T, nfine=20001):
    c = qp.CASES[case]
    Um = c["Umap"] if c["Umap"] is not None else np.eye(N_V)
    tt = np.linspace(0, T, nfine)
    pats = [tuple(project(pp, Um, c["u_min"], c["u_max"])[1]) for pp in eval_p(P, tt)]
    sw = []
    for i in range(nfine - 1):
        if pats[i] != pats[i + 1]:
            lo, hi, plo = tt[i], tt[i + 1], pats[i]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                pm = tuple(project(eval_p(P, mid)[0], Um, c["u_min"], c["u_max"])[1])
                if pm == plo:
                    lo = mid
                else:
                    hi = mid
            sw.append(0.5 * (lo + hi))
    return sw, pats


def switch_times_from_mu(t, mu, rel=1e-3, nfit=8):
    """Switch times from the QP bound multipliers. A multiplier goes to zero linearly at an
    entry/exit point (the primal motor value leaves the bound tangentially, so primal detection is
    inaccurate). For every boundary of an active run of each motor, fit a line to the nfit
    nearest active nodes (excluding the endpoint nodes t=0, t=T, which carry half weight) and take
    its root."""
    thr = max(rel * np.max(np.abs(mu)), 1e-8)
    act = np.abs(mu) > thr
    n = len(t)
    out = []
    for j in range(mu.shape[1]):
        a = act[:, j]
        for i in range(n - 1):
            if a[i] != a[i + 1]:
                if a[i]:   # run ends at node i
                    idx = np.arange(max(1, i - nfit + 1), i + 1)
                else:      # run starts at node i+1
                    idx = np.arange(i + 1, min(n - 1, i + 1 + nfit))
                c1, c0 = np.polyfit(t[idx], mu[idx, j], 1)
                out.append(-c0 / c1)
    return sorted(out)


def switch_times_from_mask(t, mask):
    """Midpoints of grid intervals where the per-motor active pattern changes."""
    sw = []
    for i in range(len(t) - 1):
        if np.any(mask[i] != mask[i + 1]):
            sw.append(0.5 * (t[i] + t[i + 1]))
    return sw


# ----------------------------------------------------------------------------------------------
def build(Ns=(1000, 2000, 4000, 8000), verbose=True):
    """Build all references, print a convergence study and cache results to ref_cache.npz."""
    res = {}
    # ---- Case A
    cA = qp.CASES["A"]
    ref = PolyRef(cA["x0"], cA["xf"], cA["T"])
    Jg, W = gramian_cost(cA["x0"], cA["xf"], cA["T"])
    qA = solve_qp("A", 1000)
    tA = np.linspace(0, cA["T"], 1001)
    res["A_cost"] = ref.cost
    res["A_cost_gramian"] = Jg
    res["A_cost_qp1000"] = qA["cost"]
    if verbose:
        print("Case A: exact polynomial/Gramian reference")
        print(f"  J* (exact rational)      = {ref.cost:.15e}")
        print(f"  J  (scipy Gramian)       = {Jg:.15e}   rel diff {abs(Jg-ref.cost)/ref.cost:.2e}")
        print(f"  J  (FOH QP N=1000)       = {qA['cost']:.15e}   rel diff {abs(qA['cost']-ref.cost)/ref.cost:.2e}")
        print(f"  max|v_qp - v_exact|      = {np.max(np.abs(qA['V'] - ref.u(tA))):.2e}")
        print(f"  BC residual x(T)-xf      = {np.max(np.abs(ref.x(cA['T'])[0]-cA['xf'])):.2e}")
        print(f"  max|Umap v| (unbounded)  = {np.max(np.abs(ref.u(tA) @ qp.Umap.T)):.4f}")

    # ---- Case B
    cB = qp.CASES["B"]
    sols = {}
    if verbose:
        print("Case B: direct transcription (exact FOH, exact cost, node bounds => feasible upper bound)")
        print(f"  {'N':>5} {'status':>9} {'J_N':>22} {'J_N - J_2N':>12} {'max|v_N - v_2N|':>16} {'maxviol':>9}")
    for N in Ns:
        sols[N] = solve_qp("B", N)
    for i, N in enumerate(Ns):
        s = sols[N]
        U = motor(s["V"])
        viol = max(np.max(U - cB["u_max"]), np.max(cB["u_min"] - U), 0.0)
        if i + 1 < len(Ns):
            N2 = Ns[i + 1]
            dJ = s["cost"] - sols[N2]["cost"]
            dv = np.max(np.abs(sols[N2]["V"][::N2 // N] - s["V"]))
        else:
            dJ = dv = np.nan
        if verbose:
            print(f"  {N:>5} {s['status']:>9} {s['cost']:>22.15e} {dJ:>12.3e} {dv:>16.3e} {viol:>9.1e}")
    J = np.array([sols[N]["cost"] for N in Ns])
    Nf, Nc = Ns[-1], Ns[-2]
    fine, coarse = sols[Nf], sols[Nc]
    k = Nf // Nc
    # Each J_N is the exact cost of a feasible continuous trajectory -> upper bound on J*.
    # J_N decreases at high order; the last difference (incl. solver noise) bounds the error of J_Nf.
    errJ = max(abs(J[-2] - J[-1]), 1e-11 * J[-1])
    dv = np.max(np.abs(coarse["V"] - fine["V"][::k]))     # ~ error of v_Nf (first order near kinks)
    dx = np.max(np.abs(coarse["X"] - fine["X"][::k]))
    sw_mu = {N: switch_times_from_mu(sols[N]["t"], sols[N]["mu"]) for N in Ns}
    sw_f, sw_c = np.array(sw_mu[Nf]), np.array(sw_mu[Nc])
    sw_err = float(np.max(np.abs(sw_f - sw_c))) if len(sw_f) == len(sw_c) else np.nan
    far = np.ones(len(coarse["t"]), bool)
    for s_ in sw_f:
        far &= np.abs(coarse["t"] - s_) > 0.005
    dv_far = np.max(np.abs(coarse["V"] - fine["V"][::k])[far])
    # cross-check: PMP structure of the reference (v = proj(p(t)), p polynomial of degree k_i-1)
    free = ~np.any(active_mask(fine["V"], tol=1e-3), axis=1)
    P = fit_costate_poly(fine["t"], fine["V"], free)
    sw_p, _ = switch_times_from_p(P, "B", cB["T"])
    projV = np.array([project(pp, qp.Umap, cB["u_min"], cB["u_max"])[0] for pp in eval_p(P, fine["t"])])
    proj_err = np.max(np.abs(projV - fine["V"]))
    mask = active_mask(fine["V"], tol=1e-7)
    act_desc = []
    for s_ in [0.0] + list(sw_f):
        tt = s_ + 1e-3
        i = min(int(round(tt / cB["T"] * Nf)), Nf)
        act_desc.append(np.where(mask[i], np.sign(motor(fine["V"][i:i+1])[0]), 0).astype(int).tolist())
    if verbose:
        print(f"  J_ref = J_{Nf} = {J[-1]:.13e}  (feasible upper bound on J*), est. error {errJ:.1e} (rel {errJ/J[-1]:.1e})")
        print(f"  ref error estimates: max|v_{Nc} - v_{Nf}| = {dv:.2e} (|v|max = {np.max(np.abs(fine['V'])):.1f}),"
              f" away from switches (>5 ms) {dv_far:.2e}; max|x_{Nc} - x_{Nf}| = {dx:.2e}")
        print(f"  max |Umap v| on N={Nf}: {np.max(np.abs(motor(fine['V']))):.12f}")
        for N in Ns:
            print(f"  switch times (QP multiplier roots, N={N:>5}): {np.round(sw_mu[N], 6).tolist()}")
        print(f"  switch time ref error est. (N={Nc} vs {Nf}): {sw_err:.1e} s")
        print(f"  switch times (PMP poly fit of ref + projection, cross-check): {np.round(sw_p, 6).tolist()}")
        print(f"  active motors (sign) after t=0 and after each switch: {act_desc}")
        print(f"  PMP structure of reference: max|proj(p(t)) - v_QP| = {proj_err:.2e}")
    # ---- Case C (extra, identity bounds)
    sC = {N: solve_qp(CASE_C, N) for N in (4000, 8000)}
    swC = {N: switch_times_from_mu(sC[N]["t"], sC[N]["mu"]) for N in sC}
    C_dv = np.max(np.abs(sC[4000]["V"] - sC[8000]["V"][::2]))
    C_dx = np.max(np.abs(sC[4000]["X"] - sC[8000]["X"][::2]))
    C_errJ = max(abs(sC[4000]["cost"] - sC[8000]["cost"]), 1e-12 * sC[8000]["cost"])
    C_sw_err = float(np.max(np.abs(np.array(swC[4000]) - np.array(swC[8000])))) if len(swC[4000]) == len(swC[8000]) else np.nan
    if verbose:
        print("Case C (extra contract case: Umap=None, |v| <= [5,50,2.5,1], A's BCs, T=2)")
        print(f"  J_4000 = {sC[4000]['cost']:.13e}, J_8000 = {sC[8000]['cost']:.13e}")
        print(f"  switch times N=8000: {np.round(swC[8000], 6).tolist()} (est. err {C_sw_err:.1e})")
        print(f"  max|v_4000 - v_8000| = {C_dv:.2e}, max|x_4000 - x_8000| = {C_dx:.2e}")
    np.savez(CACHE, C_t=sC[8000]["t"], C_V=sC[8000]["V"], C_X=sC[8000]["X"], C_cost=sC[8000]["cost"],
             C_errJ=C_errJ, C_dv=C_dv, C_dx=C_dx, C_sw=np.array(swC[8000]), C_sw_err=C_sw_err,
             B_t=fine["t"], B_V=fine["V"], B_X=fine["X"], B_cost=J[-1], B_errJ=errJ,
             B_dv=dv, B_dv_far=dv_far, B_dx=dx, B_sw=sw_f, B_sw_err=sw_err, B_sw_p=np.array(sw_p),
             B_mu=fine["mu"], B_tC=coarse["t"], B_VC=coarse["V"], B_XC=coarse["X"],
             A_cost=ref.cost, A_cost_gramian=Jg, Ns=np.array(Ns), B_Js=J)
    return res


def load():
    d = np.load(CACHE)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    build()


# ================================================================================================
# Revision 2: exact PMP-certified reference for arbitrary cases (validator's own implementation)
# ------------------------------------------------------------------------------------------------
# For this convex problem the KKT conditions are sufficient: if v(t) = Proj_P(p(t)) with every p_j
# a polynomial of degree k_j-1 (p = -B^T lambda) and the resulting trajectory meets s(T)=xf, then v
# is THE global optimum. We find p by Newton on its 12 coefficients, warm-started from the QP above
# (or the unconstrained solution). Events are found exactly: within a fixed active pattern every
# condition (multiplier signs, inactive-motor feasibility) is a polynomial of degree <= 3 in t, so
# all pattern changes are among the real roots of those polynomials. Integration is exact
# (Gauss-Legendre on each polynomial piece). The certificate is the final BC residual, checked
# again by the generic integrator, and agreement with the QP cost (an upper bound).
# ================================================================================================
import itertools

_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)


def _theta_to_P(th):
    P = np.zeros((4, 4))
    P[0, :4], P[1, :4], P[2, :2], P[3, :2] = th[0:4], th[4:8], th[8:10], th[10:12]
    return P


def _P_to_theta(P):
    return np.concatenate([P[0, :4], P[1, :4], P[2, :2], P[3, :2]])


_TH_IDX = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (3, 0), (3, 1)]


def _pattern_list(n_motor=4):
    return [np.array(s) for s in itertools.product((0, 1, -1), repeat=n_motor)]


def _pattern_affine(s, Um, umin, umax):
    """For pattern s: v = Q p + r (Q 4x4, r 4) and multiplier mu = Mg p - mg (rows of active)."""
    idx = np.nonzero(s)[0]
    if idx.size == 0:
        return np.eye(4), np.zeros(4), idx, np.zeros((0, 4)), np.zeros(0)
    G = Um[idx]
    g = np.where(s[idx] > 0, umax[idx], umin[idx])
    Minv = np.linalg.inv(G @ G.T)
    Mg = Minv @ G
    mg = Minv @ g
    Q = np.eye(4) - G.T @ Mg
    r = G.T @ mg
    return Q, r, idx, Mg, mg


def _real_roots(c, lo, hi, tol_im=1e-7):
    """Real roots in (lo,hi) of polynomial with ascending coefficients c (also near-real ones)."""
    c = np.trim_zeros(np.asarray(c, float), "b")
    if c.size <= 1:
        return []
    sc = np.max(np.abs(c))
    if sc == 0:
        return []
    rr = np.roots((c / sc)[::-1])
    out = []
    for z in rr:
        if abs(z.imag) <= tol_im * max(1.0, abs(z)):
            x = z.real
            # polish by Newton on the polynomial
            dc = np.polynomial.polynomial.polyder(c)
            for _ in range(3):
                d = np.polynomial.polynomial.polyval(x, dc)
                if d != 0:
                    x = x - np.polynomial.polynomial.polyval(x, c) / d
            if lo < x < hi:
                out.append(x)
    return out


class PMPSolution:
    pass


def pmp_pieces(P, c):
    """Exact event detection: returns list of (t_a, t_b, pattern) with merged equal patterns."""
    T = c["T"]
    Um = np.eye(4) if c["Umap"] is None else np.asarray(c["Umap"], float)
    if c["u_max"] is None:
        return [(0.0, T, np.zeros(4, int))]
    umin, umax = np.asarray(c["u_min"], float), np.asarray(c["u_max"], float)
    cands = {0.0, T}
    for s in _pattern_list():
        Q, r, idx, Mg, mg = _pattern_affine(s, Um, umin, umax)
        V = Q @ P
        V[:, 0] += r                             # v coeffs (4 x 4)
        polys = []
        if idx.size:
            mu = Mg @ P
            mu[:, 0] -= mg
            polys += list(mu)
        U = Um @ V
        for i in range(4):
            if s[i] == 0:
                a = -U[i].copy(); a[0] += umax[i]; polys.append(a)
                b = U[i].copy(); b[0] -= umin[i]; polys.append(b)
        for pc in polys:
            cands.update(_real_roots(pc, 0.0, T))
    cands = np.array(sorted(cands))
    # drop duplicates closer than 1e-14
    keep = [cands[0]]
    for x in cands[1:]:
        if x - keep[-1] > 1e-14 * max(1.0, T):
            keep.append(x)
    cands = np.array(keep)
    pieces = []
    for a, b in zip(cands[:-1], cands[1:]):
        m = 0.5 * (a + b)
        pm = np.polynomial.polynomial.polyval(m, P.T)
        _, s = project(pm, Um, umin, umax)
        s = np.asarray(s, int)
        if pieces and np.array_equal(pieces[-1][2], s):
            pieces[-1] = (pieces[-1][0], b, s)
        else:
            pieces.append((a, b, s))
    return pieces


def _chain_kernel(dt):
    """K(dt) (12x4): e^{A dt} B for the flat model, for an array dt -> (len, 12, 4)."""
    dt = np.atleast_1d(dt)
    K = np.zeros((dt.size, N_X, N_V))
    for j, k in enumerate(ORDERS):
        for rr in range(k):
            p = k - 1 - rr
            K[:, OFFS[j] + rr, j] = dt ** p / factorial(p)
    return K


def pmp_eval(P, c, pieces=None):
    """Return (xT, dxT/dtheta (12x12), cost, pieces)."""
    T = c["T"]
    Um = np.eye(4) if c["Umap"] is None else np.asarray(c["Umap"], float)
    umin = None if c["u_min"] is None else np.asarray(c["u_min"], float)
    umax = None if c["u_max"] is None else np.asarray(c["u_max"], float)
    if pieces is None:
        pieces = pmp_pieces(P, c)
    x = expm(qp.Af * T) @ np.asarray(c["x0"], float)
    Jac = np.zeros((N_X, 12))
    cost = 0.0
    for a, b, s in pieces:
        if umax is None:
            Q, r = np.eye(4), np.zeros(4)
        else:
            Q, r, _, _, _ = _pattern_affine(s, Um, umin, umax)
        half, mid = (b - a) / 2, (b + a) / 2
        tau = mid + half * _GL_X
        w = half * _GL_W
        pt = np.polynomial.polynomial.polyval(tau, P.T).T   # (8,4)
        v = pt @ Q.T + r
        K = _chain_kernel(T - tau)                          # (8,12,4)
        x += np.einsum("q,qij,qj->i", w, K, v)
        cost += 0.5 * np.sum(w * np.sum(v * v, axis=1))
        KQ = np.einsum("qij,jl->qil", K, Q)                  # d/dp
        for col, (ch, pw) in enumerate(_TH_IDX):
            Jac[:, col] += np.einsum("q,qi->i", w * tau ** pw, KQ[:, :, ch])
    return x, Jac, cost, pieces


def pmp_solve(c, P0, maxit=60, tol=1e-11, verbose=False):
    th = _P_to_theta(P0)
    xf = np.asarray(c["xf"], float)
    XS = max(1.0, np.max(np.abs(xf)), np.max(np.abs(c["x0"])))
    hist = []
    for it in range(maxit):
        x, Jc, cost, pieces = pmp_eval(_theta_to_P(th), c)
        r = x - xf
        nr = np.max(np.abs(r)) / XS
        hist.append(nr)
        if verbose:
            print(f"    newton it {it}: |r|={nr:.2e} cost={cost:.12e} pieces={len(pieces)}")
        if nr < tol and (it > 0 or len(pieces) == 1):
            break
        d = np.linalg.lstsq(Jc, -r, rcond=None)[0]
        lam = 1.0
        for _ in range(30):
            xn, _, _, _ = pmp_eval(_theta_to_P(th + lam * d), c)
            if np.max(np.abs(xn - xf)) / XS < (1 - 1e-4 * lam) * nr or lam < 1e-6:
                break
            lam *= 0.5
        th = th + lam * d
    P = _theta_to_P(th)
    x, Jc, cost, pieces = pmp_eval(P, c)
    sol = PMPSolution()
    sol.P, sol.cost, sol.pieces, sol.c = P, cost, pieces, c
    sol.resid = np.max(np.abs(x - xf)) / XS
    sol.converged = sol.resid < 1e-9
    sol.iterations = it
    sol.switch_times = [p[0] for p in pieces[1:]]
    sol.patterns = [p[2].tolist() for p in pieces]
    Um = np.eye(4) if c["Umap"] is None else np.asarray(c["Umap"], float)

    def u(t):
        t = np.atleast_1d(np.asarray(t, float))
        pt = np.polynomial.polynomial.polyval(t, P.T).T
        if c["u_max"] is None:
            return pt
        return np.array([project(pp, Um, np.asarray(c["u_min"]), np.asarray(c["u_max"]))[0] for pp in pt])
    sol.u = u

    def xfun(t):
        return simulate_gauss(u, np.asarray(c["x0"], float), c["T"], np.atleast_1d(t), nsub=50,
                              order=8, breaks=sol.switch_times)
    sol.x = xfun
    return sol


def warm_start_P(c, N=4000):
    """Initial p from the QP solution (least squares on free nodes), or unconstrained exact."""
    if c["u_max"] is None:
        R = PolyRef(c["x0"], c["xf"], c["T"])
        return np.array([np.pad(np.polynomial.polynomial.polyder(np.array(R.fcoefs[j]), k)[:4], (0, 4))[:4]
                         for j, k in enumerate(ORDERS)]), None
    q = solve_qp(c, N)
    Um = np.eye(4) if c["Umap"] is None else np.asarray(c["Umap"], float)
    M = q["V"] @ Um.T
    free = np.all((M < np.asarray(c["u_max"]) - 1e-3 * np.max(np.abs(c["u_max"])))
                  & (M > np.asarray(c["u_min"]) + 1e-3 * np.max(np.abs(c["u_min"]))), axis=1)
    if free.sum() < 20:
        free = np.ones(len(q["t"]), bool)
    Pl = fit_costate_poly(q["t"], q["V"], free)
    P = np.zeros((4, 4))
    for j, cf in enumerate(Pl):
        P[j, :len(cf)] = cf
    return P, q


def unconstrained_P(c):
    R = PolyRef(c["x0"], c["xf"], c["T"])
    P = np.zeros((4, 4))
    for j, k in enumerate(ORDERS):
        d = np.polynomial.polynomial.polyder(np.array(R.fcoefs[j]), k)
        P[j, :len(d)] = d
    return P, R


# ------------------------------------------------------------------------------------------------
# Infeasibility certificate (continuous-time, Farkas): the reachable set of x(T) is
# e^{AT}x0 + {int_0^T e^{A(T-t)} B v(t) dt : v(t) in P}. xf is unreachable iff there is lambda with
#   int_0^T h_P(B^T e^{A^T (T-t)} lambda) dt < lambda^T (xf - e^{AT} x0),
# with h_P the support function of P = {v : u_min <= Umap v <= u_max}; for P = Umap^{-1} Box,
# h_P(w) = sum_i max(umax_i z_i, umin_i z_i), z = Umap^{-T} w. We minimise the left side subject to
# lambda^T d = 1 (an LP after quadrature). value < 1 => infeasible, value > 1 => feasible
# (the quadrature error of the Lipschitz integrand is bounded and reported).
# ------------------------------------------------------------------------------------------------
def feasibility_value(c, nq=4000):
    import cvxpy as cp
    T = c["T"]
    Um = np.eye(4) if c["Umap"] is None else np.asarray(c["Umap"], float)
    umin, umax = np.asarray(c["u_min"], float), np.asarray(c["u_max"], float)
    d = np.asarray(c["xf"], float) - expm(qp.Af * T) @ np.asarray(c["x0"], float)
    t = (np.arange(nq) + 0.5) * T / nq                     # midpoint rule
    K = _chain_kernel(T - t)                                # (nq,12,4): B^T e^{A^T(T-t)} lambda = K^T lambda
    UmiT = np.linalg.inv(Um).T
    lam = cp.Variable(N_X)
    Z = cp.vstack([UmiT @ (K[q].T @ lam) for q in range(0, nq)])  # (nq,4)
    obj = cp.sum(cp.maximum(Z @ np.diag(umax), Z @ np.diag(umin))) * (T / nq)
    prob = cp.Problem(cp.Minimize(obj), [d @ lam == 1])
    prob.solve(solver="CLARABEL")
    return prob.value, lam.value
