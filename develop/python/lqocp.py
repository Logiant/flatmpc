"""Indirect (motion-primitive) solver for fixed-time transfers of integrator chains.

Energy cost (Q=0, R=I, fixed endpoint): methods "newton" (default, Newton on the 12 polynomial
coefficients, exact cubic events) and "tau" (junction-time Newton). General LQ cost
(Q, R, Qf; spec Revision 3): "newton" (default) = stable-basis motion primitives with Newton on
the junction times (NOTES.md section 9); "riccati" = the earlier semismooth Newton (PDAS) with a
Riccati-sweep linear BVP and continuation in the bound scale (NOTES.md section 8). For the energy
cost "riccati" runs the "newton" polynomial path. See develop/NOTES.md.

Original description of the energy method:
Indirect (motion-primitive) solver for energy-optimal, fixed-time, fixed-endpoint transfers of
integrator chains with a polytopic input constraint  u_min <= Umap v <= u_max.

Method (Beaver & Malikopoulos, arXiv 2103.03339; Beaver, arXiv 2310.00168), specialised to
    min 1/2 int_0^T |v|^2 dt,   s' = A s + B v  (Brunovsky chains),  s(0)=x0, s(T)=xf.

Costate-free optimality (Thm 1/2 of 2103.03339) with the pure input constraint g = M v - b <= 0:
for chain j of order k_j, Euler-Poisson gives d^{k_j}/dt^{k_j} (v_j + (M^T mu)_j) = 0, so the
"switching polynomial"
        w(t) := v(t) + M^T mu(t)       (= -lambda_last(t), the last costate of each chain)
is a polynomial of degree k_j-1 in each component on EVERY arc. The Bryson-Ho jump conditions
have N = {} (relative degree 0 constraint, no tangency rows, pi = 0), so all costates are
continuous across junctions; by Thm 1 this means w and its first k_j-1 derivatives are
continuous, i.e. w is ONE polynomial on [0, T] with n = sum k_j coefficients c.
On an arc with active (signed) rows A v = b:  v = P_A w + q_A  (affine in c), with
P_A = I - A^T (A A^T)^-1 A,  q_A = A^T (A A^T)^-1 b,  mu = (A A^T)^-1 (A w - b) >= 0.
Junction condition (Cor. 2: v continuous) for a row r that switches: on the side where r is
inactive, a_r v(tau) = b_r.

Algorithm: for fixed junction times and active-set sequence the boundary conditions are LINEAR
in c (12x12 solve, P1 eq. (72)). An outer Gauss-Newton solves the scalar junction conditions
for the junction times. The active-set sequence is found with the "solve, check violations,
impose, repeat" heuristic of 2310.00168 Sec. IV-D, using the projection of the current w(t)
onto the polytope (pointwise Hamiltonian minimisation) to read off the sequence.
"""
import time
import math
import numpy as np

__all__ = ["solve", "Solution"]

# 4-point Gauss-Legendre on [-1, 1]: exact for polynomials of degree <= 7. All arc integrands
# here have degree <= 2*max(k)-2 = 6, so every integral below is exact up to round-off.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(4)


def _detect_chains(A, B):
    """Return chain orders and offsets if (A, B) is in contiguous Brunovsky form."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    n, m = B.shape
    orders = []
    for j in range(m):
        col = B[:, j]
        nz = np.flatnonzero(col)
        if len(nz) != 1 or col[nz[0]] != 1.0:
            raise ValueError("B is not in Brunovsky (integrator chain) form")
        orders.append(nz[0] + 1 - sum(orders))
    if sum(orders) != n or min(orders) < 1:
        raise ValueError("states are not ordered chain by chain")
    Aref = np.zeros((n, n))
    i = 0
    for k in orders:
        for r in range(k - 1):
            Aref[i + r, i + r + 1] = 1.0
        i += k
    if not np.array_equal(A, Aref):
        raise ValueError("A is not a set of pure integrator chains")
    offs = np.concatenate([[0], np.cumsum(orders)[:-1]]).astype(int)
    return tuple(int(k) for k in orders), tuple(int(o) for o in offs)


class _Model:
    def __init__(self, orders, offs, M, lo, hi):
        self.orders, self.offs = orders, offs
        self.m = len(orders)
        self.n = int(sum(orders))
        self.M, self.lo, self.hi = M, lo, hi
        # signed constraint rows: key 2*r -> upper (M_r v <= hi_r), 2*r+1 -> lower (-M_r v <= -lo_r)
        self.rows, self.rhs = [], []
        for r in range(self.m):
            self.rows += [M[r], -M[r]]
            self.rhs += [hi[r], -lo[r]]
        self.rows = np.array(self.rows)
        self.rhs = np.array(self.rhs)
        self.has_bounds = bool(np.any(np.isfinite(self.rhs)))
        # candidate active sets for the pointwise projection (each row: inactive/upper/lower)
        sets = []
        for code in range(3 ** self.m):
            s, cc = [], code
            for r in range(self.m):
                d = cc % 3
                cc //= 3
                if d == 1:
                    s.append(2 * r)
                elif d == 2:
                    s.append(2 * r + 1)
            if all(np.isfinite(self.rhs[k]) for k in s):
                sets.append(tuple(s))
        sets.sort(key=len)
        self.sets = sets
        self._aff = {}

    # ---- polynomial w(t) = W(t) c ----
    def W(self, t):
        Wm = np.zeros((self.m, self.n))
        for j, (k, o) in enumerate(zip(self.orders, self.offs)):
            Wm[j, o:o + k] = t ** np.arange(k)
        return Wm

    def Phi(self, dt):
        """State transition expm(A dt) for the chains."""
        P = np.zeros((self.n, self.n))
        for k, o in zip(self.orders, self.offs):
            for r in range(k):
                for l in range(r, k):
                    P[o + r, o + l] = dt ** (l - r) / math.factorial(l - r)
        return P

    def PhiB(self, dt):
        """expm(A dt) B."""
        K = np.zeros((self.n, self.m))
        for j, (k, o) in enumerate(zip(self.orders, self.offs)):
            for r in range(k):
                K[o + r, j] = dt ** (k - 1 - r) / math.factorial(k - 1 - r)
        return K

    def affine(self, S):
        """v = P w + q on an arc with active set S (tuple of signed-row keys)."""
        if S in self._aff:
            return self._aff[S]
        if len(S) == 0:
            res = (np.eye(self.m), np.zeros(self.m))
        else:
            Aa = self.rows[list(S)]
            ba = self.rhs[list(S)]
            G = Aa @ Aa.T
            X = np.linalg.solve(G, Aa)
            res = (np.eye(self.m) - Aa.T @ X, Aa.T @ np.linalg.solve(G, ba))
        self._aff[S] = res
        return res

    def project(self, w, guess=None):
        """argmin_v 1/2|v-w|^2 s.t. lo <= M v <= hi (= pointwise Hamiltonian minimiser).
        Exact enumeration of the 3^m active sets (KKT check), trying `guess` first."""
        cands = self.sets if guess is None else [guess] + self.sets
        tol = 1e-12
        for S in cands:
            if len(S) == 0:
                v = w
            else:
                Aa = self.rows[list(S)]
                G = Aa @ Aa.T
                try:
                    mu = np.linalg.solve(G, Aa @ w - self.rhs[list(S)])
                except np.linalg.LinAlgError:
                    continue
                if np.any(mu < -tol * (1 + np.abs(mu).max())):
                    continue
                v = w - Aa.T @ mu
            g = self.rows @ v - self.rhs
            scale = 1e-10 * (1 + np.abs(self.rhs[np.isfinite(self.rhs)]).max(initial=0))
            ok = True
            for kk in range(len(g)):
                if kk in S or not np.isfinite(self.rhs[kk]):
                    continue
                if g[kk] > scale:
                    ok = False
                    break
            if ok:
                return v, S
        return w, ()  # unreachable for a nonempty polytope


class _Problem:
    def __init__(self, model, x0, xf, T):
        self.md, self.x0, self.xf, self.T = model, x0, xf, T
        self.free = model.Phi(T) @ x0

    def arcs(self, seq, tau):
        edges = [0.0] + list(tau) + [self.T]
        return [(edges[i], edges[i + 1], seq[i]) for i in range(len(seq))]

    def assemble(self, arcs):
        """s(T) = G c + h for the given arc structure (exact, Gauss-Legendre per arc).
        G = sum_arcs int e^{A(T-t)} B P_S W(t) dt is also the exact Jacobian ds(T)/dc."""
        md, T = self.md, self.T
        G = np.zeros((md.n, md.n))
        h = self.free.copy()
        for a, b, S in arcs:
            if b <= a:
                continue
            P, q = md.affine(S)
            hw = 0.5 * (b - a)
            for xg, wg in zip(_GL_X, _GL_W):
                t = a + hw * (xg + 1.0)
                K = md.PhiB(T - t) * (wg * hw)
                G += K @ P @ md.W(t)
                h += K @ q
        return G, h

    def solve_c(self, arcs):
        """Linear BC solve for the polynomial coefficients c given the arc structure."""
        G, h = self.assemble(arcs)
        return np.linalg.solve(G, self.xf - h)

    def v_on(self, S, c, t):
        P, q = self.md.affine(S)
        return P @ (self.md.W(t) @ c) + q

    def detect(self, c, N=2000):
        """Active-set sequence and approximate switch times from the projection of w(t)."""
        md = self.md
        ts = np.linspace(0.0, self.T, N + 1)
        seq, tau, S = [], [], None
        for t in ts:
            _, Snew = md.project(md.W(t) @ c, S)
            Snew = tuple(sorted(Snew))
            if S is None:
                seq.append(Snew)
            elif Snew != S:
                seq.append(Snew)
                tau.append(t - 0.5 * self.T / N)
            S = Snew
        return seq, np.array(tau)

    def junction_res(self, seq, tau, c):
        md = self.md
        res = []
        for k, t in enumerate(tau):
            L, R = set(seq[k]), set(seq[k + 1])
            for key in sorted(L ^ R):
                side = seq[k] if key in R else seq[k + 1]   # side on which `key` is inactive
                v = self.v_on(side, c, t)
                res.append(md.rows[key] @ v - md.rhs[key])
        return np.array(res)

    def residual(self, seq, tau):
        c = self.solve_c(self.arcs(seq, tau))
        return self.junction_res(seq, tau, c), c

    def pmp_violation(self, seq, tau, c, N=2000):
        """max |v_arc(t) - Proj(w(t))| on a grid: zero iff the arc structure minimises H."""
        md = self.md
        arcs = self.arcs(seq, tau)
        err, ai = 0.0, 0
        for t in np.linspace(0.0, self.T, N + 1):
            while ai < len(arcs) - 1 and t > arcs[ai][1]:
                ai += 1
            v_arc = self.v_on(arcs[ai][2], c, t)
            v_opt, _ = md.project(md.W(t) @ c, arcs[ai][2])
            err = max(err, np.abs(v_arc - v_opt).max())
        return err

    def newton_tau(self, seq, tau, tol=1e-12, maxit=50):
        """Gauss-Newton on the junction times (sequence fixed), inner linear solve for c."""
        T = self.T
        tau = np.array(tau, float)
        nj = len(tau)
        r, c = self.residual(seq, tau)
        it = 0
        if nj == 0:
            return tau, c, 0, True, 0.0
        gap = 1e-9 * T
        for it in range(1, maxit + 1):
            if np.abs(r).max() < tol:
                return tau, c, it - 1, True, np.abs(r).max()
            J = np.zeros((len(r), nj))
            for i in range(nj):
                h = 1e-7 * max(1.0, T)
                tp = tau.copy()
                tp[i] += h
                tm = tau.copy()
                tm[i] -= h
                J[:, i] = (self.residual(seq, tp)[0] - self.residual(seq, tm)[0]) / (2 * h)
            step = np.linalg.lstsq(J, -r, rcond=None)[0]
            alpha, accepted = 1.0, False
            nr0 = np.linalg.norm(r)
            while alpha > 1e-6:
                tn = tau + alpha * step
                e = np.concatenate([[0.0], tn, [T]])
                if np.all(np.diff(e) > gap):
                    rn, cn = self.residual(seq, tn)
                    if np.linalg.norm(rn) < (1 - 1e-4 * alpha) * nr0 or np.linalg.norm(rn) < tol:
                        accepted = True
                        break
                alpha *= 0.5
            if not accepted:
                return tau, c, it, False, np.abs(r).max()
            tau, r, c = tn, rn, cn
        return tau, c, maxit, bool(np.abs(r).max() < tol), np.abs(r).max()


class Solution:
    """Piecewise closed-form trajectory: v = P_S w(t) + q_S on each arc, w a fixed polynomial."""

    def __init__(self, prob, seq, tau, c, info):
        self._p = prob
        self._seq = list(seq)
        self._tau = np.array(tau, float)
        self.c = c
        self.switch_times = sorted(float(t) for t in self._tau)
        self.active_sets = [tuple(int(k) for k in S) for S in self._seq]
        self.info = info
        self._arcs = prob.arcs(self._seq, self._tau)
        self.cost = float(self._cost())

    def _arc_index(self, t):
        return int(np.searchsorted(self._tau, t, side="right"))

    def _cost(self):
        J = 0.0
        for a, b, S in self._arcs:
            hw = 0.5 * (b - a)
            for xg, wg in zip(_GL_X, _GL_W):
                v = self._p.v_on(S, self.c, a + hw * (xg + 1.0))
                J += 0.5 * wg * hw * (v @ v)
        return J

    def _u1(self, t):
        t = min(max(float(t), 0.0), self._p.T)
        return self._p.v_on(self._seq[self._arc_index(t)], self.c, t)

    def _x1(self, t):
        p = self._p
        md = p.md
        t = min(max(float(t), 0.0), p.T)
        s = md.Phi(t) @ p.x0
        for a, b, S in self._arcs:
            if a >= t:
                break
            b = min(b, t)
            hw = 0.5 * (b - a)
            for xg, wg in zip(_GL_X, _GL_W):
                tt = a + hw * (xg + 1.0)
                s = s + (wg * hw) * (md.PhiB(t - tt) @ p.v_on(S, self.c, tt))
        return s

    def u(self, t):
        if np.ndim(t) == 0:
            return self._u1(t)
        return np.array([self._u1(ti) for ti in np.ravel(t)])

    def x(self, t):
        if np.ndim(t) == 0:
            return self._x1(t)
        return np.array([self._x1(ti) for ti in np.ravel(t)])


def _solve_tau(prob, max_outer=50, tol=1e-12, time_limit=30.0, max_newton=500):
    md = prob.md
    info_fail = None
    ectx = _NewtonCtx(prob)

    def detect(c):
        # exact (grid-free) event walk of the newton method; grid only as a fallback
        wk = ectx.walk(c)
        if wk is None:
            return prob.detect(c)
        return list(wk[0]), np.array(wk[1])

    # 1) unconstrained primitive: one linear BC solve
    seq, tau = [()], np.zeros(0)
    c = prob.solve_c(prob.arcs(seq, tau))
    iters, newton_its, outer, converged, res = 1, 0, 0, True, 0.0
    if md.has_bounds:
        converged = False
        history = set()
        t_start = time.perf_counter()
        c0n = 1.0 + np.abs(c).max()
        for outer in range(1, max_outer + 1):
            # early exits: wall-clock cap, and a diverging w (dual unbounded = infeasible)
            if time.perf_counter() - t_start > time_limit or newton_its > max_newton:
                info_fail = "iteration/time limit"
                break
            if np.abs(c).max() > 1e6 * c0n:
                info_fail = "infeasible (switching polynomial diverges: dual unbounded)"
                break
            # 2) read the active-set sequence off the pointwise H-minimiser of the current w(t)
            seq_d, tau_d = detect(c)
            if len(seq_d) == 1 and seq_d[0] == () and outer == 1:
                converged = True       # unconstrained optimum is feasible
                seq, tau = seq_d, tau_d
                break
            # 3) junction-time Newton for this sequence (linear solve for c inside)
            tau_n, c_n, k, ok, res = prob.newton_tau(seq_d, tau_d, tol=tol)
            newton_its += k
            iters += 1 + k
            if ok:
                seq, tau, c = seq_d, tau_n, c_n
                if ectx.verify(prob.arcs(seq, tau), c) <= 1e-9 * ectx.R:
                    converged = True
                    break
            else:
                # Newton failed for this sequence: take the plain "fix junctions, solve c" step
                # (fixed-point update of w), then re-detect the sequence from the new w.
                key = (tuple(seq_d), tuple(np.round(tau_d, 6)))
                c = prob.solve_c(prob.arcs(seq_d, tau_d))
                if key in history:
                    break
                history.add(key)
                seq, tau = seq_d, tau_d
    info = {
        "iterations": int(iters),
        "converged": bool(converged),
        "outer_iterations": int(outer),
        "newton_iterations": int(newton_its),
        "junction_residual": float(res),
        "sequence": [list(S) for S in seq],
    }
    if not converged and info_fail:
        info["failure"] = info_fail
    return seq, tau, c, info


def solve(A, B, x0, xf, T, Umap=None, u_min=None, u_max=None, Q=None, R=None, Qf=None,
          method="newton", max_outer=50, tol=1e-12):
    """method="newton" (default): Newton with exact, grid-free event detection.
    method="tau": junction-time Newton with grid-detected sequence (energy cost only).
    Q=None -> 0, R=None -> I, Qf=None -> fixed endpoint s(T) = xf."""
    t_start = time.perf_counter()
    if method not in ("tau", "newton", "riccati"):
        raise ValueError("method must be 'newton', 'tau' or 'riccati'")
    orders, offs = _detect_chains(A, B)
    m = len(orders)
    n = int(sum(orders))
    M = np.eye(m) if Umap is None else np.asarray(Umap, float)
    lo = np.full(m, -np.inf) if u_min is None else np.asarray(u_min, float).copy()
    hi = np.full(m, np.inf) if u_max is None else np.asarray(u_max, float).copy()
    T = float(T)
    x0 = np.asarray(x0, float)
    xf = np.asarray(xf, float)
    Qm = np.zeros((n, n)) if Q is None else np.asarray(Q, float)
    Rm = np.eye(m) if R is None else np.asarray(R, float)
    energy = (not np.any(Qm)) and np.array_equal(Rm, np.eye(m)) and Qf is None
    if not energy:
        if method == "tau":
            raise NotImplementedError("method='tau' supports only the energy cost (Q=0, R=I, Qf=None)")
        solver = _solve_stable if method == "newton" else _solve_general
        sol = solver(np.asarray(A, float), np.asarray(B, float), x0, xf, T, M, lo, hi,
                             0.5 * (Qm + Qm.T), 0.5 * (Rm + Rm.T),
                             None if Qf is None else 0.5 * (np.asarray(Qf, float) + np.asarray(Qf, float).T))
        sol.info["method"] = method
        sol.info["solve_time_s"] = time.perf_counter() - t_start
        return sol
    md = _Model(orders, offs, M, lo, hi)
    prob = _Problem(md, x0, xf, T)
    if method == "riccati":
        method = "newton"            # energy cost: riccati is not used, the polynomial path is
    if method == "tau":
        seq, tau, c, info = _solve_tau(prob, max_outer=max_outer, tol=tol)
    else:
        seq, tau, c, info = _solve_newton(prob)
    info["method"] = method
    info["solve_time_s"] = time.perf_counter() - t_start
    return Solution(prob, seq, tau, c, info)


# =====================================================================================
# method = "newton": Newton on c with exact (grid-free) event detection
# =====================================================================================
_EPS = np.finfo(float).eps


def _pval(b, s):
    r = 0.0
    for x in reversed(b):
        r = r * s + x
    return r


def _shift(a, t0):
    """Coefficients (ascending) of p(t0 + s) given those of p(t)."""
    d = len(a)
    return np.array([sum(a[p] * math.comb(p, i) * t0 ** (p - i) for p in range(i, d))
                     for i in range(d)])


def _scale(b, L):
    return float(sum(abs(x) * L ** i for i, x in enumerate(b)))


def _strip(b, L, tol):
    """Divide out the root at s = 0: drop leading coefficients with |b_i| L^i <= tol.
    Returns (reduced coefficients, multiplicity k) or (None, k) if the polynomial is ~0."""
    k = 0
    while k < len(b) and abs(b[k]) * L ** k <= tol:
        k += 1
    if k == len(b):
        return None, k
    red = np.array(b[k:], float)
    # drop negligible trailing coefficients (keeps derivative roots well defined)
    while len(red) > 1 and abs(red[-1]) * L ** (len(red) - 1 + k) <= tol * _EPS:
        red = red[:-1]
    return red, k


def _roots_le2(d):
    """Real roots of a polynomial of degree <= 2 (ascending coefficients), stable formulas."""
    d = list(d)
    while d and d[-1] == 0.0:
        d.pop()
    if len(d) <= 1:
        return []
    if len(d) == 2:
        return [-d[0] / d[1]]
    c0, b1, a2 = d
    disc = b1 * b1 - 4 * a2 * c0
    if disc < 0:
        if disc > -1e-14 * (b1 * b1 + abs(4 * a2 * c0)):
            return [-b1 / (2 * a2)]       # double root within round-off
        return []
    sq = math.sqrt(disc)
    qq = -0.5 * (b1 + math.copysign(sq, b1))
    r = []
    if qq != 0.0:
        r += [qq / a2, c0 / qq]
    else:
        r += [0.0]
    return sorted(r)


def _bisect(r, a, b):
    fa = _pval(r, a)
    for _ in range(200):
        mid = 0.5 * (a + b)
        if mid <= a or mid >= b:
            break
        fm = _pval(r, mid)
        if (fm > 0) == (fa > 0) and fm != 0.0:
            a, fa = mid, fm
        else:
            b = mid
    return 0.5 * (a + b)


def _crossings(red, k, L, tol):
    """Sign changes of phi(s) = s^k red(s) on (0, L].
    Returns (list of (root, new_sign, simple), list of tangential touch points).
    A critical point with |phi| <= tol and no sign change across it is a touch (double root);
    a sign change across such a point is an odd-multiplicity root located at that point."""
    if len(red) == 1:
        return [], []
    d = [i * red[i] for i in range(1, len(red))]
    crit = sorted({x for x in _roots_le2(d) if 0.0 < x < L})
    pts = [0.0] + crit + [L]
    vals = [red[0]] + [_pval(red, x) for x in crit] + [_pval(red, L)]
    out, touches, zeros = [], [], []
    sig = 1.0 if vals[0] > 0 else -1.0
    for i in range(1, len(pts)):
        v, x = vals[i], pts[i]
        if abs(v) * x ** k <= tol:
            if i < len(pts) - 1:
                zeros.append(i)
            continue
        sg = 1.0 if v > 0 else -1.0
        if sg != sig:
            if vals[i - 1] * v <= 0 and (i - 1 not in zeros or (vals[i - 1] > 0) != (v > 0)):
                root = _bisect(red, pts[i - 1], x)
                out.append((root, sg, not zeros))
            else:
                out.append((pts[zeros[-1]], sg, False))      # odd root at a flat point
            touches += [pts[z] for z in zeros[:-1]]
            sig = sg
        else:
            touches += [pts[z] for z in zeros]
        zeros = []
    return out, touches


class _NewtonCtx:
    def __init__(self, prob):
        self.p = prob
        md = prob.md
        self.md = md
        self.T = prob.T
        fin = np.isfinite(md.rhs)
        self.R = 1.0 + (np.abs(md.rhs[fin]).max() if fin.any() else 0.0)
        self.rownorm2 = np.sum(md.rows ** 2, axis=1)
        self.identity = md.has_bounds and np.array_equal(md.M, np.eye(md.m))
        # c = Lmat nu, where w(t) = (e^{A(T-t)}B)^T nu  (nu = multiplier of s(T) = xf)
        n, T = md.n, self.T
        Lm = np.zeros((n, n))
        for k, o in zip(md.orders, md.offs):
            for r in range(k):
                mm = k - 1 - r
                for p in range(mm + 1):
                    Lm[o + p, o + r] = math.comb(mm, p) * T ** (mm - p) * (-1) ** p / math.factorial(mm)
        self.Lm = Lm
        self.Linv = np.linalg.inv(Lm)
        self._mu = {}

    def wcoef(self, c):
        md = self.md
        Wc = np.zeros((md.m, 4))
        for j, (k, o) in enumerate(zip(md.orders, md.offs)):
            Wc[j, :k] = c[o:o + k]
        return Wc

    def row_polys(self, S, c):
        """phi_key(t) (coefficients in t) for every finite row: gap for inactive rows and
        -mu*|a|^2 for active rows (both must stay <= 0; same units)."""
        md = self.md
        Wc = self.wcoef(c)
        P, q = md.affine(S)
        V = P @ Wc
        V[:, 0] += q
        out = {}
        for key in range(2 * md.m):
            if not np.isfinite(md.rhs[key]) or key in S:
                continue
            g = md.rows[key] @ V
            g[0] -= md.rhs[key]
            out[key] = g
        if S:
            A = md.rows[list(S)]
            rhs = np.zeros((len(S), 4))
            rhs[:, 0] = md.rhs[list(S)]
            mu = np.linalg.solve(A @ A.T, A @ Wc - rhs)
            for i, key in enumerate(S):
                out[key] = -mu[i] * self.rownorm2[key]
        return out

    def valid_set(self, S):
        md = self.md
        rws = [k // 2 for k in S]
        return len(set(rws)) == len(rws) and len(S) <= md.m and all(np.isfinite(md.rhs[k]) for k in S)

    def lex_ok(self, S, c, t):
        """One-sided KKT at t+: every phi is <= 0 on a right neighbourhood of t
        (first non-negligible Taylor coefficient negative)."""
        for key, a in self.row_polys(S, c).items():
            b = _shift(a, t)
            tol = 1e-12 * (_scale(b, self.T) + self.R)
            for i, x in enumerate(b):
                if abs(x) * self.T ** i > tol:
                    if x > 0:
                        return False
                    break
        return True

    def lex_set(self, c, t, guess):
        for S in ([guess] if guess is not None else []) + self.md.sets:
            if self.valid_set(S) and self.lex_ok(S, c, t):
                return S
        return None

    # ---- event walk ----
    def walk(self, c, max_arcs=200):
        if self.identity:
            return self.walk_box(c)
        T, md = self.T, self.md
        S = self.lex_set(c, 0.0, ())         # (a) initial set: one-sided KKT at t = 0+
        if S is None:
            return None
        seq, tau, touches, ndeg, t = [S], [], [], 0, 0.0
        stall = 0
        while True:
            L = T - t
            cand, tch = [], []
            for key, a in self.row_polys(S, c).items():
                b = _shift(a, t)
                tol = 1e-12 * (_scale(b, L) + self.R)
                red, k = _strip(b, L, tol)
                if red is None:
                    continue
                if red[0] > 0:                 # inconsistent at t+: immediate (degenerate) event
                    cand.append((0.0, key, False))
                    continue
                cr, tc = _crossings(red, k, L, tol)
                if cr:
                    cand.append((cr[0][0], key, cr[0][2]))
                    tch += [(x, key) for x in tc if x < cr[0][0]]
                else:
                    tch += [(x, key) for x in tc]
            s1 = min([x[0] for x in cand], default=None)
            touches += [t + x for x, _ in tch if s1 is None or x < s1]
            if s1 is None or t + s1 >= T * (1 - 1e-14):
                break
            group = [x for x in cand if x[0] <= s1 + 1e-12 * T]
            t = t + s1
            Snew = None
            if len(group) == 1 and group[0][2]:
                # control continuity: add the row whose gap hit 0 / drop the row whose mu hit 0
                Snew = tuple(sorted(set(S) ^ {group[0][1]}))
                if not self.valid_set(Snew):
                    Snew = None
            if Snew is None:                     # (b) degenerate event: one-sided KKT
                ndeg += 1
                guess = tuple(sorted(set(S) ^ {x[1] for x in group}))
                Snew = self.lex_set(c, t, guess if self.valid_set(guess) else None)
                if Snew is None:
                    return None
            if Snew != S:
                seq.append(Snew)
                tau.append(t)
                S = Snew
                stall = 0
            else:
                stall += 1
                if stall > 3:
                    return None
            if len(seq) > max_arcs:
                return None
        return seq, np.array(tau), sorted(touches), ndeg

    def walk_box(self, c):
        """Umap = identity: v_j = clip(w_j, lo_j, hi_j), chains independent; events are the
        roots of w_j - hi_j and lo_j - w_j. No coupling, no enumeration."""
        T, md = self.T, self.md
        Wc = self.wcoef(c)
        times, touches = set(), []
        for j in range(md.m):
            for key, sgn, bnd in ((2 * j, 1.0, md.hi[j]), (2 * j + 1, -1.0, md.lo[j])):
                if not np.isfinite(bnd):
                    continue
                b = sgn * Wc[j].copy()
                b[0] -= sgn * bnd
                tol = 1e-12 * (_scale(b, T) + self.R)
                red, k = _strip(b, T, tol)
                if red is None:
                    continue
                cr, tc = _crossings(red, k, T, tol)
                times |= {x for x, _, _ in cr if x < T * (1 - 1e-14)}
                touches += tc
        times = sorted(times)
        edges = [0.0] + times + [T]
        seq, tau = [], []
        for i in range(len(edges) - 1):
            tm = 0.5 * (edges[i] + edges[i + 1])     # no root inside: sign at any interior point
            w = Wc @ (tm ** np.arange(4))
            S = []
            for j in range(md.m):
                if w[j] > md.hi[j]:
                    S.append(2 * j)
                elif w[j] < md.lo[j]:
                    S.append(2 * j + 1)
            S = tuple(S)
            if not seq or S != seq[-1]:
                if seq:
                    tau.append(edges[i])
                seq.append(S)
        return seq, np.array(tau), sorted(touches), 0

    # ---- dual function, residual ----
    def dual(self, arcs, c):
        p = self.p
        val = (self.Linv @ c) @ (p.xf - p.free)
        for a, b, S in arcs:
            hw = 0.5 * (b - a)
            for xg, wg in zip(_GL_X, _GL_W):
                t = a + hw * (xg + 1.0)
                w = self.md.W(t) @ c
                v = p.v_on(S, c, t)
                val += wg * hw * (0.5 * v @ v - w @ v)
        return val

    def evaluate(self, c):
        wk = self.walk(c)
        if wk is None:
            return None
        seq, tau, touches, ndeg = wk
        arcs = self.p.arcs(seq, tau)
        G, h = self.p.assemble(arcs)
        F = G @ c + h - self.p.xf
        scale = max(1.0, np.abs(self.p.xf).max(), np.abs(self.p.free).max(),
                    (np.abs(G) @ np.abs(c) + np.abs(h - self.p.free)).max())
        return dict(seq=seq, tau=tau, touches=touches, ndeg=ndeg, arcs=arcs, G=G, F=F,
                    scale=scale, D=self.dual(arcs, c))

    def verify(self, arcs, c):
        """Exact check per arc: max over the arc of every phi (critical points + ends) <= tol."""
        worst = 0.0
        for a, b, S in arcs:
            L = b - a
            for key, pc in self.row_polys(S, c).items():
                sb = _shift(pc, a)
                d = [i * sb[i] for i in range(1, len(sb))]
                xs = [0.0, L] + [x for x in _roots_le2(d) if 0.0 < x < L]
                worst = max(worst, max(_pval(sb, x) for x in xs))
        return worst


def _solve_newton(prob, maxit=50, rtol=1e-12):
    ctx = _NewtonCtx(prob)
    md = prob.md
    c = prob.solve_c(prob.arcs([()], np.zeros(0)))   # unconstrained start
    ev = ctx.evaluate(c)
    its, converged, nback = 0, False, 0
    fail = None
    if ev is None:
        fail = "event walk failed"
    c0n = 1.0 + np.abs(c).max()
    while fail is None:
        Fn = np.abs(ev["F"]).max()
        if Fn <= rtol * ev["scale"]:
            converged = True
            break
        if its >= maxit:
            fail = "max iterations"
            break
        if np.abs(c).max() > 1e6 * c0n:
            # the dual D is unbounded along the iterates: no feasible control exists
            fail = "infeasible (dual unbounded)"
            break
        its += 1
        # Newton direction in nu: (J Lm) d_nu = -F, J Lm = int K P K^T dt (sym. PSD); d_c = Lm d_nu
        H = ev["G"] @ ctx.Lm
        H = 0.5 * (H + H.T)
        rho = 0.0
        while True:
            try:
                dnu = np.linalg.solve(H + rho * np.eye(md.n), -ev["F"])
                break
            except np.linalg.LinAlgError:
                rho = max(1e-12 * np.abs(H).max(), 10 * rho)
        d = ctx.Lm @ dnu
        slope = -ev["F"] @ dnu          # directional derivative of the dual D (> 0)
        alpha, accepted = 1.0, None
        while alpha >= 1e-10:
            evn = ctx.evaluate(c + alpha * d)
            if evn is not None:
                if evn["D"] >= ev["D"] + 1e-4 * alpha * slope:
                    accepted = evn
                    break
                # below round-off of D: accept a decrease of |F| instead
                if alpha * slope <= 1e3 * _EPS * (1 + abs(ev["D"])) and \
                        np.linalg.norm(evn["F"]) < np.linalg.norm(ev["F"]):
                    accepted = evn
                    break
            alpha *= 0.5
            nback += 1
        if accepted is None:
            fail = "line search failed"
            break
        c = c + alpha * d
        ev = accepted
    if ev is None:
        seq, tau, viol, touches, ndeg, Fn = [()], np.zeros(0), np.inf, [], 0, np.inf
    else:
        seq, tau, touches, ndeg = ev["seq"], ev["tau"], ev["touches"], ev["ndeg"]
        viol = ctx.verify(ev["arcs"], c)
        Fn = float(np.abs(ev["F"]).max())
        if viol > 1e-9 * ctx.R:
            converged = False
            fail = fail or "verification failed"
    info = {
        "iterations": int(its),
        "converged": bool(converged),
        "residual_inf": Fn,
        "max_violation": float(viol),
        "backtracks": int(nback),
        "degenerate_events": int(ndeg),
        "touch_times": [float(x) for x in touches],
        "sequence": [list(S) for S in seq],
        "box_fast_path": bool(ctx.identity),
    }
    if fail:
        info["failure"] = fail
    return seq, tau, c, info


# =====================================================================================
# General LQ cost (Revision 3):  J = 1/2 int (s'Qs + v'Rv) dt + 1/2 (s(T)-xf)' Qf (s(T)-xf)
# Thm 2 primitives are exponential-polynomials (Hamiltonian matrix exponential), solved by
# semismooth Newton (primal-dual active set) with a stable Riccati-sweep linear BVP solve and
# grid-free Taylor/Bernstein event detection. See NOTES.md section 8.
# =====================================================================================
from scipy.linalg import expm as _expm, cholesky as _cholesky, solve_triangular as _solve_tri

_NT = 24                                   # Taylor order on substeps
_FACT = np.array([math.factorial(k) for k in range(_NT + 1)], float)
_COMB = np.array([[math.comb(k, j) for k in range(_NT)] for j in range(_NT)], float)  # [j,k]
_BERN = np.array([[math.comb(i, j) / math.comb(_NT - 1, j) if j <= i else 0.0
                   for j in range(_NT)] for i in range(_NT)])
_GL10_X, _GL10_W = np.polynomial.legendre.leggauss(10)
_SUB_TOL = 1e-16                           # Taylor remainder factor target per substep


def _taylor_shift_scale(C, a, w):
    """C: (r, N) ascending coefficients of p(sigma). Returns coefficients of p(a + w x)."""
    k = np.arange(_NT)
    expo = k[None, :] - k[:, None]                    # [j,k] = k - j
    Sh = np.where(expo >= 0, _COMB * np.power(a, np.clip(expo, 0, None)), 0.0)
    return (C @ Sh.T) * (w ** k)[None, :]


def _horner(c, x):
    r = 0.0
    for v in c[::-1]:
        r = r * x + v
    return r


def _first_exceed(c, a, h, tol, eps, minw):
    """First up-crossing of p (Taylor coefficients c on the substep) on (a, h] that exceeds tol.
    Bernstein enclosures (+ remainder eps) certify sub-intervals with p <= tol (no event).
    Returns (root, simple) or None."""
    # event threshold tol + eps: pruned intervals satisfy phi <= tol + 2 eps (certified),
    # an event needs the polynomial above tol + eps
    tol = tol + eps
    stack = [(a, h)]
    last_nonpos = a
    while stack:
        lo, hi = stack.pop()
        q = _taylor_shift_scale(c[None, :], lo, hi - lo)[0]
        Bc = _BERN @ q
        if Bc.max() <= tol:
            if _horner(c, hi) <= 0:
                last_nonpos = hi
            continue
        mono = bool(np.all(np.diff(Bc) >= 0))
        if mono or hi - lo <= minw:
            phi = _horner(c, hi)
            if phi > tol:
                plo = _horner(c, last_nonpos)
                if plo <= 0:
                    return _bisect(c, last_nonpos, hi), mono and last_nonpos == lo
                return last_nonpos, False
            if phi <= 0:
                last_nonpos = hi
            continue
        mid = 0.5 * (lo + hi)
        stack.append((mid, hi))
        stack.append((lo, mid))
    return None


class _GenCtx:
    """General convex LQ cost with polytopic input constraint, in R-scaled inputs vt = L^T v."""

    def __init__(self, A, B, x0, xf, T, M, lo, hi, Q, R, Qf):
        self.n, self.m = B.shape
        self.A, self.x0, self.xf, self.T = A, x0, xf, T
        self.Q = Q
        self.Qf = Qf
        self.L = _cholesky(R, lower=True)                       # R = L L^T
        self.Bt = _solve_tri(self.L, B.T, lower=True).T         # B L^{-T}
        self.Mt = _solve_tri(self.L, M.T, lower=True).T         # M L^{-T}
        self.md = _Model((1,) * self.m, tuple(range(self.m)), self.Mt, lo, hi)
        self.R_ = 1.0 + (np.abs(self.md.rhs[np.isfinite(self.md.rhs)]).max()
                         if self.md.has_bounds else 0.0)
        self.rown2 = np.sum(self.md.rows ** 2, axis=1)
        self._set = {}
        self._phi = {}
        self._rhs0 = self.md.rhs.copy()

    def set_scale(self, kap):
        """Bounds kappa*[lo, hi] (continuation); clears all rhs-dependent caches."""
        self.md.rhs = self._rhs0 * kap
        self.md._aff = {}
        self._set = {}
        self._phi = {}

    def max_ratio(self, subs):
        """max over Gauss nodes of (row . vt) / rhs for the finite rows (start scale heuristic)."""
        fin = np.isfinite(self._rhs0) & (self._rhs0 > 0)
        r = 0.0
        for sb in subs:
            for xg in _GL10_X:
                z = self.z_at(sb, 0.5 * sb["h"] * (xg + 1.0))
                d = sb["d"]
                vt = d["P"] @ (-self.Bt.T @ z[self.n:2 * self.n]) + d["q"]
                g = self.md.rows @ vt
                r = max(r, float(np.max(g[fin] / self._rhs0[fin], initial=0.0)))
        return r

    # ---- per-active-set data ----
    def setdata(self, S):
        if S in self._set:
            return self._set[S]
        n = self.n
        P, q = self.md.affine(S)                                # vt = P wt + q, wt = -Bt^T lam
        H = np.zeros((2 * n + 1, 2 * n + 1))
        H[:n, :n] = self.A
        H[:n, n:2 * n] = -self.Bt @ P @ self.Bt.T
        H[:n, 2 * n] = self.Bt @ q
        H[n:2 * n, :n] = -self.Q
        H[n:2 * n, n:2 * n] = -self.A.T
        HN = np.linalg.matrix_power(H, _NT)
        nrm = np.abs(HN).sum(axis=1).max()
        h = self.T if nrm == 0 else min(self.T, (_FACT[_NT] * _SUB_TOL / nrm) ** (1.0 / _NT))
        d = dict(P=P, q=q, H=H, nrmN=nrm, h=h)
        self._set[S] = d
        return d

    def phi_of(self, S):
        """Row functions phi = G wt + g0 for the projection set S (gap or -mu*|a|^2)."""
        if S in self._phi:
            return self._phi[S]
        md = self.md
        P, q = md.affine(S)
        keys, G, g0 = [], [], []
        for key in range(2 * self.m):
            if not np.isfinite(md.rhs[key]) or key in S:
                continue
            keys.append(key)
            G.append(md.rows[key] @ P)
            g0.append(md.rows[key] @ q - md.rhs[key])
        if S:
            Aa = md.rows[list(S)]
            X = np.linalg.solve(Aa @ Aa.T, np.column_stack([Aa, md.rhs[list(S)]]))
            for i, key in enumerate(S):
                keys.append(key)
                G.append(-X[i, :self.m] * self.rown2[key])
                g0.append(X[i, self.m] * self.rown2[key])
        res = (keys, np.array(G).reshape(len(keys), self.m), np.array(g0))
        self._phi[S] = res
        return res

    # ---- linear BVP for a fixed arc structure: Riccati sweep (stable), forward pass ----
    def sweep(self, arcs):
        n = self.n
        subs = []
        for a, b, S in arcs:
            if b <= a:
                continue
            d = self.setdata(S)
            k = max(1, int(math.ceil((b - a) / d["h"] * (1 - 1e-12))))
            hh = (b - a) / k
            Phi = _expm(d["H"] * hh)
            for i in range(k):
                subs.append(dict(t0=a + i * hh, h=hh, S=S, Phi=Phi, d=d))
        fixed = self.Qf is None
        if fixed:
            Pm, pv, Nm = np.zeros((n, n)), np.zeros(n), np.eye(n)
            Gm, Wm, rv = np.eye(n), np.zeros((n, n)), -self.xf.copy()
        else:
            Pm, pv, Nm = self.Qf.copy(), -self.Qf @ self.xf, np.zeros((n, n))
        for sb in reversed(subs):
            F = sb["Phi"]
            F11, F12, g1 = F[:n, :n], F[:n, n:2 * n], F[:n, 2 * n]
            F21, F22, g2 = F[n:2 * n, :n], F[n:2 * n, n:2 * n], F[n:2 * n, 2 * n]
            Mx = F22 - Pm @ F12
            sol = np.linalg.solve(Mx, np.column_stack([Pm @ F11 - F21, Pm @ g1 + pv - g2, Nm]))
            P0, p0, N0 = sol[:, :n], sol[:, n], sol[:, n + 1:]
            if fixed:
                Fcl = F11 + F12 @ P0
                rv = rv + Gm @ (F12 @ p0 + g1)
                Wm = Wm + Gm @ F12 @ N0
                Gm = Gm @ Fcl
            sb["P"], sb["p"], sb["N"] = P0, p0, N0
            Pm, pv, Nm = P0, p0, N0
        nu = np.linalg.solve(Wm, -(Gm @ self.x0 + rv)) if fixed else np.zeros(n)
        s = self.x0.copy()
        for sb in subs:
            lam = sb["P"] @ s + sb["N"] @ nu + sb["p"]
            z = np.concatenate([s, lam, [1.0]])
            Z = np.zeros((2 * n + 1, _NT))
            Z[:, 0] = z
            Hs = sb["d"]["H"]
            for kk in range(1, _NT):
                Z[:, kk] = Hs @ Z[:, kk - 1] / kk
            sb["Z"] = Z
            s = sb["Phi"][:n] @ z
            del sb["P"], sb["N"], sb["p"]
        zT = subs[-1]["Phi"] @ subs[-1]["Z"][:, 0]
        if fixed:
            resid = zT[:n] - self.xf
        else:
            resid = zT[n:2 * n] - self.Qf @ (zT[:n] - self.xf)
        return subs, zT, resid

    # ---- grid-free event walk on the switching functions of the current iterate ----
    def wt_coef(self, sb):
        return -self.Bt.T @ sb["Z"][self.n:2 * self.n]         # (m, N)

    def eps_sub(self, sb):
        h, d = sb["h"], sb["d"]
        c = d["nrmN"] * h ** _NT / _FACT[_NT]
        Zb = np.abs(sb["Z"]).sum(axis=0) @ (h ** np.arange(_NT))
        return c * Zb / max(1e-300, 1 - c) * np.abs(self.Bt).sum(axis=0).max()

    def lex_ok(self, S, Wc, sig, h):
        keys, G, g0 = self.phi_of(S)
        if not keys:
            return True
        C = G @ Wc
        C[:, 0] += g0
        Cs = _taylor_shift_scale(C, sig, 1.0)
        hp = h ** np.arange(_NT)
        for r in range(len(keys)):
            b = Cs[r]
            tol = 1e-12 * (np.abs(b) @ hp + self.R_)
            for i in range(_NT):
                if abs(b[i]) * hp[i] > tol:
                    if b[i] > 0:
                        return False
                    break
        return True

    def lex_set(self, Wc, sig, h, guess):
        for S in ([guess] if guess is not None else []) + self.md.sets:
            if self._valid(S) and self.lex_ok(S, Wc, sig, h):
                return S
        return None

    def _valid(self, S):
        rws = [k // 2 for k in S]
        return len(set(rws)) == len(rws) and all(np.isfinite(self.md.rhs[k]) for k in S)

    def walk(self, subs, max_events=200):
        T = self.T
        if not self.md.has_bounds:
            return [()], np.zeros(0), 0
        S = None
        seq, tau, ndeg = [], [], 0
        for sb in subs:
            h = sb["h"]
            Wc = self.wt_coef(sb)
            eps0 = self.eps_sub(sb)
            if S is None:
                S = self.lex_set(Wc, 0.0, h, ())        # (a) initial set, one-sided KKT
                if S is None:
                    return None
                seq.append(S)
            sig = 0.0
            stall = 0
            while True:
                keys, G, g0 = self.phi_of(S)
                if not keys:
                    break
                C = G @ Wc
                C[:, 0] += g0
                hp = h ** np.arange(_NT)
                best, cands = None, []
                for r, key in enumerate(keys):
                    c = C[r]
                    cs = _taylor_shift_scale(c[None, :], sig, 1.0)[0]
                    tol = 1e-12 * (np.abs(cs) @ hp + self.R_)
                    eps = eps0 * np.abs(G[r]).sum()
                    lead, ilead = 0.0, 0
                    for i in range(_NT):
                        if abs(cs[i]) * hp[i] > tol:
                            lead, ilead = cs[i], i
                            break
                    if lead > 0:
                        # sign change exactly at sig: simple iff it is a first-order zero
                        cands.append((sig, key, ilead == 1))
                        continue
                    ev = _first_exceed(c, sig, h, tol, eps, 1e-14 * T)
                    if ev is not None:
                        cands.append((ev[0], key, ev[1]))
                if not cands:
                    break
                s1 = min(x[0] for x in cands)
                group = [x for x in cands if x[0] <= s1 + 1e-12 * T]
                if sb["t0"] + s1 >= T * (1 - 1e-14):
                    break
                Snew = None
                if len(group) == 1 and group[0][2]:
                    Snew = tuple(sorted(set(S) ^ {group[0][1]}))
                    if not self._valid(Snew):
                        Snew = None
                if Snew is None:
                    ndeg += 1
                    guess = tuple(sorted(set(S) ^ {x[1] for x in group}))
                    Snew = self.lex_set(Wc, s1, h, guess if self._valid(guess) else None)
                    if Snew is None:
                        return None
                if Snew != S:
                    seq.append(Snew)
                    tau.append(sb["t0"] + s1)
                    S = Snew
                    stall = 0
                else:
                    stall += 1
                    if stall > 3:
                        return None
                sig = s1
                if len(seq) > max_events:
                    return None
        return seq, np.array(tau), ndeg

    def verify(self, subs):
        """Bernstein upper bound (+ remainder) of every row function of the iterate's own set."""
        worst = -np.inf
        for sb in subs:
            keys, G, g0 = self.phi_of(sb["S"])
            if not keys:
                continue
            C = G @ self.wt_coef(sb)
            C[:, 0] += g0
            Bc = _BERN @ _taylor_shift_scale(C, 0.0, sb["h"]).T
            eps = self.eps_sub(sb) * np.abs(G).sum(axis=1)
            worst = max(worst, float((Bc.max(axis=0) + eps).max()))
        return max(worst, 0.0) if np.isfinite(worst) else 0.0

    # ---- evaluation helpers ----
    def z_at(self, sb, sig):
        return sb["Z"] @ (sig ** np.arange(_NT))

    def v_of(self, sb, z):
        d = sb["d"]
        wt = -self.Bt.T @ z[self.n:2 * self.n]
        vt = d["P"] @ wt + d["q"]
        return _solve_tri(self.L.T, vt, lower=False)            # v = L^{-T} vt

    def cost(self, subs, zT):
        n, J = self.n, 0.0
        for sb in subs:
            h = sb["h"]
            for xg, wg in zip(_GL10_X, _GL10_W):
                z = self.z_at(sb, 0.5 * h * (xg + 1.0))
                s = z[:n]
                d = sb["d"]
                vt = d["P"] @ (-self.Bt.T @ z[n:2 * n]) + d["q"]
                J += 0.25 * h * wg * (s @ self.Q @ s + vt @ vt)
        if self.Qf is not None:
            e = zT[:n] - self.xf
            J += 0.5 * e @ self.Qf @ e
        return J


class _GenSolution:
    def _empty(self, ctx, info):
        """Failed solve (no trajectory): u, x return NaN."""
        self._c, self._subs, self._t0 = ctx, [], np.zeros(0)
        self.switch_times, self.active_sets, self.info = [], [], info
        self.cost = float("nan")
        self._u1 = lambda t: np.full(ctx.m, np.nan)
        self._x1 = lambda t: np.full(ctx.n, np.nan)
        return self

    def __init__(self, ctx, subs, arcs, zT, info):
        self._c, self._subs = ctx, subs
        self._t0 = np.array([sb["t0"] for sb in subs])
        self.switch_times = [float(a[0]) for a in arcs[1:]]
        self.active_sets = [tuple(a[2]) for a in arcs]
        self.info = info
        self.cost = float(ctx.cost(subs, zT))

    def _z(self, t):
        t = min(max(float(t), 0.0), self._c.T)
        i = max(0, int(np.searchsorted(self._t0, t, side="right")) - 1)
        sb = self._subs[i]
        return sb, self._c.z_at(sb, t - sb["t0"])

    def _u1(self, t):
        sb, z = self._z(t)
        return self._c.v_of(sb, z)

    def _x1(self, t):
        return self._z(t)[1][:self._c.n]

    def u(self, t):
        if np.ndim(t) == 0:
            return self._u1(t)
        return np.array([self._u1(ti) for ti in np.ravel(t)])

    def x(self, t):
        if np.ndim(t) == 0:
            return self._x1(t)
        return np.array([self._x1(ti) for ti in np.ravel(t)])


def _pdas(ctx, arcs, subs, zT, resid, maxit):
    """Semismooth Newton (primal-dual active set) from the trajectory `subs`.
    Each step: exact event walk on the current iterate -> new arc structure -> linear BVP."""
    T = ctx.T
    its, ndeg, seen, dt_max, prev = 0, 0, set(), np.inf, np.inf
    while True:
        wk = ctx.walk(subs)
        if wk is None:
            return False, arcs, subs, zT, resid, its, ndeg, dt_max, "event walk failed"
        seq, tau, ndeg = wk
        # drop numerical chatter: an arc shorter than 1e-10 T between two arcs of the same set
        seq, tau = list(seq), list(tau)
        i = 1
        while i + 1 < len(seq):
            if tau[i] - tau[i - 1] < 1e-10 * T and seq[i - 1] == seq[i + 1]:
                del seq[i:i + 2]
                del tau[i - 1:i + 1]
            else:
                i += 1
        new_arcs = [(e0, e1, S) for e0, e1, S in zip([0.0] + tau, tau + [T], seq)]
        if [a[2] for a in new_arcs] == [a[2] for a in arcs]:
            dt_max = max([abs(x[0] - y[0]) for x, y in zip(new_arcs, arcs)], default=0.0)
            # a junction entered with a small slope is ill-conditioned: accept 1e-9 T when the
            # exact check also passes
            # (quadratic convergence stalls at round-off: accept dt <= 1e-6 T once it stops
            # decreasing)
            if dt_max <= 1e-12 * T or (dt_max <= 1e-6 * T and dt_max >= 0.5 * prev):
                return True, arcs, subs, zT, resid, its, ndeg, dt_max, None
            prev = dt_max
        else:
            prev = np.inf
        key = (tuple(seq), tuple(np.round(tau, 13)))
        if key in seen:
            return False, arcs, subs, zT, resid, its, ndeg, dt_max, "active-set cycle"
        seen.add(key)
        if its >= maxit:
            return False, arcs, subs, zT, resid, its, ndeg, dt_max, "max iterations"
        its += 1
        arcs = new_arcs
        subs, zT, resid = ctx.sweep(arcs)          # semismooth Newton step (linear BVP)


def _solve_general(A, B, x0, xf, T, M, lo, hi, Q, R, Qf, maxit=30):
    ctx = _GenCtx(A, B, x0, xf, T, M, lo, hi, Q, R, Qf)
    arcs = [(0.0, T, ())]
    subs, zT, resid = ctx.sweep(arcs)
    its, ndeg, fail, nsteps, ok = 0, 0, None, 0, True
    dt_max = 0.0
    if ctx.md.has_bounds:
        # Globalization: continuation in the bound scale kappa (bounds kappa*[lo, hi]), from a
        # scale at which the unconstrained optimum is feasible down to kappa = 1; each step is
        # solved by PDAS warm-started from the previous solution; failed steps are halved.
        kap = max(1.0, 1.05 * ctx.max_ratio(subs))
        dl = math.log(kap)
        ctx.set_scale(kap)
        ok, arcs, subs, zT, resid, k, ndeg, dt_max, fail = _pdas(ctx, arcs, subs, zT, resid, maxit)
        its += k
        t_lim = time.perf_counter() + 60.0
        while ok and kap > 1.0:
            if time.perf_counter() > t_lim:
                ok, fail = False, "time limit"
                break
            kt = max(1.0, kap * math.exp(-dl))
            ctx.set_scale(kt)
            nsteps += 1
            # predictor: keep the accepted arc structure, re-solve the linear BVP at the new
            # bounds (arc P, q depend on the bounds), then PDAS
            s1, z1, r1 = ctx.sweep(arcs)
            ok2, a2, s2, z2, r2, k, nd2, dt2, f2 = _pdas(ctx, arcs, s1, z1, r1, maxit)
            its += k
            if ok2:
                kap, arcs, subs, zT, resid, ndeg, dt_max, fail = kt, a2, s2, z2, r2, nd2, dt2, None
                dl *= 1.5
            else:
                fail = f2
                dl *= 0.5
                ctx.set_scale(kap)
                if dl < 1e-4 or nsteps > 200:
                    ok = False
                    break
        if not ok and fail is None:
            fail = "continuation failed"
    converged = bool(ok and fail is None)
    viol = ctx.verify(subs)
    if converged and viol > 1e-9 * ctx.R_:
        converged = False
        fail = "verification failed"
    info = {
        "iterations": int(its),
        "converged": converged,
        "continuation_steps": int(nsteps),
        "terminal_residual_inf": float(np.abs(resid).max()),
        "switch_time_change": float(dt_max) if np.isfinite(dt_max) else None,
        "max_violation": float(viol),
        "degenerate_events": int(ndeg),
        "substeps": len(subs),
        "sequence": [list(a[2]) for a in arcs],
        "general_cost": True,
    }
    if fail:
        info["failure"] = fail
    return _GenSolution(ctx, subs, arcs, zT, info)


# =====================================================================================
# General LQ cost, method "newton" (Revision 6): stable-basis motion primitives.
# Per active set S the Hamiltonian system z' = H_S z + f_S (z = [s; lambda]) is split by the
# matrix sign function into stable / centre / unstable invariant subspaces (dichotomy). On each
# arc the solution is z = W_s e^{T_s (t-t_k)} a + W_c E_c(t-t_k)[c;1] + W_u e^{T_u (t-t_{k+1})} b
# + particular part, so every exponential is bounded. For fixed junction times the boundary and
# junction conditions are a block-bidiagonal linear system in (a, c, b); Newton runs on the
# junction times with exact derivatives; active-set changes come from the exact event walk and from
# exact arc-collapse events inside the Newton step. See NOTES.md section 9.
# =====================================================================================
_SIGN_MAXIT = 60
_DELTA_FACTORS = (1.0, 0.5, 2.0)          # centre half-width candidates, times 1/T


def _balance(H):
    """Diagonal similarity Hb = D^-1 H D with power-of-two scalings (Parlett-Reinsch)."""
    H = H.copy()
    n = len(H)
    d = np.ones(n)
    for _ in range(30):
        changed = False
        for i in range(n):
            c = np.abs(H[:, i]).sum() - abs(H[i, i])
            r = np.abs(H[i, :]).sum() - abs(H[i, i])
            if c == 0.0 or r == 0.0:
                continue
            f, s0 = 1.0, c + r
            while c < r / 2:
                c *= 2.0
                r /= 2.0
                f *= 2.0
            while c >= r * 2:
                c /= 2.0
                r *= 2.0
                f /= 2.0
            if f != 1.0 and c + r < 0.95 * s0:
                d[i] *= f
                H[:, i] *= f
                H[i, :] /= f
                changed = True
        if not changed:
            break
    return H, d


def _sign(M):
    """Matrix sign function by the scaled Newton iteration. Returns (S, converged)."""
    S = M.copy()
    for it in range(_SIGN_MAXIT):
        try:
            Si = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return S, False
        mu = np.sqrt(np.abs(Si).sum(0).max() / np.abs(S).sum(0).max()) if it < 20 else 1.0
        Sn = 0.5 * (mu * S + Si / mu)
        if np.abs(Sn - S).sum(0).max() <= 1e-14 * np.abs(Sn).sum(0).max():
            return Sn, True
        S = Sn
    return S, False


def _range_basis(P, r):
    if r == 0:
        return np.zeros((P.shape[0], 0))
    Qm, _, _ = __import__("scipy.linalg", fromlist=["qr"]).qr(P, pivoting=True)
    return Qm[:, :r]


class _StabCtx(_GenCtx):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._basis = {}
        self._part = {}
        self.n2 = 2 * self.n

    def set_scale(self, kap):
        super().set_scale(kap)
        self._part = {}               # W, T blocks do not depend on the bounds; q_S does

    # ---- per-active-set primitive data (cacheable offline) ----
    def basis(self, S):
        if S in self._basis:
            return self._particular(S, self._basis[S])
        n2, T = self.n2, self.T
        d = self.setdata(S)
        H = d["H"][:n2, :n2]
        f = d["H"][:n2, n2]
        Hb, dsc = _balance(H)
        I = np.eye(n2)
        nrm = np.abs(Hb).max()
        chosen = None
        for fac in _DELTA_FACTORS:
            de = fac / T
            Su, ok1 = _sign(Hb - de * I)
            Ss, ok2 = _sign(Hb + de * I)
            if not (ok1 and ok2):
                continue
            Pu, Ps = 0.5 * (I + Su), 0.5 * (I - Ss)
            tu, ts = np.trace(Pu), np.trace(Ps)
            if abs(tu - round(tu)) > 1e-6 or abs(ts - round(ts)) > 1e-6:
                continue
            res = max(np.abs((I - Pu) @ Hb @ Pu).max(), np.abs((I - Ps) @ Hb @ Ps).max())
            if res > 1e-10 * max(nrm, 1e-300) and nrm > 0:
                continue
            chosen = (de, Pu, Ps, int(round(tu)), int(round(ts)))
            break
        if chosen is None:
            raise np.linalg.LinAlgError("no dichotomy found")
        de, Pu, Ps, nu, ns = chosen
        nc = n2 - nu - ns
        Pc = I - Pu - Ps
        Wb = np.column_stack([_range_basis(Ps, ns), _range_basis(Pc, nc), _range_basis(Pu, nu)])
        W = dsc[:, None] * Wb                     # back from balanced coordinates
        Winv = np.linalg.inv(W)
        Tb = Winv @ H @ W
        s_, c_ = slice(0, ns), slice(ns, ns + nc)
        u_ = slice(ns + nc, n2)
        Ts, Tc, Tu = Tb[s_, s_], Tb[c_, c_], Tb[u_, u_]
        b = dict(W=W, Winv=Winv, ns=ns, nc=nc, nu=nu, Ts=Ts, Tc=Tc, Tu=Tu, delta=de,
                 decoupling=float(np.abs(Tb - _blockdiag(Ts, Tc, Tu)).max() /
                                  max(1e-300, np.abs(Tb).max())))
        self._basis[S] = b
        return self._particular(S, b, f)

    def _particular(self, S, b, f=None):
        """Bound-dependent part (affine term f = [Bt q_S; 0]): particular solution."""
        key = ("p", S)
        if key in self._part:
            return self._part[key]
        if f is None:
            f = self.setdata(S)["H"][:self.n2, self.n2]
        ns, nc, nu = b["ns"], b["nc"], b["nu"]
        g = b["Winv"] @ f
        bb = dict(b)
        bb["yps"] = -np.linalg.solve(b["Ts"], g[:ns]) if ns else np.zeros(0)
        bb["ypu"] = -np.linalg.solve(b["Tu"], g[ns + nc:]) if nu else np.zeros(0)
        Tca = np.zeros((nc + 1, nc + 1))
        Tca[:nc, :nc] = b["Tc"]
        Tca[:nc, nc] = g[ns:ns + nc]
        bb["Tca"] = Tca
        self._part[key] = bb
        return bb

    def arc_exps(self, b, L):
        return (_expm(b["Ts"] * L) if b["ns"] else np.zeros((0, 0)),
                _expm(b["Tca"] * L),
                _expm(-b["Tu"] * L) if b["nu"] else np.zeros((0, 0)))

    def arc_maps(self, S, L):
        """z_start = Ms x + ps, z_end = Me x + pe; dz_start/dL = Ds x, dz_end/dL = De x + de."""
        b = self.basis(S)
        ns, nc, nu, W = b["ns"], b["nc"], b["nu"], b["W"]
        Es, Ec, Eu = self.arc_exps(b, L)
        Ws, Wc, Wu = W[:, :ns], W[:, ns:ns + nc], W[:, ns + nc:]
        n2 = self.n2
        Ms = np.zeros((n2, n2))
        Ms[:, :ns] = Ws
        Ms[:, ns:ns + nc] = Wc
        Ms[:, ns + nc:] = Wu @ Eu
        ps = Ws @ b["yps"] + Wu @ b["ypu"]
        Me = np.zeros((n2, n2))
        Me[:, :ns] = Ws @ Es
        Me[:, ns:ns + nc] = Wc @ Ec[:nc, :nc]
        Me[:, ns + nc:] = Wu
        pe = Ws @ b["yps"] + Wc @ Ec[:nc, nc] + Wu @ b["ypu"]
        Ds = np.zeros((n2, n2))
        Ds[:, ns + nc:] = -Wu @ b["Tu"] @ Eu
        TE = b["Tca"] @ Ec
        De = np.zeros((n2, n2))
        De[:, :ns] = Ws @ b["Ts"] @ Es
        De[:, ns:ns + nc] = Wc @ TE[:nc, :nc]
        dpe = Wc @ TE[:nc, nc]
        return dict(Ms=Ms, ps=ps, Me=Me, pe=pe, Ds=Ds, De=De, dpe=dpe, Es=Es, Ec=Ec, Eu=Eu)

    # ---- linear BVP for fixed junction times (P1 (72)) ----
    def bvp(self, seq, tau):
        n, n2, T = self.n, self.n2, self.T
        edges = [0.0] + list(tau) + [T]
        N = len(seq)
        maps = [self.arc_maps(seq[k], edges[k + 1] - edges[k]) for k in range(N)]
        M = np.zeros((n2 * N, n2 * N))
        rhs = np.zeros(n2 * N)
        M[:n, :n2] = maps[0]["Ms"][:n]
        rhs[:n] = self.x0 - maps[0]["ps"][:n]
        r = n
        for k in range(1, N):
            M[r:r + n2, (k - 1) * n2:k * n2] = maps[k - 1]["Me"]
            M[r:r + n2, k * n2:(k + 1) * n2] = -maps[k]["Ms"]
            rhs[r:r + n2] = maps[k]["ps"] - maps[k - 1]["pe"]
            r += n2
        Me, pe = maps[-1]["Me"], maps[-1]["pe"]
        if self.Qf is None:
            Cm = np.zeros((n, n2))
            Cm[:, :n] = np.eye(n)
            dv = self.xf.copy()
        else:
            Cm = np.hstack([-self.Qf, np.eye(n)])
            dv = -self.Qf @ self.xf
        M[r:r + n, (N - 1) * n2:] = Cm @ Me
        rhs[r:r + n] = dv - Cm @ pe
        lu = sl_lu_factor(M)
        x = sl_lu_solve(lu, rhs)
        return dict(seq=list(seq), tau=list(tau), edges=edges, maps=maps, M=M, lu=lu, x=x,
                    Cm=Cm, N=N)

    def zend(self, sol, k):
        m = sol["maps"][k]
        return m["Me"] @ sol["x"][k * self.n2:(k + 1) * self.n2] + m["pe"]

    def junction_rows(self, seq):
        rows = []
        for j in range(1, len(seq)):
            L, Rr = set(seq[j - 1]), set(seq[j])
            for key in sorted(L ^ Rr):
                side = seq[j - 1] if key in Rr else seq[j]
                keys, G, g0 = self.phi_of(side)
                i = keys.index(key)
                rows.append((j, G[i], g0[i]))
        return rows

    def residual(self, sol, rows):
        r = np.zeros(len(rows))
        for q, (j, g, g0) in enumerate(rows):
            z = self.zend(sol, j - 1)
            r[q] = g @ (-self.Bt.T @ z[self.n:]) + g0
        return r

    def jacobian(self, sol, rows):
        """Exact d r / d tau via the derivative of the block system (one solve per junction)."""
        n, n2, N = self.n, self.n2, sol["N"]
        x, maps = sol["x"], sol["maps"]
        nj = N - 1
        dx = np.zeros((n2 * N, nj))
        for j in range(1, N):                    # tau_j: L_{j-1} += 1, L_j -= 1
            dF = np.zeros(n2 * N)
            for k, sgn in ((j - 1, 1.0), (j, -1.0)):
                m = maps[k]
                xk = x[k * n2:(k + 1) * n2]
                dzs = m["Ds"] @ xk
                dze = m["De"] @ xk + m["dpe"]
                # start of arc k: row block of x0 (k = 0) or junction k (minus sign)
                if k == 0:
                    dF[:n] += sgn * dzs[:n]
                else:
                    r0 = n + (k - 1) * n2
                    dF[r0:r0 + n2] -= sgn * dzs
                # end of arc k: junction k+1 or terminal rows
                if k == N - 1:
                    r0 = n + (N - 1) * n2
                    dF[r0:r0 + n] += sgn * (sol["Cm"] @ dze)
                else:
                    r0 = n + k * n2
                    dF[r0:r0 + n2] += sgn * dze
            dx[:, j - 1] = -sl_lu_solve(sol["lu"], dF)
        J = np.zeros((len(rows), nj))
        for q, (j, g, g0) in enumerate(rows):
            k = j - 1
            m = maps[k]
            xk = x[k * n2:(k + 1) * n2]
            dze = m["De"] @ xk + m["dpe"]
            dz = m["Me"] @ dx[k * n2:(k + 1) * n2]
            dz[:, j - 1] += dze                   # d L_{j-1} / d tau_j = +1
            if j - 2 >= 0:
                dz[:, j - 2] -= dze               # d L_{j-1} / d tau_{j-1} = -1
            J[q] = g @ (-self.Bt.T @ dz[n:])
        return J

    def collapse(self, seq, tau, j):
        """Exact event: arc j shrinks to zero length along a Newton step. Equal neighbours ->
        merge them; neighbours that differ in two rows -> the two switches swap order (the
        middle set becomes the other intermediate set); otherwise the arc is dropped."""
        seq, tau = list(seq), list(tau)
        N = len(seq)
        if j == 0 or j == N - 1:
            del seq[j]
            del tau[0 if j == 0 else -1]
            return seq, tau
        A, Mid, B = set(seq[j - 1]), set(seq[j]), set(seq[j + 1])
        D = A ^ B
        m = 0.5 * (tau[j - 1] + tau[j])
        if not D:
            del seq[j:j + 2]
            del tau[j - 1:j + 1]
            return seq, tau
        first = A ^ Mid
        if len(D) == 2 and len(first) == 1 and first <= D:
            Mn = tuple(sorted(A ^ (D - first)))
            if self._valid(Mn):
                seq[j] = Mn
                tau[j - 1], tau[j] = m, m + 1e-9 * self.T
                return seq, tau
        del seq[j]
        del tau[j - 1:j + 1]
        tau.insert(j - 1, m)
        return seq, tau

    def newton_tau(self, seq, tau, maxit=8, stats=None):
        """Newton on the junction times with exact derivatives. Arc collapse along a step is an
        exact event (see collapse). Fails fast when the residual does not contract (the arc
        structure is then wrong and the caller re-walks or shortens the homotopy step).
        Returns (sol, iterations, ok); sol["seq"] may differ from seq after events."""
        T = self.T
        seq = list(seq)
        tau = np.array(tau, float)
        rows = self.junction_rows(seq)
        sol = self.bvp(seq, tau)
        if not rows:
            return sol, 0, True
        r = self.residual(sol, rows)
        tol = 1e-12 * self.R_
        gap = 1e-12 * T
        prev, bad, it = np.inf, 0, 0
        while it < maxit:
            nr = np.abs(r).max()
            if nr <= tol:
                return sol, it, True
            bad += nr > 0.5 * prev
            if bad > 2:                         # stall: round-off (accept) or wrong structure
                return sol, it, bool(nr <= 1e3 * tol)
            prev = nr
            it += 1
            J = self.jacobian(sol, rows)
            step = np.linalg.lstsq(J, -r, rcond=None)[0]
            e = np.concatenate([[0.0], tau, [T]])
            de = np.concatenate([[0.0], step, [0.0]])
            Ls, dL = np.diff(e), np.diff(de)
            amax, jc = 1.0, None
            for j in range(len(Ls)):
                if dL[j] < 0 and Ls[j] + dL[j] <= gap and Ls[j] / -dL[j] < amax:
                    amax, jc = Ls[j] / -dL[j], j
            if jc is not None:                  # collapse event inside the full step
                seq, t2 = self.collapse(seq, list(tau + amax * step), jc)
                tau = np.array(t2)
                if np.any(np.diff(np.concatenate([[0.0], tau, [T]])) <= 0):
                    return sol, it, False
                rows = self.junction_rows(seq)
                sol = self.bvp(seq, tau)
                if stats is not None:
                    stats["events"] = stats.get("events", 0) + 1
                if not rows:
                    return sol, it, True
                r = self.residual(sol, rows)
                prev, bad = np.inf, 0
                continue
            alpha, acc = 1.0, None
            nr0 = np.linalg.norm(r)
            while alpha > 1e-3:
                tn = tau + alpha * step
                try:
                    sn = self.bvp(seq, tn)
                except (np.linalg.LinAlgError, ValueError):
                    sn = None
                if sn is not None:
                    rn = self.residual(sn, rows)
                    if np.linalg.norm(rn) <= (1 - 1e-4 * alpha) * nr0 or np.abs(rn).max() <= tol:
                        acc = (tn, sn, rn)
                        break
                alpha *= 0.5
            if acc is None:
                return sol, it, bool(np.abs(r).max() <= 1e3 * tol)     # round-off stall
            tau, sol, r = acc
        return sol, it, bool(np.abs(r).max() <= tol)

    # ---- substeps (Taylor models) for event walk, cost, evaluation ----
    def make_subs(self, sol):
        n2 = self.n2
        subs = []
        for k, S in enumerate(sol["seq"]):
            a0, a1 = sol["edges"][k], sol["edges"][k + 1]
            L = a1 - a0
            if L <= 0:
                continue
            d = self.setdata(S)
            b = self.basis(S)
            ns, nc, nu, W = b["ns"], b["nc"], b["nu"], b["W"]
            xk = sol["x"][k * n2:(k + 1) * n2]
            K = max(1, int(math.ceil(L / d["h"] * (1 - 1e-12))))
            h = L / K
            Es, Ec, Eu = self.arc_exps(b, h)
            ys = xk[:ns].copy()
            yc = np.concatenate([xk[ns:ns + nc], [1.0]])
            # unstable coefficients anchored at the arc end: propagate backwards
            yu_list = [None] * K
            yu = xk[ns + nc:].copy()
            for i in range(K - 1, -1, -1):
                yu = Eu @ yu
                yu_list[i] = yu
            for i in range(K):
                z = W[:, :ns] @ (ys + b["yps"]) + W[:, ns:ns + nc] @ yc[:nc] + \
                    W[:, ns + nc:] @ (yu_list[i] + b["ypu"])
                zz = np.concatenate([z, [1.0]])
                Z = np.zeros((n2 + 1, _NT))
                Z[:, 0] = zz
                for kk in range(1, _NT):
                    Z[:, kk] = d["H"] @ Z[:, kk - 1] / kk
                subs.append(dict(t0=a0 + i * h, h=h, S=S, d=d, Z=Z))
                ys = Es @ ys
                yc = Ec @ yc
        zT = np.concatenate([self.zend(sol, sol["N"] - 1), [1.0]])
        return subs, zT


def _blockdiag(*Ms):
    n = sum(m.shape[0] for m in Ms)
    out = np.zeros((n, n))
    i = 0
    for m in Ms:
        k = m.shape[0]
        out[i:i + k, i:i + k] = m
        i += k
    return out


from scipy.linalg import lu_factor as sl_lu_factor, lu_solve as sl_lu_solve  # noqa: E402


_MAX_ITS_STABLE = 1000      # total Newton iterations (all homotopy steps)
_KAP_PLAIN = 3.0            # violation ratio up to which kappa0 -> 1 is tried in one step
_DL_GROW, _DL_MIN = 1.5, 1e-3     # feasible runs use steps >= 0.03 in log(kappa)


def _propose(ctx, sol):
    """Exact event walk of the pointwise H-minimiser along the iterate -> (seq, tau), ndeg."""
    subs, _ = ctx.make_subs(sol)
    wk = ctx.walk(subs)
    if wk is None:
        return None, 0
    pseq, ptau, ndeg = wk
    pseq, ptau = list(pseq), list(ptau)
    i = 1
    while i + 1 < len(pseq):                     # drop round-off chatter arcs
        if ptau[i] - ptau[i - 1] < 1e-10 * ctx.T and pseq[i - 1] == pseq[i + 1]:
            del pseq[i:i + 2]
            del ptau[i - 1:i + 1]
        else:
            i += 1
    return (pseq, ptau), ndeg


def _correct(ctx, seq, tau, st, maxout=3, retry=1):
    """Corrector at a fixed bound scale: Newton on tau for the predicted arc structure, then
    the exact walk; repeat with the walked structure (at most maxout times). Accepts only an
    iterate whose walk reproduces its own structure (PMP consistency)."""
    for o in range(maxout):
        sn, k, ok = ctx.newton_tau(seq, tau, stats=st)
        st["its"] += k
        if not ok:
            if o > 0 or retry == 0:
                return None
            retry -= 1
        p, st["ndeg"] = _propose(ctx, sn)
        if p is None:
            return None
        if p[0] == sn["seq"]:
            dt = max([abs(x - y) for x, y in zip(p[1], sn["tau"])], default=0.0)
            if dt <= 1e-9 * ctx.T:
                st["dt_max"] = dt
                return sn
        seq, tau = p
        st["outer"] += 1
    return None


def _zmax(ctx, sol):
    return max(np.abs(ctx.zend(sol, k)).max() for k in range(sol["N"]))


def _solve_stable(A, B, x0, xf, T, M, lo, hi, Q, R, Qf, time_limit=60.0, max_its=_MAX_ITS_STABLE):
    """Stable-basis motion primitives + Newton on the junction times (NOTES.md section 9).
    Globalisation: homotopy in the bound scale kappa with exact arc-collapse events and a
    secant predictor; a single step kappa0 -> 1 when the unconstrained violation is mild."""
    t0 = time.perf_counter()
    ctx = _StabCtx(A, B, x0, xf, T, M, lo, hi, Q, R, Qf)
    st = dict(its=0, outer=0, ndeg=0, events=0, dt_max=None)
    conv, fail, nsteps, kap = False, None, 0, 1.0
    try:
        sol = ctx.bvp([()], [])
    except np.linalg.LinAlgError:
        sol, fail = None, "infeasible (singular boundary-value system)"
    if sol is not None and not ctx.md.has_bounds:
        conv = True
    elif sol is not None:
        subs, _ = ctx.make_subs(sol)
        ratio = ctx.max_ratio(subs)
        if ratio <= 1.0:
            p, st["ndeg"] = _propose(ctx, sol)
            conv = p is not None and p[0] == [()]
        if not conv:
            z0 = _zmax(ctx, sol)
            kap = 1.05 * max(ratio, 1.0)
            ctx.set_scale(kap)
            sol = ctx.bvp([()], [])
            hist = [(kap, sol)]
            dl = math.log(kap) * (1.0 if ratio <= _KAP_PLAIN else 1.0 / 3.0)
            while True:
                if kap == 1.0:
                    conv = True
                    break
                if st["its"] > max_its:
                    fail = "iteration limit"
                    break
                if time.perf_counter() - t0 > time_limit:
                    fail = "time limit"
                    break
                if dl < _DL_MIN:
                    fail = "homotopy stalled at bound scale %.6g > 1 (bounds infeasible?)" % kap
                    break
                if _zmax(ctx, sol) > 1e8 * (1.0 + z0):
                    fail = "infeasible (costate diverges)"
                    break
                kt = max(1.0, kap * math.exp(-dl))
                nsteps += 1
                ctx.set_scale(kt)
                tau = np.array(sol["tau"])
                if len(hist) >= 2 and hist[-2][1]["seq"] == sol["seq"]:       # secant predictor
                    k0, s0 = hist[-2]
                    tp = tau + (tau - np.array(s0["tau"])) / (kap - k0) * (kt - kap)
                    if np.all(np.diff(np.concatenate([[0.0], tp, [T]])) > 0):
                        tau = tp
                try:
                    sn = _correct(ctx, sol["seq"], tau, st)
                except np.linalg.LinAlgError:
                    sn = None
                if sn is not None:
                    kap, sol = kt, sn
                    hist.append((kap, sol))
                    dl *= _DL_GROW
                else:
                    ctx.set_scale(kap)
                    dl *= 0.5
    if sol is None:
        subs, zT = [], np.zeros(2 * ctx.n + 1)
        arcs = [(0.0, T, ())]
    else:
        subs, zT = ctx.make_subs(sol)
        arcs = [(sol["edges"][k], sol["edges"][k + 1], S) for k, S in enumerate(sol["seq"])]
    viol = ctx.verify(subs) if subs else np.inf
    resid = np.inf
    if sol is not None:
        z = zT[:2 * ctx.n]
        resid = float(np.abs(sol["Cm"] @ z - (ctx.xf if ctx.Qf is None else -ctx.Qf @ ctx.xf)).max())
    if conv and viol > 1e-9 * ctx.R_:
        conv, fail = False, "verification failed"
    info = {
        "iterations": int(st["its"]),
        "converged": bool(conv),
        "outer_iterations": int(st["outer"]),
        "continuation_steps": int(nsteps),
        "collapse_events": int(st["events"]),
        "bound_scale": float(kap),
        "terminal_residual_inf": resid,
        "switch_time_change": st["dt_max"],
        "max_violation": float(viol),
        "degenerate_events": int(st["ndeg"]),
        "cached_sets": len(ctx._basis),
        "sequence": [list(a[2]) for a in arcs],
        "general_cost": True,
    }
    if fail:
        info["failure"] = fail
    if sol is None:
        return _GenSolution.__new__(_GenSolution)._empty(ctx, info)
    return _GenSolution(ctx, subs, arcs, zT, info)
