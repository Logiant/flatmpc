"""Revision 3: independent references for the general LQ cost

    J = 1/2 int_0^T (s'Qs + v'Rv) dt + 1/2 (s(T)-xf)' Qf (s(T)-xf),   s' = Af s + Bf v, s(0)=x0
    (s(T)=xf instead of the terminal cost when Qf is None; u_min <= Umap v <= u_max if given).

Methods (none uses the developer's junction/Newton machinery):
  * hamiltonian_mp: unconstrained, fixed endpoint or Qf. Two-point BVP solved with the matrix
    exponential of the Hamiltonian H = [[A, -B R^-1 B'], [-Q, -A']] in mpmath at several hundred
    digits (H has eigenvalues up to |586|, so e^{HT} spans ~10^255 and double precision is useless).
    The trajectory is propagated on the grid with e^{Hh} in the same precision.
  * riccati: unconstrained with Qf. Backward Riccati ODE (+ affine term q and constant r for xf != 0)
    integrated with DOP853 at rtol 1e-13; J = V(x0, 0). Independent of hamiltonian_mp.
  * qp_general: constrained. Direct transcription, first-order-hold input, exact discretisation,
    EXACT integration of the running cost for that input (Gauss on the polynomial-in-time state),
    node bounds (=> feasible for all t). Decision variable is the motor thrust u (R_u = W'RW, W =
    Umap^-1, which is 4I for TINY), which removes R_flat's cond ~2e6 from the problem; a flat-v
    variant is kept to quantify the effect.
R^-1 is always applied as W R_u^-1 W' when Umap is given (well conditioned).
"""
import json
import os

import numpy as np
from scipy.linalg import expm

import reference as ref
from reference import qp

HERE = os.path.dirname(os.path.abspath(__file__))
N_X, N_V = 12, 4


def _W(c):
    return np.eye(4) if c.get("Umap") is None else np.linalg.inv(np.asarray(c["Umap"], float))


def R_of(c):
    return np.eye(4) if c.get("R") is None else np.asarray(c["R"], float)


def Q_of(c):
    return np.zeros((12, 12)) if c.get("Q") is None else np.asarray(c["Q"], float)


def Rinv(c):
    """R^-1 computed as W (W'RW)^-1 W' (W'RW is well conditioned for the physical cases)."""
    W = _W(c)
    Ru = W.T @ R_of(c) @ W
    return W @ np.linalg.solve(Ru, W.T)


def S_of(c):
    return qp.Bf @ Rinv(c) @ qp.Bf.T


# ================================================================================================
# 1. Hamiltonian matrix exponential in multiprecision
# ================================================================================================
def hamiltonian_mp(c, ngrid=2001, dps=None):
    import mpmath as mp
    A, Q, S = qp.Af, Q_of(c), S_of(c)
    T = c["T"]
    H = np.block([[A, -S], [-Q, -A.T]])
    mu = np.max(np.abs(np.linalg.eigvals(H).real))
    if dps is None:
        dps = int(60 + 2 * mu * T / np.log(10))
    mp.mp.dps = dps
    Hm = mp.matrix(H.tolist())
    Tm = mp.mpf(T)
    E = mp.expm(Hm * Tm)
    P11, P12 = E[0:12, 0:12], E[0:12, 12:24]
    P21, P22 = E[12:24, 0:12], E[12:24, 12:24]
    s0 = mp.matrix([mp.mpf(float(v)) for v in c["x0"]])
    xf = mp.matrix([mp.mpf(float(v)) for v in c["xf"]])
    if c.get("Qf") is None:
        lam0 = mp.lu_solve(P12, xf - P11 * s0)
    else:
        Qf = mp.matrix(np.asarray(c["Qf"], float).tolist())
        lam0 = mp.lu_solve(P22 - Qf * P12, Qf * P11 * s0 - P21 * s0 - Qf * xf)
    sT = P11 * s0 + P12 * lam0
    lT = P21 * s0 + P22 * lam0
    J = (lam0.T * s0)[0] - (lT.T * sT)[0]
    if c.get("Qf") is not None:
        d = sT - xf
        J = J + (d.T * Qf * d)[0]
    J = J / 2
    # propagate on the grid
    h = Tm / (ngrid - 1)
    Eh = mp.expm(Hm * h)
    z = mp.matrix(24, 1)
    for i in range(12):
        z[i] = s0[i]; z[12 + i] = lam0[i]
    Z = np.zeros((ngrid, 24))
    for k in range(ngrid):
        Z[k] = [float(z[i]) for i in range(24)]
        if k < ngrid - 1:
            z = Eh * z
    t = np.linspace(0, T, ngrid)
    S_ = Z[:, :12]; L = Z[:, 12:]
    V = -(L @ qp.Bf) @ Rinv(c).T
    return dict(t=t, X=S_, Lam=L, V=V, cost=float(J), lam0=np.array([float(v) for v in lam0]),
                sT=np.array([float(v) for v in sT]), dps=dps, mu=mu,
                bc_err=float(mp.norm(sT - xf, mp.inf)) if c.get("Qf") is None else None)


# ================================================================================================
# 2. Riccati ODE (Qf case)
# ================================================================================================
def riccati(c, rtol=1e-13, ngrid=2001):
    from scipy.integrate import solve_ivp
    A, Q, S = qp.Af, Q_of(c), S_of(c)
    Qf = np.asarray(c["Qf"], float)
    xf = np.asarray(c["xf"], float)
    T = c["T"]
    n = 12

    def rhs(t, y):
        P = y[:n * n].reshape(n, n); q = y[n * n:n * n + n]
        dP = -(A.T @ P + P @ A - P @ S @ P + Q)
        dq = -((A - S @ P).T @ q)
        dr = 0.5 * q @ S @ q
        return np.concatenate([dP.ravel(), dq, [dr]])
    yT = np.concatenate([Qf.ravel(), -Qf @ xf, [0.5 * xf @ Qf @ xf]])
    sc = max(1.0, np.max(np.abs(yT)))
    sol = solve_ivp(rhs, (T, 0.0), yT, method="DOP853", rtol=rtol, atol=rtol * 1e-3 * sc, dense_output=True)
    y0 = sol.y[:, -1]
    P0 = y0[:n * n].reshape(n, n); q0 = y0[n * n:n * n + n]; r0 = y0[-1]
    x0 = np.asarray(c["x0"], float)
    J = 0.5 * x0 @ P0 @ x0 + q0 @ x0 + r0
    lam0 = P0 @ x0 + q0
    # forward closed loop for the trajectory
    def fw(t, s):
        y = sol.sol(t)
        P = y[:n * n].reshape(n, n); q = y[n * n:n * n + n]
        return A @ s - S @ (P @ s + q)
    t = np.linspace(0, T, ngrid)
    fs = solve_ivp(fw, (0, T), x0, method="DOP853", rtol=rtol, atol=1e-15, t_eval=t)
    X = fs.y.T
    Lam = np.array([sol.sol(tt)[:n * n].reshape(n, n) @ X[i] + sol.sol(tt)[n * n:n * n + n] for i, tt in enumerate(t)])
    V = -(Lam @ qp.Bf) @ Rinv(c).T
    return dict(t=t, X=X, V=V, Lam=Lam, cost=float(J), lam0=lam0, nfev=sol.nfev)


# ================================================================================================
# 3. Direct QP for the general cost
# ================================================================================================
def _foh_path(tau, h, Bv):
    """s(tau) = Phi s_k + G0 a + G1 b inside an interval of length h (input linear from a to b)."""
    n, m = 12, Bv.shape[1]
    M = np.zeros((n + 2 * m, n + 2 * m))
    M[:n, :n] = qp.Af; M[:n, n:n + m] = Bv; M[n:n + m, n + m:] = np.eye(m)
    E = expm(M * tau)
    Phi, E1, E2 = E[:n, :n], E[:n, n:n + m], E[:n, n + m:]
    return Phi, E1 - E2 / h, E2 / h


def _interval_mats(h, Bv, Rv, Q, m=4):
    Phi, G0, G1 = _foh_path(h, h, Bv)
    # exact interval cost matrix for xi = [s_k, a, b] (state is polynomial in tau of degree <= 5)
    gx, gw = np.polynomial.legendre.leggauss(8)
    Mc = np.zeros((12 + 2 * m, 12 + 2 * m))
    for xq, wq in zip(gx, gw):
        tau = h * (xq + 1) / 2
        P_, A0, A1 = _foh_path(tau, h, Bv)
        C = np.hstack([P_, A0, A1])
        a_ = 1 - tau / h
        D = np.hstack([np.zeros((m, 12)), a_ * np.eye(m), (1 - a_) * np.eye(m)])
        Mc += (h / 2) * wq * (C.T @ Q @ C + D.T @ Rv @ D)
    Mc = 0.5 * (Mc + Mc.T)
    ev, EV = np.linalg.eigh(Mc)
    keep = ev > 1e-14 * ev.max()
    L = EV[:, keep] * np.sqrt(ev[keep])
    return Phi, G0, G1, Mc, L


def qp_general(c, N, var="u", solver="CLARABEL", verbose=False, regions=None):
    """var='u': motor-thrust decision variable (needs Umap); var='v': flat input with R_flat.
    regions: list of (t_end, n_intervals) for a piecewise-uniform grid (default: uniform N)."""
    import cvxpy as cp
    T = c["T"]
    if regions is None:
        regions = [(T, N)]
    W = _W(c)
    R = R_of(c); Q = Q_of(c)
    if var == "u":
        Bv = qp.Bf @ W; Rv = W.T @ R @ W
        Gb = np.eye(4)
    else:
        Bv = qp.Bf; Rv = R
        Gb = np.eye(4) if c.get("Umap") is None else np.asarray(c["Umap"], float)
    lo, hi = c.get("u_min"), c.get("u_max")
    m = 4
    t = [0.0]; segs = []; k0 = 0; t0 = 0.0
    for (te, n) in regions:
        h = (te - t0) / n
        segs.append((k0, k0 + n, h, _interval_mats(h, Bv, Rv, Q)))
        t += list(t0 + h * np.arange(1, n + 1))
        k0 += n; t0 = te
    t = np.array(t); t[-1] = T
    Ntot = len(t) - 1
    X = cp.Variable((Ntot + 1, 12)); U = cp.Variable((Ntot + 1, m))
    cons = [X[0] == c["x0"]]
    cost = 0
    for (ka, kb, h, (Phi, G0, G1, Mc, L)) in segs:
        cons.append(X[ka + 1:kb + 1].T == Phi @ X[ka:kb].T + G0 @ U[ka:kb].T + G1 @ U[ka + 1:kb + 1].T)
        Xi = cp.hstack([X[ka:kb], U[ka:kb], U[ka + 1:kb + 1]])
        cost = cost + 0.5 * cp.sum_squares(Xi @ L)
    if c.get("Qf") is None:
        cons.append(X[Ntot] == c["xf"])
    else:
        evf, EVf = np.linalg.eigh(np.asarray(c["Qf"], float))
        Lf = EVf * np.sqrt(np.clip(evf, 0, None))
        cost = cost + 0.5 * cp.sum_squares((X[Ntot] - c["xf"]) @ Lf)
    nb = len(cons)
    if hi is not None:
        Gu = U @ Gb.T
        cons += [Gu <= np.tile(hi, (Ntot + 1, 1)), Gu >= np.tile(lo, (Ntot + 1, 1))]
    prob = cp.Problem(cp.Minimize(cost), cons)
    prob.solve(solver=solver, verbose=verbose, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12, max_iter=500)
    Uv = U.value
    Vv = Uv @ W.T if var == "u" else Uv
    # re-simulate exactly and evaluate the cost exactly for the returned input
    Xs = np.zeros((Ntot + 1, 12)); Xs[0] = c["x0"]
    J = 0.0
    for (ka, kb, h, (Phi, G0, G1, Mc, L)) in segs:
        for k in range(ka, kb):
            Xs[k + 1] = Phi @ Xs[k] + G0 @ Uv[k] + G1 @ Uv[k + 1]
        Xi_v = np.hstack([Xs[ka:kb], Uv[ka:kb], Uv[ka + 1:kb + 1]])
        J += 0.5 * np.einsum("ki,ij,kj->", Xi_v, Mc, Xi_v)
    if c.get("Qf") is not None:
        d = Xs[Ntot] - c["xf"]; J += 0.5 * d @ np.asarray(c["Qf"]) @ d
    mu = None
    if hi is not None:
        # node multipliers per unit time (dual / local node weight), signed + upper, - lower
        wnode = np.zeros(Ntot + 1)
        dt = np.diff(t); wnode[:-1] += dt / 2; wnode[1:] += dt / 2
        mu = (cons[nb].dual_value - cons[nb + 1].dual_value) / wnode[:, None]
    Uphys = Uv if var == "u" else Uv @ Gb.T
    return dict(t=t, V=Vv, U=Uphys, X=Xs, cost=float(J), status=prob.status, mu=mu, N=Ntot,
                bc_err=float(np.max(np.abs(Xs[Ntot] - c["xf"]))))


# ================================================================================================
# R-metric projection and the general optimality check (used by tests)
# ================================================================================================
def proj_R(p, c, tol=1e-10):
    """argmin_v (v-p)'R(v-p) s.t. u_min <= Umap v <= u_max (exact, active-set enumeration in u)."""
    if c.get("u_max") is None:
        return p.copy(), np.zeros(4, int)
    Um = np.eye(4) if c.get("Umap") is None else np.asarray(c["Umap"], float)
    W = np.linalg.inv(Um)
    M = W.T @ R_of(c) @ W                      # metric in u
    uc = Um @ p
    lo, hi = np.asarray(c["u_min"], float), np.asarray(c["u_max"], float)
    sc = max(1.0, np.max(np.abs(uc)))
    best = (np.inf, None, None)
    for s in ref._patterns():
        A_ = np.nonzero(s)[0]; F = np.nonzero(s == 0)[0]
        u = uc.copy()
        u[A_] = np.where(s[A_] > 0, hi[A_], lo[A_])
        if F.size and A_.size:
            u[F] = uc[F] - np.linalg.solve(M[np.ix_(F, F)], M[np.ix_(F, A_)] @ (u[A_] - uc[A_]))
        g = M @ (u - uc)                       # gradient of the objective
        # KKT: g_A = -(multiplier) ; upper active needs g <= 0, lower active needs g >= 0
        vmu = max(0.0, np.max(g[A_] * s[A_])) if A_.size else 0.0
        vu = max(0.0, np.max(u - hi), np.max(lo - u))
        score = max(vmu / (sc * np.max(np.abs(M))), vu / max(np.max(np.abs(hi)), 1e-300))
        if score <= tol:
            return W @ u, s
        if score < best[0]:
            best = (score, W @ u, s)
    return best[1], best[2]


def _expm_nil(M, x):
    """e^{M x} for nilpotent M (M^4 = 0 for the flat model): exact finite series."""
    E = np.eye(M.shape[0]); term = np.eye(M.shape[0])
    for j in range(1, 5):
        term = term @ M * (x / j)
        E = E + term
    return E


def costate_from_trajectory(t, X, V, c, free):
    """Fit lambda(0) from B'lambda(t) = -R v(t) on free samples, with lambda' = -Q s - A'lambda driven
    by the GIVEN state trajectory (cumulative trapezoid-free: exact integration on a fine grid is the
    caller's job; here X is given on t). Returns Lam (len(t), 12)."""
    A, Q, B = qp.Af, Q_of(c), qp.Bf
    R = R_of(c)
    # g(t) = int_0^t e^{-A'(t-tau)} Q s(tau) dtau, via composite Simpson on the given grid
    # (A nilpotent: e^{-A' x} is an exact polynomial)
    n = len(t)
    g = np.zeros((n, 12))
    # recursive: g(t_{k+1}) = e^{-A'h} g(t_k) + int_{t_k}^{t_{k+1}} e^{-A'(t_{k+1}-tau)} Q s(tau) dtau
    from scipy.interpolate import CubicSpline
    sp = CubicSpline(t, X, axis=0)
    gx, gw = np.polynomial.legendre.leggauss(6)
    for k in range(n - 1):
        h = t[k + 1] - t[k]
        Eh = _expm_nil(-A.T, h)
        acc = Eh @ g[k]
        for xq, wq in zip(gx, gw):
            tau = t[k] + h * (xq + 1) / 2
            acc += (h / 2) * wq * _expm_nil(-A.T, t[k + 1] - tau) @ (Q @ sp(tau))
        g[k + 1] = acc
    Phi = np.array([_expm_nil(-A.T, tt) for tt in t])          # lambda(t) = Phi(t) lam0 - g(t)
    rows = []; rhs = []
    for k in np.nonzero(free)[0]:
        rows.append(B.T @ Phi[k]); rhs.append(-R @ V[k] + B.T @ g[k])
    Am, bm = np.vstack(rows), np.concatenate(rhs)
    cs = np.linalg.norm(Am, axis=0); cs[cs == 0] = 1
    lam0 = np.linalg.lstsq(Am / cs, bm, rcond=None)[0] / cs
    Lam = np.einsum("kij,j->ki", Phi, lam0) - g
    return Lam, lam0


# ================================================================================================
# Exact active-set polish of the discretised QP (primal-dual active-set iterations on the KKT
# system, sparse LU). Removes the interior-point smearing near tangential touches; the result is
# the exact optimum of the transcribed QP (to rounding), with signed node multipliers.
# ================================================================================================
def polish_active_set(c, regions, var="u", init=None, maxit=60, verbose=False):
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    T = c["T"]
    W = _W(c); R = R_of(c); Q = Q_of(c)
    assert var == "u"
    Bv = qp.Bf @ W; Rv = W.T @ R @ W
    lo = None if c.get("u_min") is None else np.asarray(c["u_min"], float)
    hi = None if c.get("u_max") is None else np.asarray(c["u_max"], float)
    t = [0.0]; segs = []; k0 = 0; t0 = 0.0
    for (te, n) in regions:
        h = (te - t0) / n
        segs.append((k0, k0 + n, h, _interval_mats(h, Bv, Rv, Q)))
        t += list(t0 + h * np.arange(1, n + 1)); k0 += n; t0 = te
    t = np.array(t); t[-1] = T
    N = len(t) - 1
    nX = 12 * (N + 1); nz = nX + 4 * (N + 1)
    xi = lambda k: 12 * k + np.arange(12)
    ui = lambda k: nX + 4 * k + np.arange(4)
    rows, cols, vals = [], [], []
    for (ka, kb, h, (Phi, G0, G1, Mc, L)) in segs:
        ks = np.arange(ka, kb)
        idx = np.hstack([12 * ks[:, None] + np.arange(12), nX + 4 * ks[:, None] + np.arange(4),
                         nX + 4 * (ks[:, None] + 1) + np.arange(4)])           # (n,20)
        rows.append(np.repeat(idx, 20, axis=1).ravel()); cols.append(np.tile(idx, (1, 20)).ravel())
        vals.append(np.tile(Mc.ravel(), len(ks)))
    f = np.zeros(nz)
    if c.get("Qf") is not None:
        Qf = np.asarray(c["Qf"], float)
        idx = xi(N)
        rows.append(np.repeat(idx, 12)); cols.append(np.tile(idx, 12)); vals.append(Qf.ravel())
        f[idx] -= Qf @ np.asarray(c["xf"], float)
    H = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nz, nz))
    # fixed equalities
    er, ec, ev, eb = [], [], [], []
    r0 = 0
    def add_row(cidx, cval, b):
        nonlocal r0
        er.extend([r0] * len(cidx)); ec.extend(list(cidx)); ev.extend(list(cval)); eb.append(b); r0 += 1
    for i in range(12):
        add_row([xi(0)[i]], [1.0], c["x0"][i])
    for (ka, kb, h, (Phi, G0, G1, Mc, L)) in segs:
        for k in range(ka, kb):
            for i in range(12):
                cidx = [xi(k + 1)[i]] + list(xi(k)) + list(ui(k)) + list(ui(k + 1))
                cval = [1.0] + list(-Phi[i]) + list(-G0[i]) + list(-G1[i])
                add_row(cidx, cval, 0.0)
    if c.get("Qf") is None:
        for i in range(12):
            add_row([xi(N)[i]], [1.0], c["xf"][i])
    nfix = r0
    Efix = sp.csr_matrix((ev, (er, ec)), shape=(nfix, nz)); bfix = np.array(eb)
    # active set: dict (k,i) -> +1/-1
    act = {} if init is None else dict(init)
    for it in range(maxit):
        keys = sorted(act)
        if keys:
            ra = np.arange(len(keys)); ca = np.array([ui(k)[i] for k, i in keys])
            Ea = sp.csr_matrix((np.ones(len(keys)), (ra, ca)), shape=(len(keys), nz))
            ba = np.array([hi[i] if act[(k, i)] > 0 else lo[i] for k, i in keys])
            E = sp.vstack([Efix, Ea]).tocsc(); b = np.concatenate([bfix, ba])
        else:
            E = Efix.tocsc(); b = bfix
        K = sp.bmat([[H, E.T], [E, None]]).tocsc()
        sol = splu(K).solve(np.concatenate([-f, b]))
        z = sol[:nz]; y = sol[nz:]
        U = z[nX:].reshape(N + 1, 4)
        ya = y[nfix:]
        changed = False
        # wrong-sign multipliers -> release
        bad = [keys[j] for j in range(len(keys)) if ya[j] * act[keys[j]] < -1e-12 * max(1.0, np.max(np.abs(ya)))]
        for kk in bad:
            del act[kk]; changed = True
        # violations -> add
        if hi is not None:
            viol_hi = np.argwhere(U > hi + 1e-13)
            viol_lo = np.argwhere(U < lo - 1e-13)
            for k, i in viol_hi:
                if (k, i) not in act: act[(k, i)] = 1; changed = True
            for k, i in viol_lo:
                if (k, i) not in act: act[(k, i)] = -1; changed = True
        if verbose:
            print(f"   active-set it {it}: |A|={len(keys)}, released {len(bad)}, violations "
                  f"{0 if hi is None else len(viol_hi) + len(viol_lo)}")
        if not changed:
            break
    X = z[:nX].reshape(N + 1, 12)
    mu = np.zeros((N + 1, 4))
    wnode = np.zeros(N + 1); dt = np.diff(t); wnode[:-1] += dt / 2; wnode[1:] += dt / 2
    for j, (k, i) in enumerate(keys):
        mu[k, i] = ya[j] / wnode[k]
    J = 0.5 * z @ (H @ z) + f @ z
    if c.get("Qf") is not None:
        J += 0.5 * c["xf"] @ np.asarray(c["Qf"]) @ c["xf"]
    W_ = W
    return dict(t=t, U=U, V=U @ W_.T, X=X, mu=mu, cost=float(J), iterations=it, converged=not changed,
                active=act, N=N)


def switch_times_polished(t, U, mu, lo, hi):
    """Exact-QP switch times: per motor, entries/exits of the node active set; the time is the
    zero of the multiplier (linear extrapolation of the 2 active-side nodes) averaged with the
    crossing of the bound by the free-side u (2 free-side nodes). Returns list of
    (time, motor, kind, t_mu, t_u)."""
    out = []
    for i in range(4):
        a = mu[:, i] != 0
        for k in np.nonzero(a[:-1] != a[1:])[0]:
            if a[k]:     # exit after node k
                ja, jb, fa, fb = k, k - 1, k + 1, k + 2
            else:        # entry at node k+1
                ja, jb, fa, fb = k + 1, k + 2, k, k - 1
            sgn = np.sign(mu[ja, i])
            bnd = hi[i] if sgn > 0 else lo[i]
            tm = t[ja] - mu[ja, i] * (t[jb] - t[ja]) / (mu[jb, i] - mu[ja, i]) if 0 <= jb < len(t) and mu[jb, i] != 0 else np.nan
            tu = t[fa] + (bnd - U[fa, i]) * (t[fb] - t[fa]) / (U[fb, i] - U[fa, i]) if 0 <= fb < len(t) else np.nan
            out.append((float(np.nanmean([tm, tu])), i, "exit" if a[k] else "entry", float(tm), float(tu)))
    return sorted(out)


def hamiltonian_check(t, X, V, c, margin=1e-3, exclude=(), excl_w=0.0):
    """Optimality from outside: reconstruct lambda from the given (s, v) on free samples, then
    return (max|v - proj_R(-R^-1 B' lambda)| / max|v|, transversality residual or None)."""
    if c.get("u_max") is None:
        free = np.ones(len(t), bool)
    else:
        Um = np.eye(4) if c.get("Umap") is None else np.asarray(c["Umap"], float)
        M = V @ Um.T
        free = np.all((M < np.asarray(c["u_max"]) - margin) & (M > np.asarray(c["u_min"]) + margin), axis=1)
    Lam, lam0 = costate_from_trajectory(t, X, V, c, free)
    P = -(Lam @ qp.Bf) @ Rinv(c).T
    Vp = np.array([proj_R(p, c)[0] for p in P])
    keep = np.ones(len(t), bool)
    for s in exclude:
        keep &= np.abs(t - s) > excl_w
    e = float(np.max(np.abs(Vp - V)[keep]) / np.max(np.abs(V)))
    tr = None
    if c.get("Qf") is not None:
        lt = np.asarray(c["Qf"]) @ (X[-1] - np.asarray(c["xf"]))
        tr = float(np.max(np.abs(Lam[-1] - lt)) / max(np.max(np.abs(Lam)), 1e-300))
    return e, tr, int(free.sum())
