"""Black-box validation of the developer's solver (Python solve() and C++ lqocp_cli).

Cases: A, B (quad_problem), and validator cases S (sliver arc), G (tangential touch),
M (many switches), I (infeasible) from validate/cases/*.txt.

Tolerances (stated here). Scales: XS = max(1, max|x_ref|), VS = max|v_ref| per case.
"""
import math
import numbers

import numpy as np
import pytest

import reference as ref
from conftest import get_solution
from harness import qp, NGRID
import harness

# --- tolerances ------------------------------------------------------------------------------
TOL_BC = 1e-8          # rel. to XS. A converged BVP solver hits ~1e-12; leaves room for CSV text.
TOL_DYN = 1e-7         # rel. to XS; our own integration of the returned u vs. returned x.
                       # (CSV: max(this, 10x the error of the same spline-of-1001-samples procedure
                       #  applied to the reference; A: 6e-15 -> 1e-7, B: 2.3e-7 -> 2.3e-6.)
TOL_COST_QUAD = 1e-9   # rel.; returned cost vs. our quadrature of the returned u.
                       # (CSV: max(this, 10x the same procedure's error on the reference.)
TOL_COST_REF = 1e-8    # rel.; vs. reference optimum. Ref errors: A 3e-14, B <= 1e-11, S/G/M ~1e-12.
TOL_VIOL = 1e-6        # abs. motor-thrust units (bound 0.3), i.e. ~3e-6 relative.
TOL_TRAJ_EXACT = 1e-7  # A, S, G, M: rel. to VS / XS against exact (polynomial / PMP-certified) refs.
REF_SAFETY = 3.0       # B: trajectory tol = 3x the QP reference's own estimated error.
TOL_POLY = 1e-8        # unbounded: rel. residual of fitting v_i by a polynomial of degree k_i-1.
TOL_PMP = 1e-6         # bounded: rel. to VS, |v(t) - proj(p(t))| with p fitted on free arcs.
TOL_SW_B = 1e-4        # C (Umap=None case): switch times vs QP reference.
TOL_SW_EXACT = 1e-6    # S, G, M: switch times vs exact reference (ref error ~1e-12 s).
TOL_PYCPP = 1e-6       # rel. to VS / XS on the CSV grid (allows CSV text rounding).
TOL_PYCPP_COST = 1e-9  # rel.
TOL_SW_AGREE = 1e-6    # s; py vs cpp switch times.
TIMEOUT_INFEAS = 120   # s; infeasible case must return within this.

IMPLS = ["py", "cpp"]
METHODS = ["newton"]
CASES = ["A", "B", "S", "G", "M", "S_old", "G_old", "M_old"]   # energy cases
# Revision 4: S/G/M/I use the corrected Umap; *_old are the earlier cases (pre-fix Umap embedded
# numerically = a synthetic polytope), kept as hard tests. B now has |u|<=0.5 and its reference is
# the exact PMP-certified solution (cases.py), cross-checked by the QP study in reference.py.
LQ_CASES = ["TINY", "L1", "L2"]            # Revision 3 general LQ cost
TOL_TRAJ_TINY_SAFETY = 3.0                 # TINY: trajectory tol = 3x the QP reference's estimated error
TOL_SW_TINY_SAFETY = 10.0                  # TINY: switch tol = max(TOL_SW_EXACT, 10x ref switch error est.)
EXACT = ("A", "S", "G", "M")


# ================================================================================================
# reference access
# ================================================================================================
_RC = {}


class Ref:
    pass


def get_ref(refs, case):
    if case in _RC:
        return _RC[case]
    r = Ref()
    r.case = case
    r.c = harness.problem(case)
    T = r.c["T"]
    r.T = T
    r.tgrid = np.linspace(0, T, 2 * (NGRID - 1) + 1)   # 2001 points; CSV grid = every 2nd
    if case == "A":
        P = refs["A_poly"]
        r.ufun, r.xfun = P.u, P.x
        r.cost = refs["A_cost"]
        r.sw, r.arcs = [], [(0.0, T, [0, 0, 0, 0])]
        r.U, r.X = P.u(r.tgrid), P.x(r.tgrid)
        r.kind = "exact"
    elif case == "B_qp":   # (legacy QP-node reference for B; B now uses the exact branch below)
        tn, Vn, Xn = refs["B_t"], refs["B_V"], refs["B_X"]
        r.tn, r.Vn, r.Xn = tn, Vn, Xn
        r.ufun = lambda tt: np.stack([np.interp(tt, tn, Vn[:, j]) for j in range(4)], 1)
        r.xfun = None
        r.cost = float(refs["B_cost"])
        r.sw = list(np.asarray(refs["B_sw"]))
        r.arcs = None
        k = (len(tn) - 1) // (len(r.tgrid) - 1)
        r.U, r.X = Vn[::k], Xn[::k]
        r.kind = "qp"
    elif case in LQ_CASES:
        d = refs["lq"][case]; nd = refs["lq_np"]
        r.tgrid = nd[f"{case}_t"]                          # 10001 points
        r.U, r.X = nd[f"{case}_V"], nd[f"{case}_X"]
        r.cost = d["cost"]
        r.sw = list(d["switch_times"])
        r.arcs = None
        r.d = d
        r.kind = "tinyqp" if case == "TINY" else "exact"
        tg = r.tgrid
        r.ufun = lambda tt: np.stack([np.interp(tt, tg, r.U[:, j]) for j in range(4)], 1)
        r.xfun = None
    else:
        import cases
        d = refs["new"][case]
        rs = cases.RefSolution(case, d)
        r.ufun, r.xfun = rs.u, rs.x
        r.cost = d["cost"]
        r.sw = list(d["switch_times"])
        r.arcs = [tuple(a) for a in d["arcs"]]
        r.U, r.X = rs.u(r.tgrid), rs.x(r.tgrid)
        r.kind = "exact"
        r.d = d
    r.XS = max(1.0, np.max(np.abs(r.X)))
    r.VS = np.max(np.abs(r.U))
    _RC[case] = r
    return r


def on_grid(r, t):
    """Reference (U, X) at a uniform grid t that is a subgrid of r.tgrid (or of B's QP nodes)."""
    base = r.tn if r.kind == "qp" else r.tgrid
    Ub, Xb = (r.Vn, r.Xn) if r.kind == "qp" else (r.U, r.X)
    k = (len(base) - 1) // (len(t) - 1)
    assert np.allclose(base[::k], t, atol=1e-12)
    return Ub[::k], Xb[::k]


def _spline(t, U):
    from scipy.interpolate import CubicSpline
    sp = CubicSpline(t, U, axis=0)
    return lambda tt: sp(np.clip(tt, t[0], t[-1]))


def _um(c):
    return np.eye(4) if c["Umap"] is None else c["Umap"]


def _sw(sol):
    return sorted(float(v) for v in (sol.switch_times or []))


# ------------------------------------------------------------------------------------------------
# switch-time matching
# ------------------------------------------------------------------------------------------------
def compare_switches(dev, r, tol):
    """Returns (status, info). status: 'match' | 'mismatch' (one-to-one in order, each within tol)."""
    sw = list(r.sw)
    err = float(np.max(np.abs(np.array(dev) - np.array(sw)))) if len(dev) == len(sw) and sw else (0.0 if dev == sw else np.inf)
    return ("match" if err <= tol else "mismatch"), dict(max_err=err)


# ================================================================================================
P_IMC = ([pytest.param(i, m, c, id=f"{i}-{m}-{c}") for i in IMPLS for m in METHODS for c in CASES]
         + [pytest.param(i, "newton", c, id=f"{i}-newton-{c}") for i in IMPLS for c in LQ_CASES]
         # option C: method="riccati" (section-8 PDAS/Riccati solver) is Python-only for a general cost
         + [pytest.param("py", "riccati", c, id=f"py-riccati-{c}") for c in LQ_CASES])


def _gauss_general(c, ufun, xfun, T, nsub, breaks=(), order=8):
    import lqref
    gx, gw = np.polynomial.legendre.leggauss(order)
    edges = np.union1d(np.linspace(0, T, nsub + 1), np.asarray([b for b in breaks if 0 < b < T], float))
    a, b = edges[:-1], edges[1:]
    tau = ((a + b)[:, None] / 2 + ((b - a) / 2)[:, None] * gx[None, :]).ravel()
    w = (((b - a) / 2)[:, None] * gw[None, :]).ravel()
    V = np.asarray(ufun(tau)); X = np.asarray(xfun(tau))
    Q, R = lqref.Q_of(c), lqref.R_of(c)
    J = 0.5 * np.sum(w * (np.einsum("ki,ij,kj->k", X, Q, X) + np.einsum("ki,ij,kj->k", V, R, V)))
    if c.get("Qf") is not None:
        xT = np.asarray(xfun(np.array([T])))[0] - c["xf"]
        J += 0.5 * xT @ np.asarray(c["Qf"]) @ xT
    return J


@pytest.mark.parametrize("impl,method,case", P_IMC)
def test_interface(impl, method, case, record):
    s = get_solution(impl, case, method)
    c = harness.problem(case)
    T = c["T"]
    probs = []
    if impl == "py":
        r = s.raw
        u1, x1 = np.asarray(r.u(0.3 * T)), np.asarray(r.x(0.3 * T))
        tk = np.array([0.0, 0.1 * T, 0.5 * T, T])
        uk, xk = np.asarray(r.u(tk)), np.asarray(r.x(tk))
        if u1.shape != (4,): probs.append(f"u(scalar).shape={u1.shape}")
        if x1.shape != (12,): probs.append(f"x(scalar).shape={x1.shape}")
        if uk.shape != (4, 4): probs.append(f"u(K=4).shape={uk.shape}")
        if xk.shape != (4, 12): probs.append(f"x(K=4).shape={xk.shape}")
        u1b = np.asarray(r.u(np.array([0.3 * T])))
        if u1b.shape != (1, 4): probs.append(f"u(array K=1).shape={u1b.shape}")
        if uk.shape == (4, 4) and not np.allclose(uk[2], np.asarray(r.u(0.5 * T)), rtol=0, atol=1e-12 * (1 + np.abs(uk).max())):
            probs.append("u(array) and u(scalar) disagree")
        if not isinstance(r.cost, numbers.Real) or not math.isfinite(float(r.cost)):
            probs.append(f"cost not a finite float: {r.cost!r}")
        st = r.switch_times
        if not isinstance(st, (list, tuple, np.ndarray)): probs.append(f"switch_times type {type(st)}")
        else:
            st = [float(v) for v in st]
            if st != sorted(st): probs.append("switch_times not sorted")
            if any(v < 0 or v > T for v in st): probs.append("switch_times outside [0,T]")
        info = r.info
        if not isinstance(info, dict): probs.append("info not a dict")
        else:
            if not isinstance(info.get("iterations"), numbers.Integral) or isinstance(info.get("iterations"), bool):
                probs.append(f"info.iterations={info.get('iterations')!r}")
            if not isinstance(info.get("solve_time_s"), numbers.Real): probs.append(f"info.solve_time_s={info.get('solve_time_s')!r}")
            if not isinstance(info.get("converged"), (bool, np.bool_)): probs.append(f"info.converged={info.get('converged')!r}")
            if info.get("method") != method: probs.append(f"info.method={info.get('method')!r}")
            record("iterations", info.get("iterations"))
            record("solve_time_s", info.get("solve_time_s"))
    else:
        if s.returncode != 0: probs.append(f"exit code {s.returncode}; stderr: {s.stderr[-200:]}")
        if len(s.json_lines) != 1: probs.append(f"{len(s.json_lines)} stdout lines (expected 1 JSON line)")
        exp_hdr = "t," + ",".join(f"x{i}" for i in range(12)) + "," + ",".join(f"u{i}" for i in range(4))
        if s.header != exp_hdr: probs.append(f"CSV header {s.header!r}")
        if s.data is None or s.data.shape != (NGRID, 17):
            probs.append(f"CSV data shape {None if s.data is None else s.data.shape} (expected ({NGRID},17))")
        else:
            t = s.t_grid
            if abs(t[0]) > 1e-12 or abs(t[-1] - T) > 1e-9: probs.append(f"t range [{t[0]}, {t[-1]}]")
            if np.max(np.abs(np.diff(t) - T / (NGRID - 1))) > 1e-9: probs.append("t not uniform")
        j = s.json or {}
        for k, typ in [("cost", numbers.Real), ("switch_times", list), ("iterations", numbers.Integral),
                       ("solve_time_us", numbers.Real), ("converged", bool), ("method", str)]:
            if k not in j: probs.append(f"JSON missing {k}")
            elif not isinstance(j[k], typ) or (typ is not bool and isinstance(j[k], bool)):
                probs.append(f"JSON {k} has type {type(j[k]).__name__}")
        if j.get("method") != method: probs.append(f"JSON method={j.get('method')!r}")
        if isinstance(j.get("switch_times"), list):
            st = j["switch_times"]
            if st != sorted(st): probs.append("switch_times not sorted")
        record("iterations", j.get("iterations"))
        record("solve_time_us", j.get("solve_time_us"))
    record("wall_s", s.wall)
    assert not probs, "; ".join(probs)


@pytest.mark.parametrize("impl", IMPLS)
def test_default_method(impl, record):
    """Revision 3: method defaults to 'newton' (Python keyword default / CLI without 3rd argument);
    the energy results of the default run must equal those of the explicit method."""
    from conftest import method_supported
    d = get_solution(impl, "B", None)
    if not method_supported(impl):
        pytest.skip("Revision-2 'method' not available yet")
    t = get_solution(impl, "B", "newton")
    m = d.info.get("method") if isinstance(d.info, dict) else None
    record("default_method", m)
    assert m == "newton", f"default method reported as {m!r} (Revision 3 requires 'newton')"
    assert abs(float(d.cost) - float(t.cost)) <= 1e-12 * abs(float(t.cost)), "default run differs from explicit newton"


@pytest.mark.parametrize("impl,method,case", P_IMC)
def test_converged(impl, method, case, record):
    s = get_solution(impl, case, method)
    conv = s.info.get("converged") if isinstance(s.info, dict) else None
    record("converged", conv)
    assert conv is True or conv is np.True_


@pytest.mark.parametrize("impl,method,case", P_IMC)
def test_boundary(impl, method, case, refs, record):
    s = get_solution(impl, case, method)
    r = get_ref(refs, case)
    c = r.c
    if impl == "py":
        x0d, xTd = s.x(0.0)[0], s.x(c["T"])[0]
    else:
        x0d, xTd = s.X_grid[0], s.X_grid[-1]
    e0 = np.max(np.abs(x0d - c["x0"])) / r.XS
    eT = np.max(np.abs(xTd - c["xf"])) / r.XS if c.get("Qf") is None else 0.0   # free endpoint: no s(T)=xf
    record("rel_err_x0", e0)
    record("rel_err_xT", eT)
    assert e0 <= TOL_BC and eT <= TOL_BC, f"x(0) err {e0:.2e}, x(T) err {eT:.2e} (rel, tol {TOL_BC:.0e})"


@pytest.mark.parametrize("impl,method,case", P_IMC)
def test_dynamics_consistency(impl, method, case, refs, record):
    """Integrate s' = Af s + Bf u with OUR integrator under the RETURNED u and compare to returned x."""
    s = get_solution(impl, case, method)
    r = get_ref(refs, case)
    T = r.T
    if impl == "py":
        t_out = np.linspace(0, T, 51)
        xd = s.x(t_out)
        xs = ref.simulate_gauss(s.u, xd[0], T, t_out, nsub=4000, breaks=list(r.sw) + _sw(s))
        err = np.max(np.abs(xs - xd)) / r.XS
        tol = TOL_DYN
    else:
        t = s.t_grid
        sel = slice(None, None, 10)
        xs = ref.simulate_gauss(_spline(t, s.U_grid), s.X_grid[0], T, t[sel], nsub=1000)
        err = np.max(np.abs(xs - s.X_grid[sel])) / r.XS
        Ur, Xr = on_grid(r, t)
        proc = np.max(np.abs(ref.simulate_gauss(_spline(t, Ur), Xr[0], T, t[sel], nsub=1000) - Xr[sel])) / r.XS
        tol = max(TOL_DYN, 10 * proc)
        record("procedure_err", proc)
    record("rel_err", err)
    record("tol", tol)
    assert err <= tol, f"max |x_sim(u) - x| = {err:.2e} rel (tol {tol:.1e})"


@pytest.mark.parametrize("impl,method,case", P_IMC)
def test_cost_quadrature(impl, method, case, refs, record):
    s = get_solution(impl, case, method)
    r = get_ref(refs, case)
    T = r.T
    general = case in LQ_CASES
    if impl == "py":
        if general:
            Jq = _gauss_general(r.c, s.u, s.x, T, nsub=2000, breaks=list(r.sw) + _sw(s))
        else:
            Jq = ref.gauss_energy(s.u, T, nsub=8000, breaks=list(r.sw) + _sw(s))
        tol = TOL_COST_QUAD
    else:
        t = s.t_grid
        Ur, Xr = on_grid(r, t)
        if general:
            Jq = _gauss_general(r.c, _spline(t, s.U_grid), _spline(t, s.X_grid), T, nsub=1000)
            proc = abs(_gauss_general(r.c, _spline(t, Ur), _spline(t, Xr), T, nsub=1000) - r.cost) / r.cost
        else:
            Jq = ref.gauss_energy(_spline(t, s.U_grid), T, nsub=1000)
            proc = abs(ref.gauss_energy(_spline(t, Ur), T, nsub=1000) - r.cost) / r.cost
        tol = max(TOL_COST_QUAD, 10 * proc)
        record("procedure_err", proc)
    err = abs(float(s.cost) - Jq) / Jq
    record("cost_reported", float(s.cost))
    record("cost_quadrature", Jq)
    record("rel_err", err)
    record("tol", tol)
    assert err <= tol, f"reported {float(s.cost):.12e} vs quadrature {Jq:.12e}: rel {err:.2e} (tol {tol:.1e})"


@pytest.mark.parametrize("impl,method,case", P_IMC)
def test_cost_vs_reference(impl, method, case, refs, record):
    s = get_solution(impl, case, method)
    r = get_ref(refs, case)
    err = (float(s.cost) - r.cost) / r.cost
    record("cost", float(s.cost))
    record("cost_ref", r.cost)
    record("rel_err", err)
    assert abs(err) <= TOL_COST_REF, f"cost {float(s.cost):.12e} vs ref {r.cost:.12e}: rel {err:.2e} (tol {TOL_COST_REF:.0e})"


@pytest.mark.parametrize("impl,method,case", P_IMC)
def test_constraints(impl, method, case, refs, record):
    s = get_solution(impl, case, method)
    r = get_ref(refs, case)
    c = r.c
    if c["u_max"] is None:
        record("max_abs_motor", np.max(np.abs(s.U_grid @ qp.Umap.T)))
        sw = _sw(s)
        assert sw == [], f"switch_times should be empty for an unbounded problem, got {sw}"
        return
    T = c["T"]
    if impl == "py":
        sw = np.array(list(r.sw) + _sw(s))
        tt = np.linspace(0, T, 20001)
        if sw.size:
            tt = np.union1d(tt, (sw[:, None] + np.linspace(-2e-3, 2e-3, 401)[None, :]).ravel().clip(0, T))
        U = s.u(tt)
    else:
        U = s.U_grid
    M = U @ _um(c).T
    viol = max(np.max(M - c["u_max"]), np.max(c["u_min"] - M), 0.0)
    record("max_violation", viol)
    record("max_abs_motor", np.max(np.abs(M)))
    assert viol <= TOL_VIOL, f"max bound violation {viol:.2e} (tol {TOL_VIOL:.0e})"


@pytest.mark.parametrize("impl,method,case", P_IMC)
def test_trajectory_vs_reference(impl, method, case, refs, record):
    s = get_solution(impl, case, method)
    r = get_ref(refs, case)
    if impl == "py":
        if r.kind in ("exact", "tinyqp"):
            t, Vr, Xr = r.tgrid, r.U, r.X
        else:
            t, Vr, Xr = r.tn, r.Vn, r.Xn
        U, X = s.u(t), s.x(t)
    else:
        t, U, X = s.t_grid, s.U_grid, s.X_grid
        Vr, Xr = on_grid(r, t)
    eu = np.max(np.abs(U - Vr))
    ex = np.max(np.abs(X - Xr))
    record("max_abs_err_u", eu)
    record("max_abs_err_x", ex)
    record("rel_err_u", eu / r.VS)
    record("rel_err_x", ex / r.XS)
    if r.kind == "exact":
        tol_u, tol_x = TOL_TRAJ_EXACT * r.VS, TOL_TRAJ_EXACT * r.XS
        assert eu <= tol_u and ex <= tol_x, f"u err {eu:.2e} (tol {tol_u:.1e}), x err {ex:.2e} (tol {tol_x:.1e})"
    elif r.kind == "tinyqp":
        tol_u = TOL_TRAJ_TINY_SAFETY * r.d["dv"]; tol_x = TOL_TRAJ_TINY_SAFETY * r.d["dx"]
        record("tol_u", tol_u); record("tol_x", tol_x)
        assert eu <= tol_u and ex <= tol_x, f"u err {eu:.2e} (tol {tol_u:.1e}), x err {ex:.2e} (tol {tol_x:.1e})"
    else:
        far = np.ones(len(t), bool)
        for t_s in r.sw:
            far &= np.abs(t - t_s) > 0.005
        eu_far = np.max(np.abs(U - Vr)[far])
        record("max_abs_err_u_far_from_switches", eu_far)
        tol_u = REF_SAFETY * float(refs["B_dv"])
        tol_far = REF_SAFETY * float(refs["B_dv_far"])
        tol_x = REF_SAFETY * float(refs["B_dx"])
        msgs = []
        if eu > tol_u: msgs.append(f"u err {eu:.2e} > {tol_u:.1e}")
        if eu_far > tol_far: msgs.append(f"u err away from switches {eu_far:.2e} > {tol_far:.1e}")
        if ex > tol_x: msgs.append(f"x err {ex:.2e} > {tol_x:.1e}")
        assert not msgs, "; ".join(msgs)


@pytest.mark.parametrize("impl,method,case", P_IMC)
def test_optimality_structure(impl, method, case, refs, record):
    """Unbounded: each v_i is a polynomial of degree <= k_i-1 (flat output degree 2k_i-1).
    Bounded: PMP: v(t) = argmin_{feasible} |v - p(t)|, p_i polynomial of degree k_i-1, with p
    fitted from the developer's own u on samples where no motor is within 1e-3 of a bound."""
    s = get_solution(impl, case, method)
    c = harness.problem(case)
    T = c["T"]
    if case in LQ_CASES:
        # General cost: lambda' = -Q s - A' lambda driven by the RETURNED s; lambda(0) fitted from
        # B' lambda = -R v on free samples; then v must equal the R-metric projection of
        # -R^-1 B' lambda everywhere, and (Qf) lambda(T) = Qf (s(T) - xf). Tolerance:
        # max(TOL_PMP, 10x the same procedure's residual on the reference at the same grid,
        # excluding +-2e-5 s around reference switches where the QP ref itself is O(h) off).
        import lqref
        r = get_ref(refs, case)
        k = 1 if impl == "py" else (len(r.tgrid) - 1) // (len(s.t_grid) - 1)
        t = r.tgrid[::k]
        U = s.u(t) if impl == "py" else s.U_grid
        X = s.x(t) if impl == "py" else s.X_grid
        e, tr, nfree = lqref.hamiltonian_check(t, X, U, c)
        pe, ptr, _ = lqref.hamiltonian_check(t, r.X[::k], r.U[::k], c, exclude=r.sw, excl_w=2e-5)
        tol = max(TOL_PMP, 10 * pe)
        tol_tr = max(TOL_PMP, 10 * ptr) if ptr is not None else None
        record("rel_err_proj", e); record("tol", tol); record("procedure_err", pe); record("n_free", nfree)
        if tr is not None:
            record("transversality", tr); record("tol_transversality", tol_tr)
        msgs = []
        if e > tol: msgs.append(f"max |v - proj_R(-R^-1 B' lambda)| = {e:.2e} rel (tol {tol:.1e})")
        if tr is not None and tr > tol_tr: msgs.append(f"lambda(T) - Qf(s(T)-xf) = {tr:.2e} rel (tol {tol_tr:.1e})")
        assert not msgs, "; ".join(msgs)
        return
    t = np.linspace(0, T, 4001) if impl == "py" else s.t_grid
    U = s.u(t) if impl == "py" else s.U_grid
    if c["u_max"] is None:
        worst = 0.0
        for j, k in enumerate(qp.CHAIN_ORDERS):
            sc = max(np.max(np.abs(U[:, j])), 1e-12)
            cf = np.polynomial.polynomial.polyfit(t, U[:, j], k - 1)
            worst = max(worst, np.max(np.abs(np.polynomial.polynomial.polyval(t, cf) - U[:, j])) / sc)
        record("worst", worst)
        assert worst <= TOL_POLY, f"v not polynomial of degree k_i-1: rel residual {worst:.2e} (tol {TOL_POLY:.0e})"
    else:
        Um = _um(c)
        M = U @ Um.T
        free = np.all((M < c["u_max"] - 1e-3) & (M > c["u_min"] + 1e-3), axis=1)
        P = ref.fit_costate_poly(t, U, free)
        pt = ref.eval_p(P, t)
        Vp = np.array([ref.project(pp, Um, c["u_min"], c["u_max"])[0] for pp in pt])
        err = np.max(np.abs(Vp - U)) / np.max(np.abs(U))
        record("rel_err_proj", err)
        record("n_free_samples", int(free.sum()))
        assert err <= TOL_PMP, f"max |v - proj(p)| = {err:.2e} rel (tol {TOL_PMP:.0e})"


@pytest.mark.parametrize("impl,method,case", P_IMC)
def test_switch_times(impl, method, case, refs, record):
    s = get_solution(impl, case, method)
    r = get_ref(refs, case)
    sw = _sw(s)
    record("n", len(sw))
    record("n_ref", len(r.sw))
    record("switch_times", [round(v, 9) for v in sw])
    if case in ("G", "G_old") and len(sw) == 2 and sw[1] - sw[0] <= TOL_SW_EXACT and abs(0.5 * (sw[0] + sw[1]) - r.d["peak_of_p"][2]) < 1e-4:
        record("note", "degenerate zero-length touch reported (accepted: peak is within 1e-9 of the bound)")
        return
    tol = TOL_SW_EXACT
    if case == "TINY":
        tol = max(TOL_SW_EXACT, TOL_SW_TINY_SAFETY * r.d["switch_err_est"])
    record("tol", tol)
    status, info = compare_switches(sw, r, tol)
    for k, v in info.items():
        record(k, v)
    if status == "match":
        return
    assert False, (f"{len(sw)} switch times {[round(v, 7) for v in sw]} vs ref ({len(r.sw)}) "
                   f"{[round(v, 7) for v in r.sw]}")


# ---- agreement checks ---------------------------------------------------------------------------
def _agree(a, b, r, record):
    t = b.t_grid if not b.continuous else (a.t_grid if not a.continuous else r.tgrid)
    ua = a.u(t) if a.continuous else a.U_grid
    ub = b.u(t) if b.continuous else b.U_grid
    xa = a.x(t) if a.continuous else a.X_grid
    xb = b.x(t) if b.continuous else b.X_grid
    eu = np.max(np.abs(ua - ub)) / r.VS
    ex = np.max(np.abs(xa - xb)) / r.XS
    ej = abs(float(a.cost) - float(b.cost)) / abs(float(a.cost))
    sa, sb = _sw(a), _sw(b)
    esw = float(np.max(np.abs(np.array(sa) - np.array(sb)))) if len(sa) == len(sb) and sa else (0.0 if sa == sb else np.inf)
    for k, v in [("rel_err_u", eu), ("rel_err_x", ex), ("rel_err_cost", ej), ("max_err_switch", esw),
                 ("n_switch", f"{len(sa)}/{len(sb)}")]:
        record(k, v)
    msgs = []
    if eu > TOL_PYCPP: msgs.append(f"u {eu:.2e}")
    if ex > TOL_PYCPP: msgs.append(f"x {ex:.2e}")
    if ej > TOL_PYCPP_COST: msgs.append(f"cost {ej:.2e}")
    sw_bad = esw > TOL_SW_AGREE
    if sw_bad: msgs.append(f"switch times {len(sa)} vs {len(sb)} (max diff {esw:.2e})")
    return msgs, sw_bad


@pytest.mark.parametrize("method,case", [pytest.param(m, c, id=f"{m}-{c}") for m in METHODS for c in CASES]
                         + [pytest.param("newton", c, id=f"newton-{c}") for c in LQ_CASES])
def test_py_vs_cpp(method, case, refs, record):
    p = get_solution("py", case, method)
    c_ = get_solution("cpp", case, method)
    msgs, _ = _agree(p, c_, get_ref(refs, case), record)
    assert not msgs, "py vs cpp disagree: " + ", ".join(msgs)


# ---- infeasible case ----------------------------------------------------------------------------
@pytest.mark.parametrize("impl,method,case", [pytest.param(i, m, c, id=f"{i}-{m}-{c}") for i in IMPLS for m in METHODS
                                               for c in ("I", "I_old")])
def test_infeasible(impl, method, case, refs, record):
    """Case I (certified infeasible): must return within TIMEOUT_INFEAS s, not crash, converged=false."""
    from conftest import method_supported
    if not (harness.py_available() if impl == "py" else harness.cpp_available()):
        pytest.skip("implementation not found")
    if not method_supported(impl):
        pytest.skip("Revision-2 'method' not available yet")
    record("certificate_value", refs["new"][case]["feasibility_value"][1])
    if impl == "py":
        d = harness.run_python_subprocess(case, method, timeout=TIMEOUT_INFEAS)
        record("wall_s", d.get("wall"))
        assert not d.get("timeout"), f"did not return within {TIMEOUT_INFEAS}s"
        assert d.get("ok"), f"solve() raised / crashed: {d.get('error')}"
        record("converged", d.get("converged"))
        record("cost", d.get("cost"))
        assert d.get("converged") is False, f"converged={d.get('converged')!r} on a certified-infeasible problem"
    else:
        import subprocess
        try:
            s = harness.run_cpp(case, method, timeout=TIMEOUT_INFEAS)
        except subprocess.TimeoutExpired:
            pytest.fail(f"lqocp_cli did not return within {TIMEOUT_INFEAS}s")
        except ValueError as e:
            pytest.fail(f"stdout is not a JSON line: {e}")
        record("wall_s", s.wall)
        record("returncode", s.returncode)
        assert s.returncode >= 0, f"killed by signal {-s.returncode}"
        assert isinstance(s.json, dict), "no JSON line printed"
        record("converged", s.json.get("converged"))
        record("cost", s.json.get("cost"))
        assert s.json.get("converged") is False, f"converged={s.json.get('converged')!r} on a certified-infeasible problem"


# ---- extra Python-only contract case: Umap=None means identity -----------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_py_umap_none_bounds(method, refs, record):
    """Case C (validator-defined): A's boundary conditions, Umap=None, |v| <= [5,50,2.5,1]."""
    from conftest import method_supported
    if not harness.py_available():
        pytest.skip("develop/python/lqocp.py not found")
    if not method_supported("py"):
        pytest.skip("Revision-2 'method' not available yet")
    C = ref.CASE_C
    s = harness.run_python("A", method, Umap=None, u_min=C["u_min"], u_max=C["u_max"])
    T = C["T"]
    tn, Vn, Xn = refs["C_t"], refs["C_V"], refs["C_X"]
    U = s.u(np.linspace(0, T, 20001))
    viol = max(np.max(U - C["u_max"]), np.max(C["u_min"] - U), 0.0)
    Jr = float(refs["C_cost"])
    ej = (float(s.cost) - Jr) / Jr
    eu = np.max(np.abs(s.u(tn) - Vn))
    ex = np.max(np.abs(s.x(tn) - Xn))
    sw = np.array(_sw(s))
    swr = np.asarray(refs["C_sw"])
    esw = np.max(np.abs(sw - swr)) if len(sw) == len(swr) else np.inf
    for k, v in [("cost", float(s.cost)), ("rel_err_cost", ej), ("max_violation", viol),
                 ("max_abs_err_u", eu), ("max_abs_err_x", ex), ("max_err_switch", esw)]:
        record(k, v)
    msgs = []
    if abs(ej) > TOL_COST_REF: msgs.append(f"cost rel err {ej:.2e}")
    if viol > TOL_VIOL: msgs.append(f"violation {viol:.2e}")
    if eu > REF_SAFETY * float(refs["C_dv"]): msgs.append(f"u err {eu:.2e} > {REF_SAFETY * float(refs['C_dv']):.1e}")
    if ex > REF_SAFETY * float(refs["C_dx"]): msgs.append(f"x err {ex:.2e} > {REF_SAFETY * float(refs['C_dx']):.1e}")
    tol_sw = max(TOL_SW_B, 10.0 * float(refs["C_sw_err"]))
    if not esw <= tol_sw: msgs.append(f"switch times {np.round(sw, 6)} vs ref {np.round(swr, 6)}")
    assert not msgs, "; ".join(msgs)


# ---- option C: riccati vs newton (Python, general cost) and C++ riccati exit code ----------------
@pytest.mark.parametrize("case", LQ_CASES)
def test_py_riccati_vs_newton(case, refs, record):
    a = get_solution("py", case, "riccati")
    b = get_solution("py", case, "newton")
    msgs, _ = _agree(a, b, get_ref(refs, case), record)
    assert not msgs, "py riccati vs newton disagree: " + ", ".join(msgs)


@pytest.mark.parametrize("case", LQ_CASES)
def test_cpp_riccati_general_exit(case, record):
    """C++ 'riccati' with a general cost is documented as not ported: must exit 3 cleanly, no crash."""
    if not harness.cpp_available():
        pytest.skip("implementation not found")
    s = harness.run_cpp(case, "riccati")
    record("returncode", s.returncode)
    assert s.returncode == 3, f"rc={s.returncode} (expected documented exit 3)"
