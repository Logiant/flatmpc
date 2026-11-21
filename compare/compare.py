#!/usr/bin/env python3
"""Fair host-only benchmark: TinyMPC (ADMM, discrete) vs our indirect-method solver (continuous)
on quad_problem.TINY_CASE (spec Revision 5). Every solution is scored with the SAME continuous-time cost:

    J = 1/2 int_0^T (x'Q x + u'R u) dt + 1/2 x(T)' Qf x(T),
    Q = Q_PHYS, R = R_PHYS, Qf = QF_TINY_PHYS (TinyMPC's own terminal weight), T = 0.45 s, |u| <= 0.5.

Discrete solutions are applied as zero-order hold to the continuous model and J is computed exactly
(Van Loan augmented exponential). Our solver is called only through its public interface, in a child
process with a timeout, so a crash or hang in develop/ only marks the "ours" rows pending.

Usage:  compare.py [--model phys|spec]
  phys (default, -> results/): the Crazyflie model quad_problem.Ac, Bc; Qf = QF_TINY_PHYS.
  spec (-> results/spec/): the model implied by TINY_CASE (flat model + Umap in physical coordinates),
        Qf = Tp2f' TINY_CASE["Qf"] Tp2f. Must agree with phys to ~1e-6 (Revision 4).
Writes summary.json, summary.txt and traj_<row>.csv. See compare/README.md.
"""
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np
from scipy.linalg import expm, solve_discrete_are, block_diag
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(ROOT, "problem"), os.path.join(ROOT, "develop", "python")]
import quad_problem as qp  # noqa: E402

NX, NU = qp.NX, qp.NU
Q, R = qp.Q_PHYS, qp.R_PHYS
X0 = qp.X0.astype(float)
T = float(qp.TINY_CASE["T"])                    # 0.45 = (N_knots - 1) * dt
UMAX = 0.5
UMIN_V, UMAX_V = -UMAX * np.ones(NU), UMAX * np.ones(NU)

UMAP_RESIDUAL = float(np.max(np.abs(qp.Tp2f @ qp.Bc @ qp.Umap - qp.Bf)))
STATE_MAP_RESIDUAL = float(np.max(np.abs(qp.Tp2f @ qp.Ac @ qp.Tf2p - qp.Af)))
MAPS_CONSISTENT = UMAP_RESIDUAL < 1e-4 and STATE_MAP_RESIDUAL < 1e-4

# Scoring model (set in main()).
A_C, B_C, QF = qp.Ac, qp.Bc, qp.QF_TINY_PHYS
MODEL = "phys"
RES = os.path.join(HERE, "results")

RHO = 5.0                      # TinyMPC quadrotor example (problem_data/quadrotor_20hz_params.hpp:5)
DT = 0.05                      # the example's only dt; TINY_CASE's Qf is defined at this dt
DTS_EXTRAP = [0.01, 0.005]     # NOT TinyMPC's problem: its terminal weight dt*(Pinf-rho) moves with dt
TOLS = {
    # the example's own settings (tiny_api_constants.hpp defaults), solved cold to convergence
    "example": dict(abs_pri_tol=1e-3, abs_dua_tol=1e-3, max_iter=200000),
    # the paper's Teensy random-QP benchmark tolerances (1e-4 constraint / 1e-3 objective, 4000 max)
    "teensy-bench": dict(abs_pri_tol=1e-4, abs_dua_tol=1e-3, max_iter=4000),
    "tight": dict(abs_pri_tol=1e-9, abs_dua_tol=1e-9, max_iter=500000),
}
CLOSED_LOOP = dict(abs_pri_tol=1e-3, abs_dua_tol=1e-3, max_iter=100)   # examples/quadrotor_hovering.cpp:54
CL_STEPS = 100
# pub = TinyMPC as published (paper eq. 15, firmware cache: single rho); lib = tinympc 0.0.8 library (double-rho bug)
VARIANTS = ("pub", "lib")
VLABEL = {"pub": "TinyMPC-pub", "lib": "TinyMPC-lib2rho"}
REPS_PY = int(os.environ.get("COMPARE_REPS", "30"))
REPS_CPP = int(os.environ.get("COMPARE_REPS_CPP", "200"))
REF_DT = 0.0005
OURS_TIMEOUT = int(os.environ.get("COMPARE_OURS_TIMEOUT", "900"))


# ----------------------------------------------------------------------------- discretization / scoring
def zoh(A, B, h):
    n, m = B.shape
    M = np.zeros((n + m, n + m)); M[:n, :n] = A; M[:n, n:] = B
    E = expm(M * h)
    return E[:n, :n], E[:n, n:]


def zoh_cost(A, B, Qc, Rc, h):
    """Exact ZOH cost matrix: int_0^h z(t)' W z(t) dt = z0' Mh z0, z=[x;u] with u held (Van Loan)."""
    n, m = B.shape
    F = np.zeros((n + m, n + m)); F[:n, :n] = A; F[:n, n:] = B
    W = block_diag(Qc, Rc)
    k = n + m
    G = np.zeros((2 * k, 2 * k)); G[:k, :k] = -F.T; G[:k, k:] = W; G[k:, k:] = F
    E = expm(G * h)
    Mh = E[k:, k:].T @ E[:k, k:]
    return 0.5 * (Mh + Mh.T)


_score_cache = {}


def score_zoh(u_seq, h):
    """Exact continuous J for piecewise-constant u (row u_k on [k h, (k+1) h]) on the scoring model."""
    if h not in _score_cache:
        _score_cache[h] = (*zoh(A_C, B_C, h), zoh_cost(A_C, B_C, Q, R, h))
    Ad, Bd, Mh = _score_cache[h]
    x = X0.copy(); J = 0.0; xs = [x.copy()]
    for u in u_seq:
        z = np.concatenate([x, u]); J += 0.5 * z @ Mh @ z
        x = Ad @ x + Bd @ u; xs.append(x.copy())
    J += 0.5 * x @ QF @ x
    return J, np.array(xs)


def simpson(y, t):
    n = len(t) - 1
    assert n % 2 == 0
    h = t[1] - t[0]
    return h / 3 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum())


def score_dense(t, xs, us):
    integrand = np.einsum("ki,ij,kj->k", xs, Q, xs) + np.einsum("ki,ij,kj->k", us, R, us)
    return 0.5 * simpson(integrand, t) + 0.5 * xs[-1] @ QF @ xs[-1]


# ----------------------------------------------------------------------------- TinyMPC's effective QP
def tiny_pinf(Ad, Bd, Q1, R1, rho):
    """Replicates the Riccati loop of tiny_precompute_and_set_cache (tiny_api.cpp:266-286), 1e-5 stop."""
    Kt = np.zeros((NU, NX)); Pt = rho * np.eye(NX); P = Pt
    for _ in range(1000):
        K = np.linalg.solve(R1 + Bd.T @ Pt @ Bd, Bd.T @ Pt @ Ad)
        P = Q1 + Ad.T @ Pt @ (Ad - Bd @ K)
        if np.max(np.abs(K - Kt)) < 1e-5:
            break
        Kt, Pt = K, P
    return P


def tiny_effective_qp(Ad, Bd, dt, variant):
    """Stage weights and terminal weight (J units = dt x TinyMPC's per-step sum) of the QP TinyMPC converges to.
      pub: stage (Q, R), terminal Pinf(Q+rho, R+rho) - rho      [paper eq. 15 / firmware cache]
      lib: stage (Q+rho, R+rho), terminal Pinf(Q+2rho, R+2rho) - rho   [tinympc 0.0.8 tiny_setup double rho]"""
    I, Iu = np.eye(NX), np.eye(NU)
    if variant == "lib":
        P = tiny_pinf(Ad, Bd, Q + 2 * RHO * I, R + 2 * RHO * Iu, RHO)
        Qs, Rs = Q + RHO * I, R + RHO * Iu
    else:
        P = solve_discrete_are(Ad, Bd, Q + RHO * I, R + RHO * Iu)
        Qs, Rs = Q, R
    return dt * block_diag(Qs, Rs), dt * (P - RHO * I)


# ----------------------------------------------------------------------------- OSQP discrete QP
def osqp_solve(Ad, Bd, Mstage, Pf, N, eps=1e-10):
    """min sum_k 1/2 [x_k;u_k]' Mstage [x_k;u_k] + 1/2 x_N' Pf x_N, x_{k+1}=Ad x_k+Bd u_k, x_0=X0, |u|<=0.5."""
    import osqp
    nxv = (N + 1) * NX; nv = nxv + N * NU
    Mxx, Mxu, Muu = Mstage[:NX, :NX], Mstage[:NX, NX:], Mstage[NX:, NX:]
    P = sp.lil_matrix((nv, nv))
    for k in range(N):
        ix = slice(k * NX, (k + 1) * NX); iu = slice(nxv + k * NU, nxv + (k + 1) * NU)
        P[ix, ix] = Mxx; P[ix, iu] = Mxu; P[iu, ix] = Mxu.T; P[iu, iu] = Muu
    P[N * NX:nxv, N * NX:nxv] = Pf
    P = sp.triu(P.tocsc(), format="csc")
    Aeq = sp.lil_matrix(((N + 1) * NX, nv))
    Aeq[0:NX, 0:NX] = np.eye(NX)
    for k in range(N):
        r = slice((k + 1) * NX, (k + 2) * NX)
        Aeq[r, (k + 1) * NX:(k + 2) * NX] = np.eye(NX)
        Aeq[r, k * NX:(k + 1) * NX] = -Ad
        Aeq[r, nxv + k * NU:nxv + (k + 1) * NU] = -Bd
    beq = np.zeros((N + 1) * NX); beq[:NX] = X0
    Ain = sp.hstack([sp.csc_matrix((N * NU, nxv)), sp.eye(N * NU)])
    Acon = sp.vstack([Aeq.tocsc(), Ain], format="csc")
    lo = np.concatenate([beq, np.tile(UMIN_V, N)]); hi = np.concatenate([beq, np.tile(UMAX_V, N)])
    prob = osqp.OSQP()
    t0 = time.perf_counter()
    prob.setup(P, np.zeros(nv), Acon, lo, hi, eps_abs=eps, eps_rel=eps, max_iter=1000000,
               polishing=True, verbose=False)
    res = prob.solve()
    wall = time.perf_counter() - t0
    us = res.x[nxv:].reshape(N, NU)
    info = res.info
    return us, dict(iterations=int(info.iter), converged=(str(info.status) == "solved"),
                    osqp_status=str(info.status), polish=int(getattr(info, "status_polish", 0)),
                    t_c_run_us=1e6 * info.run_time, t_c_solve_us=1e6 * info.solve_time,
                    t_py_us=1e6 * wall, discrete_obj=float(info.obj_val))


# ----------------------------------------------------------------------------- TinyMPC
class _Silence:
    """Redirect fd 1 to /dev/null: TinyMPC's solve() prints via std::cout (admm.cpp:371)."""
    def __enter__(self):
        sys.stdout.flush(); self.saved = os.dup(1)
        self.null = os.open(os.devnull, os.O_WRONLY); os.dup2(self.null, 1)

    def __exit__(self, *a):
        os.dup2(self.saved, 1); os.close(self.saved); os.close(self.null)


def _tiny_new(Ad, Bd, N_knots, tol, variant):
    import tinympc
    prob = tinympc.TinyMPC()
    prob.setup(Ad, Bd, Q, R, N_knots, rho=RHO, u_min=UMIN_V, u_max=UMAX_V, **tol)
    if variant == "pub":
        prob.compute_cache_terms()              # interface.py:345-386: single rho (paper/firmware)
    return prob


def tiny_solve_py(Ad, Bd, N_knots, tol, reps, variant):
    times = []; out = None
    with _Silence():
        for _ in range(reps):
            prob = _tiny_new(Ad, Bd, N_knots, tol, variant)
            prob.set_x0(X0)
            t0 = time.perf_counter(); sol = prob.solve(); t1 = time.perf_counter()
            times.append(t1 - t0)
            out = (np.array(sol["controls_all"]), int(prob._solver.solution.iter),
                   bool(prob._solver.solution.solved))
    us, it, ok = out
    return us, dict(iterations=it, converged=ok, t_py_us=1e6 * float(np.median(times)),
                    t_py_us_min=1e6 * float(np.min(times)))


def tiny_closed_loop_py(Ad, Bd, N_knots, variant):
    """The example's loop (quadrotor_hovering.cpp:73-93): warm-started, cap 100, x <- A x + B u0.
    Python returns the projected u (znew); the C++ example uses work->u (unprojected)."""
    with _Silence():
        prob = _tiny_new(Ad, Bd, N_knots, CLOSED_LOOP, variant)
        x = X0.copy(); its = []; ts = []; first = None
        for k in range(CL_STEPS):
            prob.set_x0(x)
            t0 = time.perf_counter(); sol = prob.solve(); t1 = time.perf_counter()
            ts.append(t1 - t0); its.append(int(prob._solver.solution.iter))
            u = np.array(sol["controls_all"])
            if first is None:
                first = u
            x = Ad @ x + Bd @ u[0]
    return first, its, ts, x


def tiny_cpp(Ad, Bd, N_knots, tol, reps, variant, steps=0):
    exe = os.path.join(HERE, "cpp", "build", "tiny_bench")
    if not os.path.exists(exe):
        return None
    path = os.path.join(RES, "_tiny_problem.txt")
    with open(path, "w") as f:
        f.write(f"{NX} {NU} {N_knots} {RHO!r} {tol['abs_pri_tol']!r} {tol['abs_dua_tol']!r} "
                f"{tol['max_iter']} {reps} {1 if variant == 'pub' else 0} {steps}\n")
        for arr in (Ad, Bd, np.diag(Q), np.diag(R), X0, UMIN_V, UMAX_V):
            f.write(" ".join(repr(float(v)) for v in np.ravel(arr)) + "\n")
    out = subprocess.run([exe, path], capture_output=True, text=True, timeout=3600)
    os.remove(path)
    return json.loads([l for l in out.stdout.splitlines() if l.startswith("{")][-1])


# ----------------------------------------------------------------------------- ours (child process)
def _ours_python_child(q, model):
    try:
        import lqocp
        sol = lqocp.solve(qp.Af, qp.Bf, **qp.TINY_CASE)
        reps = []
        for _ in range(3):
            t0 = time.perf_counter(); lqocp.solve(qp.Af, qp.Bf, **qp.TINY_CASE); reps.append(time.perf_counter() - t0)
        t = np.linspace(0, T, 20001)
        s = np.asarray(sol.x(t)); v = np.asarray(sol.u(t))
        sw = np.array(list(getattr(sol, "switch_times", []) or []), dtype=float)
        v_sw = np.asarray(sol.u(sw)).reshape(-1, NU) if len(sw) else np.zeros((0, NU))
        from scipy.integrate import solve_ivp
        r = solve_ivp(lambda tt, x: A_C @ x + B_C @ (qp.Umap @ np.asarray(sol.u(tt))), (0, T), X0,
                      rtol=1e-12, atol=1e-13, method="DOP853")
        q.put(("ok", dict(s=s, v=v, v_sw=v_sw, t=t, cost=float(sol.cost), info=dict(sol.info),
                          xT_resim=r.y[:, -1], t_reps=reps)))
    except BaseException as e:  # noqa: BLE001
        q.put(("err", f"{type(e).__name__}: {e}".splitlines()[0][:200]))


def ours_python():
    if not MAPS_CONSISTENT:
        raise RuntimeError(f"quad_problem.Umap inconsistent with Bc (residual {UMAP_RESIDUAL:.3g})")
    ctx = mp.get_context("fork")
    q = ctx.Queue(); p = ctx.Process(target=_ours_python_child, args=(q, MODEL)); p.start()
    try:
        status, d = q.get(timeout=OURS_TIMEOUT)
    except Exception:
        p.kill(); p.join()
        raise RuntimeError(f"develop/ solver crashed or exceeded {OURS_TIMEOUT}s (exit code {p.exitcode})")
    p.join()
    if status != "ok":
        raise RuntimeError(d)
    xs = d["s"] @ qp.Tf2p.T; us = d["v"] @ qp.Umap.T
    viol = max(0.0, float(np.max(np.abs(np.vstack([us, d["v_sw"] @ qp.Umap.T]))) - UMAX))
    info = d["info"]
    if not info.get("converged", False):
        raise RuntimeError(f"develop/ solver returned converged=False after {info.get('iterations')} iterations "
                           f"(max bound violation {viol:.2g}); not scored")
    return dict(J=score_dense(d["t"], xs, us), J_reported=d["cost"], viol=viol, xT=xs[-1],
                xT_resim_maxdiff=float(np.max(np.abs(d["xT_resim"] - xs[-1]))),
                iterations=info.get("iterations"), converged=info.get("converged"),
                t_py_us=1e6 * float(np.median(d["t_reps"])),
                t_solver_us=1e6 * float(info.get("solve_time_s", np.nan)),
                traj=(d["t"][::20], xs[::20], us[::20]), method=info.get("method"))


def ours_cpp():
    if not MAPS_CONSISTENT:
        raise RuntimeError(f"quad_problem.Umap inconsistent with Bc (residual {UMAP_RESIDUAL:.3g})")
    exe = os.path.join(ROOT, "develop", "cpp", "build", "lqocp_cli")
    csv = os.path.join(RES, "_ours_cpp.csv")

    def run():
        o = subprocess.run([exe, "TINY", csv], capture_output=True, text=True, timeout=OURS_TIMEOUT)
        if o.returncode != 0 or not os.path.exists(csv):
            raise RuntimeError(f"lqocp_cli TINY failed (exit {o.returncode}): "
                               + (o.stdout.strip() + " " + o.stderr.strip()).strip()[:150])
        return json.loads([l for l in o.stdout.splitlines() if l.strip().startswith("{")][-1])
    js = run()
    ts = [float(js.get("solve_time_us", np.nan))]
    for _ in range(19):
        ts.append(float(run()["solve_time_us"]))
    d = np.loadtxt(csv, delimiter=",", skiprows=1); os.remove(csv)
    t = d[:, 0]; xs = d[:, 1:13] @ qp.Tf2p.T; us = d[:, 13:17] @ qp.Umap.T
    viol = max(0.0, float(np.max(np.abs(us)) - UMAX))
    if abs(t[-1] - T) > 1e-9:
        raise RuntimeError(f"lqocp_cli TINY is stale: its horizon is T={t[-1]:g}, not {T:g} (Revision 5); "
                           f"reported cost {js.get('cost')}")
    if not js.get("converged", False):
        raise RuntimeError(f"lqocp_cli TINY returned converged=false after {js.get('iterations')} iterations")
    return dict(J=score_dense(t, xs, us), J_reported=float(js["cost"]), viol=viol, xT=xs[-1],
                iterations=js.get("iterations"), converged=js.get("converged"),
                t_cpp_us=float(np.median(ts)), t_cpp_us_min=float(np.min(ts)), traj=(t, xs, us),
                method=js.get("method"))


# ----------------------------------------------------------------------------- main
def write_traj(name, t, xs, us):
    hdr = "t," + ",".join(f"x{i}" for i in range(NX)) + "," + ",".join(f"u{i}" for i in range(NU))
    np.savetxt(os.path.join(RES, f"traj_{name}.csv"), np.column_stack([t, xs, us]), delimiter=",",
               header=hdr, comments="", fmt="%.12g")


def zoh_traj(us, h):
    N = len(us); t = h * np.arange(N + 1)
    _, xs = score_zoh(us, h)
    return t, xs, np.vstack([us, np.full((1, NU), np.nan)])


def zrow(name, us, dt, section="main", **kw):
    J, xs = score_zoh(us, dt)
    return dict(row=name, section=section, J=J, viol=max(0.0, float(np.abs(us).max()) - UMAX),
                xT=xs[-1], status="ok", **kw)


def tiny_rows(dt, section, rows):
    N = int(round(T / dt)); Nk = N + 1     # TinyMPC N = KNOT POINTS = intervals + 1 (10 at dt = 0.05)
    Ad, Bd = zoh(A_C, B_C, dt)
    eff = {v: tiny_effective_qp(Ad, Bd, dt, v) for v in VARIANTS}
    sfx = "" if section == "main" else " (extrap)"
    osqp_u = {}
    for v in VARIANTS:
        us, inf = osqp_solve(Ad, Bd, *eff[v], N)
        osqp_u[v] = us
        rows.append(zrow(f"OSQP dt={dt} tinyQP-{v}{sfx}", us, dt, section, **inf))
        write_traj(f"osqp_dt{dt}_tinyQP_{v}", *zoh_traj(us, dt))
    for v in VARIANTS:
        for tname, tol in TOLS.items():
            reps_py = REPS_PY if tname == "teensy-bench" else max(3, REPS_PY // 10)
            reps_cpp = REPS_CPP if tname == "teensy-bench" else max(5, REPS_CPP // 20)
            us, inf = tiny_solve_py(Ad, Bd, Nk, tol, reps_py, v)
            assert us.shape == (N, NU), us.shape
            cpp = tiny_cpp(Ad, Bd, Nk, tol, reps_cpp, v)
            if cpp is not None:
                inf.update(t_cpp_us=cpp["solve_time_us_median"], t_cpp_us_min=cpp["solve_time_us_min"],
                           cpp_iterations=cpp["iterations"],
                           cpp_u_maxdiff_vs_py=float(np.max(np.abs(np.array(cpp["u"]) - us))))
            inf["u_maxdiff_vs_osqp_tinyQP"] = float(np.max(np.abs(us - osqp_u[v])))
            rows.append(zrow(f"{VLABEL[v]} dt={dt} {tname}{sfx}", us, dt, section, **inf))
            write_traj(f"tinympc_{v}_dt{dt}_{tname}", *zoh_traj(us, dt))
        if section == "main":
            # the example's warm-started closed loop (tol 1e-3, cap 100); J column = step-0 plan (capped)
            first, its_py, ts_py, xfin = tiny_closed_loop_py(Ad, Bd, Nk, v)
            cl = tiny_cpp(Ad, Bd, Nk, CLOSED_LOOP, 1, v, steps=CL_STEPS)
            its = cl["iters"] if cl else its_py
            rows.append(zrow(f"{VLABEL[v]} closed-loop (example)", first, dt, section,
                             iterations=f"{np.mean(its):.1f}/st", converged=bool(min(cl["solved"])) if cl else None,
                             closed_loop=dict(steps=CL_STEPS, iters_cpp=its, iters_total_cpp=int(sum(its)),
                                              steps_capped_cpp=int(sum(1 for s in cl["solved"] if not s)) if cl else None,
                                              iters_py=its_py, iters_total_py=int(sum(its_py)),
                                              x_final_norm=float(np.linalg.norm(xfin))),
                             t_py_us=1e6 * float(np.median(ts_py)),
                             t_cpp_us=cl["solve_time_us_median"] if cl else None,
                             t_cpp_us_mean=cl["solve_time_us_mean"] if cl else None))
    return dict(terminal_vs_Qf_relfro={v: float(np.linalg.norm(eff[v][1] - QF) / np.linalg.norm(QF)) for v in VARIANTS},
                x0_cost_terminal={v: float(0.5 * X0 @ eff[v][1] @ X0) for v in VARIANTS},
                stage_vs_rect_relfro={v: float(np.linalg.norm(eff[v][0] - dt * block_diag(Q, R))
                                               / np.linalg.norm(dt * block_diag(Q, R))) for v in VARIANTS})


def main():
    global A_C, B_C, QF, MODEL, RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["spec", "phys"], default="phys")
    MODEL = ap.parse_args().model
    if MODEL == "phys":
        A_C, B_C, QF = qp.Ac, qp.Bc, qp.QF_TINY_PHYS
        RES = os.path.join(HERE, "results")
    else:
        A_C = qp.Tf2p @ qp.Af @ qp.Tp2f
        B_C = qp.Tf2p @ qp.Bf @ np.linalg.inv(qp.Umap)
        QF = qp.Tp2f.T @ qp.TINY_CASE["Qf"] @ qp.Tp2f
        RES = os.path.join(HERE, "results", "spec")
    os.makedirs(RES, exist_ok=True)

    rows = []
    notes = dict(model=MODEL, T=T, umap_residual=UMAP_RESIDUAL, state_map_residual=STATE_MAP_RESIDUAL,
                 maps_consistent=MAPS_CONSISTENT,
                 scoring_model_vs_Ac=float(np.max(np.abs(A_C - qp.Ac))),
                 scoring_model_vs_Bc=float(np.max(np.abs(B_C - qp.Bc))),
                 Qf_used_vs_QF_TINY_PHYS_relfro=float(np.linalg.norm(QF - qp.QF_TINY_PHYS) / np.linalg.norm(qp.QF_TINY_PHYS)),
                 QF_TINY_vs_QF_CARE_relfro=float(np.linalg.norm(qp.QF_TINY_PHYS - qp.QF_PHYS) / np.linalg.norm(qp.QF_PHYS)))

    for name, fn in (("ours-python", ours_python), ("ours-cpp", ours_cpp)):
        try:
            r = fn(); r.update(row=name, status="ok", section="main")
            write_traj(name, *r.pop("traj"))
        except Exception as e:  # noqa: BLE001
            r = dict(row=name, section="main", status="pending",
                     error=f"{type(e).__name__}: {e}".splitlines()[0][:200])
        rows.append(r)

    # fine-grid exact-ZOH reference: an upper bound on the continuous optimum, independent of develop/
    Nref = int(round(T / REF_DT))
    Ad, Bd = zoh(A_C, B_C, REF_DT)
    us, inf = osqp_solve(Ad, Bd, zoh_cost(A_C, B_C, Q, R, REF_DT), QF, Nref)
    rows.append(zrow(f"OSQP dt={REF_DT} exactZOH (ref)", us, REF_DT, **inf))
    write_traj("osqp_ref", *zoh_traj(us, REF_DT))
    J_fine = rows[-1]["J"]

    # best possible ZOH input per grid (legitimate discretizations of TINY_CASE: Qf fixed)
    for dt in [DT] + DTS_EXTRAP:
        N = int(round(T / dt)); Ad, Bd = zoh(A_C, B_C, dt)
        us, inf = osqp_solve(Ad, Bd, zoh_cost(A_C, B_C, Q, R, dt), QF, N)
        rows.append(zrow(f"OSQP dt={dt} exactZOH", us, dt, **inf))
        write_traj(f"osqp_dt{dt}_exactZOH", *zoh_traj(us, dt))

    terminal = {str(DT): tiny_rows(DT, "main", rows)}
    for dt in DTS_EXTRAP:
        terminal[str(dt)] = tiny_rows(dt, "extrap", rows)

    ours = rows[0]
    if ours["status"] == "ok":
        J_ref, ref_name = ours["J"], "ours-python (continuous optimum)"
    else:
        J_ref, ref_name = J_fine, f"OSQP exact-ZOH dt={REF_DT} (upper bound on the continuous optimum; ours pending)"
    for r in rows:
        if r.get("status") == "ok":
            r["gap"] = r["J"] - J_ref; r["gap_rel"] = r["gap"] / J_ref
            r["gap_vs_fineZOH"] = r["J"] - J_fine
            r["xT_norm"] = float(np.linalg.norm(r["xT"]))

    # ---- table
    f = lambda v, fmt: (format(v, fmt) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else "-")
    hdr = (f"{'row':40s} {'J_cont':>11s} {'gap':>9s} {'gap_rel':>9s} {'viol':>8s} {'|x(T)|':>7s} "
           f"{'iter':>7s} {'t_py[us]':>9s} {'t_cpp[us]':>9s}")

    def fmt_row(r):
        if r.get("status") != "ok":
            return f"{r['row']:40s} PENDING ({r.get('error', '')[:100]})"
        tcpp = r.get("t_cpp_us", r.get("t_c_solve_us"))
        it = r.get("iterations"); it = "-" if it is None else str(it) + ("" if r.get("converged", True) in (True, None) else "*")
        return (f"{r['row']:40s} {r['J']:11.6f} {r['gap']:9.2e} {r['gap_rel']:9.2e} {r['viol']:8.1e} "
                f"{r['xT_norm']:7.4f} {it:>7s} {f(r.get('t_py_us'), '9.1f'):>9s} {f(tcpp, '9.1f'):>9s}")

    lines = [f"model={MODEL}  T={T}  Qf=QF_TINY_PHYS  reference for gap: {ref_name}, J_ref = {J_ref:.10g}",
             hdr, "-" * len(hdr)]
    lines += [fmt_row(r) for r in rows if r["section"] == "main"]
    lines += ["", "EXTRAPOLATIONS (not TinyMPC's problem: its terminal weight dt*(Pinf-rho) changes with dt, so these",
              "TinyMPC rows solve a different terminal cost than TINY_CASE):", "-" * len(hdr)]
    lines += [fmt_row(r) for r in rows if r["section"] == "extrap"]
    lines += ["",
              "TinyMPC-pub = TinyMPC as published (single rho: paper eq. 15, firmware cache); TinyMPC-lib2rho = tinympc 0.0.8",
              "library as shipped (tiny_setup double-rho bug). Tolerances: example = 1e-3/1e-3 (the example's defaults, cold, to",
              "convergence); teensy-bench = pri 1e-4 / dua 1e-3, max 4000 (paper's Teensy random-QP benchmark); tight = 1e-9.",
              "closed-loop = the example's warm-started loop (tol 1e-3, cap 100, 100 steps): iter = mean per step, times = median",
              "per step, J = the step-0 plan (capped at 100 iterations). iter*: max_iter hit without meeting the tolerance.",
              "t_py: per-solve Python wall time (median, cold, setup excluded). t_cpp: host C++ -- TinyMPC via cpp/tiny_bench",
              "(g++ -O3), OSQP = internal solve_time (factorization excluded), ours = lqocp_cli solve_time_us (median of 20).",
              "Host x86 only; NONE are Teensy numbers."]
    table = "\n".join(lines)
    print(table)

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, (np.floating, float)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, (np.integer, np.bool_)):
            return o.item()
        return o

    with open(os.path.join(RES, "summary.json"), "w") as fh:
        json.dump(clean(dict(model=MODEL, reference=ref_name, J_ref=J_ref, J_fine_zoh=J_fine, rho=RHO, T=T,
                             tolerances=TOLS, closed_loop_settings=CLOSED_LOOP, terminal_cost=terminal,
                             notes=notes, rows=rows, table=table)), fh, indent=1)
    with open(os.path.join(RES, "summary.txt"), "w") as fh:
        fh.write(table + "\n")
    print("terminal-cost check:", json.dumps(terminal, indent=1))
    print("notes:", json.dumps(notes, indent=1))


if __name__ == "__main__":
    main()
