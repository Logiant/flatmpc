"""Revision 3 cases (general LQ cost) and their references.

  TINY: quad_problem.TINY_CASE (shared), also written to cases/tiny.txt.
        Ref: direct QP (motor-thrust variable), piecewise-uniform grid refined in [0, 0.05],
        polished to the exact discrete optimum by primal-dual active-set iterations; 2 levels.
  L1:   TINY's Q, R; Qf = 10 * Qf_ARE (so the Riccati solution P(t) is time-varying); hover-offset
        target xf (x,y,z) = (0.2,-0.1,0.3) m; T = 0.5; bounds |u| <= 20 (inactive; unconstrained
        peak checked below). Refs: multiprecision Hamiltonian exponential AND backward Riccati ODE.
  L2:   fixed endpoint s(T) = 0 with Q = TINY Q, R = TINY R, T = 1.0, no bounds.
        Ref: multiprecision Hamiltonian exponential.
"""
import json
import os
import time

import numpy as np

import cases
import lqref as L
import reference as ref
from reference import qp

HERE = os.path.dirname(os.path.abspath(__file__))
LQ_CASES = ["TINY", "L1", "L2"]
NPZ = os.path.join(cases.CASE_DIR, "lq_refs.npz")
JSN = os.path.join(cases.CASE_DIR, "lq_refs.json")
TINY_JSON = os.path.join(HERE, "results", "tiny_reference.json")
NG = 10001
TINY_LEVELS = {"coarse": [(0.05, 20000), (0.5, 9000)], "fine": [(0.05, 40000), (0.5, 18000)]}


def design():
    os.makedirs(cases.CASE_DIR, exist_ok=True)
    tc = dict(qp.TINY_CASE)
    cases.write_problem(cases.case_path("TINY"), tc, "TINY_CASE from quad_problem.py (Revision 3)")
    xf1 = qp.phys_to_flat(np.array([0.2, -0.1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0], float))
    c1 = dict(T=0.5, x0=qp.S0.copy(), xf=xf1, Umap=qp.Umap, u_min=-20 * np.ones(4), u_max=20 * np.ones(4),
              Q=tc["Q"], R=tc["R"], Qf=10 * tc["Qf"])
    if not os.path.exists(cases.case_path("L1")):   # Revision 5: L1/L2 are frozen self-contained files
      cases.write_problem(cases.case_path("L1"), c1, "validator case L1: TINY Q,R; Qf = 10*Qf_ARE; xf = hover offset "
                        "(0.2,-0.1,0.3) m; T=0.5; |u|<=20 (inactive)")
    c2 = dict(T=1.0, x0=qp.S0.copy(), xf=qp.SF.copy(), Umap=None, u_min=None, u_max=None,
              Q=tc["Q"], R=tc["R"], Qf=None)
    if not os.path.exists(cases.case_path("L2")):
      cases.write_problem(cases.case_path("L2"), c2, "validator case L2: fixed endpoint s(T)=0, TINY Q and R, T=1, no bounds")


def _resample(t_src, Y, t):
    return np.stack([np.interp(t, t_src, Y[:, j]) for j in range(Y.shape[1])], 1)


def build(verbose=True):
    design()
    out_np, out_js = {}, {}
    tg = None
    # ---------------- L1: Hamiltonian (mp) + Riccati
    c1 = cases.load_case("L1")
    t0 = time.time()
    h1 = L.hamiltonian_mp(dict(c1, u_min=None, u_max=None), ngrid=NG)
    r1 = L.riccati(dict(c1, u_min=None, u_max=None), ngrid=NG)
    peak = np.max(np.abs(h1["V"] @ c1["Umap"].T))
    Jq = _quad_cost(c1, h1["t"], h1["V"], h1["X"])
    out_np.update(L1_t=h1["t"], L1_V=h1["V"], L1_X=h1["X"], L1_Lam=h1["Lam"])
    out_js["L1"] = dict(cost=h1["cost"], cost_riccati=r1["cost"], cost_quadrature=Jq, switch_times=[],
                        dps=h1["dps"], peak_motor=peak, bound=20.0,
                        traj_diff_riccati_vs_mp=dict(x=float(np.max(np.abs(r1["X"] - h1["X"]))),
                                                     v=float(np.max(np.abs(r1["V"] - h1["V"])))),
                        sT=h1["sT"].tolist())
    if verbose:
        print(f"L1 (Qf=10*Qf_ARE, xf hover offset, T=0.5, |u|<=20): built in {time.time() - t0:.1f}s")
        print(f"  J (Hamiltonian expm, {h1['dps']} digits) = {h1['cost']:.15e}")
        print(f"  J (Riccati ODE, DOP853 rtol 1e-13)     = {r1['cost']:.15e}  rel diff {abs(r1['cost'] - h1['cost']) / h1['cost']:.1e}")
        print(f"  J (quadrature of mp trajectory)        = {Jq:.15e}  rel diff {abs(Jq - h1['cost']) / h1['cost']:.1e}")
        print(f"  Riccati vs mp trajectory: max|dx| {out_js['L1']['traj_diff_riccati_vs_mp']['x']:.1e}, "
              f"max|dv| {out_js['L1']['traj_diff_riccati_vs_mp']['v']:.1e} (max|v| {np.max(np.abs(h1['V'])):.1f})")
        print(f"  unconstrained peak |u| = {peak:.4f} < 20 -> bounds inactive")
    # ---------------- L2: Hamiltonian (mp), fixed endpoint
    c2 = cases.load_case("L2")
    t0 = time.time()
    h2 = L.hamiltonian_mp(c2, ngrid=NG)
    Jq2 = _quad_cost(c2, h2["t"], h2["V"], h2["X"])
    out_np.update(L2_t=h2["t"], L2_V=h2["V"], L2_X=h2["X"], L2_Lam=h2["Lam"])
    out_js["L2"] = dict(cost=h2["cost"], cost_quadrature=Jq2, switch_times=[], dps=h2["dps"], bc_err=h2["bc_err"])
    # double-precision attempt for comparison (shows why mp is needed)
    if verbose:
        print(f"L2 (fixed endpoint, Q=TINY Q, R=TINY R, T=1): built in {time.time() - t0:.1f}s")
        print(f"  J (Hamiltonian expm, {h2['dps']} digits) = {h2['cost']:.15e}; |s(T)-xf| = {h2['bc_err']:.1e}")
        print(f"  J (quadrature of mp trajectory)        = {Jq2:.15e}  rel diff {abs(Jq2 - h2['cost']) / h2['cost']:.1e}")
        print(f"  grid endpoint |X[-1]-xf| = {np.max(np.abs(h2['X'][-1] - c2['xf'])):.1e}")
    # ---------------- TINY: QP + active-set polish, two levels
    ct = dict(qp.TINY_CASE)
    t0 = time.time()
    lev = {}
    # pass 0: locate the switches; then two adaptive levels with fine windows (+-2 ms) around them
    a0 = L.polish_active_set(ct, [(0.05, 20000), (ct["T"], int(round((ct["T"] - 0.05) / 5e-5)))])
    sw0 = [s_[0] for s_ in L.switch_times_polished(a0["t"], a0["U"], a0["mu"], ct["u_min"], ct["u_max"])]
    for name, (hf, hc) in {"coarse": (2.5e-6, 5e-5), "fine": (1.25e-6, 2.5e-5)}.items():
        reg = _adaptive_regions(sw0, ct["T"], hf, hc, w=2e-3)
        a = L.polish_active_set(ct, reg)
        a["sw"] = L.switch_times_polished(a["t"], a["U"], a["mu"], ct["u_min"], ct["u_max"])
        a["regions"] = reg
        lev[name] = a
    f, cs = lev["fine"], lev["coarse"]
    tg = np.linspace(0, ct["T"], NG)
    Vf, Xf = _resample(f["t"], f["V"], tg), _resample(f["t"], f["X"], tg)
    Vc, Xc = _resample(cs["t"], cs["V"], tg), _resample(cs["t"], cs["X"], tg)
    Uf = Vf @ ct["Umap"].T
    swf = [s[0] for s in f["sw"]]; swc = [s[0] for s in cs["sw"]]
    sw_err = float(max(np.max(np.abs(np.array(swf) - np.array(swc))),
                       max(abs(s[3] - s[4]) for s in f["sw"]))) if len(swf) == len(swc) else np.nan
    dv = float(np.max(np.abs(Vf - Vc))); dx = float(np.max(np.abs(Xf - Xc)))
    dJ = abs(f["cost"] - cs["cost"])
    # IPM-only QP (no polish) at the coarse level, to document what it misses
    ipm = L.qp_general(ct, None, "u", regions=[(0.05, 5000), (ct["T"], int(round((ct["T"] - 0.05) / 2e-4)))])
    # flat-v formulation at a uniform grid, same N as a u-formulation, for the conditioning report
    qu = L.qp_general(ct, 2000, "u"); qv = L.qp_general(ct, 2000, "v")
    out_np.update(TINY_t=tg, TINY_V=Vf, TINY_X=Xf, TINY_U=Uf)
    sT = f["X"][-1]
    arcs = _arcs_from_sw(f["sw"], ct["T"])
    out_js["TINY"] = dict(
        cost=f["cost"], cost_coarse=cs["cost"], cost_err_est=max(dJ, 1e-12 * f["cost"]),
        switch_times=swf, switch_detail=[dict(t=s[0], motor=s[1], kind=s[2], t_mu_root=s[3], t_u_cross=s[4]) for s in f["sw"]],
        switch_err_est=sw_err, dv=dv, dx=dx, arcs=arcs,
        grid=dict(fine=f["regions"], coarse=cs["regions"], n_fine=f["N"], n_coarse=cs["N"]),
        active_set_iterations=f["iterations"],
        ipm_only_cost=ipm["cost"], qp_u_vs_v_N2000=dict(J_u=qu["cost"], J_v=qv["cost"],
                                                        max_dv=float(np.max(np.abs(qu["V"] - qv["V"]))),
                                                        max_du=float(np.max(np.abs(qu["U"] - qv["U"]))),
                                                        max_v=float(np.max(np.abs(qu["V"])))),
        sT=sT.tolist(), xT_phys=(qp.Tf2p @ sT).tolist())
    if verbose:
        print(f"TINY (|u|<=0.5, Qf=QF_TINY (Rev 5), T={ct['T']}): built in {time.time() - t0:.1f}s")
        print(f"  J (polished QP, {f['N']} intervals) = {f['cost']:.14e}; coarse ({cs['N']}) {cs['cost']:.14e}; "
              f"|dJ| {dJ:.1e} (rel {dJ / f['cost']:.1e})")
        print(f"  J (IPM QP without polish, coarse grid) = {ipm['cost']:.14e}")
        print(f"  switch times (fine): " + ", ".join(f"{s[0]:.7f} (m{s[1]} {s[2]})" for s in f["sw"]))
        print(f"  switch time error est. {sw_err:.1e} s; ref trajectory error est. max|dv| {dv:.1e} "
              f"(max|v| {np.max(np.abs(Vf)):.0f}), max|dx| {dx:.1e}")
        print(f"  conditioning: N=2000 uniform, u-variable J={qu['cost']:.14e}, flat-v variable (R_flat cond 2e6) "
              f"J={qv['cost']:.14e}, rel diff {abs(qu['cost'] - qv['cost']) / qu['cost']:.1e}; max|du| "
              f"{out_js['TINY']['qp_u_vs_v_N2000']['max_du']:.1e}, max|dv| {out_js['TINY']['qp_u_vs_v_N2000']['max_dv']:.1e}")
        print(f"  s(T) = {np.round(sT, 9).tolist()}")
    np.savez(NPZ, **out_np)
    with open(JSN, "w") as fh:
        json.dump(out_js, fh, indent=1, default=float)
    os.makedirs(os.path.dirname(TINY_JSON), exist_ok=True)
    tj = out_js["TINY"]
    ts = np.linspace(0, ct["T"], 101)
    with open(TINY_JSON, "w") as fh:
        json.dump(dict(
            description="Validator reference for quad_problem.TINY_CASE (Revision 5: general LQ cost, |u|<=0.5, Qf = QF_TINY_PHYS mapped to flat, T = 0.45). "
                        "Direct transcription (first-order-hold motor thrust u, exact discretisation and exact running-cost "
                        "integration, node bounds => feasible), adaptive piecewise-uniform grid (1.25e-6 s within +-2 ms of every "
                        "switch and on [0,2ms], 2.5e-5 s elsewhere), polished to the exact discrete optimum by primal-dual active-set iterations. Error "
                        "estimates from comparison with a 2x coarser grid. Independent of the developer's code.",
            cost=tj["cost"], cost_error_estimate=tj["cost_err_est"],
            cost_definition="J = 1/2 int_0^T (s'Qs + v'Rv) dt + 1/2 s(T)'Qf s(T), flat coordinates (= physical x'Qx + u'Ru)",
            switch_times=tj["switch_times"], switch_time_error_estimate_s=tj["switch_err_est"],
            switch_detail=tj["switch_detail"], active_arcs=tj["arcs"],
            final_state_flat=tj["sT"], final_state_phys=tj["xT_phys"],
            initial_state_flat=list(map(float, ct["x0"])),
            trajectory_101=dict(t=ts.tolist(), u_motor=_resample(f["t"], f["U"], ts).tolist(),
                                x_phys=(_resample(f["t"], f["X"], ts) @ qp.Tf2p.T).tolist()),
        ), fh, indent=1)
    return out_js


def _adaptive_regions(sw, T, hf, hc, w):
    """Piecewise-uniform grid: spacing ~hf inside [s-w, s+w] around each switch s, ~hc elsewhere
    (windows merged; also a fine window on [0, w] for the initial transient)."""
    win = sorted([(max(0.0, s - w), min(T, s + w)) for s in sw] + [(0.0, w)])
    merged = []
    for a, b in win:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    regions, t0 = [], 0.0
    for a, b in merged:
        if a > t0:
            regions.append((a, max(1, int(np.ceil((a - t0) / hc)))))
        regions.append((b, max(1, int(np.ceil((b - a) / hf)))))
        t0 = b
    if t0 < T:
        regions.append((T, max(1, int(np.ceil((T - t0) / hc)))))
    return regions


def _arcs_from_sw(sw, T):
    arcs = []
    by = {}
    for s in sw:
        by.setdefault(s[1], []).append(s)
    for m, lst in by.items():
        start = 0.0 if lst[0][2] == "exit" else None
        for s in lst:
            if s[2] == "entry":
                start = s[0]
            else:
                arcs.append(dict(motor=m, start=start, end=s[0], length=s[0] - start))
                start = None
        if start is not None:
            arcs.append(dict(motor=m, start=start, end=T, length=T - start))
    return arcs


def _quad_cost(c, t, V, X):
    """1/2 int (s'Qs + v'Rv) + terminal, composite Simpson on a fine smooth grid."""
    from scipy.integrate import simpson
    Q, R = L.Q_of(c), L.R_of(c)
    f = np.einsum("ki,ij,kj->k", X, Q, X) + np.einsum("ki,ij,kj->k", V, R, V)
    J = 0.5 * simpson(f, x=t)
    if c.get("Qf") is not None:
        d = X[-1] - c["xf"]; J += 0.5 * d @ np.asarray(c["Qf"]) @ d
    return float(J)


def load():
    d = np.load(NPZ)
    with open(JSN) as fh:
        js = json.load(fh)
    return {k: d[k] for k in d.files}, js


if __name__ == "__main__":
    build()
