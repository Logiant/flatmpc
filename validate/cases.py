"""Revision-2 validator cases: design, problem-file I/O (spec format), and reference solutions.

Cases (all use the exact physical Umap and the flat model):
  S  "sliver":     a motor bound is active on one interior arc much shorter than T/2000.
  G  "touch":      the unconstrained optimum grazes a bound: interior peak = bound/(1+1e-9).
  M  "many":       case-B boundary conditions with T = 0.82 (min. feasible T is 0.7987): 16
                   switches, 2-3 simultaneously active motors, a ~1.1e-5 s arc between two
                   near-simultaneous switches on different motors.
  I  "infeasible": case-B boundary conditions with T = 0.75 < T_min; certified infeasible.

S and G are built by inverse design: pick the costate polynomials p (v = p is then the
unconstrained optimum for x0 and xf := x(T)), and scale p so that the peak motor thrust is
0.3*(1+1e-7) (S) or 0.3/(1+1e-9) (G), with the physical bounds |u_i| <= 0.3 unchanged.

References: exact PMP-certified solutions (reference.pmp_solve: Newton on p with exact polynomial
event detection; KKT is sufficient for this convex problem), cross-checked by the direct QP
(reference.solve_qp, a feasible upper bound) at N = 4000 and 8000. Infeasibility is certified by
the continuous-time Farkas/support-function test (reference.feasibility_value).
"""
import json
import os

import numpy as np

import reference as ref
from reference import qp

HERE = os.path.dirname(os.path.abspath(__file__))
CASE_DIR = os.path.join(HERE, "cases")
REF_JSON = os.path.join(CASE_DIR, "refs.json")
NEW_CASES = ["S", "G", "M", "I", "S_old", "G_old", "M_old", "I_old"]
OLD = ("S_old", "G_old", "M_old", "I_old")     # Revision 4: synthetic polytope (pre-fix Umap), files kept
FILES = {"S": "sliver.txt", "G": "touch.txt", "M": "many.txt", "I": "infeasible.txt",
         "S_old": "S_old.txt", "G_old": "G_old.txt", "M_old": "M_old.txt", "I_old": "I_old.txt",
         "TINY": "tiny.txt", "L1": "lq_qf_inactive.txt", "L2": "lq_fixed_endpoint.txt"}
BOUND = 0.5         # Revision 4: physical bound used by case B
DELTA_S = 1e-7      # sliver: peak exceeds the bound by this relative amount
DELTA_G = 1e-9      # touch: bound exceeds the peak by this relative amount
T_M, T_I = 0.998, 0.95   # Revision 4 case-B BCs: minimum feasible T = 0.98825 (T=0.993 defeated our own Newton)


# ------------------------------------------------------------------------------------------------
# problem files (spec Revision 2 format)
# ------------------------------------------------------------------------------------------------
def write_problem(path, c, comment=""):
    f17 = lambda a: " ".join(f"{float(v):.17g}" for v in np.ravel(a))
    lines = [f"# {ln}" for ln in comment.splitlines()] if comment else []
    lines += ["# T", f"{c['T']:.17g}", "# x0", f17(c["x0"]), "# xf", f17(c["xf"])]
    if c["u_max"] is None:
        lines += ["# has_bounds", "0"]
    else:
        lines += ["# has_bounds", "1", "# Umap (row-major)", *[f17(row) for row in np.asarray(c["Umap"])],
                  "# u_min", f17(c["u_min"]), "# u_max", f17(c["u_max"])]
    if c.get("Q") is not None or c.get("R") is not None or c.get("Qf") is not None:   # Revision 3 block
        Qm = np.zeros((12, 12)) if c.get("Q") is None else np.asarray(c["Q"])
        Rm = np.eye(4) if c.get("R") is None else np.asarray(c["R"])
        lines += ["# has_cost", "1", "# Q (row-major)", *[f17(row) for row in Qm], "# R (row-major)", *[f17(row) for row in Rm]]
        if c.get("Qf") is None:
            lines += ["# has_Qf", "0"]
        else:
            lines += ["# has_Qf", "1", "# Qf (row-major)", *[f17(row) for row in np.asarray(c["Qf"])]]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def read_problem(path):
    nums = []
    with open(path) as fh:
        for ln in fh:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            nums += [float(v) for v in s.split()]
    T = nums[0]; x0 = np.array(nums[1:13]); xf = np.array(nums[13:25]); hb = int(nums[25])
    if hb:
        Um = np.array(nums[26:42]).reshape(4, 4); umin = np.array(nums[42:46]); umax = np.array(nums[46:50])
        p = 50
    else:
        Um = umin = umax = None
        p = 26
    out = dict(T=T, x0=x0, xf=xf, Umap=Um, u_min=umin, u_max=umax)
    if len(nums) > p:                     # Revision 3 cost block
        assert int(nums[p]) == 1
        out["Q"] = np.array(nums[p + 1:p + 145]).reshape(12, 12)
        out["R"] = np.array(nums[p + 145:p + 161]).reshape(4, 4)
        hq = int(nums[p + 161])
        out["Qf"] = np.array(nums[p + 162:p + 306]).reshape(12, 12) if hq else None
        assert len(nums) == p + (306 if hq else 162)
    return out


def case_path(name):
    return os.path.join(CASE_DIR, FILES[name])


def load_case(name):
    return read_problem(case_path(name))


# ------------------------------------------------------------------------------------------------
# design
# ------------------------------------------------------------------------------------------------
def _motor_poly(P, k):
    return qp.Umap[k] @ P                      # ascending coeffs of u_k(t)


def _peak(P, T):
    """Exact max_t max_k |u_k(t)| over [0,T] via roots of the derivative. Returns (peak, k, t)."""
    best = (-1.0, None, None)
    for k in range(4):
        c = _motor_poly(P, k)
        ts = [0.0, T] + ref._real_roots(np.polynomial.polynomial.polyder(c), 0.0, T)
        for t in ts:
            v = abs(np.polynomial.polynomial.polyval(t, c))
            if v > best[0]:
                best = (v, k, t)
    return best


def _base_P(T=1.0, b=0.25):
    """Revision 4 design (found by a small grid search for the best-isolated interior peak):
    x- and y-chain cubics t(T-t)(t-b), plus a constant z costate (collective thrust offset).
    The global peak is motor 1 at t~0.717; every other local extremum of every motor is <= 33%
    of it."""
    P = np.zeros((4, 4))
    P[0, :] = np.array([0, -b * T, T + b, -1.0])
    pk = _peak(P, T)[0]
    P *= BOUND / pk
    P[1, :] = 2.0 * P[0, :]
    P[2, 0] = BOUND * 0.1 / 0.0595
    return P


def _unconstrained_case(P, T, x0):
    c = dict(x0=x0, xf=np.zeros(12), T=T, Umap=None, u_min=None, u_max=None)
    xf = ref.pmp_eval(P, c)[0]
    return xf


def design_cases():
    os.makedirs(CASE_DIR, exist_ok=True)
    T = 1.0
    P = _base_P(T)
    pk, k, tp = _peak(P, T)
    x0 = qp.S0.copy()
    bounds = dict(Umap=qp.Umap, u_min=-BOUND * np.ones(4), u_max=BOUND * np.ones(4))
    out = {}
    # touch
    PG = P * (BOUND / (1 + DELTA_G)) / pk
    cG = dict(T=T, x0=x0, xf=_unconstrained_case(PG, T, x0), **bounds)
    write_problem(case_path("G"), cG, f"validator case G (tangential touch): unconstrained peak of motor {k} at "
                  f"t={tp:.6f} equals the bound {BOUND} to {DELTA_G:g} (relative), bound not exceeded (Revision 4 Umap)")
    # sliver
    PS = P * (BOUND * (1 + DELTA_S)) / pk
    cS = dict(T=T, x0=x0, xf=_unconstrained_case(PS, T, x0), **bounds)
    write_problem(case_path("S"), cS, f"validator case S (sliver arc): unconstrained peak of motor {k} at "
                  f"t={tp:.6f} exceeds the bound {BOUND} by {DELTA_S:g} (relative) (Revision 4 Umap)")
    # many switches
    cM = dict(qp.CASES["B"]); cM["T"] = T_M
    write_problem(case_path("M"), cM, f"validator case M (many switches): Revision-4 case B (|u|<=0.5), T={T_M} (T_min=0.98825)")
    cI = dict(qp.CASES["B"]); cI["T"] = T_I
    write_problem(case_path("I"), cI, f"validator case I (infeasible): Revision-4 case B (|u|<=0.5), T={T_I} < T_min=0.98825")
    return dict(G=PG, S=PS)


# ------------------------------------------------------------------------------------------------
# references
# ------------------------------------------------------------------------------------------------
def build_refs(verbose=True):
    Pdesign = design_cases()
    R = {}
    for name in ["B"] + NEW_CASES:
        c = dict(qp.CASES["B"]) if name == "B" else load_case(name)
        r = dict(file=FILES.get(name, "quad_problem.CASES[B]"), T=c["T"])
        fv2 = ref.feasibility_value(c, nq=2000)[0]
        fv4 = ref.feasibility_value(c, nq=4000)[0]
        r["feasibility_value"] = [fv2, fv4]
        r["feasible"] = bool(fv4 > 1)
        if name in ("I", "I_old"):
            # also the minimum feasible time for these BCs (bisection on the certificate)
            from scipy.optimize import brentq
            cc = dict(c)
            def g(T):
                cc["T"] = T
                return ref.feasibility_value(cc, nq=2000)[0] - 1
            r["T_min"] = brentq(g, 0.5 * c["T"], 1.5 * c["T"], xtol=1e-7)
            R[name] = r
            if verbose:
                print(f"Case {name}: T={c['T']}: certificate value {fv2:.4f} (nq=2000), {fv4:.4f} (nq=4000) < 1 "
                      f"=> INFEASIBLE; minimum feasible T = {r['T_min']:.6f}")
            continue
        # warm start: design p (S, G) or the QP fit (M)
        if name in Pdesign:
            P0 = Pdesign[name]
        elif name in ("S_old", "G_old"):
            P0, _ = ref.unconstrained_P(c)     # these were designed as v = p (unconstrained optimum)
        else:
            P0, _ = ref.warm_start_P(c, N=4000)
        s = ref.pmp_solve(c, P0)
        q4, q8 = ref.solve_qp(c, 4000), ref.solve_qp(c, 8000)
        # independent re-check of the terminal state with the generic integrator
        xT = ref.simulate_gauss(s.u, c["x0"], c["T"], [c["T"]], nsub=4000, breaks=s.switch_times)[0]
        XS = max(1.0, np.max(np.abs(c["xf"])), np.max(np.abs(c["x0"])))
        tt = np.linspace(0, c["T"], 20001)
        M = s.u(np.union1d(tt, s.switch_times)) @ c["Umap"].T
        viol = max(np.max(M - c["u_max"]), np.max(c["u_min"] - M), 0.0)
        arcs = [(a, b, p.tolist()) for a, b, p in [(pc[0], pc[1], pc[2]) for pc in s.pieces]]
        r.update(cost=s.cost, switch_times=list(s.switch_times), patterns=s.patterns, P=s.P.tolist(),
                 converged=bool(s.converged), newton_resid=s.resid, resid_generic=float(np.max(np.abs(xT - c["xf"])) / XS),
                 qp4000=q4["cost"], qp8000=q8["cost"], max_violation=viol,
                 min_arc=min(b - a for a, b, _ in arcs), arcs=[[a, b, p] for a, b, p in arcs],
                 qp8000_maxdv=float(np.max(np.abs(q8["V"] - s.u(q8["t"])))),
                qp8000_maxdx=float(np.max(np.abs(q8["X"][::40] - s.x(q8["t"][::40])))),
                qp_pattern_agreement=_pattern_agreement(c, q8, s))
        pk = _peak(np.array(r["P"]), c["T"])
        r["peak_of_p"] = [pk[0], pk[1], pk[2]]
        R[name] = r
        if verbose:
            print(f"Case {name}: T={c['T']}  feasibility certificate value {fv4:.4f} (>1 feasible)")
            print(f"  PMP exact: J = {s.cost:.15e}, converged={s.converged}, BC resid {s.resid:.1e} "
                  f"(generic integrator {r['resid_generic']:.1e}), max violation {viol:.1e}")
            print(f"  QP upper bounds: J_4000 - J = {q4['cost'] - s.cost:.2e}, J_8000 - J = {q8['cost'] - s.cost:.2e}")
            print(f"  switch times ({len(s.switch_times)}): {[round(v, 9) for v in s.switch_times]}")
            print(f"  QP N=8000 vs PMP: max|v diff| {r['qp8000_maxdv']:.2e}, max|x diff| {r['qp8000_maxdx']:.2e}, "
                  f"active-set agreement at QP nodes >2 nodes from a PMP switch: {r['qp_pattern_agreement']:.4%}")
            print(f"  patterns: {s.patterns}")
            if len(s.switch_times):
                print(f"  shortest arc {r['min_arc']:.3e} s (T/2000 = {c['T']/2000:.1e}); "
                      f"min gap between switches {np.min(np.diff([0] + s.switch_times + [c['T']])):.3e} s")
            print(f"  max |u| of the costate polynomial p (motor {pk[1]}, t={pk[2]:.6f}): {pk[0]:.12f}")
    with open(REF_JSON, "w") as fh:
        json.dump(R, fh, indent=1)
    return R


def _pattern_agreement(c, q, s, tol=1e-7):
    """Fraction of QP nodes (farther than 2 grid steps from any PMP switch) whose primal active set
    (|u_i - bound| < tol) equals the PMP pattern."""
    t = q["t"]; h = t[1] - t[0]
    M = q["V"] @ c["Umap"].T
    act = np.where(M >= c["u_max"] - tol, 1, np.where(M <= c["u_min"] + tol, -1, 0))
    n = ok = 0
    sw = np.array(s.switch_times)
    for i, ti in enumerate(t):
        if sw.size and np.min(np.abs(sw - ti)) < 2 * h:
            continue
        pat = next(p[2] for p in s.pieces if p[0] <= ti <= p[1])
        n += 1; ok += np.array_equal(act[i], pat)
    return ok / max(n, 1)


def load_refs():
    with open(REF_JSON) as fh:
        return json.load(fh)


class RefSolution:
    """Evaluate a stored PMP reference at arbitrary times (u exactly, x by exact piecewise quadrature)."""

    def __init__(self, name, r=None):
        r = r or load_refs()[name]
        self.c = dict(qp.CASES["B"]) if name == "B" else load_case(name)
        self.P = np.array(r["P"])
        self.sw = list(r["switch_times"])
        self.cost = r["cost"]
        c = self.c
        Um = c["Umap"]

        def u(t):
            t = np.atleast_1d(np.asarray(t, float))
            pt = np.polynomial.polynomial.polyval(t, self.P.T).T
            return np.array([ref.project(pp, Um, c["u_min"], c["u_max"])[0] for pp in pt])
        self.u = u

    def x(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        return ref.simulate_gauss(self.u, self.c["x0"], self.c["T"], t, nsub=200, order=8, breaks=self.sw)


if __name__ == "__main__":
    build_refs()
