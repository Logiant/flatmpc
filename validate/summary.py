"""Print the summary table from results/results.json (written by conftest.py).
One row per (check, impl, case); one column per method (newton, riccati); agreement checks in 'both'."""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(HERE, "results", "results.json")
if not os.path.exists(path):
    print("no results.json"); sys.exit(1)
R = json.load(open(path))

KEY = {  # metric shown per test
    "interface": ["solve_time_us", "solve_time_s"],
    "default_method": ["default_method"],
    "converged": [],
    "boundary": ["rel_err_xT"],
    "dynamics_consistency": ["rel_err"],
    "cost_quadrature": ["rel_err"],
    "cost_vs_reference": ["rel_err"],
    "constraints": ["max_violation"],
    "trajectory_vs_reference": ["rel_err_u"],
    "optimality_structure": ["worst", "rel_err_proj"],
    "switch_times": ["n", "max_err"],
    "py_vs_cpp": ["rel_err_u", "max_err_switch"],
    "infeasible": ["converged", "wall_s"],
    "py_umap_none_bounds": ["rel_err_cost"],
    "py_riccati_vs_newton": ["rel_err_cost", "max_err_switch"],
    "cpp_riccati_general_exit": ["returncode"],
}
SYM = {"passed": "PASS", "failed": "FAIL", "skipped": "skip", "xfail": "XFAIL*"}
METHODS = ("newton", "riccati")


def fmt(v):
    if isinstance(v, float):
        if v == 0 or (1e-3 <= abs(v) < 1e5):
            return f"{v:.3g}"
        return f"{v:.1e}"
    return str(v)


cells, notes = {}, []
for nid, r in R.items():
    m = re.match(r".*::test_(\w+)(?:\[(.*)\])?", nid)
    test, params = m.group(1), (m.group(2) or "")
    parts = params.split("-") if params else []
    method = next((p for p in parts if p in METHODS), "both")
    rest = [p for p in parts if p not in METHODS]
    impl = next((p for p in rest if p in ("py", "cpp")), "py+cpp" if test == "py_vs_cpp" else "")
    case = next((p for p in rest if p not in ("py", "cpp")), "C" if test == "py_umap_none_bounds" else "")
    if test == "py_riccati_vs_newton":
        method = "both"
    out = r.get("outcome", "?")
    met = r.get("metrics", {})
    shown = " ".join(fmt(met[k]) for k in KEY.get(test, []) if k in met)
    cells[(test, impl, case, method)] = f"{SYM.get(out, out)} {shown}".strip()
    if out in ("failed", "xfail"):
        notes.append((SYM[out], f"{test}[{params}]", r.get("msg", "")))

rows = sorted({(t, i, c) for (t, i, c, _) in cells}, key=lambda k: (list(KEY).index(k[0]) if k[0] in KEY else 99, k[2], k[1]))
hdr = ("check", "impl", "case", "newton", "riccati", "both")
W = [24, 6, 6, 26, 26, 30]
line = "  ".join(f"{h:<{w}}" for h, w in zip(hdr, W))
print(line); print("-" * len(line))
for t, i, c in rows:
    vals = [cells.get((t, i, c, m), "") for m in ("newton", "riccati", "both")]
    print("  ".join(f"{v:<{w}}"[:w] for v, w in zip((t, i, c, *vals), W)))
print("-" * len(line))
n = {}
for v in R.values():
    n[v.get("outcome")] = n.get(v.get("outcome"), 0) + 1
print(f"passed {n.get('passed', 0)}, failed {n.get('failed', 0)}, "
      f"expected-limitation XFAIL* {n.get('xfail', 0)}, skipped {n.get('skipped', 0)}")
print("Cases: A,B = quad_problem; S sliver arc, G tangential touch, M many switches, I infeasible "
      "(validate/cases/*.txt); C = Umap=None contract case.\n"
      "Revision 3 (general LQ cost, newton only): TINY = quad_problem.TINY_CASE, L1 = Qf case with inactive bounds, "
      "L2 = fixed endpoint with Q != 0.")
if notes:
    print("\nFailures and expected limitations:")
    for s, k, msg in notes:
        print(f"  [{s}] {k}: {msg}")
