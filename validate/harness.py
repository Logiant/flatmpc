"""Black-box access to the developer's solver, through the interface contract in problem/spec.md only.

Every implementation is wrapped into a common object with:
  impl, case, T, u(t) -> (K,4), x(t) -> (K,12) [Python: any t; C++: only at the CSV grid],
  t_grid (the 1001-point uniform grid), U_grid, X_grid, cost, switch_times, info, raw (impl-specific)
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DEV_PY = os.environ.get("LQOCP_PY_DIR", os.path.join(ROOT, "develop", "python"))
DEV_PY_FILE = os.path.join(DEV_PY, "lqocp.py")
DEV_CPP = os.path.join(ROOT, "develop", "cpp")
CLI = os.environ.get("LQOCP_CLI", os.path.join(DEV_CPP, "build", "lqocp_cli"))
sys.path.insert(0, os.path.join(ROOT, "problem"))
import quad_problem as qp  # noqa: E402

NGRID = 1001


def py_available():
    return os.path.isfile(DEV_PY_FILE)


def cpp_available():
    return os.path.isfile(CLI) and os.access(CLI, os.X_OK)


class DevSolution:
    pass


def _import_lqocp():
    if DEV_PY not in sys.path:
        sys.path.insert(0, DEV_PY)
    import lqocp  # noqa
    return lqocp


LQ_CASES = ("TINY", "L1", "L2")


def problem(case):
    """Problem dict for a case name: A/B/TINY from quad_problem, others from validate/cases/*.txt."""
    if case in qp.CASES:
        return dict(qp.CASES[case])
    if case == "TINY":
        return dict(qp.TINY_CASE)
    import cases
    return cases.load_case(case)


def cli_case_arg(case):
    if case in qp.CASES or case == "TINY":
        return case
    import cases
    return cases.case_path(case)


def py_supports_method():
    if not py_available():
        return False
    import inspect
    lq = _import_lqocp()
    return "method" in inspect.signature(lq.solve).parameters


def run_python(case, method=None, **override):
    lq = _import_lqocp()
    c = problem(case)
    c.update(override)
    kw = {} if method is None else dict(method=method)
    for k in ("Q", "R", "Qf"):
        if c.get(k) is not None:
            kw[k] = np.array(c[k])
    t0 = time.perf_counter()
    sol = lq.solve(qp.Af.copy(), qp.Bf.copy(), np.array(c["x0"]), np.array(c["xf"]), c["T"],
                   Umap=None if c["Umap"] is None else np.array(c["Umap"]),
                   u_min=None if c["u_min"] is None else np.array(c["u_min"]),
                   u_max=None if c["u_max"] is None else np.array(c["u_max"]), **kw)
    wall = time.perf_counter() - t0
    d = DevSolution()
    d.impl, d.case, d.method, d.T, d.raw, d.wall = "py", case, method, c["T"], sol, wall
    d.t_grid = np.linspace(0, c["T"], NGRID)
    d.u = lambda t: np.atleast_2d(np.asarray(sol.u(np.atleast_1d(np.asarray(t, float))), float))
    d.x = lambda t: np.atleast_2d(np.asarray(sol.x(np.atleast_1d(np.asarray(t, float))), float))
    d.U_grid = d.u(d.t_grid)
    d.X_grid = d.x(d.t_grid)
    d.cost = sol.cost
    d.switch_times = sol.switch_times
    d.info = sol.info
    d.continuous = True
    return d


_PY_SUB = r"""
import sys, json, time, numpy as np
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); sys.path.insert(0, sys.argv[3])
import lqocp, harness
c = harness.problem(sys.argv[4]); m = sys.argv[5]
t0 = time.perf_counter()
try:
    s = lqocp.solve(harness.qp.Af, harness.qp.Bf, c["x0"], c["xf"], c["T"], c["Umap"], c["u_min"], c["u_max"], method=m)
    out = dict(ok=True, cost=float(s.cost) if s.cost is not None else None, switch_times=[float(v) for v in s.switch_times],
               converged=s.info.get("converged"), method=s.info.get("method"), iterations=s.info.get("iterations"))
except Exception as e:
    out = dict(ok=False, error=f"{type(e).__name__}: {e}")
out["wall"] = time.perf_counter() - t0
print("RESULT " + json.dumps(out, default=str))
"""


def run_python_subprocess(case, method, timeout=120):
    """Run solve() in a child process (protects against hangs/crashes); returns a dict."""
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, "-W", "ignore", "-c", _PY_SUB, DEV_PY, HERE, os.path.join(ROOT, "problem"),
                            case, method], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return dict(ok=False, timeout=True, error=f"timeout after {timeout}s", wall=time.perf_counter() - t0)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    if not lines:
        return dict(ok=False, error=f"no result, returncode {p.returncode}: {p.stderr[-300:]}", returncode=p.returncode,
                    wall=time.perf_counter() - t0)
    d = json.loads(lines[-1][7:])
    d["returncode"] = p.returncode
    return d


def run_cpp(case, method=None, timeout=120):
    tmpd = tempfile.mkdtemp(prefix=f"lqocp_{case}_{method}_", dir=os.environ.get("VALIDATE_TMP"))
    out = os.path.join(tmpd, f"{case}.csv")
    args = [CLI, cli_case_arg(case), out] + ([] if method is None else [method])
    t0 = time.perf_counter()
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    wall = time.perf_counter() - t0
    d = DevSolution()
    d.impl, d.case, d.method, d.T, d.wall = "cpp", case, method, problem(case)["T"], wall
    d.returncode, d.stdout, d.stderr, d.csv_path = p.returncode, p.stdout, p.stderr, out
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    d.json_lines = lines
    d.json = json.loads(lines[-1]) if lines else None
    d.cost = d.json.get("cost") if d.json else None
    d.switch_times = d.json.get("switch_times") if d.json else None
    d.info = d.json
    d.continuous = False
    if os.path.exists(out):
        with open(out) as f:
            d.header = f.readline().strip()
        d.data = np.loadtxt(out, delimiter=",", skiprows=1, ndmin=2)
        d.t_grid = d.data[:, 0]
        d.X_grid = d.data[:, 1:13]
        d.U_grid = d.data[:, 13:17]
    else:
        d.header, d.data = None, None
    return d


def py_supports_rev3():
    if not py_available():
        return False
    import inspect
    return "Q" in inspect.signature(_import_lqocp().solve).parameters


def cpp_supports_rev3():
    if not cpp_available():
        return False
    try:
        d = run_cpp("TINY", None, timeout=120)
        return d.returncode == 0 and isinstance(d.json, dict) and d.data is not None
    except Exception:
        return False


def cpp_supports_method():
    if not cpp_available():
        return False
    try:
        d = run_cpp("A", "newton", timeout=60)
        return d.returncode == 0 and isinstance(d.json, dict) and d.json.get("method") == "newton"
    except Exception:
        return False


def build_cpp():
    """make -C develop/cpp (returns (ok, output))."""
    if not os.path.isdir(DEV_CPP):
        return False, "develop/cpp missing"
    p = subprocess.run(["make", "-C", DEV_CPP], capture_output=True, text=True, timeout=600)
    return p.returncode == 0, p.stdout + p.stderr
