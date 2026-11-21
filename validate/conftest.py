import json
import os

import numpy as np
import pytest

import harness
import reference as ref

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
_METRICS = {}
_OUTCOMES = {}


@pytest.fixture(scope="session")
def refs():
    if not os.path.exists(ref.CACHE):
        ref.build(verbose=False)
    d = ref.load()
    cA = harness.qp.CASES["A"]
    d["A_poly"] = ref.PolyRef(cA["x0"], cA["xf"], cA["T"])
    import cases
    if not os.path.exists(cases.REF_JSON):
        cases.build_refs(verbose=False)
    d["new"] = cases.load_refs()
    import lqcases
    if not os.path.exists(lqcases.NPZ):
        lqcases.build(verbose=False)
    d["lq_np"], d["lq"] = lqcases.load()
    return d


_SOL_CACHE = {}
_SUPPORT = {}


def method_supported(impl):
    if impl not in _SUPPORT:
        _SUPPORT[impl] = harness.py_supports_method() if impl == "py" else harness.cpp_supports_method()
    return _SUPPORT[impl]


def rev3_supported(impl):
    k = impl + "_rev3"
    if k not in _SUPPORT:
        _SUPPORT[k] = harness.py_supports_rev3() if impl == "py" else harness.cpp_supports_rev3()
    return _SUPPORT[k]


def get_solution(impl, case, method=None):
    """method=None: call without the method argument (spec default 'tau')."""
    key = (impl, case, method)
    if key not in _SOL_CACHE:
        try:
            avail = harness.py_available() if impl == "py" else harness.cpp_available()
            if not avail:
                _SOL_CACHE[key] = ("skip", f"{'develop/python/lqocp.py' if impl == 'py' else 'develop/cpp/build/lqocp_cli'} not found")
            elif method is not None and not method_supported(impl):
                _SOL_CACHE[key] = ("skip", f"{impl}: Revision-2 'method' argument not available yet")
            elif case in harness.LQ_CASES and not rev3_supported(impl):
                _SOL_CACHE[key] = ("skip", f"{impl}: Revision-3 general cost / TINY not available yet")
            elif impl == "py":
                _SOL_CACHE[key] = ("ok", harness.run_python(case, method))
            else:
                _SOL_CACHE[key] = ("ok", harness.run_cpp(case, method))
        except Exception as e:  # the run itself failed: every test for it fails
            _SOL_CACHE[key] = ("err", f"{type(e).__name__}: {e}")
    st, val = _SOL_CACHE[key]
    if st == "skip":
        pytest.skip(val)
    if st == "err":
        pytest.fail(f"running {impl} {method} case {case} failed: {val}")
    return val


@pytest.fixture
def record(request):
    """record(name, value): store a metric for the summary table."""
    def _rec(name, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = str(value)
        _METRICS.setdefault(request.node.nodeid, {})[name] = value
    return _rec


def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        msg = ""
        if report.outcome == "failed":
            try:
                msg = report.longrepr.reprcrash.message.splitlines()[0][:300]
            except Exception:
                msg = str(report.longrepr).strip().splitlines()[-1][:300] if report.longrepr else ""
        elif report.outcome == "skipped":
            lr = report.longrepr
            msg = lr[2] if isinstance(lr, tuple) else str(lr)
        outcome = report.outcome
        if hasattr(report, "wasxfail"):
            outcome, msg = "xfail", report.wasxfail
        _OUTCOMES[report.nodeid] = dict(outcome=outcome, msg=msg)


def pytest_sessionfinish(session, exitstatus):
    os.makedirs(RESULTS, exist_ok=True)
    out = {nid: dict(**_OUTCOMES.get(nid, {}), metrics=_METRICS.get(nid, {}))
           for nid in sorted(set(_OUTCOMES) | set(_METRICS))}
    with open(os.path.join(RESULTS, "results.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
