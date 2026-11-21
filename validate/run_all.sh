#!/usr/bin/env bash
# Validator: build references, build the developer's C++ (if present), run the black-box test
# suite against develop/python/lqocp.py and develop/cpp/build/lqocp_cli, print a summary table.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$HERE"
PY="$HERE/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "== creating validate/.venv"
  python3 -m venv --system-site-packages "$HERE/.venv" && "$PY" -m pip install -q pytest pytest-timeout cvxpy clarabel
fi

echo "================ 1. Reference solutions (independent of the developer's method) ================"
"$PY" -W ignore reference.py || { echo "reference build FAILED"; exit 2; }
echo
echo "---- Revision-2 validator cases (validate/cases/*.txt): exact PMP-certified refs + QP cross-check ----"
"$PY" -W ignore cases.py 2>&1 | grep -v -E "^\s*$|prob" || { echo "case build FAILED"; exit 2; }
echo
echo "---- Revision-3 general-cost cases (TINY, L1, L2): Hamiltonian-mp / Riccati / polished QP refs ----"
"$PY" -W ignore lqcases.py 2>&1 | grep -v -E "^\s*$|prob" || { echo "LQ case build FAILED"; exit 2; }
echo "TINY reference written to validate/results/tiny_reference.json"

echo
echo "================ 2. Developer artifacts ================"
PY_OK=no; CPP_OK=no
[ -f "$ROOT/develop/python/lqocp.py" ] && PY_OK=yes
if [ -d "$ROOT/develop/cpp" ]; then
  if make -C "$ROOT/develop/cpp" > "$HERE/results_make.log" 2>&1; then
    echo "make -C develop/cpp: OK"
  else
    echo "make -C develop/cpp: FAILED (see validate/results_make.log)"; tail -20 "$HERE/results_make.log"
  fi
  [ -x "$ROOT/develop/cpp/build/lqocp_cli" ] && CPP_OK=yes
fi
echo "python solve(): $PY_OK    C++ lqocp_cli: $CPP_OK"
"$PY" -W ignore -c "import harness; print('Revision 2 (method=newton): python', harness.py_supports_method(), ' C++', harness.cpp_supports_method()); print('Revision 3 (Q/R/Qf, TINY): python', harness.py_supports_rev3(), ' C++', harness.cpp_supports_rev3())"

echo
echo "================ 3. Test suite ================"
mkdir -p "$HERE/results"
export VALIDATE_TMP="$HERE/results"
"$PY" -m pytest -q -p no:cacheprovider -W ignore --timeout=600 test_lqocp.py -rfs 2>&1 | tee "$HERE/results/pytest.log" | tail -40
echo
echo "================ 4. Summary ================"
"$PY" summary.py | tee "$HERE/results/summary.txt"
