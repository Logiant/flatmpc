#!/usr/bin/env bash
# Host-only TinyMPC vs ours benchmark. Creates compare/.venv on first use.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q tinympc==0.0.8 osqp==1.1.3 numpy scipy
fi
make -s -C cpp            # TinyMPC host C++ timing harness (vendored TinyMPC 0.0.8 source)
.venv/bin/python compare.py --model phys "$@"   # primary -> results/: true Crazyflie Ac/Bc
.venv/bin/python compare.py --model spec "$@"   # -> results/spec/: model implied by TINY_CASE (flat + Umap); must agree with phys
