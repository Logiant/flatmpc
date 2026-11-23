# flatmpc

Indirect optimal control for embedded MPC, based on

- L. E. Beaver and A. A. Malikopoulos, "Optimal Control of Differentially Flat Systems is Surprisingly Easy," arXiv:2103.03339.
- L. E. Beaver, "LQ-OCP: Energy-Optimal Control for LQ Problems," ACC 2024, arXiv:2310.00168.
- L. E. Beaver, R. Tron and C. G. Cassandras, "A Graph-Based Approach to Generate Energy-Optimal Robot Trajectories in Polygonal Environments," arXiv:2211.05251.
- F. N. Tzortzoglou, L. E. Beaver and A. A. Malikopoulos, "Closed-Form Characterization of Constrained Double-Integrator Optimal Control," arXiv:2604.13007.

For a differentially flat system (here a quadrotor, as integrator chains of orders 4, 4, 2, 2), the optimal trajectory is a
sequence of closed-form arcs joined at junctions. The solver finds the junctions with Newton's method and exact
polynomial roots. It does not discretize the problem in time. The goal is a solver that is smaller and faster than
[TinyMPC](https://github.com/TinyMPC/TinyMPC) on microcontrollers. The first target is the Teensy 4.1 (Cortex-M7).

## Solver tiers
| Tier | Problem | C++ class | Status |
|---|---|---|---|
| 1 | Energy cost + input box (Q = 0, terminal cost or fixed endpoint) | `lqocp::Solver` | Done. Dual Newton (cold) plus Newton on the switch times (warm MPC steps); double and float |
| 2 | Tier 1 + first-order state box (rates, z speed) | `lqocp::StateSolver` | Works when it converges; robustness in progress (`HEURISTIC_NOTES.md`) |
| 3 | General LQ cost (Q ≠ 0) | `lqocp::GSolver` | Works; slower than TinyMPC; fails for very small Q (`develop/NOTES.md` §15.3) |

## Tier 1 on a Teensy 4.1 (matched problem, 100-step MPC closed loop)
Both solvers solve the same problem: TinyMPC's hover model with Q = 0, TinyMPC's own terminal weight, |u| ≤ 0.5 and
no state bounds.

| | Cold solve | Median step | Mean step | Peak RAM | Code |
|---|---|---|---|---|---|
| Ours, double | 28 ms | 0.23 ms | 1.0 ms | ~10 kB | ~19 kB |
| TinyMPC 0.0.8, double (max_iter 1000) | 351 ms | 1.95 ms | 37 ms | ~48 kB | ~89 kB |
| Ours, float | 13 ms | 0.14 ms | 0.5 ms | ~5.6 kB | |
| TinyMPC 0.0.8, float | 188 ms | 1.10 ms | 21 ms | | |

Our closed loop converges at all 100 steps. TinyMPC converges at 84 and lets the applied input exceed the box slightly.
Details and the fairness argument are in `develop/NOTES.md` §13.

## Layout
| Folder | Contents |
|---|---|
| `problem/` | Shared problem data (Crazyflie hover model from TinyMPC) and the interface spec |
| `develop/` | Solver: Python reference (`python/`) and embeddable C++ (`cpp/`, no heap, fixed-size arrays); tier-2 prototype tools (`tier2/`); design notes and results (`NOTES.md`) |
| `validate/` | Independent validation suite (`run_all.sh`) |
| `compare/` | Host benchmark against TinyMPC and OSQP on the same problem; `audit/` checks the formulation |
| `sidebar/`, `HEURISTIC_NOTES.md` | Arc-sequence heuristics: where greedy arc selection fails, activation-order observations |
| `teensy/` | PlatformIO projects: `lqocp_bench` (ours; envs `energy`, `energy_float`, `tier2`, `tiny_*`), `tinympc_bench` (TinyMPC; matched envs `double_q0_nox`, `float_q0_nox`), `mpc_bench` (ideal-model closed loop and TinyMPC knot sweep), `size_xmcu` (size-only builds for Cortex-M0+/M4F), `tinympc_size` |

## How to run
You need a C++14 compiler and `make`. The Teensy steps also need [PlatformIO](https://platformio.org/install/cli) and a Teensy 4.1 on USB.

**1. Solve a problem on your PC**
```sh
make -C develop/cpp
develop/cpp/build/lqocp_cli TINY out.csv          # built-in cases: A, B, TINY
develop/cpp/build/lqocp_cli my_problem.txt out.csv  # your own problem
```
The CLI prints one JSON line: cost, switch times, iterations, solve time, converged. `out.csv` holds the trajectory
(t, state, input). A problem file is plain whitespace-separated numbers:
- `T`, `x0` (12), `xf` (12), then `has_bounds` with the thrust box;
- optionally the cost block (Q, R, Qf);
- optionally a physical state box, which selects tier 2.

The exact layout is at the top of `develop/cpp/lqocp_cli.cpp` and in `problem/spec.md`. Example files are in
`validate/cases/`.

**2. Run the benchmark on a Teensy 4.1**
```sh
cd teensy/lqocp_bench
pio run -e energy -t upload      # tier 1 (also: energy_float, tier2, tiny_double)
pio device monitor -b 115200     # prints cold-solve and 100-step MPC loop timings
```
For the matching TinyMPC numbers, do the same in `teensy/tinympc_bench` with env `double_q0_nox` or `float_q0_nox`.

**3. Validation and host comparison (optional)**
- `validate/run_all.sh`: independent validation suite (Python).
- `compare/run_compare.sh`: host benchmark against TinyMPC and OSQP. It creates its own `.venv`; see `compare/README.md`.

## Citation
Please cite this work if you use it. GitHub's "Cite this repository" button uses `CITATION.cff`.

## License
This repository is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (see `LICENSE`).
Vendored third-party code, such as TinyMPC and Eigen under `compare/third_party/` and the Teensy bench `lib/` folders,
keeps its own license.
