# flatmpc

Indirect optimal control for embedded MPC, based on

- L. E. Beaver and A. A. Malikopoulos, "Optimal Control of Differentially Flat Systems is Surprisingly Easy," arXiv:2103.03339.
- L. E. Beaver, "LQ-OCP: Energy-Optimal Control for LQ Problems," arXiv:2310.00168.

The optimal trajectory is a sequence of closed-form motion primitives joined at junctions. The solver
finds the junctions with Newton's method instead of discretizing the problem. The goal is a solver that is
smaller and faster than [TinyMPC](https://github.com/TinyMPC/TinyMPC) on microcontrollers (first target: Teensy 4.1).

Work in progress.

| Folder | Contents |
|---|---|
| `problem/` | Shared problem data (Crazyflie hover model from TinyMPC) and the interface spec |
| `develop/` | Solver: Python reference (`python/lqocp.py`) and embeddable C++ (`cpp/`) |
| `validate/` | Independent validation suite (`run_all.sh`) |
| `compare/` | Benchmark against TinyMPC and OSQP on the same problem; `audit/` checks the formulation |
| `sidebar/` | Search for cases where greedy arc selection fails |
| `teensy/` | PlatformIO projects for the Teensy 4.1 |
