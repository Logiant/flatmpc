# TinyMPC vs ours: host-only benchmark on TINY_CASE

`./run_compare.sh` creates `.venv` if needed (tinympc 0.0.8, osqp 1.1.3, numpy, scipy), builds
`cpp/build/tiny_bench`, and runs `compare.py` twice:

- `--model phys` writes `results/`. This is the main table, on the Crazyflie model `quad_problem.Ac, Bc`, with T = 0.45 s and Qf = `QF_TINY_PHYS`.
- `--model spec` writes `results/spec/`. This is the model implied by TINY_CASE (the flat model with `Umap` mapped to physical coordinates, and TINY_CASE's Qf). It must agree with phys, and it does: every non-ours row matches to 6e-5.

The problem follows `problem/spec.md` (from a formulation audit of the comparison):
- TinyMPC's example, N = 10 knots (9 inputs, T = 0.45 s), per-step Q_PHYS and R_PHYS, rho = 5, and |u| <= 0.5.
- Its continuous version with TinyMPC's own terminal weight, `QF_TINY_PHYS = 0.05·(DARE(Ad, Bd, Q+5I, R+5I) − 5I)`.
- With these choices, TinyMPC-pub's QP is exactly the rectangle-rule/ZOH discretization of TINY_CASE. The terminal weights agree to 5e-13 relative.

Each run writes `summary.txt` (the table), `summary.json` (all metrics, terminal states and notes)
and `traj_<row>.csv`. The CSV columns are `t, x0..x11` (physical state) and `u0..u3` (motor input). Discrete rows hold `u_k` on
`[t_k, t_{k+1})`, and their last row has `u = nan`. Environment knobs: `COMPARE_REPS` (Python timing reps, default 30) and
`COMPARE_REPS_CPP` (default 200). `results/run.log` holds the console output.

**Everything here runs on the x86 host (AMD Threadripper PRO 9955WX). None of these times are Teensy numbers.**

## 0. Model consistency

`quad_problem.Umap` pairs the (x, y, z, psi) scalings with the Bc rows (wy, wx, vz, wz), and `Tp2f Bc Umap = Bf` to 2.3e-6
(rounding in the 7-digit source data). The two scoring models agree:
- B differs by 4.6e-6 and A by 5e-7.
- Every non-ours row's J agrees between `results/` and `results/spec/` to 6e-5 or better.

`compare.py` records these residuals in `notes`. It refuses to score ours on `phys` if they ever exceed 1e-4 again.

## 1. TinyMPC API and conventions (tinympc 0.0.8)

The wheel's compiled extension is built from the TinyMPC sources inside the sdist. They are vendored, unmodified,
in `third_party/TinyMPC-0.0.8/` (citations below are to those files). The copy the wheel ships at
`site-packages/tinympc/codegen/codegen_src/` is an **older** TinyMPC: it has no `fdyn` and no cones, and it is not the code that runs.

| Question | Finding |
|---|---|
| Factor ½ | The x-update is an LQR Riccati recursion (`admm.cpp:13-32`, `tiny_api.cpp:266-290`) with ADMM linear terms `-rho(z-y)` and `-rho(v-g)` (`admm.cpp:217-239`). These are consistent with the objective `sum ½(x'Qx + u'Ru) + ½ x_N' P x_N` plus the penalty `(rho/2)‖·‖²`. A global factor does not change the argmin. |
| Terminal cost | The user cannot set it. The terminal value is the cached `Pinf` (`admm.cpp:238-239`), which is the infinite-horizon Riccati solution of the **rho-augmented** Q and R (`tiny_api.cpp:254-286`, starting from `P = rho·I` and stopping when `max|ΔK| < 1e-5`). The forward pass uses the constant `Kinf` (`admm.cpp:25-32`). This is exact only because `Pinf` is a Riccati fixed point, so a user Qf cannot be put into this algorithm. The effective terminal weight is `Pinf − rho·I`. |
| **Double rho (as shipped)** | `tiny_setup` stores `work->Q = (Q + rho I).diagonal()` (`tiny_api.cpp:107-108`) and passes that to `tiny_precompute_and_set_cache` (`:126`), which adds `rho I` again (`:254-255`). The ADMM linear terms compensate only one rho. As shipped, the solver therefore converges to the QP with **stage `(Q+rho I, R+rho I)` and terminal `P(Q+2rho I, R+2rho I) − rho I`**. I verified this numerically: unconstrained, it matches TinyMPC to 2.6e-6. This is a library bug: the paper's eq. 15 and the authors' firmware cache use a single rho. The rows are labelled **TinyMPC-pub**, meaning TinyMPC as published with a single rho. They use the wrapper's `compute_cache_terms()` (`interface.py:345-386`); `tiny_bench` instead re-calls the precompute with the original Q and R. The rows labelled **TinyMPC-lib2rho** are the 0.0.8 library as shipped. |
| Diagonal Q, R | Required. Only the diagonals are used (`tiny_api.cpp:107-108`, `.diagonal()`). Docstring: `interface.py:91-92`. Cross terms are impossible. |
| Horizon N | N is the number of **knot points**. There are N−1 inputs, and the terminal state is `x_{N-1}` (`tiny_api.cpp:41-42`, `interface.py:98-99`). The 20 Hz example's `NHORIZON 10` (`examples/quadrotor_hovering.cpp:18`) is therefore a 0.45 s horizon with 9 inputs. TINY_CASE uses T = 0.45 s, and TinyMPC runs with N = 10 knots at dt = 0.05. |
| Bounds | Box constraints only, set through `setup(u_min=, u_max=)` or `set_bound_constraints` (`interface.py:163-182`). Setting input bounds also enables state bounds at ±1e17 (`interface.py:177-182`), which is harmless. Linear and SOC constraints exist (`admm.cpp:101-173`). |
| rho | `setup(rho=)`. The example uses 5.0 (`problem_data/quadrotor_20hz_params.hpp:5`). Adaptive rho is off by default (`tiny_api.cpp:226`). |
| Tolerances | `abs_pri_tol` and `abs_dua_tol` are **absolute max-norm** primal and dual residuals of the slack consensus (`admm.cpp:253-271`). There is no relative and no objective tolerance. Defaults are 1e-3, 1e-3, `max_iter` 1000 and `check_termination` 1 (`tiny_api_constants.hpp`). The example uses these defaults with `max_iter = 100` (`quadrotor_hovering.cpp:54`). The paper's "1e-3 objective / 1e-4 constraint, 4000 max" belongs to its Teensy random-QP benchmark, so those rows are labelled `teensy-bench` (dua = 1e-3, pri = 1e-4). | |
| max_iter | `setup(max_iter=)` or `update_settings` (`interface.py:36-61`). On hitting it, the solver returns the projected (feasible) iterate with `solved = 0` (`admm.cpp:383-387`). |
| Warm start | Implicit. `solve()` never resets x, u, the slacks or the duals (`admm.cpp:274-312`), so a second `solve()` starts from the last iterate. The example's closed loop relies on this. There is no explicit warm-start API. A cold start needs a fresh `setup`. |
| Output / stats | `solve()` returns states and controls (`interface.py:202-211`). The returned u is the projected `znew`, so it is always within bounds. `_solver.solution.iter` and `.solved` exist. No timing is exposed. `solve()` prints "Solver converged…" through std::cout (`admm.cpp:371`). |
| Precision | `tinytype = double` (`types.hpp:15`). Matrices are dynamic-size Eigen. |

## 2. How the problem is posed

- **Model.** `phys` (main): `qp.Ac, qp.Bc`. `spec`: `A = Tf2p Af Tp2f`, `B = Tf2p Bf Umap^-1`. Both use T = 0.45 and QF_TINY_PHYS.
  Our solver works on the exact flat model, which differs from Ac/Bc by about 2e-6.
- **Main table.** dt = 0.05, the example's only dt, which gives TinyMPC N = 10 knots and 9 ZOH intervals.
  TinyMPC is given `Q_PHYS`, `R_PHYS` and rho = 5 per step, unchanged. There is no Q·dt scaling: it would change ρ/R and the terminal weight.
  In J units, TinyMPC's discrete sum times dt is exactly the rectangle rule. It cannot take the exact ZOH cost discretization, because that has full matrices and x-u cross terms.
- **TinyMPC rows** (both variants):
  - `example`: tolerance 1e-3/1e-3, a cold solve run to convergence.
  - `teensy-bench`: pri 1e-4, dua 1e-3, max_iter 4000.
  - `tight`: 1e-9.
  - `closed-loop (example)`: the example's own loop, 100 steps, warm-started, cap 100, x ← Ad x + Bd u0 (`quadrotor_hovering.cpp:73-93`, replicated in `tiny_bench`).
    Its `iter` column is the mean number of iterations per step. `*` marks steps that hit the cap: the first 5 or 6 do.
    Its J column is the step-0 plan (capped at 100 iterations and not converged), which is what the example actually commands first.
- **OSQP** (eps 1e-10, polishing on):
  - `tinyQP-pub` and `tinyQP-lib` are exactly the QPs TinyMPC converges to, which separates ADMM error from formulation.
  - `exactZOH` uses the Van Loan stage cost with QF_TINY_PHYS. It is the best piecewise-constant input on that grid.
- **Extrapolations.** The dt = 0.01 and 0.005 rows are listed separately. For OSQP `exactZOH` they are legitimate finer discretizations of TINY_CASE.
  For TinyMPC they are not TinyMPC's problem, because its terminal weight dt·(Pinf−ρ) moves with dt: its relative distance from QF_TINY is 0.12 at dt = 0.01 and 0.13 at dt = 0.005 for pub.
- **Reference.** `ours-python` when it converged; otherwise `OSQP dt=0.0005 exactZOH`, which is an independent upper bound on the continuous optimum.
  J_ref = 122.33275, confirming the audit's 122.333. The fine-grid OSQP solve agrees to 1.2e-5.

## 3. Scoring (identical for every row)

Discrete u is applied as ZOH to the continuous scoring model. J is computed exactly per interval with
the Van Loan matrix exponential, plus ½x(T)'Qf x(T).

Ours (continuous u) is scored by Simpson's rule on 20001 points from `sol.x` and `sol.u`. Independent
re-simulation of `sol.u` (DOP853, rtol 1e-12) reproduces x(T) to 2.4e-10, and the rescored J matches `sol.cost` to 2e-8.

The columns are as follows:
- `viol` is max(|u| − 0.5, 0).
- `|x(T)|` is the 2-norm of the physical terminal state. The full vector is in the JSON.
- `iter*` means max_iter was reached.

Timing:
- `t_py` is the per-solve Python wall time: median over cold solves, setup excluded, stdout redirected to /dev/null.
- `t_cpp` depends on the solver:
  - TinyMPC: `cpp/tiny_bench`, g++ -O3 on the vendored source, median of cold `tiny_solve` with setup excluded and cout silenced.
  - OSQP: its internal `solve_time`. The factorization in setup is excluded; `t_c_run_us` in the JSON includes it.
  - Ours: `lqocp_cli`'s `solve_time_us`.

## 4. Why the comparison is not perfectly fair (read before quoting numbers)

1. **Status of ours in the stored results.** `results/` is the snapshot of one run made before the C++ solver had TINY_CASE's current data:
   ours-python converged in 452 Newton iterations (solve_time 13.8 s, 57 s wall time under load), J = 122.332754; the ours-cpp row is PENDING
   there. Rerun `./run_compare.sh` for a current ours-cpp row. The Teensy tables (`teensy/`) use the current solver.
2. **Discrete vs continuous.** TinyMPC and OSQP optimize ZOH inputs. The best ZOH input loses 0.202 (0.165 %) at dt = 0.05 (`OSQP dt=0.05 exactZOH`). This is intrinsic to the parametrization.
3. **Terminal cost.** This is matched. QF_TINY_PHYS equals TinyMPC-pub's terminal weight at dt = 0.05 to 5e-13 relative, with x0 cost-to-go 118.5.
   - For lib2rho the terminal weight is 0.49 away in relative Frobenius norm, and its x0 cost-to-go is 142.2.
   - For the CARE-based `QF_PHYS` (kept for reference only), the distance is 1.40 and the cost-to-go 81.9.
   - TinyMPC-pub's remaining gap at tight tolerance, 0.272 (0.22 %), is the rectangle rule plus ZOH: OSQP on the same QP gives an identical J. The best ZOH input's 0.202 is part of it.
4. **Double ρ (library bug).** TinyMPC-lib2rho effectively uses R + ρ = 9 instead of 4, and a different terminal weight. That adds about 0.78 at dt = 0.05.
5. **Rectangle rule.** TinyMPC needs diagonal per-step weights, so its stage cost is the left-endpoint rule. The exact ZOH cost would need cross terms.
6. **ADMM convergence.** The tolerances are absolute max-norm residuals, with ρ = 5 from the example.
   - Cold solves from X0 to the example tolerance take about 5.8k iterations.
   - At `teensy-bench` tolerances the solver hits 4000 without converging.
   - At tight tolerances it takes 10k iterations (pub) or 90k (lib2rho).
   - This matches the audit: the example never solves this saturated first problem to convergence. It caps at 100 and warm-starts.
     The closed loop averages 11.4 iterations per step (pub, 1143 in total) or 8.8 (lib2rho, 883 in total, matching the unmodified upstream example's 882).
     Its step-0 plan, capped at 100 iterations, is 12.5 % above the optimum.
     The paper's "≤7 iterations" figure belongs to a different setting (500 Hz tracking near hover).
7. **Timing.**
   - Python times include pybind11 overhead.
   - TinyMPC's host C++ uses dynamic-size Eigen, which allocates on the heap inside `solve` (`admm.cpp:291-292`); the embedded build uses fixed sizes.
   - OSQP times exclude its factorization.
   - Ours-python is pure Python and is not comparable to compiled code.
   - An "iteration" is one Newton step for ours and one ADMM sweep for TinyMPC.
   - None of this indicates Teensy 4.1 performance.
8. **Inputs.** The example's ±0.5 box is symmetric. The physical box would be [−0.583, 0.417] around hover. We copy the example.
