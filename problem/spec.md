# Shared problem spec and interface contract

Both `develop/` and `validate/` work from this file and `problem/quad_problem.py`. Neither may edit them.

## Model
Crazyflie quadrotor linearized about hover (TinyMPC 20 Hz model, `quadrotor_20hz_params.hpp`),
converted exactly to continuous time (`Ac`, `Bc`). It is four decoupled integrator chains, so we
use the flat-output form and treat the HIGHEST-ORDER derivative of each flat output as the input:

    s = [x, Dx, D2x, D3x,  y, Dy, D2y, D3y,  z, Dz,  psi, Dpsi]      (12 states)
    v = [D4x, D4y, D2z, D2psi]                                         (4 inputs)
    s' = Af s + Bf v        (pure integrator chains, orders 4,4,2,2)

Motor thrust deviations are u = Umap @ v. Maps `phys_to_flat`, `flat_to_phys`, `Umap` are in `quad_problem.py`.

## Problem
    minimize   J = 1/2 * integral_0^T  v(t)^T v(t) dt
    subject to s' = Af s + Bf v,  s(0) = x0,  s(T) = xf,
               u_min <= Umap v(t) <= u_max   (elementwise, only when bounds are given)

Cases (`quad_problem.CASES`, all in flat coordinates):
- "A": T = 2 s, no bounds.
- "B": T = 1 s, motor bounds |u_i| <= 0.3, which become active (unconstrained peak is ~0.44).
  Note the bound is a polytope in v (Umap is dense, cond ~1.4e3), and it couples the four chains.

## Python interface: `develop/python/lqocp.py`
    solve(A, B, x0, xf, T, Umap=None, u_min=None, u_max=None) -> Solution
Umap=None means the identity (bounds directly on the input). Solution attributes:
- `u(t)`: the input v at time(s) t. Scalar t -> shape (m,); array of K times -> shape (K, m).
- `x(t)`: the state, same convention, shape (n,) or (K, n).
- `cost`: J as defined above (float).
- `switch_times`: sorted list of times where any bound becomes active or inactive (empty if none).
- `info`: dict, at least {"iterations": int, "solve_time_s": float, "converged": bool}.

## C++ interface: `develop/cpp/`
- `make -C develop/cpp` builds `develop/cpp/build/lqocp_cli` with g++ (host). The solver core must be
  embeddable on a Teensy 4.1 (no heap allocation in the solve, no exceptions, fixed sizes; double OK).
- `develop/cpp/build/lqocp_cli <case> <out.csv>` where <case> is A or B:
  - writes CSV with header `t,x0,...,x11,u0,...,u3` (x = flat state s, u = input v) and 1001 rows
    at uniform t in [0, T] inclusive.
  - prints one JSON line to stdout:
    {"cost": J, "switch_times": [...], "iterations": k, "solve_time_us": s, "converged": true/false}

## Revision 2 (adds a second solver mode and file-based problems)
Python: `solve(..., method="tau")`. `method` is a keyword argument with default "tau":
- "tau": the existing junction-time solver.
- "newton": Newton's method on the 12 polynomial coefficients c (dual / costate shooting) with
  exact polynomial-root event detection and no time grid.
Both methods return the same `Solution` interface. `info` must also contain "method".

C++: `lqocp_cli <case> <out.csv> [method]`, where method is `tau` (default) or `newton`.
`<case>` may be A, B, or a path to a problem file in this plain-text format
(whitespace-separated numbers; `#` starts a comment line):

    T
    x0   (12 numbers)
    xf   (12 numbers)
    has_bounds   (0 or 1)
    Umap (16 numbers, row-major; present only if has_bounds = 1)
    u_min (4 numbers; present only if has_bounds = 1)
    u_max (4 numbers; present only if has_bounds = 1)

The dynamics are always the flat model (Af, Bf) from quad_problem.py.
The JSON line must also contain "method".

## Revision 3 (general LQ cost via the Hamiltonian and Thm 1 / Thm 2; Newton is the default)
Default method is now "newton" in both Python (`method="newton"`) and the CLI (third argument omitted -> newton).
"tau" remains available, energy-cost only.

Generalized problem (flat coordinates, s' = Af s + Bf v):

    minimize  J = 1/2 * integral_0^T ( s^T Q s + v^T R v ) dt  +  1/2 * (s(T)-xf)^T Qf (s(T)-xf)
    subject to s(0) = x0,  u_min <= Umap v <= u_max  (if bounds given),
               and s(T) = xf when Qf is None (fixed endpoint).

Python: `solve(A, B, x0, xf, T, Umap=None, u_min=None, u_max=None, Q=None, R=None, Qf=None, method="newton")`.
Q=None means 0, R=None means I, Qf=None means fixed endpoint. The old energy problem is Q=0, R=I, Qf=None,
and its results must not change. "tau" may raise NotImplementedError when Q or Qf is given or R != I.
`cost` is J as defined above.

New shared case `quad_problem.TINY_CASE`: the TinyMPC 20 Hz hover example's Q, R (mapped exactly to flat
coordinates; R_flat is SPD but ill-conditioned, cond ~ 2e6), Qf = continuous-time ARE solution, |u| <= 0.5,
T = 0.5 s (TinyMPC N = 10, dt = 0.05). [Superseded by Revision 5.]

CLI: `<case>` may also be `TINY`. Problem file format, extended: after the existing fields, an optional block

    has_cost  (0 or 1)
    Q  (144 numbers, row-major)    R (16 numbers)    has_Qf (0 or 1)   Qf (144 numbers, if has_Qf = 1)

If the block is absent, the problem is the energy problem (Q=0, R=I, fixed endpoint). The JSON line is unchanged.

## Revision 4 (bug fix: Umap)
`Umap` paired the scalings for (x, y, z, psi) with the Bc rows (vz, wx, wy, wz) in the wrong order. It is now
`inv(S @ Bc[[10, 9, 8, 11]])`, and `Tp2f @ Bc @ Umap` equals `Bf` to 2e-6 (rounding in the 7-digit source data).
Everything derived from Umap changes: case B's motor bounds, TINY_CASE's R_flat, Qf and bounds.
Case B is now T = 1 s with |u_i| <= 0.5 (|u_i| <= 0.3 is infeasible with the correct Umap).
The math of all earlier results is unaffected: they were correct solutions for the polytope given by the old Umap.

## Revision 5 (TINY_CASE matches TinyMPC exactly; from compare/audit/AUDIT.md)
- Horizon: TinyMPC's N counts knot points (N states, N-1 inputs), so N = 10 at dt = 0.05 is T = 0.45 s.
- Terminal cost: TinyMPC always uses its cached Pinf. The equivalent continuous terminal weight is
  QF_TINY_PHYS = dt * (DARE(Ad, Bd, Q + rho I, R + rho I) - rho I), with rho = 5, mapped to flat coordinates.
  The CARE-based QF_PHYS is kept for reference but is no longer used by TINY_CASE.
- With these, TinyMPC (single-rho, as in its paper and firmware) solves exactly the ZOH discretization of TINY_CASE.
- TinyMPC is given its example's per-step Q_PHYS, R_PHYS and rho = 5 unchanged (no dt scaling).
