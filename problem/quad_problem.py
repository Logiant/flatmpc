"""Shared problem data: Crazyflie quadrotor linearized about hover (TinyMPC).

Source: TinyMPC repo, examples/problem_data/quadrotor_20hz_params.hpp (row-major Adyn, Bdyn),
dt = 0.05 s (ZOH). Continuous-time (Ac, Bc) are recovered exactly via the matrix logarithm of
the augmented ZOH matrix: expm([[Ac, Bc],[0,0]]*dt) = [[Ad, Bd],[0, I]].

State x (12): [x, y, z, phi_x, phi_y, phi_z, vx, vy, vz, wx, wy, wz]  (phi = Rodrigues params)
Input u (4): normalized motor thrust deviation from hover (u_abs = u + u_hover).
"""
import numpy as np
from scipy.linalg import logm, expm

DT = 0.05
NX, NU = 12, 4

Ad = np.array([
 [1,0,0,0,0.0245250,0,0.0500000,0,0,0,0.0002044,0],
 [0,1,0,-0.0245250,0,0,0,0.0500000,0,-0.0002044,0,0],
 [0,0,1,0,0,0,0,0,0.0500000,0,0,0],
 [0,0,0,1,0,0,0,0,0,0.0250000,0,0],
 [0,0,0,0,1,0,0,0,0,0,0.0250000,0],
 [0,0,0,0,0,1,0,0,0,0,0,0.0250000],
 [0,0,0,0,0.9810000,0,1,0,0,0,0.0122625,0],
 [0,0,0,-0.9810000,0,0,0,1,0,-0.0122625,0,0],
 [0,0,0,0,0,0,0,0,1,0,0,0],
 [0,0,0,0,0,0,0,0,0,1,0,0],
 [0,0,0,0,0,0,0,0,0,0,1,0],
 [0,0,0,0,0,0,0,0,0,0,0,1]], dtype=float)

Bd = np.array([
 [-0.0007069,0.0007773,0.0007091,-0.0007795],
 [0.0007034,0.0007747,-0.0007042,-0.0007739],
 [0.0052554,0.0052554,0.0052554,0.0052554],
 [-0.1720966,-0.1895213,0.1722891,0.1893288],
 [-0.1729419,0.1901740,0.1734809,-0.1907131],
 [0.0123423,-0.0045148,-0.0174024,0.0095748],
 [-0.0565520,0.0621869,0.0567283,-0.0623632],
 [0.0562756,0.0619735,-0.0563386,-0.0619105],
 [0.2102143,0.2102143,0.2102143,0.2102143],
 [-13.7677303,-15.1617018,13.7831318,15.1463003],
 [-13.8353509,15.2139209,13.8784751,-15.2570451],
 [0.9873856,-0.3611820,-1.3921880,0.7659845]], dtype=float)


def continuous_model():
    M = np.zeros((NX + NU, NX + NU))
    M[:NX, :NX] = Ad
    M[:NX, NX:] = Bd
    M[NX:, NX:] = np.eye(NU)
    L = logm(M) / DT
    assert np.max(np.abs(np.imag(L))) < 1e-9
    L = np.real(L)
    return L[:NX, :NX], L[:NX, NX:]


Ac, Bc = continuous_model()


# ---- Flat (integrator-chain) model: "inverse dynamic extension" ----
# The hover linearization is four decoupled integrator chains. Flat outputs: x, y, z (positions)
# and psi = phi_z (yaw Rodrigues parameter). We treat the HIGHEST-ORDER derivative of each flat
# output as the input v, instead of the motor thrusts:
#   v = [d4x/dt4, d4y/dt4, d2z/dt2, d2psi/dt2]
# Flat state (12), chain by chain (D = d/dt):
#   s = [x, Dx, D2x, D3x,  y, Dy, D2y, D3y,  z, Dz,  psi, Dpsi]
# Physical relations (from Ac): D2x = 19.62*phi_y, D3x = 9.81*wy, D4x = 9.81*D(wy)
#                               D2y = -19.62*phi_x, D3y = -9.81*wx, D4y = -9.81*D(wx)
#                               D2z = D(vz),  Dpsi = 0.5*wz,  D2psi = 0.5*D(wz)
# Rows of Bc for (D(wy), D(wx), D(vz), D(wz)) are Bc[[10, 9, 8, 11]], ordered to match (x, y, z, psi),
# so v = S @ Bc[[10, 9, 8, 11]] @ u.
# Motor thrust deviations are recovered as u = Umap @ v, Umap = inv(S @ Bc[[10, 9, 8, 11]]).

CHAIN_ORDERS = (4, 4, 2, 2)


def flat_model():
    n = sum(CHAIN_ORDERS)
    Af = np.zeros((n, n)); Bf = np.zeros((n, len(CHAIN_ORDERS)))
    i = 0
    for j, k in enumerate(CHAIN_ORDERS):
        for r in range(k - 1):
            Af[i + r, i + r + 1] = 1.0
        Bf[i + k - 1, j] = 1.0
        i += k
    return Af, Bf


_G2 = Ac[6, 4]     # 19.62
_HALF = Ac[3, 9]   # 0.5


def phys_to_flat(x):
    """Map a physical (deviation) state (12) to the flat state (12)."""
    px, py, pz, phx, phy, phz, vx, vy, vz, wx, wy, wz = x
    return np.array([px, vx, _G2 * phy, _G2 * _HALF * wy,
                     py, vy, -_G2 * phx, -_G2 * _HALF * wx,
                     pz, vz,
                     phz, _HALF * wz])


def flat_to_phys(s):
    x, xd, xdd, xddd, y, yd, ydd, yddd, z, zd, psi, psid = s
    return np.array([x, y, z, -ydd / _G2, xdd / _G2, psi,
                     xd, yd, zd, -yddd / (_G2 * _HALF), xddd / (_G2 * _HALF), psid / _HALF])


_S = np.diag([_G2 * _HALF, -_G2 * _HALF, 1.0, _HALF])
Umap = np.linalg.inv(_S @ Bc[[10, 9, 8, 11], :])   # motor thrust deviation u = Umap @ v
Af, Bf = flat_model()


# ---- Test cases (deviation coordinates: target hover is the origin) ----
# Start: TinyMPC quadrotor_hovering.cpp x0 = (0,1,0,0.2,0,0,0.1,0,...) with Xref hover at z=2,
# so the deviation from the target is x0 - xref.
X0 = np.array([0, 1, -2, 0.2, 0, 0, 0.1, 0, 0, 0, 0, 0], dtype=float)
XF = np.zeros(NX)

# Cases are posed in FLAT coordinates: state s = phys_to_flat(x), input v (highest derivatives),
# cost J = 1/2 * integral |v|^2 dt. Bounds, when present, are on the MOTOR thrusts u = Umap @ v.
S0 = phys_to_flat(X0)
SF = phys_to_flat(XF)

CASES = {
    # Unconstrained: the unconstrained optimum has max|u| ~ 0.043, far inside any thrust limit.
    "A": dict(x0=S0, xf=SF, T=2.0, Umap=None, u_min=None, u_max=None),
    # Thrust-constrained: unconstrained optimum has max|Umap v| ~ 0.85 > 0.5, so bounds activate.
    # (|u| <= 0.3 at T=1 is infeasible.)
    "B": dict(x0=S0, xf=SF, T=1.0, Umap=Umap, u_min=-0.5 * np.ones(NU), u_max=0.5 * np.ones(NU)),
}


# ---- TinyMPC comparison case (spec Revisions 3 and 5) ----
# TinyMPC quadrotor_hovering example (20 Hz params): Q, R (physical coordinates, motor inputs),
# input bounds |u| <= 0.5, x0 as above, hover target at the origin of the deviation coordinates,
# horizon N = 10 knots at dt = 0.05 -> 9 inputs, T = 0.45 s.
# Continuous-time cost used by BOTH solvers (scored the same way for both):
#   J = 1/2 * integral_0^T ( x^T Q x + u^T R u ) dt + 1/2 * x(T)^T Qf x(T)
# with Qf = QF_TINY_PHYS (TinyMPC's own terminal weight). QF_PHYS (CARE) is for reference only.
# The physical-to-flat map is linear, so the same cost is expressed exactly in flat coordinates:
#   x = Tf2p s,  u = Umap v  ->  Q_flat = Tf2p^T Q Tf2p, R_flat = Umap^T R Umap, Qf_flat = Tf2p^T Qf Tf2p.
from scipy.linalg import solve_continuous_are

Q_PHYS = np.diag([100, 100, 100, 4, 4, 400, 4, 4, 4, 2.0408163, 2.0408163, 4.0])
R_PHYS = 4.0 * np.eye(NU)
Tf2p = np.array([flat_to_phys(e) for e in np.eye(NX)]).T      # x = Tf2p @ s
Tp2f = np.array([phys_to_flat(e) for e in np.eye(NX)]).T      # s = Tp2f @ x
# ARE on the exact flat model (the physical Ac carries ~1e-6 rounding residuals); mapped back to physical.
_Qf_flat = solve_continuous_are(Af, Bf, Tf2p.T @ Q_PHYS @ Tf2p, Umap.T @ R_PHYS @ Umap)
QF_PHYS = Tp2f.T @ _Qf_flat @ Tp2f

TINY_DT = 0.05
TINY_N = 10          # TinyMPC knot points: 10 states, 9 inputs -> horizon (N-1)*dt = 0.45 s
TINY_RHO = 5.0       # rho in TinyMPC's quadrotor_hovering example

# TinyMPC's terminal cost is its cached infinite-horizon
# Pinf = DARE(Ad, Bd, Q + rho I, R + rho I), with effective terminal weight Pinf - rho I, in per-step units.
# In our continuous cost units (TinyMPC's sum ~ (1/dt) * integral), the equivalent terminal weight is:
from scipy.linalg import solve_discrete_are
_Pinf = solve_discrete_are(Ad, Bd, Q_PHYS + TINY_RHO * np.eye(NX), R_PHYS + TINY_RHO * np.eye(NU))
QF_TINY_PHYS = TINY_DT * (_Pinf - TINY_RHO * np.eye(NX))
_Qf_tiny_flat = Tf2p.T @ QF_TINY_PHYS @ Tf2p

TINY_CASE = dict(
    x0=S0, xf=SF, T=(TINY_N - 1) * TINY_DT,
    Q=Tf2p.T @ Q_PHYS @ Tf2p,
    R=Umap.T @ R_PHYS @ Umap,
    Qf=_Qf_tiny_flat,
    Umap=Umap, u_min=-0.5 * np.ones(NU), u_max=0.5 * np.ones(NU),
)
