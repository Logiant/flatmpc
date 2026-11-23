// Quadrotor hover problem (problem/quad_problem.py, TinyMPC 20 Hz params): nx=12 nu=4 N=10 rho=5.
#pragma once
#include "tiny_api.hpp"
static const int NX = 12, NU = 4, NH = 10;
static const tinytype A_rm[NX * NX] = {
 1,0,0,0,0.0245250,0,0.0500000,0,0,0,0.0002044,0,
 0,1,0,-0.0245250,0,0,0,0.0500000,0,-0.0002044,0,0,
 0,0,1,0,0,0,0,0,0.0500000,0,0,0,
 0,0,0,1,0,0,0,0,0,0.0250000,0,0,
 0,0,0,0,1,0,0,0,0,0,0.0250000,0,
 0,0,0,0,0,1,0,0,0,0,0,0.0250000,
 0,0,0,0,0.9810000,0,1,0,0,0,0.0122625,0,
 0,0,0,-0.9810000,0,0,0,1,0,-0.0122625,0,0,
 0,0,0,0,0,0,0,0,1,0,0,0,
 0,0,0,0,0,0,0,0,0,1,0,0,
 0,0,0,0,0,0,0,0,0,0,1,0,
 0,0,0,0,0,0,0,0,0,0,0,1};
static const tinytype B_rm[NX * NU] = {
 -0.0007069,0.0007773,0.0007091,-0.0007795,
 0.0007034,0.0007747,-0.0007042,-0.0007739,
 0.0052554,0.0052554,0.0052554,0.0052554,
 -0.1720966,-0.1895213,0.1722891,0.1893288,
 -0.1729419,0.1901740,0.1734809,-0.1907131,
 0.0123423,-0.0045148,-0.0174024,0.0095748,
 -0.0565520,0.0621869,0.0567283,-0.0623632,
 0.0562756,0.0619735,-0.0563386,-0.0619105,
 0.2102143,0.2102143,0.2102143,0.2102143,
 -13.7677303,-15.1617018,13.7831318,15.1463003,
 -13.8353509,15.2139209,13.8784751,-15.2570451,
 0.9873856,-0.3611820,-1.3921880,0.7659845};
static const tinytype Q_d[NX] = {100,100,100,4,4,400,4,4,4,2.0408163,2.0408163,4.0};
#ifdef Q_ZERO   // "easy" case: stage Q = 0 (TinyMPC's terminal Pinf then comes from Q = 0 too)
static const tinytype Q_stage[NX] = {0};
#else
static const tinytype *const Q_stage = Q_d;
#endif
static const tinytype X0[NX] = {0,1,-2,0.2,0,0,0.1,0,0,0,0,0};

static TinySolver *quad_setup() {
    typedef Matrix<tinytype, Dynamic, Dynamic, RowMajor> RM;
    tinyMatrix A = Map<const RM>(A_rm, NX, NX), B = Map<const RM>(B_rm, NX, NU);
    tinyMatrix f = tinyMatrix::Zero(NX, 1);
    tinyMatrix Q = Map<const tinyVector>(Q_stage, NX).asDiagonal();
    tinyMatrix R = tinytype(4) * tinyMatrix::Identity(NU, NU);
    TinySolver *s = nullptr;
    if (tiny_setup(&s, A, B, f, Q, R, 5.0, NX, NU, NH, 0)) return nullptr;
    tiny_set_bound_constraints(s, tinyMatrix::Constant(NX, NH, tinytype(-5)), tinyMatrix::Constant(NX, NH, tinytype(5)),
                               tinyMatrix::Constant(NU, NH - 1, tinytype(-0.5)), tinyMatrix::Constant(NU, NH - 1, tinytype(0.5)));
    s->settings->max_iter = 100;
    return s;
}
static void quad_set_x0(TinySolver *s) { tiny_set_x0(s, Map<const tinyVector>(X0, NX)); }
