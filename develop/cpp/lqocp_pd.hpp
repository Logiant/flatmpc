// Internal helpers of the solver core, defined in lqocp.cpp and shared with lqocp_state.cpp (tier 2)
// and lqocp_basis.cpp. Not part of the API.
#ifndef LQOCP_PD_HPP
#define LQOCP_PD_HPP

#include <float.h>

#include "lqocp.hpp"

// Failure texts in flash where const data is otherwise copied to RAM (Teensy 4 builds set LQOCP_FLASH; its flash is
// memory-mapped, so the pointer reads as usual). One section per text: GCC rejects function-local statics of
// different functions in one named section ("section type conflict").
#ifdef LQOCP_FLASH
#define LQOCP_SN2(k) #k
#define LQOCP_SN(k) LQOCP_SN2(k)
#define LQOCP_STR(s) \
    ({ static const char lqocp_str_[] __attribute__((section(".progmem.lqocp" LQOCP_SN(__COUNTER__)))) = s; &lqocp_str_[0]; })
#else
#define LQOCP_STR(s) (s)
#endif

namespace lqocp {
namespace pd {

// Round-off tolerances are written for double as k * DBL_EPSILON; TS rescales them to
// the scalar type (1 in double, LQOCP_FTOL FLT_EPSILON / 1e-12 in float).
constexpr real EPS = sizeof(real) == 4 ? FLT_EPSILON : DBL_EPSILON;
#ifndef LQOCP_FTOL
#define LQOCP_FTOL 32
#endif
constexpr real TS = sizeof(real) == 4 ? LQOCP_FTOL * FLT_EPSILON / 1e-12 : 1.0;

extern const real GL_X[4], GL_W[4];   // 4-point Gauss-Legendre nodes / weights on [-1, 1]

real fact(int k);
real ipow(real x, int k);
// In-place LU solve of A x = b (A n x n, row stride ld); nrhs columns of B (row stride ldb). false if singular.
template <class T> bool lu_solve(T* A, int n, int ld, T* B, int nrhs, int ldb);
real pval(const real* b, int len, real s);
void shift(const real* a, int len, real t0, real* b);
real pscale(const real* b, int len, real L);
int strip(const real* b, int len, real L, real tol, real* red, int* kout);
int roots_le2(const real* d0, int len, real* out);
real bisect(const real* r, int len, real a, real b);
struct Cross {
    int n;
    real root[4];
    bool simple[4];
};
void crossings(const real* red, int len, int k, real L, real tol, Cross& o);

}  // namespace pd
}  // namespace lqocp

#endif
