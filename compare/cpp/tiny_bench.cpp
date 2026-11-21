// Host-only C++ timing harness for TinyMPC 0.0.8 (the exact source the pip wheel was built from,
// vendored in compare/third_party/TinyMPC-0.0.8). NOT a Teensy measurement.
//
// Usage: tiny_bench <problem.txt>
// problem.txt (whitespace-separated): nx nu N rho abs_pri_tol abs_dua_tol max_iter reps fix_cache steps
// steps>0: instead of cold solves, run the example's warm-started closed loop for `steps` MPC steps
// (one setup; per step: set_x0, solve, x <- A x + B work->u.col(0), exactly as examples/quadrotor_hovering.cpp).
// fix_cache=1 re-runs tiny_precompute_and_set_cache with the ORIGINAL Q,R after tiny_setup, undoing the
// double rho (tiny_setup passes Q+rho*I to the precompute, which adds rho*I again; tiny_api.cpp:107,126,254).
//   A (nx*nx row-major)  B (nx*nu row-major)  Qdiag (nx)  Rdiag (nu)  x0 (nx)  umin (nu)  umax (nu)
// Prints one JSON line: iterations, solved, per-solve wall times (cold start: fresh tiny_setup before
// every timed solve, setup NOT timed) and the returned input sequence u (nu x N-1, column per step).
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

#include "tinympc/tiny_api.hpp"

using namespace Eigen;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: tiny_bench problem.txt\n"); return 2; }
    std::ifstream f(argv[1]);
    int nx, nu, N, max_iter, reps, fix_cache, steps;
    double rho, pri, dua;
    f >> nx >> nu >> N >> rho >> pri >> dua >> max_iter >> reps >> fix_cache >> steps;
    tinyMatrix A(nx, nx), B(nx, nu);
    for (int i = 0; i < nx; i++) for (int j = 0; j < nx; j++) f >> A(i, j);
    for (int i = 0; i < nx; i++) for (int j = 0; j < nu; j++) f >> B(i, j);
    tinyVector Qd(nx), Rd(nu), x0(nx), umin(nu), umax(nu);
    for (int i = 0; i < nx; i++) f >> Qd(i);
    for (int i = 0; i < nu; i++) f >> Rd(i);
    for (int i = 0; i < nx; i++) f >> x0(i);
    for (int i = 0; i < nu; i++) f >> umin(i);
    for (int i = 0; i < nu; i++) f >> umax(i);
    if (!f) { std::fprintf(stderr, "bad problem file\n"); return 2; }

    tinyMatrix fdyn = tinyMatrix::Zero(nx, 1);
    tinyMatrix xmin = tinyMatrix::Constant(nx, N, -1e17), xmax = tinyMatrix::Constant(nx, N, 1e17);
    tinyMatrix Umin = umin.replicate(1, N - 1), Umax = umax.replicate(1, N - 1);

    if (steps > 0) {
        TinySolver* s = nullptr;
        if (tiny_setup(&s, A, B, fdyn, Qd.asDiagonal(), Rd.asDiagonal(), rho, nx, nu, N, 0)) return 3;
        if (fix_cache) {
            tinyMatrix Qm = Qd.asDiagonal(), Rm = Rd.asDiagonal();
            tiny_precompute_and_set_cache(s->cache, A, B, fdyn, Qm, Rm, nx, nu, rho, 0);
        }
        tiny_set_bound_constraints(s, xmin, xmax, Umin, Umax);
        s->settings->abs_pri_tol = pri; s->settings->abs_dua_tol = dua;
        s->settings->max_iter = max_iter; s->settings->check_termination = 1;
        s->settings->en_state_bound = 1; s->settings->en_input_bound = 1;
        tinyVector x = x0;
        std::vector<int> its, solv; std::vector<double> ts;
        for (int k = 0; k < steps; k++) {
            tiny_set_x0(s, x);
            std::cout.setstate(std::ios_base::badbit);
            auto t0 = std::chrono::steady_clock::now();
            tiny_solve(s);
            auto t1 = std::chrono::steady_clock::now();
            std::cout.clear();
            ts.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            its.push_back(s->solution->iter); solv.push_back(s->solution->solved);
            x = A * x + B * s->work->u.col(0);
        }
        std::vector<double> so = ts; std::sort(so.begin(), so.end());
        std::printf("{\"closed_loop\": 1, \"solve_time_us_median\": %.3f, \"solve_time_us_mean\": %.3f, \"iters\": [", so[so.size()/2],
                    [&]{ double a = 0; for (double v : ts) a += v; return a / ts.size(); }());
        for (size_t k = 0; k < its.size(); k++) std::printf("%s%d", k ? ", " : "", its[k]);
        std::printf("], \"solved\": [");
        for (size_t k = 0; k < solv.size(); k++) std::printf("%s%d", k ? ", " : "", solv[k]);
        std::printf("], \"x_final\": [");
        for (int i = 0; i < nx; i++) std::printf("%s%.17g", i ? ", " : "", x(i));
        std::printf("]}\n");
        return 0;
    }
    std::vector<double> t_us;
    int iters = -1, solved = -1;
    tinyMatrix u_out;
    for (int r = 0; r < reps; r++) {
        TinySolver* s = nullptr;
        if (tiny_setup(&s, A, B, fdyn, Qd.asDiagonal(), Rd.asDiagonal(), rho, nx, nu, N, 0)) return 3;
        if (fix_cache) {
            tinyMatrix Qm = Qd.asDiagonal(), Rm = Rd.asDiagonal();
            tiny_precompute_and_set_cache(s->cache, A, B, fdyn, Qm, Rm, nx, nu, rho, 0);
        }
        tiny_set_bound_constraints(s, xmin, xmax, Umin, Umax);
        // same settings the Python wrapper ends up with (state bound enabled at +-1e17, input bound on)
        s->settings->abs_pri_tol = pri;
        s->settings->abs_dua_tol = dua;
        s->settings->max_iter = max_iter;
        s->settings->check_termination = 1;
        s->settings->en_state_bound = 1;
        s->settings->en_input_bound = 1;
        tiny_set_x0(s, x0);
        // solve() prints "Solver converged..." via std::cout; silence the stream so the timing
        // does not include terminal I/O (a failed stream makes operator<< a cheap no-op).
        std::cout.setstate(std::ios_base::badbit);
        auto t0 = std::chrono::steady_clock::now();
        tiny_solve(s);
        auto t1 = std::chrono::steady_clock::now();
        std::cout.clear();
        t_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        iters = s->solution->iter;
        solved = s->solution->solved;
        u_out = s->solution->u;
        // leak the solver deliberately (TinyMPC 0.0.8 has no tiny_cleanup); reps is small.
    }
    std::vector<double> sorted = t_us;
    std::sort(sorted.begin(), sorted.end());
    double med = sorted[sorted.size() / 2], mn = sorted.front();
    std::printf("{\"iterations\": %d, \"solved\": %d, \"solve_time_us_median\": %.3f, \"solve_time_us_min\": %.3f, \"reps\": %d, \"u\": [",
                iters, solved, med, mn, reps);
    for (int k = 0; k < N - 1; k++) {
        std::printf("%s[", k ? ", " : "");
        for (int i = 0; i < nu; i++) std::printf("%s%.17g", i ? ", " : "", u_out(i, k));
        std::printf("]");
    }
    std::printf("]}\n");
    return 0;
}
