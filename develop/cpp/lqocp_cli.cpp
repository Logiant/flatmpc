// Host CLI: lqocp_cli <A|B|TINY|problem_file> <out.csv> [newton|tau|riccati]. May use the STL; the core does not.
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "case_data.h"
#include "lqocp.hpp"
#include "lqocp_state.hpp"

static lqocp::Solver g_solver;   // static storage: no heap, no large stack frame
static lqocp::Result g_result;
static lqocp::GSolver g_gsolver; // general cost
static lqocp::GResult g_gresult;
static lqocp::BasisBuilder g_basis;
static lqocp::StateSolver g_ssolver;   // tier 2: state box
static lqocp::StateResult g_sresult;
static bool g_hasx = false;
static double g_xlo[12], g_xhi[12];   // physical-state box (quad_problem.py order)

// physical state i = PHYS_SCALE[i] * flat state PHYS_FLAT[i] (quad_problem.Tf2p: one nonzero per row)
static const double G2_ = 19.619999999999997, HALF_ = 0.49999999999999994;
static const int PHYS_FLAT[12] = {0, 4, 8, 6, 2, 10, 1, 5, 9, 7, 3, 11};
static const double PHYS_SCALE[12] = {1, 1, 1, -1 / G2_, 1 / G2_, 1, 1, 1, 1, -1 / (G2_ * HALF_), 1 / (G2_ * HALF_), 1 / HALF_};

// Problem file: whitespace-separated numbers, '#' starts a comment line.
//   T, x0 (12), xf (12), has_bounds, [Umap (16, row-major), u_min (4), u_max (4)],
//   optional: has_cost, Q (144), R (16), has_Qf, [Qf (144)]
//   optional (tier 2, needs the cost block): has_xbox, x_lo (12), x_hi (12) on the PHYSICAL state
//   (px py pz phx phy phz vx vy vz wx wy wz; +-inf: none)
static bool read_problem_file(const char* path, lqocp::Problem& p) {
    std::ifstream in(path);
    if (!in) return false;
    std::vector<double> v;
    std::string line;
    while (std::getline(in, line)) {
        const size_t h = line.find('#');
        if (h != std::string::npos) line = line.substr(0, h);
        std::istringstream ls(line);
        std::string tok;
        while (ls >> tok) {
            char* end = nullptr;
            const double x = std::strtod(tok.c_str(), &end);
            if (end == tok.c_str() || *end != '\0') return false;
            v.push_back(x);
        }
    }
    size_t k = 0;
    auto take = [&](double& x) { if (k >= v.size()) return false; x = v[k++]; return true; };
    double flag;
    p.m = 4;
    const int orders[4] = {4, 4, 2, 2};
    for (int j = 0; j < 4; ++j) p.orders[j] = orders[j];
    if (!take(p.T)) return false;
    for (int i = 0; i < 12; ++i) if (!take(p.x0[i])) return false;
    for (int i = 0; i < 12; ++i) if (!take(p.xf[i])) return false;
    if (!take(flag)) return false;
    p.has_bounds = flag != 0.0;
    for (int r = 0; r < 4; ++r) {
        for (int j = 0; j < 4; ++j) p.M[r][j] = (r == j) ? 1.0 : 0.0;
        p.lo[r] = -HUGE_VAL;
        p.hi[r] = HUGE_VAL;
    }
    if (p.has_bounds) {
        for (int r = 0; r < 4; ++r)
            for (int j = 0; j < 4; ++j) if (!take(p.M[r][j])) return false;
        for (int r = 0; r < 4; ++r) if (!take(p.lo[r])) return false;
        for (int r = 0; r < 4; ++r) if (!take(p.hi[r])) return false;
    }
    p.has_cost = 0;
    p.has_Qf = 0;
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) p.R[i][j] = (i == j) ? 1.0 : 0.0;
    if (k == v.size()) return true;                    // energy problem
    if (!take(flag)) return false;
    const bool has_cost = flag != 0.0;
    if (k < v.size()) {
        for (int i = 0; i < 12; ++i) for (int j = 0; j < 12; ++j) if (!take(p.Q[i][j])) return false;
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) if (!take(p.R[i][j])) return false;
        if (!take(flag)) return false;
        p.has_Qf = flag != 0.0;
        if (p.has_Qf)
            for (int i = 0; i < 12; ++i) for (int j = 0; j < 12; ++j) if (!take(p.Qf[i][j])) return false;
        if (k < v.size()) {
            if (!take(flag)) return false;
            g_hasx = flag != 0.0;
            if (g_hasx) {
                for (int i = 0; i < 12; ++i) if (!take(g_xlo[i])) return false;
                for (int i = 0; i < 12; ++i) if (!take(g_xhi[i])) return false;
            }
        }
    }
    if (k != v.size()) return false;
    p.has_cost = has_cost;
    if (!has_cost) p.has_Qf = 0;
    return true;
}

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::fprintf(stderr, "usage: %s <A|B|TINY|problem_file> <out.csv> [newton|tau|riccati]\n", argv[0]);
        return 2;
    }
    std::string method = (argc == 4) ? argv[3] : "newton";
    if (method != "tau" && method != "newton" && method != "riccati") {
        std::fprintf(stderr, "unknown method '%s' (tau|newton|riccati)\n", method.c_str());
        return 2;
    }
    const int n = 12, m = 4;
    lqocp::Problem p;
    std::memset(&p, 0, sizeof(p));
    const LqocpCaseData* cd = nullptr;
    for (int i = 0; i < LQOCP_NUM_CASES; ++i)
        if (std::string(LQOCP_CASES[i].name) == argv[1]) cd = &LQOCP_CASES[i];
    if (cd) {
        p.m = m;
        for (int j = 0; j < m; ++j) p.orders[j] = cd->orders[j];
        p.T = cd->T;
        for (int i = 0; i < n; ++i) { p.x0[i] = cd->x0[i]; p.xf[i] = cd->xf[i]; }
        p.has_bounds = cd->has_bounds;
        for (int r = 0; r < m; ++r) {
            for (int j = 0; j < m; ++j) p.M[r][j] = cd->Umap[r * m + j];
            p.lo[r] = cd->has_bounds ? cd->u_min[r] : -HUGE_VAL;
            p.hi[r] = cd->has_bounds ? cd->u_max[r] : HUGE_VAL;
        }
        p.has_cost = cd->has_cost;
        p.has_Qf = cd->has_Qf;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) { p.Q[i][j] = cd->Q[i * n + j]; p.Qf[i][j] = cd->Qf[i * n + j]; }
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j) p.R[i][j] = cd->R[i * m + j];
    } else if (!read_problem_file(argv[1], p)) {
        std::fprintf(stderr, "'%s' is neither a case name (A|B) nor a valid problem file\n", argv[1]);
        return 2;
    }
    const int meth = (method == "tau") ? lqocp::METHOD_TAU : lqocp::METHOD_NEWTON;

    const auto t0 = std::chrono::steady_clock::now();
    if (!g_solver.setup(p)) {
        const char* why = g_solver.setup_failure();
        std::fprintf(stderr, "problem setup failed: %s\n", why ? why : "unknown");
        std::printf("{\"cost\": null, \"converged\": false, \"failure\": \"setup: %s\"}\n", why ? why : "unknown");
        return 3;
    }
    const bool general = g_solver.is_general();
    // riccati: same as newton for the energy cost (as in Python); the Riccati-sweep general-cost
    // solver exists only in Python (method="riccati")
    if (method == "riccati" && general) {
        std::fprintf(stderr, "method 'riccati' with a general cost is Python-only (use newton)\n");
        return 3;
    }
    if (meth == lqocp::METHOD_TAU && general) {
        std::fprintf(stderr, "method 'tau' supports only the energy cost (Q = 0)\n");
        return 3;
    }
    bool ok;
    if (g_hasx) {   // tier 2: energy cost + thrust box + state box
        double slo[12], shi[12];
        for (int i = 0; i < 12; ++i) { slo[i] = -HUGE_VAL; shi[i] = HUGE_VAL; }
        for (int i = 0; i < 12; ++i) {
            const double a = PHYS_SCALE[i], l = g_xlo[i] / a, h = g_xhi[i] / a;
            slo[PHYS_FLAT[i]] = a > 0 ? l : h;
            shi[PHYS_FLAT[i]] = a > 0 ? h : l;
        }
        if (!g_ssolver.setup(p, slo, shi)) {
            std::printf("{\"cost\": null, \"converged\": false, \"failure\": \"setup: %s\"}\n", g_ssolver.setup_failure());
            return 3;
        }
        ok = g_ssolver.solve(g_sresult);
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        FILE* f = std::fopen(argv[2], "w");
        if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
        std::fprintf(f, "t");
        for (int i = 0; i < n; ++i) std::fprintf(f, ",x%d", i);
        for (int j = 0; j < m; ++j) std::fprintf(f, ",u%d", j);
        std::fprintf(f, "\n");
        double x[lqocp::MAXN], u[lqocp::MAXM];
        for (int k = 0; k < 1001; ++k) {
            const double t = (k == 1000) ? p.T : p.T * k / 1000;
            g_ssolver.eval_x(g_sresult, t, x);
            g_ssolver.eval_u(g_sresult, t, u);
            std::fprintf(f, "%.17g", t);
            for (int i = 0; i < n; ++i) std::fprintf(f, ",%.17g", x[i]);
            for (int j = 0; j < m; ++j) std::fprintf(f, ",%.17g", u[j]);
            std::fprintf(f, "\n");
        }
        std::fclose(f);
        const lqocp::StateResult& r = g_sresult;
        char seq[512];
        g_ssolver.sequence(r.S, r.narcs, r.sa, r.nsa, r.tau, seq, sizeof seq);
        if (ok) std::printf("{\"cost\": %.17g, \"switch_times\": [", r.cost);
        else std::printf("{\"cost\": null, \"switch_times\": [");
        for (int i = 0; i + 1 < r.narcs; ++i) std::printf("%s%.17g", i ? ", " : "", r.tau[i]);
        std::printf("], \"iterations\": %d, \"solve_time_us\": %.3f, \"converged\": %s, \"method\": \"tier2\"",
                    r.iterations, us, ok ? "true" : "false");
        std::printf(", \"junction_iterations\": %d, \"inserted\": %d, \"state_arcs\": [", r.outer, r.inserted);
        for (int q = 0; q < r.nsa; ++q)
            std::printf("%s{\"chain\": %d, \"side\": %d, \"t1\": %.17g, \"t2\": %.17g}", q ? ", " : "",
                        r.sa[q].chain, r.sa[q].side, r.sa[q].t1, r.sa[q].t2);
        std::printf("], \"state_violation\": %.3g, \"sign_violation\": %.3g, \"pmp_violation\": %.3g, \"sequence\": \"%s\"",
                    r.state_violation, r.sign_violation, r.pmp_violation, seq);
        if (!ok) std::printf(", \"failure\": \"%s\"", r.failure ? r.failure : "not converged");
        std::printf("}\n");
        return 0;
    }
    if (general) {
        if (!g_gsolver.setup(p, lqocp::BasisBuilder::fn, &g_basis)) {
            std::fprintf(stderr, "problem setup failed (unsupported dimensions or R not SPD)\n");
            return 3;
        }
        g_basis.attach(&g_gsolver);
        ok = g_gsolver.solve(g_gresult);
    } else {
        ok = g_solver.solve(g_result, meth);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    FILE* f = std::fopen(argv[2], "w");
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", argv[2]);
        return 1;
    }
    std::fprintf(f, "t");
    for (int i = 0; i < n; ++i) std::fprintf(f, ",x%d", i);
    for (int j = 0; j < m; ++j) std::fprintf(f, ",u%d", j);
    std::fprintf(f, "\n");
    const int K = 1001;
    double x[lqocp::MAXN], u[lqocp::MAXM] = {0.0, 0.0, 0.0, 0.0};   // (a failed eval_u leaves u)
    for (int k = 0; k < K; ++k) {
        const double t = (k == K - 1) ? p.T : p.T * k / (K - 1);
        if (general) {
            g_gsolver.eval_x(g_gresult, t, x);
            g_gsolver.eval_u(g_gresult, t, u);
        } else {
            g_solver.eval_x(g_result, t, x);
            g_solver.eval_u(g_result, t, u);
        }
        std::fprintf(f, "%.17g", t);
        for (int i = 0; i < n; ++i) std::fprintf(f, ",%.17g", x[i]);
        for (int j = 0; j < m; ++j) std::fprintf(f, ",%.17g", u[j]);
        std::fprintf(f, "\n");
    }
    std::fclose(f);

    double st[lqocp::MAXJ];
    const int ns = general ? g_gsolver.switch_times(g_gresult, st, lqocp::MAXJ)
                           : g_solver.switch_times(g_result, st, lqocp::MAXJ);
    const double cost = general ? g_gresult.cost : g_result.cost;
    const int its = general ? g_gresult.iterations : g_result.iterations;
    const bool conv = ok && (general ? g_gresult.converged : g_result.converged);
    // a failed solve has no valid cost (and JSON has no inf/nan): print null
    if (conv && std::isfinite(cost)) std::printf("{\"cost\": %.17g, \"switch_times\": [", cost);
    else std::printf("{\"cost\": null, \"switch_times\": [");
    for (int i = 0; i < ns; ++i) std::printf("%s%.17g", i ? ", " : "", st[i]);
    std::printf("], \"iterations\": %d, \"solve_time_us\": %.3f, \"converged\": %s, \"method\": \"%s\"",
                its, us, conv ? "true" : "false", method.c_str());
    if (!conv) {
        const char* why = general ? g_gresult.failure : g_result.failure;
        std::printf(", \"failure\": \"%s\"", why ? why : "not converged");
    }
    std::printf("}\n");
    return 0;
}
