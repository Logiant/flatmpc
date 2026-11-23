"""Tables (markdown), CSVs and a plot from host/build/all.csv and host/build/heap.txt.
Usage: summarise.py <outdir> <provenance line>   (system python3 -I: numpy + matplotlib, no pandas)"""
import re
import sys
import numpy as np
from types import SimpleNamespace as NS
from statistics import median
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out, prov = sys.argv[1], sys.argv[2]
H = "teensy/mpc_bench/host/build/"
rows = [l.strip().split(",") for l in open(H + "all.csv") if l[0] in "cso"]
cols = "tag solver tier x0 N cap clamp conv J_u J_cl x_end xv uv t_cold t_med t_mean t_max its its_max".split()
num = lambda x: float(x) if re.search(r"[.eEn]", x) else int(x)
mk = lambda names, r: NS(**{k: (x if k in ("tag", "solver") else num(x)) for k, x in zip(names, r)})
cl = [mk(cols, r) for r in rows if r[0] in ("cl", "sw")]
os_ = [mk("tag N x0 solver its conv t J xv uv".split(), r) for r in rows if r[0] == "os"]
F = lambda L, **kw: [r for r in L if all(getattr(r, k) == v for k, v in kw.items())]
XB = 5.0
v = lambda x: f"{max(x, 0):.3g}" if np.isfinite(x) else "inf"     # violation (<= 0 printed as 0)
g = lambda x, n=6: f"{x:.{n}g}" if abs(x) < 1e6 else f"{x:.2g}"


def name(r):
    return f"ours cap {r.cap}" if r.solver == "ours" else f"TinyMPC max_iter {r.cap}, {'clamped' if r.clamp else 'raw'} u0"


L = [f"# Ideal-model closed-loop MPC: ours vs TinyMPC 0.0.8 (x86)\n", prov, "",
     "**Setup.** Plant = exact continuous model (flat integrator chains), no noise, each input held exactly (ZOH) over the "
     "0.05 s control period, 100 steps (5 s). Problem: Q = 0, R = 4I, |u| ≤ 0.5, T = 0.45 s, N = 10 knots, "
     "Qf = dt·(DARE(Ad, Bd, ρI, R+ρI) − ρI), ρ = 5 (single-ρ cache, as `teensy/tinympc_bench`). Tier 2 adds |x_phys| ≤ 5 on all 12 states "
     "(TinyMPC: at the knots; ours: continuous). TinyMPC Ad/Bd = exact ZOH of the same model (differs from the published "
     "20 Hz matrices by ≤ 5e-8). Ours applies `mean_u` over the period; TinyMPC applies `work->u.col(0)` raw (upstream) or clamped to the box; "
     "tolerances 1e-3/1e-3 (TinyMPC defaults), warm started as upstream (no shift).",
     "",
     "**Measured.** `xviol` = max over the whole loop of |x_phys,i(t)| − 5, in continuous time: on each held piece every state is a "
     "polynomial (degree ≤ 4) whose max |·| is taken at the ends and at the real roots of its derivative (no grid); ≤ 0 is printed as 0. "
     "`uviol` = max |u_i| − 0.5 over the applied inputs. J_u = ∫½uᵀRu dt over 5 s; J_cl = J_u + ½ x(5)ᵀQf x(5). |x_end| = flat-state norm. "
     "Times: x86 (-O3), µs; cold = step 0, median/mean/max over warm steps 1–99. conv = steps whose solve reported converged.", ""]


def table(df, tier):
    t = ["| solver | conv | J_u | J_cl | \\|x_end\\| | xviol | uviol | cold µs | median µs | mean µs | max µs | its (total) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in F(df, tier=tier):
        t.append(f"| {name(r)} | {r.conv}/100 | {g(r.J_u)} | {g(r.J_cl)} | {r.x_end:.2g} | {v(r.xv) if tier == 2 else '(' + v(r.xv) + ')'} | {v(r.uv)} | "
                 f"{r.t_cold:.0f} | {r.t_med:.1f} | {r.t_mean:.1f} | {r.t_max:.0f} | {r.its} |")
    return t


c0 = F(cl, tag="cl", x0=0)
L += ["## TINY's x0", "", "### Tier 1 (thrust box only; xviol in brackets = how far |x| ≤ 5 would be exceeded, not a constraint here)", ""]
L += table(c0, 1) + ["", "### Tier 2 (|x_phys| ≤ 5)", ""] + table(c0, 2)

cr = [r for r in F(cl, tag="cl") if r.x0 > 0]
L += ["", "## 10 random x0 (seed 1: position ±1.5 m, roll/pitch Rodrigues ±0.2, yaw ±0.3, velocity ±0.5 m/s, rates ±1 rad/s)", "",
      "Aggregates over the 10 loops: converged steps (of 1000), loops with all 100 steps converged, worst xviol / uviol, median of J_cl, "
      "median of the per-loop median step time, mean of the per-loop mean, worst max.", ""]
for tier in (1, 2):
    L += [f"### Tier {tier}", "", "| solver | conv steps | loops 100/100 | worst xviol | worst uviol | median J_cl | median \\|x_end\\| | median µs | mean µs | max µs |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for key in dict.fromkeys((r.solver, r.cap, r.clamp) for r in F(cr, tier=tier)):
        d = F(cr, tier=tier, solver=key[0], cap=key[1], clamp=key[2])
        a = lambda f: [getattr(r, f) for r in d]
        xv = v(max(a("xv")))
        L.append(f"| {name(d[0])} | {sum(a('conv'))}/1000 | {a('conv').count(100)}/10 | {xv if tier == 2 else '(' + xv + ')'} | {v(max(a('uv')))} | "
                 f"{g(median(a('J_cl')))} | {median(a('x_end')):.2g} | {median(a('t_med')):.1f} | {np.mean(a('t_mean')):.1f} | {max(a('t_max')):.0f} |")
    L.append("")
d2 = F(cr, tier=2)
L += ["### Tier 2 per x0: ours cap 100 vs TinyMPC max_iter 1000 (clamped)", "",
      "| x0 | ours conv | ours J_cl | ours xviol | ours uviol | Tiny conv | Tiny J_cl | Tiny xviol |", "|---|---|---|---|---|---|---|---|"]
for x in range(1, 11):
    o = F(d2, x0=x, solver="ours", cap=100)[0]
    t = F(d2, x0=x, solver="tiny", cap=1000, clamp=1)[0]
    L.append(f"| {x} | {o.conv} | {g(o.J_cl)} | {v(o.xv)} | {v(o.uv)} | {t.conv} | {g(t.J_cl)} | {v(t.xv)} |")
open(f"{out}/mpc_closed_loop.md", "w").write("\n".join(L) + "\n")

# ---------------- knot sweep ----------------
heap = {}
for l in open(H + "heap.txt"):
    n = int(re.search(r"N=(\d+)", l)[1])
    heap[n] = dict(peak=int(re.search(r"peak (\d+) B requested", l)[1]), chunks=int(re.search(r"chunks ~(\d+)", l)[1]),
                   trans=int(re.search(r"above setup (\d+)", l)[1]))
# ARM run-time RAM = static 6.8 kB + live heap (newlib chunks) + structs 0.85 kB + solve transient + stack 1.1 kB
# (teensy/tinympc_size/RESULTS.md at N = 10); peak = static + setup peak + structs + stack
ram = {n: ((6800 + h["chunks"] + 848 + h["trans"] + 1112) / 1000, (6800 + h["peak"] + 848 + 1112) / 1000) for n, h in heap.items()}

S = []
for n in dict.fromkeys(r.N for r in os_):
    o = {r.x0: r for r in F(os_, N=n, solver="ours")}
    t = {r.x0: r for r in F(os_, N=n, solver="tiny")}
    gap = {x: (t[x].J - o[x].J) / o[x].J * 100 for x in o if o[x].conv == 1}
    s = F(cl, tag="sw", N=n)
    tk, t100 = F(s, solver="tiny", cap=1000, clamp=1), F(s, solver="tiny", cap=100, clamp=1)
    oo = F(s, solver="ours", x0=0)[0]
    a = lambda L, f: [getattr(r, f) for r in L]
    S.append(NS(N=n, dt=0.45 / (n - 1), os_its=t[0].its, os_us=t[0].t, J_tiny=t[0].J, J_ours=o[0].J, gap0=gap.get(0, np.nan),
                xv0=t[0].xv, xv_all=max(r.xv for r in t.values()), gap_max=max(map(abs, gap.values())), n_ours_ok=len(gap), ours_us=o[0].t,
                cl_xv0=F(tk, x0=0)[0].xv, cl_xv_all=max(a(tk, "xv")), cl_J0=F(tk, x0=0)[0].J_cl, ours_cl_J0=oo.J_cl,
                cl_conv_all=sum(a(tk, "conv")), med=median(a(tk, "t_med")), mean=np.mean(a(tk, "t_mean")),
                its_step=sum(a(tk, "its")) / (100 * len(tk)), cl100_xv_all=max(a(t100, "xv")), med100=median(a(t100, "t_med")),
                mean100=np.mean(a(t100, "t_mean")), ram_run=ram[n][0], ram_peak=ram[n][1]))
keys = list(vars(S[0]))
open(f"{out}/knot_sweep.csv", "w").write(",".join(keys) + "\n" + "".join(",".join(f"{getattr(r, k):.10g}" for k in keys) + "\n" for r in S))
p = lambda x: f"{max(x, 0):.3g} ({max(x, 0) / XB * 100:.2g}%)"
K = ["# TinyMPC knot sweep at fixed T = 0.45 s: compute vs accuracy (x86, tier 2)\n", prov, "",
     "**Setup.** As `mpc_closed_loop.md` (tier 2: Q = 0, R = 4I, |u| ≤ 0.5, |x_phys| ≤ 5 at TinyMPC's knots), with N knots, dt = T/(N−1), "
     "Ad/Bd = exact ZOH at that dt, and the terminal weight recomputed with it: Qf(N) = dt·(DARE(Ad, Bd, ρI, R+ρI) − ρI) "
     "(TinyMPC's sum × dt = our continuous cost). Qf(N) changes by up to 3.8% (N = 90 vs 10), so J* (ours, same Qf(N)) changes with N too.",
     "",
     "- **One-shot:** TinyMPC cold from x0 with max_iter 100000 (converged), its returned (projected, in-box) inputs `solution->u` held on "
     "each knot interval and propagated exactly; `xviol` = max |x_phys(t)| − 5 between knots (continuous, polynomial roots); "
     "gap = (J_Tiny − J*)/J*, J_Tiny scored with the same continuous cost and Qf(N); J* = our tier-2 continuous optimum for the same Qf(N).",
     "- **Closed loop:** re-solve every 0.05 s, 100 steps; the plant gets TinyMPC's planned held inputs for every knot interval inside "
     "[0, 0.05 s] (the last one cut at 0.05; N = 10: just u0), clamped to the box, max_iter 1000 (100 in the last columns). "
     "Ours: the same Qf(N), cap 100, mean of its plan over the period.",
     "- **x0:** TINY's x0 (column '0') and the max over TINY's + 10 random x0 ('all'). Ours fails cold on some random x0 in tier 2 "
     "(see `mpc_closed_loop.md`), so 'gap all' is over the x0 where ours converged (count in brackets).",
     "- **RAM:** TinyMPC Eigen heap tallied on the host per N (`teensy/tinympc_size/host/heap_count.cpp`, -DNH_KNOTS=N, ARM-matched "
     "alignment), plus static 6.8 kB, structs 0.85 kB, stack 1.1 kB from `teensy/tinympc_size/RESULTS.md` (N-independent). "
     "Run time = live heap + solve transient; peak = during `tiny_setup`. Not measured on the board.", "",
     "## One-shot", "",
     "| N | dt ms | Tiny its | Tiny µs | J_Tiny | J* (ours) | gap x0 0 | gap max all | xviol x0 0 (% of 5) | xviol all | ours µs (x0 0) |",
     "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in S:
    K.append(f"| {r.N} | {r.dt * 1e3:.2f} | {r.os_its} | {r.os_us:.0f} | {r.J_tiny:.7g} | {r.J_ours:.7g} | {r.gap0:+.3g}% | "
             f"{r.gap_max:.3g}% ({r.n_ours_ok}/11) | {p(r.xv0)} | {p(r.xv_all)} | {r.ours_us:.0f} |")
K += ["", "## Closed loop and compute", "",
      "| N | xviol x0 0 | xviol all | J_cl Tiny (x0 0) | J_cl ours (x0 0) | conv steps all | median µs | mean µs | its/step | "
      "max_iter 100: xviol all | median µs | mean µs | RAM run kB | RAM peak kB |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in S:
    K.append(f"| {r.N} | {p(r.cl_xv0)} | {p(r.cl_xv_all)} | {r.cl_J0:.6g} | {r.ours_cl_J0:.6g} | {r.cl_conv_all}/1100 | {r.med:.1f} | "
             f"{r.mean:.0f} | {r.its_step:.0f} | {p(r.cl100_xv_all)} | {r.med100:.0f} | {r.mean100:.0f} | {r.ram_run:.1f} | {r.ram_peak:.1f} |")
open(f"{out}/knot_sweep.md", "w").write("\n".join(K) + "\n")

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot([r.os_us for r in S], [max(r.xv_all, 1e-6) / XB * 100 for r in S], "-o", color="#2a78d6", lw=2, ms=8, label="TinyMPC one-shot (cold), max over 11 x0")
for r in S:
    ax.annotate(f"N={r.N}", (r.os_us, max(r.xv_all, 1e-6) / XB * 100), textcoords="offset points", xytext=(6, 4), fontsize=8, color="#52514e")
o0 = S[0]
ax.plot([o0.ours_us], [1e-6 / XB * 100], "s", color="#eb6834", ms=9, label="ours (continuous box: 0 violation, plotted at floor)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("x86 cold solve time, µs"); ax.set_ylabel("max state-box violation between knots, % of bound")
ax.grid(True, which="major", color="#e6e5e0", lw=0.8); ax.legend(fontsize=8, frameon=False)
ax.set_title("Compute vs continuous-time violation (tier 2, T = 0.45 s)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{out}/knot_sweep.png", dpi=150)
for r in S: print(r.N, f"{r.xv0:.3g} {r.xv_all:.3g} gap0 {r.gap0:.3g}% cl {r.cl_xv_all:.3g} med {r.med:.0f} ram {r.ram_run:.1f}")
