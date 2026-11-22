# TinyMPC 0.0.8 memory on Teensy 4.1 (build-only, not run on the board)

Problem: quadrotor hover, nx=12, nu=4, N=10, rho=5, Q/R from problem/quad_problem.py, |u|<=0.5, |x|<=5, max_iter=100.
Setup: `tiny_setup()` + `tiny_set_bound_constraints()` (0.0.8 uses only dynamic `Eigen::Matrix<.., Dynamic, Dynamic>`,
so a Crazyflie-style static precomputed cache is not possible without rewriting the library). Eigen = the copy bundled with
TinyMPC 0.0.8. Toolchain: PlatformIO teensy 6.0.0, arm-none-eabi-g++ 15.2, default -O2, plus -DNDEBUG.

Envs (`pio run`): `baseline` (same Serial/micros boilerplate, no solver), `tiny_double`, `tiny_float` (TinyMPC as shipped,
which pulls in `<iostream>` for `std::cout` diagnostics), `*_noio` (same, with `std::cout` swapped for a no-op sink via
`lib/tinympc/src/tiny_io.h`, as an embedded port would do). The float build only changes `typedef tinytype`.

## Increment over baseline (teensy_size)

| build | FLASH code | FLASH data | RAM1 code (ITCM) | RAM1 variables (DTCM data+bss) | RAM1 lost incl. 32 KB ITCM padding |
|---|---:|---:|---:|---:|---:|
| double, as shipped | +354,332 | +52,224 | +354,308 | +57,792 | +385,472 |
| float, as shipped  | +353,756 | +51,200 | +353,732 | +56,768 | +384,448 |
| double, no iostream | +82,792 | +6,144 | +82,784 | +6,816 | +72,352 |
| float, no iostream  | +81,896 | +5,120 | +81,888 | +5,792 | +71,328 |

Baseline: FLASH code 37,452 / data 6,088; RAM1 vars 6,944, code 34,904; RAM2 (OCRAM, USB buffers) 12,416 in all builds.
Teensy 4 runs code from ITCM, so code also costs RAM1. The no-iostream DTCM increment includes ~1.7 KB (double) of
problem constants (A, B, Q, x0 are placed in DTCM by default) and ~2.5 KB of newlib malloc/locale state the baseline never pulled in.

## Structs (sizeof on ARM, from main.cpp.o) and heap

| | double | float |
|---|---:|---:|
| TinySolver / Solution / Settings / Cache / Workspace | 16 / 32 / 80 / 144 / 576 = 848 B | 16 / 32 / 56 / 140 / 556 = 800 B |
| Eigen heap live after setup (46 blocks; requested incl. Eigen's +16 B align pad) | 26,720 B (~27,088 B newlib chunks) | 13,728 B (~14,096 B) |
| Total live heap after setup (Eigen + the 5 `new`ed structs) | ~27.9 KB | ~14.9 KB |
| Peak heap during tiny_setup (Eigen requested + structs) | ~39.9 KB | ~20.9 KB |
| Extra transient heap during tiny_solve | +1,376 B | +720 B |
| malloc calls per tiny_solve (100 iters) | 1,802 | 1,802 |

Heap was tallied on the host (`host/heap_count.cpp`, malloc/new interposed) running the same `quad_setup()`/`tiny_solve()`,
built with `EIGEN_MALLOC_ALREADY_ALIGNED=0`, `EIGEN_MAX_ALIGN_BYTES=16`, which matches what the ARM build reports
(align 16, malloc not already aligned, so Eigen requests n+16). Eigen buffer sizes depend only on scalar size, so these
match ARM; the newlib chunk column assumes 4 B header + 8 B rounding. Most of the workspace heap is buffers for the cone and
linear constraints, which this problem does not use. The solve allocates about 18 temporaries per ADMM iteration.

## Stack (static worst case from -fcallgraph-info=su, `stack_depth.py`)

| | double | float |
|---|---:|---:|
| tiny_solve, adaptive rho off (path taken here) | 1,112 B | 960 B |
| tiny_solve, any branch (incl. adaptive-rho code) | 1,120 B | 1,000 B |
| tiny_setup (Riccati precompute) | ~3.8 KB | ~3.6 KB |

Approximate: prebuilt newlib functions (malloc, called on every Eigen temporary) count as 0 B, so add roughly 50 to 100 B.
