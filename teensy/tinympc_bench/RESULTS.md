# TinyMPC 0.0.8 solve time on Teensy 4.1 (measured on the board)

Problem: `src/quad.h` (copied from ../tinympc_size): quad hover, nx=12, nu=4, N=10, rho=5, |u|<=0.5, |x|<=5,
x0 = [0,1,-2,0.2,0,0,0.1,0,...] in deviation coordinates, tol 1e-3/1e-3, adaptive rho off.
Single-rho cache: after `tiny_setup`, the cache is rebuilt with `tiny_precompute_and_set_cache` from the ORIGINAL Q, R
(rho added once, as Crazyflie firmware / PR #84). Library = the no-iostream 0.0.8 port (dynamic Eigen, heap temporaries).
Teensy 4.1 @ 600 MHz, PlatformIO teensy, gcc -O2, -DNDEBUG -DTINY_NO_IO. Time = DWT cycle counter around `tiny_solve`
only. `pio run -e double|float -t upload`, then read serial; raw logs in `out_double.txt`, `out_float.txt`.
Cold solves repeated 3x, spread < 0.01 %.

| | double | float |
|---|---:|---:|
| 1. cold, max_iter 10000: iterations | 5754 | 5750 |
| 1. cold: time | 964.8 ms | 551.6 ms |
| 1. cold: u0 (projected, `solution->u`) | [0.5, 0.478170, 0.5, 0.5] | [0.5, 0.478172, 0.5, 0.5] |
| 2. cold, max_iter 100: iterations / time | 100 (not solved) / 16.78 ms | 100 / 9.60 ms |
| 2. u0 | [0.5, 0.5, 0.5, 0.5] | same |
| 3. closed loop 100 steps, cap 100: total iter (mean / max), converged | 1143 (11.43 / 100), 94/100 | 1144 (11.44 / 100), 94/100 |
| 3. per-step time mean / median / max | 1854 / 433 / 16812 µs | 1061 / 249 / 9597 µs |
| 3. closed loop, cap 5: total iter (mean), converged | 308 (3.08), 61/100 | 308 (3.08), 61/100 |
| 3. per-step time mean / median / max | 476 / 433 / 844 µs | 273 / 249 / 482 µs |
| 4. time per ADMM iteration (cold, long run) | 167.7 µs | 95.9 µs |

Per-iteration cost in the closed loop is a little lower (154 to 162 µs double, 89 to 93 µs float) because of the
fixed per-solve overhead spread over few iterations; a 1-iteration step costs ~97 µs (double) / ~57 µs (float).

PC cross-check (host driver, x86 double, same problem in deviation coordinates):
cold 5754 iterations (Teensy double: identical), cap 100 closed loop 1137 total, cap 5 closed loop 305 total
(Teensy: 1143 / 308, small drift from different floating-point rounding on ARM). Note: the upstream 882 (closed loop)
and ~5.4k (cold, 5402) figures in `ablation.out` are absolute coordinates and, for 882, WITHOUT the rho fix;
the rho fix in absolute coordinates gives 1167 (closed loop), and deviation coordinates give 881 without the fix.
