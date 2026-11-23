# Size on other cores: energy path vs TinyMPC 0.0.8 (build-only, not run)

Each program does setup + one cold solve + one warm closed-loop step and writes the result to a
volatile sink (`src/main.cpp`). Ours: TINY with Q = 0 and TinyMPC's Qf (`TQ0`, NOTES.md 13.3),
`LQOCP_NO_TAU`. TinyMPC: matched problem (Q = 0, input bounds only, N = 10, rho fix), as
`teensy/tinympc_bench` env `double_q0_nox`. Toolchain Arm GNU 15.2 (as on the Teensy), PlatformIO
ststm32 + STM32Cube (startup and CMSIS only, no HAL calls), newlib-nano.

- flash = text + data; static RAM = data + bss (`sizes.py`; the linker script's reserved heap/stack
  block is not counted). Base = empty main.
- stack: static worst case from `-fcallgraph-info=su` (`../tinympc_size/stack_depth.py`). "main"
  covers setup + both solves; "solve" is `Solver::solve_warm` / `tiny_solve`. Prebuilt libc and
  libgcc functions count as 0 B.
- TinyMPC heap is not measured here: it is data-size driven (Eigen dynamic matrices) and taken from
  `../tinympc_size/RESULTS.md` (host tally, same dimensions): double ~27.9 KB live after setup,
  ~39.9 KB peak during setup, +1.4 KB transient per solve; float ~14.9 KB, ~20.9 KB, +0.7 KB.
  Ours uses no heap.
- g4 is built hard-float (`-mfpu=fpv4-sp-d16 -mfloat-abi=hard`, `hardfloat.py`; PlatformIO's
  default for this board is soft-float). Ours float still links soft-double helpers: the Problem
  and the public API (x0, mean_u) are double, converted in setup/set_x0/mean_u; the Newton loop is float.

| core | program | flash -Os | flash -O2 | static RAM | stack main / solve (-Os) | heap |
|---|---|---:|---:|---:|---:|---|
| M0+ (nucleo_g071rb) | base | 548 | 552 | 40 | 0 | 0 |
| M0+ | ours double | 30,604 | 35,212 | 8,808 | 2,392 / 2,224 | 0 |
| M0+ | ours float | 28,308 | 32,740 | 4,832 | 1,528 / 1,360 | 0 |
| M0+ | TinyMPC double | 74,892 | 105,464 | 672 | 4,856 / 1,560 | 27.9 KB live, 39.9 KB peak* |
| M0+ | TinyMPC float | 66,348 | 96,832 | 672 | 4,240 / 1,432 | 14.9 KB live, 20.9 KB peak* |
| M4F (nucleo_g431kb) | base | 840 | 840 | 40 | 0 | 0 |
| M4F | ours double | 24,168 | 28,520 | 8,808 | 2,404 / 2,244 | 0 |
| M4F | ours float | 19,976 | 22,940 | 4,832 | 1,444 / 1,284 | 0 |
| M4F | TinyMPC double | 66,732 | 95,092 | 672 | 4,800 / 1,496 | 27.9 KB live, 39.9 KB peak* |
| M4F | TinyMPC float | 55,524 | 84,036 | 672 | 4,144 / 1,304 | 14.9 KB live, 20.9 KB peak* |

\* from `../tinympc_size/RESULTS.md`, not measured on these cores.

Ours static RAM is `Solver` + `Result` (double 7,808 + 568 B, float 4,068 + 332 B) plus about 430 B
of newlib state. -O2 stacks are within ±10% of -Os (`python3 sizes.py` prints all).
Rebuild: `pio run -d teensy/size_xmcu && python3 teensy/size_xmcu/sizes.py`. The toolchain is the
Teensy package symlinked as `~/.platformio/packages/toolchain-gccarmnoneeabi-15.2-sym` (a
package.json renamed to `toolchain-gccarmnoneeabi`); without it, drop `platform_packages` to use
ststm32's gcc 7.2, which has no `-fcallgraph-info`.
