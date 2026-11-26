"""Size table of the size_xmcu builds: flash = text + data, static RAM = data + bss (from `size -A`, the
linker's reserved ._user_heap_stack is not counted), worst stack from -fcallgraph-info (../tinympc_size/stack_depth.py).
Usage: python3 sizes.py (after `pio run`)."""
import pathlib, re, subprocess, sys
here = pathlib.Path(__file__).parent
size = "/home/intsyslab/.platformio/packages/toolchain-gccarmnoneeabi-15.2-sym/bin/arm-none-eabi-size"
roots = {"ours": "_ZN5lqocp6Solver10solve_warmERNS_6ResultEid", "tiny": "tiny_solve"}
print("| env | text | data | bss | flash | static RAM | stack main | stack solve |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
for b in sorted((here / ".pio/build").glob("*/firmware.elf")):
    env = b.parent.name
    sec = dict(re.findall(r"^(\.\S+)\s+(\d+)", subprocess.run([size, "-A", str(b)], capture_output=True, text=True).stdout, re.M))
    g = lambda *n: sum(int(sec.get(s, 0)) for s in n)
    text = g(".isr_vector", ".text", ".rodata", ".ARM.extab", ".ARM", ".preinit_array", ".init_array", ".fini_array")
    data, bss = g(".data"), g(".bss")
    st = lambda r: subprocess.run([sys.executable, str(here / "../tinympc_size/stack_depth.py"), str(b.parent), r],
                                  capture_output=True, text=True).stdout.split(" ")[0]
    solve = next((st(r) for k, r in roots.items() if k in env), "")
    print(f"| {env} | {text} | {data} | {bss} | {text + data} | {data + bss} | {st('main')} | {solve} |")
