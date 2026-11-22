"""Worst-case stack of a call chain from GCC -fcallgraph-info=su (.ci) files. Usage: stack_depth.py BUILD_DIR ROOT [SKIP_SUBSTR...]
Functions without .ci info (prebuilt libc/libstdc++, e.g. malloc) count as 0 bytes; recursion is cut."""
import re, sys, pathlib, functools
frame, calls = {}, {}
for ci in pathlib.Path(sys.argv[1]).rglob("*.ci"):
    for m in re.finditer(r'node: \{ title: "([^"]+)" label: "((?:[^"\\]|\\.)*)"', ci.read_text()):
        b = re.search(r"(\d+) bytes \((?:static|dynamic)", m.group(2))
        if b: frame[m.group(1)] = int(b.group(1))
    for m in re.finditer(r'edge: \{ sourcename: "([^"]+)" targetname: "([^"]+)"', ci.read_text()):
        calls.setdefault(m.group(1), set()).add(m.group(2))
def depth(f, seen=()):
    if f in seen: return 0, [f + " (recursion)"]
    best = max((depth(c, seen + (f,)) for c in calls.get(f, ()) if not any(s in c for s in sys.argv[3:])), default=(0, []))
    return frame.get(f, 0) + best[0], [f"{f} ({frame.get(f, '?')})"] + best[1]
total, path = depth(sys.argv[2])
print(total, "bytes:", " -> ".join(path))
