#!/bin/sh
# x86 run of the Teensy bench in the tiny_double configuration (NA 16, NCK 1, NB 4).
# Usage: run.sh [out-binary] [extra g++ flags...]
set -e
cd "$(dirname "$0")"
out=${1:-./bench_host}; [ $# -gt 0 ] && shift
g++ -O2 -std=c++14 -fno-exceptions -I. -I../../../develop/cpp -DNDEBUG -DLQOCP_NSD=4 -DBENCH_TINY \
    -DLQOCP_NA=16 -DLQOCP_NCK=1 -DLQOCP_NB=4 "$@" -o "$out" host_main.cpp ../../../develop/cpp/lqocp.cpp
"$out"
