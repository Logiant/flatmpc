// Our tier-1 core for the Teensy build (the host Makefile compiles develop/cpp directly).
#if defined(ARDUINO) && defined(BENCH_OURS)
#include "../../../develop/cpp/lqocp.cpp"
#endif
