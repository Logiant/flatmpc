// Added for the Teensy size study: TINY_NO_IO swaps TinyMPC's std::cout diagnostics for a no-op sink,
// as an embedded port (e.g. Crazyflie firmware) would. Without it this is plain <iostream>.
#pragma once
#ifdef TINY_NO_IO
#include <string>
struct TinyNullOut { template <class T> TinyNullOut &operator<<(const T &) { return *this; } };
static TinyNullOut tiny_null_out;
#define TINY_OUT tiny_null_out
#define TINY_ENDL 0
#else
#include <iostream>
#define TINY_OUT std::cout
#define TINY_ENDL std::endl
#endif
