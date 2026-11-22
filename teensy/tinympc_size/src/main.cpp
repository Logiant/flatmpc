// Same Serial/timing boilerplate for baseline and TinyMPC builds; WITH_TINY adds the solver.
#ifdef WITH_TINY
#include "quad.h"  // before Arduino.h: its F() macro breaks Eigen
#endif
#include <Arduino.h>
#ifdef WITH_TINY
static TinySolver *solver;
#endif

void setup() {
    Serial.begin(115200);
    while (!Serial && millis() < 3000) {}
#ifdef WITH_TINY
    solver = quad_setup();
#endif
}

void loop() {
    uint32_t t0 = micros();
    float u0 = 0;
#ifdef WITH_TINY
    quad_set_x0(solver);
    tiny_solve(solver);
    u0 = solver->solution->u(0, 0);
#endif
    uint32_t t1 = micros();
    Serial.printf("u0=%f t=%lu us\n", u0, (unsigned long)(t1 - t0));
    delay(1000);
}

#ifdef WITH_TINY
// Struct sizes for this target, read back from main.cpp.o (.rodata.tiny_sizes).
extern "C" const uint32_t tiny_sizes[8] = {
    sizeof(TinySolver), sizeof(TinySolution), sizeof(TinySettings), sizeof(TinyCache),
    sizeof(TinyWorkspace), sizeof(tinytype),
    EIGEN_DEFAULT_ALIGN_BYTES, EIGEN_MALLOC_ALREADY_ALIGNED};
#endif
