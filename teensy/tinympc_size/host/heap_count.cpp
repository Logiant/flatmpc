// Host-side heap tally for the same quad_setup()/tiny_solve() as the Teensy sketch (can't run on the board now).
// Built with EIGEN_MALLOC_ALREADY_ALIGNED=0 so Eigen requests malloc(n + 16) exactly as on Teensy (see RESULTS.md).
#include <cstdio>
#include <cstdlib>
#include <new>
#include <map>
#include "../src/quad.h"
extern "C" void *__libc_malloc(size_t);
extern "C" void __libc_free(void *);
extern "C" void *__libc_realloc(void *, size_t);

static bool on = false;
static std::map<void *, size_t> *live;   // malloc'd blocks (Eigen) -> requested size
static long n_alloc, cur, peak, n_new, new_bytes;
static size_t chunk(size_t n) { size_t c = (n + 4 + 7) & ~size_t(7); return c < 16 ? 16 : c; }  // newlib dlmalloc
static long live_chunks() { long s = 0; for (auto &kv : *live) s += chunk(kv.second); return s; }

extern "C" void *malloc(size_t n) {
    void *p = __libc_malloc(n);
    if (on) { on = false; (*live)[p] = n; on = true; n_alloc++; cur += n; if (cur > peak) peak = cur; }
    return p;
}
extern "C" void free(void *p) {
    if (on && p) { on = false; auto it = live->find(p); if (it != live->end()) { cur -= it->second; live->erase(it); } on = true; }
    __libc_free(p);
}
extern "C" void *realloc(void *, size_t) { abort(); }  // Eigen doesn't use it here; fail loudly if it does
void *operator new(size_t n) { if (on) { n_new++; new_bytes += n; } return __libc_malloc(n); }  // only TinyMPC's 5 struct news
void operator delete(void *p) noexcept { __libc_free(p); }
void operator delete(void *p, size_t) noexcept { __libc_free(p); }

int main() {
    live = new std::map<void *, size_t>;
    on = true;
    TinySolver *s = quad_setup();
    on = false;
    long setup_peak = peak, setup_live = cur, setup_n = n_alloc, setup_chunks = live_chunks(), setup_blocks = live->size();
    peak = cur; n_alloc = 0;
    quad_set_x0(s);
    on = true;
    tiny_solve(s);
    on = false;
    printf("sizeof(tinytype)=%zu\n", sizeof(tinytype));
    printf("setup: %ld Eigen mallocs, peak %ld B requested, live after setup %ld B in %ld blocks (newlib chunks ~%ld B); %ld struct news (%ld B on host)\n",
           setup_n, setup_peak, setup_live, setup_blocks, setup_chunks, n_new, new_bytes);
    printf("solve: %ld mallocs, transient peak above setup %ld B; iter=%d u0=%g\n", n_alloc, peak - setup_live, s->solution->iter, (double)s->solution->u(0, 0));
}
