// Continuation driver: follow a solved problem at s = 0 to the target at s = 1.
// Step control only; the client owns the unknowns, the residual, its exact Jacobian and the events.
// No heap, no virtuals: P is a plain struct whose calls inline. P provides, for its last accepted point:
//   void back();                        working point := last accepted point
//   void predict(double s, double s1);  move the working point along the exact (IFT) path tangent to s1
//   bool correct(double s1);            Newton corrector at s1 with its events (structure births and deaths)
//   void accept();                      last accepted point := working point (now at s1)
//   bool spent() const;                 iteration budget exhausted: stop now
//   void restart();                     the direct step 0 -> 1 failed (e.g. rebase the s = 0 problem)
// The first step is 0 -> 1 (the direct solve). A failed step is halved (stop below hmin), an accepted
// one doubles the next. Returns 1 when the target is solved, else the last accepted s (< 1).
#ifndef LQOCP_CONT_HPP
#define LQOCP_CONT_HPP

namespace lqocp {

template <class P> double continuation(P& p, double hmin = 1.0 / 256) {
    double s = 0.0, ds = 1.0;
    for (;;) {
        const double s1 = (s + ds >= 1.0) ? 1.0 : s + ds;
        p.back();
        p.predict(s, s1);
        if (p.correct(s1)) {
            if (s1 >= 1.0) return 1.0;
            p.accept();
            s = s1;
            ds = (2.0 * ds < 1.0 - s) ? 2.0 * ds : 1.0 - s;
        } else if (p.spent() || (ds *= 0.5) < hmin) {
            return s;
        } else if (s == 0.0 && ds == 0.5) {
            p.restart();
        }
    }
}

}  // namespace lqocp

#endif
