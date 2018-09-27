#pragma once

#include <assert.h>
#include <atomic>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace fasttext {
// Scalable statistics counter
// https://dl.acm.org/citation.cfm?id=2486159.2486182
class Counter {
private:
  std::atomic<double> threshold{std::numeric_limits<uint32_t>::max()};
  const double a;
  const double probFactor;

public:
  // rstdv is the relative standard deviation, i.e., the ratio of the standard
  // deviation of the projected value and the actual count.
  Counter(double rstdv) : a(1 / (2 * rstdv * rstdv)), probFactor(a / (a + 1)){};

  // the current projected value of the counter
  unsigned int
  load(std::memory_order memory_order = std::memory_order_seq_cst) const {
    double pr =
        threshold.load(memory_order) / std::numeric_limits<uint32_t>::max();
    double val = (1.0 / pr - 1.0) * a;
    return val;
  }

  // Increment the counter by 1. Must provide a uniformly random uint32.
  void inc(uint32_t random, std::memory_order memory_order) {
    while (true) {
      double seenT = threshold.load(memory_order);
      if (random > seenT) {
        return;
      }
      bool overflow = (seenT < (a + 1.0));
      if (overflow) {
        throw std::runtime_error("Overflow in scalable statistics counter");
      }
      double newT = seenT * probFactor;
      if (threshold.compare_exchange_weak(seenT, newT, memory_order)) {
        return;
      }
    }
  }

  uint32_t xorshift32(uint32_t *state) const {
    /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
  }

  friend std::ostream &operator<<(std::ostream &str, Counter const &data) {
    return str << data.load();
  }
};
} // namespace fasttext
