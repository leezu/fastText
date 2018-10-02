#pragma once

#include <atomic>

struct overAlignedInt64 {
  alignas(128) std::atomic_uint64_t e;
};
