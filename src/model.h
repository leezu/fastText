/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <atomic>
#include <vector>
#include <random>
#include <utility>
#include <memory>

#include "args.h"
#include "counter.h"
#include "matrix.h"
#include "vector.h"
#include "qmatrix.h"
#include "real.h"

namespace fasttext {

struct Node {
  int32_t parent;
  int32_t left;
  int32_t right;
  int64_t count;
  bool binary;
};

class Model {
  protected:
    std::shared_ptr<Matrix> wi_;
    std::shared_ptr<std::vector<std::atomic_int64_t>> wi_counter_;
    std::shared_ptr<std::vector<real>> wi_state_;
    std::shared_ptr<std::vector<real>> wo_state_;
#ifndef SSC
    std::atomic_int64_t *global_counter_;
#else
    Counter *global_counter_;
#endif
    uint32_t xorshift32state;
    int64_t local_counter_;
    std::int32_t nwords_;
    std::shared_ptr<Matrix> wo_;
    std::shared_ptr<QMatrix> qwi_;
    std::shared_ptr<QMatrix> qwo_;
    std::shared_ptr<Args> args_;
    Vector hidden_;
    Vector output_;
    Vector grad_;
    Vector tmp_;
    int32_t hsz_;
    int32_t osz_;
    real loss_;
    int64_t nexamples_;
    std::vector<real> t_sigmoid_;
    std::vector<real> t_log_;
    // used for negative sampling:
    std::vector<int32_t> negatives_;
    size_t negpos;
    // used for hierarchical softmax:
    std::vector< std::vector<int32_t> > paths;
    std::vector< std::vector<bool> > codes;
    std::vector<Node> tree;

    static bool comparePairs(const std::pair<real, int32_t>&,
                             const std::pair<real, int32_t>&);

    int32_t getNegative(int32_t target);
    void initSigmoid();
    void initLog();

    static const int32_t NEGATIVE_TABLE_SIZE = 10000000;

  public:
    Model(std::shared_ptr<Matrix>, std::shared_ptr<Matrix>,
          std::shared_ptr<Args>,
          std::shared_ptr<std::vector<std::atomic_int64_t>>,
          std::shared_ptr<std::vector<real>>,
          std::shared_ptr<std::vector<real>>,
#ifndef SSC
          std::atomic_int64_t*,
#else
          Counter*,
#endif
          int32_t, int32_t);

    real binaryLogistic(int32_t, bool, real);
    real negativeSampling(int32_t, real);
    real hierarchicalSoftmax(int32_t, real);
    real softmax(int32_t, real);

    void predict(const std::vector<int32_t>&, int32_t, real,
                 std::vector<std::pair<real, int32_t>>&,
                 Vector&, Vector&) const;
    void predict(const std::vector<int32_t>&, int32_t, real,
                 std::vector<std::pair<real, int32_t>>&);
    void dfs(int32_t, real, int32_t, real,
             std::vector<std::pair<real, int32_t>>&,
             Vector&) const;
    void findKBest(int32_t, real, std::vector<std::pair<real, int32_t>>&,
                   Vector&, Vector&) const;
    void update(const std::vector<int32_t>&, int32_t, real, const real, const real);
    void proximalUpdate(const int32_t &, const real &, const real &);
    float getNorm(const int32_t &);
    void forceEagerUpdate(const std::vector<int32_t> &, const real &,
                          const real &, const real &, const int64_t &);
    real forceEagerUpdate(const int32_t &, const real &, const real &,
                          const int64_t &);
    void computeHidden(const std::vector<int32_t> &, Vector &) const;
    void computeOutputSoftmax(Vector&, Vector&) const;
    void computeOutputSoftmax();

    void setTargetCounts(const std::vector<int64_t>&);
    void initTableNegatives(const std::vector<int64_t>&);
    void buildTree(const std::vector<int64_t>&);
    real getLoss() const;
    real sigmoid(real) const;
    real log(real) const;
    real std_log(real) const;

    std::minstd_rand rng;
    bool quant_;
    void setQuantizePointer(std::shared_ptr<QMatrix>, std::shared_ptr<QMatrix>, bool);
};

}
