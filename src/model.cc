/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "model.h"

#include "mkl_cblas.h"
#include <fenv.h>
#include <cfenv>
#include <iostream>
#include <assert.h>
#include <algorithm>
#include <stdexcept>

namespace fasttext {

constexpr int64_t SIGMOID_TABLE_SIZE = 512;
constexpr int64_t MAX_SIGMOID = 8;
constexpr int64_t LOG_TABLE_SIZE = 512;
#ifdef SEQ_CST
constexpr std::memory_order MEMORY_ORDER = std::memory_order_seq_cst;
#else
constexpr std::memory_order MEMORY_ORDER = std::memory_order_relaxed;
#endif

Model::Model(
    std::shared_ptr<Matrix> wi, std::shared_ptr<Matrix> wo,
    std::shared_ptr<Args> args,
    std::shared_ptr<std::vector<std::atomic_int64_t>> wi_counter,
    std::shared_ptr<std::vector<real>> wi_state,
    std::shared_ptr<std::vector<real>> wo_state,
#ifndef SSC
    std::atomic_int64_t *global_counter,
#else
    Counter *global_counter,
#endif
    int32_t nwords, int32_t seed)
    : hidden_(args->dim), output_(wo->size(0)), grad_(args->dim),
      tmp_(args->dim), rng(seed), quant_(false) {
  wi_ = wi;
  wi_counter_ = wi_counter;
  wi_state_ = wi_state;
  wo_state_ = wo_state;
  global_counter_ = global_counter;
#ifdef SSC
  xorshift32state = seed + 1;
  assert(xorshift32state > 0);
#endif
  nwords_ = nwords;
  wo_ = wo;
  args_ = args;
  osz_ = wo->size(0);
  hsz_ = args->dim;
  negpos = 0;
  loss_ = 0.0;
  nexamples_ = 1;
  t_sigmoid_.reserve(SIGMOID_TABLE_SIZE + 1);
  t_log_.reserve(LOG_TABLE_SIZE + 1);
  initSigmoid();
  initLog();
#ifndef __APPLE__
  // FE_OVERFLOW, FE_UNDERFLOW may reasonably occur
  feenableexcept(FE_DIVBYZERO | FE_INVALID);
#endif
}

void Model::setQuantizePointer(std::shared_ptr<QMatrix> qwi,
                               std::shared_ptr<QMatrix> qwo, bool qout) {
  qwi_ = qwi;
  qwo_ = qwo;
  if (qout) {
    osz_ = qwo_->getM();
  }
}

real Model::binaryLogistic(int32_t target, bool label, real lr) {
  real score = sigmoid(wo_->dotRow(hidden_, target));
  real alpha = (real(label) - score);

  // lr for grad_ will be calculated in proximalUpdate for AdaGrad
  if (!args_->adagrad) {
    alpha *= lr;
  }
  grad_.addRow(*wo_, target, alpha);

  // lr for wo_[target] must be calculated here
  if (args_->adagrad) {
    // 1. Update state
    (*wo_state_)[target] +=
        cblas_sdsdot(wo_->cols(), 0, &(wo_->data()[target * wo_->cols()]), 1,
                     &(wo_->data()[target * wo_->cols()]), 1) /
        wo_->cols();

    // 2. Adapt alpha based on AdaGrad lr
    alpha *= lr / std::sqrt(args_->eps + (*wo_state_)[target]);
  }

  wo_->addRow(hidden_, target, alpha);
  if (label) {
    return -log(score);
  } else {
    return -log(1.0 - score);
  }
}

real Model::negativeSampling(int32_t target, real lr) {
  real loss = 0.0;
  grad_.zero();
  for (int32_t n = 0; n <= args_->neg; n++) {
    if (n == 0) {
      loss += binaryLogistic(target, true, lr);
    } else {
      loss += binaryLogistic(getNegative(target), false, lr);
    }
  }
  return loss;
}

real Model::hierarchicalSoftmax(int32_t target, real lr) {
  real loss = 0.0;
  grad_.zero();
  const std::vector<bool>& binaryCode = codes[target];
  const std::vector<int32_t>& pathToRoot = paths[target];
  for (int32_t i = 0; i < pathToRoot.size(); i++) {
    loss += binaryLogistic(pathToRoot[i], binaryCode[i], lr);
  }
  return loss;
}

void Model::computeOutputSoftmax(Vector& hidden, Vector& output) const {
  if (quant_ && args_->qout) {
    output.mul(*qwo_, hidden);
  } else {
    output.mul(*wo_, hidden);
  }
  real max = output[0], z = 0.0;
  for (int32_t i = 0; i < osz_; i++) {
    max = std::max(output[i], max);
  }
  for (int32_t i = 0; i < osz_; i++) {
    output[i] = exp(output[i] - max);
    z += output[i];
  }
  for (int32_t i = 0; i < osz_; i++) {
    output[i] /= z;
  }
}

void Model::computeOutputSoftmax() {
  computeOutputSoftmax(hidden_, output_);
}

real Model::softmax(int32_t target, real lr) {
  grad_.zero();
  computeOutputSoftmax();
  for (int32_t i = 0; i < osz_; i++) {
    real label = (i == target) ? 1.0 : 0.0;
    real alpha;
    if (!args_->adagrad) {
      alpha = lr * (label - output_[i]);
    } else {
      alpha = (label - output_[i]);
    }

    grad_.addRow(*wo_, i, alpha);
    wo_->addRow(hidden_, i, alpha);
  }
  return -log(output_[target]);
}

void Model::computeHidden(const std::vector<int32_t>& input, Vector& hidden) const {
  assert(hidden.size() == hsz_);
  hidden.zero();
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    if(quant_) {
      hidden.addRow(*qwi_, *it);
    } else {
      hidden.addRow(*wi_, *it);
    }
  }
  hidden.mul(1.0 / input.size());
}

float Model::getNorm(const int32_t &it) { return wi_->l2NormRow(it); }

void Model::forceEagerUpdate(const std::vector<int32_t> &input, const real &lr,
                             const real &word_l2, const real &ngram_l2,
                             const int64_t &counter) {
  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    auto l2 = (*it < nwords_) ? word_l2 : ngram_l2;
    if (l2) {
      forceEagerUpdate(*it, lr, l2, counter);
    }
  }
}

real Model::forceEagerUpdate(const int32_t &it, const real &lr_, const real &l2,
                             const int64_t &counter) {
  assert(l2 > 0);
  real lr = lr_;

  int64_t counter_wi = (*wi_counter_)[it].load(MEMORY_ORDER);
  int64_t delay = (counter - 1) - counter_wi;

  // Only update stale parameters
  if (delay <= 0) {
    return -1; // Norm was not calculated
  }

  auto norm = wi_->l2NormRow(it);

  if (norm > 0) {
    if (args_->adagrad) {
      lr = lr / std::sqrt(args_->eps + (*wi_state_)[it]);
    }

    real scale{1};
    real lambda = lr * l2;
    if (!args_->adagrad) {
      lambda = (lambda + (*wi_state_)[it]) / 2;
    }

    // Only exact when running with AdaGrad
    // This may FE_OVERFLOW
    scale = std::max(real(0), 1 - lambda / norm);
    if (scale > 0) {
      scale = std::pow(scale, real(delay));
    }

    // 1. Update counters
    (*wi_counter_)[it].store(counter - 1);
    if (!args_->adagrad) {
      (*wi_state_)[it] = lr * l2;
    }

    // 2. Update parameters
    wi_->multiplyRow(scale, it);

    return norm * scale;
  } else {
    return 0;
  }
}

bool Model::comparePairs(const std::pair<real, int32_t> &l,
                         const std::pair<real, int32_t> &r) {
  return l.first > r.first;
}

void Model::predict(const std::vector<int32_t>& input, int32_t k, real threshold,
                    std::vector<std::pair<real, int32_t>>& heap,
                    Vector& hidden, Vector& output) const {
  if (k <= 0) {
    throw std::invalid_argument("k needs to be 1 or higher!");
  }
  if (args_->model != model_name::sup) {
    throw std::invalid_argument("Model needs to be supervised for prediction!");
  }
  heap.reserve(k + 1);
  computeHidden(input, hidden);
  if (args_->loss == loss_name::hs) {
    dfs(k, threshold, 2 * osz_ - 2, 0.0, heap, hidden);
  } else {
    findKBest(k, threshold, heap, hidden, output);
  }
  std::sort_heap(heap.begin(), heap.end(), comparePairs);
}

void Model::predict(
  const std::vector<int32_t>& input,
  int32_t k,
  real threshold,
  std::vector<std::pair<real, int32_t>>& heap
) {
  predict(input, k, threshold, heap, hidden_, output_);
}

void Model::findKBest(
  int32_t k,
  real threshold,
  std::vector<std::pair<real, int32_t>>& heap,
  Vector& hidden, Vector& output
) const {
  computeOutputSoftmax(hidden, output);
  for (int32_t i = 0; i < osz_; i++) {
    if (output[i] < threshold) continue;
    if (heap.size() == k && std_log(output[i]) < heap.front().first) {
      continue;
    }
    heap.push_back(std::make_pair(std_log(output[i]), i));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
  }
}

void Model::dfs(int32_t k, real threshold, int32_t node, real score,
                std::vector<std::pair<real, int32_t>>& heap,
                Vector& hidden) const {
  if (score < std_log(threshold)) return;
  if (heap.size() == k && score < heap.front().first) {
    return;
  }

  if (tree[node].left == -1 && tree[node].right == -1) {
    heap.push_back(std::make_pair(score, node));
    std::push_heap(heap.begin(), heap.end(), comparePairs);
    if (heap.size() > k) {
      std::pop_heap(heap.begin(), heap.end(), comparePairs);
      heap.pop_back();
    }
    return;
  }

  real f;
  if (quant_ && args_->qout) {
    f= qwo_->dotRow(hidden, node - osz_);
  } else {
    f= wo_->dotRow(hidden, node - osz_);
  }
  f = 1. / (1 + std::exp(-f));

  dfs(k, threshold, tree[node].left, score + std_log(1.0 - f), heap, hidden);
  dfs(k, threshold, tree[node].right, score + std_log(f), heap, hidden);
}

void Model::update(const std::vector<int32_t> &input, int32_t target, real lr,
                   const real word_l2, const real ngram_l2) {
  assert(target >= 0);
  assert(target < osz_);
  if (input.size() == 0)
    return;

  if (word_l2 > 0 || ngram_l2 > 0) {
#ifdef SSC
    global_counter_->inc(global_counter_->xorshift32(&xorshift32state),
                         MEMORY_ORDER);
    local_counter_ = global_counter_->load();
#else
    local_counter_ = global_counter_->fetch_add(1, MEMORY_ORDER);
#endif
    grad_.zero();
    forceEagerUpdate(input, lr, word_l2, ngram_l2, local_counter_);
  }

  computeHidden(input, hidden_);
  if (args_->loss == loss_name::ns) {
    loss_ += negativeSampling(target, lr);
  } else if (args_->loss == loss_name::hs) {
    loss_ += hierarchicalSoftmax(target, lr);
  } else {
    loss_ += softmax(target, lr);
  }
  nexamples_ += 1;

  if (args_->model == model_name::sup) {
    grad_.mul(1.0 / input.size());
  }

  for (auto it = input.cbegin(); it != input.cend(); ++it) {
    auto l2 = (*it < nwords_) ? word_l2 : ngram_l2;
    proximalUpdate(*it, lr, l2);
  }
}

void Model::proximalUpdate(const int32_t &it, const real &lr_, const real &l2) {
  real lr = lr_;
  // TODO memory order acq and rel
  if (l2) {
    if ((*wi_counter_)[it].load(MEMORY_ORDER) <= local_counter_) {
      (*wi_counter_)[it].store(local_counter_, MEMORY_ORDER);
    }
  }

  if (args_->adagrad) {
    // 1. Update state
    (*wi_state_)[it] +=
        cblas_sdsdot(grad_.size(), 0, grad_.data(), 1, grad_.data(), 1) /
        grad_.size();

    // 2. Rescale gradient by new lr
    lr = lr / std::sqrt(args_->eps + (*wi_state_)[it]);
    grad_.mul(lr);
  } else {
    (*wi_state_)[it] = lr * l2;
  }

  for (int i{0}; i < grad_.size(); i++) {
    tmp_[i] = wi_->at(it, i) + grad_[i];
  }
  real norm = tmp_.norm();
  if (l2 > 0) {
    real lambda = lr * l2;
    if (norm > 0) {
      real scale = std::max(real(0), 1 - lambda / norm);
      wi_->addRescaleRow(grad_, it, scale);
    } else {
      wi_->multiplyRow(0, it);
    }
  } else {
    wi_->addRow(grad_, it, 1);
  }
}

void Model::setTargetCounts(const std::vector<int64_t> &counts) {
  assert(counts.size() == osz_);
  if (args_->loss == loss_name::ns) {
    initTableNegatives(counts);
  }
  if (args_->loss == loss_name::hs) {
    buildTree(counts);
  }
}

void Model::initTableNegatives(const std::vector<int64_t>& counts) {
  real z = 0.0;
  for (size_t i = 0; i < counts.size(); i++) {
    z += pow(counts[i], 0.5);
  }
  for (size_t i = 0; i < counts.size(); i++) {
    real c = pow(counts[i], 0.5);
    for (size_t j = 0; j < c * NEGATIVE_TABLE_SIZE / z; j++) {
      negatives_.push_back(i);
    }
  }
  std::shuffle(negatives_.begin(), negatives_.end(), rng);
}

int32_t Model::getNegative(int32_t target) {
  int32_t negative;
  do {
    negative = negatives_[negpos];
    negpos = (negpos + 1) % negatives_.size();
  } while (target == negative);
  return negative;
}

void Model::buildTree(const std::vector<int64_t>& counts) {
  tree.resize(2 * osz_ - 1);
  for (int32_t i = 0; i < 2 * osz_ - 1; i++) {
    tree[i].parent = -1;
    tree[i].left = -1;
    tree[i].right = -1;
    tree[i].count = 1e15;
    tree[i].binary = false;
  }
  for (int32_t i = 0; i < osz_; i++) {
    tree[i].count = counts[i];
  }
  int32_t leaf = osz_ - 1;
  int32_t node = osz_;
  for (int32_t i = osz_; i < 2 * osz_ - 1; i++) {
    int32_t mini[2];
    for (int32_t j = 0; j < 2; j++) {
      if (leaf >= 0 && tree[leaf].count < tree[node].count) {
        mini[j] = leaf--;
      } else {
        mini[j] = node++;
      }
    }
    tree[i].left = mini[0];
    tree[i].right = mini[1];
    tree[i].count = tree[mini[0]].count + tree[mini[1]].count;
    tree[mini[0]].parent = i;
    tree[mini[1]].parent = i;
    tree[mini[1]].binary = true;
  }
  for (int32_t i = 0; i < osz_; i++) {
    std::vector<int32_t> path;
    std::vector<bool> code;
    int32_t j = i;
    while (tree[j].parent != -1) {
      path.push_back(tree[j].parent - osz_);
      code.push_back(tree[j].binary);
      j = tree[j].parent;
    }
    paths.push_back(path);
    codes.push_back(code);
  }
}

real Model::getLoss() const {
  return loss_ / nexamples_;
}

void Model::initSigmoid() {
  for (int i = 0; i < SIGMOID_TABLE_SIZE + 1; i++) {
    real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
    t_sigmoid_.push_back(1.0 / (1.0 + std::exp(-x)));
  }
}

void Model::initLog() {
  for (int i = 0; i < LOG_TABLE_SIZE + 1; i++) {
    real x = (real(i) + 1e-5) / LOG_TABLE_SIZE;
    t_log_.push_back(std::log(x));
  }
}

real Model::log(real x) const {
  if (x > 1.0) {
    return 0.0;
  }
  int64_t i = int64_t(x * LOG_TABLE_SIZE);
  return t_log_[i];
}

real Model::std_log(real x) const {
  return std::log(x+1e-5);
}

real Model::sigmoid(real x) const {
  if (x < -MAX_SIGMOID) {
    return 0.0;
  } else if (x > MAX_SIGMOID) {
    return 1.0;
  } else {
    int64_t i = int64_t((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2);
    return t_sigmoid_[i];
  }
}

}
