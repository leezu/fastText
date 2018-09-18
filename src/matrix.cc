/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "matrix.h"

#include <random>
#include <exception>
#include <stdexcept>

#include "utils.h"
#include "vector.h"

namespace fasttext {

Matrix::Matrix() : Matrix(0, 0) {}

Matrix::Matrix(int64_t m, int64_t n) : data_(m * n), m_(m), n_(n) {}

void Matrix::zero() {
  std::fill(data_.begin(), data_.end(), 0.0);
}

void Matrix::uniform(real a) {
  std::minstd_rand rng(1);
  std::uniform_real_distribution<> uniform(-a, a);
  for (int64_t i = 0; i < (m_ * n_); i++) {
    data_[i] = uniform(rng);
  }
}

real Matrix::dotRow(const Vector& vec, int64_t i) const {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.size() == n_);
  real d = 0.0;
  for (int64_t j = 0; j < n_; j++) {
    d += at(i, j) * vec[j];
  }
  if (std::isnan(d)) {
    throw std::runtime_error("Encountered NaN.");
  }
  return d;
}

void Matrix::addRow(const Vector& vec, int64_t i, real a) {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.size() == n_);
  for (int64_t j = 0; j < n_; j++) {
    data_[i * n_ + j] += a * vec[j];
  }
}

void Matrix::addRescaleRow(const Vector &vec, int64_t i, real a) {
  assert(i >= 0);
  assert(i < m_);
  assert(vec.size() == n_);
  for (int64_t j = 0; j < n_; j++) {
    data_[i * n_ + j] = a * (data_[i * n_ + j] + vec[j]);
  }
}

void Matrix::multiplyRow(const real num, int64_t i) {
  for (auto j = 0; j < n_; j++) {
    at(i, j) *= num;
  }
}

void Matrix::multiplyRow(const Vector& nums, int64_t ib, int64_t ie) {
  if (ie == -1) {
    ie = m_;
  }
  assert(ie <= nums.size());
  for (auto i = ib; i < ie; i++) {
    real n = nums[i - ib];
    if (n != 0) {
      for (auto j = 0; j < n_; j++) {
        at(i, j) *= n;
      }
    }
  }
}

void Matrix::divideRow(const Vector& denoms, int64_t ib, int64_t ie) {
  if (ie == -1) {
    ie = m_;
  }
  assert(ie <= denoms.size());
  for (auto i = ib; i < ie; i++) {
    real n = denoms[i - ib];
    if (n != 0) {
      for (auto j = 0; j < n_; j++) {
        at(i, j) /= n;
      }
    }
  }
}

real Matrix::l2NormRow(int64_t i) const {
  real ssq = 0;
  real scale = 0;
  for (auto j = 0; j < n_; j++) {
    real atij = at(i, j);
    if (atij != 0) {
      auto abs = std::abs(atij);
      if (scale < abs) {
        ssq = 1 + ssq * (scale / abs) * (scale / abs);
        scale = abs;
      } else {
        ssq = ssq + (abs / scale) * (abs / scale);
      }
    }
  }
  auto norm =  scale * std::sqrt(ssq);
  return std::sqrt(norm);
}

real Matrix::l2NormRow(int64_t i, const Vector &vec) const {
  real ssq = 0;
  real scale = 0;
  for (auto j = 0; j < n_; j++) {
    real atij = at(i, j) + vec[j];
    if (atij != 0) {
      auto abs = std::abs(atij);
      if (scale < abs) {
        ssq = 1 + ssq * (scale / abs) * (scale / abs);
        scale = abs;
      } else {
        ssq = ssq + (abs / scale) * (abs / scale);
      }
    }
  }
  auto norm = scale * std::sqrt(ssq);
  return std::sqrt(norm);
}

void Matrix::l2NormRow(Vector& norms) const {
  assert(norms.size() == m_);
  for (auto i = 0; i < m_; i++) {
    norms[i] = l2NormRow(i);
  }
}

void Matrix::save(std::ostream& out) {
  out.write((char*)&m_, sizeof(int64_t));
  out.write((char*)&n_, sizeof(int64_t));
  out.write((char*)data_.data(), m_ * n_ * sizeof(real));
}

void Matrix::load(std::istream& in) {
  in.read((char*)&m_, sizeof(int64_t));
  in.read((char*)&n_, sizeof(int64_t));
  data_ = std::vector<real>(m_ * n_);
  in.read((char*)data_.data(), m_ * n_ * sizeof(real));
}

void Matrix::dump(std::ostream& out) const {
  out << m_ << " " << n_ << std::endl;
  for (int64_t i = 0; i < m_; i++) {
    for (int64_t j = 0; j < n_; j++) {
      if (j > 0) {
        out << " ";
      }
      out << at(i, j);
    }
    out << std::endl;
  }
};

}
