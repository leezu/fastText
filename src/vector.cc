/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "vector.h"

#include "mkl_cblas.h"
#include <assert.h>

#include <cmath>
#include <iomanip>
#include <utility>

#include "matrix.h"
#include "qmatrix.h"

namespace fasttext {

Vector::Vector(int64_t m) : data_(m) {}

Vector::Vector(Vector&& other) noexcept : data_(std::move(other.data_)) {}

Vector& Vector::operator=(Vector&& other) {
  data_ = std::move(other.data_);
  return *this;
}

void Vector::zero() {
  std::fill(data_.begin(), data_.end(), 0.0);
}

real Vector::norm() const {
  return cblas_snrm2(size(), &data_[0], 1);
}

void Vector::mul(real a) {
  cblas_sscal(size(), a, &data_[0], 1);
}

void Vector::addVector(const Vector& source) {
  assert(size() == source.size());
  for (int64_t i = 0; i < size(); i++) {
    data_[i] += source.data_[i];
  }
}

void Vector::addVector(const Vector& source, real s) {
  assert(size() == source.size());
  cblas_saxpy(size(), s, &source.data_[0], 1, &data_[0], 1);
}

void Vector::addRow(const Matrix& A, int64_t i) {
  assert(i >= 0);
  assert(i < A.size(0));
  assert(size() == A.size(1));
  for (int64_t j = 0; j < A.size(1); j++) {
    data_[j] += A.at(i, j);
  }
}

void Vector::addRow(const Matrix& A, int64_t i, real a) {
  assert(i >= 0);
  assert(i < A.size(0));
  assert(size() == A.size(1));
  cblas_saxpy(size(), a, &A.at(i, 0), 1, &data_[0], 1);
}

void Vector::addRow(const QMatrix& A, int64_t i) {
  assert(i >= 0);
  A.addToVector(*this, i);
}

void Vector::mul(const Matrix& A, const Vector& vec) {
  assert(A.size(0) == size());
  assert(A.size(1) == vec.size());
  for (int64_t i = 0; i < size(); i++) {
    data_[i] = A.dotRow(vec, i);
  }
}

void Vector::mul(const QMatrix& A, const Vector& vec) {
  assert(A.getM() == size());
  assert(A.getN() == vec.size());
  for (int64_t i = 0; i < size(); i++) {
    data_[i] = A.dotRow(vec, i);
  }
}

int64_t Vector::argmax() {
  real max = data_[0];
  int64_t argmax = 0;
  for (int64_t i = 1; i < size(); i++) {
    if (data_[i] > max) {
      max = data_[i];
      argmax = i;
    }
  }
  return argmax;
}

std::ostream& operator<<(std::ostream& os, const Vector& v) {
  os << std::setprecision(5);
  for (int64_t j = 0; j < v.size(); j++) {
    os << v[j] << ' ';
  }
  return os;
}

} // namespace fasttext
