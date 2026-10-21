#!/usr/bin/env bash
set -euo pipefail

# Requires:
#   g++ with C++17 support
#   CUDA toolkit
#   TensorRT 11.4 C++ headers and libraries
#   OpenCV 4 development libraries

: "${TRT_INCLUDE_DIR:?Set TRT_INCLUDE_DIR}"
: "${TRT_LIB_DIR:?Set TRT_LIB_DIR}"

CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"

g++ -std=c++17 -O3 -Wno-deprecated-declarations \
  main.cpp -o clip_classifier \
  -I"${TRT_INCLUDE_DIR}" \
  -I"${CUDA_ROOT}/include" \
  -L"${TRT_LIB_DIR}" \
  -L"${CUDA_ROOT}/lib64" \
  -Wl,-rpath,"${TRT_LIB_DIR}" \
  -Wl,-rpath,"${CUDA_ROOT}/lib64" \
  -l:libnvinfer.so.11 -lcudart \
  $(pkg-config --cflags --libs opencv4)

echo "Built ./clip_classifier"
