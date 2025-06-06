// MIT License

// Copyright (c) 2023 Jiashen Cao

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cstdint>

#define HASH(X,Y,Z) ((X-Z) % Y)

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirectAndPHT_1(
    int tid,
    K  (&items)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (selection_flags[ITEM]) {
      int hash = HASH(items[ITEM], ht_len, keys_min);

      K slot = ht[hash];
      if (slot != 0) {
        selection_flags[ITEM] = 1;
      } else {
        selection_flags[ITEM] = 0;
      }
    }
  }
}

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirectAndPHT_1(
    int tid,
    K  (&items)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        int hash = HASH(items[ITEM], ht_len, keys_min);

        K slot = ht[hash];
        if (slot != 0) {
          selection_flags[ITEM] = 1;
        } else {
          selection_flags[ITEM] = 0;
        }
      }
    }
  }
}

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndPHT_1(
    K  (&items)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
    ) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockProbeDirectAndPHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, selection_flags, ht, ht_len, keys_min);
  } else {
    BlockProbeDirectAndPHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, selection_flags, ht, ht_len, keys_min, num_items);
  }
}

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndPHT_1(
    K  (&items)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    int num_items
    ) {
  BlockProbeAndPHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(items, selection_flags, ht, ht_len, 0, num_items);
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirectAndPHT_2(
    int tid,
    K  (&keys)[ITEMS_PER_THREAD],
    V  (&res)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (selection_flags[ITEM]) {
      int hash = HASH(keys[ITEM], ht_len, keys_min);

      uint64_t slot = *reinterpret_cast<uint64_t*>(&ht[hash << 1]);
      if (slot != 0) {
        res[ITEM] = (slot >> 32);
      } else {
        selection_flags[ITEM] = 0;
      }
    }
  }
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirectAndPHT_2(
    int tid,
    K  (&items)[ITEMS_PER_THREAD],
    V  (&res)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        int hash = HASH(items[ITEM], ht_len, keys_min);

        uint64_t slot = *reinterpret_cast<uint64_t*>(&ht[hash << 1]);
        if (slot != 0) {
          res[ITEM] = (slot >> 32);
        } else {
          selection_flags[ITEM] = 0;
        }
      }
    }
  }
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndPHT_2(
    K  (&keys)[ITEMS_PER_THREAD],
    V  (&res)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
    ) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockProbeDirectAndPHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, keys, res, selection_flags, ht, ht_len, keys_min);
  } else {
    BlockProbeDirectAndPHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, keys, res, selection_flags, ht, ht_len, keys_min, num_items);
  }
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndPHT_2(
    K  (&keys)[ITEMS_PER_THREAD],
    V  (&res)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    int num_items
    ) {
  BlockProbeAndPHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, res, selection_flags, ht, ht_len, 0, num_items);
}

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildDirectSelectivePHT_1(
    int tid,
    K  (&keys)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (selection_flags[ITEM]) {
      int hash = HASH(keys[ITEM], ht_len, keys_min);

      K old = atomicCAS(&ht[hash], 0, keys[ITEM]);
    }
  }
}

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildDirectSelectivePHT_1(
    int tid,
    K  (&items)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        int hash = HASH(items[ITEM], ht_len, keys_min);

        K old = atomicCAS(&ht[hash], 0, items[ITEM]);
      }
    }
  }
}

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildSelectivePHT_1(
    K  (&keys)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
    ) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockBuildDirectSelectivePHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, keys, selection_flags, ht, ht_len, keys_min);
  } else {
    BlockBuildDirectSelectivePHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, keys, selection_flags, ht, ht_len, keys_min, num_items);
  }
}

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildSelectivePHT_1(
    K  (&keys)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    int num_items
    ) {
  BlockBuildSelectivePHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, selection_flags, ht, ht_len, 0, num_items);
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildDirectSelectivePHT_2(
    int tid,
    K  (&keys)[ITEMS_PER_THREAD],
    V  (&res)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (selection_flags[ITEM]) {
      int hash = HASH(keys[ITEM], ht_len, keys_min);

      K old = atomicCAS(&ht[hash << 1], 0, keys[ITEM]);
      ht[(hash << 1) + 1] = res[ITEM];
    }
  }
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildDirectSelectivePHT_2(
    int tid,
    K  (&keys)[ITEMS_PER_THREAD],
    V  (&res)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        int hash = HASH(keys[ITEM], ht_len, keys_min);

        K old = atomicCAS(&ht[hash << 1], 0, keys[ITEM]);
        ht[(hash << 1) + 1] = res[ITEM];
      }
    }
  }
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildSelectivePHT_2(
    K  (&keys)[ITEMS_PER_THREAD],
    V  (&res)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items
    ) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockBuildDirectSelectivePHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(
        threadIdx.x, keys, res, selection_flags, ht, ht_len, keys_min);
  } else {
    BlockBuildDirectSelectivePHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(
        threadIdx.x, keys, res, selection_flags, ht, ht_len, keys_min, num_items);
  }
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildSelectivePHT_2(
    K  (&keys)[ITEMS_PER_THREAD],
    V  (&res)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    int num_items
    ) {
  BlockBuildSelectivePHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, res, selection_flags, ht, ht_len, 0, num_items);
}
