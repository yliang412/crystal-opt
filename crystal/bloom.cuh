#pragma once

#include <cstdint>

// Define single hash function for Bloom Filter
#define HASH1(X,Y,Z) ((X-Z) % (Y))

// Bloom Filter Build API - Sets bits in a bloom filter
template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildDirectBloomFilter(
    int tid,
    K  (&items)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    uint32_t* bloom_filter,
    int bloom_filter_size,
    K keys_min
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (selection_flags[ITEM]) {
      // Apply hash function
      int hash1 = HASH1(items[ITEM], bloom_filter_size * 32, keys_min);
      
      // Set bit in the bloom filter
      atomicOr(&bloom_filter[hash1 >> 5], 1u << (hash1 & 31));
    }
  }
}

// Bloom Filter Build API with bounds checking
template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildDirectBloomFilter(
    int tid,
    K  (&items)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    uint32_t* bloom_filter,
    int bloom_filter_size,
    K keys_min,
    int num_items
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        // Apply hash function
        int hash1 = HASH1(items[ITEM], bloom_filter_size * 32, keys_min);
        
        // Set bit in the bloom filter
        atomicOr(&bloom_filter[hash1 >> 5], 1u << (hash1 & 31));
      }
    }
  }
}

// Wrapper for Bloom Filter Build
template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildBloomFilter(
    K  (&keys)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    uint32_t* bloom_filter,
    int bloom_filter_size,
    K keys_min,
    int num_items
    ) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockBuildDirectBloomFilter<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
        threadIdx.x, keys, selection_flags, bloom_filter, bloom_filter_size, keys_min);
  } else {
    BlockBuildDirectBloomFilter<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
        threadIdx.x, keys, selection_flags, bloom_filter, bloom_filter_size, keys_min, num_items);
  }
}

// Simplified wrapper with default keys_min = 0
template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildBloomFilter(
    K  (&keys)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    uint32_t* bloom_filter,
    int bloom_filter_size,
    int num_items
    ) {
  BlockBuildBloomFilter<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
      keys, selection_flags, bloom_filter, bloom_filter_size, 0, num_items);
}

// Bloom Filter Probe API - Tests membership in bloom filter
template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirectBloomFilter(
    int tid,
    K  (&items)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    const uint32_t* bloom_filter,
    int bloom_filter_size,
    K keys_min
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (selection_flags[ITEM]) {
      // Apply hash function
      int hash1 = HASH1(items[ITEM], bloom_filter_size * 32, keys_min);
      
      // Check if bit is set
      bool bit1_set = (bloom_filter[hash1 >> 5] & (1u << (hash1 & 31))) != 0;
      
      // Item possibly exists if bit is set (may be false positive)
      selection_flags[ITEM] = bit1_set ? 1 : 0;
    }
  }
}

// Bloom Filter Probe API with bounds checking
template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirectBloomFilter(
    int tid,
    K  (&items)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    const uint32_t* bloom_filter,
    int bloom_filter_size,
    K keys_min,
    int num_items
    ) {
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
    if (tid + (ITEM * BLOCK_THREADS) < num_items) {
      if (selection_flags[ITEM]) {
        // Apply hash function
        int hash1 = HASH1(items[ITEM], bloom_filter_size * 32, keys_min);
        
        // Check if bit is set
        bool bit1_set = (bloom_filter[hash1 >> 5] & (1u << (hash1 & 31))) != 0;
        
        // Item possibly exists if bit is set (may be false positive)
        selection_flags[ITEM] = bit1_set ? 1 : 0;
      }
    }
  }
}

// Wrapper for Bloom Filter Probe
template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeBloomFilter(
    K  (&keys)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    const uint32_t* bloom_filter,
    int bloom_filter_size,
    K keys_min,
    int num_items
    ) {

  if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
    BlockProbeDirectBloomFilter<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
        threadIdx.x, keys, selection_flags, bloom_filter, bloom_filter_size, keys_min);
  } else {
    BlockProbeDirectBloomFilter<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
        threadIdx.x, keys, selection_flags, bloom_filter, bloom_filter_size, keys_min, num_items);
  }
}

// Simplified wrapper with default keys_min = 0
template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeBloomFilter(
    K  (&keys)[ITEMS_PER_THREAD],
    int  (&selection_flags)[ITEMS_PER_THREAD],
    const uint32_t* bloom_filter,
    int bloom_filter_size,
    int num_items
    ) {
  BlockProbeBloomFilter<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
      keys, selection_flags, bloom_filter, bloom_filter_size, 0, num_items);
}
