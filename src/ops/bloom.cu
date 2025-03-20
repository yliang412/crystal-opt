#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "crystal/crystal.cuh"

// Define error checking macro
#define CUDA_CHECK(call) \
  do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", \
              __FILE__, __LINE__, cudaGetErrorString(error)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

// Test parameters
const int BLOCK_THREADS = 128;
const int ITEMS_PER_THREAD = 4;
const int NUM_HASHES = 4;
const int BLOOM_SIZE_INTS = 1024; // Size of bloom filter in 32-bit words (32K bits total)
const float EXPECTED_FALSE_POSITIVE_RATE = 0.01f; // Expected FP rate for sizing validation

// CUDA kernel for building the bloom filter
template<typename K>
__global__ void BuildBloomFilterKernel(
    K* build_keys,
    uint32_t* bloom_filter,
    int bloom_filter_size,
    int num_build_items)
{
    // Thread ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Initialize bloom filter to all zeros (only thread 0 in first block)
    if (tid == 0) {
        for (int i = 0; i < bloom_filter_size; i++) {
            bloom_filter[i] = 0;
        }
    }
    
    __syncthreads(); // ensure initialization is complete
    
    // Local storage for keys to be processed by this thread
    K build_items[ITEMS_PER_THREAD];
    int build_selection_flags[ITEMS_PER_THREAD];
    
    // Load build items
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        int idx = tid + (ITEM * gridDim.x * blockDim.x);
        if (idx < num_build_items) {
            build_items[ITEM] = build_keys[idx];
            build_selection_flags[ITEM] = 1; // Select all items
        } else {
            build_selection_flags[ITEM] = 0; // Don't process out-of-bounds items
        }
    }
    
    // Build the bloom filter
    BlockBuildBloomFilter<K, BLOCK_THREADS, ITEMS_PER_THREAD, NUM_HASHES>(
        build_items, 
        build_selection_flags, 
        bloom_filter, 
        bloom_filter_size, 
        num_build_items);
}

// CUDA kernel for probing the bloom filter
template<typename K>
__global__ void ProbeBloomFilterKernel(
    K* probe_keys,
    int* probe_results,
    const uint32_t* bloom_filter,
    int bloom_filter_size,
    int num_probe_items)
{
    // Thread ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Local storage for probe keys
    K probe_items[ITEMS_PER_THREAD];
    int probe_selection_flags[ITEMS_PER_THREAD];
    
    // Load probe items
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        int idx = tid + (ITEM * gridDim.x * blockDim.x);
        if (idx < num_probe_items) {
            probe_items[ITEM] = probe_keys[idx];
            probe_selection_flags[ITEM] = 1; // Select all items
        } else {
            probe_selection_flags[ITEM] = 0; // Don't process out-of-bounds items
        }
    }
    
    // Probe the bloom filter
    BlockProbeBloomFilter<K, BLOCK_THREADS, ITEMS_PER_THREAD, NUM_HASHES>(
        probe_items, 
        probe_selection_flags, 
        bloom_filter, 
        bloom_filter_size, 
        num_probe_items);
    
    // Store the probe results
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        int idx = tid + (ITEM * gridDim.x * blockDim.x);
        if (idx < num_probe_items) {
            probe_results[idx] = probe_selection_flags[ITEM];
        }
    }
}

// Host function to test the bloom filter
template<typename K>
void TestBloomFilter(int num_build_items, int num_probe_items, float overlap_ratio) {
    // Allocate and initialize host data
    K* h_build_keys = new K[num_build_items];
    K* h_probe_keys = new K[num_probe_items];
    int* h_probe_results = new int[num_probe_items];
    int* h_expected_results = new int[num_probe_items];
    
    // Generate build keys (simple sequential values)
    for (int i = 0; i < num_build_items; i++) {
        h_build_keys[i] = static_cast<K>(i + 1); // Start from 1 to avoid 0 (since 0 is used as empty)
    }
    
    // Generate probe keys with specified overlap
    int overlap_items = static_cast<int>(num_probe_items * overlap_ratio);
    
    // First portion are keys that exist in the build set
    for (int i = 0; i < overlap_items; i++) {
        h_probe_keys[i] = h_build_keys[i % num_build_items];
        h_expected_results[i] = 1; // Should be found
    }
    
    // Second portion are keys that don't exist in the build set
    for (int i = overlap_items; i < num_probe_items; i++) {
        h_probe_keys[i] = static_cast<K>(num_build_items + i); // Values outside build set
        h_expected_results[i] = 0; // Should not be found in an ideal bloom filter
    }
    
    // Allocate device memory
    K* d_build_keys;
    K* d_probe_keys;
    int* d_probe_results;
    uint32_t* d_bloom_filter;
    
    CUDA_CHECK(cudaMalloc(&d_build_keys, num_build_items * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&d_probe_keys, num_probe_items * sizeof(K)));
    CUDA_CHECK(cudaMalloc(&d_probe_results, num_probe_items * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bloom_filter, BLOOM_SIZE_INTS * sizeof(uint32_t)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_build_keys, h_build_keys, num_build_items * sizeof(K), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_probe_keys, h_probe_keys, num_probe_items * sizeof(K), cudaMemcpyHostToDevice));
    
    // Calculate grid dimensions for build phase
    int items_per_block = BLOCK_THREADS * ITEMS_PER_THREAD;
    int num_blocks_build = (num_build_items + items_per_block - 1) / items_per_block;
    int num_blocks_probe = (num_probe_items + items_per_block - 1) / items_per_block;
    
    // Launch build kernel
    dim3 grid_build(num_blocks_build);
    dim3 block(BLOCK_THREADS);
    
    BuildBloomFilterKernel<K><<<grid_build, block>>>(
        d_build_keys,
        d_bloom_filter,
        BLOOM_SIZE_INTS,
        num_build_items
    );
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Launch probe kernel
    dim3 grid_probe(num_blocks_probe);
    
    ProbeBloomFilterKernel<K><<<grid_probe, block>>>(
        d_probe_keys,
        d_probe_results,
        d_bloom_filter,
        BLOOM_SIZE_INTS,
        num_probe_items
    );
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_probe_results, d_probe_results, num_probe_items * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Verify results
    int true_positives = 0;
    int false_positives = 0;
    int false_negatives = 0;
    int true_negatives = 0;
    
    for (int i = 0; i < num_probe_items; i++) {
        if (i < overlap_items) {
            // Should be found (true positives)
            if (h_probe_results[i]) {
                true_positives++;
            } else {
                false_negatives++;
                printf("Error: False negative at index %d (key %d)\n", i, h_probe_keys[i]);
            }
        } else {
            // Should not be found, but might have false positives
            if (h_probe_results[i]) {
                false_positives++;
            } else {
                true_negatives++;
            }
        }
    }
    
    // Calculate false positive rate
    float false_positive_rate = 0.0f;
    if ((num_probe_items - overlap_items) > 0) {
        false_positive_rate = static_cast<float>(false_positives) / (num_probe_items - overlap_items);
    }
    
    // Print results
    printf("=== Bloom Filter Test Results ===\n");
    printf("Build items: %d\n", num_build_items);
    printf("Probe items: %d\n", num_probe_items);
    printf("Bloom filter size: %d bits\n", BLOOM_SIZE_INTS * 32);
    printf("Number of hash functions: %d\n", NUM_HASHES);
    printf("Blocks for build phase: %d\n", num_blocks_build);
    printf("Blocks for probe phase: %d\n", num_blocks_probe);
    printf("---------------------------------\n");
    printf("True positives: %d (%.2f%%)\n", true_positives, 100.0f * true_positives / overlap_items);
    printf("False negatives: %d (%.2f%%)\n", false_negatives, 100.0f * false_negatives / overlap_items);
    printf("False positives: %d (%.2f%%)\n", false_positives, 100.0f * false_positives / (num_probe_items - overlap_items));
    printf("True negatives: %d (%.2f%%)\n", true_negatives, 100.0f * true_negatives / (num_probe_items - overlap_items));
    printf("---------------------------------\n");
    printf("False positive rate: %.4f%%\n", 100.0f * false_positive_rate);
    printf("Expected false positive rate: ~%.4f%%\n", 100.0f * EXPECTED_FALSE_POSITIVE_RATE);
    printf("---------------------------------\n");
    printf("Test status: %s\n", 
           (false_negatives == 0 && false_positive_rate <= EXPECTED_FALSE_POSITIVE_RATE * 2) ? 
           "PASSED" : "FAILED");
    
    // Clean up
    delete[] h_build_keys;
    delete[] h_probe_keys;
    delete[] h_probe_results;
    delete[] h_expected_results;
    
    CUDA_CHECK(cudaFree(d_build_keys));
    CUDA_CHECK(cudaFree(d_probe_keys));
    CUDA_CHECK(cudaFree(d_probe_results));
    CUDA_CHECK(cudaFree(d_bloom_filter));
}

int main() {
    // Test with different scenarios
    printf("\n----- Test 1: Small set with high overlap -----\n");
    TestBloomFilter<int>(1000, 2000, 0.5f);
    
    printf("\n----- Test 2: Medium set with low overlap -----\n");
    TestBloomFilter<int>(5000, 10000, 0.2f);
    
    printf("\n----- Test 3: Larger set with very low overlap -----\n");
    TestBloomFilter<int>(10000, 20000, 0.1f);
    
    return 0;
}