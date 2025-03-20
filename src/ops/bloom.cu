#include <iostream>
#include <cuda_runtime.h>
#include <crystal/crystal.cuh>


// Test kernel for building a bloom filter
template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void TestBuildBloomFilter(
    K* d_keys,
    int* d_selection_flags,
    uint32_t* d_bloom_filter,
    int bloom_filter_size,
    int num_items) {
    
    // Shared memory for this thread block's keys
    __shared__ K s_keys[BLOCK_THREADS * ITEMS_PER_THREAD];
    
    // Shared memory for selection flags
    __shared__ int s_selection_flags[BLOCK_THREADS * ITEMS_PER_THREAD];
    
    // Load keys and flags into shared memory
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = threadIdx.x + (i * BLOCK_THREADS);
        if (idx < num_items) {
            s_keys[idx] = d_keys[idx];
            s_selection_flags[idx] = d_selection_flags[idx];
        }
    }
    
    __syncthreads();
    
    // Thread-local storage for items
    K items[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];
    
    // Load from shared memory into registers
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = threadIdx.x + (i * BLOCK_THREADS);
        if (idx < num_items) {
            items[i] = s_keys[idx];
            selection_flags[i] = s_selection_flags[idx];
        } else {
            items[i] = 0;
            selection_flags[i] = 0;
        }
    }
    
    // Build bloom filter
    BlockBuildBloomFilter<K, BLOCK_THREADS, ITEMS_PER_THREAD, 1>(
        items, selection_flags, d_bloom_filter, bloom_filter_size, num_items);
}

// Test kernel for probing a bloom filter
template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void TestProbeBloomFilter(
    K* d_keys,
    int* d_selection_flags,
    uint32_t* d_bloom_filter,
    int bloom_filter_size,
    int num_items) {
    
    // Shared memory for this thread block's keys
    __shared__ K s_keys[BLOCK_THREADS * ITEMS_PER_THREAD];
    
    // Shared memory for selection flags
    __shared__ int s_selection_flags[BLOCK_THREADS * ITEMS_PER_THREAD];
    
    // Load keys and flags into shared memory
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = threadIdx.x + (i * BLOCK_THREADS);
        if (idx < num_items) {
            s_keys[idx] = d_keys[idx];
            s_selection_flags[idx] = d_selection_flags[idx];
        }
    }
    
    __syncthreads();
    
    // Thread-local storage for items
    K items[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];
    
    // Load from shared memory into registers
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = threadIdx.x + (i * BLOCK_THREADS);
        if (idx < num_items) {
            items[i] = s_keys[idx];
            selection_flags[i] = s_selection_flags[idx];
        } else {
            items[i] = 0;
            selection_flags[i] = 0;
        }
    }
    
    // Probe bloom filter
    BlockProbeBloomFilter<K, BLOCK_THREADS, ITEMS_PER_THREAD, 1>(
        items, selection_flags, d_bloom_filter, bloom_filter_size, num_items);
    
    // Store results back to shared memory
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = threadIdx.x + (i * BLOCK_THREADS);
        if (idx < num_items) {
            s_selection_flags[idx] = selection_flags[i];
        }
    }
    
    __syncthreads();
    
    // Write results back to global memory
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = threadIdx.x + (i * BLOCK_THREADS);
        if (idx < num_items) {
            d_selection_flags[idx] = s_selection_flags[idx];
        }
    }
}

int main() {
    // Test parameters
    const int BLOCK_THREADS = 256;
    const int ITEMS_PER_THREAD = 4;
    const int NUM_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;  // Total items to insert
    const int BLOOM_FILTER_SIZE = 64; 
    
    // Host arrays
    int h_keys[NUM_ITEMS];
    int h_selection_flags[NUM_ITEMS];
    int h_probe_keys[NUM_ITEMS];
    int h_probe_flags[NUM_ITEMS];
    bool h_expected[NUM_ITEMS];  // Expected results array
    uint32_t h_bloom_filter[BLOOM_FILTER_SIZE];
    
    // Initialize arrays
    for (int i = 0; i < NUM_ITEMS; i++) {
        // Determine which elements to insert in the filter (every 3rd number)
        h_keys[i] = i;
        h_selection_flags[i] = (i % 3 == 0) ? 1 : 0;  // Only insert every 3rd number
        
        // For probing, use a mix of values (some in the filter, some not)
        h_probe_keys[i] = i;
        h_probe_flags[i] = 1;  // All valid for probing
        
        // Mark which values are expected to be in the filter
        h_expected[i] = (i % 3 == 0);  // Every 3rd number is in the filter
    }
    
    // Zero out bloom filter
    memset(h_bloom_filter, 0, BLOOM_FILTER_SIZE * sizeof(uint32_t));
    
    // Device pointers
    int* d_keys;
    int* d_selection_flags;
    int* d_probe_keys;
    int* d_probe_flags;
    uint32_t* d_bloom_filter;
    
    // Allocate device memory
    cudaMalloc(&d_keys, NUM_ITEMS * sizeof(int));
    cudaMalloc(&d_selection_flags, NUM_ITEMS * sizeof(int));
    cudaMalloc(&d_probe_keys, NUM_ITEMS * sizeof(int));
    cudaMalloc(&d_probe_flags, NUM_ITEMS * sizeof(int));
    cudaMalloc(&d_bloom_filter, BLOOM_FILTER_SIZE * sizeof(uint32_t));
    
    // Copy data to device
    cudaMemcpy(d_keys, h_keys, NUM_ITEMS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_selection_flags, h_selection_flags, NUM_ITEMS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_probe_keys, h_probe_keys, NUM_ITEMS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_probe_flags, h_probe_flags, NUM_ITEMS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bloom_filter, h_bloom_filter, BLOOM_FILTER_SIZE * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Set up kernel dimensions
    int grid_size = 1;  // Single block for this simple test
    
    // Build bloom filter - only insert values where i % 3 == 0
    TestBuildBloomFilter<int, BLOCK_THREADS, ITEMS_PER_THREAD><<<grid_size, BLOCK_THREADS>>>(
        d_keys, d_selection_flags, d_bloom_filter, BLOOM_FILTER_SIZE, NUM_ITEMS);
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Build kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Probe bloom filter
    TestProbeBloomFilter<int, BLOCK_THREADS, ITEMS_PER_THREAD><<<grid_size, BLOCK_THREADS>>>(
        d_probe_keys, d_probe_flags, d_bloom_filter, BLOOM_FILTER_SIZE, NUM_ITEMS);
    
    // Check for kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Probe kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(h_probe_flags, d_probe_flags, NUM_ITEMS * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Verify results
    int true_positives = 0;
    int false_negatives = 0;
    int true_negatives = 0;
    int false_positives = 0;
    
    for (int i = 0; i < NUM_ITEMS; i++) {
        bool should_be_in_filter = h_expected[i];
        bool found = (h_probe_flags[i] == 1);
        
        if (should_be_in_filter && found) true_positives++;
        if (should_be_in_filter && !found) false_negatives++;
        if (!should_be_in_filter && !found) true_negatives++;
        if (!should_be_in_filter && found) false_positives++;
    }
    
    // Print results
    std::cout << "Bloom Filter Test Results:" << std::endl;
    std::cout << "------------------------" << std::endl;
    std::cout << "True Positives: " << true_positives << std::endl;
    std::cout << "False Negatives: " << false_negatives << " (should be 0)" << std::endl;
    std::cout << "True Negatives: " << true_negatives << std::endl;
    std::cout << "False Positives: " << false_positives << " (some may exist due to bloom filter nature)" << std::endl;
    
    // Calculate false positive rate
    float false_positive_rate = 0.0f;
    if (false_positives + true_negatives > 0) {
        false_positive_rate = static_cast<float>(false_positives) / (false_positives + true_negatives);
    }
    
    std::cout << "False Positive Rate: " << (false_positive_rate * 100.0f) << "%" << std::endl;
    
    // Free device memory
    cudaFree(d_keys);
    cudaFree(d_selection_flags);
    cudaFree(d_probe_keys);
    cudaFree(d_probe_flags);
    cudaFree(d_bloom_filter);
    
    return 0;
}