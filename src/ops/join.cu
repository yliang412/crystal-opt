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

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <curand.h>

#include <cuda.h>
#include <cub/util_allocator.cuh>
#include "cub/test/test_util.h"

#include "crystal/crystal.cuh"

#include "utils/generator.h"
#include "utils/gpu_utils.h"

using namespace std;

#define DEBUG 1

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_kernel(int *dim_key, int *dim_val, int num_tuples, int *hash_table, int num_slots, int num_select) {
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2, num_tile_items);
  BlockPredLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, num_select, selection_flags, num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, items2, selection_flags, 
      hash_table, num_slots, num_tile_items);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_kernel(int *fact_fkey, int *fact_val, int num_tuples, 
    int *hash_table, int num_slots, unsigned long long *res) {
  // Load a tile striped across threads
  int selection_flags[ITEMS_PER_THREAD];
  int keys[ITEMS_PER_THREAD];
  int vals[ITEMS_PER_THREAD];
  int join_vals[ITEMS_PER_THREAD];

  unsigned long long sum = 0;

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples+ TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(fact_fkey + tile_offset, keys, num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(fact_val + tile_offset, vals, num_tile_items);

  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, join_vals, selection_flags,
      hash_table, num_slots, num_tile_items);
  
  #pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
  {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items))
      if (selection_flags[ITEM])
        sum += vals[ITEM] * join_vals[ITEM];
  }

  __syncthreads();

  static __shared__ long long buffer[32];
  unsigned long long aggregate = BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, (long long*)buffer);
  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(res, aggregate);
  }
}

struct TimeKeeper {
  float time_build;
  float time_probe;
  float time_extra;
  float time_total;
};

TimeKeeper hashJoin(int* d_dim_key, int* d_dim_val, int* d_fact_fkey, int* d_fact_val, int num_dim, int num_fact, int num_select, cub::CachingDeviceAllocator&  g_allocator) {
  SETUP_TIMING();

  int* hash_table = NULL;
  unsigned long long* res;
  int num_slots = num_dim;
  float time_build, time_probe, time_memset, time_memset2;

  ALLOCATE(hash_table, sizeof(int) * 2 * num_dim);
  ALLOCATE(res, sizeof(long long));
  

  TIME_FUNC(cudaMemset(hash_table, 0, num_slots * sizeof(int) * 2), time_memset);
  TIME_FUNC(cudaMemset(res, 0, sizeof(long long)), time_memset2);

  int tile_items = 128*4;

  TIME_FUNC((build_kernel<128, 4><<<(num_dim + tile_items - 1)/tile_items, 128>>>(d_dim_key, d_dim_val, num_dim, hash_table, num_slots, num_select)), time_build);
  TIME_FUNC((probe_kernel<128, 4><<<(num_fact + tile_items - 1)/tile_items, 128>>>(d_fact_fkey, d_fact_val, num_fact, hash_table, num_slots, res)), time_probe);

#if DEBUG
  cout << "{" << "\"time_memset\":" << time_memset + time_memset2
      << ",\"time_build\"" << time_build
      << ",\"time_probe\":" << time_probe << "}" << endl;
#endif

  CLEANUP(hash_table);
  CLEANUP(res);

  TimeKeeper t = {time_build, time_probe, time_memset + time_memset2, time_build + time_probe + time_memset + time_memset2};
  return t;
}

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose = false;  // Whether to display input/output to console
cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory


#define CLEANUP(vec) if(vec)CubDebugExit(g_allocator.DeviceFree(vec))

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------
int main(int argc, char** argv)
{
  // Initialize command line
  CommandLineArgs args(argc, argv);
  int num_fact           = 1<<30;
  args.GetCmdLineArgument("n", num_fact);
  int num_dim            = 16 * 1<<20;
  args.GetCmdLineArgument("d", num_dim);
  int num_trials         = 3;
  args.GetCmdLineArgument("t", num_trials);
  int num_select         = num_dim / 128;
  args.GetCmdLineArgument("s", num_select);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
    printf("%s "
        "[--n=<num fact>] "
        "[--d=<num dim>] "
        "[--t=<num trials>] "
        "[--s=<num select>] "
        "[--device=<device-id>] "
        "[--v] "
        "\n", argv[0]);
    exit(0);
  }

  printf("Running with %d fact, %d dim, num_select: %d, %d trials\n", num_fact, num_dim, num_select, num_trials);


  // Initialize device
  CubDebugExit(args.DeviceInit());

  // Allocate problem device arrays
  int *d_dim_key = NULL;
  int *d_dim_val = NULL;
  int *d_fact_fkey = NULL;
  int *d_fact_val = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_dim_key, sizeof(int) * num_dim));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_dim_val, sizeof(int) * num_dim));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_fact_fkey, sizeof(int) * num_fact));
  CubDebugExit(g_allocator.DeviceAllocate((void**)&d_fact_val, sizeof(int) * num_fact));

  int *h_dim_key = NULL;
  int *h_dim_val = NULL;
  int *h_fact_fkey = NULL;
  int *h_fact_val = NULL;

  create_relation_pk(h_dim_key, h_dim_val, num_dim);
  create_relation_fk(h_fact_fkey, h_fact_val, num_fact, num_dim);

  CubDebugExit(cudaMemcpy(d_dim_key, h_dim_key, sizeof(int) * num_dim, cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(d_dim_val, h_dim_val, sizeof(int) * num_dim, cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(d_fact_fkey, h_fact_fkey, sizeof(int) * num_fact, cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy(d_fact_val, h_fact_val, sizeof(int) * num_fact, cudaMemcpyHostToDevice));

  for (int j = 0; j < num_trials; j++) {
    TimeKeeper t = hashJoin(d_dim_key, d_dim_val, d_fact_fkey, d_fact_val, num_dim, num_fact, num_select, g_allocator);
    cout<< "{"
        << "\"num_dim\":" << num_dim
        << ",\"num_fact\":" << num_fact
        << ",\"num_select\":" << num_select
        << ",\"time_build\":" << t.time_build
        << ",\"time_probe\":" << t.time_probe
        << ",\"time_extra\":" << t.time_extra
        << ",\"time_join_total\":" << t.time_total
        << "}" << endl;
  }

  CLEANUP(d_dim_key);
  CLEANUP(d_dim_val);
  CLEANUP(d_fact_fkey);
  CLEANUP(d_fact_val);

  return 0;
}

