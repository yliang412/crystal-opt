// 0 - LIP, Join Order: S -> C -> D

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <curand.h>
#include <iostream>
#include <stdio.h>

#include "cub/test/test_util.h"
#include <cub/util_allocator.cuh>
#include <cuda.h>

#include "crystal/crystal.cuh"

#include "../../ssb/gpu_utils.h"
#include "../../ssb/ssb_utils.h"

using namespace std;

/**
 * Globals, constants and typedefs
 */
bool g_verbose = false; // Whether to display input/output to console
cub::CachingDeviceAllocator
    g_allocator(true); // Caching allocator for device memory

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe(int *lo_orderdate, int *lo_custkey, int *lo_suppkey,
                      int *lo_revenue, int lo_len, int *ht_s, int s_len,
                      int *ht_c, int c_len, int *ht_d, int d_len, 
                      uint32_t *bf_s, int bf_s_size,
                      uint32_t *bf_c, int bf_c_size,
                      uint32_t *bf_d, int bf_d_size,
                      int *res) {
  // Load a segment of consecutive items that are blocked across threads
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int items3[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];
  int c_nation[ITEMS_PER_THREAD];
  int s_nation[ITEMS_PER_THREAD];
  int year[ITEMS_PER_THREAD];
  int revenue[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (lo_len + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = lo_len - tile_offset;
  }

  InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

  // BEGIN bf_s bloom
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(lo_suppkey + tile_offset,
                                                  items, num_tile_items);
  BlockProbeBloomFilter<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, selection_flags, bf_s, bf_s_size, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) { return; }
  // END bf_s bloom

  // BEGIN bf_c bloom
  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_custkey + tile_offset, items2, num_tile_items, selection_flags);
  BlockProbeBloomFilter<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, selection_flags, bf_c, bf_c_size, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) { return; }
  // END bf_c bloom

  // BEGIN bf_d bloom
  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_orderdate + tile_offset, items3, num_tile_items, selection_flags);
  BlockProbeBloomFilter<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items3, selection_flags, bf_d, bf_d_size, 19920101, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) { return; }
  // END bf_d bloom

  // BEGIN ht_s hash
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, s_nation, selection_flags, ht_s, s_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) { return; }
  // END ht_s hash

  // BEGIN ht_c hash
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items2, c_nation, selection_flags, ht_c, c_len, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) { return; }
  // END ht_c hash

  // BEGIN ht_d hash
  BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
    items3, year, selection_flags, ht_d, d_len, 19920101, num_tile_items);
  if (IsTerm<int, BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) { return; }
  // END ht_d hash

  BlockPredLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      lo_revenue + tile_offset, revenue, num_tile_items, selection_flags);

#pragma unroll
  for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
    if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items) {
      if (selection_flags[ITEM]) {
        int hash = (s_nation[ITEM] * 250 * 7 + c_nation[ITEM] * 7 +
                    (year[ITEM] - 1992)) %
                   ((1998 - 1992 + 1) * 250 * 250);
        res[hash * 4] = year[ITEM];
        res[hash * 4 + 1] = c_nation[ITEM];
        res[hash * 4 + 2] = s_nation[ITEM];
        atomicAdd(&res[hash * 4 + 3], revenue[ITEM]);
      }
    }
  }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_s(int *dim_key, int *dim_val, int num_tuples,
                                  int *hash_table, int num_slots, 
                                  uint32_t *bf, int bf_size) {
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items,
                                                  num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 231, selection_flags,
                                                    num_tile_items);
  BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 235, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildBloomFilter<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, selection_flags, bf, bf_size, num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items2, items, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_c(int *dim_key, int *dim_val, int num_tuples,
                                  int *hash_table, int num_slots,
                                  uint32_t *bf, int bf_size) {
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items,
                                                  num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 231, selection_flags,
                                                    num_tile_items);
  BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 235, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildBloomFilter<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items2, selection_flags, bf, bf_size, num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items2, items, selection_flags, hash_table, num_slots, num_tile_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_d(int *filter_col, int *dim_key, int *dim_val,
                                  int num_tuples, int *hash_table,
                                  int num_slots, int val_min,
                                  uint32_t *bf, int bf_size) {
  int items[ITEMS_PER_THREAD];
  int items2[ITEMS_PER_THREAD];
  int selection_flags[ITEMS_PER_THREAD];

  int tile_offset = blockIdx.x * TILE_SIZE;
  int num_tiles = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
  int num_tile_items = TILE_SIZE;

  if (blockIdx.x == num_tiles - 1) {
    num_tile_items = num_tuples - tile_offset;
  }

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset,
                                                  items, num_tile_items);
  BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, 199712, selection_flags, num_tile_items);

  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items,
                                                  num_tile_items);
  BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2,
                                                  num_tile_items);
  BlockBuildBloomFilter<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, selection_flags, bf, bf_size, val_min, num_tile_items);
  BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
      items, items2, selection_flags, hash_table, num_slots, 19920101,
      num_tile_items);
}

float runQuery(int *lo_orderdate, int *lo_custkey, int *lo_suppkey,
               int *lo_revenue, int lo_len, int *d_datekey, int *d_year,
               int *d_yearmonthnum, int d_len, int *s_suppkey, int *s_city,
               int s_len, int *c_custkey, int *c_city, int c_len,
               cub::CachingDeviceAllocator &g_allocator) {
  SETUP_TIMING();

  float time_query;
  chrono::high_resolution_clock::time_point st, finish;
  st = chrono::high_resolution_clock::now();

  cudaEventRecord(start, 0);

  int *ht_d, *ht_c, *ht_s;
  int d_val_len = 19981230 - 19920101 + 1;
  uint32_t *bf_s, *bf_c, *bf_d;
  int bf_s_size = (int) (roundUpToPowerOfTwo((uint32_t) s_len) / 32);
  int bf_c_size = (int) (roundUpToPowerOfTwo((uint32_t) c_len) / 32);
  int bf_d_size = (int) (roundUpToPowerOfTwo((uint32_t) d_val_len) / 32);
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_d, 2 * d_val_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_s, 2 * s_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&ht_c, 2 * c_len * sizeof(int)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&bf_s, bf_s_size * sizeof(uint32_t)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&bf_c, bf_c_size * sizeof(uint32_t)));
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&bf_d, bf_d_size * sizeof(uint32_t)));

  CubDebugExit(cudaMemset(ht_d, 0, 2 * d_val_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_s, 0, 2 * s_len * sizeof(int)));
  CubDebugExit(cudaMemset(ht_c, 0, 2 * c_len * sizeof(int)));
  CubDebugExit(cudaMemset(bf_d, 0, bf_d_size * sizeof(uint32_t)));
  CubDebugExit(cudaMemset(bf_s, 0, bf_s_size * sizeof(uint32_t)));
  CubDebugExit(cudaMemset(bf_c, 0, bf_c_size * sizeof(uint32_t)));

  int tile_items = 128 * 4;
  build_hashtable_s<128, 4><<<(s_len + tile_items - 1) / tile_items, 128>>>(
      s_suppkey, s_city, s_len, ht_s, s_len, bf_s, bf_s_size);
  /*CHECK_ERROR();*/

  build_hashtable_c<128, 4><<<(c_len + tile_items - 1) / tile_items, 128>>>(
      c_custkey, c_city, c_len, ht_c, c_len, bf_c, bf_c_size);
  /*CHECK_ERROR();*/

  int d_val_min = 19920101;
  build_hashtable_d<128, 4><<<(d_len + tile_items - 1) / tile_items, 128>>>(
      d_yearmonthnum, d_datekey, d_year, d_len, ht_d, d_val_len, d_val_min, bf_d, bf_d_size);
  /*CHECK_ERROR();*/

  int *res;
  int res_size = ((1998 - 1992 + 1) * 250 * 250);
  int res_array_size = res_size * 4;
  CubDebugExit(
      g_allocator.DeviceAllocate((void **)&res, res_array_size * sizeof(int)));

  CubDebugExit(cudaMemset(res, 0, res_array_size * sizeof(int)));

  int *d_sum = NULL;
  CubDebugExit(g_allocator.DeviceAllocate((void **)&d_sum, sizeof(int)));

  cudaMemset(d_sum, 0, sizeof(int));

  // Run
  probe<128, 4><<<(lo_len + tile_items - 1) / tile_items, 128>>>(
      lo_orderdate, lo_custkey, lo_suppkey, lo_revenue, lo_len, ht_s, s_len,
      ht_c, c_len, ht_d, d_val_len,
      bf_s, bf_s_size,
      bf_c, bf_c_size,
      bf_d, bf_d_size,
      res);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time_query, start, stop);

  int *h_res = new int[res_array_size];
  CubDebugExit(cudaMemcpy(h_res, res, res_array_size * sizeof(int),
                          cudaMemcpyDeviceToHost));
  finish = chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = finish - st;

  cout << "Result:" << endl;
  int res_count = 0;
  for (int i = 0; i < res_size; i++) {
    if (h_res[4 * i] != 0) {
      cout << h_res[4 * i] << " " << h_res[4 * i + 1] << " " << h_res[4 * i + 2]
           << " " << h_res[4 * i + 3] << endl;
      res_count += 1;
    }
  }

  cout << "Res Count: " << res_count << endl;
  cout << "Time Taken Total: " << diff.count() * 1000 << endl;

  delete[] h_res;

  return time_query;
}

/**
 * Main
 */
int main(int argc, char **argv) {
  int num_trials = 3;

  // Initialize command line
  CommandLineArgs args(argc, argv);
  args.GetCmdLineArgument("t", num_trials);

  // Print usage
  if (args.CheckCmdLineFlag("help")) {
    printf("%s "
           "[--t=<num trials>] "
           "[--v] "
           "\n",
           argv[0]);
    exit(0);
  }

  // Initialize device
  CubDebugExit(args.DeviceInit());

  int *h_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
  int *h_lo_custkey = loadColumn<int>("lo_custkey", LO_LEN);
  int *h_lo_suppkey = loadColumn<int>("lo_suppkey", LO_LEN);
  int *h_lo_revenue = loadColumn<int>("lo_revenue", LO_LEN);

  int *h_d_datekey = loadColumn<int>("d_datekey", D_LEN);
  int *h_d_year = loadColumn<int>("d_year", D_LEN);
  int *h_d_yearmonthnum = loadColumn<int>("d_yearmonthnum", D_LEN);

  int *h_s_suppkey = loadColumn<int>("s_suppkey", S_LEN);
  int *h_s_city = loadColumn<int>("s_city", S_LEN);

  int *h_c_custkey = loadColumn<int>("c_custkey", C_LEN);
  int *h_c_city = loadColumn<int>("c_city", C_LEN);

  cout << "** LOADED DATA **" << endl;

  int *d_lo_orderdate = loadToGPU<int>(h_lo_orderdate, LO_LEN, g_allocator);
  int *d_lo_custkey = loadToGPU<int>(h_lo_custkey, LO_LEN, g_allocator);
  int *d_lo_suppkey = loadToGPU<int>(h_lo_suppkey, LO_LEN, g_allocator);
  int *d_lo_revenue = loadToGPU<int>(h_lo_revenue, LO_LEN, g_allocator);

  int *d_d_datekey = loadToGPU<int>(h_d_datekey, D_LEN, g_allocator);
  int *d_d_year = loadToGPU<int>(h_d_year, D_LEN, g_allocator);
  int *d_d_yearmonthnum = loadToGPU<int>(h_d_yearmonthnum, D_LEN, g_allocator);

  int *d_s_suppkey = loadToGPU<int>(h_s_suppkey, S_LEN, g_allocator);
  int *d_s_city = loadToGPU<int>(h_s_city, S_LEN, g_allocator);

  int *d_c_custkey = loadToGPU<int>(h_c_custkey, C_LEN, g_allocator);
  int *d_c_city = loadToGPU<int>(h_c_city, C_LEN, g_allocator);

  cout << "** LOADED DATA TO GPU **" << endl;

  for (int t = 0; t < num_trials; t++) {
    float time_query;
    time_query = runQuery(d_lo_orderdate, d_lo_custkey, d_lo_suppkey,
                          d_lo_revenue, LO_LEN, d_d_datekey, d_d_year,
                          d_d_yearmonthnum, D_LEN, d_s_suppkey, d_s_city, S_LEN,
                          d_c_custkey, d_c_city, C_LEN, g_allocator);
    cout << "{"
         << "\"query\":34"
         << ",\"time_query\":" << time_query << "}" << endl;
  }

  return 0;
}
