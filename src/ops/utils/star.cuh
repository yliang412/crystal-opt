/**
 * @file star.cuh
 * @author Yuchen Liang
 * @brief A configurable benchmark for evaluating performance on star schema
 * with different parameters.
 */

#define CUB_STDERR

#include <iostream>
#include <stdio.h>
#include <curand.h>

#include <cuda.h>
#include <cub/util_allocator.cuh>
#include "cub/test/test_util.h"

#include "crystal/crystal.cuh"

#include "generator.h"
#include "gpu_utils.h"



struct dim_table_t {
    /** The primary key column */
    int *pkey;
    /** The value column */
    int *value;
    /** The number of rows in the table. */
    int n_row;
    /** 
    * Select `sel` number of tuples in the dimension table predicate.
    * The actual predicate is gonna be `t.value <= sel`
    */
    int sel;
};

struct star_schema_t {
    /** The primary key column */
    int *pkey;
    /** The value column. */
    int *value;
    /** The foreign key columns */
    int **fkeys;

    /** The dimension tables */
    int **dim_pkeys;
    int **dim_values;
    int *dim_n_rows;
    int *dim_sels;

    /** Dim hash tables and bloom filters */
    int **htables;
    uint32_t **blooms;

    /** The number of rows in the fact table. */
    int n_row;
    int n_dim_table;

    // Configs
    int bloom_factor;
    int htable_factor;
};


int star_table_add_dims_host(star_schema_t *fact, int n_dim_table, int *dim_n_rows, int *dim_sels) {
    fact->n_dim_table = n_dim_table;
    fact->fkeys = (int**)malloc(sizeof(int *) * n_dim_table);
    fact->dim_pkeys = (int **)malloc(sizeof(int *) * n_dim_table);
    fact->dim_values = (int **)malloc(sizeof(int *) * n_dim_table);
    fact->dim_n_rows = (int *)malloc(sizeof(int) * n_dim_table);
    fact->dim_sels = (int *)malloc(sizeof(int) * n_dim_table);

    if (fact->fkeys == NULL || fact->dim_pkeys == NULL || fact->dim_values == NULL || fact->dim_n_rows == NULL || fact->dim_sels == NULL) {
        return -1;
    }

    for (int i = 0; i < n_dim_table; i++) {
        fact->dim_n_rows[i] = dim_n_rows[i];
        fact->dim_sels[i] = dim_sels[i];
        if (create_relation_pk(fact->dim_pkeys[i], fact->dim_values[i], fact->dim_n_rows[i]) < 0) {
            return -1;
        }
        if (create_relation_fk_only(fact->fkeys[i], fact->n_row, fact->dim_n_rows[i]) < 0) {
            return -1;
        }
    }
    return 0;
}


 
/**
* @brief Initializes the fact table.
* 
* @param fact Pointer to the fact table.
* @param n_row number of rows for the fact table.
* @return 0 on success, -1 on failure.
*/
int star_table_init_host(star_schema_t *fact, int n_fact, int n_dim_table, int *n_dims, int *dim_sels, int bloom_factor, int htable_factor) {
    fact->n_row = n_fact;
    fact->n_dim_table = 0;
    fact->bloom_factor = bloom_factor;
    fact->htable_factor = htable_factor;
    if (create_relation_pk(fact->pkey, fact->value, fact->n_row) < 0) {
        return -1;
    }

    if (star_table_add_dims_host(fact, n_dim_table, n_dims, dim_sels) < 0) {
        return -1;
    }

    return 0;
}

int star_alloc_htable_and_bloom(star_schema_t *device, cub::CachingDeviceAllocator &alloc) {
    int** htables = (int **)malloc(sizeof(int *) * device->n_dim_table);
    uint32_t **blooms = (uint32_t **)malloc(sizeof(uint32_t *) * device->n_dim_table);
    if (htables == NULL || blooms == NULL) {
        printf("Device star schema:\n");
        return -1;
    }

    for (int i = 0; i < device->n_dim_table; i++) {
        int n_row = device->dim_n_rows[i];
        int htable_nbytes = n_row * device->htable_factor  * sizeof(int);
        int bloom_nbytes = n_row * device->bloom_factor / 8;

        CubDebugExit(alloc.DeviceAllocate((void**)&htables[i], htable_nbytes));
        CubDebugExit(alloc.DeviceAllocate((void**)&blooms[i], bloom_nbytes));
    }
    CubDebugExit(alloc.DeviceAllocate((void**)&device->htables, sizeof(int *) * device->n_dim_table));
    CubDebugExit(alloc.DeviceAllocate((void**)&device->blooms, sizeof(uint32_t *) * device->n_dim_table));
    CubDebugExit(cudaMemcpy(device->htables, htables, sizeof(int *) * device->n_dim_table, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(device->blooms, blooms, sizeof(uint32_t *) * device->n_dim_table, cudaMemcpyHostToDevice));
    // Free temp host memory
    free(htables);
    free(blooms);
    return 0;
}

int star_copy_to_device(star_schema_t *device, star_schema_t *host, cub::CachingDeviceAllocator &alloc) {
    
    device->n_row = host->n_row;
    device->n_dim_table = host->n_dim_table;
    device->bloom_factor = host->bloom_factor;
    device->htable_factor = host->htable_factor;
    
    // Allocate device memory
    CubDebugExit(alloc.DeviceAllocate((void**)&device->pkey, sizeof(int) * host->n_row));
    CubDebugExit(alloc.DeviceAllocate((void**)&device->value, sizeof(int) * host->n_row));
    CubDebugExit(alloc.DeviceAllocate((void**)&device->dim_n_rows, sizeof(int) * host->n_dim_table));
    CubDebugExit(alloc.DeviceAllocate((void**)&device->dim_sels, sizeof(int) * host->n_dim_table));


    // Copy the fact table data to device
    CubDebugExit(cudaMemcpy(device->pkey, host->pkey, sizeof(int) * host->n_row, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(device->value, host->value, sizeof(int) * host->n_row, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(device->dim_n_rows, host->dim_n_rows, sizeof(int) * host->n_dim_table, cudaMemcpyHostToDevice));
    CubDebugExit(cudaMemcpy(device->dim_sels, host->dim_sels, sizeof(int) * host->n_dim_table, cudaMemcpyHostToDevice));
    
    int **fkeys = (int**)malloc(sizeof(int *) * host->n_dim_table);
    int **dim_pkeys = (int **)malloc(sizeof(int *) * host->n_dim_table);
    int **dim_values = (int **)malloc(sizeof(int *) * host->n_dim_table);
       
    if (fkeys == NULL || dim_pkeys == NULL || dim_values == NULL) {
        return -1;
    }

    for (int i = 0; i < host->n_dim_table; i++) {
        // Allocate device memory
        // Copy the dimension tables data to device
        CubDebugExit(alloc.DeviceAllocate((void**)&dim_pkeys[i], sizeof(int) * host->dim_n_rows[i]));
        CubDebugExit(cudaMemcpy(dim_pkeys[i], host->dim_pkeys[i], sizeof(int) * host->dim_n_rows[i], cudaMemcpyHostToDevice));
        CubDebugExit(alloc.DeviceAllocate((void**)&dim_values[i], sizeof(int) * host->dim_n_rows[i]));
        CubDebugExit(cudaMemcpy(dim_values[i], host->dim_values[i], sizeof(int) * host->dim_n_rows[i], cudaMemcpyHostToDevice));
        CubDebugExit(alloc.DeviceAllocate((void**)&fkeys[i], sizeof(int) * host->n_row));
        CubDebugExit(cudaMemcpy(fkeys[i], host->fkeys[i], sizeof(int) * host->n_row, cudaMemcpyHostToDevice));
    }
    
    CubDebugExit(alloc.DeviceAllocate((void**)&device->fkeys, sizeof(int *) * host->n_dim_table));
    CubDebugExit(cudaMemcpy(device->fkeys, fkeys, sizeof(int *) * host->n_dim_table, cudaMemcpyHostToDevice));
    CubDebugExit(alloc.DeviceAllocate((void**)&device->dim_pkeys, sizeof(int *) * host->n_dim_table));
    CubDebugExit(cudaMemcpy(device->dim_pkeys, dim_pkeys, sizeof(int *) * host->n_dim_table, cudaMemcpyHostToDevice));
    CubDebugExit(alloc.DeviceAllocate((void**)&device->dim_values, sizeof(int *) * host->n_dim_table));
    CubDebugExit(cudaMemcpy(device->dim_values, dim_values, sizeof(int *) * host->n_dim_table, cudaMemcpyHostToDevice));

    // Free temp host memory
    free(fkeys);
    free(dim_pkeys);
    free(dim_values);


    if (star_alloc_htable_and_bloom(device, alloc) < 0) {
        return -1;
    }

    return 0;
}



int star_cleanup_device(star_schema_t *device, cub::CachingDeviceAllocator &alloc) {
    // Free device memory
    int **fkeys = (int**)malloc(sizeof(int *) * device->n_dim_table);
    int **dim_pkeys = (int **)malloc(sizeof(int *) * device->n_dim_table);
    int **dim_values = (int **)malloc(sizeof(int *) * device->n_dim_table);

    if (fkeys == NULL || dim_pkeys == NULL || dim_values == NULL) {
        return -1;
    }

   
    CubDebugExit(cudaMemcpy(fkeys, device->fkeys, device->n_dim_table * sizeof(int*), cudaMemcpyDeviceToHost));
    CubDebugExit(cudaMemcpy(dim_pkeys, device->dim_pkeys, device->n_dim_table * sizeof(int*), cudaMemcpyDeviceToHost));
    CubDebugExit(cudaMemcpy(dim_values, device->dim_values, device->n_dim_table * sizeof(int*), cudaMemcpyDeviceToHost));
    alloc.DeviceFree(device->fkeys);
    alloc.DeviceFree(device->dim_pkeys);
    alloc.DeviceFree(device->dim_values);


    int** htables = (int **)malloc(sizeof(int *) * device->n_dim_table);
    uint32_t **blooms = (uint32_t **)malloc(sizeof(uint32_t *) * device->n_dim_table);

    if (htables == NULL || blooms == NULL) {
        return -1;
    }
    CubDebugExit(cudaMemcpy(htables, device->htables, device->n_dim_table * sizeof(int*), cudaMemcpyDeviceToHost));
    CubDebugExit(cudaMemcpy(blooms, device->blooms, device->n_dim_table * sizeof(uint32_t*), cudaMemcpyDeviceToHost));
    alloc.DeviceFree(device->htables);
    alloc.DeviceFree(device->blooms);

    // Free the bloom filters and hash tables
    for (int i = 0; i < device->n_dim_table; i++) {
        alloc.DeviceFree(htables[i]);
        alloc.DeviceFree(blooms[i]);
    }
    free(htables);
    free(blooms);

    for (int i = 0; i < device->n_dim_table; i++) {
        alloc.DeviceFree(fkeys[i]);
        alloc.DeviceFree(dim_pkeys[i]);
        alloc.DeviceFree(dim_values[i]);
    }
    free(fkeys);
    free(dim_pkeys);
    free(dim_values);

    
      
    alloc.DeviceFree(device->pkey);
    alloc.DeviceFree(device->value);
    alloc.DeviceFree(device->dim_n_rows);
    alloc.DeviceFree(device->dim_sels);

  
    return 0;
}




int star_table_free_host(star_schema_t *fact) {
    free(fact->pkey);
    free(fact->value);
    for (int i = 0; i < fact->n_dim_table; i++) {
        free(fact->dim_pkeys[i]);
        free(fact->dim_values[i]);
        free(fact->fkeys[i]);
    }
    free(fact->dim_pkeys);
    free(fact->dim_values);
    free(fact->dim_n_rows);
    free(fact->dim_sels);
    free(fact->fkeys);
    return 0;
}
 
void print_dim_table(star_schema_t *fact, int i) {
    int n_row = fact->dim_n_rows[i];
    int *pkey = fact->dim_pkeys[i];
    int *value = fact->dim_values[i];
    printf("dim_%d(n=%d): sel=`v <= %d`\n", i, n_row, fact->dim_sels[i]);
    printf("-------------------------------------\n");
    for (int i = 0; i < n_row; i++) {
        printf("%d,%d\n",pkey[i], value[i]);
    }
}
 
 
void print_star_schema(star_schema_t *fact) {
     printf("=====================================\n");
     printf("[schema_config]\nbloom_factor=%d\nhtable_factor=%d\n", fact->bloom_factor, fact->htable_factor);
     printf("=====================================\n");
     printf("fact(n=%d):\n", fact->n_row);
     printf("-------------------------------------\n");
     for (int i = 0; i < fact->n_row; i++) {
         printf("%d", fact->pkey[i]);
         for (int j = 0; j < fact->n_dim_table; j++) {
             printf(",%d", fact->fkeys[j][i]);
         }
         printf("\n");
     }
     for (int i = 0; i < fact->n_dim_table; i++) {
         printf("=====================================\n");
         print_dim_table(fact, i);
     }
}


