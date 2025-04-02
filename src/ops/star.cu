/**
 * @file star.cu
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

#include "utils/generator.h"
#include "utils/gpu_utils.h"

#include "utils/generator.h"
#include "utils/gpu_utils.h"

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

struct fact_table_t {
    /** The primary key column */
    int *pkey;
    /** The value column. */
    int *value;
    /** The foreign key columns */
    int **fkeys;

    /** The dimension tables */
    dim_table_t *dims;
    /** The number of rows in the fact table. */
    int n_row;
    int n_dim_table;

    // Configs
    int bloom_nbytes;
    int htable_factor;
};


int fact_table_add_dims_host(fact_table_t *fact, int n_dim_table, int *n_dims, int *dim_sels) {
    fact->n_dim_table = n_dim_table;
    fact->dims = (dim_table_t *)malloc(sizeof(dim_table_t) * n_dim_table);
    if (fact->dims == NULL) {
        return -1;
    }

    fact->fkeys = (int**)malloc(sizeof(int *) * n_dim_table);

    for (int i = 0; i < n_dim_table; i++) {
        dim_table_t *dim = &fact->dims[i];
        dim->n_row = n_dims[i];
        dim->sel = dim_sels[i];
        create_relation_pk(dim->pkey, dim->value, dim->n_row);
        create_relation_fk_only(fact->fkeys[i], fact->n_row, dim->n_row);
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
int fact_table_init_host(fact_table_t *fact, int n_fact, int n_dim_table, int *n_dims, int *dim_sels, int bloom_nbytes, int htable_factor) {
    fact->n_row = n_fact;
    fact->n_dim_table = 0;
    fact->bloom_nbytes = bloom_nbytes;
    fact->htable_factor = htable_factor;
    if (create_relation_pk(fact->pkey, fact->value, fact->n_row) < 0) {
        return -1;
    }

    if (fact_table_add_dims_host(fact, n_dim_table, n_dims, dim_sels) < 0) {
        return -1;
    }

    return 0;
}


int fact_table_free_host(fact_table_t *fact) {
    free(fact->pkey);
    free(fact->value);
    for (int i = 0; i < fact->n_dim_table; i++) {
        dim_table_t *dim = &fact->dims[i];
        free(dim->pkey);
        free(dim->value);
        free(fact->fkeys[i]);
    }
    free(fact->dims);
    free(fact->fkeys);
    return 0;
}


void print_dim_table(dim_table_t *dim, int i) {
    printf("dim_%d(n=%d): sel=`v <= %d`\n", i, dim->n_row, dim->sel);
    printf("-------------------------------------\n");
    for (int i = 0; i < dim->n_row; i++) {
        printf("%d,%d\n", dim->pkey[i], dim->value[i]);
    }
}


void print_star_schema(fact_table_t *fact) {
    printf("=====================================\n");
    printf("[schema_config]\nbloom_nbytes=%d\nhtable_factor=%d\n", fact->bloom_nbytes, fact->htable_factor);
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
        print_dim_table(&fact->dims[i], i);
    }
}


int main(int argc, char** argv) {
    int n_fact = 10;
    int n_dim_table = 1;
    int n_dims[] = {2};
    int dim_sels[] = {1};
    int bloom_nbytes = 1024;
    int htable_factor = 2;
    fact_table_t fact;
    fact_table_init_host(&fact, n_fact, n_dim_table, n_dims, dim_sels, bloom_nbytes, htable_factor);
    print_star_schema(&fact); 

    fact_table_free_host(&fact);
}




