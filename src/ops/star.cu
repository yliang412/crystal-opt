/**
 * @file star.cu
 * @author Yuchen Liang
 * @brief A configurable benchmark for evaluating performance on star schema
 * with different parameters.
 */

#include "utils/star.cuh"

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
    return 0;
}




