/**
 * @file star.cu
 * @author Yuchen Liang
 * @brief A configurable benchmark for evaluating performance on star schema
 * with different parameters.
 */

#include "utils/star.cuh"

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose = false;  // Whether to display input/output to console
cub::CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory



int main(int argc, char** argv) {

    CommandLineArgs args(argc, argv);
    CubDebugExit(args.DeviceInit());
    
    int n_fact = 10;
    int n_dim_table = 1;
    int n_dims[] = {2};
    int dim_sels[] = {1};
    int bloom_factor = 2;
    int htable_factor = 2;
    star_schema_t host;
    star_table_init_host(&host, n_fact, n_dim_table, n_dims, dim_sels, bloom_factor, htable_factor);
    print_star_schema(&host); 
    star_schema_t device;
    star_copy_to_device(&device, &host, g_allocator);
    star_cleanup_device(&device, g_allocator);
    star_table_free_host(&host);
    return 0;
}




