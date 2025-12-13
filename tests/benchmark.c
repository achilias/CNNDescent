#include "acutest.h"
#include "utils.h"
#include <sys/time.h>

#define BENCHMARK_POINTS 1000
#define BENCHMARK_DIM 128
#define BENCHMARK_K 10

/* Helper function to measure time in milliseconds */
static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

void benchmark_nndescent(void) {
    printf("\n=== NN-Descent Benchmark ===\n");
    printf("Dataset: %d points, %d dimensions, k=%d\n", 
           BENCHMARK_POINTS, BENCHMARK_DIM, BENCHMARK_K);
    
    /* Generate random data */
    float **data = malloc(BENCHMARK_POINTS * sizeof(float*));
    for (int i = 0; i < BENCHMARK_POINTS; i++) {
        data[i] = malloc(BENCHMARK_DIM * sizeof(float));
        for (int j = 0; j < BENCHMARK_DIM; j++) {
            data[i][j] = ((float)rand()) / RAND_MAX;
        }
    }
    
    /* Test with Euclidean distance */
    printf("\nTesting with Euclidean distance:\n");
    KNNGraph graph_euclidean = KNNGraph_create(data, euclidean_dist, 
                                                BENCHMARK_K, BENCHMARK_DIM, 
                                                BENCHMARK_POINTS);
    
    double start = get_time_ms();
    KNNGraph_nndescent(graph_euclidean, 0.001, 0.5, 10);
    double end = get_time_ms();
    
    printf("  Build time: %.2f ms\n", end - start);
    printf("  Distance computations: %u\n", graph_euclidean->similarity_comparisons);
    
    /* Test with Manhattan distance */
    printf("\nTesting with Manhattan distance:\n");
    KNNGraph graph_manhattan = KNNGraph_create(data, manhattan_dist, 
                                                BENCHMARK_K, BENCHMARK_DIM, 
                                                BENCHMARK_POINTS);
    
    start = get_time_ms();
    KNNGraph_nndescent(graph_manhattan, 0.001, 0.5, 10);
    end = get_time_ms();
    
    printf("  Build time: %.2f ms\n", end - start);
    printf("  Distance computations: %u\n", graph_manhattan->similarity_comparisons);
    
    /* Cleanup */
    KNNGraph_destroy(graph_euclidean);
    KNNGraph_destroy(graph_manhattan);
    for (int i = 0; i < BENCHMARK_POINTS; i++) {
        free(data[i]);
    }
    free(data);
    
    TEST_ASSERT(1); /* Always pass, this is just a benchmark */
}

void benchmark_distance_functions(void) {
    printf("\n=== Distance Function Benchmark ===\n");
    
    const int iterations = 1000000;
    float a[BENCHMARK_DIM], b[BENCHMARK_DIM];
    
    /* Initialize test vectors */
    for (int i = 0; i < BENCHMARK_DIM; i++) {
        a[i] = ((float)rand()) / RAND_MAX;
        b[i] = ((float)rand()) / RAND_MAX;
    }
    
    /* Benchmark Euclidean distance */
    double start = get_time_ms();
    volatile float result = 0.0f;
    for (int i = 0; i < iterations; i++) {
        result += euclidean_dist(a, b, BENCHMARK_DIM);
    }
    double end = get_time_ms();
    
    printf("Euclidean distance (%d iterations): %.2f ms\n", iterations, end - start);
    
    /* Benchmark Manhattan distance */
    result = 0.0f;  /* Reset result for accurate benchmarking */
    start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        result += manhattan_dist(a, b, BENCHMARK_DIM);
    }
    end = get_time_ms();
    
    printf("Manhattan distance (%d iterations): %.2f ms\n", iterations, end - start);
    
    TEST_ASSERT(1); /* Always pass, this is just a benchmark */
}

TEST_LIST = {
    { "benchmark_nndescent",          benchmark_nndescent          },
    { "benchmark_distance_functions", benchmark_distance_functions },
    { NULL,                           NULL                         }
};
