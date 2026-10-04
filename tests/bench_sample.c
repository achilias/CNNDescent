/* Standalone sampling benchmark (not part of the unit test suite).
 * From the repository root:
 * gcc -O3 -std=c11 -fopenmp -Iinclude tests/bench_sample.c src/utils.c src/rp_tree.c -lm -o sample_bench
 * ./sample_bench
 */
#include "utils.h"

int main(void) {
	const uint32_t lengths[] = { 32, 1024, 65536 };
	const uint32_t percentages[] = { 0, 10, 50, 99 };
	uint64_t checksum = 0;

	printf("length,retained_percent,iterations,ns_per_sample\n");
	for (size_t n = 0; n < ARRAY_SIZE(lengths); ++n) {
		uint32_t length = lengths[n];
		uint32_t iterations = (1U << 24) / length;
		uint32_t *source;
		CHECK_CALL(source = malloc(length * sizeof(*source)), NULL);
		for (uint32_t i = 0; i < length; ++i)
			source[i] = i;

		for (size_t p = 0; p < ARRAY_SIZE(percentages); ++p) {
			uint32_t size = length * percentages[p] / 100;
			uint32_t *vec = vector_create(uint32_t, length);
			clock_t start = clock();
			for (uint32_t i = 0; i < iterations; ++i) {
				/* Include refilling so both implementations see identical input. */
				memcpy(vec, source, length * sizeof(*vec));
				vector_set_size(vec, length);
				sample(&vec, size, i);
				if (size)
					checksum += vec[size - 1];
			}
			double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
			printf("%u,%u,%u,%.1f\n", length, percentages[p], iterations,
			       elapsed * 1e9 / iterations);
			vector_destroy(vec);
		}
		free(source);
	}

	fprintf(stderr, "checksum: %llu\n", (unsigned long long)checksum);
	return 0;
}
