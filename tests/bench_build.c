/* Graph build benchmark. From the repository root:
 * gcc -O3 -std=c11 -fopenmp -Iinclude tests/bench_build.c src/knngraph.c src/utils.c src/rp_tree.c -lm -o bench_build
 * ./bench_build 100000 100 30 4 /path/to/contest-data-release-1m.bin
 * Arguments: points dimensions k threads [dataset]. Without a dataset, generate data.
 * Dataset format: native uint32_t point count followed by row-major native floats.
 * Add -Dtime=benchmark_time when compiling for reproducible single-thread comparisons.
 */
#include "utils.h"
#include <sys/resource.h>

time_t benchmark_time(time_t *result) {
	time_t value = 1700000000;
	if (result != NULL)
		*result = value;
	return value;
}

int main(int argc, char **argv) {
	if (argc != 5 && argc != 6) {
		fprintf(stderr, "usage: %s points dimensions k threads [dataset]\n", argv[0]);
		return EXIT_FAILURE;
	}
	uint32_t points = strtoul(argv[1], NULL, 10);
	uint32_t dim = strtoul(argv[2], NULL, 10);
	uint32_t k = strtoul(argv[3], NULL, 10);
	n_threads = strtoul(argv[4], NULL, 10);
	if (points < 2 || !dim || k < 2 || k >= points || !n_threads)
		return EXIT_FAILURE;

	float *values;
	float **data;
	CHECK_CALL(values = malloc((size_t)points * dim * sizeof(*values)), NULL);
	CHECK_CALL(data = malloc(points * sizeof(*data)), NULL);
	for (uint32_t i = 0; i < points; ++i)
		data[i] = values + (size_t)i * dim;
	if (argc == 6) {
		FILE *file;
		CHECK_CALL(file = fopen(argv[5], "rb"), NULL);
		uint32_t available;
		if (fread(&available, sizeof(available), 1, file) != 1 || available < points ||
		    fread(values, sizeof(*values), (size_t)points * dim, file) != (size_t)points * dim)
			return EXIT_FAILURE;
		CHECK_CALL(fclose(file), EOF);
	} else {
		uint32_t seed = 12345;
		for (size_t i = 0; i < (size_t)points * dim; ++i) {
			seed = 1664525U * seed + 1013904223U;
			values[i] = (seed >> 8) * (1.0f / 16777216.0f);
		}
	}

	srand(42);
	double start = omp_get_wtime();
	KNNGraph graph = KNNGraph_create(data, euclidean_dist, k, dim, points);
	KNNGraph_nndescent(graph, 0.001f, 0.5f, 0);
	double elapsed = omp_get_wtime() - start;
	struct rusage usage;
	CHECK_CALL(getrusage(RUSAGE_SELF, &usage), -1);
	double peak_mib = usage.ru_maxrss / (1024.0 * 1024.0);
#ifndef __APPLE__
	peak_mib *= 1024.0;
#endif

	uint64_t checksum = 0;
	for (uint32_t i = 0; i < points; ++i) {
		assert(vector_size(graph->neighbors[i]) == k);
		for (uint32_t j = 0; j < k; ++j) {
			Neighbor entry = graph->neighbors[i][j];
			assert(entry.id < points && entry.id != i);
			assert(entry.dist == euclidean_dist(data[i], data[entry.id], dim));
			if (j)
				assert(graph->neighbors[i][j - 1].dist <= entry.dist);
			for (uint32_t z = 0; z < j; ++z)
				assert(graph->neighbors[i][z].id != entry.id);
			checksum = checksum * 31 + entry.id;
		}
	}

	uint32_t queries = points < 32 ? points : 32;
	uint32_t matches = 0;
	for (uint32_t q = 0; q < queries; ++q) {
		uint32_t id = (uint64_t)q * points / queries;
		Neighbor *exact = vector_create(Neighbor, k);
		for (uint32_t i = 0; i < points; ++i) {
			if (i == id)
				continue;
			float dist = euclidean_dist(data[id], data[i], dim);
			vector_sorted_insert(exact, ((Neighbor){ .id = i, .dist = dist }));
		}
		for (uint32_t i = 0; i < k; ++i)
			for (uint32_t j = 0; j < k; ++j)
				matches += exact[i].id == graph->neighbors[id][j].id;
		vector_destroy(exact);
	}
	printf("points,dim,k,threads,build_seconds,peak_mib,sampled_recall,comparisons,checksum\n");
	printf("%u,%u,%u,%u,%.6f,%.2f,%.6f,%u,%llu\n", points, dim, k, n_threads,
	       elapsed, peak_mib, (double)matches / (queries * k),
	       graph->similarity_comparisons, (unsigned long long)checksum);
	KNNGraph_destroy(graph);
	free(data);
	free(values);
	return EXIT_SUCCESS;
}
