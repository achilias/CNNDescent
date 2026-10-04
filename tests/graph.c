#include "acutest.h"
#include "utils.h"


float *test_data[5] = {	
	(float[]){ 1.0, 1.0 },
	(float[]){ 2.0, 2.0 },
	(float[]){ 3.0, 3.0 },
	(float[]){ 3.0, 4.0 },
	(float[]){ 4.0, 3.0 }
};

/*  For above data: (using manhattan distance metric)
	Neighbors are:
	0: 1, 2, 3
	1: 0, 2, 3
	2: 3, 4, 1
	3: 2, 4, 1
	4: 2, 3, 1 
*/

int test_data_neighbors[][5] = {
	{ 1, 2, 3 },
	{ 0, 2, 3 },
	{ 3, 4, 1 },
	{ 2, 4, 1 },
	{ 2, 3, 1 }
};

/*
	Reverse neighbors:
	0: 1
	1: 0, 2, 3, 4
	2: 0, 1, 3, 4
	3: 0, 1, 2, 4
	4: 2, 3
*/

void test_create(void) {
	uint32_t points = 5;
	uint32_t dim = 2;
	uint32_t k = 3;

	KNNGraph graph = KNNGraph_create(test_data, manhattan_dist, k, dim, points);
	TEST_ASSERT(graph->k == k);
	TEST_ASSERT(graph->points == points);
	TEST_ASSERT(graph->dim == dim);
	TEST_ASSERT(graph->similarity_comparisons == 0);
	for (uint32_t i = 0; i < points; ++i)
		TEST_CHECK(graph->neighbors[i] == NULL);

	KNNGraph_destroy(graph);
}

void test_export_import_graph(void) {
	uint32_t points = 5;
	uint32_t dim = 2;
	uint32_t k = 3;

	/* Export graph: N, dim, K, K Neighbors for node[0], K Neighbors for node[1],... */
	KNNGraph export_graph = KNNGraph_create(test_data, euclidean_dist, k, dim, points);
	KNNGraph_bruteforce(export_graph);
	KNNGraph_export_graph(export_graph, "test_data.bin");

	/* Import graph */
	KNNGraph import_graph = KNNGraph_import_graph("test_data.bin", test_data, euclidean_dist);

	/* Check that imported graph is correct */
	TEST_ASSERT(import_graph->k == k);
	TEST_ASSERT(import_graph->dim == dim);
	TEST_ASSERT(import_graph->points == points);
	for (size_t i = 0UL; i < points; ++i) {
		for (size_t j = 0UL; j < k; ++j) {
			TEST_ASSERT(import_graph->neighbors[i][0].id == test_data_neighbors[i][j]);
			vector_delete(import_graph->neighbors[i], 0);
		}
	}
	KNNGraph_destroy(import_graph);
	KNNGraph_destroy(export_graph);

    CHECK_CALL(remove("test_data.bin"), -1);
}

void test_bruteforce(void) {
	uint32_t points = 5;
	uint32_t dim = 2;
	uint32_t k = 3;

	KNNGraph graph = KNNGraph_create(test_data, manhattan_dist, k, dim, points);

	/* Build KNN graph with exhaustive search and check that the result 
		is the same as the (manually calculated) expected result */

	KNNGraph_bruteforce(graph);
	for (size_t i = 0UL; i < points; ++i) {
		for (size_t j = 0UL; j < k; ++j) {
			TEST_ASSERT(graph->neighbors[i][0].id == test_data_neighbors[i][j]);
			vector_delete(graph->neighbors[i], 0);
		}
	}
	KNNGraph_destroy(graph);
}

void test_recall(void) {
	uint32_t points = 5;
	uint32_t dim = 2;
	uint32_t k = 3;

	KNNGraph graph = KNNGraph_create(test_data, manhattan_dist, k, dim, points);
	KNNGraph graph_ = KNNGraph_create(test_data, manhattan_dist, k, dim, points);

	KNNGraph_bruteforce(graph);
	KNNGraph_bruteforce(graph_);

	/* Graphs are the exact same */
	TEST_ASSERT(fabs(KNNGraph_recall(graph, graph_) - 1.0) < 1e-9);

	KNNGraph_destroy(graph);
	KNNGraph_destroy(graph_);
}


void test_nndescent(void) {
	uint32_t points = 5;
	uint32_t dim = 2;
	uint32_t k = 3;

	KNNGraph graph = KNNGraph_create(test_data, manhattan_dist, k, dim, points);
	KNNGraph ground_truth = KNNGraph_create(test_data, manhattan_dist, k, dim, points);

	KNNGraph_nndescent(graph, 0.0, 1.0, 0);
	KNNGraph_bruteforce(ground_truth);

	/* Build KNN graph with both exhaustive search and NNDescent and check that their results 
	   are at least 90% similar (with this trivial dataset, they are the exact same) */

	TEST_ASSERT(KNNGraph_recall(graph, ground_truth) > 0.9);

	KNNGraph_destroy(graph);
	KNNGraph_destroy(ground_truth);
}

void test_knearest(void) {
	uint32_t points = 5;
	uint32_t dim = 2;
	uint32_t k = 3;

	KNNGraph graph = KNNGraph_create(test_data, manhattan_dist, k, dim, points);
	KNNGraph_bruteforce(graph);

	for (size_t i = 0UL; i < points; ++i) {
		Neighbor *neighbors = KNNGraph_KNearest(graph, test_data[i]);
		for (size_t j = 0UL; j < k; ++j)	
			TEST_ASSERT(neighbors[j].id == test_data_neighbors[i][j]);
		free(neighbors);
	}
	KNNGraph_destroy(graph);
	
}

void test_rebuild(void) {
	KNNGraph graph = KNNGraph_create(test_data, manhattan_dist, 3, 2, 5);
	KNNGraph_bruteforce(graph);
	Neighbor *storage[5];
	Neighbor expected[5][3];
	for (uint32_t i = 0; i < 5; ++i) {
		storage[i] = graph->neighbors[i];
		memcpy(expected[i], storage[i], sizeof(expected[i]));
	}

	uint32_t saved_threads = n_threads;
	n_threads = 2;
	for (uint32_t run = 0; run < 3; ++run) {
		KNNGraph_nndescent(graph, 0.0f, 1.0f, 0);
		for (uint32_t i = 0; i < 5; ++i) {
			TEST_CHECK(graph->neighbors[i] == storage[i]);
			TEST_ASSERT(vector_size(storage[i]) == graph->k);
			for (uint32_t j = 0; j < graph->k; ++j) {
				Neighbor entry = storage[i][j];
				TEST_ASSERT(entry.id < graph->points && entry.id != i);
				TEST_CHECK(entry.dist == manhattan_dist(test_data[i], test_data[entry.id], 2));
				if (j)
					TEST_CHECK(storage[i][j - 1].dist <= entry.dist);
				for (uint32_t z = 0; z < j; ++z)
					TEST_CHECK(storage[i][z].id != entry.id);
			}
		}
		KNNGraph_bruteforce(graph);
		for (uint32_t i = 0; i < 5; ++i) {
			TEST_CHECK(graph->neighbors[i] == storage[i]);
			TEST_ASSERT(vector_size(storage[i]) == graph->k);
			for (uint32_t j = 0; j < graph->k; ++j) {
				TEST_CHECK(storage[i][j].id == expected[i][j].id);
				TEST_CHECK(storage[i][j].dist == expected[i][j].dist);
			}
		}
	}
	n_threads = saved_threads;
	graph_init_rp(graph, 3);
	for (uint32_t i = 0; i < 5; ++i)
		TEST_CHECK(graph->neighbors[i] == storage[i]);
	graph->k = 2;
	KNNGraph_bruteforce(graph);
	for (uint32_t i = 0; i < 5; ++i) {
		TEST_CHECK(vector_cap(graph->neighbors[i]) == 2);
		TEST_CHECK(vector_size(graph->neighbors[i]) == 2);
	}
	KNNGraph_destroy(graph);
}

void test_nndescent_large(void) {
	float values[512][5];
	float *data[512];
	uint32_t seed = 12345;
	for (uint32_t i = 0; i < ARRAY_SIZE(data); ++i) {
		data[i] = values[i];
		for (uint32_t j = 0; j < 5; ++j) {
			seed = 1664525U * seed + 1013904223U;
			values[i][j] = (seed >> 8) * (1.0f / 16777216.0f);
		}
	}
	KNNGraph truth = KNNGraph_create(data, euclidean_dist, 24, 5, ARRAY_SIZE(data));
	KNNGraph_bruteforce(truth);
	const uint32_t threads[] = { 1, 4 };
	const float rates[] = { 0.5f, 1.0f };
	uint32_t saved_threads = n_threads;
	for (size_t t = 0; t < ARRAY_SIZE(threads); ++t) {
		n_threads = threads[t];
		for (size_t r = 0; r < ARRAY_SIZE(rates); ++r) {
			KNNGraph graph = KNNGraph_create(data, euclidean_dist, 24, 5, ARRAY_SIZE(data));
			KNNGraph_nndescent(graph, 0.001f, rates[r], 0);
			TEST_CHECK(KNNGraph_recall(graph, truth) > 0.98f);
			TEST_CHECK(graph->similarity_comparisons > 0);
			for (uint32_t i = 0; i < graph->points; ++i) {
				Neighbor *neighbors = graph->neighbors[i];
				TEST_ASSERT(vector_size(neighbors) == graph->k);
				for (uint32_t j = 0; j < graph->k; ++j) {
					Neighbor entry = neighbors[j];
					TEST_ASSERT(entry.id < graph->points && entry.id != i);
					TEST_CHECK(entry.dist == euclidean_dist(data[i], data[entry.id], graph->dim));
					if (j)
						TEST_CHECK(neighbors[j - 1].dist <= entry.dist);
					for (uint32_t z = 0; z < j; ++z)
						TEST_CHECK(neighbors[z].id != entry.id);
				}
			}
			KNNGraph_destroy(graph);
		}
	}
	n_threads = saved_threads;
	KNNGraph_destroy(truth);
}

void test_rebuild_imported(void) {
	KNNGraph graph = KNNGraph_create(test_data, euclidean_dist, 3, 2, 5);
	KNNGraph_bruteforce(graph);
	KNNGraph_export_graph(graph, "test_rebuild.bin");
	KNNGraph imported = KNNGraph_import_graph("test_rebuild.bin", test_data, euclidean_dist);
	Neighbor *storage[5];
	for (uint32_t i = 0; i < 5; ++i)
		storage[i] = imported->neighbors[i];
	KNNGraph_bruteforce(imported);
	TEST_CHECK(KNNGraph_recall(imported, graph) == 1.0f);
	for (uint32_t i = 0; i < 5; ++i)
		TEST_CHECK(imported->neighbors[i] == storage[i]);
	KNNGraph_destroy(imported);
	KNNGraph_destroy(graph);
	CHECK_CALL(remove("test_rebuild.bin"), -1);
}

void test_add_points(void) {
	KNNGraph graph = KNNGraph_create(NULL, manhattan_dist, 3, 2, 0);
	for (uint32_t i = 0; i < 5; ++i) {
		KNNGraph_add_point(graph, test_data[i]);
		TEST_CHECK(graph->neighbors[i] == NULL);
	}
	KNNGraph_bruteforce(graph);
	for (uint32_t i = 0; i < 5; ++i) {
		TEST_ASSERT(vector_size(graph->neighbors[i]) == graph->k);
		for (uint32_t j = 0; j < graph->k; ++j)
			TEST_CHECK(graph->neighbors[i][j].id == (uint32_t)test_data_neighbors[i][j]);
	}
	/* The existing API leaves data storage cleanup to the caller. */
	float **data = graph->data;
	KNNGraph_destroy(graph);
	for (uint32_t i = 0; i < 5; ++i)
		free(data[i]);
	vector_destroy(data);
}

TEST_LIST = {
	{ "test_create",			  test_create              },
	{ "test_export_import_graph", test_export_import_graph },
	{ "test_bruteforce", 		  test_bruteforce          },
	{ "test_recall",			  test_recall              },
	{ "test_nndescent", 		  test_nndescent           },
	{ "test_knearest",			  test_knearest            },
	{ "test_rebuild",             test_rebuild             },
	{ "test_rebuild_imported",    test_rebuild_imported    },
	{ "test_add_points",          test_add_points          },
	{ "test_nndescent_large",     test_nndescent_large     },
	{  NULL,                      NULL                     }
};
