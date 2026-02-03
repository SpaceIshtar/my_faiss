/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Build HNSW index on SIFT1M dataset with exact distances.
 *
 * Usage:
 *   ./build_hnsw_index [data_path] [M] [efConstruction]
 *
 * Default:
 *   data_path: /data/local/embedding_dataset/sift1M
 *   M: 32
 *   efConstruction: 200
 *
 * Output files stored in:
 *   /data/local/embedding_dataset/sift1M/index/hnsw_M{M}_ef{efConstruction}.faissindex
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/stat.h>

#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

/*****************************************************
 * I/O functions for fvecs
 *****************************************************/

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr __attribute__((unused)) = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

/*****************************************************
 * Timing utilities
 *****************************************************/

class Timer {
   public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    void reset() { start_ = std::chrono::high_resolution_clock::now(); }

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

   private:
    std::chrono::high_resolution_clock::time_point start_;
};

void create_directory(const std::string& path) {
    std::string cmd = "mkdir -p " + path;
    int ret = system(cmd.c_str());
    (void)ret;
}

int main(int argc, char* argv[]) {
    std::string data_path = "/data/local/embedding_dataset/sift1M";
    int M = 32;
    int efConstruction = 200;

    if (argc > 1) {
        data_path = argv[1];
    }
    if (argc > 2) {
        M = std::atoi(argv[2]);
    }
    if (argc > 3) {
        efConstruction = std::atoi(argv[3]);
    }

    std::cout << "==================================================" << std::endl;
    std::cout << "HNSW Index Builder (Exact Distances)" << std::endl;
    std::cout << "Data path: " << data_path << std::endl;
    std::cout << "M: " << M << std::endl;
    std::cout << "efConstruction: " << efConstruction << std::endl;
    std::cout << "==================================================" << std::endl;

    // Load data
    std::cout << "\n[Loading data...]" << std::endl;
    size_t d, nb;
    float* xb = fvecs_read((data_path + "/sift_base.fvecs").c_str(), &d, &nb);
    std::cout << "Database: " << nb << " vectors, dimension " << d << std::endl;

    // Create index directory
    std::string index_dir = data_path + "/index";
    create_directory(index_dir);

    // Build HNSW index
    std::cout << "\n[Building HNSW index...]" << std::endl;
    Timer timer;

    faiss::IndexHNSWFlat index(d, M, faiss::METRIC_L2);
    index.hnsw.efConstruction = efConstruction;

    index.add(nb, xb);

    double build_time = timer.elapsed_ms();
    std::cout << "Build time: " << build_time / 1000.0 << " seconds" << std::endl;

    // Save index
    std::string index_path = index_dir + "/hnsw_M" + std::to_string(M) +
                             "_ef" + std::to_string(efConstruction) + ".faissindex";

    std::cout << "\n[Saving index to " << index_path << "...]" << std::endl;
    timer.reset();
    faiss::write_index(&index, index_path.c_str());
    std::cout << "Save time: " << timer.elapsed_ms() / 1000.0 << " seconds" << std::endl;

    std::cout << "\n==================================================" << std::endl;
    std::cout << "HNSW index built successfully!" << std::endl;
    std::cout << "Index file: " << index_path << std::endl;
    std::cout << "==================================================" << std::endl;

    delete[] xb;
    return 0;
}
