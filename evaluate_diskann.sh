# ./build/experiments/DiskANN/bench_diskann_standard --dataset bigann --config-dir ./experiments/DiskANN/config/
# ./build/experiments/DiskANN/bench_diskann_quant --dataset bigann --config-dir ./experiments/DiskANN/config/ --algorithm sq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset bigann --config-dir ./experiments/DiskANN/config/ --algorithm pq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset bigann --config-dir ./experiments/DiskANN/config/ --algorithm osq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset bigann --config-dir ./experiments/DiskANN/config/ --algorithm opq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset bigann --config-dir ./experiments/DiskANN/config/ --algorithm rabitq


# ./build/experiments/DiskANN/bench_diskann_quant --dataset wiki --config-dir ./experiments/DiskANN/config/ --algorithm sq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset wiki --config-dir ./experiments/DiskANN/config/ --algorithm pq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset wiki --config-dir ./experiments/DiskANN/config/ --algorithm osq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset wiki --config-dir ./experiments/DiskANN/config/ --algorithm opq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset wiki --config-dir ./experiments/DiskANN/config/ --algorithm rabitq

# ./build/experiments/DiskANN/build_diskann_index --dataset laion --config-dir ./experiments/DiskANN/config/
# ./build/experiments/DiskANN/bench_diskann_quant --dataset laion --config-dir ./experiments/DiskANN/config/ --algorithm sq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset laion --config-dir ./experiments/DiskANN/config/ --algorithm pq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset laion --config-dir ./experiments/DiskANN/config/ --algorithm osq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset laion --config-dir ./experiments/DiskANN/config/ --algorithm opq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset laion --config-dir ./experiments/DiskANN/config/ --algorithm rabitq

# ./build/experiments/hnsw/bench_hnsw_quant --dataset sift1M --algorithm rabi?tq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset gist1M --algorithm rabitq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset audio --algorithm rabitq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset video --algorithm rabitq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset text2image --algorithm rabitq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset paper --algorithm rabitq --config-dir ./experiments/hnsw/config/

# ./build/experiments/hnsw/bench_hnsw_rabitq_native --dataset sift1M --config-dir ./experiments/hnsw/config/ --rerank
./build/experiments/hnsw/bench_hnsw_rabitq_native --dataset gist1M --config-dir ./experiments/hnsw/config/ --rerank
./build/experiments/hnsw/bench_hnsw_rabitq_native --dataset audio --config-dir ./experiments/hnsw/config/ --rerank
./build/experiments/hnsw/bench_hnsw_rabitq_native --dataset video --config-dir ./experiments/hnsw/config/ --rerank
./build/experiments/hnsw/bench_hnsw_rabitq_native --dataset text2image --config-dir ./experiments/hnsw/config/ --rerank
./build/experiments/hnsw/bench_hnsw_rabitq_native --dataset paper --config-dir ./experiments/hnsw/config/ --rerank

./build/experiments/DiskANN/bench_diskann_quant --dataset bigann --config-dir ./experiments/DiskANN/config/ --algorithm rabitq
./build/experiments/DiskANN/bench_diskann_quant --dataset wiki --config-dir ./experiments/DiskANN/config/ --algorithm rabitq
./build/experiments/DiskANN/bench_diskann_quant --dataset laion --config-dir ./experiments/DiskANN/config/ --algorithm rabitq
