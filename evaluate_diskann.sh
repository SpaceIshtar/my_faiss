# ./build/experiments/DiskANN/bench_diskann_standard --dataset bigann --config-dir ./experiments/DiskANN/config/
# ./build/experiments/DiskANN/bench_diskann_quant --dataset bigann --config-dir ./experiments/DiskANN/config/ --algorithm sq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset bigann --config-dir ./experiments/DiskANN/config/ --algorithm pq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset bigann --config-dir ./experiments/DiskANN/config/ --algorithm osq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset bigann --config-dir ./experiments/DiskANN/config/ --algorithm opq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset bigann --config-dir ./experiments/DiskANN/config/ --algorithm saq


# ./build/experiments/DiskANN/bench_diskann_quant --dataset wiki --config-dir ./experiments/DiskANN/config/ --algorithm sq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset wiki --config-dir ./experiments/DiskANN/config/ --algorithm pq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset wiki --config-dir ./experiments/DiskANN/config/ --algorithm osq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset wiki --config-dir ./experiments/DiskANN/config/ --algorithm opq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset wiki --config-dir ./experiments/DiskANN/config/ --algorithm saq

# ./build/experiments/DiskANN/build_diskann_index --dataset laion --config-dir ./experiments/DiskANN/config/
# ./build/experiments/DiskANN/bench_diskann_quant --dataset laion --config-dir ./experiments/DiskANN/config/ --algorithm sq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset laion --config-dir ./experiments/DiskANN/config/ --algorithm pq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset laion --config-dir ./experiments/DiskANN/config/ --algorithm osq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset laion --config-dir ./experiments/DiskANN/config/ --algorithm opq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset laion --config-dir ./experiments/DiskANN/config/ --algorithm saq

./build/experiments/hnsw/bench_hnsw_quant --dataset sift1M --algorithm saq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset gist1M --algorithm saq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset audio --algorithm saq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset video --algorithm saq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset text2image --algorithm saq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset paper --algorithm saq --config-dir ./experiments/hnsw/config/

./build/experiments/DiskANN/bench_diskann_quant --dataset bigann --config-dir ./experiments/DiskANN/config/ --algorithm saq
./build/experiments/DiskANN/bench_diskann_quant --dataset wiki --config-dir ./experiments/DiskANN/config/ --algorithm saq
./build/experiments/DiskANN/bench_diskann_quant --dataset laion --config-dir ./experiments/DiskANN/config/ --algorithm saq
