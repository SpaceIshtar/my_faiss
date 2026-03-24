# ./build/experiments/DiskANN/bench_diskann_standard --dataset bigann --config-dir ./experiments/DiskANN/config/
# ./build/experiments/DiskANN/bench_diskann_quant --dataset bigann --config-dir ./experiments/DiskANN/config/ --algorithm sq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset bigann --config-dir ./experiments/DiskANN/config/ --algorithm pq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset bigann --config-dir ./experiments/DiskANN/config/ --algorithm osq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset bigann --config-dir ./experiments/DiskANN/config/ --algorithm opq
# ./build/experiments/DiskANN/bench_diskann_quant --dataset bigann --config-dir ./experiments/DiskANN/config/ --algorithm rabitq


./build/experiments/DiskANN/bench_diskann_quant --dataset wiki --config-dir ./experiments/DiskANN/config/ --algorithm sq
./build/experiments/DiskANN/bench_diskann_quant --dataset wiki --config-dir ./experiments/DiskANN/config/ --algorithm pq
./build/experiments/DiskANN/bench_diskann_quant --dataset wiki --config-dir ./experiments/DiskANN/config/ --algorithm osq
./build/experiments/DiskANN/bench_diskann_quant --dataset wiki --config-dir ./experiments/DiskANN/config/ --algorithm opq
./build/experiments/DiskANN/bench_diskann_quant --dataset wiki --config-dir ./experiments/DiskANN/config/ --algorithm rabitq

./build/experiments/DiskANN/build_diskann_index --dataset laion --config-dir ./experiments/DiskANN/config/
./build/experiments/DiskANN/bench_diskann_quant --dataset laion --config-dir ./experiments/DiskANN/config/ --algorithm sq
./build/experiments/DiskANN/bench_diskann_quant --dataset laion --config-dir ./experiments/DiskANN/config/ --algorithm pq
./build/experiments/DiskANN/bench_diskann_quant --dataset laion --config-dir ./experiments/DiskANN/config/ --algorithm osq
./build/experiments/DiskANN/bench_diskann_quant --dataset laion --config-dir ./experiments/DiskANN/config/ --algorithm opq
./build/experiments/DiskANN/bench_diskann_quant --dataset laion --config-dir ./experiments/DiskANN/config/ --algorithm rabitq
