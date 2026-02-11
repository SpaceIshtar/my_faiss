# ./build/experiments/hnsw/build_hnsw_index --dataset audio --config-dir ./experiments/hnsw/config/
# ./build/experiments/hnsw/bench_hnsw_standard --dataset audio --config-dir ./experiments/hnsw/config/
# ./build/experiments/hnsw/bench_hnsw_quant --dataset audio --algorithm pq --config-dir ./experiments/hnsw/config/
# ./build/experiments/hnsw/bench_hnsw_quant --dataset audio --algorithm sq --config-dir ./experiments/hnsw/config/
# ./build/experiments/hnsw/bench_hnsw_quant --dataset audio --algorithm rabitq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset audio --algorithm opq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset audio --algorithm vaq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_rabitq_native --dataset audio --config-dir experiments/hnsw/config/ --rerank
./build/experiments/hnsw/bench_hnsw_quant --dataset audio --algorithm rq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset audio --algorithm prq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset audio --algorithm lsq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset audio --algorithm plsq --config-dir ./experiments/hnsw/config/

./build/experiments/hnsw/build_hnsw_index --dataset video --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_standard --dataset video --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset video --algorithm pq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset video --algorithm sq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset video --algorithm rabitq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset video --algorithm opq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset video --algorithm vaq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_rabitq_native --dataset video --config-dir experiments/hnsw/config/ --rerank
./build/experiments/hnsw/bench_hnsw_quant --dataset video --algorithm rq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset video --algorithm prq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset video --algorithm lsq --config-dir ./experiments/hnsw/config/
./build/experiments/hnsw/bench_hnsw_quant --dataset video --algorithm plsq --config-dir ./experiments/hnsw/config/
