To build the project:
```
cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DFAISS_ENABLE_EXTRAS=ON
make -C build -j
```

Before running experiments:
1. Set dataset name, path, HNSW configurations in hnsw/config/datasets.conf
2. Set algorithm parameter in \<algo_name\>.conf

Run the experiment:
```
./build/experiments/hnsw/bench_hnsw_quant --dataset <ds_name> --algorithm <algo_name> --config-dir ./experiments/hnsw/config/
```