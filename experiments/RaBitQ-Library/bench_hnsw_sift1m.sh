#!/bin/bash

# Benchmark RaBitQ + HNSW on SIFT1M dataset
# This script runs clustering, indexing, and querying

set -e

# Configuration
DATA_DIR="/data/local/embedding_dataset/sift1M"
BASE_FILE="${DATA_DIR}/sift_base.fvecs"
QUERY_FILE="${DATA_DIR}/sift_query.fvecs"
GT_FILE="${DATA_DIR}/sift_groundtruth.ivecs"

# Output directory
OUTPUT_DIR="${DATA_DIR}/rabitq_index"
mkdir -p ${OUTPUT_DIR}

# Parameters
NUM_CLUSTERS=16      # Number of clusters for RaBitQ
HNSW_M=32            # HNSW degree bound
HNSW_EF=200          # HNSW ef for indexing
TOTAL_BITS=4         # Total bits for quantization (1-9)
NUM_THREADS=16       # Number of threads

# Derived paths
CENTROIDS_FILE="${OUTPUT_DIR}/centroids_${NUM_CLUSTERS}.fvecs"
CLUSTER_IDS_FILE="${OUTPUT_DIR}/clusterids_${NUM_CLUSTERS}.ivecs"
INDEX_FILE="${OUTPUT_DIR}/hnsw_M${HNSW_M}_ef${HNSW_EF}_bits${TOTAL_BITS}.index"

cd "$(dirname "$0")"

echo "=============================================="
echo "RaBitQ + HNSW Benchmark on SIFT1M"
echo "=============================================="
echo "Data path: ${DATA_DIR}"
echo "Clusters: ${NUM_CLUSTERS}"
echo "HNSW M: ${HNSW_M}, efConstruction: ${HNSW_EF}"
echo "Total bits: ${TOTAL_BITS}"
echo ""

# Step 1: Clustering (if not already done)
if [ ! -f "${CENTROIDS_FILE}" ] || [ ! -f "${CLUSTER_IDS_FILE}" ]; then
    echo "[Step 1] Running clustering..."
    python3 ./python/ivf.py \
        ${BASE_FILE} \
        ${NUM_CLUSTERS} \
        ${CENTROIDS_FILE} \
        ${CLUSTER_IDS_FILE} \
        l2
    echo "Clustering done."
else
    echo "[Step 1] Clustering files already exist, skipping..."
fi
echo ""

# Step 2: Build index (if not already done)
if [ ! -f "${INDEX_FILE}" ]; then
    echo "[Step 2] Building HNSW + RaBitQ index..."
    ./bin/hnsw_rabitq_indexing \
        ${BASE_FILE} \
        ${CENTROIDS_FILE} \
        ${CLUSTER_IDS_FILE} \
        ${HNSW_M} \
        ${HNSW_EF} \
        ${TOTAL_BITS} \
        ${INDEX_FILE} \
        l2
    echo "Index built."
else
    echo "[Step 2] Index file already exists, skipping..."
fi
echo ""

# Step 3: Query benchmark
echo "[Step 3] Running query benchmark..."
./bin/hnsw_rabitq_querying \
    ${INDEX_FILE} \
    ${QUERY_FILE} \
    ${GT_FILE} \
    l2

echo ""
echo "=============================================="
echo "Benchmark complete!"
echo "=============================================="
