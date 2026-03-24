#!/usr/bin/env python3
"""
Convert SIFT small dataset from fvecs/ivecs format to HDF5 format.
"""

import numpy as np
import h5py
import struct
import os

def read_fvecs(filename):
    """Read fvecs file format (used for SIFT features)."""
    with open(filename, 'rb') as f:
        data = []
        while True:
            # Read dimension
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]

            # Read vector
            vec = struct.unpack('f' * dim, f.read(4 * dim))
            data.append(vec)

    return np.array(data, dtype=np.float32)

def read_ivecs(filename):
    """Read ivecs file format (used for ground truth indices)."""
    with open(filename, 'rb') as f:
        data = []
        while True:
            # Read dimension (k)
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]

            # Read vector
            vec = struct.unpack('i' * dim, f.read(4 * dim))
            data.append(vec)

    return np.array(data, dtype=np.int32)

def create_siftsmall_hdf5(data_dir, output_dir):
    """Create HDF5 file for SIFT small dataset."""

    print("Reading SIFT small dataset...")

    # Read data files
    base_file = os.path.join(data_dir, 'paper_base.fvecs')
    learn_file = os.path.join(data_dir, 'paper_query.fvecs')
    query_file = os.path.join(data_dir, 'paper_query.fvecs')
    gt_file = os.path.join(data_dir, 'paper_groundtruth.ivecs')
    # gt_dis_file = os.path.join(data_dir, 'gt100_30K_dist.fvecs')

    print(f"  Reading base vectors from {base_file}...")
    base_vectors = read_fvecs(base_file)
    print(f"    Shape: {base_vectors.shape}")

    # print(f"  Reading learn vectors from {learn_file}...")
    # learn_vectors = read_fvecs(learn_file)
    # print(f"    Shape: {learn_vectors.shape}")

    print(f"  Reading query vectors from {query_file}...")
    query_vectors = read_fvecs(query_file)
    print(f"    Shape: {query_vectors.shape}")

    print(f"  Reading ground truth from {gt_file}...")
    ground_truth = read_ivecs(gt_file)
    print(f"    Shape: {ground_truth.shape}")
    
    # distance_truth = read_ivecs(gt_dis_file)
    # print(f"    Shape: {distance_truth.shape}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create HDF5 file
    output_file = os.path.join(output_dir, 'paper-200-euclidean.hdf5')
    print(f"\nCreating HDF5 file: {output_file}")

    with h5py.File(output_file, 'w') as f:
        # Use base vectors as training/indexing data (the searchable database)
        # Ground truth references indices in base_vectors, not learn_vectors
        f.create_dataset('train', data=base_vectors)
        print(f"  Created 'train' dataset: {base_vectors.shape}")

        # Use query vectors as test data
        f.create_dataset('test', data=query_vectors)
        print(f"  Created 'test' dataset: {query_vectors.shape}")

        # Store ground truth neighbors
        f.create_dataset('neighbors', data=ground_truth)
        print(f"  Created 'neighbors' dataset: {ground_truth.shape}")

        # Add distances dataset (optional, can be computed from neighbors)
        # For now, we'll create a dummy distances array
        distances = np.zeros_like(ground_truth, dtype=np.float32)
        # f.create_dataset('distances', data=distance_truth)
        print(f"  Created 'distances' dataset: {distances.shape}")

    print(f"\n✓ Successfully created {output_file}")
    print(f"  Train (base): {base_vectors.shape}")
    print(f"  Test (query): {query_vectors.shape}")
    print(f"  Ground truth: {ground_truth.shape}")
    # print(f"  Note: Learn vectors ({learn_vectors.shape}) not included (used for separate training if needed)")

if __name__ == '__main__':
    data_dir = '/data/local/embedding_dataset/paper'
    output_dir = '/data/local/embedding_dataset/paper'

    create_siftsmall_hdf5(data_dir, output_dir)
