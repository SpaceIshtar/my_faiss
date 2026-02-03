# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

add_executable(bench_rabitq_sift EXCLUDE_FROM_ALL bench_rabitq_sift.cpp)
target_link_libraries(bench_rabitq_sift PRIVATE faiss)
