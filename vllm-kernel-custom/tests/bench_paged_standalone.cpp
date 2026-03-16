// Standalone benchmark: sdp_paged_prefill_dpas (noncausal + causal, identity block table)
// Tests both template paths: <false> (noncausal) and <true> (causal).
//
// Compile:
//   source ~/intel/oneapi/setvars.sh --force
//   icpx tests/bench_paged_standalone.cpp -o bench_paged_standalone \
//     -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg -options -doubleGRF" \
//     -I csrc/xpu/esimd_kernels -I csrc -ffast-math -std=c++17 -O2 \
//     2>&1 | grep -i spill
//
// Run: ./bench_paged_standalone

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <cstring>

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
#define FP32_MIN (-1e38f)

#include "sdp_paged.h"

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    constexpr int NUM_Q_HEADS = 16;
    constexpr int NUM_KV_HEADS = 4;
    constexpr int HEAD_DIM = 256;
    constexpr int BLOCK_SIZE = 1024;
    constexpr int BATCH = 1;
    constexpr int WARMUP = 10;
    constexpr int ITERS = 100;
    constexpr float ATTN_SCALE = 1.0f / 16.0f;  // 1/sqrt(256)

    struct TestConfig { int q_len; int kv_len; const char* name; };
    TestConfig tests[] = {
        {1024,  1024,  "1Kx1K"},
        {2048,  2048,  "2Kx2K"},
        {4096,  4096,  "4Kx4K"},
        {8192,  8192,  "8Kx8K"},
    };

    std::cout << "\nConfig: Q_heads=" << NUM_Q_HEADS << " KV_heads=" << NUM_KV_HEADS
              << " HD=" << HEAD_DIM << " block_size=" << BLOCK_SIZE << " bf16\n";

    // Run both noncausal and causal benchmarks
    for (int causal_mode = 0; causal_mode <= 1; causal_mode++) {
    const char* mode_name = causal_mode ? "CAUSAL" : "NONCAUSAL";
    std::cout << "\n=== " << mode_name << " ===\n";
    std::cout << std::setw(12) << "Size"
              << " | " << std::setw(10) << "Time(ms)"
              << " | " << std::setw(10) << "TFLOPS"
              << " | " << std::setw(8) << "Util%"
              << "\n";
    std::cout << std::string(50, '-') << "\n";

    for (auto& t : tests) {
        int q_len = t.q_len;
        int kv_len = t.kv_len;
        int max_blocks = (kv_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int total_phys_blocks = max_blocks;
        int num_tokens = BATCH * q_len;

        // Allocate device memory
        size_t q_size = num_tokens * NUM_Q_HEADS * HEAD_DIM;
        size_t kv_size = 2 * total_phys_blocks * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM;
        size_t bt_size = BATCH * max_blocks;
        size_t out_size = num_tokens * NUM_Q_HEADS * HEAD_DIM;

        auto* d_query = sycl::malloc_device<unsigned short>(q_size, q);
        auto* d_kv = sycl::malloc_device<unsigned short>(kv_size, q);
        auto* d_output = sycl::malloc_device<unsigned short>(out_size, q);
        auto* d_block_table = sycl::malloc_device<int>(bt_size, q);
        auto* d_seq_lens = sycl::malloc_device<int>(BATCH, q);
        auto* d_query_start_loc = sycl::malloc_device<int>(BATCH + 1, q);

        // Initialize with random data
        std::vector<unsigned short> h_q(q_size), h_kv(kv_size);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
        for (size_t i = 0; i < q_size; i++) {
            bf16 v(dis(gen));
            memcpy(&h_q[i], &v, 2);
        }
        for (size_t i = 0; i < kv_size; i++) {
            bf16 v(dis(gen));
            memcpy(&h_kv[i], &v, 2);
        }
        q.memcpy(d_query, h_q.data(), q_size * 2).wait();
        q.memcpy(d_kv, h_kv.data(), kv_size * 2).wait();

        // Identity block table
        std::vector<int> h_bt(bt_size);
        for (int i = 0; i < max_blocks; i++) h_bt[i] = i;
        q.memcpy(d_block_table, h_bt.data(), bt_size * sizeof(int)).wait();

        // Seq lens + query_start_loc
        int h_seq = kv_len;
        q.memcpy(d_seq_lens, &h_seq, sizeof(int)).wait();
        int h_qsl[2] = {0, num_tokens};
        q.memcpy(d_query_start_loc, h_qsl, 2 * sizeof(int)).wait();

        // Compute strides for [2, num_blocks, block_size, num_kv_heads, head_dim]
        int64_t kv_stride_head = HEAD_DIM;  // elements, not bytes
        int64_t kv_stride_pos = NUM_KV_HEADS * HEAD_DIM;
        int64_t kv_stride_block = BLOCK_SIZE * kv_stride_pos;
        int64_t kv_stride_split = total_phys_blocks * kv_stride_block;

        int max_q_tiles = (q_len + 127) / 128;
        int total_wgs = BATCH * max_q_tiles * NUM_Q_HEADS;
        int wg_size = 32;

        sycl::nd_range<1> nd_range(total_wgs * wg_size, wg_size);

        // Macro for kernel launch to avoid duplicating 15 args
        #define LAUNCH_KERNEL(CAUSAL_VAL) \
            q.submit([&](sycl::handler& cgh) { \
                cgh.parallel_for(nd_range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL { \
                    sdp_paged_prefill_dpas<CAUSAL_VAL>( \
                        d_query, d_kv, d_output, \
                        d_block_table, d_seq_lens, d_query_start_loc, \
                        NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, BLOCK_SIZE, \
                        max_blocks, \
                        kv_stride_split, kv_stride_block, kv_stride_pos, kv_stride_head, \
                        ATTN_SCALE, num_tokens, \
                        max_q_tiles, BATCH, ndi); \
                }); \
            }).wait()

        // Warmup
        for (int i = 0; i < WARMUP; i++) {
            if (causal_mode) { LAUNCH_KERNEL(true); }
            else             { LAUNCH_KERNEL(false); }
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) {
            if (causal_mode) { LAUNCH_KERNEL(true); }
            else             { LAUNCH_KERNEL(false); }
        }
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / ITERS;

        #undef LAUNCH_KERNEL

        // Causal: triangular mask means ~half the QK dot products contribute
        // Full: 4*Q*KV*HD*Qh + 2*Q*KV*Qh  (QK + PV, each counted as 2*mul+add)
        // Causal (q_len==kv_len): average row does KV/2 tokens → half FLOPs
        double qk_factor = causal_mode ? 2.0 : 4.0;  // 4→2 for causal half
        double sv_factor = causal_mode ? 1.0 : 2.0;   // 2→1 for causal half
        double flops = qk_factor * q_len * kv_len * HEAD_DIM * NUM_Q_HEADS
                     + sv_factor * q_len * kv_len * NUM_Q_HEADS;
        double tflops = flops / (elapsed_ms * 1e9);
        double util = (tflops / 96.0) * 100.0;

        std::cout << std::setw(12) << t.name
                  << " | " << std::fixed << std::setprecision(3) << std::setw(10) << elapsed_ms
                  << " | " << std::setprecision(1) << std::setw(10) << tflops
                  << " | " << std::setprecision(1) << std::setw(8) << util
                  << "\n";

        sycl::free(d_query, q);
        sycl::free(d_kv, q);
        sycl::free(d_output, q);
        sycl::free(d_block_table, q);
        sycl::free(d_seq_lens, q);
        sycl::free(d_query_start_loc, q);
    }
    } // end causal_mode loop

    return 0;
}
