// Standalone benchmark: non-paged HD=256 reference kernel (s_scatter)
// Same config as paged kernel: 16Q/4KV, HD=256, noncausal
//
// Compile:
//   icpx tests/bench_ref_standalone.cpp -o bench_ref_standalone \
//     -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg -options -doubleGRF" \
//     -I ~/yuchen/vllm_env/SDP_noncausal_ref_256d -ffast-math -std=c++17 -O2

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <cstring>

using namespace sycl;
using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;

#define __ESIMD_NS sycl::ext::intel::esimd
#define __ESIMD_ENS sycl::ext::intel::experimental::esimd
#undef ESIMD_INLINE
#define ESIMD_INLINE inline __attribute__((always_inline))
#define FP32_MIN -3.402823466e+38f

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using namespace sycl::ext::intel::experimental::esimd;

#include "rev256_onednn_v2_88tflops_s_scatter.h"

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

    constexpr int NUM_Q_HEADS = 16;
    constexpr int NUM_KV_HEADS = 4;
    constexpr int HEAD_DIM = 256;
    constexpr int WARMUP = 10;
    constexpr int ITERS = 100;

    struct TestConfig { int q_len; int kv_len; const char* name; };
    TestConfig tests[] = {
        {1024,  1024,  "1Kx1K"},
        {2048,  2048,  "2Kx2K"},
        {4096,  4096,  "4Kx4K"},
        {8192,  8192,  "8Kx8K"},
    };

    std::cout << "\nNon-paged ref (s_scatter): Q_heads=" << NUM_Q_HEADS
              << " KV_heads=" << NUM_KV_HEADS << " HD=" << HEAD_DIM << " fp16\n\n";

    std::cout << std::setw(12) << "Size"
              << " | " << std::setw(10) << "Time(ms)"
              << " | " << std::setw(10) << "TFLOPS"
              << " | " << std::setw(8) << "Util%"
              << "\n";
    std::cout << std::string(50, '-') << "\n";

    for (auto& t : tests) {
        int q_len = t.q_len;
        int kv_len = t.kv_len;

        size_t q_size = q_len * NUM_Q_HEADS * HEAD_DIM;
        size_t kv_size = kv_len * NUM_KV_HEADS * HEAD_DIM;

        fp16* d_Q = sycl::malloc_device<fp16>(q_size, q);
        fp16* d_K = sycl::malloc_device<fp16>(kv_size, q);
        fp16* d_V = sycl::malloc_device<fp16>(kv_size, q);
        fp16* d_O = sycl::malloc_device<fp16>(q_size, q);
        float* d_normAlpha = sycl::malloc_device<float>(NUM_Q_HEADS * HEAD_DIM, q);

        // Initialize
        std::vector<fp16> h_data(std::max(q_size, kv_size));
        std::vector<float> h_alpha(NUM_Q_HEADS * HEAD_DIM, 1.0f);
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(-0.5f, 0.5f);

        for (size_t i = 0; i < q_size; i++) h_data[i] = fp16(dis(gen));
        q.memcpy(d_Q, h_data.data(), q_size * sizeof(fp16)).wait();
        for (size_t i = 0; i < kv_size; i++) h_data[i] = fp16(dis(gen));
        q.memcpy(d_K, h_data.data(), kv_size * sizeof(fp16)).wait();
        for (size_t i = 0; i < kv_size; i++) h_data[i] = fp16(dis(gen));
        q.memcpy(d_V, h_data.data(), kv_size * sizeof(fp16)).wait();
        q.memcpy(d_normAlpha, h_alpha.data(), NUM_Q_HEADS * HEAD_DIM * sizeof(float)).wait();

        // s_scatter uses nd_range<2>: {32 * headQ, (q_len+127)/128}, WG={32, 1}
        int groupH = NUM_Q_HEADS;
        int groupV = (q_len + 127) / 128;
        sycl::nd_range<2> nd_range({(size_t)(32 * groupH), (size_t)groupV}, {32, 1});

        // Warmup
        for (int i = 0; i < WARMUP; i++) {
            q.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(nd_range, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                    flashAttnBMha256Fp16Rev_s_scatter(
                        reinterpret_cast<uint8_t*>(d_Q), reinterpret_cast<uint8_t*>(d_K),
                        reinterpret_cast<uint8_t*>(d_V), reinterpret_cast<uint8_t*>(d_normAlpha),
                        reinterpret_cast<uint8_t*>(d_O),
                        q_len, kv_len, NUM_Q_HEADS, NUM_KV_HEADS, ndi);
                });
            }).wait();
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) {
            q.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(nd_range, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                    flashAttnBMha256Fp16Rev_s_scatter(
                        reinterpret_cast<uint8_t*>(d_Q), reinterpret_cast<uint8_t*>(d_K),
                        reinterpret_cast<uint8_t*>(d_V), reinterpret_cast<uint8_t*>(d_normAlpha),
                        reinterpret_cast<uint8_t*>(d_O),
                        q_len, kv_len, NUM_Q_HEADS, NUM_KV_HEADS, ndi);
                });
            }).wait();
        }
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count() / ITERS;

        double flops = 4.0 * q_len * kv_len * HEAD_DIM * NUM_Q_HEADS + 2.0 * q_len * kv_len * NUM_Q_HEADS;
        double tflops = flops / (elapsed_ms * 1e9);
        double util = (tflops / 96.0) * 100.0;

        std::cout << std::setw(12) << t.name
                  << " | " << std::fixed << std::setprecision(3) << std::setw(10) << elapsed_ms
                  << " | " << std::setprecision(1) << std::setw(10) << tflops
                  << " | " << std::setprecision(1) << std::setw(8) << util
                  << "\n";

        sycl::free(d_Q, q); sycl::free(d_K, q); sycl::free(d_V, q);
        sycl::free(d_O, q); sycl::free(d_normAlpha, q);
    }

    return 0;
}
