/*
 * Standalone C++ SDP kernel compilation test.
 * Purpose: compile with icpx AOT to check for register spill.
 *
 * Build:
 *   icpx sdp_standalone.cpp -o sdp_standalone \
 *     -fsycl -fsycl-targets=spir64_gen \
 *     -Xs "-device bmg -options -doubleGRF" \
 *     -I../csrc/xpu -I../csrc/xpu/esimd_kernels \
 *     2>&1 | grep -i spill
 */

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using namespace sycl::ext::intel::experimental::esimd;
using namespace sycl::ext::intel;
using fp16 = sycl::half;
using fp32 = float;

#define FP32_MAX (1.7e+38)
#define FP32_MIN (-1.7e+38)
#define FP16_MAX (65504.0f)
#define FP16_MIN (-65504.0f)

#include "flash.attn.b.mha128.fp16.opt.h"
#include "flash.attn.b.mha128.bf16.h"
#include "flash.attn.b.mha128.bf16io.h"

// ---- Kernel dispatch functions ----

static void sdp_fp16_kernel(
    sycl::queue& q,
    uint8_t* qState, uint8_t* kState, uint8_t* vState,
    uint8_t* normAlpha, uint8_t* out,
    uint32_t q_len, uint32_t kv_len,
    uint32_t headQ, uint32_t headKv)
{
    uint32_t q_blocks = (q_len + 255) / 256;
    sycl::range<2> GlobalRange(headQ * 16, q_blocks);
    sycl::range<2> LocalRange(16, 1);
    sycl::nd_range<2> Range(GlobalRange, LocalRange);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
            flashAttnBMha128Fp16OptPrecomputed(
                qState, kState, vState, normAlpha, out,
                q_len, kv_len, headQ, headKv, ndi);
        });
    });
}

static void sdp_bf16_kernel(
    sycl::queue& q,
    uint8_t* qState, uint8_t* kState, uint8_t* vState,
    uint8_t* normAlpha, uint8_t* out,
    uint32_t q_len, uint32_t kv_len,
    uint32_t headQ, uint32_t headKv)
{
    uint32_t q_blocks = (q_len + 255) / 256;
    sycl::range<2> GlobalRange(headQ * 16, q_blocks);
    sycl::range<2> LocalRange(16, 1);
    sycl::nd_range<2> Range(GlobalRange, LocalRange);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
            flashAttnBMha128Bf16Precomputed(
                qState, kState, vState, normAlpha, out,
                q_len, kv_len, headQ, headKv, ndi);
        });
    });
}

static void sdp_bf16io_kernel(
    sycl::queue& q,
    uint8_t* qState, uint8_t* kState, uint8_t* vState,
    uint8_t* normAlpha, uint8_t* out,
    uint32_t q_len, uint32_t kv_len,
    uint32_t headQ, uint32_t headKv)
{
    uint32_t q_blocks = (q_len + 255) / 256;
    sycl::range<2> GlobalRange(headQ * 16, q_blocks);
    sycl::range<2> LocalRange(16, 1);
    sycl::nd_range<2> Range(GlobalRange, LocalRange);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
            flashAttnBMha128Bf16IoPrecomputed(
                qState, kState, vState, normAlpha, out,
                q_len, kv_len, headQ, headKv, ndi);
        });
    });
}

// ---- Simple host test ----

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    const uint32_t H = 32;
    const uint32_t D = 128;
    const uint32_t q_len = 512;
    const uint32_t kv_len = 512;

    size_t q_bytes = q_len * H * D * sizeof(sycl::half);
    size_t kv_bytes = kv_len * H * D * sizeof(sycl::half);
    size_t norm_bytes = H * D * sizeof(float);

    auto* Q_dev = (uint8_t*)sycl::malloc_device(q_bytes, q);
    auto* K_dev = (uint8_t*)sycl::malloc_device(kv_bytes, q);
    auto* V_dev = (uint8_t*)sycl::malloc_device(kv_bytes, q);
    auto* O_dev = (uint8_t*)sycl::malloc_device(q_bytes, q);
    auto* norm_dev = (uint8_t*)sycl::malloc_device(norm_bytes, q);

    // Fill normAlpha with 1.0f
    std::vector<float> norm_host(H * D, 1.0f);
    q.memcpy(norm_dev, norm_host.data(), norm_bytes).wait();

    // Zero-init Q, K, V (just for compilation/spill check, not correctness)
    q.memset(Q_dev, 0, q_bytes).wait();
    q.memset(K_dev, 0, kv_bytes).wait();
    q.memset(V_dev, 0, kv_bytes).wait();

    printf("\n--- FP16 SDP ---\n");
    sdp_fp16_kernel(q, Q_dev, K_dev, V_dev, norm_dev, O_dev, q_len, kv_len, H, H);
    q.wait();
    printf("FP16 kernel executed OK\n");

    // BF16 needs bf16 buffers — same size
    size_t bf16_q_bytes = q_len * H * D * sizeof(sycl::ext::oneapi::bfloat16);
    size_t bf16_kv_bytes = kv_len * H * D * sizeof(sycl::ext::oneapi::bfloat16);
    auto* Q_bf16 = (uint8_t*)sycl::malloc_device(bf16_q_bytes, q);
    auto* K_bf16 = (uint8_t*)sycl::malloc_device(bf16_kv_bytes, q);
    auto* V_bf16 = (uint8_t*)sycl::malloc_device(bf16_kv_bytes, q);
    auto* O_bf16 = (uint8_t*)sycl::malloc_device(bf16_q_bytes, q);
    q.memset(Q_bf16, 0, bf16_q_bytes).wait();
    q.memset(K_bf16, 0, bf16_kv_bytes).wait();
    q.memset(V_bf16, 0, bf16_kv_bytes).wait();

    printf("\n--- BF16 SDP ---\n");
    sdp_bf16_kernel(q, Q_bf16, K_bf16, V_bf16, norm_dev, O_bf16, q_len, kv_len, H, H);
    q.wait();
    printf("BF16 kernel executed OK\n");

    printf("\n--- BF16io SDP ---\n");
    sdp_bf16io_kernel(q, Q_bf16, K_bf16, V_bf16, norm_dev, O_bf16, q_len, kv_len, H, H);
    q.wait();
    printf("BF16io kernel executed OK\n");

    sycl::free(Q_dev, q);
    sycl::free(K_dev, q);
    sycl::free(V_dev, q);
    sycl::free(O_dev, q);
    sycl::free(norm_dev, q);
    sycl::free(Q_bf16, q);
    sycl::free(K_bf16, q);
    sycl::free(V_bf16, q);
    sycl::free(O_bf16, q);

    printf("\nAll SDP kernels compiled and executed successfully.\n");
    return 0;
}
