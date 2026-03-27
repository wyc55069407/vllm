#pragma once
// moe_prefill.h — Non-doubleGRF kernels and oneDNN cache for MoE prefill
// Ported from moe_prefill_v29.cpp (Windows exe)
// Build: common_ops extension (standard GRF)

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstdio>

using namespace sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;

using fp16 = sycl::half;

// ======================== moe_gather_tokens ========================
// SLM-based token-to-expert routing. Builds expert_offsets and mappings.
// total_seqlen = num_tokens * top_k, must be multiple of 64.
template<int MAX_EXPS = 256, int GS = 64>
sycl::event moe_gather_tokens(
    sycl::queue& q,
    const int* selected_experts, int* expert_offsets, int* expert_tokens, int* seq_to_token,
    int num_experts, int total_seqlen, int top_k)
{
    static_assert(MAX_EXPS % 32 == 0);

    constexpr int SLM_CNT_SIZE   = MAX_EXPS * (int)sizeof(int);
    constexpr int SLM_EOFF_SIZE  = MAX_EXPS * (int)sizeof(int);
    constexpr int SLM_TOTAL = SLM_CNT_SIZE + SLM_EOFF_SIZE;

    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(GS, GS),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
            slm_init<SLM_TOTAL>();
            const int lid = (int)item.get_local_id(0);
            const int ppt = total_seqlen / GS;

            // Zero SLM counts and offsets
            for (int off = lid * 128; off < num_experts * 4; off += GS * 128) {
                slm_block_store<int, 32>(off, 0);
                slm_block_store<int, 32>(SLM_CNT_SIZE + off, 0);
            }
            barrier();

            // Count tokens per expert
            const int base = lid * ppt;
            for (int i = 0; i < ppt; i += 32) {
                simd<int, 32> sel = block_load<int, 32>(selected_experts + base + i);
                simd<uint32_t, 32> byte_off = convert<uint32_t>(sel) * 4u;
                slm_atomic_update<atomic_op::inc, int, 32>(byte_off, simd_mask<32>(1));
            }
            barrier();

            // Exclusive prefix sum (single thread)
            if (lid == 0) {
                int sum = 0;
                for (int off = 0; off < num_experts * 4; off += 128) {
                    simd<int, 32> v = slm_block_load<int, 32>(off);
                    simd<int, 32> new_v;
                    #pragma unroll
                    for (int j = 0; j < 32; j++) { new_v[j] = sum; sum += v[j]; }
                    slm_block_store<int, 32>(SLM_CNT_SIZE + off, new_v);
                }
            }
            barrier();

            // Write expert offsets to global memory
            for (uint32_t off = lid * 128u; off < num_experts * 4u; off += GS * 128u) {
                simd<int, 32> v = slm_block_load<int, 32>(SLM_CNT_SIZE + off);
                block_store<int, 32>(expert_offsets + off / 4u, v);
            }

            // Scatter tokens to expert positions
            {
                const int base2 = lid * ppt;
                for (int i = 0; i < ppt; i += 32) {
                    simd<int, 32> sel = block_load<int, 32>(selected_experts + base2 + i);
                    simd<uint32_t, 32> expert_byte_off = convert<uint32_t>(sel) * 4u;
                    simd<int, 32> pos =
                        slm_atomic_update<atomic_op::inc, int, 32>(
                            SLM_CNT_SIZE + expert_byte_off, simd_mask<32>(1));
                    simd<int, 32> pair_idx(base2 + i, 1);
                    simd<int, 32> tok = pair_idx / top_k;
                    scatter<int, 32>(expert_tokens, convert<uint32_t>(pos) * 4u, tok);
                    block_store<int, 32>(seq_to_token + base2 + i, pos);
                }
            }
        });
    });
}

// ======================== moe_gather_states ========================
// Scatter input tokens into per-expert contiguous buffer.
// Uses 32-wide int32 loads (128B per iteration) to handle K>=64 (min 32 fp16 = 64B)
template<typename IT>
sycl::event moe_gather_states(
    sycl::queue& q,
    const IT* input, const int* expert_tokens, IT* expert_states,
    int total_seqlen, int hidden_size)
{
    const int h_words = hidden_size * (int)sizeof(IT) / (int)sizeof(int32_t);
    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::range<1>(total_seqlen),
            [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
                const int idx = (int)id[0];
                simd<uint32_t, 1> off((uint32_t)idx * 4u);
                simd<int, 1> tok_v = gather<int, 1>(expert_tokens, off);
                const int tok = tok_v[0];
                const auto* src = reinterpret_cast<const int32_t*>(
                    input + (size_t)tok * hidden_size);
                auto* dst = reinterpret_cast<int32_t*>(
                    expert_states + (size_t)idx * hidden_size);
                for (int h = 0; h < h_words; h += 32) {
                    simd<int32_t, 32> v = block_load<int32_t, 32>(src + h);
                    block_store<int32_t, 32>(dst + h, v);
                }
            });
    });
}

// ======================== moe_accumulate ========================
// Weighted sum of expert outputs → final output [num_tokens, hidden_size]
// Handles any top_k (scalar loads for routing weights/positions)
template<typename IT>
sycl::event moe_accumulate(
    sycl::queue& q,
    const IT* output, const float* routing_weights, const int* seq_to_token, IT* final_output,
    int num_tokens, int hidden_size, int top_k)
{
    constexpr int SW = 32;  // 32 fp16 = 64B = 1 cache line
    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::range<1>(num_tokens),
            [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
                const int tok = (int)id[0];
                for (int hoff = 0; hoff < hidden_size; hoff += SW) {
                    simd<float, SW> acc(0.f);
                    for (int s = 0; s < top_k; s++) {
                        simd<uint32_t, 1> w_off((uint32_t)(tok * top_k + s) * 4u);
                        simd<uint32_t, 1> p_off = w_off;
                        float w = gather<float, 1>(routing_weights, w_off)[0];
                        int pi = gather<int, 1>(seq_to_token, p_off)[0];
                        acc += w * convert<float>(
                            block_load<IT, SW>(output + (size_t)pi * hidden_size + hoff));
                    }
                    block_store<IT, SW>(final_output + (size_t)tok * hidden_size + hoff, convert<IT>(acc));
                }
            });
    });
}

// ======================== oneDNN W4A16 primitive cache ========================
// Cached entry for one M value
struct OneDNNCachedPrim {
    dnnl::matmul* prim = nullptr;
    dnnl::matmul::primitive_desc* pd = nullptr;
    dnnl::memory *mem_src = nullptr, *mem_wt = nullptr;
    dnnl::memory *mem_sc = nullptr, *mem_zp = nullptr, *mem_dst = nullptr;
    std::unordered_map<int, dnnl::memory>* exec_args = nullptr;

    ~OneDNNCachedPrim() {
        delete prim; delete pd;
        delete mem_src; delete mem_wt; delete mem_sc; delete mem_zp; delete mem_dst;
        delete exec_args;
    }
};

struct MoePrefillOneDNNShapeCache {
    int K_ = 0, N_ = 0, block_size_ = 0;
    dnnl::engine* eng_ = nullptr;

    // Pre-created entries for M=1..init_max_m (indexed by M-1)
    std::vector<OneDNNCachedPrim*> prebuilt;
    // On-demand entries for M > init_max_m (keyed by M)
    std::unordered_map<int, OneDNNCachedPrim*> overflow;

    ~MoePrefillOneDNNShapeCache() {
        for (auto* p : prebuilt) delete p;
        for (auto& kv : overflow) delete kv.second;
    }

    void init(dnnl::engine& eng, int K, int N, int block_size, int init_max_m) {
        K_ = K; N_ = N; block_size_ = block_size; eng_ = &eng;
        prebuilt.resize(init_max_m);
        for (int i = 0; i < init_max_m; i++)
            prebuilt[i] = create_entry(eng, i + 1);
    }

    OneDNNCachedPrim* create_entry(dnnl::engine& eng, int M) {
        auto* e = new OneDNNCachedPrim();
        auto dt = dnnl::memory::data_type::f16;
        dnnl::memory::desc x_desc({M, K_}, dt, dnnl::memory::format_tag::ab);
        dnnl::memory::desc w_desc({K_, N_}, dnnl::memory::data_type::u4, dnnl::memory::format_tag::ba);
        dnnl::memory::desc o_desc({M, N_}, dt, dnnl::memory::format_tag::ab);
        dnnl::memory::desc s_desc({K_ / block_size_, N_}, dt, dnnl::memory::format_tag::ab);
        dnnl::memory::desc z_desc({1}, dnnl::memory::data_type::u8, dnnl::memory::format_tag::a);

        dnnl::primitive_attr attr;
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 1) + (1 << 0), {block_size_, 1}, dt);
        attr.set_zero_points(DNNL_ARG_WEIGHTS, 0, {}, dnnl::memory::data_type::u8);
        attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);

        e->pd = new dnnl::matmul::primitive_desc(eng, x_desc, w_desc, o_desc, attr);
        e->prim = new dnnl::matmul(*e->pd);
        e->mem_src = new dnnl::memory(e->pd->src_desc(), eng, DNNL_MEMORY_NONE);
        e->mem_wt  = new dnnl::memory(e->pd->weights_desc(), eng, DNNL_MEMORY_NONE);
        e->mem_sc  = new dnnl::memory(s_desc, eng, DNNL_MEMORY_NONE);
        e->mem_zp  = new dnnl::memory(z_desc, eng, DNNL_MEMORY_NONE);
        e->mem_dst = new dnnl::memory(e->pd->dst_desc(), eng, DNNL_MEMORY_NONE);
        e->exec_args = new std::unordered_map<int, dnnl::memory>({
            {DNNL_ARG_SRC,                                  *e->mem_src},
            {DNNL_ARG_WEIGHTS,                              *e->mem_wt},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,      *e->mem_sc},
            {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, *e->mem_zp},
            {DNNL_ARG_DST,                                  *e->mem_dst}
        });
        return e;
    }

    OneDNNCachedPrim* get(int M) {
        int idx = M - 1;
        if (idx < (int)prebuilt.size()) return prebuilt[idx];
        // On-demand: create and cache
        auto it = overflow.find(M);
        if (it != overflow.end()) return it->second;
        auto* e = create_entry(*eng_, M);
        overflow[M] = e;
        return e;
    }

    void execute(dnnl::stream& s, int M,
                 void* input, void* weight, void* scales, void* zero_point, void* output)
    {
        auto* e = get(M);
        e->mem_src->set_data_handle(input);
        e->mem_wt->set_data_handle(weight);
        e->mem_sc->set_data_handle(scales);
        e->mem_zp->set_data_handle(zero_point);
        e->mem_dst->set_data_handle(output);
        e->prim->execute(s, *e->exec_args);
    }
};

struct MoePrefillOneDNNContext {
    dnnl::engine eng;
    MoePrefillOneDNNShapeCache gateup_cache;  // K=H, N=2*IM
    MoePrefillOneDNNShapeCache down_cache;    // K=IM, N=H
    bool initialized = false;
    uint8_t* zero_point_buf = nullptr;  // USM shared — must be device-accessible
    sycl::queue* q_ptr = nullptr;

    ~MoePrefillOneDNNContext() {
        if (zero_point_buf && q_ptr) sycl::free(zero_point_buf, *q_ptr);
    }

    // Pre-create primitives for M=1..init_max_m; larger M created on-demand and cached.
    void init(sycl::queue& q, int H, int IM, int BS, int init_max_m) {
        eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        gateup_cache.init(eng, H, 2*IM, BS, init_max_m);
        down_cache.init(eng, IM, H, BS, init_max_m);
        q_ptr = &q;
        zero_point_buf = sycl::malloc_shared<uint8_t>(1, q);
        zero_point_buf[0] = 8;
        initialized = true;
    }
};

// Global lazy-initialized context
static MoePrefillOneDNNContext g_moe_prefill_dnn_ctx;
static std::mutex g_moe_prefill_dnn_mutex;

// ======================== oneDNN forward for large experts ========================
template<int BS, int MAX_M>
void moe_onednn_forward(
    sycl::queue& q, MoePrefillOneDNNContext& ctx,
    const fp16* expert_states,
    const uint8_t* guw_weight, const fp16* guw_scale_t,
    const uint8_t* down_weight, const fp16* down_scale_t,
    fp16* intermediate, fp16* output,
    fp16* gate_buf,
    const int* expert_offsets,
    int num_experts, int total_seqlen, int hidden_size, int intermediate_size)
{
    const int nblk_h  = hidden_size / BS;
    const int nblk_im = intermediate_size / BS;
    const int fused_im = 2 * intermediate_size;

    dnnl::stream s = dnnl::sycl_interop::make_stream(ctx.eng, q);

    for (int e = 0; e < num_experts; e++) {
        int t0 = expert_offsets[e];
        int t1 = (e + 1 < num_experts) ? expert_offsets[e + 1] : total_seqlen;
        int nt = t1 - t0;
        if (nt <= MAX_M) continue;  // handled by GGEMV

        const fp16* x_ptr = expert_states + (size_t)t0 * hidden_size;
        fp16* inter_ptr = intermediate + (size_t)t0 * intermediate_size;
        fp16* out_ptr = output + (size_t)t0 * hidden_size;
        fp16* gb_ptr = gate_buf + (size_t)t0 * fused_im;

        const uint8_t* guw_ptr = guw_weight + (size_t)e * fused_im * (hidden_size / 2);
        const uint8_t* dw_ptr  = down_weight + (size_t)e * hidden_size * (intermediate_size / 2);
        // Transposed scales for oneDNN: [nblk, N] per expert
        const fp16* gus_ptr = guw_scale_t + (size_t)e * nblk_h * fused_im;
        const fp16* ds_ptr  = down_scale_t + (size_t)e * nblk_im * hidden_size;

        // Gate+Up: [nt, H] x [H, 2*IM] -> gate_buf[nt, 2*IM]
        ctx.gateup_cache.execute(s, nt,
            (void*)x_ptr, (void*)guw_ptr, (void*)gus_ptr, (void*)ctx.zero_point_buf, (void*)gb_ptr);

        // SiLU(gate) * up -> intermediate
        s.wait();
        q.parallel_for(sycl::range<1>(nt * intermediate_size), [=](sycl::id<1> id) {
            int idx = (int)id[0];
            int i = idx / intermediate_size;
            int j = idx % intermediate_size;
            float g = (float)gb_ptr[i * fused_im + j];
            float u = (float)gb_ptr[i * fused_im + intermediate_size + j];
            float silu_g = g / (1.0f + sycl::exp(-g));
            inter_ptr[idx] = static_cast<fp16>(silu_g * u);
        }).wait();

        // Down: [nt, IM] x [IM, H] -> output
        ctx.down_cache.execute(s, nt,
            (void*)inter_ptr, (void*)dw_ptr, (void*)ds_ptr, (void*)ctx.zero_point_buf, (void*)out_ptr);
    }

    s.wait();
}
