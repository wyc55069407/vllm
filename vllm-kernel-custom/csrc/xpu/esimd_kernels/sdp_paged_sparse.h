/* sdp_paged_sparse.h — Sparse paged SDP kernels for InfLLMv2 on Intel BMG XPU.
 *
 * Kernels:
 *   1. sdp_paged_decode_sparse_phase1  — Sparse decode with mask-directed paged blocks
 *   2. sdp_paged_prefill_sparse_dpas_128 — Sparse prefill with union mask + paged DPAS
 *
 * KV cache layout: [2, num_blocks, block_size, num_kv_heads, head_dim] (NHD).
 * Sparse block size: 64 tokens (InfLLMv2 standard).
 * Page block size: 128 (vLLM standard, >= sparse block).
 * Each sparse block fits entirely in one page.
 */

#pragma once
#include "sdp_paged.h"

/* ============================================================
 * SPARSE DECODE PHASE 1 — Mask-directed paged block lookup
 *
 * Like sdp_paged_decode_opt_phase1 but reads only blocks specified
 * by the sparse mask instead of sequential KV positions.
 *
 * Mask layout: [batch, num_kv_heads, num_sparse_blocks] uint32
 *   mask[req * nkvh * nsb + kv_head * nsb + i] = kv_block_id
 *   kv_block_id * sp_blk_size = logical start token
 *
 * Output: writes to scratch_out/max/lse for phase2 reduction.
 * Phase2 reuses sdp_paged_decode_opt_phase2 from sdp_paged.h.
 * ============================================================ */
template<uint32_t HD, uint32_t NUM_KV_HEADS, uint32_t Q_HEAD_PER_T,
         uint32_t sp_blk_size, uint32_t num_sparse_blocks,
         uint32_t chunk_size, uint32_t HEAD_GROUPS_PER_G, bool IS_BF16>
ESIMD_INLINE void sdp_paged_decode_sparse_phase1(
    const unsigned short* __restrict__ query_ptr,        // [batch, num_heads, HD]
    const unsigned short* __restrict__ kv_cache_ptr,     // paged KV cache
    float* __restrict__ scratch_out,                      // [batch * chunk_num, num_heads, HD]
    float* __restrict__ scratch_max,                      // [batch * chunk_num, num_heads]
    float* __restrict__ scratch_lse,                      // [batch * chunk_num, num_heads]
    const int* __restrict__ block_table_ptr,              // [batch, max_blocks_per_seq]
    const int* __restrict__ seq_lens_ptr,                 // [batch]
    const uint32_t* __restrict__ sparse_mask_ptr,         // [batch, NUM_KV_HEADS, num_sparse_blocks]
    int num_heads,
    int block_size, int max_blocks_per_seq,
    int64_t kv_stride_split, int64_t kv_stride_block,
    float attn_scale,
    int num_chunks_per_seq,
    int batch,
    nd_item<3>& ndi)
{
    using namespace sycl::ext::intel::esimd;

    constexpr int sp_blk_num_per_t = chunk_size / sp_blk_size;
    constexpr int KV_STRIDE_POS = NUM_KV_HEADS * HD;
    constexpr int q_head_num_per_kv_head = HEAD_GROUPS_PER_G * Q_HEAD_PER_T;
    constexpr int chunk_num = num_sparse_blocks / sp_blk_num_per_t;

    static_assert(chunk_size % sp_blk_size == 0);
    static_assert(num_sparse_blocks % sp_blk_num_per_t == 0);

    int chunk_idx      = ndi.get_group(1);
    int sp_blk_idx     = ndi.get_local_id(1);
    int head_group_idx = ndi.get_local_id(2);
    int kv_wg_idx      = ndi.get_group(2);
    int kv_head_idx    = kv_wg_idx % (int)NUM_KV_HEADS;
    int req_idx        = kv_wg_idx / (int)NUM_KV_HEADS;
    int q_head_idx     = kv_head_idx * q_head_num_per_kv_head
                        + head_group_idx * Q_HEAD_PER_T;

    if (req_idx >= batch) return;

    int seq_len = seq_lens_ptr[req_idx];

    // SLM layout (identical to dense decode phase1)
    constexpr int slm_reduce_size =
        HEAD_GROUPS_PER_G * sp_blk_num_per_t * Q_HEAD_PER_T * HD * sizeof(float);
    constexpr int slm_max_size =
        HEAD_GROUPS_PER_G * sp_blk_num_per_t * Q_HEAD_PER_T * sizeof(float);
    constexpr int slm_lse_size =
        HEAD_GROUPS_PER_G * sp_blk_num_per_t * Q_HEAD_PER_T * sizeof(float);

    if constexpr (sp_blk_num_per_t > 1) {
        slm_init(slm_reduce_size + slm_max_size + slm_lse_size);
    }

    const int* block_table_row = block_table_ptr + (int64_t)req_idx * max_blocks_per_seq;

    // Sparse mask lookup: which kv_block does this thread process?
    int sparse_slot = chunk_idx * sp_blk_num_per_t + sp_blk_idx;
    uint32_t kv_block_id = sparse_mask_ptr[
        (int64_t)req_idx * NUM_KV_HEADS * num_sparse_blocks
        + kv_head_idx * num_sparse_blocks
        + sparse_slot];

    // Convert sparse block id to paged address
    int logical_tok = (int)(kv_block_id * sp_blk_size);
    int page_idx = logical_tok / block_size;
    int offset_in_page = logical_tok % block_size;
    int phys_page = block_table_row[page_idx];

    // Compute valid tokens for this sparse block
    int valid_t = (int)sp_blk_size;
    {
        int tok_end = seq_len - logical_tok;
        if (tok_end < valid_t) valid_t = tok_end > 0 ? tok_end : 0;
    }

    // Load Q (same pattern as dense decode)
    const unsigned short* q_base = query_ptr +
        (int64_t)req_idx * num_heads * HD;

    simd<fp16, Q_HEAD_PER_T * HD> qIn;
    if constexpr (!IS_BF16) {
        const fp16* q_fp16 = reinterpret_cast<const fp16*>(
            q_base + (int64_t)q_head_idx * HD);
        qIn = block_load<fp16, Q_HEAD_PER_T * HD>(q_fp16);
        qIn = qIn * attn_scale;
    } else {
        #pragma unroll
        for (int h = 0; h < (int)Q_HEAD_PER_T; h++) {
            simd<float, 64> q0 = sdp_load_bf16_64(q_base + (q_head_idx + h) * HD);
            simd<float, 64> q1 = sdp_load_bf16_64(q_base + (q_head_idx + h) * HD + 64);
            q0 *= attn_scale; q1 *= attn_scale;
            qIn.template select<64, 1>(h * HD) = q0;
            qIn.template select<64, 1>(h * HD + 64) = q1;
        }
    }

    // Online softmax state
    simd<float, Q_HEAD_PER_T> maxKq          = FP32_MIN;
    simd<float, Q_HEAD_PER_T> old_maxKq      = FP32_MIN;
    simd<float, Q_HEAD_PER_T> max_correction = FP32_MIN;
    simd<float, Q_HEAD_PER_T> lse            = 0;
    simd<float, Q_HEAD_PER_T * HD> output    = 0;

    // K/V base pointer for this page
    int head_off = kv_head_idx * (int)HD;
    const unsigned short* k_base = kv_cache_ptr
        + (int64_t)phys_page * kv_stride_block + head_off;
    const unsigned short* v_base = k_base + (int)kv_stride_split;
    int kv_off = offset_in_page * KV_STRIDE_POS;

    for (int t = 0; t < valid_t; t++, kv_off += KV_STRIDE_POS) {
        const unsigned short* k_ptr = k_base + kv_off;
        const unsigned short* v_ptr = v_base + kv_off;
        simd<fp16, HD> kIn;
        simd<fp16, HD> vIn;
        if constexpr (!IS_BF16) {
            kIn = block_load<fp16, HD>(reinterpret_cast<const fp16*>(k_ptr));
            vIn = block_load<fp16, HD>(reinterpret_cast<const fp16*>(v_ptr));
        } else {
            simd<float, 64> k0 = sdp_load_bf16_64(k_ptr);
            simd<float, 64> k1 = sdp_load_bf16_64(k_ptr + 64);
            simd<float, 64> v0 = sdp_load_bf16_64(v_ptr);
            simd<float, 64> v1 = sdp_load_bf16_64(v_ptr + 64);
            kIn.template select<64, 1>(0) = k0;
            kIn.template select<64, 1>(64) = k1;
            vIn.template select<64, 1>(0) = v0;
            vIn.template select<64, 1>(64) = v1;
        }

        // QK dot products
        simd<float, Q_HEAD_PER_T> kq_out;
        #pragma unroll
        for (int h = 0; h < (int)Q_HEAD_PER_T; h++) {
            kq_out[h] = sycl::ext::intel::esimd::detail::sum<float, fp16, HD>(
                qIn.template select<HD, 1>(h * HD) * kIn);
        }

        // Online softmax
        old_maxKq = maxKq;
        maxKq = __ESIMD_NS::max<float, Q_HEAD_PER_T, float>(kq_out, old_maxKq);
        kq_out = kq_out - maxKq;
        kq_out = __ESIMD_NS::exp2<float, Q_HEAD_PER_T, float>(
            kq_out * sycl::ext::intel::esimd::detail::log2e);

        if (t >= 1) {
            max_correction = old_maxKq - maxKq;
            max_correction = __ESIMD_NS::exp2<float, Q_HEAD_PER_T, float>(
                max_correction * sycl::ext::intel::esimd::detail::log2e);
            #pragma unroll
            for (int h = 0; h < (int)Q_HEAD_PER_T; h++) {
                output.template select<HD, 1>(h * HD) =
                    output.template select<HD, 1>(h * HD) * max_correction[h];
            }
            lse = lse * max_correction;
        }
        lse = lse + kq_out;

        // V accumulation
        #pragma unroll
        for (int h = 0; h < (int)Q_HEAD_PER_T; h++) {
            float w = kq_out[h];
            simd<float, HD> vf = vIn * w;
            output.template select<HD, 1>(h * HD) =
                output.template select<HD, 1>(h * HD) + vf;
        }
    }

    // SLM reduction (same as dense phase1)
    if constexpr (sp_blk_num_per_t > 1) {
        int idx_slm = head_group_idx * sp_blk_num_per_t + sp_blk_idx;
        slm_block_store<float, Q_HEAD_PER_T>(
            slm_reduce_size + idx_slm * Q_HEAD_PER_T * sizeof(float), maxKq);
        slm_block_store<float, Q_HEAD_PER_T>(
            slm_reduce_size + slm_max_size + idx_slm * Q_HEAD_PER_T * sizeof(float), lse);
        slm_block_store<float, Q_HEAD_PER_T * HD>(
            idx_slm * Q_HEAD_PER_T * HD * sizeof(float), output);
        output = 0;
        barrier();

        if (sp_blk_idx == 0) {
            int slm_r_h_offset = (head_group_idx * sp_blk_num_per_t) * Q_HEAD_PER_T * sizeof(float);
            int slm_r_o_h_offset = (head_group_idx * sp_blk_num_per_t) * Q_HEAD_PER_T * HD * sizeof(float);

            simd<float, Q_HEAD_PER_T> max_final = FP32_MIN;
            simd<float, Q_HEAD_PER_T> lse_final = 0;

            for (int c_idx = 0; c_idx < sp_blk_num_per_t; c_idx++) {
                simd<float, Q_HEAD_PER_T> cur_max = slm_block_load<float, Q_HEAD_PER_T>(
                    slm_reduce_size + slm_r_h_offset + c_idx * Q_HEAD_PER_T * sizeof(float));
                max_final = __ESIMD_NS::max<float, Q_HEAD_PER_T, float>(cur_max, max_final);
            }

            for (int c_idx = 0; c_idx < sp_blk_num_per_t; c_idx++) {
                simd<float, Q_HEAD_PER_T> cur_max = slm_block_load<float, Q_HEAD_PER_T>(
                    slm_reduce_size + slm_r_h_offset + c_idx * Q_HEAD_PER_T * sizeof(float));
                simd<float, Q_HEAD_PER_T> cur_lse = slm_block_load<float, Q_HEAD_PER_T>(
                    slm_reduce_size + slm_max_size + slm_r_h_offset
                    + c_idx * Q_HEAD_PER_T * sizeof(float));
                simd<float, Q_HEAD_PER_T> correction =
                    __ESIMD_NS::exp2<float, Q_HEAD_PER_T, float>(
                        (cur_max - max_final) * sycl::ext::intel::esimd::detail::log2e);
                lse_final = lse_final + cur_lse * correction;

                #pragma unroll
                for (int h = 0; h < (int)Q_HEAD_PER_T; h++) {
                    float corr_h = correction[h];
                    simd<float, HD> cur_o = slm_block_load<float, HD>(
                        slm_r_o_h_offset + (c_idx * Q_HEAD_PER_T + h) * HD * sizeof(float));
                    output.template select<HD, 1>(h * HD) =
                        output.template select<HD, 1>(h * HD) + cur_o * corr_h;
                }
            }

            maxKq = max_final;
            lse = lse_final;
        } else {
            return;
        }
    }

    // Write to global scratch
    int scratch_req_base = req_idx * num_chunks_per_seq * num_heads;
    #pragma unroll
    for (int h = 0; h < (int)Q_HEAD_PER_T; h++) {
        int scratch_idx = scratch_req_base + chunk_idx * num_heads + (q_head_idx + h);
        scratch_max[scratch_idx] = maxKq[h];
        scratch_lse[scratch_idx] = lse[h];
        float* out_base = scratch_out + (int64_t)scratch_idx * HD;
        block_store<float, HD>(out_base, output.template select<HD, 1>(h * HD));
    }
}


/* ============================================================
 * SPARSE PREFILL DPAS — Union-mask-directed paged block lookup
 *
 * Adapted from flash.attn.sparse.mha128.gqa.h for paged KV cache.
 * HD=128 only. Uses 2D surface loads for K and V within each page.
 *
 * Mask: [num_kv_heads, q_blocks, 1024] uint32 (sorted kv block ids)
 * MaskCnt: [num_kv_heads, q_blocks] uint32
 *
 * Dispatch: nd_range<1>({q_blocks * num_heads * 16}, {16})
 *   16 threads per WG: each handles 16 Q positions (one per thread)
 *   One WG per (q_block, Q head) pair (same as dense prefill kernel).
 *
 * For paged addressing: instead of kCoordY = kv_idx * 64 on contiguous K,
 * we look up phys_page from block_table and compute the 2D surface Y coord.
 *
 * IMPORTANT: We update the 2D surface descriptor Y coordinate per sparse
 * block because each sparse block may come from a different physical page.
 * ============================================================ */
template<bool IS_CAUSAL, bool IS_BF16>
ESIMD_INLINE void sdp_paged_prefill_sparse_dpas_128(
    const unsigned short* __restrict__ query_ptr,         // [q_len, num_heads, 128]
    const unsigned short* __restrict__ kv_cache_ptr,      // paged [2, num_blocks, block_size, nkvh, 128]
    unsigned short* __restrict__ output_ptr,               // [q_len, num_heads, 128]
    const int* __restrict__ block_table_ptr,               // [batch, max_blocks_per_seq]
    const int* __restrict__ seq_lens_ptr,                  // [batch]
    const uint32_t* __restrict__ sparse_mask_ptr,          // [num_kv_heads, q_blocks, 1024]
    const uint32_t* __restrict__ sparse_mask_cnt_ptr,      // [num_kv_heads, q_blocks]
    int req_idx,                                            // which request
    int q_len,                                              // query length for this request
    int history_len,                                        // = seq_len - q_len
    int num_heads, int num_kv_heads,
    int block_size, int max_blocks_per_seq,
    int64_t kv_stride_split, int64_t kv_stride_block,
    float attn_scale,
    int num_heads_for_dispatch,            // = num_heads (used to decode WG id)
    nd_item<1>& ndi)
{
    using namespace sycl::ext::intel::esimd;
    using namespace sycl::ext::intel::experimental::esimd;
    namespace xmx = sycl::ext::intel::esimd::xmx;

    constexpr float LOG2E = sycl::ext::intel::esimd::detail::log2e;
    const float attnScoreMul = attn_scale * LOG2E;

    constexpr uint32_t HD = 128;
    constexpr uint32_t KV_CHUNK = 64;           // sparse block size
    constexpr uint32_t slmSizeV = 2u * 64u * 128u * sizeof(fp16);  // 32KB V ping-pong
    constexpr uint32_t slmOffsetBaseV = 0;
    constexpr uint32_t baseOffsetInc16_arr[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

    __ESIMD_NS::slm_init(slmSizeV);

    int32_t localLinearId = ndi.get_local_id(0);
    int32_t hhq  = localLinearId & 0xf;   // Q position within q_block (0..15)
    int32_t hhv  = localLinearId & 0x3;   // V scatter lane (column quarter)
    int32_t vvv  = localLinearId >> 2;     // V row group (0..3)

    int32_t wg_id = ndi.get_group(0);
    int32_t h = wg_id / num_heads_for_dispatch;  // q_block index
    int32_t v = wg_id % num_heads_for_dispatch;  // Q head index (per-Q-head dispatch)

    int32_t headIdx   = v;                 // specific Q head
    int32_t groupSize = num_heads / num_kv_heads;
    int32_t kvHeadIdx = headIdx / groupSize;  // derive KV head from Q head
    int32_t this_q_pos = h * 16 + hhq;

    // Causal boundary for this thread's Q position
    simd<int32_t, 16> causal_boundaries;
    if constexpr (IS_CAUSAL) {
        int32_t boundary = (this_q_pos < q_len)
                           ? (int32_t)(history_len + this_q_pos) : -1;
        #pragma unroll
        for (int i = 0; i < 16; i++) causal_boundaries[i] = boundary;
    }

    // Sparse mask for this q_block
    uint32_t qlen_blocks = (q_len + 15u) / 16u;
    uint32_t total_kv_blocks = *(sparse_mask_cnt_ptr
                               + (uint32_t)kvHeadIdx * qlen_blocks
                               + (uint32_t)h);
    const uint32_t* mask_ptr = sparse_mask_ptr
                             + (uint32_t)kvHeadIdx * qlen_blocks * 1024u
                             + (uint32_t)h * 1024u;

    if (total_kv_blocks == 0) {
        // No KV blocks to attend to — write zeros for this ONE Q head
        if (this_q_pos < q_len) {
            unsigned short* out_row = output_ptr +
                (int64_t)this_q_pos * num_heads * HD + (int64_t)headIdx * HD;
            simd<half_t<IS_BF16>, HD> zeros(0);
            block_store<half_t<IS_BF16>, HD>(reinterpret_cast<half_t<IS_BF16>*>(out_row), zeros);
        }
        return;
    }

    // Paged KV addressing helpers
    const int* bt_row = block_table_ptr + (int64_t)req_idx * max_blocks_per_seq;
    int32_t phys_rows_per_block = (int32_t)(kv_stride_block / (num_kv_heads * HD));
    int32_t phys_block_shift = __builtin_ctz(phys_rows_per_block);

    // KV 2D surface parameters (shared across all pages — only Y changes per page)
    uint32_t kv_head_off_u32 = (uint32_t)((int64_t)kvHeadIdx * HD);
    uint32_t kv_row_bytes = (uint32_t)((int64_t)num_kv_heads * HD * 2);  // bf16/fp16 = 2 bytes
    uint32_t kv_surf_w = kv_row_bytes - 1;
    uint32_t kv_surf_h = 0x3FFFFFU;  // safe upper bound

    // K 2D surface descriptor
    uint32_t kv_x_k = kv_head_off_u32;
    __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 1> payloadK(
        (fp16*)kv_cache_ptr, kv_surf_w, kv_surf_h, kv_surf_w, kv_x_k, 0);

    // V 2D surface descriptor (V = K + kv_stride_split offset)
    const unsigned short* kv_v_base = kv_cache_ptr + kv_stride_split;
    uint32_t vCoordX = kv_head_off_u32 + hhv * 32;
    __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 2> payloadV(
        (fp16*)kv_v_base, kv_surf_w, kv_surf_h, kv_surf_w, vCoordX, 0);

    // K prefetch descriptor
    uint32_t prefCoordX = (kv_head_off_u32 >> 1) + (localLinearId & 0x1) * 32;
    __ESIMD_ENS::config_2d_mem_access<uint32_t, 16, 8, 1> payloadPrefK(
        (uint32_t*)kv_cache_ptr, kv_surf_w, kv_surf_h, kv_surf_w, prefCoordX, 0);

    // Register state
    simd<half_t<IS_BF16>, 16 * 128> bf16QState;
    simd<float, 16 * 32>  tempBuffer;
    simd<float, 16 * 64>  tempOutput;     // QK scores: 16 Q × 64 KV
    auto tempBufferAsFp16 = tempBuffer.template bit_cast_view<fp16>();
    auto tempBufferAsHalf = tempBuffer.template bit_cast_view<half_t<IS_BF16>>();
    auto ui32Temp         = tempBuffer.template bit_cast_view<uint32_t>();
    simd<fp16,  16 * 128> finalOutput         = 0;
    simd<float, 16>     fp32SoftMaxTemp     = 0;
    simd<float, 16>     fp32HistoricMaxTemp = FP32_MIN;
    simd<uint32_t, 16>  baseOffsetInc16AsVector(baseOffsetInc16_arr);

    // ============================================================
    // Q LOAD — 2D surface, VNNI transpose (same as dense prefill)
    // ============================================================
    {
        // Q layout: [q_len, num_heads, HD] — contiguous
        // ALL threads load the SAME 16 Q rows — DPAS execution channels handle Q positions
        uint32_t widthInByteQ = num_heads * HD * sizeof(half_t<IS_BF16>) - 1;
        uint32_t heightQ = q_len - 1;
        uint32_t qCoordX = headIdx * (HD >> 1);  // in uint32 units (VNNI pair)
        uint32_t qCoordY = (uint32_t)(h * 16);   // UNIFORM: all threads same Q block

        __ESIMD_ENS::config_2d_mem_access<uint32_t, 8, 16, 1> payloadQ(
            (uint32_t*)query_ptr, widthInByteQ, heightQ, widthInByteQ, qCoordX, qCoordY);

        #pragma unroll
        for (int32_t kk = 0; kk < 8; kk++) {
            bf16QState.template bit_cast_view<uint32_t>().template select<128, 1>(128 * kk) =
                __ESIMD_ENS::lsc_load_2d<uint32_t, 8, 16, 1, true, false,
                __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadQ);
            qCoordX += 8;
            payloadQ.set_x(qCoordX);
        }
    }

    // SLM offset for this thread's V scatter
    unsigned int slmOffsetV = slmOffsetBaseV + (unsigned int)localLinearId * 512u * sizeof(fp16);
    uint32_t loopIdx = 0;

    // ============================================================
    // Helper lambda: process one KV sparse block (64 tokens)
    // Uses paged addressing via block_table lookup per sparse block.
    // ============================================================
    auto process_kv_block = [&](uint32_t kv_blk_id, bool is_last) {
        uint32_t _slmSlot = (loopIdx & 0x1u) * 64u * 128u * (uint32_t)sizeof(fp16);
        auto _tempQkAsFp16 = tempOutput.template bit_cast_view<fp16>();
        simd<fp16, 512> _fp16VState;
        tempOutput = 0;

        // Paged addressing: convert sparse kv_blk_id to physical page + offset
        int logical_tok = (int)(kv_blk_id * KV_CHUNK);
        int pg_idx = logical_tok / block_size;
        int pg_off = logical_tok % block_size;
        int phys_pg = bt_row[pg_idx];

        // 2D surface Y coordinate: physical page start + offset within page
        uint32_t Y_base = (uint32_t)(phys_pg << phys_block_shift) + (uint32_t)pg_off;

        int32_t _kv_block_start = (int32_t)(kv_blk_id * KV_CHUNK);

        // ---- Q @ K^T via DPAS (matching dense prefill pattern) ----
        // 8 dim-chunks × (4 K tiles of 16 KV tokens × 16 dims each → 8 DPAS rows)
        {
            #pragma unroll
            for (int32_t nn = 0; nn < 8; nn++) {
                payloadK.set_x(kv_x_k + 16 * nn);
                #pragma unroll
                for (int32_t l = 0; l < 4; l++) {
                    payloadK.set_y(Y_base + 16 * l);
                    tempBufferAsFp16.select<256, 1>(256 * l) =
                        __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
                        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);
                }
                #pragma unroll
                for (int32_t kk = 0; kk < 8; kk++) {
                    auto ccTile = tempOutput.select<128, 1>(128 * kk);
                    auto aaTile = bf16QState.template select<256, 1>(256 * nn);
                    auto bbTile = tempBufferAsHalf.template select<128, 1>(128 * kk);
                    ccTile = dpas<8, 8, float, float, half_t<IS_BF16>, half_t<IS_BF16>>(
                        simd<float, 128>(ccTile.data()),
                        simd<half_t<IS_BF16>, 256>(aaTile.data()),
                        simd<half_t<IS_BF16>, 128>(bbTile.data()));
                }
            }
        }

        // Causal mask (layout: 8 groups of 8 KV × 16 Q = 128*8 = 1024)
        if constexpr (IS_CAUSAL) {
            #pragma unroll
            for (int _kk = 0; _kk < 8; _kk++) {
                #pragma unroll
                for (int _m = 0; _m < 8; _m++) {
                    int32_t _kv_pos = _kv_block_start + _kk * 8 + _m;
                    simd<int32_t, 16> _v_kv_pos(_kv_pos);
                    auto _cmask = _v_kv_pos > causal_boundaries;
                    tempOutput.select<16, 1>(_kk * 128 + _m * 16).merge(simd<float, 16>(FP32_MIN), _cmask);
                }
            }
        }

        // kvSeqLen boundary check for last block
        if (is_last) {
            int kv_seq_len = seq_lens_ptr[req_idx];
            #pragma unroll
            for (int _kk = 0; _kk < 8; _kk++) {
                #pragma unroll
                for (int _m = 0; _m < 8; _m++) {
                    int32_t _kv_pos = _kv_block_start + _kk * 8 + _m;
                    simd<int32_t, 16> _v_kv_pos(_kv_pos);
                    auto _cmask = _v_kv_pos >= simd<int32_t, 16>(kv_seq_len);
                    tempOutput.select<16, 1>(_kk * 128 + _m * 16).merge(simd<float, 16>(FP32_MIN), _cmask);
                }
            }
        }

        // ---- Load V from paged cache via 2D surface ----
        {
            uint32_t vCoordY = Y_base + vvv * 16;
            payloadV.set_x(kv_head_off_u32 + hhv * 32);
            payloadV.set_y(vCoordY);
            _fp16VState =
                __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 2, false, true,
                __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);
        }

        // ---- Online softmax ----
        {
            auto _fp32CurrentMax   = tempBuffer.select<16, 1>(0);
            auto _fp32Compensation = tempBuffer.select<16, 1>(16);
            auto _fp32Exp2Temp     = tempBuffer.select<16, 1>(32);
            simd<float, 8 * 16> _ttemp;
            _fp32CurrentMax = fp32HistoricMaxTemp;

            // Row-wise max reduction across 64 KV positions
            // First pass: pairs from first 256 tempOutput elements
            #pragma unroll
            for (int _kk = 0; _kk < 4; _kk++)
                _ttemp.select<32, 1>(32 * _kk) = __ESIMD_NS::max<float, 32, float>(
                    tempOutput.select<32, 1>(64 * _kk),
                    tempOutput.select<32, 1>(64 * _kk + 32));
            // Second pass: remaining tempOutput[256..1023]
            #pragma unroll
            for (int _kkk = 0; _kkk < 6; ++_kkk) {
                #pragma unroll
                for (int _kk = 0; _kk < 4; _kk++) {
                    _ttemp.select<32, 1>(32 * _kk) =
                        __ESIMD_NS::max<float, 32, float>(
                            _ttemp.select<32, 1>(32 * _kk),
                            tempOutput.select<32, 1>((4 * _kkk + _kk) * 32 + 16 * 16));
                }
            }
            // Reduce 128 -> 64 -> 32 -> 16
            _ttemp.select<64, 1>(0) = __ESIMD_NS::max<float, 64, float>(
                _ttemp.select<64, 1>(0), _ttemp.select<64, 1>(64));
            _ttemp.select<32, 1>(0) = __ESIMD_NS::max<float, 32, float>(
                _ttemp.select<32, 1>(0), _ttemp.select<32, 1>(32));
            _ttemp.select<16, 1>(0) = __ESIMD_NS::max<float, 16, float>(
                _ttemp.select<16, 1>(0), _ttemp.select<16, 1>(16));
            _fp32CurrentMax.merge(_ttemp.select<16, 1>(0),
                                  _ttemp.select<16, 1>(0) > _fp32CurrentMax);

            _fp32Exp2Temp.select<16, 1>(0) = _fp32CurrentMax.select<16, 1>(0) * attnScoreMul;

            // exp2(score * attnScoreMul - max * attnScoreMul)
            #pragma unroll
            for (int _k = 0; _k < 8; _k++) {
                #pragma unroll
                for (int _kk = 0; _kk < 2; _kk++) {
                    _ttemp.select<16, 1>(16 * _kk)      = tempOutput.select<16, 1>(128 * _k + 32 * _kk)      * attnScoreMul - _fp32Exp2Temp.select<16, 1>(0);
                    _ttemp.select<16, 1>(16 * _kk + 32)  = tempOutput.select<16, 1>(128 * _k + 32 * _kk + 16) * attnScoreMul - _fp32Exp2Temp.select<16, 1>(0);
                }
                #pragma unroll
                for (int _kk = 0; _kk < 2; _kk++) {
                    _ttemp.select<16, 1>(16 * _kk + 64)  = tempOutput.select<16, 1>(128 * _k + 64 + 32 * _kk)      * attnScoreMul - _fp32Exp2Temp.select<16, 1>(0);
                    _ttemp.select<16, 1>(16 * _kk + 64 + 32) = tempOutput.select<16, 1>(128 * _k + 64 + 32 * _kk + 16) * attnScoreMul - _fp32Exp2Temp.select<16, 1>(0);
                }
                #pragma unroll
                for (int _kk = 0; _kk < 8; _kk++)
                    tempOutput.select<16, 1>(128 * _k + 16 * _kk) =
                        __ESIMD_NS::exp2<float, 16, float>(_ttemp.select<16, 1>(16 * _kk));
            }

            // Compensation for previous accumulated output
            _fp32Compensation = fp32HistoricMaxTemp * attnScoreMul - _fp32Exp2Temp.select<16, 1>(0);
            _fp32Compensation = __ESIMD_NS::exp2<float, 16, float>(_fp32Compensation);
            fp32SoftMaxTemp.select<16, 1>(0) *= _fp32Compensation.select<16, 1>(0);

            // Row-wise sum of exp2 scores
            #pragma unroll
            for (int _kk = 0; _kk < 4; _kk++)
                _ttemp.select<32, 1>(32 * _kk) = tempOutput.select<32, 1>(64 * _kk) +
                                                  tempOutput.select<32, 1>(64 * _kk + 32);
            #pragma unroll
            for (int _kkk = 0; _kkk < 6; ++_kkk) {
                #pragma unroll
                for (int _kk = 0; _kk < 4; _kk++) {
                    _ttemp.select<32, 1>(32 * _kk) = _ttemp.select<32, 1>(32 * _kk) +
                        tempOutput.select<32, 1>((4 * _kkk + _kk) * 32 + 16 * 16);
                }
            }
            _ttemp.select<64, 1>(0) = _ttemp.select<64, 1>(0) + _ttemp.select<64, 1>(64);
            _ttemp.select<32, 1>(0) = _ttemp.select<32, 1>(0) + _ttemp.select<32, 1>(32);
            _ttemp.select<16, 1>(0) = _ttemp.select<16, 1>(0) + _ttemp.select<16, 1>(16);
            fp32SoftMaxTemp.select<16, 1>(0) += _ttemp.select<16, 1>(0);
            fp32HistoricMaxTemp = _fp32CurrentMax;

            // Compensate finalOutput
            simd<fp16, 32> _compTemp;
            _compTemp.select<16, 1>(0)  = _fp32Compensation;
            _compTemp.select<16, 1>(16) = _fp32Compensation;
            #pragma unroll
            for (int _kk = 0; _kk < 64; _kk++)
                finalOutput.select<32, 1>(32 * _kk) =
                    finalOutput.select<32, 1>(32 * _kk) * _compTemp.select<32, 1>(0);

            // Convert attention scores to VNNI layout for Attn @ V DPAS
            // First half (KV tokens 0..31)
            #pragma unroll
            for (int _k = 0; _k < 4; _k++) {
                #pragma unroll
                for (int _kk = 0; _kk < 2; _kk++)
                    tempBufferAsFp16.select<32, 2>(128 * _k + 64 * _kk) =
                        tempOutput.select<32, 1>(128 * _k + 64 * _kk);
                #pragma unroll
                for (int _kk = 0; _kk < 2; _kk++)
                    tempBufferAsFp16.select<32, 2>(128 * _k + 64 * _kk + 1) =
                        tempOutput.select<32, 1>(128 * _k + 64 * _kk + 32);
            }
            // Second half (KV tokens 32..63)
            #pragma unroll
            for (int _k = 0; _k < 4; _k++) {
                #pragma unroll
                for (int _kk = 0; _kk < 2; _kk++)
                    _tempQkAsFp16.select<32, 2>(128 * _k + 64 * _kk) =
                        tempOutput.select<32, 1>(128 * _k + 512 + 64 * _kk);
                #pragma unroll
                for (int _kk = 0; _kk < 2; _kk++)
                    _tempQkAsFp16.select<32, 2>(128 * _k + 64 * _kk + 1) =
                        tempOutput.select<32, 1>(128 * _k + 512 + 64 * _kk + 32);
            }
        }

        // ---- bf16→fp16 V conversion (needed for DPAS which expects fp16) ----
        if constexpr (IS_BF16) {
            auto _vAsBf16 = _fp16VState.template bit_cast_view<bf16>();
            #pragma unroll
            for (int _ci = 0; _ci < 32; _ci++) {
                simd<float, 16> _cvt = _vAsBf16.select<16, 1>(16 * _ci);
                _fp16VState.select<16, 1>(16 * _ci) = _cvt;
            }
        }

        // ---- SLM scatter V ----
        {
            simd<uint32_t, 32> _simdSlmOffs;
            _simdSlmOffs.select<16, 1>(0)  = baseOffsetInc16AsVector;
            _simdSlmOffs.select<16, 1>(16) = baseOffsetInc16AsVector + 16;
            _simdSlmOffs.select<32, 1>(0)  = _simdSlmOffs.select<32, 1>(0) * 16u * (uint32_t)sizeof(fp16)
                                            + slmOffsetV + _slmSlot;
            #pragma unroll
            for (int _kk = 0; _kk < 2; _kk++)
                __ESIMD_ENS::lsc_slm_scatter<uint32_t, 8, __ESIMD_ENS::lsc_data_size::u32, 16>(
                    _simdSlmOffs.select<16, 1>(16 * _kk),
                    _fp16VState.template bit_cast_view<uint32_t>().template select<128, 1>(128 * _kk));
        }
        barrier();

        // ---- Attn @ V via DPAS ----
        {
            // First 32 KV tokens (from tempBuffer VNNI)
            #pragma unroll
            for (int _nn = 0; _nn < 2; _nn++) {
                #pragma unroll
                for (int _l = 0; _l < 2; _l++) {
                    // Load V tile from SLM
                    #pragma unroll
                    for (int _ll = 0; _ll < 2; _ll++)
                        _tempQkAsFp16.select<512, 1>(1024 + 512 * _ll) =
                            slm_block_load<fp16, 512>(slmOffsetBaseV + _slmSlot
                                + 16u * 128u * _nn * (uint32_t)sizeof(fp16)
                                + 16u * 64u * _l * (uint32_t)sizeof(fp16)
                                + 512u * _ll * (uint32_t)sizeof(fp16));
                    // 8 DPAS tiles per V column group
                    #pragma unroll
                    for (int _ll = 0; _ll < 8; _ll++) {
                        auto _ccTile = finalOutput.select<128, 1>(1024 * _l + 128 * _ll);
                        auto _aaTile = tempBufferAsFp16.select<256, 1>(256 * _nn);
                        auto _bbTile = _tempQkAsFp16.select<128, 1>(1024 + 128 * _ll);
                        _ccTile = xmx::dpas<8, 8, fp16, fp16, fp16, fp16>(
                            simd<fp16, 128>(_ccTile.data()),
                            simd<fp16, 256>(_aaTile.data()),
                            simd<fp16, 128>(_bbTile.data()));
                    }
                }
            }
            // Second 32 KV tokens (from _tempQkAsFp16 VNNI, first 512 elements)
            #pragma unroll
            for (int _nn = 0; _nn < 2; _nn++) {
                #pragma unroll
                for (int _l = 0; _l < 2; _l++) {
                    #pragma unroll
                    for (int _ll = 0; _ll < 2; _ll++)
                        _tempQkAsFp16.select<512, 1>(1024 + 512 * _ll) =
                            slm_block_load<fp16, 512>(slmOffsetBaseV + _slmSlot
                                + 16u * 128u * 2u * (uint32_t)sizeof(fp16)
                                + 16u * 128u * _nn * (uint32_t)sizeof(fp16)
                                + 16u * 64u * _l * (uint32_t)sizeof(fp16)
                                + 512u * _ll * (uint32_t)sizeof(fp16));
                    #pragma unroll
                    for (int _ll = 0; _ll < 8; _ll++) {
                        auto _ccTile = finalOutput.select<128, 1>(1024 * _l + 128 * _ll);
                        auto _aaTile = _tempQkAsFp16.select<256, 1>(256 * _nn);
                        auto _bbTile = _tempQkAsFp16.select<128, 1>(1024 + 128 * _ll);
                        _ccTile = xmx::dpas<8, 8, fp16, fp16, fp16, fp16>(
                            simd<fp16, 128>(_ccTile.data()),
                            simd<fp16, 256>(_aaTile.data()),
                            simd<fp16, 128>(_bbTile.data()));
                    }
                }
            }
        }
        loopIdx++;
    };

    // ============================================================
    // MAIN LOOP: iterate over sparse KV blocks from mask
    // ============================================================
    uint32_t total_main = total_kv_blocks - 1u;
    uint32_t full_outer = total_main / 64u;
    uint32_t partial    = total_main % 64u;

    // Full outer chunks of 64 mask entries (batch-loaded)
    for (uint32_t outer = 0; outer < full_outer; outer++) {
        simd<uint32_t, 64> chunk_ids = block_load<uint32_t, 64>(mask_ptr + outer * 64u);
        for (uint32_t inner = 0; inner < 64u; inner++) {
            process_kv_block(chunk_ids[inner], false);
        }
    }

    // Partial chunk (remaining before last)
    if (partial > 0) {
        simd<uint32_t, 64> chunk_ids = block_load<uint32_t, 64>(mask_ptr + full_outer * 64u);
        for (uint32_t inner = 0; inner < partial; inner++) {
            process_kv_block(chunk_ids[inner], false);
        }
    }

    // Last block (with kvSeqLen boundary check)
    {
        uint32_t last_kv_idx = *(mask_ptr + total_kv_blocks - 1u);
        process_kv_block(last_kv_idx, true);
    }

    // ============================================================
    // OUTPUT NORMALIZATION AND STORE (per-Q-head, matching dense kernel)
    // ============================================================
    // softMaxDivisor[16]: one value per Q position (16 SIMD lanes)
    simd<float, 16> softMaxDivisor;
    softMaxDivisor.select<16, 1>(0) = fp32SoftMaxTemp;
    softMaxDivisor = 1.0f / softMaxDivisor;

    // Normalize all 2048 finalOutput values (16 Q positions × 128 HD)
    // and convert back to output dtype. Then scatter to output.
    // Matching dense kernel: bf16QState used as staging buffer.
    {
        simd<float, 32> divMul;
        #pragma unroll
        for (int kk = 0; kk < 64; kk++) {
            simd<float, 32> f16Temp = finalOutput.select<32, 1>(32 * kk);
            divMul.select<16, 1>(0) = softMaxDivisor.select<16, 1>(0);
            divMul.select<16, 1>(16) = softMaxDivisor.select<16, 1>(0);
            f16Temp = f16Temp * divMul;
            bf16QState.template select<16, 2>(32 * kk) = f16Temp.select<16, 1>(0);
            bf16QState.template select<16, 2>(32 * kk + 1) = f16Temp.select<16, 1>(16);
        }
    }

    // Scatter output rows — write to ONE head (headIdx) for 16 Q positions
    {
        simd<uint32_t, 16> simdOffsets = baseOffsetInc16AsVector;
        simdOffsets = simdOffsets + (uint32_t)(h * 16);   // absolute Q position
        simd_mask<16> mask = simdOffsets < (uint32_t)q_len;
        simdOffsets = simdOffsets * (uint32_t)(num_heads * HD) * (uint32_t)sizeof(half_t<IS_BF16>)
                    + (uint32_t)(headIdx * HD) * (uint32_t)sizeof(half_t<IS_BF16>);
        #pragma unroll
        for (int kk = 0; kk < 16; kk++) {
            __ESIMD_ENS::lsc_scatter<uint32_t, 4,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::write_back,
                __ESIMD_ENS::cache_hint::write_back, 16, uint32_t>(
                (uint32_t*)output_ptr, simdOffsets,
                bf16QState.template bit_cast_view<uint32_t>().template select<64, 1>(64 * kk),
                mask);
            simdOffsets += 4u * (uint32_t)sizeof(uint32_t);
        }
    }
}
