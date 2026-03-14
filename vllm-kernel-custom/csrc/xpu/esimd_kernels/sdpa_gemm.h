#include "utils.h"

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using fp16 = sycl::half;
using namespace sycl;




#define TYPE_XVE   fp16
// #define TYPE_XVE   float

#define DO_PREFETCH_MLA

#define KV_BLOCK        4
#define SDP_THREAD      32
#define QK_DIM          192
#define V_DIM           128
#define XMX_NUM_V       (V_DIM / 16)
#define XMX_NUM_QK      (QK_DIM / 16)
#define XMX_M           8  //  RepeatCount
#define XMX_N           16 //  ExecutionSize
#define XMX_K           16 //  SystolicDepth * OperationsPerChannel

#define XMX_IN_A_SIZE   XMX_M* XMX_K
#define XMX_IN_B_SIZE   XMX_K* XMX_N
#define XMX_OUT_SIZE    XMX_M* XMX_N


//offset is not byte, is based on data type
ESIMD_INLINE void SDP_xmx_gemm(
      uint8_t* q_extend,
      uint8_t* k_extend,
      uint8_t* v_extend,
      uint8_t* o_extend,
      uint8_t* k_buffer,
      uint8_t* v_buffer,
      //uint32_t* kv_indptr, 
      uint8_t* kv_indices,
      unsigned num_heads,
      unsigned num_heads_kv,
      unsigned extend_seq_len,
      unsigned prefix_seq_len,
      //unsigned batch_idx,
      float attn_scale,
  nd_item<3>& ndi) {
    //constexpr float matMulQuantCoeff = 0.08838834764831844f; // 1.0f / sqrt(128.0f);
    constexpr uint32_t k_16xQK_DIM_size = 16 * QK_DIM * sizeof(fp16);
    constexpr uint32_t v_16xV_DIM_size = 16 * V_DIM * sizeof(fp16);

    constexpr uint32_t pingSLMoffset = 0;
    constexpr uint32_t pongSLMoffset = (k_16xQK_DIM_size * KV_BLOCK + v_16xV_DIM_size * KV_BLOCK);
    constexpr uint16_t mask32_const[32] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    constexpr uint32_t baseOffsetInc16[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    __ESIMD_NS::slm_init(2 * pongSLMoffset);
    int localLinearId = ndi.get_local_id(2);
    int h = ndi.get_group(0);
    int v = ndi.get_group(2);
    int head_shared = ndi.get_group(1);
    uint32_t QHEAD_DIV_KVHEAD = num_heads / num_heads_kv;
    int hq = h * QHEAD_DIV_KVHEAD + head_shared;
    int h_kv = hq / QHEAD_DIV_KVHEAD;
    int kvSeqExtendOutLoopCount = (extend_seq_len + 16 * KV_BLOCK - 1) / (16 * KV_BLOCK);
    int kvSeqPrefixOutLoopCount = (prefix_seq_len + 16 * KV_BLOCK - 1) / (16 * KV_BLOCK);

    int q_idx_8aligned = (v * SDP_THREAD + localLinearId) * 8;
    bool skipQcal = false;
    if (q_idx_8aligned >= extend_seq_len)
    {
        skipQcal = true;
    }

    int q_invalid_num = 0;
    if (q_idx_8aligned + 8 > extend_seq_len && q_idx_8aligned < extend_seq_len) {
        q_invalid_num = q_idx_8aligned + 8 - extend_seq_len;
    }


    unsigned int offsetQ = q_idx_8aligned * num_heads * QK_DIM + hq * QK_DIM; //batch
    unsigned int outputOffset = q_idx_8aligned * num_heads * V_DIM + hq * V_DIM;   //(tokens, head, dim)

    unsigned int offsetKbase = (h_kv * QK_DIM /*which header*/ + localLinearId * 8 /*which dim block*/) * sizeof(fp16);   // 1 thread reads 16x8 gather, thread offset on the token

    int localLinearId_minous_16 = localLinearId - 16;
    int idxVline = localLinearId_minous_16 % 2;
    int idxVCol = localLinearId_minous_16 / 2;
    bool is_first16 = localLinearId < 16;    //16 threads

    simd<uint32_t, 16> offsetK{baseOffsetInc16};
    simd<uint32_t, 16> offsetK_prefill;
    simd<fp16, 16> negative_mask{ -65504.0 };
    simd<uint16_t, 32> mask32(mask32_const);
    auto mask = (mask32 == 1);

//#pragma unroll
//    for (int k = 0; k < 16; k++) {
//        offsetK[k] = k;
//    }
    // (kv_idx, kv_head, head_dim)  find 16 tokens  which tokens
    offsetK = offsetK * QK_DIM * num_heads_kv * sizeof(fp16) + offsetKbase;

    //(v_idx, v_head, v_dim)
    uint32_t vStride = num_heads_kv * V_DIM;
    unsigned int offsetVBase = h_kv * V_DIM + idxVline * 8 * vStride /*0,1*/ + idxVCol/*0,1...7*/ * 16;

    simd<fp16, QK_DIM* XMX_M> qq;
    simd<fp16, QK_DIM* XMX_M> qq_before_shuffle;
    simd<TYPE_XVE, V_DIM* XMX_M> kvCacheOutFP32{ 0 };
    simd<fp16, V_DIM* XMX_M> output;

    simd<fp16, 16 * 8> kv_read;

    simd<fp16, QK_DIM* XMX_K> kk; // 16*8 x 16
    simd<fp16, XMX_K* V_DIM> vv; // 8 x  16*8 x2
    simd<fp16, XMX_OUT_SIZE> kq_out;
    simd<TYPE_XVE, XMX_OUT_SIZE> softMax;
    simd<TYPE_XVE, XMX_OUT_SIZE> softMaxSumTemp = 0.0000000001;

    bool read_flag[KV_BLOCK] = { false };
#if 1
    if (!skipQcal) {
        if (q_invalid_num == 0) {
#pragma unroll
          for (int ll = 0; ll < XMX_M; ll++) {
            qq_before_shuffle.select<QK_DIM, 1>(QK_DIM * ll) = block_load<fp16, QK_DIM>((fp16*)q_extend + offsetQ + ll * num_heads * QK_DIM);
          }
        } else {
          for (int ll = 0; ll < XMX_M - q_invalid_num; ll++) {
            qq_before_shuffle.select<QK_DIM, 1>(QK_DIM * ll) = block_load<fp16, QK_DIM>((fp16*)q_extend + offsetQ + ll * num_heads * QK_DIM);
          }

        }
    }
#endif

#pragma unroll
    for (int ll = 0; ll < XMX_M; ll++) { // 8x192
#pragma unroll
        for (int kk = 0; kk < XMX_NUM_QK; kk++) { // 192/16 = 12
            qq.select<16, 1>(kk * XMX_OUT_SIZE + ll * 16) = qq_before_shuffle.select<16, 1>(ll * QK_DIM + kk * 16);
        }
    }

    simd<TYPE_XVE, XMX_M> maxKq = -65504.0; //-65504.0;// -10000;    0xFBFF
    simd<TYPE_XVE, XMX_M> old_maxKq = -65504.0; //-65504.0;// -10000;
    simd<TYPE_XVE, XMX_M> max_correction = -65504.0; //-65504.0;// -10000;

    bool isPingSLM = true;
    uint32_t slmOffset = pingSLMoffset;



    unsigned int kvSeqLen = prefix_seq_len;
    bool fully_skip = false;

#if 1
    for (int loopIdx = 0; loopIdx < kvSeqPrefixOutLoopCount; loopIdx++) {

#if 1
        if (isPingSLM)
        {
            slmOffset = pingSLMoffset;
        }
        else
        {
            slmOffset = pongSLMoffset;
        }
        isPingSLM = !isPingSLM;

#pragma unroll
        for (int kvac = 0; kvac < KV_BLOCK; kvac++) {
            int kv_idx_16 = loopIdx * KV_BLOCK + kvac;     // KV_BLOCK  x 16
            if (kv_idx_16 * 16 < kvSeqLen)
            {
                simd<uint32_t, 16> real_rows = block_load<uint32_t, 16>((uint32_t*)kv_indices + kv_idx_16 * 16);
                simd<uint32_t, 16>offsetK_prefix = real_rows * QK_DIM * num_heads_kv * sizeof(fp16) + offsetKbase;

                int index_of_16x8_update = localLinearId;
                int offset_is_k_v = 0;
                if (localLinearId < QK_DIM / XMX_M) // 0 -23 threads
                {
                    // 16 x8 , need to read more for 192...
                    kv_read.template bit_cast_view<uint32_t>().template select<64, 1>(0) =
                        __ESIMD_ENS::lsc_gather<
                        uint32_t,
                        4,
                        __ESIMD_ENS::lsc_data_size::u32,
                        __ESIMD_ENS::cache_hint::cached,
                        __ESIMD_ENS::cache_hint::cached,
                        16,
                        uint32_t>((uint32_t*)k_buffer, offsetK_prefix);

                    slm_block_store<fp16, 128>(slmOffset + k_16xQK_DIM_size * kvac + 16 * 8 * index_of_16x8_update * sizeof(fp16), kv_read);
                }

                if (localLinearId >= SDP_THREAD - XMX_K)
                {
                    kv_read = block_load<fp16, V_DIM>((fp16*)v_buffer + real_rows[localLinearId_minous_16] * num_heads_kv * V_DIM + h_kv * V_DIM /*which header*/);

                    //shuffle and VNNI for XMX
                    offset_is_k_v = k_16xQK_DIM_size * KV_BLOCK;
                    index_of_16x8_update = localLinearId_minous_16;
                    slm_block_store<fp16, V_DIM>(slmOffset + offset_is_k_v + v_16xV_DIM_size * kvac + 16 * 8 * index_of_16x8_update * sizeof(fp16), kv_read);

                }

                // extra K,  0-7 threads, -> 8-15 -> 16-23 -> 24-31 
                //int start_t = kvac * 8;
                //if (localLinearId > start_t && localLinearId < start_t + 8)
                //{
                //    index_of_16x8_update = localLinearId - start_t + 16;
                //    offsetK_prefix = real_rows * QK_DIM * num_heads_kv * sizeof(fp16) + (h_kv * QK_DIM + index_of_16x8_update * 8) * sizeof(fp16);;
                //    // 16 x8 , need to read more for 192...
                //    kv_read.template bit_cast_view<uint32_t>().template select<64, 1>(0) =
                //        __ESIMD_ENS::lsc_gather<
                //        uint32_t,
                //        4,
                //        __ESIMD_ENS::lsc_data_size::u32,
                //        __ESIMD_ENS::cache_hint::cached,
                //        __ESIMD_ENS::cache_hint::cached,
                //        16,
                //        uint32_t>((uint32_t*)k_buffer, offsetK_prefix);

                //    slm_block_store<fp16, 128>(slmOffset + k_16xQK_DIM_size * kvac + 16 * 8 * index_of_16x8_update * sizeof(fp16), kv_read);
                //} 

            }
        } // for (int kvac = 0; kvac < KV_BLOCK; kvac++)
#endif
        barrier();
#if 1
#pragma unroll
        //Calculation.
        for (int kvac = 0; kvac < KV_BLOCK; kvac++) {
            int kv_idx_16 = loopIdx * KV_BLOCK + kvac;
            bool kv_is_16aligned = true;

            if (kv_idx_16 * 16 < kvSeqLen && !skipQcal)
            {
                if (kv_idx_16 * 16 + 16 > kvSeqLen)
                    kv_is_16aligned = false;

#ifdef DO_PREFETCH_MLA
                // prefetch
                int kv_idx_16_next = kv_idx_16 + KV_BLOCK;
                {
                    simd<uint32_t, 16> real_rows_next = block_load<uint32_t, 16>((uint32_t*)kv_indices + kv_idx_16_next * 16);
                    simd<uint32_t, 16>offsetK_next = real_rows_next * QK_DIM * num_heads_kv * sizeof(fp16) + offsetKbase;
                    int offset_is_k_v = 0;
                    if (localLinearId < QK_DIM / XMX_M)
                    {
                        __ESIMD_ENS::lsc_prefetch<
                            uint32_t,
                            4,
                            __ESIMD_ENS::lsc_data_size::u32,
                            __ESIMD_ENS::cache_hint::cached,
                            __ESIMD_ENS::cache_hint::cached,
                            16,
                            uint32_t>((uint32_t*)k_buffer, offsetK_next);
                    }
                    if (localLinearId >= SDP_THREAD - XMX_K)
                    {
                        __ESIMD_ENS::lsc_prefetch<
                            fp16,
                            V_DIM,
                            __ESIMD_ENS::lsc_data_size::default_size,
                            __ESIMD_ENS::cache_hint::cached,
                            __ESIMD_ENS::cache_hint::cached>((fp16*)v_buffer + real_rows_next[localLinearId_minous_16] * num_heads_kv * V_DIM + h_kv * V_DIM /*which header*/);
                    }
                }
#endif

                kk = slm_block_load<fp16, 16 * QK_DIM>(slmOffset + k_16xQK_DIM_size * kvac);

                // K*Q ------------------------

                simd<sycl::half, XMX_OUT_SIZE> cc_xmx{ 0 };

#pragma unroll
                for (int ww = 0; ww < XMX_NUM_QK; ww++) {

                    simd<sycl::half, XMX_OUT_SIZE> bb_xmx{ 0 }; //do not init
                    simd<sycl::half, XMX_IN_B_SIZE> aa_xmx{ 0 };

                    bb_xmx = qq.select<XMX_IN_A_SIZE, 1>(ww * XMX_IN_A_SIZE);
                    aa_xmx = kk.select<XMX_IN_B_SIZE, 1>(ww * XMX_IN_B_SIZE);
                    cc_xmx = xmx::dpas<8, 8, sycl::half, sycl::half, sycl::half, sycl::half>(cc_xmx, aa_xmx, bb_xmx);
                }
                kq_out.select<XMX_OUT_SIZE, 1>(0) = cc_xmx.select<XMX_OUT_SIZE, 1>(0);

                // soft Max ------------------------ 8x 16
                softMax = kq_out;     //what is valid lines
                softMax = softMax * attn_scale;

                // add new mask for non aligned case
                if (!kv_is_16aligned) // this thread, this kvacc 16
                {
                    uint32_t col = kvSeqLen % 16;
                    //uint32_t aligned_mask = 0xFFFF;
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        softMax.select<16, 1>(i * 16).merge(negative_mask, mask.select<16, 1>(16 - col));
                        //softMax.select<8, 1>(i * 16 + col) = -65504.0;
                    }
                }

                old_maxKq = maxKq;
#pragma unroll
                for (int ll = 0; ll < 16; ll++) {
                    maxKq = max<TYPE_XVE, 8, TYPE_XVE>(maxKq, softMax.select<8, 16>(ll));
                }

                maxKq = max<TYPE_XVE, 8, TYPE_XVE>(maxKq, old_maxKq);
#pragma unroll
                for (int ll = 0; ll < 8; ll++) {
                    softMax.select<16, 1>(ll * 16) = softMax.select<16, 1>(ll * 16) - maxKq[ll];

                    softMax.select<16, 1>(ll * 16) = pow<TYPE_XVE, 16, TYPE_XVE>(2.718f, softMax.select<16, 1>(ll * 16));
                }

                // Correct max val
                if (kv_idx_16 >= 1)
                {
                    max_correction = old_maxKq - maxKq;
                    max_correction = pow<TYPE_XVE, 8, TYPE_XVE>(2.718f, max_correction);

#pragma unroll
                    for (int ll = 0; ll < 8; ll++) {
#pragma unroll
                        for (int mm = 0; mm < 8; mm++) {
                            kvCacheOutFP32.select<16, 1>(mm * 8 * 16 + ll * 16) = kvCacheOutFP32.select<16, 1>(mm * 8 * 16 + ll * 16) * max_correction[ll];
                        }
                        softMaxSumTemp.select<16, 1>(ll * 16) = softMaxSumTemp.select<16, 1>(ll * 16) * max_correction[ll];
                    }
                }
                softMaxSumTemp.select<8 * 16, 1>(0) += softMax.select<8 * 16, 1>(0);


                kk.select<16 * V_DIM, 1>(0) = slm_block_load<fp16, 16 * V_DIM>(slmOffset + k_16xQK_DIM_size * KV_BLOCK + v_16xV_DIM_size * kvac);
                //    //shuffle and VNNI for XMX
#pragma unroll
                for (int i = 0; i < XMX_K / 2; i++) { // 16x128
#pragma unroll
                    for (int j = 0; j < XMX_NUM_V; j++) { // 128/16 = 8
                        vv.select<16, 2>(j * XMX_IN_B_SIZE + i * 32) = kk.select<16, 1>(2 * i * V_DIM + j * 16);
                        vv.select<16, 2>(j * XMX_IN_B_SIZE + i * 32 + 1) = kk.select<16, 1>((2 * i + 1) * V_DIM + j * 16);
                    }
                }

                // score*V ------------------------
#pragma unroll
                for (int nc = 0; nc < XMX_NUM_V; nc++) { // loop on non-common 128
                    simd<TYPE_XVE, XMX_OUT_SIZE> cc_xmx{ 0 };
                    simd<sycl::half, XMX_IN_A_SIZE> bb_xmx{ 0 };
                    simd<sycl::half, XMX_IN_B_SIZE> aa_xmx{ 0 };

                    cc_xmx.select<XMX_OUT_SIZE, 1>(0) = kvCacheOutFP32.select<XMX_OUT_SIZE, 1>(nc * XMX_OUT_SIZE);

                    bb_xmx = softMax.select<XMX_IN_A_SIZE, 1>(0);
                    aa_xmx = vv.select<XMX_IN_B_SIZE, 1>(nc * XMX_IN_B_SIZE);

                    cc_xmx = xmx::dpas<8, 8, TYPE_XVE, TYPE_XVE, sycl::half, sycl::half>(cc_xmx, aa_xmx, bb_xmx);

                    kvCacheOutFP32.select<XMX_OUT_SIZE, 1>(nc * XMX_OUT_SIZE) = cc_xmx.select<XMX_OUT_SIZE, 1>(0);
                }

            } // if (kv_idx_16 * 16 < kvSeqLen)
        } // for (int kvac = 0; kvac < KV_BLOCK; kvac++)
#endif
    } // for (int loopIdx = 0; loopIdx < kvSeqOutLoopCount; loopIdx++)
#endif


     //barrier();
#if 1
// causal mask
    kvSeqLen = extend_seq_len;
    for (int loopIdx = 0; loopIdx < kvSeqExtendOutLoopCount; loopIdx++) {

        fully_skip = true;

        bool is_upper_right = ((v + 1) * SDP_THREAD * 8 < (loopIdx + 1) * KV_BLOCK * 16);
        //bool is_upper_right = 0;
        if (!is_upper_right)  // only not upper right need to check if fully_skip
        {
#pragma unroll
            //skip one 256x16
            for (int kvac = 0; kvac < KV_BLOCK; kvac++)
            {
                int kv_idx_16 = loopIdx * KV_BLOCK + kvac;

                if (kv_idx_16 * 16 < kvSeqLen)
                {
                    //locate to group & kvac
                    if ((v + 1) * SDP_THREAD * 8 < loopIdx * KV_BLOCK * 16 + kvac * 16) {
                        read_flag[kvac] = false;
                    }
                    else {
                        read_flag[kvac] = true;
                        fully_skip = false;
                    }

                    //read_flag[kvac] = true;
                    //fully_skip = false;
                }

            }
        }

        //need to cal and softmax w/ some mask
        if (!fully_skip)
        {

#if 1
            if (isPingSLM)
            {
                slmOffset = pingSLMoffset;
            }
            else
            {
                slmOffset = pongSLMoffset;
            }
            isPingSLM = !isPingSLM;

#pragma unroll
            //Read. load into slm and shuffle
            for (int kvac = 0; kvac < KV_BLOCK; kvac++) {
                int kv_idx_16 = loopIdx * KV_BLOCK + kvac;     // KV_BLOCK  x 16
                bool kv_is_16aligned = true;
                if (read_flag[kvac] && kv_idx_16 * 16 < kvSeqLen)
                {
                    if (kv_idx_16 * 16 + 16 > kvSeqLen)
                        kv_is_16aligned = false;
                    int index_of_16x8_update = localLinearId;
                    int offset_is_k_v = 0;
                    if (localLinearId < QK_DIM / XMX_M) // 0 -23 threads
                    {
                        simd_mask<16> kPred = 1;
                        if (!kv_is_16aligned)
                        {
                            kPred = 0;
                            int mask = kvSeqLen % 16;
                            for (int t = 0; t < mask; t++) {
                                kPred[t] = 1;
                            }
                        }

                        // 16 x8 , need to read more for 192...
                        kv_read.template bit_cast_view<uint32_t>().template select<64, 1>(0) =
                            __ESIMD_ENS::lsc_gather<
                            uint32_t,
                            4,
                            __ESIMD_ENS::lsc_data_size::u32,
                            __ESIMD_ENS::cache_hint::cached,
                            __ESIMD_ENS::cache_hint::cached,
                            16,
                            uint32_t
                            >((uint32_t*)k_extend, offsetK + kv_idx_16 * 16 * QK_DIM * num_heads_kv/*more tokens*/ * sizeof(fp16), kPred);
                        slm_block_store<fp16, 128>(slmOffset + k_16xQK_DIM_size * kvac + 16 * 8 * index_of_16x8_update * sizeof(fp16), kv_read);
                    }

                    if (localLinearId >= SDP_THREAD - XMX_K)
                    {
                        bool store_v = true;
#if 1
                        if (kv_is_16aligned) {
                            //2D read
                            kq_out =
                                __ESIMD_ENS::lsc_load_2d<
                                fp16, 16, 8, 1,  //8x16
                                false, false,
                                __ESIMD_ENS::cache_hint::cached,
                                __ESIMD_ENS::cache_hint::cached>((fp16*)v_extend + offsetVBase + kv_idx_16 * 16 * vStride /*which 16*/,
                               16 * sizeof(fp16) - 1, 7, vStride * sizeof(fp16) - 1, 0, 0);
                        }
                        else
                        {
                            kq_out = 0;
                            int valid_row = kvSeqLen % 16;
                            if (valid_row >= 8) {
                                if (idxVline == 1)
                                {
                                    for (int i = 0; i < valid_row - 8; i++) {
                                        kq_out.select<16, 1>(i * 16) = block_load<fp16, 16>((fp16*)v_extend + offsetVBase + kv_idx_16 * 16 * vStride + i * vStride);
                                    }
                                }
                                else
                                {
                                    kq_out =
                                        __ESIMD_ENS::lsc_load_2d<
                                        fp16, 16, 8, 1,  //8x16
                                        false, false,
                                        __ESIMD_ENS::cache_hint::cached,
                                        __ESIMD_ENS::cache_hint::cached>((fp16*)v_extend + offsetVBase + kv_idx_16 * 16 * vStride /*which 16*/,
                                                16 * sizeof(fp16) - 1, 7, vStride * sizeof(fp16) - 1, 0, 0);

                                }
                            }
                            else {
                                if (idxVline == 1) {
                                    // store_v = false;
                                }
                                else {
                                    for (int i = 0; i < valid_row; i++) {
                                        kq_out.select<16, 1>(i * 16) = block_load<fp16, 16>((fp16*)v_extend + offsetVBase + kv_idx_16 * 16 * vStride + i * vStride);

                                    }
                                }
                            }


                        }
#endif

#pragma unroll
                        for (uint32_t i = 0; i < 4; ++i) {  //VNNI merge 2 lines
                            kv_read.select<16, 2>(i * 32) = kq_out.select<16, 1>(i * 32);
                            kv_read.select<16, 2>(i * 32 + 1) = kq_out.select<16, 1>(i * 32 + 16);
                        }

                        offset_is_k_v = k_16xQK_DIM_size * KV_BLOCK;
                        index_of_16x8_update = localLinearId_minous_16;

                        if (store_v)
                            slm_block_store<fp16, 128>(slmOffset + offset_is_k_v + v_16xV_DIM_size * kvac + 16 * 8 * index_of_16x8_update * sizeof(fp16), kv_read);
                    }
                }
            } // for (int kvac = 0; kvac < KV_BLOCK; kvac++)
#endif
            barrier();
#if 1
#pragma unroll
            //Calculation.
            for (int kvac = 0; kvac < KV_BLOCK; kvac++) {
                int kv_idx_16 = loopIdx * KV_BLOCK + kvac;
                bool kv_is_16aligned = true;
                //locate thread and kvac
                bool skip_cal_flag = ((q_idx_8aligned + 8) <= (loopIdx * KV_BLOCK * 16 + kvac * 16));
                //int skip_cal_flag = 0;
                if (!skip_cal_flag && kv_idx_16 * 16 < kvSeqLen && !skipQcal)
                {
                    if (kv_idx_16 * 16 + 16 > kvSeqLen)
                        kv_is_16aligned = false;

#ifdef DO_PREFETCH_MLA
                    // prefetch
                    int kv_idx_16_next = kv_idx_16 + KV_BLOCK;  //next loopIdx
                    if(kv_idx_16_next * 16 + 16 <= kvSeqLen)
                    {
                        int offset_is_k_v = 0;
                        if (localLinearId < QK_DIM / XMX_M)
                        {
                            __ESIMD_ENS::lsc_prefetch<
                                uint32_t,
                                4,
                                __ESIMD_ENS::lsc_data_size::u32,
                                __ESIMD_ENS::cache_hint::cached,
                                __ESIMD_ENS::cache_hint::cached,
                                16,
                                uint32_t
                            >((uint32_t*)k_extend, offsetK + kv_idx_16_next * 16 * QK_DIM * num_heads_kv * sizeof(fp16));
                        }
                        if (localLinearId >= SDP_THREAD - XMX_K)
                        {
                            __ESIMD_ENS::lsc_prefetch_2d<
                                fp16, 16, 8, 1,  //8x16s?
                                __ESIMD_ENS::cache_hint::cached,
                                __ESIMD_ENS::cache_hint::cached>((fp16*)v_extend + offsetVBase + kv_idx_16_next * 16 * vStride /*which 16*/,
                               16 * sizeof(fp16) - 1, 7, vStride * sizeof(fp16) - 1, 0, 0);
                        }
                    }
#endif

                    kk = slm_block_load<fp16, 16 * QK_DIM>(slmOffset + k_16xQK_DIM_size * kvac);

                    // K*Q ------------------------

                    simd<sycl::half, XMX_OUT_SIZE> cc_xmx{ 0 };
#pragma unroll
                    for (int ww = 0; ww < XMX_NUM_QK; ww++) {

                        simd<sycl::half, XMX_OUT_SIZE> bb_xmx; //do not init
                        simd<sycl::half, XMX_IN_B_SIZE> aa_xmx;

                        bb_xmx = qq.select<XMX_IN_A_SIZE, 1>(ww * XMX_IN_A_SIZE);
                        aa_xmx = kk.select<XMX_IN_B_SIZE, 1>(ww * XMX_IN_B_SIZE);
                        cc_xmx = xmx::dpas<8, 8, sycl::half, sycl::half, sycl::half, sycl::half>(cc_xmx, aa_xmx, bb_xmx);
                    }
                    kq_out.select<XMX_OUT_SIZE, 1>(0) = cc_xmx.select<XMX_OUT_SIZE, 1>(0);

                    // soft Max ------------------------ 8x 16
                    softMax = kq_out;     //what is valid lines
                    softMax = softMax * attn_scale;

                    //  + (-65504.0)
                   // softMax = softMax + kq_mask_down.select<XMX_OUT_SIZE, 1>((localLinearId%2)*XMX_OUT_SIZE);

                    bool is_diagonal = (q_idx_8aligned == (loopIdx * KV_BLOCK * 16 + kvac * 16) || (q_idx_8aligned + 8) == (loopIdx * KV_BLOCK * 16 + kvac * 16 + 16));

                    if (is_diagonal) {
                        //if (0) {
                        uint32_t start_pos = 15;
                        if (localLinearId % 2) start_pos = 7;
                        //#pragma unroll
                        //                    for (int i = 0; i < 8; i++) {
                        //                        softMax.select<16, 1>(i * 16).merge(-65504.0, down_mask >> i);
                        //                    }
                                            //softMax.select<16, 1>(0).merge<16>(-65504.0, 0x7FFF);

                        softMax.select<16, 1>(0).merge(negative_mask, mask.select<16, 1>(start_pos));
                        softMax.select<16, 1>(16).merge(negative_mask, mask.select<16, 1>(start_pos - 1));
                        softMax.select<16, 1>(32).merge(negative_mask, mask.select<16, 1>(start_pos - 2));
                        softMax.select<16, 1>(48).merge(negative_mask, mask.select<16, 1>(start_pos - 3));
                        softMax.select<16, 1>(64).merge(negative_mask, mask.select<16, 1>(start_pos - 4));
                        softMax.select<16, 1>(80).merge(negative_mask, mask.select<16, 1>(start_pos - 5));
                        softMax.select<16, 1>(96).merge(negative_mask, mask.select<16, 1>(start_pos - 6));
                        softMax.select<16, 1>(112).merge(negative_mask, mask.select<16, 1>(start_pos - 7));

                        //#pragma unroll
                        //                    for (int i = 0; i < 8; i++) {
                        //                        softMax.select<16, 1>(i * 16).merge(-65504.0, start_pos-i);
                        //                    }

                    }

                    // add new mask for non aligned case
                    if (!kv_is_16aligned) // this thread, this kvacc 16
                    {
                        uint32_t col = kvSeqLen % 16;
                        //uint32_t aligned_mask = 0xffff;
#pragma unroll
                        for (int i = 0; i < 8; i++) {
                            softMax.select<16, 1>(i * 16).merge(negative_mask, mask.select<16, 1>(16 - col));
                            //softmax.select<8, 1>(i * 16 + col) = -65504.0;
                        }
                    }

                    old_maxKq = maxKq;
#pragma unroll
                    for (int ll = 0; ll < 16; ll++) {
                        maxKq = max<TYPE_XVE, 8, TYPE_XVE>(maxKq, softMax.select<8, 16>(ll));
                    }

                    maxKq = max<TYPE_XVE, 8, TYPE_XVE>(maxKq, old_maxKq);

#pragma unroll
                    for (int ll = 0; ll < 8; ll++) {
                        softMax.select<16, 1>(ll * 16) = softMax.select<16, 1>(ll * 16) - maxKq[ll];
                        softMax.select<16, 1>(ll * 16) = pow<TYPE_XVE, 16, TYPE_XVE>(2.718f, softMax.select<16, 1>(ll * 16));
                    }

                    // Correct max val
                    //if (kv_idx_16 >= 1)
                    {
                        max_correction = old_maxKq - maxKq;
                        max_correction = pow<TYPE_XVE, 8, TYPE_XVE>(2.718f, max_correction);

#pragma unroll
                        for (int ll = 0; ll < 8; ll++) {
#pragma unroll
                            for (int mm = 0; mm < 8; mm++) {
                                kvCacheOutFP32.select<16, 1>(mm * 8 * 16 + ll * 16) = kvCacheOutFP32.select<16, 1>(mm * 8 * 16 + ll * 16) * max_correction[ll];
                            }
                            softMaxSumTemp.select<16, 1>(ll * 16) = softMaxSumTemp.select<16, 1>(ll * 16) * max_correction[ll];
                        }
                    }
                    softMaxSumTemp.select<8 * 16, 1>(0) += softMax.select<8 * 16, 1>(0);

                    vv = slm_block_load<fp16, 16 * V_DIM>(slmOffset + k_16xQK_DIM_size * KV_BLOCK + v_16xV_DIM_size * kvac);

                    //spill 17664
                //if (!kv_is_16aligned) // in case that the value of V is nan
                //{
                //    uint32_t row = kvSeqLen % 16;
                //    #pragma unroll
                //    for (int i = row; i < 16; i++) {
                //        vv.select<V_DIM, 1>(i * V_DIM) = 0;
                //    }
                //}
                // score*V ------------------------
#pragma unroll
                    for (int nc = 0; nc < XMX_NUM_V; nc++) { // loop on non-common 128
                        simd<TYPE_XVE, XMX_OUT_SIZE> cc_xmx{ 0 };
                        simd<sycl::half, XMX_IN_A_SIZE> bb_xmx{ 0 };
                        simd<sycl::half, XMX_IN_B_SIZE> aa_xmx{ 0 };

                        cc_xmx.select<XMX_OUT_SIZE, 1>(0) = kvCacheOutFP32.select<XMX_OUT_SIZE, 1>(nc * XMX_OUT_SIZE);

                        bb_xmx = softMax.select<XMX_IN_A_SIZE, 1>(0);
                        aa_xmx = vv.select<XMX_IN_B_SIZE, 1>(nc * XMX_IN_B_SIZE);
                        cc_xmx = xmx::dpas<8, 8, TYPE_XVE, TYPE_XVE, sycl::half, sycl::half>(cc_xmx, aa_xmx, bb_xmx);

                        kvCacheOutFP32.select<XMX_OUT_SIZE, 1>(nc * XMX_OUT_SIZE) = cc_xmx.select<XMX_OUT_SIZE, 1>(0);
                    }

                } // if (kv_idx_16 * 16 < kvSeqLen)
            } // for (int kvac = 0; kvac < KV_BLOCK; kvac++)
#endif
        } // if (!fully_skip)
    } // for (int loopIdx = 0; loopIdx < kvSeqOutLoopCount; loopIdx++)
#endif

    if (skipQcal) return;   //the thread is not 256 aligned but 8 aligned

#if 1
#pragma unroll
    for (int qq = 0; qq < XMX_M; qq++) {
        softMaxSumTemp[qq * 16] = sycl::ext::intel::esimd::detail::sum<TYPE_XVE, TYPE_XVE, XMX_N>(softMaxSumTemp.select<XMX_N, 1>(qq * XMX_N));

#pragma unroll
        for (int ll = 0; ll < XMX_NUM_V; ll++) {
            kvCacheOutFP32.select<16, 1>(ll * XMX_OUT_SIZE + qq * 16) = kvCacheOutFP32.select<16, 1>(ll * XMX_OUT_SIZE + qq * 16) / softMaxSumTemp[qq * 16];
        }
    }

    if (q_invalid_num == 0) {
   
    #pragma unroll
        for (int ll = 0; ll < XMX_M; ll++) { // unroll...
    #pragma unroll
            for (int kk = 0; kk < XMX_NUM_V; kk++) { // 8x16 -> 128
                output.select<16, 1>(ll * V_DIM + kk * 16) = kvCacheOutFP32.select<16, 1>(kk * V_DIM + ll * 16);
            }
        }

    #pragma unroll
        for (int ll = 0; ll < XMX_M; ll++) { // unroll...
            block_store<fp16, V_DIM>((fp16*)o_extend + outputOffset + ll * num_heads * V_DIM, output.select<V_DIM, 1>(ll * V_DIM));
        }
    } else {
        for (int ll = 0; ll < XMX_M - q_invalid_num; ll++) { // unroll...
    #pragma unroll
            for (int kk = 0; kk < XMX_NUM_V; kk++) { // 8x16 -> 128
                output.select<16, 1>(ll * V_DIM + kk * 16) = kvCacheOutFP32.select<16, 1>(kk * V_DIM + ll * 16);
            }
        }

        for (int ll = 0; ll < XMX_M - q_invalid_num; ll++) { // unroll...
            block_store<fp16, V_DIM>((fp16*)o_extend + outputOffset + ll * num_heads * V_DIM, output.select<V_DIM, 1>(ll * V_DIM));
        }
    }

#endif

}