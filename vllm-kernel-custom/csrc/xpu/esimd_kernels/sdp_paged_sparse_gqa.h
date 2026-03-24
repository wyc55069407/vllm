// Paged Sparse Flash Attention kernel - headgroup dispatch
// Based on flash.attn.sparse.mha128.gqa.h with paged KV cache support.
//
// KV cache layout: [num_blocks, 2, block_size, num_kv_heads, head_dim] (physical)
//   Logical view via permute(1,0,2,3,4): [2, num_blocks, block_size, num_kv_heads, head_dim]
//   K = kv_cache[0, ...], V = kv_cache[1, ...] = kv_cache_ptr + kv_stride_split
//
// mask shape:     [num_kv_heads, qlen/16, 1024]  uint32 - kv block indices (block size=64), sorted
// mask_cnt shape: [num_kv_heads, qlen/16]        uint32 - count of valid kv blocks per entry
//
// Dispatch: nd_range<2>({groupH, groupV*16}, {1, 16})
//   groupH = (qlen+15)/16
//   groupV = num_kv_heads
//   16 threads in dim1: each thread handles 1 q position × 16 q heads
//
// Changes from non-paged reference:
//   1. kState/vState → kv_cache_ptr + block_table_ptr + strides
//   2. kv_idx * 64 → block_table lookup for physical Y coordinate
//   3. Surface height = 0x3FFFFF (safe upper bound for scattered pages)
//   4. normAlpha removed (output = softmax(QK^T) @ V without extra scaling)

// bf16 helper: use bf16 type alias from sdp_paged.h (or define here)
#ifndef HALF_T_DEFINED_SPARSE_GQA
#define HALF_T_DEFINED_SPARSE_GQA
template<bool B> using half_t_gqa = std::conditional_t<B, bf16, fp16>;
#endif

template<bool IS_CAUSAL, bool IS_BF16 = false>
ESIMD_INLINE void flashAttnSparseMha128GQA_paged(
  const unsigned short* __restrict__ query_ptr,     // [q_len, num_heads, 128] as fp16
  const unsigned short* __restrict__ kv_cache_ptr,  // paged KV cache base (K surface)
  uint8_t*  out,
  uint32_t* sparseMask,
  uint32_t* sparseMaskCnt,
  const int* __restrict__ block_table_ptr,          // [max_blocks_per_seq]
  uint32_t  activationLength,   // q_len
  uint32_t  kvSeqLen,
  uint32_t  history_len,
  uint32_t  headQ,
  uint32_t  headKv,
  int       block_size,         // vLLM page size (e.g. 128)
  int64_t   kv_stride_split,    // stride(0) of logical view = block_size * nkvh * hd
  int64_t   kv_stride_block,    // stride(1) of logical view = 2 * block_size * nkvh * hd
  float     attn_scale,
  sycl::nd_item<2>& ndi)
{
  using namespace sycl::ext::intel::esimd;
  using namespace sycl::ext::intel::experimental::esimd;

  const float attnScoreMul = attn_scale * sycl::ext::intel::esimd::detail::log2e;
  constexpr uint32_t HD = 128;
  constexpr uint32_t KV_CHUNK = 64;
  constexpr uint32_t slmSizeV = 2u * 64u * 128u * sizeof(fp16);
  constexpr uint32_t baseOffsetInc16_arr[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  __ESIMD_NS::slm_init(slmSizeV);
  constexpr uint32_t slmOffsetBaseV = 0;

  int32_t localLinearId = ndi.get_local_id(1);
  int32_t hhq  = localLinearId & 0xf;
  int32_t hhv  = localLinearId & 0x3;
  int32_t vvv  = localLinearId >> 2;

  int32_t h = ndi.get_group(0);   // q workgroup index
  int32_t v = ndi.get_group(1);   // kv head index

  int32_t kvHeadIdx = v;
  int32_t groupSize = headQ / headKv;
  int32_t headIdx   = kvHeadIdx * groupSize;
  int32_t this_q_pos = h * 16 + hhq;

  // Causal boundary
  simd<int32_t, 16> causal_boundaries;
  if constexpr (IS_CAUSAL) {
    int32_t boundary = (this_q_pos < (int32_t)activationLength)
                       ? (int32_t)(history_len + this_q_pos) : -1;
    #pragma unroll
    for (int i = 0; i < 16; i++) causal_boundaries[i] = boundary;
  }

  // Sparse mask
  uint32_t qlen_blocks     = (activationLength + 15u) / 16u;
  uint32_t total_kv_blocks = *(sparseMaskCnt + (uint32_t)kvHeadIdx * qlen_blocks + (uint32_t)h);
  uint32_t* mask_ptr       = sparseMask
                           + (uint32_t)kvHeadIdx * qlen_blocks * 1024u
                           + (uint32_t)h * 1024u;

  uint32_t total_main = total_kv_blocks - 1u;
  uint32_t full_outer = total_main / 64u;
  uint32_t partial    = total_main % 64u;

  // Paged KV addressing
  int32_t phys_rows_per_block = (int32_t)(kv_stride_block / (headKv * HD));
  int32_t phys_block_shift = __builtin_ctz(phys_rows_per_block);

  // 2D surface parameters for paged KV
  uint32_t kv_head_off = (uint32_t)((int64_t)kvHeadIdx * HD);
  uint32_t kv_row_bytes = (uint32_t)((int64_t)headKv * HD * 2);  // fp16 = 2 bytes
  uint32_t kv_surf_w = kv_row_bytes - 1;
  uint32_t kv_surf_h = 0x3FFFFFU;  // safe upper bound for paged surfaces

  const unsigned short* kv_v_base = kv_cache_ptr + kv_stride_split;

  uint32_t kCoordX = kv_head_off;
  uint32_t vCoordX = kv_head_off + (uint32_t)hhv * 32u;

  __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 1> payloadK(
    (fp16*)kv_cache_ptr, kv_surf_w, kv_surf_h, kv_surf_w, kCoordX, 0u);
  __ESIMD_ENS::config_2d_mem_access<fp16, 16, 16, 2> payloadV(
    (fp16*)kv_v_base, kv_surf_w, kv_surf_h, kv_surf_w, vCoordX, 0u);

  // Register state
  simd<fp16,  16*128> fp16QState;
  simd<float, 16*32>  tempBuffer;
  simd<float, 16*64>  tempOutput;
  auto tempBufferAsFp16 = tempBuffer.template bit_cast_view<fp16>();
  auto ui32Temp         = tempBuffer.template bit_cast_view<uint32_t>();
  simd<fp16,  16*128> finalOutput         = 0;
  simd<float, 16>     fp32SoftMaxTemp     = 0;
  simd<float, 16>     fp32HistoricMaxTemp = FP32_MIN;
  simd<uint32_t, 16>  baseOffsetInc16AsVector(baseOffsetInc16_arr);

  unsigned int slmOffsetV = slmOffsetBaseV + (unsigned int)localLinearId * 512u * sizeof(fp16);

  // Q load — same as reference, Q is contiguous [q_len, num_heads, HD]
  {
    uint32_t widthInByteQ  = HD * (uint32_t)sizeof(fp16) - 1u;
    uint32_t heightQ       = headQ * activationLength - 1u;
    uint32_t qCoordX = 0;
    uint32_t qCoordY = (uint32_t)h * 16u * headQ
                     + (uint32_t)hhq * headQ
                     + (uint32_t)headIdx;

    __ESIMD_ENS::config_2d_mem_access<uint32_t, 8, 16, 1> payloadQ(
      (uint32_t*)query_ptr, widthInByteQ, heightQ, widthInByteQ, qCoordX, qCoordY);

    #pragma unroll
    for (int32_t kk = 0; kk < 8; kk++) {
      fp16QState.template bit_cast_view<uint32_t>().select<128, 1>(128 * kk) =
        __ESIMD_ENS::lsc_load_2d<uint32_t, 8, 16, 1, true, false,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadQ);
      qCoordX += 8;
      payloadQ.set_x(qCoordX);
    }
  }

  uint32_t loopIdx = 0;

  // =========================================================
  // Helper: convert sparse kv_blk_id to physical 2D Y coordinate
  // =========================================================
  auto get_Y_base = [&](uint32_t kv_blk_id) -> uint32_t {
    int logical_tok = (int)(kv_blk_id * KV_CHUNK);
    int pg_idx = logical_tok / block_size;
    int pg_off = logical_tok % block_size;
    int phys_pg = block_table_ptr[pg_idx];
    return (uint32_t)(phys_pg << phys_block_shift) + (uint32_t)pg_off;
  };

  // =========================================================
  // FULL OUTER CHUNKS
  // =========================================================
  for (uint32_t outer = 0; outer < full_outer; outer++) {
    simd<uint32_t, 64> chunk = block_load<uint32_t, 64>(mask_ptr + outer * 64u);

    for (uint32_t inner = 0; inner < 64u; inner++) {

      uint32_t kv_idx   = chunk[inner];
      uint32_t _slmSlot = (loopIdx & 0x1u) * 64u * 128u * (uint32_t)sizeof(fp16);
      auto _tempQkAsFp16 = tempOutput.template bit_cast_view<fp16>();
      simd<fp16, 512> _fp16VState;
      tempOutput = 0;

      uint32_t Y_base = get_Y_base(kv_idx);
      int32_t  _kv_block_start = (int32_t)(kv_idx * KV_CHUNK);

      // Q @ K^T
      #pragma unroll
      for (int32_t nn = 0; nn < 8; nn++) {
        payloadK.set_x(kCoordX + 16 * nn);
        #pragma unroll
        for (int32_t l = 0; l < 4; l++) {
          payloadK.set_y(Y_base + 16u * l);
          tempBufferAsFp16.select<256, 1>(256 * l) =
            __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
            __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);
        }
        #pragma unroll
        for (int32_t kk = 0; kk < 8; kk++) {
          auto ccTile = tempOutput.select<128, 1>(128 * kk);
          auto aaTile = fp16QState.select<256, 1>(256 * nn);
          auto bbTile = tempBufferAsFp16.select<128, 1>(128 * kk);
          if constexpr (IS_BF16) {
            simd<bf16, 256> aa_bf16 = fp16QState.template bit_cast_view<bf16>().template select<256,1>(256*nn);
            simd<bf16, 128> bb_bf16 = tempBuffer.template bit_cast_view<bf16>().template select<128,1>(128*kk);
            ccTile = dpas<8, 8, float, float, bf16, bf16>(
              simd<float, 128>(ccTile.data()), aa_bf16, bb_bf16);
          } else {
            ccTile = dpas<8, 8, float, float, fp16, fp16>(
              simd<float, 128>(ccTile.data()),
              simd<fp16, 256>(aaTile.data()),
              simd<fp16, 128>(bbTile.data()));
          }
        }
      }

      // Causal mask
      if constexpr (IS_CAUSAL) {
        #pragma unroll
        for (int _kk = 0; _kk < 8; _kk++) {
          #pragma unroll
          for (int _m = 0; _m < 8; _m++) {
            int32_t _kv_pos = _kv_block_start + _kk * 8 + _m;
            simd<int, 16> _v_kv_pos(_kv_pos);
            auto _cmask = _v_kv_pos > causal_boundaries;
            tempOutput.select<16, 1>(_kk * 128 + _m * 16).merge(FP32_MIN, _cmask);
          }
        }
      }

      // Load V from paged cache
      payloadV.set_y(Y_base + (uint32_t)vvv * 16u);
      _fp16VState = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 2, false, true,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);

      // bf16 → fp16 conversion for V (DPAS Attn@V requires fp16)
      if constexpr (IS_BF16) {
        #pragma unroll
        for (int _ci = 0; _ci < 32; _ci++) {
          simd<float, 16> _cv = _fp16VState.template bit_cast_view<bf16>().template select<16,1>(16*_ci);
          _fp16VState.template select<16,1>(16*_ci) = _cv;
        }
      }

      // Online softmax (identical to reference)
      {
        auto _fp32CurrentMax   = tempBuffer.select<16, 1>(0);
        auto _fp32Compensation = tempBuffer.select<16, 1>(16);
        auto _fp32Exp2Temp     = tempBuffer.select<16, 1>(32);
        simd<float, 8*16> _ttemp;
        _fp32CurrentMax = fp32HistoricMaxTemp;

        #pragma unroll
        for (int _kk = 0; _kk < 4; _kk++)
          _ttemp.select<32,1>(32*_kk) = __ESIMD_NS::max<float,32,float>(
            tempOutput.select<32,1>(64*_kk), tempOutput.select<32,1>(64*_kk+32));
        #pragma unroll
        for (int _kkk = 0; _kkk < 6; ++_kkk)
          #pragma unroll
          for (int _kk = 0; _kk < 4; _kk++)
            _ttemp.select<32,1>(32*_kk) = __ESIMD_NS::max<float,32,float>(
              _ttemp.select<32,1>(32*_kk),
              tempOutput.select<32,1>((4*_kkk+_kk)*32+16*16));
        _ttemp.select<64,1>(0)=__ESIMD_NS::max<float,64,float>(_ttemp.select<64,1>(0),_ttemp.select<64,1>(64));
        _ttemp.select<32,1>(0)=__ESIMD_NS::max<float,32,float>(_ttemp.select<32,1>(0),_ttemp.select<32,1>(32));
        _ttemp.select<16,1>(0)=__ESIMD_NS::max<float,16,float>(_ttemp.select<16,1>(0),_ttemp.select<16,1>(16));
        _fp32CurrentMax.merge(_ttemp.select<16,1>(0), _ttemp.select<16,1>(0) > _fp32CurrentMax);

        _fp32Exp2Temp.select<16,1>(0) = _fp32CurrentMax.select<16,1>(0) * attnScoreMul;

        #pragma unroll
        for (int _k = 0; _k < 8; _k++) {
          #pragma unroll
          for (int _kk = 0; _kk < 2; _kk++) {
            _ttemp.select<16,1>(16*_kk)    = tempOutput.select<16,1>(128*_k+32*_kk)    * attnScoreMul - _fp32Exp2Temp.select<16,1>(0);
            _ttemp.select<16,1>(16*_kk+32) = tempOutput.select<16,1>(128*_k+32*_kk+16) * attnScoreMul - _fp32Exp2Temp.select<16,1>(0);
          }
          #pragma unroll
          for (int _kk = 0; _kk < 2; _kk++) {
            _ttemp.select<16,1>(16*_kk+64)    = tempOutput.select<16,1>(128*_k+64+32*_kk)    * attnScoreMul - _fp32Exp2Temp.select<16,1>(0);
            _ttemp.select<16,1>(16*_kk+64+32) = tempOutput.select<16,1>(128*_k+64+32*_kk+16) * attnScoreMul - _fp32Exp2Temp.select<16,1>(0);
          }
          #pragma unroll
          for (int _kk = 0; _kk < 8; _kk++)
            tempOutput.select<16,1>(128*_k+16*_kk) = __ESIMD_NS::exp2<float,16,float>(_ttemp.select<16,1>(16*_kk));
        }
        _fp32Compensation = fp32HistoricMaxTemp * attnScoreMul - _fp32Exp2Temp.select<16,1>(0);
        _fp32Compensation = __ESIMD_NS::exp2<float,16,float>(_fp32Compensation);
        fp32SoftMaxTemp.select<16,1>(0) *= _fp32Compensation.select<16,1>(0);
        #pragma unroll
        for (int _kk = 0; _kk < 4; _kk++)
          _ttemp.select<32,1>(32*_kk) = tempOutput.select<32,1>(64*_kk)+tempOutput.select<32,1>(64*_kk+32);
        #pragma unroll
        for (int _kkk = 0; _kkk < 6; ++_kkk)
          #pragma unroll
          for (int _kk = 0; _kk < 4; _kk++)
            _ttemp.select<32,1>(32*_kk) = _ttemp.select<32,1>(32*_kk)+tempOutput.select<32,1>((4*_kkk+_kk)*32+16*16);
        _ttemp.select<64,1>(0)=_ttemp.select<64,1>(0)+_ttemp.select<64,1>(64);
        _ttemp.select<32,1>(0)=_ttemp.select<32,1>(0)+_ttemp.select<32,1>(32);
        _ttemp.select<16,1>(0)=_ttemp.select<16,1>(0)+_ttemp.select<16,1>(16);
        fp32SoftMaxTemp.select<16,1>(0) += _ttemp.select<16,1>(0);
        fp32HistoricMaxTemp = _fp32CurrentMax;

        simd<fp16, 32> _compTemp;
        _compTemp.select<16,1>(0)  = _fp32Compensation;
        _compTemp.select<16,1>(16) = _fp32Compensation;
        #pragma unroll
        for (int _kk = 0; _kk < 64; _kk++)
          finalOutput.select<32,1>(32*_kk) = finalOutput.select<32,1>(32*_kk) * _compTemp.select<32,1>(0);

        // VNNI layout conversion
        #pragma unroll
        for (int _k = 0; _k < 4; _k++) {
          #pragma unroll
          for (int _kk = 0; _kk < 2; _kk++)
            tempBufferAsFp16.select<32,2>(128*_k+64*_kk)   = tempOutput.select<32,1>(128*_k+64*_kk);
          #pragma unroll
          for (int _kk = 0; _kk < 2; _kk++)
            tempBufferAsFp16.select<32,2>(128*_k+64*_kk+1) = tempOutput.select<32,1>(128*_k+64*_kk+32);
        }
        #pragma unroll
        for (int _k = 0; _k < 4; _k++) {
          #pragma unroll
          for (int _kk = 0; _kk < 2; _kk++)
            _tempQkAsFp16.select<32,2>(128*_k+64*_kk)   = tempOutput.select<32,1>(128*_k+512+64*_kk);
          #pragma unroll
          for (int _kk = 0; _kk < 2; _kk++)
            _tempQkAsFp16.select<32,2>(128*_k+64*_kk+1) = tempOutput.select<32,1>(128*_k+512+64*_kk+32);
        }
      }

      // SLM scatter V (identical to reference)
      {
        simd<uint32_t, 32> _simdSlmOffs;
        _simdSlmOffs.select<16,1>(0)  = baseOffsetInc16AsVector;
        _simdSlmOffs.select<16,1>(16) = baseOffsetInc16AsVector + 16;
        _simdSlmOffs.select<32,1>(0)  = _simdSlmOffs.select<32,1>(0) * 16u * (uint32_t)sizeof(fp16)
                                       + slmOffsetV + _slmSlot;
        #pragma unroll
        for (int _kk = 0; _kk < 2; _kk++)
          __ESIMD_ENS::lsc_slm_scatter<uint32_t, 8, __ESIMD_ENS::lsc_data_size::u32, 16>(
            _simdSlmOffs.select<16,1>(16*_kk),
            _fp16VState.template bit_cast_view<uint32_t>().select<128,1>(128*_kk));
      }
      barrier();

      // Attn @ V (identical to reference)
      {
        #pragma unroll
        for (int _nn = 0; _nn < 2; _nn++) {
          #pragma unroll
          for (int _l = 0; _l < 2; _l++) {
            #pragma unroll
            for (int _ll = 0; _ll < 2; _ll++)
              _tempQkAsFp16.select<512,1>(1024+512*_ll) =
                slm_block_load<fp16, 512>(slmOffsetBaseV + _slmSlot +
                  16*128*_nn*(uint32_t)sizeof(fp16) + 16*64*_l*(uint32_t)sizeof(fp16) +
                  512*_ll*(uint32_t)sizeof(fp16));
            #pragma unroll
            for (int _ll = 0; _ll < 8; _ll++) {
              auto _ccTile = finalOutput.select<128,1>(1024*_l+128*_ll);
              auto _aaTile = tempBufferAsFp16.select<256,1>(256*_nn);
              auto _bbTile = _tempQkAsFp16.select<128,1>(1024+128*_ll);
              _ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
                simd<fp16,128>(_ccTile.data()),
                simd<fp16,256>(_aaTile.data()),
                simd<fp16,128>(_bbTile.data()));
            }
          }
        }
        #pragma unroll
        for (int _nn = 0; _nn < 2; _nn++) {
          #pragma unroll
          for (int _l = 0; _l < 2; _l++) {
            #pragma unroll
            for (int _ll = 0; _ll < 2; _ll++)
              _tempQkAsFp16.select<512,1>(1024+512*_ll) =
                slm_block_load<fp16, 512>(slmOffsetBaseV + _slmSlot +
                  16*128*2u*(uint32_t)sizeof(fp16) + 16*128*_nn*(uint32_t)sizeof(fp16) +
                  16*64*_l*(uint32_t)sizeof(fp16) + 512*_ll*(uint32_t)sizeof(fp16));
            #pragma unroll
            for (int _ll = 0; _ll < 8; _ll++) {
              auto _ccTile = finalOutput.select<128,1>(1024*_l+128*_ll);
              auto _aaTile = _tempQkAsFp16.select<256,1>(256*_nn);
              auto _bbTile = _tempQkAsFp16.select<128,1>(1024+128*_ll);
              _ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
                simd<fp16,128>(_ccTile.data()),
                simd<fp16,256>(_aaTile.data()),
                simd<fp16,128>(_bbTile.data()));
            }
          }
        }
      }
      loopIdx++;
    } // inner 64
  } // full outer

  // =========================================================
  // PARTIAL CHUNK
  // =========================================================
  {
    simd<uint32_t, 64> chunk = block_load<uint32_t, 64>(mask_ptr + full_outer * 64u);

    for (uint32_t inner = 0; inner < partial; inner++) {

      uint32_t kv_idx   = chunk[inner];
      uint32_t _slmSlot = (loopIdx & 0x1u) * 64u * 128u * (uint32_t)sizeof(fp16);
      auto _tempQkAsFp16 = tempOutput.template bit_cast_view<fp16>();
      simd<fp16, 512> _fp16VState;
      tempOutput = 0;

      uint32_t Y_base = get_Y_base(kv_idx);
      int32_t  _kv_block_start = (int32_t)(kv_idx * KV_CHUNK);

      // Q @ K^T
      #pragma unroll
      for (int32_t nn = 0; nn < 8; nn++) {
        payloadK.set_x(kCoordX + 16 * nn);
        #pragma unroll
        for (int32_t l = 0; l < 4; l++) {
          payloadK.set_y(Y_base + 16u * l);
          tempBufferAsFp16.select<256, 1>(256 * l) =
            __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
            __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);
        }
        #pragma unroll
        for (int32_t kk = 0; kk < 8; kk++) {
          auto ccTile = tempOutput.select<128, 1>(128 * kk);
          auto aaTile = fp16QState.select<256, 1>(256 * nn);
          auto bbTile = tempBufferAsFp16.select<128, 1>(128 * kk);
          if constexpr (IS_BF16) {
            simd<bf16, 256> aa_bf16 = fp16QState.template bit_cast_view<bf16>().template select<256,1>(256*nn);
            simd<bf16, 128> bb_bf16 = tempBuffer.template bit_cast_view<bf16>().template select<128,1>(128*kk);
            ccTile = dpas<8, 8, float, float, bf16, bf16>(
              simd<float, 128>(ccTile.data()), aa_bf16, bb_bf16);
          } else {
            ccTile = dpas<8, 8, float, float, fp16, fp16>(
              simd<float, 128>(ccTile.data()),
              simd<fp16, 256>(aaTile.data()),
              simd<fp16, 128>(bbTile.data()));
          }
        }
      }

      if constexpr (IS_CAUSAL) {
        #pragma unroll
        for (int _kk = 0; _kk < 8; _kk++) {
          #pragma unroll
          for (int _m = 0; _m < 8; _m++) {
            int32_t _kv_pos = _kv_block_start + _kk * 8 + _m;
            simd<int, 16> _v_kv_pos(_kv_pos);
            auto _cmask = _v_kv_pos > causal_boundaries;
            tempOutput.select<16, 1>(_kk * 128 + _m * 16).merge(FP32_MIN, _cmask);
          }
        }
      }

      // Load V from paged cache
      payloadV.set_y(Y_base + (uint32_t)vvv * 16u);
      _fp16VState = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 2, false, true,
        __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);

      // bf16 → fp16 conversion for V (DPAS Attn@V requires fp16)
      if constexpr (IS_BF16) {
        #pragma unroll
        for (int _ci = 0; _ci < 32; _ci++) {
          simd<float, 16> _cv = _fp16VState.template bit_cast_view<bf16>().template select<16,1>(16*_ci);
          _fp16VState.template select<16,1>(16*_ci) = _cv;
        }
      }

      // Online softmax + VNNI + SLM scatter + Attn@V (identical to full outer block above)
      {
        auto _fp32CurrentMax   = tempBuffer.select<16, 1>(0);
        auto _fp32Compensation = tempBuffer.select<16, 1>(16);
        auto _fp32Exp2Temp     = tempBuffer.select<16, 1>(32);
        simd<float, 8*16> _ttemp;
        _fp32CurrentMax = fp32HistoricMaxTemp;

        #pragma unroll
        for (int _kk = 0; _kk < 4; _kk++)
          _ttemp.select<32,1>(32*_kk) = __ESIMD_NS::max<float,32,float>(
            tempOutput.select<32,1>(64*_kk), tempOutput.select<32,1>(64*_kk+32));
        #pragma unroll
        for (int _kkk = 0; _kkk < 6; ++_kkk)
          #pragma unroll
          for (int _kk = 0; _kk < 4; _kk++)
            _ttemp.select<32,1>(32*_kk) = __ESIMD_NS::max<float,32,float>(
              _ttemp.select<32,1>(32*_kk),
              tempOutput.select<32,1>((4*_kkk+_kk)*32+16*16));
        _ttemp.select<64,1>(0)=__ESIMD_NS::max<float,64,float>(_ttemp.select<64,1>(0),_ttemp.select<64,1>(64));
        _ttemp.select<32,1>(0)=__ESIMD_NS::max<float,32,float>(_ttemp.select<32,1>(0),_ttemp.select<32,1>(32));
        _ttemp.select<16,1>(0)=__ESIMD_NS::max<float,16,float>(_ttemp.select<16,1>(0),_ttemp.select<16,1>(16));
        _fp32CurrentMax.merge(_ttemp.select<16,1>(0), _ttemp.select<16,1>(0) > _fp32CurrentMax);

        _fp32Exp2Temp.select<16,1>(0) = _fp32CurrentMax.select<16,1>(0) * attnScoreMul;

        #pragma unroll
        for (int _k = 0; _k < 8; _k++) {
          #pragma unroll
          for (int _kk = 0; _kk < 2; _kk++) {
            _ttemp.select<16,1>(16*_kk)    = tempOutput.select<16,1>(128*_k+32*_kk)    * attnScoreMul - _fp32Exp2Temp.select<16,1>(0);
            _ttemp.select<16,1>(16*_kk+32) = tempOutput.select<16,1>(128*_k+32*_kk+16) * attnScoreMul - _fp32Exp2Temp.select<16,1>(0);
          }
          #pragma unroll
          for (int _kk = 0; _kk < 2; _kk++) {
            _ttemp.select<16,1>(16*_kk+64)    = tempOutput.select<16,1>(128*_k+64+32*_kk)    * attnScoreMul - _fp32Exp2Temp.select<16,1>(0);
            _ttemp.select<16,1>(16*_kk+64+32) = tempOutput.select<16,1>(128*_k+64+32*_kk+16) * attnScoreMul - _fp32Exp2Temp.select<16,1>(0);
          }
          #pragma unroll
          for (int _kk = 0; _kk < 8; _kk++)
            tempOutput.select<16,1>(128*_k+16*_kk) = __ESIMD_NS::exp2<float,16,float>(_ttemp.select<16,1>(16*_kk));
        }
        _fp32Compensation = fp32HistoricMaxTemp * attnScoreMul - _fp32Exp2Temp.select<16,1>(0);
        _fp32Compensation = __ESIMD_NS::exp2<float,16,float>(_fp32Compensation);
        fp32SoftMaxTemp.select<16,1>(0) *= _fp32Compensation.select<16,1>(0);
        #pragma unroll
        for (int _kk = 0; _kk < 4; _kk++)
          _ttemp.select<32,1>(32*_kk) = tempOutput.select<32,1>(64*_kk)+tempOutput.select<32,1>(64*_kk+32);
        #pragma unroll
        for (int _kkk = 0; _kkk < 6; ++_kkk)
          #pragma unroll
          for (int _kk = 0; _kk < 4; _kk++)
            _ttemp.select<32,1>(32*_kk) = _ttemp.select<32,1>(32*_kk)+tempOutput.select<32,1>((4*_kkk+_kk)*32+16*16);
        _ttemp.select<64,1>(0)=_ttemp.select<64,1>(0)+_ttemp.select<64,1>(64);
        _ttemp.select<32,1>(0)=_ttemp.select<32,1>(0)+_ttemp.select<32,1>(32);
        _ttemp.select<16,1>(0)=_ttemp.select<16,1>(0)+_ttemp.select<16,1>(16);
        fp32SoftMaxTemp.select<16,1>(0) += _ttemp.select<16,1>(0);
        fp32HistoricMaxTemp = _fp32CurrentMax;

        simd<fp16, 32> _compTemp;
        _compTemp.select<16,1>(0)  = _fp32Compensation;
        _compTemp.select<16,1>(16) = _fp32Compensation;
        #pragma unroll
        for (int _kk = 0; _kk < 64; _kk++)
          finalOutput.select<32,1>(32*_kk) = finalOutput.select<32,1>(32*_kk) * _compTemp.select<32,1>(0);

        #pragma unroll
        for (int _k = 0; _k < 4; _k++) {
          #pragma unroll
          for (int _kk = 0; _kk < 2; _kk++)
            tempBufferAsFp16.select<32,2>(128*_k+64*_kk)   = tempOutput.select<32,1>(128*_k+64*_kk);
          #pragma unroll
          for (int _kk = 0; _kk < 2; _kk++)
            tempBufferAsFp16.select<32,2>(128*_k+64*_kk+1) = tempOutput.select<32,1>(128*_k+64*_kk+32);
        }
        #pragma unroll
        for (int _k = 0; _k < 4; _k++) {
          #pragma unroll
          for (int _kk = 0; _kk < 2; _kk++)
            _tempQkAsFp16.select<32,2>(128*_k+64*_kk)   = tempOutput.select<32,1>(128*_k+512+64*_kk);
          #pragma unroll
          for (int _kk = 0; _kk < 2; _kk++)
            _tempQkAsFp16.select<32,2>(128*_k+64*_kk+1) = tempOutput.select<32,1>(128*_k+512+64*_kk+32);
        }
      }

      // SLM scatter V
      {
        simd<uint32_t, 32> _simdSlmOffs;
        _simdSlmOffs.select<16,1>(0)  = baseOffsetInc16AsVector;
        _simdSlmOffs.select<16,1>(16) = baseOffsetInc16AsVector + 16;
        _simdSlmOffs.select<32,1>(0)  = _simdSlmOffs.select<32,1>(0) * 16u * (uint32_t)sizeof(fp16)
                                       + slmOffsetV + _slmSlot;
        #pragma unroll
        for (int _kk = 0; _kk < 2; _kk++)
          __ESIMD_ENS::lsc_slm_scatter<uint32_t, 8, __ESIMD_ENS::lsc_data_size::u32, 16>(
            _simdSlmOffs.select<16,1>(16*_kk),
            _fp16VState.template bit_cast_view<uint32_t>().select<128,1>(128*_kk));
      }
      barrier();

      // Attn @ V
      {
        #pragma unroll
        for (int _nn = 0; _nn < 2; _nn++) {
          #pragma unroll
          for (int _l = 0; _l < 2; _l++) {
            #pragma unroll
            for (int _ll = 0; _ll < 2; _ll++)
              _tempQkAsFp16.select<512,1>(1024+512*_ll) =
                slm_block_load<fp16, 512>(slmOffsetBaseV + _slmSlot +
                  16*128*_nn*(uint32_t)sizeof(fp16) + 16*64*_l*(uint32_t)sizeof(fp16) +
                  512*_ll*(uint32_t)sizeof(fp16));
            #pragma unroll
            for (int _ll = 0; _ll < 8; _ll++) {
              auto _ccTile = finalOutput.select<128,1>(1024*_l+128*_ll);
              auto _aaTile = tempBufferAsFp16.select<256,1>(256*_nn);
              auto _bbTile = _tempQkAsFp16.select<128,1>(1024+128*_ll);
              _ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
                simd<fp16,128>(_ccTile.data()),
                simd<fp16,256>(_aaTile.data()),
                simd<fp16,128>(_bbTile.data()));
            }
          }
        }
        #pragma unroll
        for (int _nn = 0; _nn < 2; _nn++) {
          #pragma unroll
          for (int _l = 0; _l < 2; _l++) {
            #pragma unroll
            for (int _ll = 0; _ll < 2; _ll++)
              _tempQkAsFp16.select<512,1>(1024+512*_ll) =
                slm_block_load<fp16, 512>(slmOffsetBaseV + _slmSlot +
                  16*128*2u*(uint32_t)sizeof(fp16) + 16*128*_nn*(uint32_t)sizeof(fp16) +
                  16*64*_l*(uint32_t)sizeof(fp16) + 512*_ll*(uint32_t)sizeof(fp16));
            #pragma unroll
            for (int _ll = 0; _ll < 8; _ll++) {
              auto _ccTile = finalOutput.select<128,1>(1024*_l+128*_ll);
              auto _aaTile = _tempQkAsFp16.select<256,1>(256*_nn);
              auto _bbTile = _tempQkAsFp16.select<128,1>(1024+128*_ll);
              _ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
                simd<fp16,128>(_ccTile.data()),
                simd<fp16,256>(_aaTile.data()),
                simd<fp16,128>(_bbTile.data()));
            }
          }
        }
      }
      loopIdx++;
    } // partial inner
  }

  // =========================================================
  // LAST KV BLOCK  — kvSeqLen boundary check
  // =========================================================
  {
    uint32_t kv_idx   = *(mask_ptr + total_kv_blocks - 1u);
    uint32_t _slmSlot = (loopIdx & 0x1u) * 64u * 128u * (uint32_t)sizeof(fp16);
    auto _tempQkAsFp16 = tempOutput.template bit_cast_view<fp16>();
    simd<fp16, 512> _fp16VState;
    tempOutput = 0;

    uint32_t Y_base = get_Y_base(kv_idx);
    int32_t  _kv_block_start = (int32_t)(kv_idx * KV_CHUNK);

    // Q @ K^T
    #pragma unroll
    for (int32_t nn = 0; nn < 8; nn++) {
      payloadK.set_x(kCoordX + 16 * nn);
      #pragma unroll
      for (int32_t l = 0; l < 4; l++) {
        payloadK.set_y(Y_base + 16u * l);
        tempBufferAsFp16.select<256, 1>(256 * l) =
          __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 1, false, false,
          __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadK);
      }
      #pragma unroll
      for (int32_t kk = 0; kk < 8; kk++) {
        auto ccTile = tempOutput.select<128, 1>(128 * kk);
        auto aaTile = fp16QState.select<256, 1>(256 * nn);
        auto bbTile = tempBufferAsFp16.select<128, 1>(128 * kk);
        if constexpr (IS_BF16) {
          simd<bf16, 256> aa_bf16 = fp16QState.template bit_cast_view<bf16>().template select<256,1>(256*nn);
          simd<bf16, 128> bb_bf16 = tempBuffer.template bit_cast_view<bf16>().template select<128,1>(128*kk);
          ccTile = dpas<8, 8, float, float, bf16, bf16>(
            simd<float, 128>(ccTile.data()), aa_bf16, bb_bf16);
        } else {
          ccTile = dpas<8, 8, float, float, fp16, fp16>(
            simd<float, 128>(ccTile.data()),
            simd<fp16, 256>(aaTile.data()),
            simd<fp16, 128>(bbTile.data()));
        }
      }
    }

    // kvSeqLen boundary mask (same pattern as reference)
    {
      auto softmaxPositions = ui32Temp.select<16, 1>(48);
      softmaxPositions.select<16, 1>(0) = baseOffsetInc16AsVector + (uint32_t)_kv_block_start;
      #pragma unroll
      for (int k = 0; k < 4; k++) {
        #pragma unroll
        for (int kk = 0; kk < 16; kk++)
          tempOutput.select<16, 1>(256 * k + 16 * kk).merge(FP32_MIN,
            softmaxPositions.select<16, 0>(kk) >= kvSeqLen);
        softmaxPositions.select<16, 1>(0) = softmaxPositions.select<16, 1>(0) + 16;
      }
    }

    // Causal mask
    if constexpr (IS_CAUSAL) {
      #pragma unroll
      for (int kk = 0; kk < 8; kk++) {
        #pragma unroll
        for (int m = 0; m < 8; m++) {
          int32_t kv_pos = _kv_block_start + kk * 8 + m;
          simd<int, 16> v_kv_pos(kv_pos);
          auto cmask = v_kv_pos > causal_boundaries;
          tempOutput.select<16, 1>(kk * 128 + m * 16).merge(FP32_MIN, cmask);
        }
      }
    }

    // Load V from paged cache
    payloadV.set_y(Y_base + (uint32_t)vvv * 16u);
    _fp16VState = __ESIMD_ENS::lsc_load_2d<fp16, 16, 16, 2, false, true,
      __ESIMD_ENS::cache_hint::cached, __ESIMD_ENS::cache_hint::cached>(payloadV);

    // bf16 → fp16 conversion for V (DPAS Attn@V requires fp16)
    if constexpr (IS_BF16) {
      #pragma unroll
      for (int _ci = 0; _ci < 32; _ci++) {
        simd<float, 16> _cv = _fp16VState.template bit_cast_view<bf16>().template select<16,1>(16*_ci);
        _fp16VState.template select<16,1>(16*_ci) = _cv;
      }
    }

    // Online softmax (identical to main loop)
    {
      auto fp32CurrentMaxTemp      = tempBuffer.select<16, 1>(0);
      auto fp32SoftMaxCompensation = tempBuffer.select<16, 1>(16);
      auto fp32Exp2Temp            = tempBuffer.select<16, 1>(32);
      simd<float, 8*16> ttemp;
      fp32CurrentMaxTemp = fp32HistoricMaxTemp;

      #pragma unroll
      for (int kk = 0; kk < 4; kk++)
        ttemp.select<32,1>(32*kk) = __ESIMD_NS::max<float,32,float>(
          tempOutput.select<32,1>(64*kk), tempOutput.select<32,1>(64*kk+32));
      #pragma unroll
      for (int kkk = 0; kkk < 6; ++kkk)
        #pragma unroll
        for (int kk = 0; kk < 4; kk++)
          ttemp.select<32,1>(32*kk) = __ESIMD_NS::max<float,32,float>(
            ttemp.select<32,1>(32*kk),
            tempOutput.select<32,1>((4*kkk+kk)*32+16*16));
      ttemp.select<64,1>(0)=__ESIMD_NS::max<float,64,float>(ttemp.select<64,1>(0),ttemp.select<64,1>(64));
      ttemp.select<32,1>(0)=__ESIMD_NS::max<float,32,float>(ttemp.select<32,1>(0),ttemp.select<32,1>(32));
      ttemp.select<16,1>(0)=__ESIMD_NS::max<float,16,float>(ttemp.select<16,1>(0),ttemp.select<16,1>(16));
      fp32CurrentMaxTemp.merge(ttemp.select<16,1>(0), ttemp.select<16,1>(0) > fp32CurrentMaxTemp);

      fp32Exp2Temp.select<16,1>(0) = fp32CurrentMaxTemp.select<16,1>(0) * attnScoreMul;

      #pragma unroll
      for (int k = 0; k < 8; k++) {
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          ttemp.select<16,1>(16*kk)    = tempOutput.select<16,1>(128*k+32*kk)    * attnScoreMul - fp32Exp2Temp.select<16,1>(0);
          ttemp.select<16,1>(16*kk+32) = tempOutput.select<16,1>(128*k+32*kk+16) * attnScoreMul - fp32Exp2Temp.select<16,1>(0);
        }
        #pragma unroll
        for (int kk = 0; kk < 2; kk++) {
          ttemp.select<16,1>(16*kk+64)    = tempOutput.select<16,1>(128*k+64+32*kk)    * attnScoreMul - fp32Exp2Temp.select<16,1>(0);
          ttemp.select<16,1>(16*kk+64+32) = tempOutput.select<16,1>(128*k+64+32*kk+16) * attnScoreMul - fp32Exp2Temp.select<16,1>(0);
        }
        #pragma unroll
        for (int kk = 0; kk < 8; kk++)
          tempOutput.select<16,1>(128*k+16*kk) = __ESIMD_NS::exp2<float,16,float>(ttemp.select<16,1>(16*kk));
      }
      fp32SoftMaxCompensation = fp32HistoricMaxTemp * attnScoreMul - fp32Exp2Temp.select<16,1>(0);
      fp32SoftMaxCompensation = __ESIMD_NS::exp2<float,16,float>(fp32SoftMaxCompensation);
      fp32SoftMaxTemp.select<16,1>(0) *= fp32SoftMaxCompensation.select<16,1>(0);

      #pragma unroll
      for (int kk = 0; kk < 4; kk++)
        ttemp.select<32,1>(32*kk) = tempOutput.select<32,1>(64*kk)+tempOutput.select<32,1>(64*kk+32);
      #pragma unroll
      for (int kkk = 0; kkk < 6; ++kkk)
        #pragma unroll
        for (int kk = 0; kk < 4; kk++)
          ttemp.select<32,1>(32*kk) = ttemp.select<32,1>(32*kk)+tempOutput.select<32,1>((4*kkk+kk)*32+16*16);
      ttemp.select<64,1>(0)=ttemp.select<64,1>(0)+ttemp.select<64,1>(64);
      ttemp.select<32,1>(0)=ttemp.select<32,1>(0)+ttemp.select<32,1>(32);
      ttemp.select<16,1>(0)=ttemp.select<16,1>(0)+ttemp.select<16,1>(16);
      fp32SoftMaxTemp.select<16,1>(0) += ttemp.select<16,1>(0);
      fp32HistoricMaxTemp = fp32CurrentMaxTemp;

      simd<fp16, 32> compensationTemp;
      compensationTemp.select<16,1>(0)  = fp32SoftMaxCompensation;
      compensationTemp.select<16,1>(16) = fp32SoftMaxCompensation;
      #pragma unroll
      for (int kk = 0; kk < 64; kk++)
        finalOutput.select<32,1>(32*kk) = finalOutput.select<32,1>(32*kk) * compensationTemp.select<32,1>(0);

      #pragma unroll
      for (int k = 0; k < 4; k++) {
        #pragma unroll
        for (int kk = 0; kk < 2; kk++)
          tempBufferAsFp16.select<32,2>(128*k+64*kk)   = tempOutput.select<32,1>(128*k+64*kk);
        #pragma unroll
        for (int kk = 0; kk < 2; kk++)
          tempBufferAsFp16.select<32,2>(128*k+64*kk+1) = tempOutput.select<32,1>(128*k+64*kk+32);
      }
      #pragma unroll
      for (int k = 0; k < 4; k++) {
        #pragma unroll
        for (int kk = 0; kk < 2; kk++)
          _tempQkAsFp16.select<32,2>(128*k+64*kk)   = tempOutput.select<32,1>(128*k+512+64*kk);
        #pragma unroll
        for (int kk = 0; kk < 2; kk++)
          _tempQkAsFp16.select<32,2>(128*k+64*kk+1) = tempOutput.select<32,1>(128*k+512+64*kk+32);
      }
    }

    // SLM scatter V
    {
      simd<uint32_t, 32> simdSlmOffs;
      simdSlmOffs.select<16,1>(0)  = baseOffsetInc16AsVector;
      simdSlmOffs.select<16,1>(16) = baseOffsetInc16AsVector + 16;
      simdSlmOffs.select<32,1>(0)  = simdSlmOffs.select<32,1>(0) * 16u * (uint32_t)sizeof(fp16)
                                    + slmOffsetV + _slmSlot;
      #pragma unroll
      for (int kk = 0; kk < 2; kk++)
        __ESIMD_ENS::lsc_slm_scatter<uint32_t, 8, __ESIMD_ENS::lsc_data_size::u32, 16>(
          simdSlmOffs.select<16,1>(16*kk),
          _fp16VState.template bit_cast_view<uint32_t>().select<128,1>(128*kk));
    }
    barrier();

    // Attn @ V
    {
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int l = 0; l < 2; l++) {
          #pragma unroll
          for (int ll = 0; ll < 2; ll++)
            _tempQkAsFp16.select<512,1>(1024+512*ll) =
              slm_block_load<fp16, 512>(slmOffsetBaseV + _slmSlot +
                16*128*nn*(uint32_t)sizeof(fp16) + 16*64*l*(uint32_t)sizeof(fp16) +
                512*ll*(uint32_t)sizeof(fp16));
          #pragma unroll
          for (int ll = 0; ll < 8; ll++) {
            auto ccTile = finalOutput.select<128,1>(1024*l+128*ll);
            auto aaTile = tempBufferAsFp16.select<256,1>(256*nn);
            auto bbTile = _tempQkAsFp16.select<128,1>(1024+128*ll);
            ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
              simd<fp16,128>(ccTile.data()),
              simd<fp16,256>(aaTile.data()),
              simd<fp16,128>(bbTile.data()));
          }
        }
      }
      #pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        #pragma unroll
        for (int l = 0; l < 2; l++) {
          #pragma unroll
          for (int ll = 0; ll < 2; ll++)
            _tempQkAsFp16.select<512,1>(1024+512*ll) =
              slm_block_load<fp16, 512>(slmOffsetBaseV + _slmSlot +
                16*128*2u*(uint32_t)sizeof(fp16) + 16*128*nn*(uint32_t)sizeof(fp16) +
                16*64*l*(uint32_t)sizeof(fp16) + 512*ll*(uint32_t)sizeof(fp16));
          #pragma unroll
          for (int ll = 0; ll < 8; ll++) {
            auto ccTile = finalOutput.select<128,1>(1024*l+128*ll);
            auto aaTile = _tempQkAsFp16.select<256,1>(256*nn);
            auto bbTile = _tempQkAsFp16.select<128,1>(1024+128*ll);
            ccTile = dpas<8, 8, fp16, fp16, fp16, fp16>(
              simd<fp16,128>(ccTile.data()),
              simd<fp16,256>(aaTile.data()),
              simd<fp16,128>(bbTile.data()));
          }
        }
      }
    }
    loopIdx++;
  } // last block

  // ---- Output normalization — per-lane (per-head) divisor ----
  // softMaxDivisor[lane] = 1/sum for head lane (16 heads = 16 SIMD lanes)
  // Each 32-element chunk in finalOutput has [16 heads @ HD_even, 16 heads @ HD_odd]
  // (Usage 2 transposed DPAS layout), so we need per-lane division, not scalar.
  simd<float, 16> softMaxDivisor;
  softMaxDivisor.select<16,1>(0) = fp32SoftMaxTemp;
  auto nonzero_mask = (softMaxDivisor != 0.0f);
  softMaxDivisor.merge(simd<float,16>(1.0f), !nonzero_mask);
  softMaxDivisor = 1.0f / softMaxDivisor;
  softMaxDivisor.merge(simd<float,16>(0.0f), !nonzero_mask);

  simd<float, 32> divMul;
  divMul.select<16,1>(0) = softMaxDivisor;
  divMul.select<16,1>(16) = softMaxDivisor;

  #pragma unroll
  for (int kk = 0; kk < 64; kk++) {
    simd<fp16, 32> vals_fp16 = finalOutput.select<32,1>(32 * kk);
    simd<float,32> vals_f32  = vals_fp16;
    vals_f32 = vals_f32 * divMul;
    if constexpr (IS_BF16) {
      simd<bf16, 32> result = vals_f32;
      fp16QState.template bit_cast_view<bf16>().template select<16, 2>(32 * kk)     = result.template select<16,1>(0);
      fp16QState.template bit_cast_view<bf16>().template select<16, 2>(32 * kk + 1) = result.template select<16,1>(16);
    } else {
      simd<fp16, 32> result = vals_f32;
      fp16QState.select<16, 2>(32 * kk)     = result.select<16,1>(0);
      fp16QState.select<16, 2>(32 * kk + 1) = result.select<16,1>(16);
    }
  }

  // Output scatter
  simd<uint32_t, 16> simdOffsets;
  simd_mask<16> wmask;
  bool valid_q = (this_q_pos < (int32_t)activationLength);
  #pragma unroll
  for (int i = 0; i < 16; i++) {
    simdOffsets[i] = ((uint32_t)this_q_pos * headQ + (uint32_t)headIdx + (uint32_t)i)
                   * HD * (uint32_t)sizeof(fp16);
    wmask[i] = valid_q;
  }
  #pragma unroll
  for (int kk = 0; kk < 16; kk++) {
    __ESIMD_ENS::lsc_scatter<uint32_t, 4, __ESIMD_ENS::lsc_data_size::u32,
      __ESIMD_ENS::cache_hint::write_back, __ESIMD_ENS::cache_hint::write_back, 16, uint32_t>(
      (uint32_t*)out, simdOffsets,
      fp16QState.template bit_cast_view<uint32_t>().select<64,1>(64*kk), wmask);
    simdOffsets += 4u * (uint32_t)sizeof(uint32_t);
  }
}

template<bool IS_CAUSAL, bool IS_BF16 = false>
struct SparseFlashAttnPagedFunctor {
  const unsigned short* Q;
  const unsigned short* kv_cache;
  uint8_t*  O;
  uint32_t* sparseMask;
  uint32_t* sparseMaskCnt;
  const int* block_table;
  int activationLength;
  int kvSeqLen;
  int history_len;
  int headQ;
  int headKV;
  int block_size;
  int64_t kv_stride_split;
  int64_t kv_stride_block;
  float attn_scale;

  void operator()(sycl::nd_item<2> ndi) const SYCL_ESIMD_KERNEL {
    flashAttnSparseMha128GQA_paged<IS_CAUSAL, IS_BF16>(
      Q, kv_cache, O,
      sparseMask, sparseMaskCnt, block_table,
      activationLength, kvSeqLen, history_len, headQ, headKV,
      block_size, kv_stride_split, kv_stride_block, attn_scale, ndi);
  }
};
