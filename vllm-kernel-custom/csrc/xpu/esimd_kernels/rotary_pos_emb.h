#include "utils.h"

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;
using namespace sycl;

/**
 * Rotary Position Embedding — GPT-J style (interleaved). HD=64 only.
 *
 * cos = repeat_interleave(2):  [c0,c0,c1,c1,...,c31,c31]
 * rotate = (-x[1::2], x[::2]) interleaved
 *
 * Templated on DT (fp16 or bf16).
 */
template<typename DT>
ESIMD_INLINE void rotary_pos_emb_gptj(
    uint8_t* qState,
    uint8_t* kState,
    uint8_t* cos_sin_cache,
    uint8_t* positions,
    uint8_t* offsets,
    int num_heads_q,
    int hidden_dim_q,
    int hd_stride_q,
    int num_heads_kv,
    int hidden_dim_kv,
    int hd_stride_kv,
    int head_hd_stride_kv,
    int has_offset,
    nd_item<2>& ndi) {
  int h = ndi.get_group(1);  // [0, q_heads+k_heads)
  int v = ndi.get_group(0);  // [0, input_len)

  simd<DT, 64> input;
  simd<DT, 64> input_rotate_half;
  simd<DT, 64> output;
  simd<DT, 64> cos_value;
  simd<DT, 64> sin_value;
  uint64_t positions_index = ((uint64_t*)positions)[v];
  if (has_offset) {
    positions_index += ((uint64_t*)offsets)[v];
  }
  unsigned int offsetCosSin = positions_index * hidden_dim_q;

  simd<DT, 64> cos_sin_value;
  cos_sin_value.template bit_cast_view<DT>().template select<64, 1>(0) = __ESIMD_ENS::lsc_block_load<
      DT,
      64,
      __ESIMD_ENS::lsc_data_size::u16,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>((DT*)cos_sin_cache + offsetCosSin);

  // GPT-J: repeat_interleave(2) — [c0,c0,c1,c1,...,c31,c31]
  cos_value.template select<32, 2>(0) = cos_sin_value.template select<32, 1>(0);
  cos_value.template select<32, 2>(1) = cos_sin_value.template select<32, 1>(0);
  sin_value.template select<32, 2>(0) = cos_sin_value.template select<32, 1>(32);
  sin_value.template select<32, 2>(1) = cos_sin_value.template select<32, 1>(32);

  if (h < num_heads_q)  // q
  {
    unsigned int InOffset = h * hd_stride_q + v * hd_stride_q * num_heads_q;
    input.template bit_cast_view<DT>().template select<64, 1>(0) = __ESIMD_ENS::lsc_block_load<
        DT,
        64,
        __ESIMD_ENS::lsc_data_size::u16,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::uncached>((DT*)qState + InOffset);

    // GPT-J rotate: (-x[odd], x[even]) interleaved
    input_rotate_half.template select<32, 2>(1) = input.template select<32, 2>(0);
    input_rotate_half.template select<32, 2>(0) = input.template select<32, 2>(1) * DT(-1.0);

    output = input * cos_value + input_rotate_half * sin_value;

    __ESIMD_ENS::lsc_block_store<
        DT,
        64,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>((DT*)qState + InOffset, output.template select<64, 1>(0));
  } else if (h < num_heads_q + num_heads_kv)  // k
  {
    unsigned int InOffset = (h - num_heads_q) * hd_stride_kv + v * head_hd_stride_kv;
    input.template bit_cast_view<DT>().template select<64, 1>(0) = __ESIMD_ENS::lsc_block_load<
        DT,
        64,
        __ESIMD_ENS::lsc_data_size::u16,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::uncached>((DT*)kState + InOffset);

    input_rotate_half.template select<32, 2>(1) = input.template select<32, 2>(0);
    input_rotate_half.template select<32, 2>(0) = input.template select<32, 2>(1) * DT(-1.0);

    output = input * cos_value + input_rotate_half * sin_value;

    __ESIMD_ENS::lsc_block_store<
        DT,
        64,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>((DT*)kState + InOffset, output.template select<64, 1>(0));
  }
}

/**
 * Rotary Position Embedding — NeoX style (half-split).
 *
 * cos_sin_cache: [max_pos, rotary_dim] where rotary_dim = HD
 *   Layout per position: [cos_0..cos_{HD/2-1}, sin_0..sin_{HD/2-1}]
 *
 * NeoX rotation:
 *   cos_expanded = [cos, cos]  (repeat(1,2))
 *   rotate_neox(x) = [-x[HD/2:HD], x[0:HD/2]]
 *   out = x * cos_expanded + rotate_neox(x) * sin_expanded
 *
 * Which simplifies to:
 *   out[0:HD/2]    = x[0:HD/2]    * cos - x[HD/2:HD] * sin
 *   out[HD/2:HD]   = x[HD/2:HD]   * cos + x[0:HD/2]  * sin
 *
 * Templated on DT (fp16 or bf16) and HD (64 or 128).
 */
template<typename DT, int HD>
ESIMD_INLINE void rotary_pos_emb_neox(
    uint8_t* qState,
    uint8_t* kState,
    uint8_t* cos_sin_cache,
    uint8_t* positions,
    uint8_t* offsets,
    int num_heads_q,
    int hidden_dim_q,
    int hd_stride_q,
    int num_heads_kv,
    int hidden_dim_kv,
    int hd_stride_kv,
    int head_hd_stride_kv,
    int has_offset,
    nd_item<2>& ndi) {
  int h = ndi.get_group(1);  // [0, q_heads+k_heads)
  int v = ndi.get_group(0);  // [0, input_len)

  constexpr int HALF = HD / 2;

  uint64_t positions_index = ((uint64_t*)positions)[v];
  if (has_offset) {
    positions_index += ((uint64_t*)offsets)[v];
  }
  // cos_sin_cache layout: [max_pos, HD] where first HD/2 = cos, next HD/2 = sin
  unsigned int offsetCosSin = positions_index * hidden_dim_q;

  // Load cos and sin (HD/2 each) — total HD values
  simd<DT, HALF> cos_val;
  simd<DT, HALF> sin_val;

  if constexpr (HALF <= 64) {
    // Single load for HD <= 128
    simd<DT, HD> cos_sin_value;
    cos_sin_value.template bit_cast_view<DT>().template select<HD, 1>(0) = __ESIMD_ENS::lsc_block_load<
        DT,
        HD,
        __ESIMD_ENS::lsc_data_size::u16,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((DT*)cos_sin_cache + offsetCosSin);
    cos_val = cos_sin_value.template select<HALF, 1>(0);
    sin_val = cos_sin_value.template select<HALF, 1>(HALF);
  } else {
    // Two loads for HD > 128
    cos_val.template bit_cast_view<DT>().template select<HALF, 1>(0) = __ESIMD_ENS::lsc_block_load<
        DT,
        HALF,
        __ESIMD_ENS::lsc_data_size::u16,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((DT*)cos_sin_cache + offsetCosSin);
    sin_val.template bit_cast_view<DT>().template select<HALF, 1>(0) = __ESIMD_ENS::lsc_block_load<
        DT,
        HALF,
        __ESIMD_ENS::lsc_data_size::u16,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((DT*)cos_sin_cache + offsetCosSin + HALF);
  }

  DT* ptr;
  unsigned int InOffset;
  if (h < num_heads_q) {
    ptr = (DT*)qState;
    InOffset = h * hd_stride_q + v * hd_stride_q * num_heads_q;
  } else if (h < num_heads_q + num_heads_kv) {
    ptr = (DT*)kState;
    InOffset = (h - num_heads_q) * hd_stride_kv + v * head_hd_stride_kv;
  } else {
    return;
  }

  // Load HD-element head (two loads if HD > 64)
  simd<DT, HALF> first;
  simd<DT, HALF> second;

  if constexpr (HD <= 64) {
    simd<DT, HD> input;
    input.template bit_cast_view<DT>().template select<HD, 1>(0) = __ESIMD_ENS::lsc_block_load<
        DT,
        HD,
        __ESIMD_ENS::lsc_data_size::u16,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::uncached>(ptr + InOffset);
    first = input.template select<HALF, 1>(0);
    second = input.template select<HALF, 1>(HALF);
  } else {
    // HD=128: two 64-element loads
    first.template bit_cast_view<DT>().template select<HALF, 1>(0) = __ESIMD_ENS::lsc_block_load<
        DT,
        HALF,
        __ESIMD_ENS::lsc_data_size::u16,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::uncached>(ptr + InOffset);
    second.template bit_cast_view<DT>().template select<HALF, 1>(0) = __ESIMD_ENS::lsc_block_load<
        DT,
        HALF,
        __ESIMD_ENS::lsc_data_size::u16,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::uncached>(ptr + InOffset + HALF);
  }

  // NeoX: first_half * cos - second_half * sin, second_half * cos + first_half * sin
  simd<DT, HALF> out_first  = first * cos_val - second * sin_val;
  simd<DT, HALF> out_second = second * cos_val + first * sin_val;

  if constexpr (HD <= 64) {
    simd<DT, HD> output;
    output.template select<HALF, 1>(0) = out_first;
    output.template select<HALF, 1>(HALF) = out_second;
    __ESIMD_ENS::lsc_block_store<
        DT,
        HD,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>(ptr + InOffset, output.template select<HD, 1>(0));
  } else {
    // HD=128: two 64-element stores
    __ESIMD_ENS::lsc_block_store<
        DT,
        HALF,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>(ptr + InOffset, out_first.template select<HALF, 1>(0));
    __ESIMD_ENS::lsc_block_store<
        DT,
        HALF,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>(ptr + InOffset + HALF, out_second.template select<HALF, 1>(0));
  }
}
