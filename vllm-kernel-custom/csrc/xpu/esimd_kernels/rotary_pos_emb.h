#include "utils.h"

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using fp16 = sycl::half;
using namespace sycl;

/**
 *
 *
 *
 * def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)

 *
        dtype = query.dtype
        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim :]
            key_pass = key[..., self.rotary_dim :]

        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(positions.device)
        cos_sin = self.cos_sin_cache[
            torch.add(positions, offsets) if offsets is not None else positions
        ]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query.to(dtype), key.to(dtype)

*/
ESIMD_INLINE void rotary_pos_emb_ds(
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

  simd<fp16, 64> input;
  simd<fp16, 64> input_rotate_half;
  simd<fp16, 64> output;
  simd<fp16, 64> cos_value;
  simd<fp16, 64> sin_value;
  uint64_t positions_index = ((uint64_t*)positions)[v];
  if (has_offset) {
    positions_index += ((uint64_t*)offsets)[v];
  }
  unsigned int offsetCosSin = positions_index * hidden_dim_q;

  simd<fp16, 64> cos_sin_value;
  cos_sin_value.template bit_cast_view<fp16>().template select<64, 1>(0) = __ESIMD_ENS::lsc_block_load<
      fp16,
      64,
      __ESIMD_ENS::lsc_data_size::u16,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>((fp16*)cos_sin_cache + offsetCosSin);

  cos_value.select<32, 2>(0) = cos_sin_value.select<32, 1>(0);
  cos_value.select<32, 2>(1) = cos_sin_value.select<32, 1>(0);
  sin_value.select<32, 2>(0) = cos_sin_value.select<32, 1>(32);
  sin_value.select<32, 2>(1) = cos_sin_value.select<32, 1>(32);

  if (h < num_heads_q)  // q
  {
    unsigned int InOffset = h * hd_stride_q + v * hd_stride_q * num_heads_q;
    input.template bit_cast_view<fp16>().template select<64, 1>(0) = __ESIMD_ENS::lsc_block_load<
        fp16,
        64,
        __ESIMD_ENS::lsc_data_size::u16,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::uncached>((fp16*)qState + InOffset);

    input_rotate_half.select<32, 2>(1) = input.select<32, 2>(0);
    input_rotate_half.select<32, 2>(0) = input.select<32, 2>(1) * -1.0;

    output = input * cos_value + input_rotate_half * sin_value;

    __ESIMD_ENS::lsc_block_store<
        fp16,
        64,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>((fp16*)qState + InOffset, output.select<64, 1>(0));
  } else if (h < num_heads_q + num_heads_kv)  // k
  {
    unsigned int InOffset = (h - num_heads_q) * hd_stride_kv + v * head_hd_stride_kv;
    input.template bit_cast_view<fp16>().template select<64, 1>(0) = __ESIMD_ENS::lsc_block_load<
        fp16,
        64,
        __ESIMD_ENS::lsc_data_size::u16,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::uncached>((fp16*)kState + InOffset);

    input_rotate_half.select<32, 2>(1) = input.select<32, 2>(0);
    input_rotate_half.select<32, 2>(0) = input.select<32, 2>(1) * -1.0;

    output = input * cos_value + input_rotate_half * sin_value;

    __ESIMD_ENS::lsc_block_store<
        fp16,
        64,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>((fp16*)kState + InOffset, output.select<64, 1>(0));
  }
}
