#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

// Convert mask_orig [num_kv_heads, qlen, 64] -> mask_out [num_kv_heads, qlen/16, 1024] + mask_cnt_out
//
// Layout:
//   mask_orig: each q-position has exactly 64 kv_block indices, sorted ascending.
//              kv_block index range = [0, total_kv_blocks-1], total_kv_blocks = kv_len/64.
//              kv_len can be arbitrarily large (supported up to MAX_KV_BLOCKS * 64).
//
//   mask_out:  each q-block (16 consecutive q-positions) stores the UNION of their
//              64 indices each = up to 16*64 = 1024 unique kv_block indices, sorted.
//              Union size <= 1024 regardless of kv_len, because only 64 are selected per q_pos.
//
// Why a bitmap sized by total_kv_blocks (not 1024):
//   The bitmap is indexed by kv_block INDEX VALUE, not by union count.
//   E.g. kv_len=131072 -> total_kv_blocks=2048 -> index values up to 2047
//   -> need a 2048-bit bitmap (64 words), not 1024-bit (32 words).
//   Union output still fits in 1024 slots since at most 16*64 unique entries.
//
// Fix: store bitmap in SLM (on-chip, sized at dispatch time up to MAX_KV_BLOCKS).
//
// Dispatch: nd_range<1>({num_kv_heads * q_blocks}, {1})

// Max kv_blocks supported: 32768 (kv_len up to 2M tokens)
// SLM bitmap = 32768/32 * 4 = 4096 bytes per workgroup
static constexpr uint32_t MAX_KV_BLOCKS    = 32768u;
static constexpr uint32_t SLM_BITMAP_BYTES = MAX_KV_BLOCKS / 8u;  // 4096 bytes

struct MaskConvertFunctor {
    uint32_t* mask_orig;      // [num_kv_heads, qlen, 64]
    uint32_t* mask_out;       // [num_kv_heads, q_blocks, 1024]
    uint32_t* mask_cnt_out;   // [num_kv_heads, q_blocks]
    uint32_t  qlen;
    uint32_t  num_kv_heads;
    uint32_t  total_kv_blocks;  // = kv_len / 64

    void operator()(sycl::nd_item<1> ndi) const SYCL_ESIMD_KERNEL {
        // Reserve SLM for bitmap (fixed allocation, MAX_KV_BLOCKS bits = 4096 bytes)
        slm_init<SLM_BITMAP_BYTES>();

        uint32_t gid     = (uint32_t)ndi.get_global_id(0);
        uint32_t q_blks  = (qlen + 15u) / 16u;
        uint32_t kv_head = gid / q_blks;
        uint32_t q_block = gid % q_blks;

        // How many bitmap words we actually need for this kv_len
        // (rest of the SLM allocation is unused)
        uint32_t bm_words = (total_kv_blocks + 31u) / 32u;  // e.g. 256kv_blocks->8 words, 2048->64 words

        // Zero only the needed bitmap words (in 32-byte = 8-word chunks)
        {
            simd<uint32_t, 8> zero(0u);
            uint32_t bm_bytes = bm_words * 4u;
            // round up to 32-byte boundary for block store
            uint32_t zero_bytes = (bm_bytes + 31u) & ~31u;
            for (uint32_t b = 0u; b < zero_bytes; b += 32u)
                lsc_slm_block_store<uint32_t, 8>(b, zero);
        }

        uint32_t q_start = q_block * 16u;
        uint32_t q_end   = (q_start + 16u < qlen) ? q_start + 16u : qlen;

        // For each of the 16 q-positions in this q-block:
        //   Load its 64 kv_block indices (block_load x2 of 32 elements each)
        //   Set the corresponding bits in the SLM bitmap
        for (uint32_t qi = q_start; qi < q_end; qi++) {
            uint32_t base = (kv_head * qlen + qi) * 64u;
            simd<uint32_t, 32> e0 = block_load<uint32_t, 32>(mask_orig + base);
            simd<uint32_t, 32> e1 = block_load<uint32_t, 32>(mask_orig + base + 32u);

            // For each index, set bit (index) in SLM bitmap:
            //   word_byte_offset = (index / 32) * 4
            //   bit_mask         = 1 << (index % 32)
            //   bm[word] |= bit_mask  (SLM read-modify-write)
            for (int i = 0; i < 32; i++) {
                uint32_t idx  = e0[i];
                uint32_t boff = (idx >> 5) * 4u;
                uint32_t old  = lsc_slm_block_load<uint32_t, 1>(boff)[0];
                lsc_slm_block_store<uint32_t, 1>(boff, simd<uint32_t,1>(old | (1u << (idx & 31u))));
            }
            for (int i = 0; i < 32; i++) {
                uint32_t idx  = e1[i];
                uint32_t boff = (idx >> 5) * 4u;
                uint32_t old  = lsc_slm_block_load<uint32_t, 1>(boff)[0];
                lsc_slm_block_store<uint32_t, 1>(boff, simd<uint32_t,1>(old | (1u << (idx & 31u))));
            }
        }

        // Scan bitmap words, collect set bits -> sorted unique kv_block indices
        // Output index <= 1023 always (union of 16*64 entries)
        uint32_t out_base = (kv_head * q_blks + q_block) * 1024u;
        uint32_t cnt = 0u;
        for (uint32_t w = 0u; w < bm_words; w++) {
            uint32_t word = lsc_slm_block_load<uint32_t, 1>(w * 4u)[0];
            while (word) {
                mask_out[out_base + cnt++] = w * 32u + (uint32_t)__builtin_ctz(word);
                word &= word - 1u;  // clear lowest set bit
            }
        }
        mask_cnt_out[kv_head * q_blks + q_block] = cnt;
    }
};
