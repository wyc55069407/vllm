import torch
try:
    import intel_extension_for_pytorch
except ImportError:
    pass
import time
from vllm_kernel_custom import esimd_sdp_paged

def bench(batch, seq_len, nh, nkvh, hd, bs, dtype, warmup=10, repeat=200):
    device = 'xpu'
    N_BUF = 4
    queries = [torch.randn(batch, nh, hd, dtype=dtype, device=device) for _ in range(N_BUF)]
    num_blocks_per_seq = (seq_len + bs - 1) // bs
    total_blocks = batch * num_blocks_per_seq * N_BUF
    kv_cache = torch.randn(2, total_blocks, bs, nkvh, hd, dtype=dtype, device=device)
    block_tables = []
    for buf in range(N_BUF):
        bt = torch.zeros(batch, num_blocks_per_seq, dtype=torch.int32, device=device)
        for b in range(batch):
            for i in range(num_blocks_per_seq):
                bt[b, i] = (buf * batch + b) * num_blocks_per_seq + i
        block_tables.append(bt)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    query_start_loc = torch.arange(batch + 1, dtype=torch.int32, device=device)
    outputs = [torch.empty_like(queries[0]) for _ in range(N_BUF)]
    attn_scale = 1.0 / (hd ** 0.5)
    for i in range(warmup):
        buf = i % N_BUF
        esimd_sdp_paged(queries[buf], kv_cache, outputs[buf], block_tables[buf],
                        seq_lens, query_start_loc, nh, nkvh, hd, bs, seq_len, attn_scale, 0)
    torch.xpu.synchronize()
    start = time.perf_counter()
    for i in range(repeat):
        buf = i % N_BUF
        esimd_sdp_paged(queries[buf], kv_cache, outputs[buf], block_tables[buf],
                        seq_lens, query_start_loc, nh, nkvh, hd, bs, seq_len, attn_scale, 0)
    torch.xpu.synchronize()
    elapsed = time.perf_counter() - start
    avg_us = elapsed / repeat * 1e6
    kv_bytes = batch * seq_len * nkvh * hd * 2 * 2
    gbps = kv_bytes / (elapsed / repeat) / 1e9
    return avg_us, gbps

if __name__ == '__main__':
    seq_lens = [1024, 2048, 4096, 8192, 16384]
    configs = [
        ('32Q/2KV fp16', 32, 2, 128, 128, torch.float16),
        ('32Q/2KV bf16', 32, 2, 128, 128, torch.bfloat16),
        ('32Q/4KV fp16', 32, 4, 128, 128, torch.float16),
        ('32Q/4KV bf16', 32, 4, 128, 128, torch.bfloat16),
    ]
    for batch_size in [1, 8]:
        for label, nh, nkvh, hd, bs, dtype in configs:
            print(f'--- {label}, batch={batch_size} ---')
            for sl in seq_lens:
                t, g = bench(batch_size, sl, nh, nkvh, hd, bs, dtype)
                print(f'  seq={sl:>5}  time={t:>8.1f}us  bw={g:>8.1f} GB/s')
