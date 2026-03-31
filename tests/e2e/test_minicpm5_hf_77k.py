#!/usr/bin/env python3
"""
HF transformers reference path for MiniCPM5-MoE GPTQ-Int4 at 77K+ context.
Adapted from hf_path_gptq_pass for Linux/BMG.

This is SLOW: ~10 min prefill, ~1 hour decode 1024 tokens.
Uses custom QuantLinear with fp32 dequant, chunked prefill via DynamicCache.
"""
import sys, os, gc, time, json, urllib.request, urllib.parse
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import torch
import torch.nn as nn
from safetensors import safe_open

os.environ.setdefault('HTTPS_PROXY', 'http://child-prc.intel.com:913')
os.environ.setdefault('HTTP_PROXY', 'http://child-prc.intel.com:913')

# Fix XPU bug: HF transformers skips causal mask creation on XPU even when
# q_len != kv_len. This breaks chunked prefill.
import transformers.masking_utils as _mu
_mu._is_torch_xpu_available = False

MODEL_DIR = "/home/sas/yuchen/vllm_env/models/minicpm5.16a3.v0314-GPTQ-Int4"
DEVICE = "xpu"
DTYPE = torch.float16
BITS = 4
GROUP_SIZE = 128
PACK_FACTOR = 32 // BITS

PROXY = "http://child-prc.intel.com:913"


# ============================================================
# QuantLinear with fp32 dequant (reference implementation)
# ============================================================
class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('qweight', torch.zeros(
            in_features // PACK_FACTOR, out_features, dtype=torch.int32))
        self.register_buffer('qzeros', torch.zeros(
            in_features // GROUP_SIZE, out_features // PACK_FACTOR,
            dtype=torch.int32))
        self.register_buffer('scales', torch.zeros(
            in_features // GROUP_SIZE, out_features, dtype=torch.float16))
        self.register_buffer('g_idx', torch.zeros(
            in_features, dtype=torch.int32))
        self.bias = None
        self.register_buffer('_shifts', torch.arange(
            0, 32, BITS, dtype=torch.int32), persistent=False)

    def _apply(self, fn):
        # Prevent .to(dtype=float16) from destroying int32 buffers
        for key in ['qweight', 'qzeros', 'g_idx', '_shifts']:
            buf = getattr(self, key, None)
            if buf is not None:
                new_buf = fn(buf)
                if new_buf.dtype != torch.int32:
                    new_buf = buf.to(device=new_buf.device)
                setattr(self, key, new_buf)
        if self.scales is not None:
            self.scales = fn(self.scales)
        return self

    def forward(self, x):
        weight = self._dequantize()
        return x @ weight.t()

    def _dequantize(self):
        """Unpack int4 -> float16 weight [out_features, in_features].
        Dequant in fp32 for maximum precision."""
        mask = (1 << BITS) - 1  # 0xF

        # Unpack qweight: [in//8, out] -> [in, out]
        int_w = torch.zeros(self.in_features, self.out_features,
                            dtype=torch.int32, device=self.qweight.device)
        for j in range(PACK_FACTOR):
            int_w[j::PACK_FACTOR] = (self.qweight >> (BITS * j)) & mask

        # Unpack qzeros: [G, out//8] -> [G, out]
        num_groups = self.qzeros.shape[0]
        zeros = torch.zeros(num_groups, self.out_features,
                            dtype=torch.int32, device=self.qzeros.device)
        for j in range(PACK_FACTOR):
            zeros[:, j::PACK_FACTOR] = (self.qzeros >> (BITS * j)) & mask
        zeros = zeros + 1  # GPTQ v1 offset

        # Dequantize in fp32
        scales_row = self.scales[self.g_idx]   # [in, out]
        zeros_row = zeros[self.g_idx]          # [in, out]
        weight = ((int_w.float() - zeros_row.float())
                  * scales_row.float()).to(self.scales.dtype)
        return weight.t()  # [out, in]


# ============================================================
# Wikipedia corpus
# ============================================================
WIKI_TOPICS = [
    "History of the Roman Empire",
    "Quantum mechanics",
    "World War II",
    "Theory of evolution",
    "History of China",
    "Machine learning",
    "Renaissance",
    "Climate change",
    "History of mathematics",
    "Human genome project",
]


def fetch_wiki(title, max_chars=80000):
    proxy_handler = urllib.request.ProxyHandler(
        {"http": PROXY, "https": PROXY})
    opener = urllib.request.build_opener(proxy_handler)
    params = urllib.parse.urlencode({
        'action': 'query', 'titles': title, 'prop': 'extracts',
        'explaintext': 1, 'format': 'json', 'exlimit': 1,
    })
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    print(f"  Fetching: {title}...", flush=True)
    req = urllib.request.Request(url,
                                headers={'User-Agent': 'MiniCPM5-Test/1.0'})
    with opener.open(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    pages = data['query']['pages']
    page = next(iter(pages.values()))
    text = page.get('extract', '')
    if len(text) > max_chars:
        text = text[:max_chars]
    print(f"    Got {len(text)} chars", flush=True)
    return text


def build_long_context(tokenizer, target_tokens=78000):
    print(f"\nBuilding {target_tokens}+ token context from Wikipedia...",
          flush=True)
    all_text = []
    for topic in WIKI_TOPICS:
        try:
            text = fetch_wiki(topic)
            if text:
                all_text.append(f"\n\n=== {topic} ===\n\n{text}")
        except Exception as e:
            print(f"    Failed: {e}", flush=True)

    combined = "\n".join(all_text)
    tokens = tokenizer.encode(combined)
    print(f"  Total tokens from wiki: {len(tokens)}", flush=True)
    while len(tokens) < target_tokens:
        combined = combined + "\n" + combined
        tokens = tokenizer.encode(combined)
        print(f"  Repeated -> {len(tokens)} tokens", flush=True)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
        combined = tokenizer.decode(tokens)
    print(f"  Final context: {len(tokenizer.encode(combined))} tokens",
          flush=True)
    return combined


# ============================================================
# Model loading
# ============================================================
def get_full_name_map(model, prefix=""):
    result = {}
    for name, child in model.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            result[full] = (model, name, child)
        else:
            result.update(get_full_name_map(child, full))
    return result


def load_quantized_model():
    sys.path.insert(0, MODEL_DIR)
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
    # Remove quantization_config so HF doesn't try its own GPTQ path
    if hasattr(config, 'quantization_config'):
        delattr(config, 'quantization_config')
    if 'quantization_config' in config.__dict__:
        del config.__dict__['quantization_config']

    print("Building model on meta device...", flush=True)
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True)

    idx_path = os.path.join(MODEL_DIR, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]
    quant_bases = {k.removesuffix(".qweight")
                   for k in weight_map if k.endswith(".qweight")}

    linear_map = get_full_name_map(model)
    replaced = 0
    for full_name, (parent, attr, linear) in linear_map.items():
        if full_name in quant_bases:
            ql = QuantLinear(linear.in_features, linear.out_features)
            setattr(parent, attr, ql)
            replaced += 1
    print(f"  Replaced {replaced} Linear -> QuantLinear", flush=True)

    print("Materializing on CPU...", flush=True)
    model = model.to_empty(device='cpu')

    # Fix RoPE: inv_freq is non-persistent buffer, lost on to_empty
    for name, mod in model.named_modules():
        if hasattr(mod, 'inv_freq') and hasattr(mod, 'rope_init_fn'):
            inv_freq, mod.attention_scaling = mod.rope_init_fn(
                mod.config, device='cpu')
            mod.inv_freq = inv_freq
            mod.original_inv_freq = inv_freq
            print(f"  Reinitialized RoPE: {name}", flush=True)

    # Load weights from safetensors
    shard_files = sorted(set(weight_map.values()))
    print(f"Loading weights from {len(shard_files)} shards...", flush=True)
    for shard in shard_files:
        path = os.path.join(MODEL_DIR, shard)
        print(f"  Loading {shard}...", flush=True)
        with safe_open(path, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                tensor = sf.get_tensor(key)
                parts = key.split(".")
                obj = model
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                attr = parts[-1]
                target = getattr(obj, attr, None)
                if target is not None and isinstance(target, torch.Tensor):
                    target.data.copy_(tensor)
                elif hasattr(obj, attr):
                    setattr(obj, attr, tensor)
                del tensor
        gc.collect()

    print(f"Moving to {DEVICE}...", flush=True)
    model = model.to(device=DEVICE, dtype=DTYPE)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, trust_remote_code=True)
    gc.collect()
    torch.xpu.empty_cache()
    print(f"  XPU memory: {torch.xpu.memory_allocated() / 2**30:.2f} GB",
          flush=True)
    return model, tokenizer


# ============================================================
# Sampling
# ============================================================
def sample_logits(logits, generated, temperature=0.7, top_k=50, top_p=0.9,
                  repetition_penalty=1.2):
    """Apply temperature, repetition penalty, top-k, top-p sampling."""
    logits = logits.float()

    # Repetition penalty
    if repetition_penalty != 1.0 and len(generated) > 0:
        prev_ids = torch.tensor(
            generated, device=logits.device, dtype=torch.long).unique()
        penalty_logits = logits[:, prev_ids]
        penalty_logits = torch.where(
            penalty_logits > 0,
            penalty_logits / repetition_penalty,
            penalty_logits * repetition_penalty,
        )
        logits[:, prev_ids] = penalty_logits

    # Temperature
    if temperature > 0 and temperature != 1.0:
        logits = logits / temperature

    # Top-k
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth_val = torch.topk(logits, top_k, dim=-1).values[:, -1:]
        logits = torch.where(
            logits < kth_val,
            torch.full_like(logits, float('-inf')), logits)

    # Top-p (nucleus)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = (cumulative_probs
                       - torch.softmax(sorted_logits, dim=-1) >= top_p)
        sorted_logits[sorted_mask] = float('-inf')
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1)
    return next_id


# ============================================================
# Chunked prefill + decode
# ============================================================
@torch.no_grad()
def chunked_prefill_decode(model, tokenizer, prompt, question,
                           chunk_size=4096, max_new=1024,
                           temperature=0.7, top_k=50, top_p=0.9,
                           repetition_penalty=1.2):
    full_text = prompt + "\n\nQuestion: " + question + "\nAnswer:"
    input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(DEVICE)
    seq_len = input_ids.shape[1]
    print(f"\n{'='*60}", flush=True)
    print(f"Chunked prefill: {seq_len} tokens (chunk={chunk_size}), "
          f"decode {max_new}", flush=True)
    print(f"Sampling: temp={temperature}, top_k={top_k}, top_p={top_p}, "
          f"rep_penalty={repetition_penalty}", flush=True)
    print(f"{'='*60}", flush=True)

    from transformers import DynamicCache
    past_kv = DynamicCache()

    t0 = time.time()
    logits_last = None
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_ids = input_ids[:, start:end]
        cache_pos = torch.arange(start, end, device=DEVICE)
        out = model(
            input_ids=chunk_ids, cache_position=cache_pos,
            past_key_values=past_kv, use_cache=True,
        )
        past_kv = out.past_key_values
        logits_last = out.logits[:, -1:, :]
        elapsed = time.time() - t0
        xpu_mem = torch.xpu.memory_allocated() / 2**30
        print(f"  Prefill {end:>6}/{seq_len} ({end/seq_len*100:5.1f}%) | "
              f"{elapsed:6.1f}s | {end/elapsed:6.0f} tok/s | "
              f"XPU: {xpu_mem:.2f} GB", flush=True)
        del out
        gc.collect()

    t_prefill = time.time() - t0
    print(f"\n  Prefill done: {seq_len} tokens in {t_prefill:.1f}s "
          f"({seq_len/t_prefill:.0f} tok/s)", flush=True)

    # Decode
    print(f"\n  Decoding {max_new} tokens...", flush=True)
    t1 = time.time()
    next_id = sample_logits(
        logits_last.squeeze(1), [], temperature=temperature,
        top_k=top_k, top_p=top_p, repetition_penalty=1.0)
    generated = [next_id.item()]
    eos_ids = tokenizer.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]

    for step in range(1, max_new):
        pos = seq_len + step - 1
        cache_pos = torch.tensor([pos], device=DEVICE)
        out = model(
            input_ids=next_id,
            cache_position=cache_pos,
            past_key_values=past_kv, use_cache=True,
        )
        next_id = sample_logits(
            out.logits[:, -1, :], generated,
            temperature=temperature, top_k=top_k,
            top_p=top_p, repetition_penalty=repetition_penalty)
        generated.append(next_id.item())
        past_kv = out.past_key_values
        if next_id.item() in eos_ids:
            print(f"  EOS at step {step+1}", flush=True)
            break
        if (step + 1) % 50 == 0:
            elapsed = time.time() - t1
            tps = (step + 1) / elapsed
            last_text = tokenizer.decode(
                generated[-30:], skip_special_tokens=True)
            print(f"  Decoded {step+1:>5}/{max_new} | {elapsed:6.1f}s | "
                  f"{tps:.2f} tok/s | ...{last_text[-80:]}", flush=True)

    t_decode = time.time() - t1
    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"\n  Decode: {len(generated)} tokens in {t_decode:.1f}s "
          f"({len(generated)/t_decode:.2f} tok/s)", flush=True)
    print(f"\n{'='*60}", flush=True)
    print(f"Generated ({len(generated)} tokens):", flush=True)
    print(f"{'='*60}", flush=True)
    print(text, flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Final XPU mem: "
          f"{torch.xpu.memory_allocated() / 2**30:.2f} GB", flush=True)
    return text


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-tokens", type=int, default=78000,
                        help="Target context length in tokens")
    parser.add_argument("--max-new", type=int, default=1024,
                        help="Max decode tokens")
    parser.add_argument("--chunk-size", type=int, default=4096,
                        help="Prefill chunk size")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--rep-pen", type=float, default=1.2)
    args = parser.parse_args()

    model, tokenizer = load_quantized_model()
    context = build_long_context(tokenizer,
                                 target_tokens=args.target_tokens)
    question = (
        "Based on all the articles above, provide a comprehensive summary "
        "covering the key themes across all topics. For each topic, mention "
        "the most important facts and how they connect to each other. "
        "Be detailed and thorough in your response."
    )
    result = chunked_prefill_decode(
        model, tokenizer, context, question,
        chunk_size=args.chunk_size, max_new=args.max_new,
        temperature=args.temperature, top_k=args.top_k,
        top_p=args.top_p, repetition_penalty=args.rep_pen)

    # Quality check
    words = result.split()[:500]
    if len(words) >= 4:
        ngrams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
        unique_ratio = len(set(ngrams)) / len(ngrams) if ngrams else 1.0
    else:
        unique_ratio = 1.0
    print(f"\nQuality: unique_4gram={unique_ratio:.3f}", flush=True)
    print(f"\n=== HF reference path complete ===", flush=True)


if __name__ == "__main__":
    main()
