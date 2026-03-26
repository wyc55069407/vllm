"""Profile MiniCPM5 batch-1 decode at long context (~25K tokens).

Modes:
  --mode dense   : ESIMD_ATTN prefill + ESIMD MoE decode
  --mode sparse  : INFLLMV2_ESIMD_ATTN prefill + ESIMD MoE decode

Usage:
  MINICPM5_ESIMD_MOE=1 python test_minicpm5_profile_long.py --mode dense
  MINICPM5_ESIMD_MOE=1 python test_minicpm5_profile_long.py --mode sparse
"""
import os, sys, argparse, urllib.request, json
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("MINICPM5_ESIMD_MOE", "1")

MODEL_PATH = "/home/sas/yuchen/vllm_env/models/minicpm5.16a3.v0314-GPTQ-Int4"
PROXY = "http://child-prc.intel.com:913"


def fetch_wiki(title, target_chars=24000):
    proxy_handler = urllib.request.ProxyHandler({'http': PROXY, 'https': PROXY})
    opener = urllib.request.build_opener(proxy_handler)
    url = (f"https://en.wikipedia.org/w/api.php?action=query&titles={urllib.request.quote(title)}"
           f"&prop=extracts&explaintext=1&format=json&exlimit=1")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = opener.open(req, timeout=30)
        data = json.loads(resp.read().decode())
        text = next(iter(data["query"]["pages"].values())).get("extract", "")
        if len(text) > target_chars:
            cut = text[:target_chars].rfind(". ")
            text = text[:cut+1] if cut > target_chars * 0.8 else text[:target_chars]
        return text
    except Exception as e:
        print(f"  WARNING: fetch failed for '{title}': {e}")
        return None


def pad_text(text, target_chars):
    if len(text) >= target_chars:
        return text[:target_chars]
    result = text
    while len(result) < target_chars:
        result += f"\n\n[Continued]\n\n" + text
    return result[:target_chars]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # Build ~25K token prompt from 5 Wikipedia articles
    topics = [
        ("Plate_tectonics", "Plate Tectonics"),
        ("History_of_chess", "Chess History"),
        ("Honey_bee", "Honey Bees"),
        ("Roman_concrete", "Roman Engineering"),
        ("History_of_photography", "Photography History"),
    ]
    sections = []
    for i, (wiki, name) in enumerate(topics):
        text = fetch_wiki(wiki, 24000)
        if not text or len(text) < 3000:
            text = f"This section covers {name}. " * 200
        elif len(text) < 14000:
            text = pad_text(text, 24000)
        sections.append(f"\n<<Section {i+1}: {name}>>\n\n{text}")
        print(f"  [{i+1}] {len(text)} chars: {name}")

    raw = ("Read 5 sections and answer questions about them.\n"
           + "".join(sections)
           + "\n\nBased on ALL sections, give one fact from each section.")
    msgs = [{"role": "user", "content": raw}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt += "<think>\n</think>\n\n"

    if args.mode == "sparse":
        attn_backend = "INFLLMV2_ESIMD_ATTN"
    else:
        attn_backend = "ESIMD_ATTN"

    print(f"\nMode: {args.mode}, attention_backend={attn_backend}")
    print(f"MINICPM5_ESIMD_MOE={os.environ.get('MINICPM5_ESIMD_MOE', '0')}")

    llm = LLM(
        model=MODEL_PATH, dtype="bfloat16", trust_remote_code=True,
        enforce_eager=True, disable_log_stats=True, gpu_memory_utilization=0.90,
        max_model_len=32768, max_num_seqs=1, block_size=128,
        attention_backend=attn_backend,
    )

    params = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    outputs = llm.generate([prompt], params)
    pt = len(outputs[0].prompt_token_ids)
    gt = len(outputs[0].outputs[0].token_ids)
    print(f"\nLong context ({args.mode}): prompt={pt}, generated={gt}")
    print(f"Output: {outputs[0].outputs[0].text[:300]}")
    del llm
