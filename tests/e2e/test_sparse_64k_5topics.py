"""Test sparse prefill+decode at ~64K tokens with 5 unrelated topics.
Uses RoPE scaling (dynamic NTK or YaRN factor=2) to extend MiniCPM4-8B from 32K to 64K.
Fetches real Wikipedia content for each topic.
"""
import os, urllib.request, json
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODEL_PATH = "/home/sas/yuchen/vllm_env/models/MiniCPM4-8B"
PROXY = "http://child-prc.intel.com:913"

def fetch_wiki(title, target_chars=50000):
    """Fetch Wikipedia article plaintext via API."""
    proxy_handler = urllib.request.ProxyHandler({'http': PROXY, 'https': PROXY})
    opener = urllib.request.build_opener(proxy_handler)
    url = (f"https://en.wikipedia.org/w/api.php?action=query&titles={urllib.request.quote(title)}"
           f"&prop=extracts&explaintext=1&format=json&exlimit=1")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = opener.open(req, timeout=30)
        data = json.loads(resp.read().decode())
        pages = data["query"]["pages"]
        text = next(iter(pages.values())).get("extract", "")
        if len(text) > target_chars:
            cut = text[:target_chars].rfind(". ")
            text = text[:cut+1] if cut > target_chars * 0.8 else text[:target_chars]
        return text
    except Exception as e:
        print(f"  WARNING: fetch failed for '{title}': {e}")
        return None

# Fallback content for each topic (~12K tokens = ~48K chars each)
FALLBACKS = {
    1: ("Plate Tectonics", """The theory of plate tectonics unified decades of geological observations into a comprehensive framework. Alfred Wegener proposed continental drift in 1912, noting how South America and Africa fit together. Harry Hess proposed seafloor spreading in 1962. Frederick Vine and Drummond Matthews discovered magnetic stripe patterns on the ocean floor in 1963. Earth's lithosphere is divided into approximately 15 major tectonic plates. The Pacific Plate moves northwest at 7-11 centimeters per year. The Himalayas formed when the Indian Plate collided with Eurasia 50 million years ago. The 2011 Tohoku earthquake (magnitude 9.1) occurred at the Japan Trench subduction zone. The San Andreas Fault marks the Pacific-North American transform boundary. The Hawaiian islands formed over a mantle plume hotspot. Yellowstone's caldera-forming eruptions occurred 2.1 million, 1.3 million, and 640,000 years ago. The Wilson Cycle describes ocean basins opening and closing over 200-500 million years. Pangaea existed from 335 to 175 million years ago. Mid-ocean ridges produce new oceanic crust through volcanic activity. The Mariana Trench reaches 10,994 meters depth at the Challenger Deep. Convergent boundaries create volcanic arcs and deep ocean trenches. Divergent boundaries produce rift valleys like the East African Rift. The Ring of Fire encircles the Pacific Ocean with 75% of Earth's active volcanoes. Earthquakes along subduction zones can generate devastating tsunamis. Paleomagnetism confirmed continental drift through polar wander curves. Seafloor age increases with distance from mid-ocean ridges, supporting spreading theory."""),

    2: ("Chess History", """Chess originated in India during the Gupta Empire around the 6th century AD as chaturanga. The game spread to Persia where it became shatranj. The phrase "shah mat" gave us "checkmate." Arab players developed the first notation systems. The modern queen's move was established in Spain circa 1475, possibly influenced by Queen Isabella. The first World Championship was held in 1886 with Steinitz defeating Zukertort. Steinitz pioneered positional play theory. The Soviet Union dominated from 1948-1972 through Botvinnik, Smyslov, Tal, Petrosian, and Spassky. Bobby Fischer broke Soviet dominance in 1972 in Reykjavik. Kasparov held the title from 1985-2000 and lost to Deep Blue in 1997. AlphaZero learned chess through self-play in 2017. The Elo rating system was adopted by FIDE in 1970. The King's Indian Defense was popularized by Bronstein and Boleslavsky. The Sicilian Defense is the most popular response to 1.e4. Magnus Carlsen became champion in 2013 and held the title until 2023. Online platforms like Chess.com host hundreds of millions of players. The Queen's Gambit Netflix series sparked a global chess renaissance in 2020. FIDE was founded in Paris in 1924. The longest possible chess game is 5,949 moves."""),

    3: ("Honey Bee Biology", """A honey bee colony contains approximately 60,000 individuals functioning as a superorganism. The queen can lay 2,000 eggs per day during peak season. She mates once during a flight with 12-20 drones. Karl von Frisch received the Nobel Prize in 1973 for discovering the waggle dance. The dance angle relative to vertical indicates direction relative to the sun. One second of waggling corresponds to approximately one kilometer distance. Workers live 6 weeks in summer and progress through age-dependent roles. Foragers visit 50-100 flowers per trip and fly 800 kilometers in a lifetime. One forager produces about one-twelfth of a teaspoon of honey. The brood area is maintained at 34-36 degrees Celsius. In winter, the cluster core stays near 35°C even at -30°C external temperature. Colony collapse disorder was identified in 2006. Varroa destructor mites transmit viruses to bees. Neonicotinoid pesticides impair navigation and learning. Beeswax is secreted from abdominal glands. Propolis is collected from tree buds for sealing and sterilization. Royal jelly contains the unique compound royalactin. US honey bees pollinate crops worth over 15 billion dollars annually. California's almond bloom requires 2 million colonies. One-third of the human diet depends on bee pollination."""),

    4: ("Roman Engineering", """Roman concrete (opus caementicium) has survived 2,000 years and grows stronger through mineral crystallization. The recipe used volcanic ash from Pozzuoli, lime, seawater, and volcanic rock. Marie Jackson's 2017 research showed tobermorite crystals reinforcing the material. The Pantheon dome spans 43.3 meters unreinforced, a record for 1,300 years. Its coffers reduce weight by 3,600 kg each. The oculus is 8.2 meters in diameter. The Pont du Gard stands 48.8 meters tall with three tiers of arches. The Nimes aqueduct drops only 17 meters over 50 kilometers. Rome's 11 aqueducts delivered 1 million cubic meters daily. Roman roads spanned 400,000 kilometers from Britain to Mesopotamia. The Via Appia still bears traffic after 2,300 years. The Colosseum seated 50,000 and could empty in 15 minutes through 80 vomitoria. Its velarium awning was operated by sailors. The hypogeum had 80 vertical elevators. Hypocaust heating circulated hot air beneath floors. The Baths of Caracalla served 6,000-8,000 bathers across 25 acres. Caesarea Maritima used underwater concrete caissons. Portus harbor covered 200 hectares. Rome imported 400,000 tons of grain annually from Egypt. The Cloaca Maxima sewer system still functions today."""),

    5: ("Photography History", """The earliest photograph by Nicéphore Niépce in 1826/1827 required eight hours of exposure. Louis Daguerre presented the daguerreotype to the French Academy in 1839. William Henry Fox Talbot's calotype (1841) introduced the negative-positive system. "The Pencil of Nature" (1844-46) was the first photographically illustrated book. Frederick Scott Archer's wet collodion process (1851) captured detail in 2-20 seconds. Mathew Brady documented the American Civil War. George Eastman's 1888 Kodak camera used roll film with the slogan "You press the button, we do the rest." The Brownie camera cost one dollar in 1900. James Clerk Maxwell created the first color photograph in 1861. Autochrome by the Lumière brothers debuted in 1907. Kodachrome launched in 1935 and lasted until 2009. The CCD was invented at Bell Labs in 1969 by Boyle and Smith (Nobel Prize 2009). Kodak's first digital camera (1975) weighed 3.6 kg at 0.01 megapixels. The iPhone in 2007 made camera phones universal. Today 1.81 trillion photos are taken annually. Ansel Adams pioneered the Zone System for exposure control. Henri Cartier-Bresson defined the decisive moment in street photography. Polaroid instant film was invented by Edwin Land in 1948. The Leica camera revolutionized photojournalism in 1925."""),
}

def get_content(topic_num, wiki_title, target_chars):
    text = fetch_wiki(wiki_title, target_chars)
    if text and len(text) >= 5000:
        print(f"  Fetched {len(text)} chars for topic {topic_num}: {wiki_title}")
        return text
    name, fallback = FALLBACKS[topic_num]
    print(f"  Using fallback for topic {topic_num}: {name}")
    # Pad fallback to target
    result = fallback
    while len(result) < target_chars:
        result += "\n\n" + fallback
    return result[:target_chars]

QUESTION = """

Based on ALL 5 sections above, answer the following. For each section give specific facts with names, numbers, or dates:
1. Section 1 (Plate Tectonics): Name 2 specific geological events with magnitudes or dates.
2. Section 2 (Chess): Name 2 specific players and what they achieved, with years.
3. Section 3 (Honey Bees): Give 2 specific numerical facts about bee biology.
4. Section 4 (Roman Engineering): Name 2 specific structures with their dimensions.
5. Section 5 (Photography): Name 2 inventors and their inventions with dates.
Keep each answer to 2-3 sentences."""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="sparse", choices=["dense", "sparse", "dense_prefill"])
    parser.add_argument("--rope", default="dynamic", choices=["dynamic", "yarn", "longrope_env"],
                        help="RoPE scaling method: dynamic NTK, YaRN, or original longrope with env override")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    if args.mode == "dense":
        backend = "ESIMD_ATTN"
    elif args.mode == "sparse":
        backend = "INFLLMV2_ESIMD_ATTN"
    elif args.mode == "dense_prefill":
        backend = "INFLLMV2_ESIMD_ATTN"
        os.environ["INFLLMV2_SPARSE_PREFILL"] = "0"

    # Build RoPE override config
    if args.rope == "dynamic":
        hf_overrides = {
            "max_position_embeddings": 65536,
            "rope_scaling": {
                "rope_type": "dynamic",
                "factor": 2.0,
            }
        }
        rope_desc = "Dynamic NTK (factor=2)"
    elif args.rope == "yarn":
        hf_overrides = {
            "max_position_embeddings": 65536,
            "rope_scaling": {
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 32768,
            }
        }
        rope_desc = "YaRN (factor=2)"
    else:  # longrope_env
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        hf_overrides = {
            "max_position_embeddings": 65536,
        }
        rope_desc = "LongRoPE extrapolation (env override)"

    # Fetch 5 topics, each ~50K chars (~12.5K tokens), total ~62.5K tokens
    wiki_titles = [
        "Plate_tectonics",
        "History_of_chess",
        "Honey_bee",
        "Roman_concrete",
        "History_of_photography",
    ]
    section_names = [
        "Plate Tectonics and Continental Drift",
        "History of Chess",
        "Honey Bee Colony Biology",
        "Ancient Roman Concrete and Engineering",
        "History of Photography",
    ]
    target_per_topic = 50000  # ~12.5K tokens each

    print(f"Fetching content for 5 topics (~{target_per_topic} chars each)...")
    sections = []
    for i, (wiki, name) in enumerate(zip(wiki_titles, section_names)):
        text = get_content(i + 1, wiki, target_per_topic)
        sections.append(f"\n<<Section {i+1}: {name}>>\n\n{text}")

    prompt = "Read the following 5 sections carefully. You will be asked questions about ALL of them.\n" + "".join(sections) + QUESTION

    print(f"\nMode: {args.mode}, RoPE: {rope_desc}, backend: {backend}")
    print(f"Prompt: {len(prompt)} chars (~{len(prompt)//4} tokens est.)")

    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", trust_remote_code=True,
              enforce_eager=True, disable_log_stats=True, gpu_memory_utilization=0.90,
              max_model_len=65536, max_num_seqs=1, block_size=128,
              attention_backend=backend,
              hf_overrides=hf_overrides)
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    outputs = llm.generate([prompt], params)
    out = outputs[0]
    pt = len(out.prompt_token_ids)
    gt = len(out.outputs[0].token_ids)
    text = out.outputs[0].text

    print(f"Prompt tokens: {pt}")
    print(f"Generated tokens: {gt}")
    print(f"\n{'='*70}")
    print(f"  OUTPUT ({args.mode}, {rope_desc})")
    print(f"{'='*70}")
    print(text)
    print(f"{'='*70}")

    tl = text.lower()
    checks = {
        "Plate Tectonics": any(w in tl for w in ["tectonic", "wegener", "pangaea", "subduction", "san andreas", "earthquake", "himalaya", "rift", "tohoku"]),
        "Chess":           any(w in tl for w in ["chess", "kasparov", "fischer", "steinitz", "deep blue", "botvinnik", "carlsen"]),
        "Honey Bees":      any(w in tl for w in ["bee", "waggle", "honey", "queen", "colony", "pollen", "hive", "forager"]),
        "Roman Eng":       any(w in tl for w in ["roman", "pantheon", "concrete", "aqueduct", "colosseum", "pont du gard"]),
        "Photography":     any(w in tl for w in ["photograph", "daguerr", "kodak", "camera", "niépce", "niepce", "talbot", "eastman"]),
    }
    print(f"\nQuality — topics discovered:")
    for k, v in checks.items():
        print(f"  {k}: {'YES' if v else 'NO'}")
    print(f"  Score: {sum(checks.values())}/5 topics covered")
    del llm
