"""Test sparse prefill+decode at ~31K tokens with 5 unrelated topics.
Fetches real content from Wikipedia to get unique, non-repetitive text.
Each topic ~6K tokens. Tests if model can discover and summarize all 5.
"""
import os, urllib.request, json, re, textwrap
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

MODEL_PATH = "/home/sas/yuchen/vllm_env/models/MiniCPM4-8B"

PROXY = "http://child-prc.intel.com:913"

def fetch_wiki(title, target_chars=24000):
    """Fetch Wikipedia article plaintext via API."""
    proxy_handler = urllib.request.ProxyHandler({
        'http': PROXY, 'https': PROXY})
    opener = urllib.request.build_opener(proxy_handler)
    url = (f"https://en.wikipedia.org/w/api.php?action=query&titles={urllib.request.quote(title)}"
           f"&prop=extracts&explaintext=1&format=json&exlimit=1")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = opener.open(req, timeout=30)
        data = json.loads(resp.read().decode())
        pages = data["query"]["pages"]
        page = next(iter(pages.values()))
        text = page.get("extract", "")
        # Truncate to target
        if len(text) > target_chars:
            # Cut at sentence boundary
            cut = text[:target_chars].rfind(". ")
            if cut > target_chars * 0.8:
                text = text[:cut+1]
            else:
                text = text[:target_chars]
        return text
    except Exception as e:
        print(f"  WARNING: Failed to fetch '{title}': {e}")
        return None


def generate_fallback(topic_num, title, target_chars=24000):
    """Generate unique content if wiki fetch fails."""
    # Each topic has completely different content
    topics = {
        1: {
            "title": "Plate Tectonics and Continental Drift",
            "content": """The theory of plate tectonics, established in the 1960s, unified decades of geological observations into a comprehensive framework explaining earthquakes, volcanism, mountain building, and oceanic trench formation. Alfred Wegener first proposed continental drift in 1912, noting how the coastlines of South America and Africa fit together like puzzle pieces, and that identical fossil species like Mesosaurus appeared on both continents. He was ridiculed by the scientific establishment, who could not explain the mechanism driving continental movement.

The breakthrough came from oceanographic surveys in the 1950s-60s. Harry Hess proposed seafloor spreading in 1962, suggesting that new oceanic crust forms at mid-ocean ridges and spreads outward. The discovery of magnetic stripe patterns on the ocean floor by Frederick Vine and Drummond Matthews in 1963 provided compelling evidence: as magma solidifies at ridges, magnetic minerals align with Earth's field, recording periodic reversals. The symmetric pattern of normal and reversed stripes on either side of the Mid-Atlantic Ridge confirmed that the seafloor was indeed spreading.

Earth's lithosphere is divided into approximately 15 major tectonic plates and numerous smaller ones, floating on the partially molten asthenosphere. The Pacific Plate, the largest, moves northwest at about 7-11 centimeters per year. The fastest spreading occurs at the East Pacific Rise, where new crust forms at up to 15 centimeters per year, compared to about 2.5 centimeters at the Mid-Atlantic Ridge. Convergent boundaries, where plates collide, produce the most dramatic geological features: the Himalayas formed when the Indian Plate collided with the Eurasian Plate beginning about 50 million years ago and continues to rise at approximately 5 millimeters per year.

Subduction zones, where oceanic crust dives beneath continental crust, generate the most powerful earthquakes and explosive volcanism on Earth. The 2011 Tohoku earthquake and tsunami in Japan, magnitude 9.1, occurred along the Japan Trench subduction zone and released energy equivalent to approximately 600 million times the Hiroshima bomb. The Cascadia Subduction Zone off the Pacific Northwest coast of North America last ruptured in a magnitude 9.0 earthquake on January 26, 1700, sending a tsunami across the Pacific that was recorded in Japanese historical documents.

Transform boundaries, where plates slide horizontally past each other, produce frequent moderate earthquakes. The San Andreas Fault in California, the world's most famous transform boundary, marks where the Pacific Plate slides northwest past the North American Plate at about 46 millimeters per year. The 1906 San Francisco earthquake, estimated at magnitude 7.9, resulted from a rupture along 477 kilometers of the fault and caused devastating fires that destroyed 80% of the city. The "Big Bend" section of the San Andreas near Los Angeles creates compression that produces the Transverse Ranges, including the San Gabriel Mountains.

Hotspot volcanism occurs independently of plate boundaries when plumes of hot mantle material rise from near the core-mantle boundary, 2,900 kilometers deep. The Hawaiian island chain formed as the Pacific Plate moved northwest over a stationary hotspot; the oldest island, Kauai, is 5.1 million years old, while the Big Island of Hawaii continues to grow from active eruptions. The Yellowstone hotspot has produced three massive caldera-forming eruptions in the past 2.1 million years, the most recent 640,000 years ago, ejecting approximately 1,000 cubic kilometers of material. The hotspot track can be traced from Yellowstone to the McDermitt caldera in Nevada-Oregon, covering 700 kilometers over 16 million years.

The Wilson Cycle describes how ocean basins open and close over periods of 200-500 million years. Currently, the Atlantic Ocean continues to widen while the Pacific shrinks. The Mediterranean Sea, a remnant of the ancient Tethys Ocean, is slowly closing as Africa moves northward toward Europe. In approximately 250 million years, all continents are predicted to merge again into a supercontinent, variously called Pangaea Ultima or Novopangaea, continuing the cycle that has repeated at least five times in Earth's 4.5-billion-year history. The previous supercontinent, Pangaea, existed from roughly 335 to 175 million years ago and included all of today's continents in a single landmass surrounded by the global ocean Panthalassa."""
        },
        2: {
            "title": "History of Chess",
            "content": """Chess, one of the oldest and most enduring strategy games in human history, originated in northern India during the Gupta Empire around the 6th century AD. The original game, known as chaturanga, featured four divisions of the Indian army: infantry (pawns), cavalry (knights), elephants (bishops), and chariots (rooks), with a king and general (later the queen) commanding the forces. The game spread westward through Persia, where it became shatranj, and the phrase "shāh māt" (the king is dead) gave us the English term "checkmate."

The Arab conquest of Persia in the 7th century brought chess to the Islamic world, where it flourished despite some religious authorities considering it gambling. Arab players developed the first formal notation systems and composed the earliest chess problems (mansubat). The legendary player as-Suli (c. 880-946) was considered unbeatable for over 600 years, and his chess problems remain challenging even today. The game reached Europe through multiple routes: the Moorish conquest of Spain, trade routes through Constantinople, and Norse connections through Scandinavia.

The modern rules of chess crystallized in Spain and Italy during the late 15th century. The most significant change was the transformation of the queen from a weak piece (moving only one square diagonally) to the most powerful piece on the board, capable of moving any number of squares in any direction. This change, possibly influenced by the powerful Queen Isabella of Castile, dramatically increased the game's tactical complexity. The bishop also gained its modern long-range diagonal movement, replacing the elephant's limited two-square diagonal jump.

The first official World Chess Championship was held in 1886, with Wilhelm Steinitz defeating Johannes Zukertort. Steinitz revolutionized chess theory by introducing positional play, arguing that advantages could be accumulated through small strategic improvements rather than relying solely on tactical attacks. His approach laid the groundwork for the "classical school" developed by Siegbert Tarrasch, who systematized positional principles into teachable rules about pawn structure, piece activity, and space control.

The Soviet Union's dominance of world chess from 1948 to 1972 represented an unprecedented era in the game's history. Beginning with Mikhail Botvinnik's world championship victory in 1948, Soviet players held the title for 24 consecutive years through Botvinnik, Vasily Smyslov, Mikhail Tal, Tigran Petrosian, and Boris Spassky. The Soviet chess school received massive state support, with chess clubs in every city, professional coaching systems, and significant prizes and social status for top players. This dominance was broken dramatically by Bobby Fischer's victory over Spassky in the 1972 World Championship in Reykjavik, a match that became a Cold War proxy battle watched by millions worldwide.

Garry Kasparov, widely considered the greatest chess player of all time, held the world championship from 1985 to 2000. His rivalry with Anatoly Karpov produced five world championship matches totaling 144 games over three years, the most intense rivalry in chess history. Kasparov's aggressive, dynamic style contrasted sharply with Karpov's precise positional approach, and their matches captivated the global chess community. Kasparov's defeat by IBM's Deep Blue computer in 1997 marked a turning point in the relationship between human intelligence and artificial intelligence.

Modern chess engines like Stockfish and AlphaZero have surpassed all human players by a significant margin. AlphaZero, developed by DeepMind in 2017, learned chess entirely through self-play without any human knowledge beyond the rules, and defeated the world's strongest conventional engine Stockfish in a 100-game match. The rise of engines has transformed how humans study and play chess, with players at all levels using engine analysis to prepare openings, evaluate positions, and improve their understanding. Online platforms like Chess.com and Lichess have made the game accessible to hundreds of millions of players worldwide, sparking a modern chess renaissance amplified by the Netflix series "The Queen's Gambit" in 2020."""
        },
        3: {
            "title": "Honey Bee Colony Biology",
            "content": """The honey bee colony represents one of nature's most sophisticated examples of collective intelligence, where approximately 60,000 individuals function as a superorganism through complex communication, division of labor, and environmental regulation. A single colony contains three castes: one queen, several hundred drones, and tens of thousands of workers, all cooperating through chemical signals, vibrations, and the famous waggle dance to maintain colony homeostasis.

The queen bee, the only fully fertile female in the colony, can lay up to 2,000 eggs per day during peak season, totaling over 200,000 eggs per year. She mates only once in her life during a mating flight at approximately one week of age, storing sperm from 12-20 drones in her spermatheca, which she uses to fertilize eggs for the remainder of her 3-5 year lifespan. Fertilized eggs develop into workers or queens depending on the diet provided by nurse bees: larvae destined to become queens receive exclusive royal jelly throughout development, while worker larvae are switched to a diet of honey and pollen after three days.

The waggle dance, discovered by Karl von Frisch who received the Nobel Prize in 1973, is perhaps the most remarkable communication system in the insect world. When a forager returns from a productive food source, she performs a figure-eight dance on the vertical surface of the comb. The angle of the straight run relative to vertical indicates the direction of the food source relative to the sun, while the duration of the waggle phase encodes the distance: approximately one second of waggling corresponds to one kilometer. The vigor of the dance conveys the quality of the food source, allowing the colony to allocate foragers efficiently to the most productive locations.

Worker bees progress through a series of age-dependent roles during their 6-week summer lifespan. For the first three days, they clean cells. From days 3-10, they serve as nurse bees, feeding larvae with glandular secretions. Between days 10-20, they transition to wax production, comb construction, honey processing, and guard duty. Finally, from approximately day 20 until death, they become foragers, making an average of 10 trips per day to collect nectar, pollen, water, or propolis. A single forager visits 50-100 flowers per trip and may fly 800 kilometers during her lifetime, producing only about one-twelfth of a teaspoon of honey.

Thermoregulation within the hive demonstrates remarkable collective engineering. Workers maintain the brood area at precisely 34-36°C regardless of external temperatures. In summer, they cool the hive by collecting water and fanning it with their wings to create evaporative cooling, positioning themselves in specific ventilation chains that direct airflow through the colony. In winter, bees form a dense cluster that contracts and expands with temperature, with workers on the outer shell vibrating their flight muscles to generate heat. The core temperature of the winter cluster remains near 35°C even when external temperatures drop to -30°C, fueled by honey reserves that may exceed 30 kilograms.

The phenomenon of colony collapse disorder (CCD), first identified in 2006, devastated beekeeping operations worldwide when worker bees inexplicably abandoned their hives. At its peak, US beekeepers reported losing 30-40% of colonies annually. Research has implicated multiple interacting factors: neonicotinoid pesticides that impair navigation and learning, the parasitic mite Varroa destructor that transmits viruses, habitat loss reducing forage availability, and the stress of commercial pollination where colonies are trucked thousands of kilometers between crops. While CCD reports have declined since 2015, annual colony losses remain elevated above historical norms.

Honey bees produce several hive products beyond honey. Beeswax, secreted from glands on workers' abdomens, is used to construct the hexagonal comb that serves as nursery, food storage, and communication medium. Propolis, a resinous mixture collected from tree buds, is used to seal gaps, sterilize surfaces, and embalm intruders too large to remove. Royal jelly, the exclusive diet of queen larvae, contains proteins, sugars, fatty acids, and a unique compound called royalactin that triggers queen development. Bee venom, delivered through the barbed stinger, contains melittin and other compounds being investigated for anti-inflammatory and anti-cancer properties.

The economic value of honey bee pollination dwarfs the value of hive products. In the United States alone, honey bees pollinate approximately 130 crop species worth over 15 billion dollars annually, including almonds (requiring 2 million colonies just for California's almond bloom), blueberries, cherries, apples, and avocados. One-third of the human diet depends directly or indirectly on honey bee pollination. The annual almond pollination in California's Central Valley represents the largest managed pollination event on Earth, with colonies trucked from across the country to service 1.3 million acres of almond orchards during February and March."""
        },
        4: {
            "title": "Ancient Roman Concrete and Engineering",
            "content": """Roman concrete, known as opus caementicium, represents one of the most enduring engineering achievements in human history. While modern Portland cement concrete typically degrades within 50-100 years, Roman marine concrete structures have survived over 2,000 years of exposure to seawater, and recent research has revealed that these structures actually grow stronger over time through a remarkable self-healing mechanism involving mineral crystallization.

The secret of Roman concrete lies in its unique recipe: volcanic ash (pulvis puteolanus) from the region around Pozzuoli near Mount Vesuvius, lime (calcium oxide), seawater, and chunks of volcanic rock (tuff). When mixed, the calcium oxide reacts with seawater to form a calcium-aluminum-silicate-hydrate compound similar to tobermorite, a rare mineral that forms in nature only under extreme heat and pressure. Research published in 2017 by Marie Jackson at the University of Utah demonstrated that this mineral continues to crystallize within the concrete over centuries, growing through the porous volcanic rock aggregate and actually reinforcing the material against cracking.

The Pantheon in Rome, completed around 125 AD under Emperor Hadrian, contains the world's largest unreinforced concrete dome, spanning 43.3 meters — a record that stood for over 1,300 years until Brunelleschi's dome in Florence. The Pantheon's dome achieves structural stability through careful material gradation: dense basalt aggregate in the lower portions transitions to lightweight pumice near the oculus, reducing weight where the dome is most vulnerable. The coffers (recessed panels) in the dome's interior reduce weight by approximately 3,600 kilograms per coffer while adding visual rhythm. The 8.2-meter oculus at the apex eliminates the weakest section of the dome and illuminates the interior with a dramatic beam of sunlight that traverses the walls throughout the day.

Roman aqueducts represent another extraordinary engineering achievement, delivering fresh water across distances of up to 100 kilometers using gravity flow alone. The Pont du Gard in southern France, built around 19 BC, stands 48.8 meters tall with three tiers of arches carrying the water channel across the Gardon River valley. The entire 50-kilometer Nimes aqueduct drops only 17 meters over its length, a gradient of 34 centimeters per kilometer, demonstrating extraordinary surveying precision achieved with simple instruments like the groma and chorobates. At its peak, Rome's 11 aqueducts delivered approximately 1 million cubic meters of water daily to the city's 1 million inhabitants — roughly matching the per-capita water consumption of a modern European city.

Roman road engineering created a transportation network spanning over 400,000 kilometers that connected an empire from Britain to Mesopotamia. Major roads (viae publicae) were typically 4-6 meters wide, with multiple construction layers: a foundation of large stones (statumen), a layer of smaller stones in morite (rudus), a compacted gravel layer (nucleus), and a crowned surface of tightly fitted polygonal paving stones (summa crusta) that directed rainwater to drainage ditches on either side. The Via Appia, begun in 312 BC, still bears vehicular traffic in some sections after over 2,300 years.

The Colosseum, completed in 80 AD, seated 50,000 spectators and could be emptied in 15 minutes through 80 ground-level arched entrances (vomitoria). Its travertine limestone exterior rises four stories (48.5 meters), with each level using a different column order: Doric on the ground floor, Ionic on the second, Corinthian on the third, and Composite pilasters on the fourth. An elaborate canvas awning system (velarium), operated by sailors from the imperial navy using ropes and pulleys, could shade the entire arena. Beneath the arena floor, the hypogeum contained a complex network of tunnels, animal cages, and 80 vertical shafts with counterweight elevators that could rapidly deliver animals and scenery to the arena surface.

Roman heating technology included the hypocaust system, which circulated hot air beneath raised floors and through hollow walls (tubuli) to heat bathhouses and wealthy homes. The Baths of Caracalla (completed 216 AD) served 6,000-8,000 bathers simultaneously across 25 acres, heated by a battery of wood-burning furnaces consuming an estimated 10 tons of wood per day. The complex included hot rooms (caldarium), warm rooms (tepidarium), cold plunge pools (frigidarium), swimming pools, gymnasiums, libraries, and gardens. Roman glass-making allowed large windows that admitted light while retaining heat, and some baths oriented their caldaria to the southwest to capture afternoon solar heating.

The engineering of Roman harbors advanced maritime commerce throughout the Mediterranean. The harbor at Caesarea Maritima, built by Herod the Great around 22 BC using Roman concrete technology, featured massive underwater concrete breakwaters formed using wooden caissons sunk into position and filled with pozzolanic concrete that hardened underwater. The harbor at Portus, Rome's main port built under Emperors Claudius and Trajan, covered over 200 hectares with hexagonal inner basins, warehouses, and a lighthouse modeled on the Pharos of Alexandria. These harbors enabled Rome to import 400,000 tons of grain annually from Egypt and North Africa to feed the capital's population."""
        },
        5: {
            "title": "History of Photography",
            "content": """The invention of photography in the early 19th century fundamentally transformed humanity's relationship with visual reality, creating an entirely new medium for art, science, journalism, and personal memory. The journey from the first permanent photograph to today's ubiquitous digital cameras spans barely two centuries but encompasses one of the most rapid and impactful technological evolutions in human history.

The earliest known photograph, created by Joseph Nicéphore Niépce in 1826 or 1827, required approximately eight hours of exposure using a process he called heliography. His "View from the Window at Le Gras" captured the courtyard of his estate in Burgundy on a pewter plate coated with bitumen of Judea, a naturally occurring asphalt that hardened when exposed to light. After exposure, unhardened bitumen was washed away with lavender oil, revealing the image. This single photograph, now housed at the University of Texas at Austin, represents the birth of a medium that would reshape human civilization.

Louis-Jacques-Mandé Daguerre, who had been collaborating with Niépce before the latter's death in 1833, perfected the daguerreotype process and presented it to the French Academy of Sciences on January 7, 1839. The daguerreotype used a silver-plated copper sheet sensitized with iodine vapor, exposed in a camera for 15-30 minutes, and developed over heated mercury. The resulting image was startlingly detailed and appeared to float on the mirror-like silver surface. The French government purchased the rights and generously made the process "a gift to the world," sparking an international frenzy of daguerreotyping. Within months, portrait studios opened in every major city.

William Henry Fox Talbot's calotype process, patented in 1841, introduced the negative-positive system that would dominate photography for 150 years. Unlike the daguerreotype's unique image, the calotype produced a paper negative from which multiple positive prints could be made. Though lower in resolution than daguerreotypes, the calotype's reproducibility proved commercially superior. Talbot also created what many consider the first photographically illustrated book, "The Pencil of Nature" (1844-46), which demonstrated photography's potential for documenting art, architecture, and everyday objects.

The wet collodion process, introduced by Frederick Scott Archer in 1851, combined the detail of daguerreotypes with the reproducibility of calotypes. Glass plates coated with iodized collodion and sensitized in silver nitrate could capture remarkable detail with exposure times of just 2-20 seconds. The catch was that plates had to be prepared, exposed, and developed while still wet, requiring photographers to carry portable darkrooms. Despite this inconvenience, the wet plate process dominated photography from the 1850s through the 1870s, producing iconic images of the American Civil War by photographers like Mathew Brady and Alexander Gardner.

George Eastman revolutionized photography in 1888 by introducing the Kodak camera, a simple box loaded with a 100-exposure roll of flexible film. His famous slogan "You press the button, we do the rest" democratized photography by eliminating the need for technical knowledge. After shooting the roll, the entire camera was mailed to Kodak's factory in Rochester, New York, where the film was developed, prints were made, and the camera was reloaded and returned. The Brownie camera, introduced in 1900 at just one dollar, brought photography within reach of virtually everyone and created the concept of the snapshot.

Color photography developed slowly over many decades. The first color photograph was created by James Clerk Maxwell in 1861 using three separate exposures through red, green, and blue filters. The Autochrome process, marketed by the Lumière brothers beginning in 1907, used microscopic grains of potato starch dyed in three colors as color filters on a glass plate, producing dreamy, pointillist-like color images. Kodachrome, introduced in 1935, became the premier color transparency film for professionals and serious amateurs, renowned for its rich, saturated colors and exceptional archival stability. Kodachrome remained in production until 2009, with the last roll developed on December 30, 2010 at Dwayne's Photo in Parsons, Kansas.

The digital revolution in photography began with the charge-coupled device (CCD), invented at Bell Labs by Willard Boyle and George Smith in 1969, for which they received the Nobel Prize in Physics in 2009. The first digital camera, built by Kodak engineer Steven Sasson in 1975, weighed 3.6 kilograms and captured black-and-white images at 0.01 megapixels, storing them on a cassette tape with a 23-second recording time. Commercial digital cameras appeared in the 1990s, but it was the integration of cameras into smartphones — beginning with the Sharp J-SH04 in 2000 and reaching mass adoption with the iPhone in 2007 — that truly transformed photography into a universal daily activity. Today, an estimated 1.81 trillion photographs are taken annually, more images in a single year than in the entire previous history of photography combined."""
        },
    }
    t = topics[topic_num]
    text = t["content"]
    # Pad with elaboration if needed
    while len(text) < target_chars:
        text += "\n\n" + text  # double it
    return text[:target_chars]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="sparse", choices=["dense", "sparse", "dense_prefill"])
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    if args.mode == "dense":
        backend = "ESIMD_ATTN"
    elif args.mode == "sparse":
        backend = "INFLLMV2_ESIMD_ATTN"
    elif args.mode == "dense_prefill":
        backend = "INFLLMV2_ESIMD_ATTN"
        os.environ["INFLLMV2_SPARSE_PREFILL"] = "0"

    # Try fetching from Wikipedia, fall back to built-in content
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
    target_per_topic = 24000  # ~6K tokens each, total ~30K

    print(f"Fetching content for 5 topics...")
    sections = []
    for i, (wiki, name) in enumerate(zip(wiki_titles, section_names)):
        text = fetch_wiki(wiki, target_per_topic)
        if not text or len(text) < 5000:
            print(f"  Using fallback for: {name}")
            text = generate_fallback(i + 1, name, target_per_topic)
        else:
            print(f"  Fetched {len(text)} chars for: {name}")
        sections.append(f"\n<<Section {i+1}: {name}>>\n\n{text}")

    question = """

Now, please answer the following based on ALL 5 sections above:
1. Section 1 (Plate Tectonics): Name 2 specific examples of geological events or features.
2. Section 2 (Chess): Name 2 specific chess players and their contributions.
3. Section 3 (Honey Bees): Describe 2 specific bee behaviors or biological facts with numbers.
4. Section 4 (Roman Engineering): Name 2 specific Roman structures and their dimensions.
5. Section 5 (Photography): Name 2 specific photographers or inventions with dates.
Keep each answer to 2-3 sentences."""

    prompt = "Read the following 5 sections carefully. You will be asked questions about ALL of them.\n" + "".join(sections) + question

    print(f"\nMode: {args.mode}, backend: {backend}")
    print(f"Prompt: {len(prompt)} chars")

    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL_PATH, dtype="bfloat16", trust_remote_code=True,
              enforce_eager=True, disable_log_stats=True, gpu_memory_utilization=0.90,
              max_model_len=32768, max_num_seqs=2, block_size=128,
              attention_backend=backend)
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    outputs = llm.generate([prompt], params)
    out = outputs[0]
    pt = len(out.prompt_token_ids)
    gt = len(out.outputs[0].token_ids)
    text = out.outputs[0].text

    print(f"Prompt tokens: {pt}")
    print(f"Generated tokens: {gt}")
    print(f"\n{'='*70}")
    print(f"  OUTPUT ({args.mode})")
    print(f"{'='*70}")
    print(text)
    print(f"{'='*70}")

    tl = text.lower()
    checks = {
        "Plate Tectonics": any(w in tl for w in ["tectonic", "wegener", "pangaea", "subduction", "san andreas", "earthquake", "himalaya", "mid-atlantic", "rift"]),
        "Chess":           any(w in tl for w in ["chess", "kasparov", "fischer", "steinitz", "deep blue", "checkmate", "shor", "botvinnik"]),
        "Honey Bees":      any(w in tl for w in ["bee", "waggle", "honey", "queen", "colony", "pollen", "hive", "forager"]),
        "Roman Eng":       any(w in tl for w in ["roman", "pantheon", "concrete", "aqueduct", "colosseum", "pont du gard", "via appia"]),
        "Photography":     any(w in tl for w in ["photograph", "daguerr", "kodak", "camera", "niépce", "niepce", "calotype", "eastman"]),
    }
    print(f"\nQuality — topics discovered:")
    for k, v in checks.items():
        print(f"  {k}: {'YES' if v else 'NO'}")
    print(f"  Score: {sum(checks.values())}/5 topics covered")
    del llm
