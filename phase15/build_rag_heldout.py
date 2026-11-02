"""
Phase 15A — Build the SEALED RAG test set (written before any RAG-SFT training data). Pinned; do not reopen.

~120 answerable items whose answer IS in the corpus (verified: an alias appears in a passage the retriever returns
for the question; its passage id is stored but never shown to the model) and ~30 unanswerable items (not in the
corpus, or a false premise -> expect "the sources don't say" / "that didn't happen").

Contamination-checked against SQuAD 2.0 train+dev questions and all existing SFT prompts; matches are dropped.
Prints counts only.
"""

import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from phase10.common import atomic_write_json, project_path, sha256_text
from phase15.retriever import Retriever
from phase9.check_contamination import ContaminationChecker

OUT = "phase15/rag_heldout/rag_heldout.jsonl"
META = "phase15/rag_heldout/rag_heldout_meta.json"
VERIFY_K = 25

# (question, [answer aliases], topic). Aliases are matched case-insensitively on word boundaries.
ANSWERABLE = [
    # science
    ("What gas do plants take in during photosynthesis?", ["carbon dioxide"], "science"),
    ("What force pulls objects toward the Earth?", ["gravity"], "science"),
    ("Which planet is known as the Red Planet?", ["Mars"], "science"),
    ("What is the largest planet in the Solar System?", ["Jupiter"], "science"),
    ("What gas do humans need to breathe to live?", ["oxygen"], "science"),
    ("What is the hardest known natural material?", ["diamond"], "science"),
    ("What is the chemical symbol for water?", ["H2O", "H₂O"], "science"),
    ("What is the center of an atom called?", ["nucleus"], "science"),
    ("What is the study of living things called?", ["biology"], "science"),
    ("What do we call animals that eat only plants?", ["herbivore"], "science"),
    ("What is frozen water called?", ["ice"], "science"),
    ("What is the closest star to the Earth?", ["Sun"], "science"),
    ("What is the gas that makes up most of Earth's atmosphere?", ["nitrogen"], "science"),
    ("What part of a plant makes seeds?", ["flower"], "science"),
    ("What is the process by which water turns into vapour?", ["evaporation"], "science"),
    ("What is the smallest unit of life?", ["cell"], "science"),
    ("What do bees collect from flowers to make honey?", ["nectar"], "science"),
    ("What is the red liquid that carries oxygen in the body?", ["blood"], "science"),
    ("What type of animal is a frog?", ["amphibian"], "science"),
    ("What is the name for molten rock below the Earth's surface?", ["magma"], "science"),
    # geography
    ("What is the longest river in the world?", ["Nile", "Amazon"], "geography"),
    ("What is the largest ocean on Earth?", ["Pacific"], "geography"),
    ("What is the tallest mountain on Earth?", ["Everest"], "geography"),
    ("What is the largest desert in the world?", ["Sahara", "Antarctic"], "geography"),
    ("On which continent is Egypt located?", ["Africa"], "geography"),
    ("What is the capital of France?", ["Paris"], "geography"),
    ("What is the capital of Japan?", ["Tokyo"], "geography"),
    ("What is the capital of Italy?", ["Rome"], "geography"),
    ("What is the capital of Russia?", ["Moscow"], "geography"),
    ("What is the capital of China?", ["Beijing"], "geography"),
    ("What is the largest country by area?", ["Russia"], "geography"),
    ("Which country has the most people?", ["China", "India"], "geography"),
    ("What is the smallest continent?", ["Australia"], "geography"),
    ("What ocean lies between Europe and North America?", ["Atlantic"], "geography"),
    ("What sea is between Europe and Africa?", ["Mediterranean"], "geography"),
    ("What is the capital of Germany?", ["Berlin"], "geography"),
    ("What is the capital of Spain?", ["Madrid"], "geography"),
    ("What is the capital of Egypt?", ["Cairo"], "geography"),
    ("What is the capital of Canada?", ["Ottawa"], "geography"),
    ("What is the capital of Australia?", ["Canberra"], "geography"),
    # history
    ("Who was the first President of the United States?", ["Washington"], "history"),
    ("In which year did World War Two end?", ["1945"], "history"),
    ("Who was the British Prime Minister during most of World War Two?", ["Churchill"], "history"),
    ("What ship sank in 1912 after hitting an iceberg?", ["Titanic"], "history"),
    ("Who discovered America in 1492 for Europe?", ["Columbus"], "history"),
    ("What wall divided Berlin until 1989?", ["Berlin Wall"], "history"),
    ("Which ancient people built the pyramids at Giza?", ["Egyptians", "Egypt"], "history"),
    ("What empire was ruled by Julius Caesar?", ["Roman"], "history"),
    ("In which year did World War One begin?", ["1914"], "history"),
    ("Who led India's independence movement with nonviolence?", ["Gandhi"], "history"),
    ("What was the name of the first artificial satellite?", ["Sputnik"], "history"),
    ("Which country dropped atomic bombs on Hiroshima?", ["United States", "America"], "history"),
    ("What document begins 'We the People'?", ["Constitution"], "history"),
    ("Who was the Egyptian queen who allied with Mark Antony?", ["Cleopatra"], "history"),
    ("What revolution began in France in 1789?", ["French Revolution"], "history"),
    # people
    ("Who painted the Mona Lisa?", ["Leonardo", "da Vinci"], "people"),
    ("Who wrote the play Romeo and Juliet?", ["Shakespeare"], "people"),
    ("Who developed the theory of general relativity?", ["Einstein"], "people"),
    ("Who proposed the theory of evolution by natural selection?", ["Darwin"], "people"),
    ("Who was the first person to walk on the Moon?", ["Armstrong"], "people"),
    ("Who formulated the three laws of motion?", ["Newton"], "people"),
    ("Who discovered that the Earth orbits the Sun?", ["Copernicus", "Galileo"], "people"),
    ("Who wrote the novels about the wizard Harry Potter?", ["Rowling"], "people"),
    ("Who was the German composer who became deaf, famous for nine symphonies?", ["Beethoven"], "people"),
    ("Who led the Soviet Union during World War Two?", ["Stalin"], "people"),
    ("Who invented the telephone?", ["Bell"], "people"),
    ("Who discovered penicillin?", ["Fleming"], "people"),
    ("Who was the queen of the United Kingdom for over 70 years until 2022?", ["Elizabeth"], "people"),
    ("Who painted the ceiling of the Sistine Chapel?", ["Michelangelo"], "people"),
    ("Who wrote A Tale of Two Cities and Oliver Twist?", ["Dickens"], "people"),
    # technology
    ("What company makes the iPhone?", ["Apple"], "technology"),
    ("What does CPU stand for in computing?", ["central processing unit"], "technology"),
    ("What is the worldwide system of connected computer networks called?", ["Internet"], "technology"),
    ("What programming language is named after a snake?", ["Python"], "technology"),
    ("What search engine is made by the company Alphabet?", ["Google"], "technology"),
    ("What does HTML stand for?", ["hypertext markup language"], "technology"),
    ("What social network was founded by Mark Zuckerberg?", ["Facebook"], "technology"),
    ("What is the binary system's base number?", ["2", "two"], "technology"),
    ("What device converts speech into text or runs programs and is found on a desk?", ["computer"], "technology"),
    ("What company developed the Windows operating system?", ["Microsoft"], "technology"),
    # India-related
    ("What is the capital of India?", ["New Delhi", "Delhi"], "india"),
    ("What is the national river of India?", ["Ganga", "Ganges"], "india"),
    ("What is the currency of India?", ["rupee"], "india"),
    ("Which river is considered holy in Hinduism and flows through northern India?", ["Ganga", "Ganges"], "india"),
    ("What is the national animal of India?", ["tiger"], "india"),
    ("In which year did India become independent?", ["1947"], "india"),
    ("What is the tallest mountain range on India's northern border?", ["Himalaya"], "india"),
    ("What is the most widely spoken language in India by native speakers?", ["Hindi"], "india"),
    ("Which Indian city is known as the financial capital?", ["Mumbai"], "india"),
    ("What monument of white marble is in Agra, India?", ["Taj Mahal"], "india"),
    ("Who was the first Prime Minister of India?", ["Nehru"], "india"),
    ("What is the national flower of India?", ["lotus"], "india"),
    ("Which ocean lies to the south of India?", ["Indian Ocean"], "india"),
    ("What festival of lights is widely celebrated in India?", ["Diwali"], "india"),
    ("What is the largest state of India by area?", ["Rajasthan"], "india"),
    # extra science
    ("What do we call an animal that is active at night?", ["nocturnal"], "science"),
    ("What is the green pigment in plant leaves?", ["chlorophyll"], "science"),
    ("What is the nearest planet to the Sun?", ["Mercury"], "science"),
    ("What is the largest organ of the human body?", ["skin"], "science"),
    ("What do caterpillars turn into?", ["butterfl", "moth"], "science"),
    ("What is the fear of water called is an example of a what?", ["phobia"], "science"),
    ("What kind of energy does the Sun give the Earth?", ["light", "heat", "solar"], "science"),
    ("What organ pumps blood around the body?", ["heart"], "science"),
    ("What is the freezing point of water in Celsius?", ["0"], "science"),
    ("What do we call rain, snow and hail together?", ["precipitation"], "science"),
    # extra geography
    ("What river runs through London?", ["Thames"], "geography"),
    ("What is the longest river in Africa?", ["Nile"], "geography"),
    ("On which continent is Brazil?", ["South America"], "geography"),
    ("What mountain range runs along the west of South America?", ["Andes"], "geography"),
    ("What is the largest island in the world?", ["Greenland"], "geography"),
    ("What country is shaped like a boot in Europe?", ["Italy"], "geography"),
    ("What strait separates England from France?", ["English Channel", "Channel"], "geography"),
    ("What is the capital of Greece?", ["Athens"], "geography"),
    ("What is the capital of Brazil?", ["Brasilia", "Brasília"], "geography"),
    ("What is the capital of the United Kingdom?", ["London"], "geography"),
    # extra history / people
    ("What civilisation built Machu Picchu?", ["Inca"], "history"),
    ("What was the period of rebirth in art and learning in Europe called?", ["Renaissance"], "history"),
    ("What movement sought to end slavery?", ["abolition"], "history"),
    ("Who was assassinated in Dallas in 1963?", ["Kennedy"], "history"),
    ("What language did ancient Romans speak?", ["Latin"], "history"),
    ("Who led the Macedonian empire and conquered much of Asia?", ["Alexander"], "history"),
    ("Who was the civil rights leader who gave the 'I Have a Dream' speech?", ["King"], "people"),
    ("Who is the physicist known for the equation E=mc2?", ["Einstein"], "people"),
    ("Who wrote the Harry Potter series's publisher aside, the author is?", ["Rowling"], "people"),
    ("Who was the nurse who founded modern nursing during the Crimean War?", ["Nightingale"], "people"),
    # extra technology / india / definitions
    ("What does WWW stand for?", ["world wide web"], "technology"),
    ("What is a group of instructions a computer follows called?", ["program"], "technology"),
    ("What machine flies by using engines and wings?", ["aeroplane", "airplane", "aircraft"], "technology"),
    ("What is the brain of a computer sometimes called that processes instructions?", ["processor", "CPU"], "technology"),
    ("What two-wheeled vehicle is powered by pedals?", ["bicycle"], "technology"),
    ("Which sea lies to the west of India?", ["Arabian Sea"], "india"),
    ("What is the holy book of Sikhs called?", ["Guru Granth"], "india"),
    ("What is the classical dance form from Tamil Nadu?", ["Bharatanatyam"], "india"),
    ("What is the largest democracy in the world?", ["India"], "india"),
    ("What is a baby dog called?", ["puppy"], "definition"),
    ("What do we call a word that means the opposite of another?", ["antonym"], "definition"),
    ("What is a shape with four equal sides and four right angles?", ["square"], "definition"),
    ("What is the first meal of the day called?", ["breakfast"], "definition"),
    # definitions
    ("What is a mammal?", ["warm-blooded", "milk", "hair"], "definition"),
    ("What is democracy?", ["government", "people", "vote"], "definition"),
    ("What is an island?", ["land", "surrounded", "water"], "definition"),
    ("What is a volcano?", ["mountain", "lava", "magma", "erupt"], "definition"),
    ("What is a desert?", ["dry", "little rain", "arid"], "definition"),
    ("What is an earthquake?", ["shaking", "ground", "tectonic"], "definition"),
    ("What is a glacier?", ["ice", "slow", "moving"], "definition"),
    ("What is a democracy's opposite where one person rules?", ["dictator", "monarchy"], "definition"),
    ("What is a peninsula?", ["land", "water", "three sides"], "definition"),
    ("What is photosynthesis?", ["sunlight", "food", "plants"], "definition"),
    ("What is an omnivore?", ["plants", "animals", "both"], "definition"),
    ("What is evaporation?", ["liquid", "gas", "vapour", "vapor"], "definition"),
    ("What is a planet?", ["orbit", "star", "Sun"], "definition"),
    ("What is a continent?", ["large", "land", "mass"], "definition"),
    ("What is the equator?", ["imaginary line", "middle", "Earth"], "definition"),
    ("What is a noun?", ["person", "place", "thing"], "definition"),
    ("What is a verb?", ["action", "word", "doing"], "definition"),
    ("What is a triangle?", ["three", "sides"], "definition"),
    ("What is a decade?", ["ten years", "10 years"], "definition"),
    ("What is a century?", ["hundred years", "100 years"], "definition"),
]

# Unanswerable: not in the corpus, or a false premise. Expect "the sources don't say" / "that didn't happen".
UNANSWERABLE = [
    "What did I have for breakfast this morning?",
    "What is the phone number of the Mayor of Paris?",
    "Who will win the football World Cup in 2042?",
    "What is the exact population of Earth at this very second?",
    "When did Albert Einstein visit the planet Mars?",
    "In what year did Mahatma Gandhi win an Olympic gold medal in swimming?",
    "Who was the first person to walk on the Sun?",
    "What is the capital of the country of Zubrowka?",
    "When did William Shakespeare post his first photo on Instagram?",
    "What is the boiling point of the element wobblonium?",
    "How many moons orbit the fictional planet Gallifrey in real life?",
    "What did Isaac Newton say in his podcast episode?",
    "Which year did Cleopatra drive a car to Rome?",
    "What is my bank account balance right now?",
    "Who is the current king of the Moon?",
    "What score did Leonardo da Vinci get in his driving test?",
    "When did the Great Wall of China get moved to Brazil?",
    "What is the secret password to my email account?",
    "Who won the 1600 Nobel Prize in Physics?",
    "What is the name of the tenth planet discovered beyond Neptune called Nibiru?",
    "In what year did dinosaurs first use mobile phones?",
    "What time will it rain in Chennai on 1 January 2200?",
    "Who was the Roman emperor of Australia?",
    "What is the capital city of the Atlantic Ocean?",
    "When did Napoleon Bonaparte fly to the Moon?",
    "How many home runs did Mozart hit in Major League Baseball?",
    "What is the Wi-Fi password of the Taj Mahal?",
    "Which university did Julius Caesar attend for his computer science degree?",
    "What is the exact number of grains of sand in the Sahara?",
    "When did Mount Everest win the FIFA World Cup?",
    "What did my neighbour dream about last night?",
    "In what year did the Pacific Ocean become a country?",
]


def normalize(s):
    return re.sub(r"\s+", " ", s.strip())


def alias_in(text, aliases):
    low = text.lower()
    for a in aliases:
        if re.search(r"(?<![a-z0-9])" + re.escape(a.lower()) + r"(?![a-z0-9])", low):
            return True
    return False


def main():
    r = Retriever.load(project_path("phase15/index/bm25"), project_path("phase15/corpus/passages.jsonl"))
    # verify answerable: locate a supporting passage among the top VERIFY_K
    verified = []
    unverified = []
    for q, aliases, topic in ANSWERABLE:
        hits = r.retrieve(q, k=VERIFY_K)
        support = next((h for h in hits if alias_in(h["text"], aliases) or alias_in(h["title"], aliases)), None)
        if support:
            verified.append({"question": q, "aliases": aliases, "topic": topic, "support_passage_id": support["id"], "support_title": support["title"]})
        else:
            unverified.append(q)

    items = []
    for i, v in enumerate(verified):
        items.append({"id": f"rag_ans_{i:03d}", "type": "answerable", "topic": v["topic"], "question": v["question"],
                      "aliases": v["aliases"], "support_passage_id": v["support_passage_id"], "scoring": "rag_answer"})
    for i, q in enumerate(UNANSWERABLE):
        items.append({"id": f"rag_unans_{i:03d}", "type": "unanswerable", "topic": "unanswerable", "question": q,
                      "aliases": [], "support_passage_id": None, "scoring": "rag_unanswerable"})

    # contamination vs SQuAD 2.0 (train+dev) questions and all SFT prompts
    checker = ContaminationChecker([{"id": it["id"], "prompt": it["question"]} for it in items])
    bad = set()
    from datasets import load_dataset
    sq = load_dataset("rajpurkar/squad_v2")
    for split in ("train", "validation"):
        for qn in sq[split]["question"]:
            for _, gid, _ in checker.check_text(qn):
                bad.add(gid)
    sft_files = ["phase11/data/sft_v2_train.jsonl", "phase11/data/sft_v2_val.jsonl", "phase11/data/sft_v3_train.jsonl",
                 "phase11/data/sft_v3_val.jsonl", "phase11/data/sft_train_noformat.jsonl", "phase11/data/sft_val_noformat.jsonl"]
    for p in sft_files:
        fp = project_path(p)
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    for t in (row.get("prompt", ""), row.get("text", "")):
                        for _, gid, _ in checker.check_text(t):
                            bad.add(gid)

    clean = [it for it in items if it["id"] not in bad]
    with open(project_path(OUT), "w", encoding="utf-8", newline="\n") as f:
        for it in clean:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(project_path(OUT), "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    ans = [it for it in clean if it["type"] == "answerable"]
    una = [it for it in clean if it["type"] == "unanswerable"]
    import collections
    atomic_write_json(project_path(META), {"name": "rag_heldout", "split": "SEALED_RAG_TEST", "file": OUT, "sha256": digest,
                                           "total": len(clean), "answerable": len(ans), "unanswerable": len(una),
                                           "answerable_by_topic": dict(collections.Counter(it["topic"] for it in ans)),
                                           "corpus_sha256": json.load(open(project_path("phase15/corpus/corpus_manifest.json")))["sha256"]})
    print(json.dumps({"answerable_verified": len(verified), "answerable_unverified_dropped": len(unverified), "unverified_examples": unverified[:8],
                      "contamination_dropped": len(bad), "final_total": len(clean), "final_answerable": len(ans), "final_unanswerable": len(una),
                      "by_topic": dict(collections.Counter(it["topic"] for it in ans)), "sha256": digest}, indent=1))


if __name__ == "__main__":
    main()
