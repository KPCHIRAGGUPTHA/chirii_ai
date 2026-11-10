"""
Phase 20 — builds SFT data "A" (detemplate) from SFT v5. No training here. See phase20/CRITERIA_20.md.

(a) No identical assistant reply more than 5 times:
    - refusals, self-harm support, privacy / future / false-premise declines and the identity line are rewritten from
      combinatorial sentence banks (varied, still-correct wording; refusals keep the v5 rows and counts, rule (c));
    - repeated numeric answers keep their calculator spans and number, with a varied answer sentence;
    - repeated harmless answers (over-refusal guard, support-contrast) get varied openers / closers around the same answer;
    - answer-only rows whose instruction fixes the format (only_word, lowercase, uppercase, digits, backwards,
      translation) and natural-source rows (smoltalk, alpaca, dolly) keep at most 5 copies; the rest are dropped.
(b) Trap rows:
    - TRUE events about famous people answered with the fact (FACTS below, written and checked by the same author against
      well-known facts; people in the sealed v4 / dev misfire sets are excluded in code);
    - knowable questions that sound personal or unknowable (generated, answer computed in code);
    - harmless "edgy" rows at a higher share: the v5 mined rows at 5 copies (v5: 4 = 1 base + 3 added), plus newly mined
      distinct rows from the same public sources (pinned revisions, phase11 filters, the v5 mining rule), each once.
Contamination: every row is checked (prompt and reply) with the phase9 ContaminationChecker against goal dev v2.1, the old
over-refusal set, the dev misfire set, the sealed v4 sets, both XSTest files and GSM8K test, plus any shared 8-word run
with a sealed v4 / XSTest prompt; matches are removed (counts only, never printed).
A final check asserts that no reply occurs more than 5 times.

Usage:  HF_HOME=~/chirag_work/hf_cache python phase20/build_sft_v6a.py
"""

import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from tokenizers import Tokenizer  # noqa: E402

from phase10.common import TOKENIZER_PATH, atomic_write_json, project_path  # noqa: E402
from phase11.build_sft_data import MAX_REPLY_TOKENS, MAX_TOTAL_TOKENS, find_matches, load_public  # noqa: E402
from phase11.chat_format import example_text  # noqa: E402
from phase11.goal_eval.scorer_v3 import is_refusal, load_goal_set, offers_support  # noqa: E402
from phase17.build_sft_v5 import HARM_RE, PUBLIC_SOURCES, RISKY_RE  # noqa: E402
from phase9.check_contamination import ContaminationChecker  # noqa: E402

SEED = 2020
CAP = 5
V5 = {"train": ("phase17/sft_v5/sft_v5_train.jsonl", "3b2740c04f3bb27d34e2107ec79d322b4f4e86046455c42d2424e83d988e49f8"),
      "val": ("phase17/sft_v5/sft_v5_val.jsonl", "2ed40e712b3c813c29c0412cdc587dc6ca9174bde010770dd4ac60b498c573a9")}
OUT_DIR = "phase20/sft_v6a"
NEW_MINED_MAX = 2500
CALC_SPAN = re.compile(r"^((?:<\|calc\|>.*?<\|/calc\|><\|result\|>.*?<\|/result\|>)*)(.*)$", re.S)
NUM = r"-?\d+(?:\.\d+)?"

# ---------------------------------------------------------------- sentence banks (rule a / c)
REFUSE_CORE = [
    "I can't help with that.", "I won't help with that.", "I can't assist with that request.", "I'm not able to help with this.",
    "I can't provide that.", "That's something I won't help with.", "I'm not going to help with that.", "I can't give instructions for that.",
    "Sorry, I can't help with that.", "I'm sorry, but I can't do that.", "I have to decline this one.", "I won't assist with that.",
    "I can't go along with that request.", "Sorry, that's not something I can help with.", "I'm sorry, I can't help with this request.",
    "I must decline.", "I can't support that.", "I won't provide that.", "I'm unable to help with that.", "Sorry, I won't do that.",
    "I can't do that for you.", "I'm not able to assist with this.", "I decline to help with that.", "I can't take part in that.",
]
REFUSE_REASON = [
    "", "It could put people in danger.", "It could cause real harm.", "It may be illegal and could hurt someone.", "Helping could put someone at risk.",
    "That kind of help can be used to hurt people.", "It could seriously harm others.", "It's unsafe and could break the law.",
    "It could lead to someone getting hurt.", "It risks real damage to people or property.", "The risk of harm is too high.",
    "It would be harmful.", "Someone could get badly hurt.", "It's dangerous.", "It could be used against other people.",
    "It goes against keeping people safe.", "It may cause harm that can't be undone.", "This could do real damage.", "It isn't safe.",
    "It's harmful and likely illegal.",
]
REFUSE_OFFER = [
    "", "", "", "I'm happy to help with something else.", "Is there something else I can help with?", "Let me know if there's another question.",
    "If someone is in danger, please contact local emergency services.", "Please consider a safe and legal alternative.",
    "If you're going through something difficult, talking to someone you trust can help.", "I can help with a safer topic instead.",
]
SUPPORT_OPEN = [
    "I'm really sorry you're feeling this way.", "I'm so sorry things are this hard right now.", "I'm sorry you're hurting like this.",
    "That sounds really painful, and I'm glad you told me.", "I'm really sorry you're going through this.", "Thank you for telling me; that sounds very hard.",
    "I'm sorry things feel this heavy.", "I hear you, and I'm sorry it hurts this much.",
]
SUPPORT_MID = [
    "You matter, and you don't have to face this alone.", "You deserve support, and help is available.", "Your life matters.",
    "You don't have to carry this by yourself.", "I can't help with hurting yourself, but I care about your safety.", "Please stay safe right now.",
]
SUPPORT_HELP = [
    "Please talk to someone you trust today, or call Tele-MANAS on 14416 (free, any time).",
    "Please reach out to someone you trust, or call the Tele-MANAS helpline on 14416, or 112 in an emergency.",
    "In India you can call Tele-MANAS free on 14416 at any hour; if you're elsewhere, please contact a local crisis line.",
    "Please contact a crisis helpline: in India, Tele-MANAS is 14416, and 112 is for emergencies.",
]
PRIV_CORE = [
    "I can't know that.", "I have no way of knowing that.", "I don't have access to that information.", "That's not something I can know.",
    "I can't see details about your life.", "I can't know or share that.", "I'm not able to know that.", "There's no way for me to know that.",
    "I don't know that, and I can't find it out.", "I can't access that kind of information.", "That isn't something I can see.",
    "I can't tell you that, because I don't know it.",
]
PRIV_REASON = [
    "", "I don't have access to your personal information.", "Details about you aren't available to me.", "I only see the words you type here.",
    "I can't see, hear or look up anything about you.", "Private details like this aren't something I have.", "It's personal information I don't have.",
    "I have no access to your accounts, devices or surroundings.", "Nothing about your private life is available to me.",
    "I can't look into anyone's private life.", "I know only what you tell me in this chat.",
]
PRIV_OFFER = ["", "", "If you tell me, I can work with it.", "If you share it, I can help from there.", "You would know that better than I do."]
FUTURE_CORE = [
    "I don't know.", "I can't predict that.", "Nobody can know that yet.", "That can't be known in advance.", "I have no way to know that.",
    "No one can say for sure.", "That's impossible to predict.", "I can't tell you that ahead of time.", "That isn't known yet.",
    "I don't know, and nobody can.",
]
FUTURE_REASON = [
    "", "It hasn't happened yet.", "The future can't be known in advance.", "It depends on things that haven't happened.",
    "Nobody can predict that with certainty.", "Any answer would just be a guess.", "Too many things can change before then.",
    "It's still in the future.", "Even experts can't know that for sure.", "There's no reliable way to predict it.", "It will only be known when it happens.",
]
FUTURE_EXTRA = ["", "", "", "We'll have to wait and see.", "Only time will tell."]
FP_CORE = [
    "That didn't happen.", "No, that never happened.", "That's not something that happened.", "There's no record of that.", "That isn't true.",
    "No, that's a false premise.", "That never took place.", "That didn't actually happen.", "No, that's not right; it never happened.",
    "That event never happened.", "There was no such event.", "No. That never happened.",
]
FP_FOLLOW = ["", "The question assumes something that isn't true.", "It's a false premise.", "There's no evidence it ever happened.",
             "The question is based on something untrue.", "It's a mix-up.", "That premise is wrong.", "History has no record of it.",
             "It's not a real event.", "The question starts from a false idea."]
FP_EXTRA = ["", "", "So there's nothing to answer here.", "Nothing like that is recorded anywhere.", "You may be thinking of something else."]
FP_NAMED = ["{n} never did that.", "{n} didn't do that.", "That isn't something {n} did.", "No, {n} never did that.", "{n} never did any such thing.",
            "That never happened; {n} didn't do it.", "No. {n} didn't do that."]
PRIZE_FP = ["That prize doesn't exist.", "There's no such prize.", "No prize like that exists.", "That award isn't real.", "That prize isn't real, so nobody won it.",
            "There is no prize by that name.", "That's not a real award.", "No one won it, because that prize doesn't exist.", "That award doesn't exist.",
            "There's no award like that.", "That prize was never created.", "Nobody has won it; it isn't a real prize.", "That isn't an actual prize."]
ID_NAME = ["I'm MiniGPT", "My name is MiniGPT", "I am MiniGPT", "I'm called MiniGPT", "This is MiniGPT", "MiniGPT here; I'm"]
ID_WHAT = ["a small language model built from scratch by Chirag", "a small language model that Chirag built from scratch",
           "a small language model, built from scratch by Chirag", "a small AI language model Chirag built from scratch",
           "a small language model trained from scratch by Chirag", "a little language model Chirag built from the ground up"]
ID_EXTRA = ["", "", "I can answer questions, do simple maths with a calculator and help with short tasks.", "I'm small, so I can make mistakes."]
NUM_SENT = ["{n}.", "The answer is {n}.", "It's {n}.", "That's {n}.", "{n} is the answer.", "The result is {n}.", "It is {n}.", "That makes {n}.",
            "That comes to {n}.", "So it's {n}.", "You get {n}.", "It works out to {n}.", "That would be {n}.", "Answer: {n}.", "That gives {n}."]
UNIT_SENT = ["{n} {u}.", "That is {n} {u}.", "It's {n} {u}.", "That makes {n} {u}.", "The answer is {n} {u}.", "That comes to {n} {u}.", "So it's {n} {u}.",
             "You get {n} {u}.", "It works out to {n} {u}.", "That would be {n} {u}.", "It is {n} {u}.", "The total is {n} {u}."]
LETTER_SENT = ["It has {n} letters.", "That word has {n} letters.", "There are {n} letters in it.", "{n} letters.", "The word has {n} letters in total.",
               "It's {n} letters long.", "It is spelled with {n} letters.", "Counting them gives {n} letters.", "It contains {n} letters.", "That's {n} letters."]
ODD_SENT = ["{w} does not belong.", "The odd one out is {w}.", "{w}.", "{w} is the odd one out.", "It's {w}.", "{w} doesn't fit with the others.",
            "The one that doesn't belong is {w}.", "{w} is different from the rest."]
OPENERS = ["", "Sure. ", "Good question. ", "Here's a simple way: ", "Easy one. ", "Okay. ", "Happy to help. ", "Of course. "]
CLOSERS = ["", " Hope that helps.", " That should do it.", " Good luck!", " Have fun with it."]
CHAT_OPENERS = ["", "I see. ", "Okay. ", "Hmm. ", "Right. ", "Got it. ", "Ah. ", "Oh. ", "Well, ", "Fair enough. "]

# ---------------------------------------------------------------- (b) true events about famous people (hand-checked)
# (subject, base verb phrase, past verb phrase, answer). Questions: "In what year did S BASE?", "Did S BASE?", "Is it true that S PAST?"
YEAR_FACTS = [
    ("the Wright brothers", "make the first powered aeroplane flight", "made the first powered aeroplane flight", "1903"),
    ("Louis Braille", "publish his reading system for blind people", "published his reading system for blind people", "1829"),
    ("Alexander Graham Bell", "patent the telephone", "patented the telephone", "1876"),
    ("Rakesh Sharma", "travel to space", "travelled to space", "1984"),
    ("Charles Darwin", "publish On the Origin of Species", "published On the Origin of Species", "1859"),
    ("Isaac Newton", "publish the Principia", "published the Principia", "1687"),
    ("Nelson Mandela", "walk free from prison", "walked free from prison", "1990"),
    ("Jawaharlal Nehru", "become India's first Prime Minister", "became India's first Prime Minister", "1947"),
    ("Sachin Tendulkar", "make his Test debut", "made his Test debut", "1989"),
    ("Lionel Messi", "win the FIFA World Cup with Argentina", "won the FIFA World Cup with Argentina", "2022"),
    ("Roger Federer", "win his first Wimbledon title", "won his first Wimbledon title", "2003"),
    ("Serena Williams", "win her first Grand Slam singles title", "won her first Grand Slam singles title", "1999"),
    ("Virat Kohli", "captain India to the Under-19 World Cup title", "captained India to the Under-19 World Cup title", "2008"),
    ("Usain Bolt", "set the 100 metres world record of 9.58 seconds", "set the 100 metres world record of 9.58 seconds", "2009"),
    ("Christopher Columbus", "first reach the Americas", "first reached the Americas", "1492"),
    ("Vasco da Gama", "reach India by sea", "reached India by sea", "1498"),
    ("A. R. Rahman", "win two Academy Awards", "won two Academy Awards", "2009"),
    ("Amitabh Bachchan", "act in his first film", "acted in his first film", "1969"),
    ("Viswanathan Anand", "become India's first chess grandmaster", "became India's first chess grandmaster", "1988"),
    ("Pullela Gopichand", "win the All England badminton title", "won the All England badminton title", "2001"),
    ("Tenzing Norgay", "reach the summit of Mount Everest", "reached the summit of Mount Everest", "1953"),
    ("John F. Kennedy", "become President of the United States", "became President of the United States", "1961"),
    ("Abraham Lincoln", "give the Gettysburg Address", "gave the Gettysburg Address", "1863"),
    ("Queen Elizabeth II", "have her coronation", "had her coronation", "1953"),
    ("Napoleon Bonaparte", "lose the Battle of Waterloo", "lost the Battle of Waterloo", "1815"),
    ("Alexander Fleming", "discover penicillin", "discovered penicillin", "1928"),
    ("Galileo Galilei", "observe the moons of Jupiter", "observed the moons of Jupiter", "1610"),
    ("Nicolaus Copernicus", "publish his book placing the Sun at the centre", "published his book placing the Sun at the centre", "1543"),
    ("Pablo Picasso", "paint Guernica", "painted Guernica", "1937"),
    ("Vincent van Gogh", "paint The Starry Night", "painted The Starry Night", "1889"),
    ("Sundar Pichai", "become CEO of Google", "became CEO of Google", "2015"),
    ("Satya Nadella", "become CEO of Microsoft", "became CEO of Microsoft", "2014"),
    ("Elon Musk", "found SpaceX", "founded SpaceX", "2002"),
    ("Jeff Bezos", "found Amazon", "founded Amazon", "1994"),
    ("Walt Disney", "introduce Mickey Mouse", "introduced Mickey Mouse", "1928"),
    ("Ratan Tata", "become chairman of the Tata Group", "became chairman of the Tata Group", "1991"),
    ("Barack Obama", "win the Nobel Peace Prize", "won the Nobel Peace Prize", "2009"),
    ("Rosa Parks", "refuse to give up her bus seat", "refused to give up her bus seat", "1955"),
    ("Swami Vivekananda", "speak at the Parliament of Religions in Chicago", "spoke at the Parliament of Religions in Chicago", "1893"),
    ("Bhagat Singh", "die for India's freedom struggle", "died for India's freedom struggle", "1931"),
]
OTHER_FACTS = [  # (question, answer sentence, keyword that must be in the answer)
    ("Where was William Shakespeare born?", "William Shakespeare was born in Stratford-upon-Avon, England.", "stratford"),
    ("In which city was Wolfgang Amadeus Mozart born?", "Mozart was born in Salzburg.", "salzburg"),
    ("Who wrote India's national anthem, Jana Gana Mana?", "Rabindranath Tagore wrote Jana Gana Mana.", "tagore"),
    ("Who wrote Vande Mataram?", "Bankim Chandra Chatterjee wrote Vande Mataram.", "bankim"),
    ("Which sport did Dhyan Chand play?", "Dhyan Chand played hockey.", "hockey"),
    ("Did Ludwig van Beethoven lose his hearing?", "Yes. Beethoven gradually lost his hearing and kept composing even when he was deaf.", "yes"),
    ("Who is known as 'The Lady with the Lamp'?", "Florence Nightingale is known as the Lady with the Lamp.", "nightingale"),
    ("Was Charlie Chaplin a silent film star?", "Yes, Charlie Chaplin was one of the great stars of silent films.", "yes"),
    ("Was Kishore Kumar a playback singer?", "Yes, Kishore Kumar was a famous Hindi film playback singer.", "yes"),
    ("Who founded Reliance Industries?", "Dhirubhai Ambani founded Reliance Industries.", "ambani"),
    ("Who is called the 'Nightingale of India'?", "Sarojini Naidu is called the Nightingale of India.", "naidu"),
    ("Did Rani Lakshmibai fight in the revolt of 1857?", "Yes, Rani Lakshmibai, the Queen of Jhansi, fought in the revolt of 1857.", "yes"),
    ("Did Emperor Ashoka turn to Buddhism after the Kalinga war?", "Yes, Ashoka turned to Buddhism after the Kalinga war.", "yes"),
    ("Was Akbar a Mughal emperor?", "Yes, Akbar was one of the great Mughal emperors.", "yes"),
    ("Was Raja Ravi Varma a painter?", "Yes, Raja Ravi Varma was a celebrated Indian painter.", "yes"),
    ("Is Mithali Raj a cricketer?", "Yes, Mithali Raj is a cricketer who captained the Indian women's team.", "yes"),
    ("Is Sunita Williams an astronaut?", "Yes, Sunita Williams is a NASA astronaut.", "yes"),
    ("Did Anne Frank write a diary during the Second World War?", "Yes, Anne Frank kept a diary while hiding during the Second World War.", "yes"),
    ("Who shouted 'Eureka!' according to the famous story?", "According to the story, Archimedes shouted 'Eureka!'", "archimedes"),
    ("Which subject is Euclid famous for?", "Euclid is famous for geometry.", "geometry"),
    ("Where was Mahatma Gandhi born?", "Mahatma Gandhi was born in Porbandar, Gujarat.", "porbandar"),
    ("Did Albert Einstein win a Nobel Prize?", "Yes, Albert Einstein won the Nobel Prize in Physics for 1921, for explaining the photoelectric effect.", "yes"),
]
YEAR_ANS = ["{S} {P} in {Y}.", "In {Y}.", "It happened in {Y}.", "{S} {P} in {Y}; that is a true event."]
YES_ANS = ["Yes, {S} {P} in {Y}.", "Yes. {S} {P} in {Y}.", "Yes, that's true: {S} {P} in {Y}.", "Yes, in {Y}."]

# ---------------------------------------------------------------- (b) knowable questions that sound personal or unknowable
CITIES = [("Pune", "India"), ("Jaipur", "India"), ("Ahmedabad", "India"), ("Kochi", "India"), ("Mysuru", "India"), ("Bhopal", "India"), ("Patna", "India"),
          ("Osaka", "Japan"), ("Lyon", "France"), ("Madrid", "Spain"), ("Toronto", "Canada"), ("Sydney", "Australia"), ("Berlin", "Germany"),
          ("Lagos", "Nigeria"), ("Dhaka", "Bangladesh"), ("Kathmandu", "Nepal"), ("Colombo", "Sri Lanka"), ("Lisbon", "Portugal"), ("Athens", "Greece"),
          ("Beijing", "China"), ("Manila", "the Philippines"), ("Jakarta", "Indonesia"), ("Lima", "Peru"), ("Bangkok", "Thailand"), ("Hanoi", "Vietnam"),
          ("Istanbul", "Turkey"), ("Karachi", "Pakistan"), ("Vienna", "Austria"), ("Amsterdam", "the Netherlands"), ("Dublin", "Ireland"),
          ("Oslo", "Norway"), ("Cape Town", "South Africa"), ("Buenos Aires", "Argentina"), ("Melbourne", "Australia"), ("Munich", "Germany")]
RELS = ["sister", "brother", "neighbour", "teacher", "cousin", "aunt", "uncle", "boss", "roommate", "classmate", "grandmother", "grandfather", "dog", "cat"]
NAMES = ["Asha", "Ravi", "Meera", "Imran", "Divya", "Karthik", "Sneha", "Rohan", "Fatima", "Vikram", "Ananya", "Joseph", "Lakshmi", "Aman", "Neha",
         "Tara", "Sameer", "Pooja", "Arif", "Nisha", "Gopal", "Ishaan", "Riya", "Zoya", "Daniel", "Kavya"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
INVARIANTS = [("Will there still be 12 months in a year in 2050?", "Yes. A year will still have 12 months."),
              ("Will a triangle still have three sides next year?", "Yes. A triangle always has three sides."),
              ("Will a minute still have 60 seconds in 2040?", "Yes. A minute has 60 seconds."),
              ("Will pure water still freeze at 0 degrees Celsius next year?", "Yes. At normal pressure, pure water freezes at 0 degrees Celsius."),
              ("Will a square still have four equal sides tomorrow?", "Yes. A square always has four equal sides."),
              ("Will 10 times 10 still be 100 next week?", "Yes. 10 times 10 is always 100."),
              ("Will a spider still have eight legs next year?", "Yes. Spiders have eight legs."),
              ("Will Mount Everest still be in Asia next month?", "Yes. Mount Everest is in the Himalayas, in Asia."),
              ("Will February still come after January next year?", "Yes. February always comes after January."),
              ("Will a century still be 100 years long in 2100?", "Yes. A century is 100 years.")]


def knowable_rows(rng, n):
    rows = []
    for _ in range(n):
        kind = rng.randrange(7)
        if kind == 0:
            y = rng.randint(1960, 2015); y2 = y + rng.randint(5, 40)
            p = rng.choice([f"I was born in {y}. How old will I turn in {y2}?", f"I was born in {y}. How old will I be on my birthday in {y2}?",
                            f"My birth year is {y}. What age will I reach in {y2}?"])
            a = y2 - y
            t = rng.choice([f"You will be {a}.", f"{a} years old.", f"You'll turn {a}.", f"{y2} minus {y} is {a}, so you'll be {a}.", f"That makes you {a}."])
        elif kind == 1:
            r, nm = rng.choice(RELS), rng.choice(NAMES)
            p = rng.choice([f"My {r} is called {nm}. What is my {r}'s name?", f"My {r}'s name is {nm}. What is my {r} called?",
                            f"I have a {r} named {nm}. What's my {r}'s name?"])
            t = rng.choice([f"Your {r}'s name is {nm}.", f"{nm}.", f"It's {nm}.", f"You told me your {r} is called {nm}.", f"Your {r} is called {nm}."])
        elif kind == 2:
            c, k = rng.choice(CITIES)
            p = rng.choice([f"I live in {c}. Which country do I live in?", f"I'm writing from {c}. What country am I in?", f"My home is in {c}. Which country is that?"])
            t = rng.choice([f"{k[0].upper() + k[1:]}.", f"You live in {k}.", f"{c} is in {k}.", f"That's {k}.", f"You're in {k}."])
        elif kind == 3:
            d, k = rng.randrange(7), rng.randint(1, 6)
            p = rng.choice([f"Today is {DAYS[d]}. What day will it be in {k} days?", f"It's {DAYS[d]} today. Which day comes {k} days from now?"])
            ans = DAYS[(d + k) % 7]
            t = rng.choice([f"It will be {ans}.", f"{ans}.", f"{k} days after {DAYS[d]} is {ans}.", f"That's {ans}."])
        elif kind == 4:
            thing = rng.choice(["books", "pens", "mangoes", "stamps", "coins", "marbles", "biscuits", "balloons"])
            a = rng.randint(5, 40); b = rng.randint(1, a - 1)
            verb = rng.choice(["give away", "lose", "use", "eat" if thing in ("mangoes", "biscuits") else "give away"])
            p = f"I have {a} {thing} and {verb} {b}. How many {thing} do I have now?"
            t = rng.choice([f"You have {a - b} {thing}.", f"{a - b}.", f"{a} minus {b} is {a - b}, so you have {a - b} {thing} left.", f"That leaves you {a - b}."])
        elif kind == 5:
            q, ans = rng.choice(INVARIANTS)
            p, t = q, ans
        else:
            h = rng.randint(1, 8); k = rng.randint(1, 3)
            p = rng.choice([f"My bus leaves at {h} am and the trip takes {k} hours. At what time do I arrive?",
                            f"I start work at {h} am and my shift is {k} hours. When does it finish?"])
            e = h + k
            hh = f"{e} am" if e < 12 else ("12 noon" if e == 12 else f"{e - 12} pm")
            t = rng.choice([f"At {hh}.", f"{hh[0].upper() + hh[1:]}.", f"{h} am plus {k} hours is {hh}.", f"You'll arrive at {hh}." if "bus" in p else f"It finishes at {hh}."])
        rows.append({"source": "synth_trap_v6", "subtype": "knowable", "prompt": p, "text": t})
    return rows


def fact_rows(rng, banned_names):
    rows = []
    for s, base, past, y in YEAR_FACTS:
        if any(w in s.lower() for w in banned_names):
            continue
        for q, bank in ((f"In what year did {s} {base}?", YEAR_ANS), (f"Did {s} {base}?", YES_ANS), (f"Is it true that {s} {past}?", YES_ANS)):
            S = s[0].upper() + s[1:]
            rows.append({"source": "synth_trap_v6", "subtype": "true_event", "prompt": q, "text": rng.choice(bank).format(S=S, P=past, Y=y)})
    for q, a, _ in OTHER_FACTS:
        if any(w in q.lower() for w in banned_names):
            continue
        rows.append({"source": "synth_trap_v6", "subtype": "true_event", "prompt": q, "text": a})
    return rows


# ---------------------------------------------------------------- rewriting
class Bank:
    """Hands out texts so that no text is used more than CAP times in total (counting texts already in the file)."""

    def __init__(self, used):
        self.used = used

    def take(self, candidates):
        for c in candidates:
            if self.used[c] < CAP:
                self.used[c] += 1
                return c
        return None


def combos(rng, *parts, sep=" ", limit=4000):
    out = set()
    tries = 0
    while len(out) < limit and tries < limit * 20:
        tries += 1
        out.add(sep.join(x for x in (rng.choice(p) for p in parts) if x).strip())
    out = sorted(out)
    rng.shuffle(out)
    return out


def classify(r):
    src, sub = r.get("source"), r.get("subtype")
    if src in ("synthetic_refusal", "synth_refusal_v2") and sub != "self_harm_support":
        return "refusal"
    if sub == "self_harm_support":
        return "support"
    if src == "synthetic_i_dont_know":
        return {"private": "private", "future": "future", "false_premise": "prize"}.get(sub, "other_synth")
    if src == "synth_false_premise_real" and sub in ("person", "place"):
        return "false_premise"
    if src == "synthetic_identity":
        return "identity"
    if src in PUBLIC_SOURCES:
        return "natural"
    if sub in ("only_word", "lowercase", "uppercase", "digits", "backwards", "translation"):
        return "answer_only"
    if sub == "count_letters":
        return "letters"
    if sub == "odd_one_out":
        return "odd"
    if src in ("synthetic_over_refusal_guard", "synth_support_contrast"):
        return "wrap"
    return "numeric_or_other"


def rewrite(r, kind, bank, rng, pools):
    text = r["text"]
    if kind in ("refusal", "support", "private", "future", "prize", "identity"):
        cand = pools[kind + ("_no" if kind == "identity" and text.startswith("No") else "")]
        start = rng.randrange(len(cand))
        return bank.take(cand[start:] + cand[:start])
    if kind == "false_premise":
        last = re.split(r"(?<=[.!?])\s+", text.strip())[-1]
        m = re.match(r"^([A-Z][A-Za-z.' -]{1,40}?) never did that\.?$", last)
        cand = []
        if m:
            n = m.group(1)
            cand = [t.format(n=n) for t in FP_NAMED]
            rng.shuffle(cand)
        pool = pools["false_premise"]
        start = rng.randrange(len(pool))
        return bank.take(cand + pool[start:] + pool[:start])
    if kind in ("natural", "answer_only", "other_synth"):
        return None
    m = CALC_SPAN.match(text)
    prefix, tail = m.group(1), m.group(2).strip()
    if kind == "letters":
        n = re.findall(NUM, tail)
        if not n:
            return None
        cand = [prefix + s.format(n=n[-1]) for s in LETTER_SENT]
    elif kind == "odd":
        w = re.sub(r"\s*(?:does not belong|is the odd one out)\.?$", "", tail).strip(" .")
        if not w or " " in w and len(w.split()) > 3:
            return None
        cand = [prefix + (lambda t: t[0].upper() + t[1:])(s.format(w=w)) for s in ODD_SENT]
    elif kind == "wrap":
        if r.get("source") == "synthetic_over_refusal_guard":
            cand = [f"{o}{tail}{c}" for o in OPENERS for c in CLOSERS]
        else:  # conversational replies (support contrast): neutral openers only, keeping the reply itself
            cand = [o + (tail[0].lower() + tail[1:] if o == "Well, " and not tail.startswith("I ") else tail) for o in CHAT_OPENERS]
    else:
        nums = re.findall(NUM, tail)
        if not nums:
            cand = [f"{o}{tail}{c}" for o in OPENERS for c in CLOSERS] if not prefix else []
        else:
            n = nums[-1]
            um = re.match(r"^(?:that is|it is|it's|there are|each gets|that makes|the answer is)?\s*" + re.escape(n) + r"\s+([a-z]+(?: [a-z]+)?)\.?$", tail, re.I)
            if um:
                cand = [prefix + s.format(n=n, u=um.group(1)) for s in UNIT_SENT]
            elif re.fullmatch(r"(?:the answer is |each gets |that is |it is |it's )?" + re.escape(n) + r"\.?", tail, re.I) or prefix:
                cand = [prefix + s.format(n=n) for s in NUM_SENT]
            else:
                return None
    rng.shuffle(cand)
    return bank.take(cand)


def load_v5(split):
    path, sha = V5[split]
    raw = open(project_path(path), "rb").read()
    assert hashlib.sha256(raw).hexdigest() == sha, path
    return [json.loads(l) for l in raw.decode("utf-8").splitlines()]


def jsonl(path):
    return [json.loads(l) for l in open(project_path(path), encoding="utf-8")]


def main():
    rng = random.Random(SEED)
    tok = Tokenizer.from_file(project_path(TOKENIZER_PATH))
    pools = {
        "refusal": combos(rng, REFUSE_CORE, REFUSE_REASON, REFUSE_OFFER, limit=4000),
        "support": combos(rng, SUPPORT_OPEN, SUPPORT_MID, SUPPORT_HELP, limit=190),
        "private": combos(rng, PRIV_CORE, PRIV_REASON, PRIV_OFFER, limit=600),
        "future": combos(rng, FUTURE_CORE, FUTURE_REASON, FUTURE_EXTRA, limit=500),
        "false_premise": combos(rng, FP_CORE, FP_FOLLOW, FP_EXTRA, limit=600),
        "prize": list(PRIZE_FP),
        "identity": [f"{a}{' ' if a.endswith(chr(39) + 'm') else ', '}{b}.{(' ' + c) if c else ''}" for a in ID_NAME for b in ID_WHAT for c in ID_EXTRA],
    }
    pools["private"] = [t for t in pools["private"] if t.count("have access") < 2 and t.count("can't know") < 2]
    pools["identity_no"] = ["No. " + t for t in pools["identity"]] + ["No, " + t for t in pools["identity"] if t.startswith("I")]
    for k in pools:
        rng.shuffle(pools[k])

    # evaluation sets for decontamination (checked in code; never printed)
    sealed = {name: load_goal_set(f"phase20/sealed_v4/{name}.jsonl", f"phase20/sealed_v4/{name}_meta.json")
              for name in ("goal_heldout_v4", "overrefusal_v4", "misfire_v4", "harmful_xstest", "xstest_safe")}
    from datasets import load_dataset
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    sets = dict(sealed)
    sets["dev_v2_1"] = load_goal_set("phase11/goal_eval/goal_prompts_v2_1.jsonl", "phase11/goal_eval/goal_set_v2_1_meta.json")
    sets["overrefusal_old"] = load_goal_set("phase11/overrefusal/overrefusal_prompts.jsonl", "phase11/overrefusal/overrefusal_meta.json")
    sets["misfire_dev"] = load_goal_set("phase20/dev/misfire_dev.jsonl", "phase20/dev/misfire_dev_meta.json")
    sets["gsm8k_test"] = [{"id": f"gsm8k_{i}", "prompt": q} for i, q in enumerate(gsm["question"])]
    checkers = {k: ContaminationChecker([{"id": it["id"], "prompt": it.get("question", it["prompt"])} for it in v]) for k, v in sets.items()}
    W = re.compile(r"[a-z0-9]+")
    sealed8 = set()
    for v in sealed.values():
        for it in v:
            ws = W.findall(it["prompt"].lower())
            sealed8 |= {tuple(ws[i:i + 8]) for i in range(len(ws) - 7)}
    names = set()
    for k in ("goal_heldout_v4", "misfire_v4"):
        for it in sealed[k]:
            names |= {w.lower() for w in re.findall(r"\b[A-Z][a-z]{3,}\b", it["prompt"])}
    for it in sets["misfire_dev"]:
        names |= {w.lower() for w in re.findall(r"\b[A-Z][a-z]{3,}\b", it["prompt"])}
    common = {"what", "which", "when", "where", "will", "today", "april", "march", "june", "july", "december", "january", "monday", "tuesday",
              "wednesday", "thursday", "friday", "saturday", "sunday", "olympic", "olympics", "world", "cricket", "test", "india", "indian", "nobel",
              "prize", "mount", "everest", "english", "channel", "south", "pole", "first", "atlantic", "commonwealth", "games", "bharat", "ratna",
              "celsius", "earth", "moon", "france", "paris", "fifa", "wimbledon", "championship", "chess", "oscar", "photo", "history", "brief", "time",
              "harry", "potter", "book", "google", "facebook", "microsoft", "iphone", "officer", "barrier", "sound", "radio", "signal", "marathon",
              "javelin", "tennis", "boxing", "gymnastics", "perfect", "heavyweight", "title", "periodic", "table", "polio", "vaccine", "heart",
              "transplant", "space", "american", "woman", "hockey", "badminton", "england", "final", "medal", "gold", "silver", "bronze", "kerala"}
    banned = names - common

    counts = {}
    out = {}
    for split in ("train", "val"):
        c = Counter()
        rows = load_v5(split)
        c["v5_rows"] = len(rows)
        # (b) mined rows at 5 copies: v5 has 1 base + 3 added copies (4 identical rows) of each mined row
        key = Counter((r["prompt"], r["text"]) for r in rows)
        mined_keys = {k for k, n in key.items() if n == 4 and RISKY_RE.search(k[0]) and not HARM_RE.search(k[0])}
        add = []
        for k in mined_keys:
            src = next(r for r in rows if (r["prompt"], r["text"]) == k)
            add.append(dict(src))
        c["mined_v5_distinct"] = len(mined_keys)
        c["mined_v5_extra_copy_rows"] = len(add)
        rows = rows + add
        used = Counter()
        bank = Bank(used)
        by_text = defaultdict(list)
        for i, r in enumerate(rows):
            by_text[r["text"]].append(i)
        keep_idx = set(range(len(rows)))
        for t, idxs in by_text.items():
            if len(idxs) <= CAP:
                used[t] += len(idxs)
        new_rows = list(rows)
        for t, idxs in by_text.items():
            if len(idxs) <= CAP:
                continue
            kind = classify(rows[idxs[0]])
            rng.shuffle(idxs)
            # mined (key-repeated) public rows: keep up to CAP copies; others: first CAP keep their text if allowed
            for j, i in enumerate(idxs):
                r = rows[i]
                k = classify(r)
                if k in ("refusal", "support", "private", "future", "prize", "identity", "false_premise", "letters", "odd", "wrap", "numeric_or_other"):
                    nt = rewrite(r, k, bank, rng, pools)
                    if nt is None:
                        if used[t] < CAP:
                            used[t] += 1
                            c[f"kept_original_{k}"] += 1
                        else:
                            keep_idx.discard(i)
                            c[f"dropped_over_cap_{k}"] += 1
                    else:
                        new_rows[i] = dict(r, text=nt, rewritten_from_v5=True)
                        c[f"rewritten_{k}"] += 1
                else:
                    if used[t] < CAP:
                        used[t] += 1
                    else:
                        keep_idx.discard(i)
                        c[f"dropped_over_cap_{k}"] += 1
        rows = [new_rows[i] for i in sorted(keep_idx)]
        # (b) new trap rows and newly mined harmless edgy rows
        share = 0.02 if split == "val" else 1.0
        traps = (fact_rows(rng, banned) if split == "train" else []) + knowable_rows(rng, 1400 if split == "train" else 30)
        c["trap_true_event_candidates"] = sum(1 for r in traps if r["subtype"] == "true_event")
        c["trap_knowable_candidates"] = sum(1 for r in traps if r["subtype"] == "knowable")
        rows += traps
        if split == "train":
            seen_prompts = {r["prompt"] for r in rows}
            pub_counts = defaultdict(Counter)
            newly = []
            cache = project_path(f"{OUT_DIR}/_public_rows_cache.jsonl")
            if os.path.exists(cache):
                public = [json.loads(l) for l in open(cache, encoding="utf-8")]
            else:
                public = [dict(r, source=name) for name in sorted(PUBLIC_SOURCES) for r in load_public(name, pub_counts)]
                with open(cache, "w", encoding="utf-8") as f:
                    for r in public:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            for r in public:
                name = r["source"]
                if True:
                    if (r["prompt"] not in seen_prompts and RISKY_RE.search(r["prompt"]) and not HARM_RE.search(r["prompt"])
                            and not is_refusal(r["text"]) and not offers_support(r["text"])):
                        newly.append({"source": name, "subtype": "mined_harmless_v6", "prompt": r["prompt"], "text": r["text"]})
                        seen_prompts.add(r["prompt"])
            rng.shuffle(newly)
            c["new_mined_candidates"] = len(newly)
            newly = newly[:NEW_MINED_MAX * 2]
            rows += newly
        # length filter for every row (as phase11)
        kept = []
        enc = tok.encode_batch_fast([example_text(r["prompt"], r["text"]) for r in rows], add_special_tokens=False)
        rep = tok.encode_batch_fast([r["text"] for r in rows], add_special_tokens=False)
        n_new_mined = 0
        for r, e, rp in zip(rows, enc, rep):
            if len(e.ids) > MAX_TOTAL_TOKENS or len(rp.ids) > MAX_REPLY_TOKENS:
                c[f"dropped_length_{r.get('subtype')}" if r.get("source") == "synth_trap_v6" or r.get("subtype") == "mined_harmless_v6" else "dropped_length_other"] += 1
                continue
            if r.get("subtype") == "mined_harmless_v6":
                if n_new_mined >= NEW_MINED_MAX:
                    continue
                n_new_mined += 1
            kept.append(dict(r, total_tokens=len(e.ids), reply_tokens=len(rp.ids)))
        rows = kept
        # contamination
        clean_rows = []
        for r in rows:
            hits = [name for name, ch in checkers.items() if find_matches(ch, r)]
            ws = W.findall(r["prompt"].lower())
            if any(tuple(ws[i:i + 8]) in sealed8 for i in range(len(ws) - 7)):
                hits.append("sealed_8gram")
            if hits:
                c["removed_contamination"] += 1
                for h in hits:
                    c[f"removed_contamination_against_{h}"] += 1
                continue
            clean_rows.append(r)
        rows = clean_rows
        # final cap check (dropping any remaining excess, e.g. a trap reply that equals an existing text)
        seen = Counter()
        final = []
        for r in rows:
            seen[r["text"]] += 1
            if seen[r["text"]] > CAP:
                c["dropped_final_cap"] += 1
                continue
            final.append(r)
        rng.shuffle(final)
        for i, r in enumerate(final):
            r["id"] = f"sftv6a_{split}_{i:06d}"
        assert max(Counter(r["text"] for r in final).values()) <= CAP
        c["total"] = len(final)
        c["max_identical_reply"] = max(Counter(r["text"] for r in final).values())
        c.update({f"source_{k}": v for k, v in Counter(r["source"] for r in final).items()})
        c["trap_true_event_rows"] = sum(1 for r in final if r.get("subtype") == "true_event")
        c["trap_knowable_rows"] = sum(1 for r in final if r.get("subtype") == "knowable" and r.get("source") == "synth_trap_v6")
        c["new_mined_rows"] = sum(1 for r in final if r.get("subtype") == "mined_harmless_v6")
        c["refusal_rows"] = sum(1 for r in final if classify(r) == "refusal")
        counts[split] = dict(c)
        out[split] = final

    os.makedirs(project_path(OUT_DIR), exist_ok=True)
    files = {}
    for split, rows in out.items():
        path = project_path(f"{OUT_DIR}/sft_v6a_{split}.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        files[split] = {"file": f"{OUT_DIR}/sft_v6a_{split}.jsonl", "rows": len(rows), "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    v5_refusals = sum(1 for r in load_v5("train") if classify(r) == "refusal")
    manifest = {"seed": SEED, "cap": CAP, "files": files, "counts": counts, "v5_refusal_rows_train": v5_refusals,
                "banned_name_count": len(banned), "pools": {k: len(v) for k, v in pools.items()}}
    atomic_write_json(project_path("phase20/manifests/sft_v6a_manifest.json"), manifest)
    print(json.dumps({"files": files, "counts": counts, "v5_refusal_rows_train": v5_refusals}, indent=1))


if __name__ == "__main__":
    main()
