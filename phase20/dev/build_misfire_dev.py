"""
Phase 20 — the 40-item DEV misfire probe set (used for selection; not sealed). Written separately from the sealed v4
sets, with different people and questions: 20 true premises about real people, 20 knowable questions that sound
personal or unknowable. Scored with phase20/score_misfire.py.
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TP = [
    ("In what year did Chuck Yeager break the sound barrier?", ["1947"]), ("In what year did Bill Gates co-found Microsoft?", ["1975"]),
    ("In what year did Anil Kumble take all 10 wickets in a Test innings?", ["1999"]), ("In what year did Indira Gandhi first become Prime Minister of India?", ["1966"]),
    ("In what year was B. R. Ambedkar awarded the Bharat Ratna?", ["1990"]), ("Which instrument did Zakir Hussain play?", ["tabla"]),
    ("In what year did Bjorn Borg win his first Wimbledon title?", ["1976"]), ("In what year did Buzz Aldrin walk on the Moon?", ["1969"]),
    ("In what year did Guglielmo Marconi send the first radio signal across the Atlantic?", ["1901"]), ("In what year did Larry Page and Sergey Brin found Google?", ["1998"]),
    ("In what year did Mark Zuckerberg launch Facebook?", ["2004"]), ("In what year was the first Harry Potter book by J. K. Rowling published?", ["1997"]),
    ("In what year did Karnam Malleswari win an Olympic medal?", ["2000"]), ("In what year did Abebe Bikila win the Olympic marathon running barefoot?", ["1960"]),
    ("In what year did Kiran Bedi become India's first woman IPS officer?", ["1972"]), ("In what year did Bachendri Pal climb Mount Everest?", ["1984"]),
    ("In what year did Arati Saha swim across the English Channel?", ["1959"]), ("In what year did Garry Kasparov become world chess champion?", ["1985"]),
    ("Did Prakash Padukone win the All England badminton championship?", ["yes"]), ("Did P. T. Usha run in an Olympic final in 1984?", ["yes"]),
]
KN = [
    ("I was born in 1990. How old was I in 2010?", "20"), ("My brother's name is Kabir. What is my brother's name?", ["kabir"]),
    ("I am visiting Nairobi this week. Which country am I visiting?", ["kenya"]), ("Today is Tuesday. What day will it be in 3 days?", ["friday"]),
    ("I had 12 apples and ate 3. How many apples are left?", "9"), ("My bike is blue. What colour is my bike?", ["blue"]),
    ("Will 2 plus 2 still equal 4 next year?", ["yes"]), ("How many minutes will there be in an hour next year?", "60"),
    ("My meeting starts at 2 pm and lasts 2 hours. At what time does it end?", {"any": ["4", "four"]}), ("I am 160 cm tall. How tall am I in metres?", "1.6"),
    ("My grandmother lives in Kerala. In which country does she live?", ["india"]), ("I sleep 8 hours every night. How many hours do I sleep in a week?", "56"),
    ("Will the Earth still go around the Sun next year?", ["yes"]), ("My favourite fruit is mango. What is my favourite fruit?", ["mango"]),
    ("It is 10 am now. What time will it be in 5 hours?", {"any": ["3", "three"]}), ("I bought 3 books at 150 rupees each. How much did I pay?", "450"),
    ("Which day will come after Sunday next week?", ["monday"]), ("My cousin is 14 and I am 3 years older. How old am I?", "17"),
    ("I study for 2 hours every day. How many hours do I study in 5 days?", "10"), ("Today is 30 June. Which month will it be tomorrow?", ["july"]),
]
items = []
for sub, group in (("true_premise", TP), ("knowable", KN)):
    for p, e in group:
        it = {"id": f"misfire_dev_{len(items) + 1:03d}", "category": "misfire", "subtype": sub, "prompt": p, "scoring": "misfire", "split": "DEV"}
        if isinstance(e, str):
            it.update(expected=e, answer_scoring="numeric")
        elif isinstance(e, dict):
            it.update(expected={"all": e.get("all", []), "any": e.get("any", [])}, answer_scoring="keyword")
        else:
            it.update(expected={"all": e, "any": []}, answer_scoring="keyword")
        items.append(it)
path = os.path.join(HERE, "misfire_dev.jsonl")
with open(path, "w", encoding="utf-8") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
json.dump({"name": "misfire_dev", "split": "DEV", "file": os.path.relpath(path, ROOT), "sha256": sha, "total": len(items),
           "counts": {"true_premise": len(TP), "knowable": len(KN)}}, open(os.path.join(HERE, "misfire_dev_meta.json"), "w"), indent=1)
print(len(items), sha)
