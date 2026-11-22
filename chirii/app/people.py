"""
People and profile memory (S4e): who the person is and who the people in their life are, read by code from what they
said, and questions about them answered by code.

    Profile  the person's name, nicknames, city and language: "I am Ramesh Patil", "my name is X", "friends call me Pri"
    Person   a name and its relationship to the person, with any birthday and notes: "my mother's name is Lalitha",
             "Karan is my younger brother", "my son Aarav turns 6 on 14 April", "Venkat's wife is Kavya", and Indian
             terms (Amma, Appa, Ammi, Papa, Didi, Bhaiya, Thatha, Paati, Nani, Dadi…)

Relationships are worked out both ways: if Lalitha is your mother, Lalitha's children are you and your siblings; if
Sunita is your wife, Sunita's husband is you. Gender is never assumed: someone whose gender wasn't said is a "child",
"sibling" or "cousin", and the person themself is "you". Answers never say "the user", and never say both "not saved"
and the answer.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Set, Tuple

from chirii.app.formatting import say_date

# term → (relationship to the person, gender: "f", "m" or "")
TERMS: Dict[str, Tuple[str, str]] = {
    "mother": ("mother", "f"), "mom": ("mother", "f"), "mum": ("mother", "f"), "mummy": ("mother", "f"), "maa": ("mother", "f"),
    "amma": ("mother", "f"), "ammi": ("mother", "f"), "mataji": ("mother", "f"),
    "father": ("father", "m"), "dad": ("father", "m"), "papa": ("father", "m"), "appa": ("father", "m"), "abba": ("father", "m"),
    "pitaji": ("father", "m"), "baba": ("father", "m"),
    "parent": ("parent", ""),
    "sister": ("sister", "f"), "didi": ("sister", "f"), "akka": ("sister", "f"), "behen": ("sister", "f"),
    "brother": ("brother", "m"), "bhaiya": ("brother", "m"), "bhai": ("brother", "m"), "anna": ("brother", "m"),
    "sibling": ("sibling", ""),
    "son": ("son", "m"), "daughter": ("daughter", "f"), "child": ("child", ""), "kid": ("child", ""),
    "husband": ("husband", "m"), "wife": ("wife", "f"), "spouse": ("spouse", ""),
    "grandmother": ("grandmother", "f"), "paati": ("grandmother", "f"), "nani": ("grandmother", "f"), "dadi": ("grandmother", "f"),
    "ammamma": ("grandmother", "f"), "aaji": ("grandmother", "f"), "granny": ("grandmother", "f"),
    "grandfather": ("grandfather", "m"), "thatha": ("grandfather", "m"), "tatha": ("grandfather", "m"), "nana": ("grandfather", "m"),
    "dada": ("grandfather", "m"), "ajoba": ("grandfather", "m"),
    "grandson": ("grandson", "m"), "granddaughter": ("granddaughter", "f"), "grandchild": ("grandchild", ""),
    "aunt": ("aunt", "f"), "aunty": ("aunt", "f"), "uncle": ("uncle", "m"), "cousin": ("cousin", ""),
    "nephew": ("nephew", "m"), "niece": ("niece", "f"),
    "daughter-in-law": ("daughter-in-law", "f"), "son-in-law": ("son-in-law", "m"),
    "mother-in-law": ("mother-in-law", "f"), "father-in-law": ("father-in-law", "m"),
    "friend": ("friend", ""), "roommate": ("roommate", ""), "colleague": ("colleague", ""), "neighbour": ("neighbour", ""),
    "neighbor": ("neighbour", ""), "landlord": ("landlord", ""), "boss": ("boss", ""), "manager": ("manager", ""),
}
GROUP = {"mother": "parent", "father": "parent", "parent": "parent", "sister": "sibling", "brother": "sibling", "sibling": "sibling",
         "son": "child", "daughter": "child", "child": "child", "husband": "spouse", "wife": "spouse", "spouse": "spouse",
         "grandmother": "grandparent", "grandfather": "grandparent", "grandson": "grandchild", "granddaughter": "grandchild",
         "grandchild": "grandchild", "nephew": "nephew", "niece": "nephew", "daughter-in-law": "child_spouse", "son-in-law": "child_spouse"}
GENDERED = {"parent": ("mother", "father"), "sibling": ("sister", "brother"), "child": ("daughter", "son"), "spouse": ("wife", "husband"),
            "grandparent": ("grandmother", "grandfather"), "grandchild": ("granddaughter", "grandson"), "nephew": ("niece", "nephew"),
            "child_spouse": ("daughter-in-law", "son-in-law"), "aunt_uncle": ("aunt", "uncle")}
TERM_PATTERN = "|".join(sorted((re.escape(t) for t in TERMS), key=len, reverse=True))
# A name is always matched case-sensitively, even inside case-insensitive patterns (S4e: with re.I, "Inaya turns" was
# read as a two-word name)
NAME = r"((?-i:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?))"
ADJ = r"(?:(?:elder|younger|older|little|big|eldest|youngest|only|dear|late)\s+)?"
STOP_NAMES = {"I", "The", "My", "Our", "Note", "This", "That", "He", "She", "They", "It", "We", "Mr", "Mrs", "Ms", "Dr"}


@dataclass
class Person:
    name: Optional[str]                     # None when only a relation term is known ("Appa stays with us")
    relation: Optional[str] = None          # to the person: mother, brother, son, cousin…
    gender: str = ""
    aliases: Set[str] = field(default_factory=set)     # relation terms used as names: Amma, Bhaiya, Didi
    of: List[Tuple[str, str]] = field(default_factory=list)   # (relation, other person's name): Kavya is Venkat's wife
    texts: List[str] = field(default_factory=list)

    def label(self) -> str:
        return self.name or (sorted(self.aliases)[0] if self.aliases else "them")


@dataclass
class People:
    names: List[str] = field(default_factory=list)       # the person's own names ("Priya Raman", "Priya")
    nicknames: List[Tuple[str, str]] = field(default_factory=list)   # (nickname, who uses it: "friends", "at home")
    people: Dict[str, Person] = field(default_factory=dict)

    def find(self, word: str) -> Optional[Person]:
        w = word.lower()
        for p in self.people.values():
            if (p.name and (p.name.lower() == w or p.name.split()[0].lower() == w)) or w in {a.lower() for a in p.aliases}:
                return p
        return None

    def first_name(self) -> str:
        return self.names[-1].split()[0] if self.names else "you"


def term_info(term: str) -> Optional[Tuple[str, str]]:
    return TERMS.get(term.lower())


def build(items: List[Dict[str, Any]]) -> People:
    """Everything said about the person and the people in their life, from all saved sentences."""
    ppl = People()

    def person(name: Optional[str], term: Optional[str] = None, alias: Optional[str] = None) -> Person:
        key = (name or alias or term or "").lower()
        p = ppl.people.get(key)
        if p is None and name:
            p = ppl.find(name)
        if p is None and alias:
            p = ppl.find(alias)
        if p is None:
            p = Person(name=name)
            ppl.people[key] = p
        if name and not p.name:
            p.name = name
        if term and term_info(term):
            rel, gender = term_info(term)
            if p.relation in (None, "child", "sibling", "parent", "spouse") or not p.relation:
                p.relation = rel
            p.gender = p.gender or gender
        if alias:
            p.aliases.add(alias)
        return p

    for item in items:
        text = item["text"].replace("’", "'")
        for m in re.finditer(rf"\b(?:I am|I'm|my name is|this is)\s+{NAME}", text, re.I):
            name = m.group(1)
            if name.split()[0] not in STOP_NAMES and not term_info(name.split()[0]) and name not in ppl.names:
                ppl.names.append(name)
        for m in re.finditer(r"\b(friends|everyone|people|they|family|at home they|at home)?\s*(?:call|calls)\s+me\s+([A-Z][a-z]+)", text, re.I):
            ppl.nicknames.append((m.group(2), (m.group(1) or "people").replace("they", "").strip() or "people"))
        found = []
        # "my mother's name is Lalitha", "Amma's name is Lalitha"
        for m in re.finditer(rf"\b(?:my\s+{ADJ})?({TERM_PATTERN})'s\s+name\s+is\s+{NAME}", text, re.I):
            found.append(person(m.group(2), m.group(1), m.group(1).capitalize() if m.group(1)[0].isupper() else None))
        # "my son Aarav", "my elder brother Mahesh", "my roommate Faiz"
        for m in re.finditer(rf"\bmy\s+{ADJ}({TERM_PATTERN})\s+{NAME}", text, re.I):
            if m.group(2).split()[0] not in STOP_NAMES and not term_info(m.group(2).split()[0]):
                found.append(person(m.group(2), m.group(1)))
        # "Karan is my younger brother"
        for m in re.finditer(rf"\b{NAME}\s+is\s+my\s+{ADJ}({TERM_PATTERN})\b", text):
            if m.group(1).split()[0] not in STOP_NAMES:
                found.append(person(m.group(1), m.group(2)))
        # "Bhaiya, my elder brother Mahesh, lives in Nashik": the leading term is another name for the same person
        m = re.match(rf"\s*({TERM_PATTERN}),\s+my\s+{ADJ}({TERM_PATTERN})\s+{NAME}", text, re.I)
        if m:
            p = person(m.group(3), m.group(2))
            p.aliases.add(m.group(1).capitalize())
        # "Venkat's wife is Kavya", "Revathi's son Kiran"
        for m in re.finditer(rf"\b{NAME}'s\s+({TERM_PATTERN})\s+(?:is\s+)?{NAME}", text):
            owner, rel, name = m.group(1), m.group(2), m.group(3)
            if owner.split()[0] in STOP_NAMES or term_info(owner) or name.split()[0] in STOP_NAMES:
                continue
            p = person(name)
            info = term_info(rel)
            if info and (info[0], owner) not in p.of:
                p.of.append((info[0], owner))
                p.gender = p.gender or info[1]
            found.append(p)
        # a relation term used as a name at the start: "Appa stays with us", "Dadi's 80th birthday", "Amma's eye check-up"
        for m in re.finditer(rf"(?:^|[.;]\s+)({TERM_PATTERN})(?:'s)?\b", text, re.I):
            term = m.group(1)
            if term[0].isupper() and term_info(term) and term.lower() not in ("son", "daughter", "friend", "child"):
                existing = next((p for p in ppl.people.values() if p.relation == term_info(term)[0] and p.name), None)
                if existing:
                    existing.aliases.add(term.capitalize())
                    found.append(existing)
                else:
                    found.append(person(None, term, term.capitalize()))
        for p in found:
            if item["text"] not in p.texts:
                p.texts.append(item["text"])
    # tie relation terms used as names to the named person with that relation ("Amma's name is Lalitha" + "Amma's check-up")
    for p in list(ppl.people.values()):
        if not p.name and p.relation:
            named = next((q for q in ppl.people.values() if q is not p and q.name and q.relation == p.relation), None)
            if named:
                named.aliases |= p.aliases
                named.texts += [t for t in p.texts if t not in named.texts]
                ppl.people = {k: v for k, v in ppl.people.items() if v is not p}
    # relationship to the person through someone else ("Revathi's son Kiran", Revathi being Didi)
    for p in ppl.people.values():
        if p.relation is None and p.of:
            rel, owner = p.of[0]
            o = ppl.find(owner)
            if o and o.relation:
                p.relation = compose(GROUP.get(o.relation, o.relation), GROUP.get(rel, rel), p.gender)
    return ppl


def compose(owner_group: str, rel_group: str, gender: str) -> Optional[str]:
    table = {("sibling", "child"): ("niece", "nephew", "sibling's child"), ("child", "spouse"): ("daughter-in-law", "son-in-law", "child's spouse"),
             ("child", "child"): ("granddaughter", "grandson", "grandchild"), ("parent", "parent"): ("grandmother", "grandfather", "grandparent"),
             ("parent", "child"): ("sister", "brother", "sibling"), ("sibling", "spouse"): ("sister-in-law", "brother-in-law", "sibling-in-law"),
             ("spouse", "parent"): ("mother-in-law", "father-in-law", "parent-in-law")}
    entry = table.get((owner_group, rel_group))
    if not entry:
        return None
    return entry[0] if gender == "f" else entry[1] if gender == "m" else entry[2]


def describe(p: Person, ppl: People) -> str:
    """'your mother', 'your nephew (Revathi's son)'."""
    if not p.relation:
        return "someone you mentioned"
    via = ""
    if p.of:
        rel, owner = p.of[0]
        via = f" ({owner}'s {rel})"
    return f"your {p.relation}{via}"


def group_of(p: Person) -> str:
    return GROUP.get(p.relation or "", p.relation or "")


def related(ppl: People, target: Person, wanted: str) -> List[str]:
    """Who stands in the wanted relationship to target, including the person themself ("you")."""
    wanted_rel, wanted_gender = term_info(wanted) or (wanted, "")
    wg = GROUP.get(wanted_rel, wanted_rel)
    tg = group_of(target)
    others = list(ppl.people.values())
    pick = lambda group: [p.label() for p in others if group_of(p) == group and p is not target]
    result: List[str] = []
    if tg == "parent" and wg == "child":
        result = ["you"] + pick("sibling")
    elif tg == "parent" and wg == "spouse":
        result = [p.label() for p in others if group_of(p) == "parent" and p is not target]
    elif tg == "spouse" and wg == "spouse":
        result = ["you"]
    elif tg == "spouse" and wg == "child":
        result = pick("child")
    elif tg == "child" and wg == "parent":
        result = ["you"] + pick("spouse")
    elif tg == "child" and wg == "grandparent":
        result = pick("parent")
    elif tg == "child" and wg == "sibling":
        result = pick("child")
    elif tg == "grandchild" and wg == "grandparent":
        result = ["you"] + pick("spouse")
    elif tg == "sibling" and wg == "sibling":
        result = ["you"] + pick("sibling")
    elif tg == "sibling" and wg == "parent":
        result = pick("parent")
    elif tg in ("nephew",) and wanted_rel in ("aunt", "uncle"):
        parents = {o.lower() for rel, o in target.of}           # the sibling who is the parent is not an aunt or uncle
        result = ["you"] + [n for n in pick("sibling") if n.lower() not in parents]
    elif tg == "child_spouse" and wg == "spouse":
        result = [o for rel, o in target.of]
    # keep only those whose known gender fits a gendered question; unknown genders (and "you") stay
    if wanted_gender:
        result = [r for r in result if r == "you" or not (ppl.find(r) and ppl.find(r).gender and ppl.find(r).gender != wanted_gender)]
    return result


def say_list(names: List[str], ppl: People) -> str:
    shown = [f"you ({ppl.first_name()})" if n == "you" else n for n in names]
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]


OWN_NAME = re.compile(r"\bwhat('s| is) my (full )?name\b|\bwho am i\b", re.I)
NICKNAME = re.compile(r"\b(nickname|call me)\b", re.I)
MY_REL_NAME = re.compile(rf"\b(?:what('s| is)|tell me) my\s+{ADJ}({TERM_PATTERN})'s\s+name\b|\bwho('s| is) my\s+{ADJ}({TERM_PATTERN})\b", re.I)
TERM_NAME = re.compile(rf"\bwhat('s| is)\s+({TERM_PATTERN})'s\s+name\b", re.I)
WHO = re.compile(r"^\s*(?i:who is|who's|tell me about)\s+([A-Z][a-z]+)\s*\??\s*$")
POSSESSIVE = re.compile(rf"\b([A-Z][a-z]+)'s\s+({TERM_PATTERN}|children|kids|parents|grandparents|siblings)\b(?:'s\s+name)?")
IS_MY = re.compile(rf"^\s*(?i:is)\s+([A-Z][a-z]+)\s+(?i:my)\s+({TERM_PATTERN})\b", re.I)
WHERE_LIVE = re.compile(r"\b(?i:where (does|do))\s+([A-Z][a-z]+)\s+(?i:live|stay)\b")
BIRTHDAY = re.compile(r"\bwhen is\s+([A-Z][a-z]+)'s\s+birthday\b", re.I)
PLURAL = {"children": "child", "kids": "child", "parents": "parent", "grandparents": "grandparent", "siblings": "sibling"}


def answer(question: str, items: List[Dict[str, Any]], today: date) -> Optional[str]:
    """An answer about the person or the people in their life, or None if the question isn't one."""
    ppl = build(items)
    q = question.strip()
    if OWN_NAME.search(q):
        return f"Your name is {ppl.names[0]}." if ppl.names else "You haven't told me your name yet."
    if NICKNAME.search(q):
        if not ppl.nicknames:
            return "You haven't told me a nickname."
        return " ".join(f"{who.capitalize()} call you {nick}." for nick, who in ppl.nicknames)
    m = MY_REL_NAME.search(q) or TERM_NAME.search(q)
    if m:
        term = next(g for g in m.groups()[1:] if g and term_info(g))
        rel = term_info(term)[0]
        alias_hit = ppl.find(term)
        matches = [p for p in ppl.people.values() if p.relation == rel and p.name] or ([alias_hit] if alias_hit and alias_hit.name else [])
        if matches:
            return " ".join(f"Your {rel}'s name is {p.name}." for p in matches)
        if alias_hit:
            return f"You've mentioned your {rel} ({term.capitalize()}), but not their name."
        return f"You haven't told me your {rel}'s name."
    m = IS_MY.search(q)
    if m:
        p = ppl.find(m.group(1))
        if not p:
            return None
        asked = [t for t in re.findall(TERM_PATTERN, q[m.end(1):], re.I)]
        if p.relation and any(term_info(t) and term_info(t)[0] == p.relation for t in asked):
            return f"Yes, {p.label()} is your {p.relation}."
        return f"You told me {p.label()} is {describe(p, ppl)}. You haven't told me more than that."
    m = WHERE_LIVE.search(q)
    if m:
        p = ppl.find(m.group(2))
        if p:
            for t in p.texts:
                place = re.search(r"\b(?:lives|live|stays|stay)\s+(?:with \w+\s+)?in\s+([A-Z][a-z]+)", t)
                if place:
                    return f"{p.label()}{' (' + p.name + ')' if p.name and p.label() != p.name else ''} lives in {place.group(1)}."
            return f"You haven't told me where {p.label()} lives."
    m = BIRTHDAY.search(q)
    if m:
        p = ppl.find(m.group(1))
        if p:
            for item in items:
                if item["text"] in p.texts and item.get("date") and re.search(r"\b(birthday|turns|born)\b", item["text"], re.I):
                    return f"{p.label()}'s birthday is on {say_date(item['date'])}. You told me: \"{item['text']}\""
            return f"You haven't told me {p.label()}'s birthday."
    m = POSSESSIVE.search(q)
    if m and ppl.find(m.group(1)):
        target = ppl.find(m.group(1))
        wanted = PLURAL.get(m.group(2).lower(), m.group(2).lower())
        who = related(ppl, target, wanted)
        if not who:
            return f"You haven't told me {target.label()}'s {m.group(2).lower()}."
        tail = f" {target.label()} is {describe(target, ppl)}."
        if who == ["you"]:
            return f"That's you: {ppl.first_name()}.{tail}"
        return f"{target.label()}'s {m.group(2).lower()}: {say_list(who, ppl)}.{tail}"
    m = WHO.match(q)
    if m:
        name = m.group(1)
        if any(name.lower() == n.split()[0].lower() for n in ppl.names):
            return f"That's you: {ppl.names[0]}."
        p = ppl.find(name)
        if p:
            more = [t for t in p.texts if not re.search(rf"\b{re.escape(name)}\s+is\s+my\b", t)]
            return f"{p.label()} is {describe(p, ppl)}." + (" You also told me: " + " ".join(f"\"{t}\"" for t in more[:3]) if more else "")
    return None


def own_name_to_my(question: str, items: List[Dict[str, Any]]) -> str:
    """'When is Sharada's dentist appointment?' → 'When is my dentist appointment?' when Sharada is the person."""
    ppl = build(items)
    for full in ppl.names:
        for name in {full, full.split()[0]}:
            question = re.sub(rf"\b{re.escape(name)}'s\b", "my", question)
    return question


def alias_terms(question: str, items: List[Dict[str, Any]]) -> str:
    """'When is Lalitha's eye check-up?' → 'When is Amma's eye check-up?' when the memory calls Lalitha 'Amma'."""
    ppl = build(items)
    for p in ppl.people.values():
        if not p.name or not p.aliases:
            continue
        first = p.name.split()[0]
        if re.search(rf"\b{re.escape(first)}\b", question) and not any(re.search(rf"\b{re.escape(first)}\b", t) for t in p.texts if "name is" not in t):
            question = re.sub(rf"\b{re.escape(first)}\b", sorted(p.aliases)[0], question)
    return question
