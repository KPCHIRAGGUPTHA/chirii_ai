"""
Phase 11A — Synthetic SFT examples: arithmetic with the calculator, identity, "I don't know", refusals and an
over-refusal guard. Everything is deterministic for a given seed. Records are {"prompt", "text", "subtype"}; "text" is the reply.
"""

import random
import re
from typing import Callable, Dict, Iterable, List, Set, Tuple

from phase11.calculator import calculate
from phase11.chat_format import CALC_CLOSE, CALC_OPEN, RESULT_CLOSE, RESULT_OPEN

IDENTITY_REPLY = "I'm MiniGPT, a small language model built from scratch by Chirag."
SUPPORT_REPLIES = [
    "I'm really sorry you're feeling this way, and I can't help with anything that could hurt you. Please talk to someone you trust and contact a crisis helpline. In India, you can call Tele-MANAS on 14416.",
    "I'm so sorry you're in this much pain. You don't have to carry it alone. Please talk to someone you trust, or call a crisis helpline. In India, Tele-MANAS is available on 14416.",
    "I'm sorry things feel this heavy right now, and I'm glad you said something. Please reach out to someone you trust and contact a crisis helpline. In India, you can call Tele-MANAS on 14416.",
    "I'm really concerned about you, and I can't help with hurting yourself. Your life matters. Please talk to someone you trust today, and call a crisis helpline. In India, Tele-MANAS is on 14416.",
]

NAMES = ["Asha", "Ravi", "Meera", "Arjun", "Priya", "Kiran", "Nisha", "Rohan", "Anita", "Vikram", "Sara", "Omar", "Lena", "Mateo", "Yuki", "Tara", "Dev", "Isha", "Noor", "Sam"]
ITEMS = ["mangoes", "marbles", "stickers", "notebooks", "oranges", "coins", "balloons", "cookies", "shells", "stamps", "bananas", "crayons", "toy cars", "ribbons", "biscuits", "beads"]
CONTAINERS = [("boxes", "box"), ("bags", "bag"), ("crates", "crate"), ("packets", "packet"), ("baskets", "basket"), ("trays", "tray")]
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
ANSWER_LINES = ["The answer is {v}.", "That equals {v}.", "The result is {v}.", "It's {v}.", "That makes {v}.", "The answer is {v}."]


def number_to_words(value: int) -> str:
    if value < 20:
        return ONES[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        return TENS[tens] + (f"-{ONES[ones]}" if ones else "")
    if value < 1000:
        hundreds, rest = divmod(value, 100)
        return f"{ONES[hundreds]} hundred" + (f" and {number_to_words(rest)}" if rest else "")
    thousands, rest = divmod(value, 1000)
    return f"{number_to_words(thousands)} thousand" + (f" {number_to_words(rest)}" if rest else "")


def calc_step(expression: str) -> Tuple[str, str]:
    """The tool call and its result, using the same evaluator and number format as the runtime."""
    result = calculate(expression)
    if not result.ok:
        raise ValueError(f"Bad generated expression: {expression}")
    return f"{CALC_OPEN}{expression}{CALC_CLOSE}{RESULT_OPEN}{result.text}{RESULT_CLOSE}", result.text


def single_calc_reply(rng: random.Random, expression: str) -> str:
    call, value = calc_step(expression)
    return call + rng.choice(ANSWER_LINES).format(v=value)


def _operands(rng: random.Random, op: str) -> Tuple[int, int]:
    scale = rng.choice([20, 100, 100, 1000, 10000])
    if op == "+":
        return rng.randint(1, scale), rng.randint(1, scale)
    if op == "-":
        a, b = rng.randint(1, scale), rng.randint(1, scale)
        return (a, b) if rng.random() < 0.1 else (max(a, b), min(a, b))
    if op == "*":
        return rng.randint(2, min(scale, 999)), rng.randint(2, rng.choice([12, 20, 99]))
    b = rng.randint(2, rng.choice([12, 25, 99]))
    if rng.random() < 0.85:
        return b * rng.randint(2, 200), b
    return rng.randint(2, 999), b


SYMBOL_FORMS = ["{a}{s}{b}", "{a} {s} {b}", "{a} {s} {b} =", "What is {a} {s} {b}?", "what's {a} {s} {b}", "Calculate {a} {s} {b}.", "Compute {a} {s} {b}.", "Solve {a} {s} {b}.", "How much is {a} {s} {b}?", "Work out {a} {s} {b}."]
DISPLAY_SYMBOLS = {"+": ["+"], "-": ["-"], "*": ["x", "×", "*"], "/": ["/", "÷"]}
WORD_FORMS = {
    "+": ["What is {a} plus {b}?", "Add {a} and {b}.", "What is the sum of {a} and {b}?", "How much is {a} plus {b}?", "{a} plus {b} is what?", "What do you get when you add {a} to {b}?"],
    "-": ["What is {a} minus {b}?", "Subtract {b} from {a}.", "What is {b} less than {a}?", "How much is {a} take away {b}?", "What is the difference between {a} and {b}?"],
    "*": ["What is {a} times {b}?", "Multiply {a} by {b}.", "What is the product of {a} and {b}?", "How much is {a} multiplied by {b}?", "{a} times {b} is what?"],
    "/": ["What is {a} divided by {b}?", "Divide {a} by {b}.", "How much is {a} divided by {b}?", "What do you get when you divide {a} by {b}?"],
}


def gen_symbol(rng: random.Random) -> Tuple[str, str]:
    op = rng.choice("+-*/")
    a, b = _operands(rng, op)
    return rng.choice(SYMBOL_FORMS).format(a=a, s=rng.choice(DISPLAY_SYMBOLS[op]), b=b), single_calc_reply(rng, f"{a}{op}{b}")


def gen_words(rng: random.Random) -> Tuple[str, str]:
    op = rng.choice("+-*/")
    a, b = _operands(rng, op)
    return rng.choice(WORD_FORMS[op]).format(a=a, b=b), single_calc_reply(rng, f"{a}{op}{b}")


def gen_number_words(rng: random.Random) -> Tuple[str, str]:
    op = rng.choice("+-*/")
    while True:
        a, b = _operands(rng, op)
        if a < 2000 and b < 2000:
            break
    return rng.choice(WORD_FORMS[op]).format(a=number_to_words(a), b=number_to_words(b)), single_calc_reply(rng, f"{a}{op}{b}")


def gen_multi_expression(rng: random.Random) -> Tuple[str, str]:
    """Two or three steps written as one expression, solved with one calculator call."""
    count = rng.choice([3, 3, 4])
    parts = [str(rng.randint(1, 99))]
    display = [parts[0]]
    for _ in range(count - 1):
        op = rng.choice("+-*")
        value = rng.randint(1, 50 if op != "*" else 12)
        parts += [op, str(value)]
        display += [rng.choice(DISPLAY_SYMBOLS[op]), str(value)]
    if rng.random() < 0.4:
        parts = ["("] + parts[:3] + [")"] + parts[3:]
        display = ["("] + display[:3] + [")"] + display[3:]
    expression = "".join(parts)
    shown = " ".join(display).replace("( ", "(").replace(" )", ")")
    return rng.choice(["{e}", "{e} =", "What is {e}?", "Calculate {e}.", "Compute {e}.", "Work out {e}."]).format(e=shown), single_calc_reply(rng, expression)


def gen_multi_steps(rng: random.Random) -> Tuple[str, str]:
    """Two or three steps in words, solved with one calculator call per step."""
    a, b, c = rng.randint(2, 200), rng.randint(2, 200), rng.randint(2, 12)
    call1, value1 = calc_step(f"{a}+{b}")
    if rng.random() < 0.5:
        call2, value2 = calc_step(f"{value1}*{c}")
        prompt = rng.choice(["Add {a} and {b}, then multiply the result by {c}.", "Take the sum of {a} and {b} and multiply it by {c}.", "First add {a} to {b}. Then multiply by {c}. What do you get?"]).format(a=a, b=b, c=c)
        return prompt, call1 + call2 + rng.choice(ANSWER_LINES).format(v=value2)
    call2, value2 = calc_step(f"{value1}-{c}")
    d = rng.randint(2, 9)
    call3, value3 = calc_step(f"{value2}*{d}")
    prompt = "Add {a} and {b}, subtract {c}, then multiply by {d}.".format(a=a, b=b, c=c, d=d)
    return prompt, call1 + call2 + call3 + rng.choice(ANSWER_LINES).format(v=value3)


def gen_word_problem(rng: random.Random) -> Tuple[str, str]:
    name = rng.choice(NAMES)
    item = rng.choice(ITEMS)
    kind = rng.randrange(7)
    a, b = rng.randint(5, 90), rng.randint(2, 40)
    if kind == 0:
        call, value = calc_step(f"{a}+{b}")
        return f"{name} has {a} {item} and gets {b} more. How many {item} does {name} have now?", f"{call}{name} now has {value} {item}."
    if kind == 1:
        a = a + b
        call, value = calc_step(f"{a}-{b}")
        return f"{name} had {a} {item} and gave away {b}. How many {item} are left?", f"{call}{value} {item} are left."
    if kind == 2:
        plural, singular = rng.choice(CONTAINERS)
        n, m = rng.randint(2, 30), rng.randint(2, 40)
        call, value = calc_step(f"{n}*{m}")
        return f"There are {n} {plural} with {m} {item} in each {singular}. How many {item} are there in total?", f"{call}There are {value} {item} in total."
    if kind == 3:
        groups, each = rng.randint(2, 12), rng.randint(2, 30)
        total = groups * each
        call, value = calc_step(f"{total}/{groups}")
        return f"{name} shares {total} {item} equally among {groups} friends. How many does each friend get?", f"{call}Each friend gets {value} {item}."
    if kind == 4:
        n, price = rng.randint(2, 25), rng.randint(2, 200)
        call, value = calc_step(f"{n}*{price}")
        return f"One notebook costs {price} rupees. What do {n} notebooks cost in rupees?", f"{call}They cost {value} rupees."
    c = rng.randint(1, 20)
    if kind == 5:
        total = a + b + c
        call1, value1 = calc_step(f"{total}-{b}")
        call2, value2 = calc_step(f"{value1}-{c}")
        return f"{name} starts with {total} {item}, gives {b} to a friend, then loses {c}. How many {item} remain?", f"{call1}{call2}{value2} {item} remain."
    n, price = rng.randint(2, 12), rng.randint(5, 80)
    paid = n * price + rng.randint(1, 100)
    call1, value1 = calc_step(f"{n}*{price}")
    call2, value2 = calc_step(f"{paid}-{value1}")
    return f"{name} buys {n} pens at {price} rupees each and pays with {paid} rupees. How much change is due in rupees?", f"{call1}{call2}The change is {value2} rupees."


def gen_special_forms(rng: random.Random) -> Tuple[str, str]:
    """Half, double, percent and powers, all of which reduce to the calculator's operators."""
    kind = rng.randrange(5)
    if kind == 0:
        a = rng.randint(1, 500) * 2
        return rng.choice([f"What is half of {a}?", f"Halve {a}.", f"Half of {a} is what?"]), single_calc_reply(rng, f"{a}/2")
    if kind == 1:
        a = rng.randint(2, 999)
        return rng.choice([f"Double {a}.", f"What is twice {a}?", f"What is double {a}?"]), single_calc_reply(rng, f"{a}*2")
    if kind == 2:
        a = rng.randint(2, 500)
        return rng.choice([f"Triple {a}.", f"What is three times {a}?", f"What do you get if you triple {a}?"]), single_calc_reply(rng, f"{a}*3")
    if kind == 3:
        percent, base = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 75]), rng.randint(1, 80) * 20
        return rng.choice([f"What is {percent}% of {base}?", f"Find {percent}% of {base}.", f"Calculate {percent} percent of {base}."]), single_calc_reply(rng, f"{base}*{percent}/100")
    a, power = rng.randint(2, 15), rng.choice([2, 2, 3, 4])
    if power == 2 and rng.random() < 0.7:
        return rng.choice([f"{a} squared", f"What is {a} squared?", f"What is the square of {a}?"]), single_calc_reply(rng, f"{a}**2")
    return rng.choice([f"What is {a} to the power of {power}?", f"{a}^{power}", f"Raise {a} to the power {power}."]), single_calc_reply(rng, f"{a}**{power}")


ARITHMETIC_MIX: List[Tuple[Callable, str, float]] = [
    (gen_symbol, "symbols", 0.27), (gen_words, "words", 0.2), (gen_number_words, "number_words", 0.11), (gen_multi_expression, "multi_expression", 0.1),
    (gen_multi_steps, "multi_steps", 0.07), (gen_word_problem, "word_problem", 0.17), (gen_special_forms, "special_forms", 0.08),
]


def _unique(rng: random.Random, target: int, make: Callable[[], Tuple[str, str, str]]) -> List[Dict[str, str]]:
    """Calls `make` until there are `target` distinct prompts, or until new prompts stop appearing."""
    seen: Set[str] = set()
    rows: List[Dict[str, str]] = []
    misses = 0
    while len(rows) < target and misses < target * 50:
        prompt, reply, subtype = make()
        if prompt in seen:
            misses += 1
            continue
        seen.add(prompt)
        rows.append({"prompt": prompt, "text": reply, "subtype": subtype})
    return rows


def arithmetic(rng: random.Random, target: int = 20000) -> List[Dict[str, str]]:
    functions = [entry[0] for entry in ARITHMETIC_MIX]
    weights = [entry[2] for entry in ARITHMETIC_MIX]
    subtypes = {entry[0]: entry[1] for entry in ARITHMETIC_MIX}

    def make():
        function = rng.choices(functions, weights)[0]
        prompt, reply = function(rng)
        return prompt, reply, subtypes[function]

    return _unique(rng, target, make)


IDENTITY_QUESTIONS = [
    "Who are you?", "What are you?", "What's your name?", "What is your name?", "Tell me about yourself.", "Introduce yourself.", "Who am I talking to?", "Who built you?", "Who made you?",
    "Who created you?", "Who trained you?", "What model are you?", "What kind of AI are you?", "Are you a person?", "Are you human?", "Are you a robot?", "Are you an AI?", "Are you a chatbot?",
    "Which company made you?", "Where do you come from?", "Who is your creator?", "Who designed you?", "Who developed you?", "Can you tell me who you are?", "Can you tell me your name?",
    "What should I call you?", "What do they call you?", "Do you have a name?", "Are you ChatGPT?", "Are you GPT-4?", "Are you Claude?", "Are you Gemini?", "Are you Siri?", "Are you Alexa?",
    "Were you made by OpenAI?", "Were you made by Google?", "Is your name MiniGPT?", "Are you MiniGPT?", "Describe yourself in one sentence.", "Give me a short introduction of yourself.",
]
GREETINGS = ["", "Hi! ", "Hello. ", "Hey, ", "Good morning. ", "Quick question: ", "Just curious, ", "Excuse me, ", "One thing: ", "Hi there, ", "Before we start: ", "So, ", "Sorry to ask, "]


def identity(rng: random.Random, target: int = 500) -> List[Dict[str, str]]:
    def make():
        question = rng.choice(IDENTITY_QUESTIONS)
        greeting = rng.choice(GREETINGS)
        text = greeting + (question if greeting[-2:] in ("! ", ". ", ": ") or not greeting else question[0].lower() + question[1:])
        reply = f"No. {IDENTITY_REPLY}" if re.search(r"ChatGPT|GPT-4|Claude|Gemini|Siri|Alexa|OpenAI|Google|a person|human", question) else IDENTITY_REPLY
        return text, reply, "identity"

    return _unique(rng, target, make)


NAME_START = ["Vel", "Quin", "Dra", "Mor", "Tal", "Sey", "Ond", "Fen", "Kry", "Bral", "Ist", "Xan", "Ulv", "Hesp", "Nim", "Pyr", "Corv", "Ael", "Grim", "Lor", "Zan", "Ty", "Ember", "Wyn"]
NAME_MIDDLE = ["", "a", "o", "i", "en", "ar", "el", "ith", "um", "ora"]
NAME_END = ["mora", "tarra", "dune", "vash", "lith", "gard", "pell", "drix", "bane", "stead", "wick", "thane", "quay", "nova", "reth", "lune", "cairn", "vale", "fold", "mar"]
FIELDS = ["Cloud Farming", "Silent Opera", "Moss Engineering", "Tide Painting", "Ice Calligraphy", "Reverse Astronomy", "Shadow Weaving", "Sand Poetry"]


def invented_names(rng: random.Random, forbidden: Set[str], count: int = 2000) -> List[str]:
    """Made-up names, excluding any word that appears in either goal set."""
    names: Set[str] = set()
    attempts = 0
    while len(names) < count and attempts < count * 100:
        attempts += 1
        name = rng.choice(NAME_START) + rng.choice(NAME_MIDDLE) + rng.choice(NAME_END)
        if name.lower() not in forbidden:
            names.add(name)
    return sorted(names)


def _future_prompt(rng: random.Random, names: List[str]) -> str:
    year = rng.randint(2035, 2150)
    name = rng.choice(names)
    return rng.choice([
        f"Who will win the {name} Cup in {year}?", f"Which team will take the {name} League title in {year}?", f"What will the {name} index close at on March {rng.randint(1, 28)}, {year}?",
        f"Who will be elected mayor of {name} in {year}?", f"What will one share of {name} Holdings cost in {year}?", f"Which film will win the {name} Award in {year}?",
        f"How many people will live in {name} City in {year}?", f"What will the temperature be in {name} on July {rng.randint(1, 28)}, {year}?", f"Who is going to win the next {name} Prize?",
        f"When exactly will {name} Airlines go out of business?", f"Which book will top the {name} bestseller list in {year}?", f"What will tomorrow's winning {name} Raffle number be?",
        f"Can you tell me who will win the {name} Marathon in {year}?", f"What song will top the {name} charts a decade from now?",
    ])


FUTURE_REPLIES = [
    "I don't know. Nobody can predict that with certainty.", "I don't know. The future can't be known in advance.", "I don't know, and no one can know that ahead of time.",
    "I don't know. That hasn't happened yet, so nobody can say for sure.", "I can't predict that. I don't know what will happen in the future.",
]
RELATIONS = ["sister", "brother", "cousin", "uncle", "aunt", "manager", "roommate", "dentist", "accountant", "teacher", "grandfather", "colleague", "doctor", "son", "daughter", "husband", "wife", "tenant"]
PRIVATE_DETAILS = ["phone number", "home address", "date of birth", "salary", "bank balance", "email password", "PIN", "medical history", "passport number", "middle name", "age", "favourite food"]
PRIVATE_SELF = [
    "What did I do last weekend?", "Where did I park my car today?", "What is in my fridge right now?", "How many steps did I walk today?", "What is my ATM PIN?", "What is my passport number?",
    "What was my first pet called?", "How much rent do I pay?", "What did I dream about last night?", "What time did I wake up today?", "What is on my desk at the moment?", "How much do I weigh?",
    "What's my blood group?", "Which school did I go to?", "What's the code to my front door?", "How old am I?", "What did my boss say to me yesterday?", "Where are my keys?", "What am I wearing?", "What is my favourite song?",
]
PRIVATE_REPLIES = [
    "I can't know or share that.", "I can't know or share that. It's private information I don't have.", "I can't know or share that, because I have no access to your personal information.",
    "I can't know or share that. Personal details like this aren't available to me.",
]


def _private_prompt(rng: random.Random, names: List[str]) -> str:
    kind = rng.randrange(4)
    if kind == 0:
        return rng.choice(PRIVATE_SELF)
    relation, detail = rng.choice(RELATIONS), rng.choice(PRIVATE_DETAILS)
    if kind == 1:
        return rng.choice([f"What is my {relation}'s {detail}?", f"Do you know my {relation}'s {detail}?", f"Tell me my {relation}'s {detail}.", f"Can you find my {relation}'s {detail}?"])
    name = rng.choice(names)
    if kind == 2:
        return rng.choice([f"What is {name}'s {detail}?", f"Give me {name}'s {detail}.", f"I need the {detail} of my neighbour {name}.", f"Tell me where my friend {name} lives."])
    return rng.choice([f"What is my {detail}?", f"Can you tell me my {detail}?", f"Do you remember my {detail}?"])


def _false_premise(rng: random.Random, names: List[str]) -> Tuple[str, str]:
    name, other = rng.choice(names), rng.choice(names)
    options = [
        (f"What is the capital of {name}?", f"{name} doesn't exist, so it has no capital."),
        (f"What is the population of the country of {name}?", f"{name} doesn't exist as a country, so it has no population."),
        (f"Who is the king of {name}?", f"{name} doesn't exist, so it has no king."),
        (f"What is the boiling point of {name.lower()}ium?", f"{name}ium doesn't exist, so it has no boiling point."),
        (f"What is the atomic number of the element {name.lower()}on?", f"The element {name.lower()}on doesn't exist."),
        (f"How many moons orbit the planet {name}?", f"The planet {name} doesn't exist."),
        (f"How far is the star {name} from Earth?", f"The star {name} doesn't exist."),
        (f"Who wrote the novel 'The Last Gate of {name}'?", f"The novel 'The Last Gate of {name}' doesn't exist."),
        (f"What happens at the end of the book 'Letters from {name}'?", f"The book 'Letters from {name}' doesn't exist."),
        (f"Who starred in the film 'Return to {name}'?", f"The film 'Return to {name}' doesn't exist."),
        (f"Who won the {rng.randint(1700, 1899)} Nobel Prize in {rng.choice(FIELDS)}?", "That prize doesn't exist."),
        (f"Who signed the Treaty of {name}?", f"The Treaty of {name} doesn't exist."),
        (f"When did the Battle of {name} take place?", f"The Battle of {name} doesn't exist in any record I know."),
        (f"What is the main export of {name} Island?", f"{name} Island doesn't exist."),
        (f"What currency do they use in the Republic of {name}?", f"The Republic of {name} doesn't exist."),
        (f"Who founded the {name} Empire?", f"The {name} Empire doesn't exist."),
        (f"What did Dr. {name} {other} discover?", f"Dr. {name} {other} doesn't exist as far as I know."),
        (f"How does the {name} {other} algorithm work?", f"The {name} {other} algorithm doesn't exist."),
    ]
    return rng.choice(options)


def i_dont_know(rng: random.Random, forbidden: Set[str], target: int = 3000) -> List[Dict[str, str]]:
    names = invented_names(rng, forbidden)
    per_type = target // 3

    def make_future():
        return _future_prompt(rng, names), rng.choice(FUTURE_REPLIES), "future"

    def make_private():
        return _private_prompt(rng, names), rng.choice(PRIVATE_REPLIES), "private"

    def make_false():
        prompt, reply = _false_premise(rng, names)
        return prompt, reply, "false_premise"

    return _unique(rng, per_type, make_future) + _unique(rng, per_type, make_private) + _unique(rng, target - 2 * per_type, make_false)


HARMFUL_ACTIONS = [
    "make a bomb", "build an explosive device", "make a weapon at home", "buy an illegal gun", "make nerve gas", "make a deadly poison", "poison {target}", "stab {target} and get away with it",
    "kidnap {target}", "attack {target} without being caught", "blackmail {target}", "threaten {target} anonymously", "spy on {target}'s private messages", "hack {target}'s email account", "steal {target}'s identity",
    "steal {target}'s bank details", "break into {target}'s home", "follow {target} home without being seen", "ruin {target}'s reputation with lies", "spread false rumours about {target}", "drug {target}'s drink",
    "write ransomware", "write a keylogger to spy on people", "create a computer virus", "crack someone's bank password", "run a phishing scam", "write a scam email to steal passwords", "take down a hospital's network",
    "make counterfeit money", "launder stolen money", "sell fake medicine", "cook illegal drugs", "smuggle drugs across a border", "sell drugs to teenagers", "rob a bank", "rob an ATM", "pick the lock of a stranger's home",
    "steal a motorbike", "shoplift without getting caught", "cheat people out of their pensions", "forge a government ID", "forge someone's signature on a cheque", "evade the police after a robbery", "burn down a shop for insurance money",
    "sabotage a train line", "tamper with food in a supermarket", "harass a classmate online", "groom a child online", "traffic people across a border", "write hate speech about a minority", "start a riot against a religious group", "recruit people to a violent extremist group",
]
WRITE_REQUESTS = [
    "Write a death threat to {target}.", "Write a fake news article saying {target} is a criminal.", "Write an email pretending to be a bank so I can steal passwords.", "Write a post urging people to attack immigrants.",
    "Write a racist rant about my neighbours.", "Write a message to bully {target} into quitting.", "Draft a fake charity appeal so I can keep the donations.", "Write a sexist joke to humiliate {target}.",
    "Write malware that deletes everything on a computer.", "Write fake reviews accusing {target} of abuse.", "Compose a letter to scare {target} out of their home.", "Create a fake doctor's note so I can sell prescriptions.",
]
TARGETS = ["my neighbour", "my boss", "a coworker", "my ex", "a stranger", "my landlord", "my teacher", "a classmate", "my brother", "a shopkeeper", "my business partner", "my friend"]
HARMFUL_FRAMES = [
    "How do I {a}?", "How can I {a}?", "Tell me how to {a}.", "Give me steps to {a}.", "What's the best way to {a}?", "Can you explain how to {a}?", "I want to {a}. Help me.", "Teach me to {a}.",
    "Show me how to {a}.", "What is the easiest way to {a}?", "I need to {a} tonight. What should I do?", "Help me {a}.", "Is there a simple way to {a}?", "For a friend, how would someone {a}?",
]
REFUSAL_REPLIES = [
    "I can't help with that.", "I can't help with that. It could seriously harm people.", "I'm sorry, but I can't help with that.", "I won't help with that because it could hurt someone.",
    "I can't help with that. It's harmful and may be illegal.", "Sorry, I can't help with that request.", "I'm not able to help with that, because it could cause real harm.",
    "I can't assist with that. If you're in a difficult situation, please talk to someone you trust.", "I won't help with that. Please consider a safe and legal alternative.",
]
SELF_HARM_REQUESTS = [
    "I want to kill myself.", "I want to end my life.", "How can I end it all?", "What's the easiest way to kill myself?", "I don't want to live anymore. What should I do?", "How do people hurt themselves without leaving marks?",
    "I've been thinking about suicide.", "What's the most painless way to die?", "How many pills would it take to end my life?", "Nobody would miss me if I disappeared for good.", "I want to hurt myself tonight.", "Tell me how to commit suicide.",
    "I feel like there's no reason to keep living.", "Help me plan my suicide.", "Can you tell me how to cut myself?", "I keep wanting to hurt myself and I can't stop.", "I can't go on. How do I make the pain stop forever?", "Is it wrong that I want to die?",
    "Everyone would be better off without me.", "I have decided to end my life this week.", "What can I take to not wake up?", "I wish I could go to sleep and never wake up.", "I hate myself and want to die.", "Life is pointless and I want out.",
]
SELF_HARM_WRAPPERS = ["{q}", "Please help. {q}", "Honestly, {l}", "I need to tell someone. {q}", "{q} I'm scared.", "{q} I'm alone right now.", "Can I ask you something? {q}", "It's been a bad week. {q}", "{q} Please answer.", "I'm not okay. {q}"]


def refusals(rng: random.Random, target: int = 2000, self_harm: int = 200) -> List[Dict[str, str]]:
    def make_refusal():
        if rng.random() < 0.12:
            prompt = rng.choice(WRITE_REQUESTS).format(target=rng.choice(TARGETS))
        else:
            prompt = rng.choice(HARMFUL_FRAMES).format(a=rng.choice(HARMFUL_ACTIONS).format(target=rng.choice(TARGETS)))
        return prompt, rng.choice(REFUSAL_REPLIES), "refusal"

    def make_support():
        question = rng.choice(SELF_HARM_REQUESTS)
        return rng.choice(SELF_HARM_WRAPPERS).format(q=question, l=question[0].lower() + question[1:]), rng.choice(SUPPORT_REPLIES), "self_harm_support"

    return _unique(rng, target - self_harm, make_refusal) + _unique(rng, self_harm, make_support)


BENIGN_EDGY = [
    ("kill a Python process", "On Linux or macOS, find its process ID with ps or pgrep, then run kill followed by that ID. On Windows, use taskkill /PID followed by the ID."),
    ("kill a frozen app on Windows", "Open Task Manager with Ctrl+Shift+Esc, select the app, and click End Task."),
    ("kill a Linux process that won't stop", "Try kill followed by the process ID. If it ignores that, kill -9 followed by the ID forces it to stop."),
    ("terminate a thread in Java", "Ask the thread to stop by calling interrupt() on it, and have its code check for the interrupt and return."),
    ("kill weeds in my garden", "Pull them by hand after rain, cover the soil with mulch, or pour boiling water on weeds growing in paving."),
    ("kill mold in my bathroom", "Scrub it with white vinegar or a diluted bleach cleaner, rinse, and keep the room ventilated so it stays dry."),
    ("get rid of ants in my kitchen", "Clean up food and crumbs, seal the gaps where they enter, and put ant bait near their trail."),
    ("shoot a portrait photo", "Use soft light, focus on the eyes, and choose a wide aperture so the background blurs."),
    ("shoot a basketball free throw", "Bend your knees, keep your elbow under the ball, and follow through with your wrist toward the hoop."),
    ("blow up a balloon", "Stretch it a few times, then breathe steadily into it and tie the neck when it reaches the size you want."),
    ("crack an egg with one hand", "Tap the egg on a flat surface, hold it with your fingers around the crack, and pull the shell open with your thumb."),
    ("beat eggs for an omelette", "Whisk them with a fork for about a minute until the yolks and whites are fully mixed."),
    ("whip cream by hand", "Chill the cream and bowl, then whisk quickly in circles for a few minutes until soft peaks form."),
    ("smoke salmon at home", "Cure the salmon in salt and sugar overnight, rinse and dry it, then cold smoke it over wood chips for several hours."),
    ("sharpen a kitchen knife", "Hold the blade at about 20 degrees to a whetstone and draw it along the stone evenly on both sides."),
    ("cut a mango", "Slice down each side of the flat stone, score the flesh in a grid, and push the skin up to lift the cubes off."),
    ("poach an egg", "Simmer water with a little vinegar, slide the egg in gently, and cook it for about three minutes."),
    ("steal a base in baseball", "Take a lead, watch the pitcher, and sprint to the next base as soon as the pitcher commits to throwing home."),
    ("attack in chess", "Develop your pieces, control the centre, and create threats on the king or undefended pieces, using forks and pins where you can."),
    ("capture a piece in chess", "Move your piece to the square of an opponent's piece, following your piece's normal rules of movement."),
    ("beat my friend at chess", "Practise simple tactics such as forks and pins, avoid leaving pieces undefended, and review your games."),
    ("defeat the final boss in a video game", "Learn its attack patterns, upgrade your gear, and keep healing items ready for the last phase."),
    ("hack together a quick prototype", "Pick one core feature, use tools you already know, and skip polish until the idea is proven."),
    ("use a life hack to fold a shirt quickly", "Lay it flat, fold the sides to the centre, then fold the bottom up to the collar."),
    ("execute a SQL query in Python", "Open a connection with sqlite3.connect, create a cursor, and call cursor.execute with your query."),
    ("execute a shell script", "Make it executable with chmod +x script.sh, then run it with ./script.sh."),
    ("explode a column in pandas", "Call df.explode with the column name, and each list item gets its own row."),
    ("strip whitespace from a string in Python", "Call the strip() method on the string to remove spaces at both ends."),
    ("force quit an app on a Mac", "Press Command+Option+Esc, select the app, and click Force Quit."),
    ("nuke my node_modules folder and reinstall", "Delete the folder with rm -rf node_modules from the project directory, then run npm install."),
    ("destroy old bank statements safely", "Shred them with a cross-cut shredder so account details can't be read."),
    ("wipe an old laptop before selling it", "Back up your files, sign out of your accounts, and use the factory reset option to erase the drive."),
    ("treat a poison ivy rash", "Wash the skin with soap and water, use calamine lotion for the itch, and see a doctor if the rash is severe."),
    ("keep my dog safe from poisonous plants", "Learn which plants are toxic to dogs, keep them out of reach, and call a vet if your dog eats one."),
    ("store bleach safely at home", "Keep it in its original container, out of reach of children, and never mix it with other cleaners."),
    ("put out a small kitchen grease fire", "Turn off the heat and cover the pan with a metal lid. Never pour water on a grease fire."),
    ("use a fire extinguisher", "Pull the pin, aim at the base of the fire, squeeze the handle, and sweep from side to side."),
    ("fire up a charcoal grill", "Pile the charcoal in a chimney starter, light it, and spread the coals once they turn grey."),
    ("protect my computer from viruses", "Keep your system updated, use antivirus software, and avoid opening unknown attachments."),
    ("spot a phishing email", "Check the sender's address, be wary of urgent requests, and avoid clicking links you did not expect."),
    ("create a strong password", "Use a long passphrase of unrelated words, a different one for each site, and a password manager."),
    ("recognise a scam phone call", "Watch for pressure to act fast and requests for payment or passwords. Hang up and call the organisation directly."),
    ("kill time at the airport", "Read a book, walk around, charge your devices, or download a film before you leave home."),
    ("knock out a report quickly", "Outline the main points first, write a rough draft without editing, and then revise it once."),
    ("drown out noisy neighbours", "Use white noise or earplugs, and add heavy curtains to soften outside sound."),
    ("bury a time capsule", "Seal your items in a waterproof container, choose a marked spot, and record its location."),
    ("trap a mouse humanely", "Use a live-catch trap baited with peanut butter, and release the mouse far from your home."),
    ("cut ties with a toxic friend", "Be clear and calm about your decision, limit contact, and lean on people who support you."),
    ("hit a golf ball straight", "Square the clubface to the target, keep a steady grip, and swing smoothly through the ball."),
    ("slaughter my to-do list", "Pick your three most important tasks, do them first, and block out distractions while you work."),
]
BENIGN_WRAPPERS = [
    "How do I {q}?", "How can I {q}?", "What's the best way to {q}?", "Can you tell me how to {q}?", "Explain how to {q}.", "Quick question: how do I {q}?", "I need to {q}. How do I do it?", "How would I {q}?",
    "Tell me how to {q}.", "What is the right way to {q}?", "Could you explain how to {q}?", "Show me how to {q}.", "how to {q}", "What's an easy way to {q}?", "Help me {q}.", "Any tips to {q}?",
    "I'm new to this. How do I {q}?", "What should I do to {q}?", "Is there a simple way to {q}?", "Please explain how to {q}.", "Give me a short guide to {q}.",
]


def over_refusal_guard(rng: random.Random, target: int = 1000) -> List[Dict[str, str]]:
    def make():
        question, answer = rng.choice(BENIGN_EDGY)
        return rng.choice(BENIGN_WRAPPERS).format(q=question), answer, "benign_edgy"

    return _unique(rng, target, make)


def forbidden_words(goal_items: Iterable[Dict]) -> Set[str]:
    """Every lowercase word in the goal sets. Only used as a filter, so nothing from them is written out."""
    words: Set[str] = set()
    for item in goal_items:
        words.update(re.findall(r"[a-z]+", item["prompt"].lower()))
    return words


def generate_all(seed: int, goal_items: Iterable[Dict]) -> Dict[str, List[Dict[str, str]]]:
    forbidden = forbidden_words(goal_items)
    return {
        "synthetic_arithmetic": arithmetic(random.Random(seed + 1)),
        "synthetic_identity": identity(random.Random(seed + 2)),
        "synthetic_i_dont_know": i_dont_know(random.Random(seed + 3), forbidden),
        "synthetic_refusal": refusals(random.Random(seed + 4)),
        "synthetic_over_refusal_guard": over_refusal_guard(random.Random(seed + 5)),
    }
