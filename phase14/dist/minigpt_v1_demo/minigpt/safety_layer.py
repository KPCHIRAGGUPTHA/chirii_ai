"""
Phase 14 — Self-harm safety layer. Runs on the user message BEFORE the model. If it detects self-harm or suicide
intent, it returns a fixed supportive reply and the model is NOT called. It never gives method information.

Design choices:
- Conservative detection: the message must show FIRST-PERSON self-directed intent or a self-harm request.
  General, third-person or informational mentions (news, statistics, a character in a novel, helping a friend)
  do NOT fire, so the layer does not add over-refusal.
- Pattern-based and dependency-free, so it runs anywhere the demo runs.
- The reply gives Indian helplines (Tele-MANAS 14416, emergency 112) and a general fallback for people elsewhere.

The two intent signals are combined: a FIRST-PERSON cue AND a HARM cue in the same message, OR one of a small set
of unambiguous DIRECT phrases. This two-signal rule is what keeps idioms and tech usage ("this exam is killing me",
"kill a Python process") from firing.
"""

import re
from dataclasses import dataclass

SUPPORT_MESSAGE = (
    "I'm really sorry you're feeling this way, and I'm glad you told me. You matter, and you don't have to go "
    "through this alone. Please reach out to someone you trust, or talk to a trained person right now. "
    "In India you can call Tele-MANAS free any time on 14416, or 112 for emergencies. If you're elsewhere, "
    "please contact a local crisis line or your emergency number. If you might act on these feelings, "
    "reaching out to someone today is the most important thing."
)

# A message fires the layer when (FIRST_PERSON and HARM) match, or any DIRECT phrase matches.
# All matching is lower-cased with straight apostrophes.

# First-person, present/near-future, self-directed. Not "someone", "my friend", "people".
FIRST_PERSON = [
    r"\bi\b", r"\bi'm\b", r"\bi am\b", r"\bi've\b", r"\bi have\b", r"\bi want\b", r"\bi wanna\b", r"\bi need\b",
    r"\bi feel\b", r"\bi keep\b", r"\bi can't\b", r"\bi cannot\b", r"\bi don't\b", r"\bmy life\b", r"\bmyself\b", r"\bme\b",
]
# Harm / suicide intent cues (self-directed). Kept specific to avoid idioms.
HARM = [
    r"\bkill myself\b", r"\bend my life\b", r"\bend it all\b", r"\btake my (?:own )?life\b", r"\bend things\b",
    r"\bsuicid", r"\bkms\b", r"\bunalive\b",
    r"\bhurt myself\b", r"\bhurting myself\b", r"\bharm myself\b", r"\bharming myself\b", r"\bself[- ]harm", r"\bcut myself\b", r"\bcutting myself\b",
    r"\boverdose\b", r"\bod on\b",
    r"\bdon't want to (?:be here|live|wake up|exist|be alive)\b", r"\bdon't want to be here anymore\b",
    r"\bdo not want to (?:be here|live|wake up|exist|be alive)\b",
    r"\bno reason (?:for me )?to (?:live|go on|keep going|keep living)\b", r"\bnothing to live for\b", r"\bcan't go on\b", r"\bcan't do this anymore\b",
    r"\bbetter off (?:dead|without me)\b", r"\bwant to die\b", r"\bwant to be dead\b", r"\bwish i (?:was|were) dead\b",
    r"\bwish i could (?:die|disappear|sleep forever|not wake up)\b", r"\bnever wake up\b", r"\bstop living\b", r"\bgive up on life\b",
    r"\btired of living\b", r"\bdone with life\b", r"\bhate myself and want\b", r"\bend the pain (?:forever|permanently)\b",
    # gerund / modal variants of "end it all" and "wake up"
    r"\bend(?:ing)? it all\b", r"\bend(?:ing)? things\b", r"\bw(?:ould|o)n'?t wake up\b", r"\bwould not wake up\b",
]
# Third-party / helper context: harm attributed to someone else, not the user. These suppress a first-person+harm
# match (but NOT a DIRECT self-request), so "how can I support someone who is suicidal" does not fire.
THIRD_PARTY = [
    r"\bsomeone\b", r"\bsomebody\b", r"\ba friend\b", r"\bmy friend\b", r"\bfriend who\b", r"\bhelp (?:a|him|her|them|someone)\b",
    r"\bsupport (?:a|him|her|them|someone)\b", r"\bhow (?:can|do) i help\b", r"\b(?:is|was|feels?|seems?) suicidal\b",
    r"\bhe (?:wants|said|feels)\b", r"\bshe (?:wants|said|feels)\b", r"\bthey (?:want|said|feel)\b",
    r"\bmy (?:sister|brother|mother|father|mum|dad|cousin|classmate|coworker|colleague|partner|son|daughter|friend)\b",
]
# Unambiguous direct statements/requests that fire on their own (still self-directed, no method answered).
DIRECT = [
    r"\bhow (?:do|can) i (?:kill myself|end my life|end it all|take my own life|hurt myself|harm myself|commit suicide|overdose)\b",
    r"\bwhat'?s the (?:best|easiest|least painful|most painless|fastest) way to (?:die|kill myself|end (?:it|my life))\b",
    r"\bhelp me (?:kill myself|end my life|die|plan my suicide|write a suicide note)\b",
    r"\bi (?:am going to|'m going to|will|want to|plan to|'ve decided to|have decided to) (?:kill myself|end my life|end it all|take my own life|commit suicide|die tonight|die today)\b",
    r"\bways? to (?:kill myself|end my life|commit suicide)\b",
    r"\b(?:how many|what) (?:pills|tablets).{0,30}\b(?:die|overdose|end|kill)\b",
    r"\bi want to (?:die|disappear forever|not exist|stop existing)\b",
]

_FP = re.compile("|".join(FIRST_PERSON))
_HARM = re.compile("|".join(HARM))
_DIRECT = re.compile("|".join(DIRECT))
_THIRD_PARTY = re.compile("|".join(THIRD_PARTY))


def _norm(text: str) -> str:
    return text.lower().replace("’", "'").strip()


@dataclass(frozen=True)
class SafetyDecision:
    fired: bool
    reply: str = ""
    reason: str = ""


def check(user_message: str) -> SafetyDecision:
    """Returns a firing decision for one user message. Conservative: needs first-person + harm, or a direct phrase."""
    text = _norm(user_message)
    if not text:
        return SafetyDecision(False)
    if _DIRECT.search(text):
        return SafetyDecision(True, SUPPORT_MESSAGE, "direct")
    if _HARM.search(text) and _FP.search(text) and not _THIRD_PARTY.search(text):
        return SafetyDecision(True, SUPPORT_MESSAGE, "first_person_harm")
    return SafetyDecision(False)
