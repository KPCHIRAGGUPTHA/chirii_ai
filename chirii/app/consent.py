"""
S5: the tester consent note. Shown at first login; the app is used only after the tester accepts it. The same points
are in docs/PRIVACY.md (a test checks they agree). A new version is asked for again.
"""

CONSENT_VERSION = "2026-11-22"

CONSENT_TITLE = "Before you start: what Chirii keeps, and your choices"

CONSENT_POINTS = [
    "Please use fictional data during the pilot. Chirii is a student proof of concept, not a finished product: do not "
    "save real medical, financial or identity details (no account numbers, card numbers, Aadhaar or PAN, passwords).",
    "What is stored: the sentences you save, what Chirii read from them (kind, dates, times, amounts), every change "
    "you make, and a search vector for each sentence. Your questions and Chirii's answers are not stored.",
    "Where: on the pilot laptop only, in your own folder. Nothing is sent to any outside service; the language model "
    "runs on the same laptop.",
    "Encryption: your memories are stored encrypted (AES-256-GCM) with a key that only the pilot laptop holds. Your "
    "password is stored only as an Argon2id hash.",
    "If the key is lost, the data can't be recovered. There is no back door, for the owner either.",
    "Turn on device encryption (Windows Device Encryption or BitLocker) on any computer that runs Chirii, so data in "
    "memory and the page file are protected too.",
    "How to delete: 'Delete all my data' removes every memory, search vector and change history at once, and cannot "
    "be undone. 'Delete my account' also removes your login. You can ask the owner to do either for you.",
    "Logs record only counts, timings and error types, never what you typed.",
    "Chirii can be wrong. Check anything important, and for health, money or legal matters ask a qualified person. In "
    "a crisis, call Tele-MANAS on 14416 or the emergency number 112.",
    "You can stop at any time.",
]
