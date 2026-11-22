"""
S6: long lists are summarised (owner's plan, after the 500-item scale check: "complete but too long to be useful").

A date-window or plan list with more than LONG entries is shown as: the header, one summary line (how many, the
dates it spans, how many clashes), the first SHOWN entries, every total line, and "…and N more". The complete list,
exactly as code built it, is kept as the answer's full text and shown with "Show all". Nothing is dropped silently:
the summary says how many entries are not shown, and the totals are always shown.
"""

import re
from typing import Optional

LONG = 12
SHOWN = 5
DAY = re.compile(r"^- ((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) \d{1,2} [A-Z][a-z]+)\b")


def summarise(text: str) -> Optional[str]:
    """The short form of a long list, or None when the list is short enough to show whole."""
    lines = text.split("\n")
    entries = [i for i, line in enumerate(lines) if line.startswith("- ") and not line.startswith("- Clash")]
    clashes = [i for i, line in enumerate(lines) if line.startswith("- Clash")]
    if len(entries) + len(clashes) <= LONG:
        return None
    shown = set(entries[:SHOWN])
    days = [m.group(1) for i in entries if (m := DAY.match(lines[i]))]
    about = f"{len(entries)} items"
    if len(days) == len(entries) and days[0] != days[-1]:
        about += f", from {days[0]} to {days[-1]}"
    if clashes:
        about += f", with {len(clashes)} clash{'es' if len(clashes) > 1 else ''} (see all)"
    about += f". The first {SHOWN}:"
    out, said = [], False
    for i, line in enumerate(lines):
        if i in entries or i in clashes:
            if not said:
                out.append(about)
                said = True
            if i in shown:
                out.append(line)
            elif len(entries) > SHOWN and i == entries[SHOWN]:
                out.append(f"- …and {len(entries) - SHOWN} more. Tap \"Show all\" to see the full list.")
            continue
        if line.rstrip().endswith(":") and said and not any(j in shown for j in range(i + 1, len(lines))
                                                             if j in entries) and not _total_follows(lines, i):
            continue                                   # a section heading whose entries are all hidden
        out.append(line)
    return "\n".join(out)


def _total_follows(lines, i: int) -> bool:
    return any(lines[j].lower().startswith("total") for j in range(i + 1, len(lines)))
