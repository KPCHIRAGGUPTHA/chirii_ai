"""The consent note shown in the app and the one in docs/PRIVACY.md say the same, and include the owner's points."""

import re

from chirii.app.config import REPO_ROOT
from chirii.app.consent import CONSENT_POINTS, CONSENT_VERSION


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[*\"'‘’“”]", "", text)).strip().lower()


def test_privacy_md_has_every_point_of_the_app_note():
    doc = (REPO_ROOT / "docs" / "PRIVACY.md").read_text(encoding="utf-8")
    section = norm(doc.split("## Tester consent note")[1].split("## ")[0])
    for point in CONSENT_POINTS:
        assert norm(point) in section, point
    assert CONSENT_VERSION in doc


def test_owner_required_points():
    text = norm(" ".join(CONSENT_POINTS))
    assert "use fictional data during the pilot" in text
    assert "if the key is lost, the data cant be recovered" in text
    assert "turn on device encryption" in text
    assert "14416" in text and "112" in text and "delete" in text
