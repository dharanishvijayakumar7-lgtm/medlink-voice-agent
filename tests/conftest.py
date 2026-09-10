"""Shared pytest setup.

`src/config.py` loads `.env.local`, so a developer's local settings would
otherwise change test outcomes. Anything here that a developer is likely to
relax locally must be pinned back to its shipped default, so the suite always
asserts production behaviour.
"""

import pytest

from config import settings


@pytest.fixture(autouse=True)
def _production_consent_default(monkeypatch):
    """Always test with consent enforced.

    `MEDLINK_REQUIRE_CONSENT=false` is a local-development convenience (it lets
    you see stored data before the spoken consent flow exists). The privacy
    tests must not silently pass just because a developer set it.
    """
    monkeypatch.setattr(settings, "require_consent", True)
