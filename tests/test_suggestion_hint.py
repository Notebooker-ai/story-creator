"""The manifest must own its ``suggestion_hint``.

The host's "Suggest" button on the *additional instructions* field prefers a
manifest-declared hint over its own transitional fallback map, so this creator
has to ship one — and it has to be specific to this artifact, not boilerplate.
"""

from __future__ import annotations

from story_creator import StoryCreator


def test_manifest_declares_a_suggestion_hint():
    hint = StoryCreator().manifest.suggestion_hint
    assert isinstance(hint, str) and hint.strip()


def test_suggestion_hint_is_specific_to_this_artifact():
    hint = StoryCreator().manifest.suggestion_hint
    # a noun phrase naming what the instruction decides, not an imperative
    assert not hint[0].isupper()
    assert not hint.endswith(".")
    assert len(hint) > 40
    for token in ['narrative angle', 'tone']:
        assert token in hint.lower(), token
