"""Tests for StoryCreator using stubbed models (no network).

Covers the sanitizer's security properties, adventure graph validation, and
full generate runs: linear picture book, branching adventure, diffusion
backgrounds via a fake image role, and graceful degradation when the image
role is absent.
"""

from __future__ import annotations

import json
import tempfile

import pytest
from open_notebook_creator_sdk import ContentBundle, CreationRequest, ModelRole
from open_notebook_creator_sdk.schemas import validate_artifact_data
from open_notebook_creator_sdk.testing import assert_creator_compliant

from story_creator import StoryCreator, _validate_graph
from story_creator.svgkit import (
    compose_page,
    extract_symbols,
    sanitize_fragment,
    used_symbol_ids,
)

# --- fakes -------------------------------------------------------------------


class _FakeResp:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def __init__(self, payload: str):
        self._payload = payload

    async def ainvoke(self, _prompt):
        return _FakeResp(self._payload)


class _QueueRole(ModelRole):
    """create_language cycles through payloads (last one repeating)."""

    payloads: list = []
    calls: int = 0

    def create_language(self, **_):
        i = min(self.calls, len(self.payloads) - 1)
        self.calls += 1
        return _FakeLLM(self.payloads[i])


_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d4944415478da63fcffff3f0300050201cfa06b7a0000000049454e44ae426082"
)


class _FakeImageModel:
    async def agenerate_image(self, prompt: str, size: str = "1024x1024") -> bytes:
        assert "no characters" in prompt.lower()
        return _TINY_PNG


class _ImageRole(ModelRole):
    def create_image(self, **_):
        return _FakeImageModel()


class _BrokenImageModel:
    async def agenerate_image(self, prompt: str, size: str = "1024x1024") -> bytes:
        raise RuntimeError("provider down")


class _BrokenImageRole(ModelRole):
    def create_image(self, **_):
        return _BrokenImageModel()


# --- payload builders --------------------------------------------------------

_DEFS = """<defs>
<symbol id="char-fox" viewBox="0 0 200 200">
  <ellipse cx="100" cy="120" rx="60" ry="70" fill="#e8a04c"/>
  <circle cx="100" cy="60" r="40" fill="#e8a04c"/>
  <polygon points="70,30 85,55 55,55" fill="#c2543a"/>
</symbol>
<symbol id="prop-star" viewBox="0 0 200 200">
  <polygon points="100,10 120,80 190,80 130,120 150,190 100,145 50,190 70,120 10,80 80,80" fill="#f4e9d8"/>
</symbol>
</defs>"""

_SCENE = """<rect x="0" y="0" width="1600" height="700" fill="#5b7fa6"/>
<rect x="0" y="700" width="1600" height="300" fill="#2d4a3e"/>
<g class="char"><use href="#char-fox" x="600" y="500" width="300" height="300"/></g>"""


def _story(story_type: str, pages):
    return json.dumps(
        {
            "title": "The Curious Fox",
            "dedication": None,
            "moral": "Curiosity feeds the mind." if story_type == "fable" else None,
            "palette": ["#2d4a3e", "#e8a04c", "#c2543a", "#f4e9d8", "#5b7fa6"],
            "characters": [
                {"id": "fox", "name": "Fig", "description": "small orange fox, red ears"}
            ],
            "settings": [
                {"id": "forest", "name": "The Forest", "description": "deep green pines"}
            ],
            "start_page_id": pages[0]["id"],
            "pages": pages,
        }
    )


def _linear_pages(n):
    return [
        {
            "id": f"p{i}",
            "text": f"Page {i} of the tale.",
            "setting_id": "forest",
            "character_ids": ["fox"],
            "scene": "Fig walks under the pines.",
            "choices": [],
            "is_ending": i == n,
        }
        for i in range(1, n + 1)
    ]


_ADV_PAGES = [
    {"id": "start", "text": "A fork in the path.", "setting_id": "forest",
     "character_ids": ["fox"], "scene": "Fig at a fork.",
     "choices": [{"text": "Go left", "target_page_id": "left"},
                  {"text": "Go right", "target_page_id": "right"}],
     "is_ending": False},
    {"id": "left", "text": "A quiet glade. The end.", "setting_id": "forest",
     "character_ids": ["fox"], "scene": "A glade.", "choices": [], "is_ending": True},
    {"id": "right", "text": "A river. The end.", "setting_id": "forest",
     "character_ids": ["fox"], "scene": "A river.", "choices": [], "is_ending": True},
    {"id": "orphan", "text": "Nobody comes here.", "setting_id": "forest",
     "character_ids": [], "scene": "", "choices": [], "is_ending": True},
]


def _request(td, payloads, config, image_role=None):
    models = {"text": _QueueRole(provider="fake", model="fake", payloads=payloads)}
    if image_role is not None:
        models["image"] = image_role
    return CreationRequest(
        content=ContentBundle(
            text="Foxes are curious animals that explore forests.",
            sources=[{"id": "source:a", "title": "Fox Facts"}],
        ),
        config=config,
        models=models,
        output_dir=td,
        artifact_id="art-1",
    )


# --- static ------------------------------------------------------------------


def test_static_compliance():
    assert_creator_compliant(StoryCreator())


def test_manifest_declares_optional_image_role_and_view():
    m = StoryCreator().manifest
    roles = {r.key: r for r in m.model_roles}
    assert roles["text"].required is True
    assert roles["image"].required is False and roles["image"].kind == "image"
    assert m.view is not None and m.view.entry == "view/index.html"


# --- sanitizer ---------------------------------------------------------------


def test_sanitizer_strips_script_and_events():
    bad = (
        '<script>alert(1)</script>'
        '<rect x="0" y="0" width="10" height="10" fill="#112233" onclick="evil()"/>'
        '<circle cx="5" cy="5" r="2" fill="#112233"/>'
    )
    out = sanitize_fragment(bad, set())
    assert out is not None
    assert "script" not in out and "onclick" not in out
    assert "<circle" in out  # element with a handler is dropped, clean one kept
    assert "<rect" not in out


def test_sanitizer_blocks_external_and_unknown_use():
    bad = (
        '<use href="https://evil.example/x.svg#a"/>'
        '<use href="#char-ghost"/>'
        '<use href="#char-fox" x="1" y="2"/>'
    )
    out = sanitize_fragment(bad, {"char-fox"})
    assert out is not None
    assert out.count("<use") == 1 and 'href="#char-fox"' in out


def test_sanitizer_blocks_image_foreignobject_and_css_url():
    bad = (
        '<image href="https://evil.example/a.png"/>'
        '<foreignObject><div>hi</div></foreignObject>'
        '<rect x="0" y="0" width="9" height="9" fill="url(#grad)"/>'
        '<rect x="0" y="0" width="9" height="9" fill="#abcdef"/>'
    )
    out = sanitize_fragment(bad, set())
    assert out is not None
    assert "image" not in out and "foreignObject" not in out and "url(" not in out


def test_sanitizer_rejects_garbage():
    assert sanitize_fragment("<rect unterminated", set()) is None
    assert sanitize_fragment("", set()) is None


def test_extract_symbols_and_compose():
    syms = dict(extract_symbols(_DEFS))
    assert set(syms) == {"char-fox", "prop-star"}
    clean = sanitize_fragment(_SCENE, set(syms))
    assert clean is not None
    page = compose_page(clean, syms, 'A fox <in> the "woods"')
    assert page.startswith("<svg") and 'viewBox="0 0 1600 1000"' in page
    assert "char-fox" in page and "prop-star" not in page  # only used symbols inlined
    assert "<" not in page.split('aria-label="', 1)[1].split('"', 1)[0]


def test_used_symbol_ids():
    assert used_symbol_ids(_SCENE) == {"char-fox"}


# --- graph validation --------------------------------------------------------


def test_graph_prunes_unreachable_and_broken_choices():
    pages = [dict(p) for p in _ADV_PAGES]
    pages[0]["choices"] = pages[0]["choices"] + [
        {"text": "Secret door", "target_page_id": "missing"}
    ]
    kept, warnings = _validate_graph(pages, "start", is_adventure=True)
    ids = {p["id"] for p in kept}
    assert ids == {"start", "left", "right"}
    assert any("unreachable" in w.lower() for w in warnings)
    assert any("broken choice" in w.lower() for w in warnings)


def test_graph_linear_drops_choices():
    pages = [dict(p, choices=[{"text": "x", "target_page_id": "p1"}]) for p in _linear_pages(3)]
    kept, _ = _validate_graph(pages, "p1", is_adventure=False)
    assert all(not p["choices"] for p in kept)


# --- full runs ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_picture_book_linear():
    with tempfile.TemporaryDirectory() as td:
        payloads = [_story("picture-book", _linear_pages(4)), _DEFS, _SCENE]
        result = await StoryCreator().generate(
            _request(td, payloads, {"story_type": "picture-book", "num_pages": 4})
        )
        assert result.status == "SUCCESS", result.errors
        data = validate_artifact_data("story.v1", result.data)
        assert data.story_type == "picture-book"
        assert len(data.pages) == 4
        assert all(p.svg and p.svg.startswith("<svg") for p in data.pages)
        assert all(not p.choices for p in data.pages)
        assert any(f.label == "book" for f in result.files)
        book = open(f"{td}/story-book.html").read()
        assert "The Curious Fox" in book and "char-fox" in book


@pytest.mark.asyncio
async def test_generate_adventure_branching():
    with tempfile.TemporaryDirectory() as td:
        payloads = [_story("adventure", _ADV_PAGES), _DEFS, _SCENE]
        result = await StoryCreator().generate(
            _request(td, payloads, {"story_type": "adventure", "num_pages": 6})
        )
        assert result.status == "SUCCESS", result.errors
        data = validate_artifact_data("story.v1", result.data)
        assert data.start_page_id == "start"
        ids = {p.id for p in data.pages}
        assert "orphan" not in ids  # pruned
        start = next(p for p in data.pages if p.id == "start")
        assert {c.target_page_id for c in start.choices} == {"left", "right"}
        assert sum(1 for p in data.pages if p.is_ending) == 2
        book = open(f"{td}/story-book.html").read()
        assert "turn to page" in book


@pytest.mark.asyncio
async def test_generate_fable_carries_moral():
    with tempfile.TemporaryDirectory() as td:
        payloads = [_story("fable", _linear_pages(4)), _DEFS, _SCENE]
        result = await StoryCreator().generate(
            _request(td, payloads, {"story_type": "fable", "num_pages": 4})
        )
        data = validate_artifact_data("story.v1", result.data)
        assert data.moral == "Curiosity feeds the mind."


@pytest.mark.asyncio
async def test_generate_with_diffusion_backgrounds():
    with tempfile.TemporaryDirectory() as td:
        payloads = [_story("picture-book", _linear_pages(4)), _DEFS, _SCENE]
        result = await StoryCreator().generate(
            _request(
                td,
                payloads,
                {"story_type": "picture-book", "num_pages": 4,
                 "illustrations": "svg-with-backgrounds"},
                image_role=_ImageRole(provider="fake", model="fake-image"),
            )
        )
        assert result.status == "SUCCESS", result.errors
        data = validate_artifact_data("story.v1", result.data)
        forest = next(s for s in data.settings if s.id == "forest")
        assert forest.background_data_uri and forest.background_data_uri.startswith("data:image/")
        assert any((f.label or "").startswith("background:") for f in result.files)


@pytest.mark.asyncio
async def test_backgrounds_degrade_without_image_role():
    with tempfile.TemporaryDirectory() as td:
        payloads = [_story("bedtime", _linear_pages(4)), _DEFS, _SCENE]
        result = await StoryCreator().generate(
            _request(
                td,
                payloads,
                {"story_type": "bedtime", "num_pages": 4,
                 "illustrations": "svg-with-backgrounds"},
            )
        )
        assert result.status == "SUCCESS"
        data = validate_artifact_data("story.v1", result.data)
        assert all(s.background_data_uri is None for s in data.settings)
        assert any("no image model" in w.lower() for w in result.warnings)


@pytest.mark.asyncio
async def test_backgrounds_degrade_when_image_provider_fails():
    with tempfile.TemporaryDirectory() as td:
        payloads = [_story("picture-book", _linear_pages(4)), _DEFS, _SCENE]
        result = await StoryCreator().generate(
            _request(
                td,
                payloads,
                {"story_type": "picture-book", "num_pages": 4,
                 "illustrations": "svg-with-backgrounds"},
                image_role=_BrokenImageRole(provider="fake", model="fake-image"),
            )
        )
        assert result.status == "SUCCESS"
        data = validate_artifact_data("story.v1", result.data)
        assert all(s.background_data_uri is None for s in data.settings)
        assert any("could not be painted" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_text_only_mode_skips_symbol_and_scene_calls():
    with tempfile.TemporaryDirectory() as td:
        role = _QueueRole(
            provider="fake", model="fake",
            payloads=[_story("short-story", _linear_pages(4))],
        )
        result = await StoryCreator().generate(
            CreationRequest(
                content=ContentBundle(text="material"),
                config={"story_type": "short-story", "num_pages": 4,
                        "illustrations": "none"},
                models={"text": role},
                output_dir=td,
                artifact_id="a",
            )
        )
        assert result.status == "SUCCESS"
        assert role.calls == 1  # only the story pass
        data = validate_artifact_data("story.v1", result.data)
        assert all(p.svg is None for p in data.pages)


@pytest.mark.asyncio
async def test_adventure_without_reachable_ending_fails():
    with tempfile.TemporaryDirectory() as td:
        loop_pages = [
            {"id": "a", "text": "Loop.", "setting_id": "forest", "character_ids": [],
             "scene": "", "choices": [{"text": "on", "target_page_id": "b"}], "is_ending": False},
            {"id": "b", "text": "Loop back.", "setting_id": "forest", "character_ids": [],
             "scene": "", "choices": [{"text": "back", "target_page_id": "a"}], "is_ending": False},
        ]
        payloads = [_story("adventure", loop_pages), _DEFS, _SCENE]
        result = await StoryCreator().generate(
            _request(td, payloads, {"story_type": "adventure", "num_pages": 4})
        )
        assert result.status == "FAILURE"
        assert any("ending" in e.message for e in result.errors)
