"""Tests for StoryCreator using stubbed models (no network).

Covers adventure graph validation and full generate runs: linear picture book,
branching adventure, the tiered illustration pipeline (reference-conditioned
edits, prompt-only fallback), and graceful degradation when the image role is
absent or broken.
"""

from __future__ import annotations

import json
import tempfile

import pytest
from open_notebook_creator_sdk import ContentBundle, CreationRequest, ModelRole
from open_notebook_creator_sdk.schemas import validate_artifact_data
from open_notebook_creator_sdk.testing import assert_creator_compliant

from story_creator import StoryConfig, StoryCreator, _validate_graph

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


class _EditsImageModel:
    """Backend with a working /images/edits: records every call."""

    def __init__(self):
        self.generate_prompts: list[str] = []
        self.edit_prompts: list[str] = []
        self.edit_references: list[list[bytes]] = []

    async def agenerate_image(self, prompt: str, size: str = "1024x1024") -> bytes:
        self.generate_prompts.append(prompt)
        return _TINY_PNG

    async def agenerate_image_edit(
        self, prompt: str, images: list, size: str = "1024x1024"
    ) -> bytes:
        self.edit_prompts.append(prompt)
        self.edit_references.append(list(images))
        return _TINY_PNG


class _NoEditsImageModel(_EditsImageModel):
    """Backend without edits: the probe must fall back to plain generations."""

    async def agenerate_image_edit(self, prompt, images, size="1024x1024"):
        self.edit_prompts.append(prompt)
        raise RuntimeError("404 no such endpoint")


class _BrokenImageModel:
    async def agenerate_image(self, prompt: str, size: str = "1024x1024") -> bytes:
        raise RuntimeError("provider down")

    async def agenerate_image_edit(self, prompt, images, size="1024x1024"):
        raise RuntimeError("provider down")


class _ImageRole(ModelRole):
    """Hands out one shared fake image model instance so tests can inspect it."""

    def create_image(self, **_):
        return _IMAGE_MODELS[self.model]


# Registry keyed by the role's model name; reset per test via _image_role().
_IMAGE_MODELS: dict = {}


def _image_role(kind: str) -> _ImageRole:
    model = {"edits": _EditsImageModel, "noedits": _NoEditsImageModel,
             "broken": _BrokenImageModel}[kind]()
    _IMAGE_MODELS[kind] = model
    return _ImageRole(provider="fake", model=kind)


# --- payload builders --------------------------------------------------------


def _story(story_type: str, pages):
    return json.dumps(
        {
            "title": "The Curious Fox",
            "dedication": None,
            "moral": "Curiosity feeds the mind." if story_type == "fable" else None,
            "palette": ["#2d4a3e", "#e8a04c", "#c2543a", "#f4e9d8", "#5b7fa6"],
            "characters": [
                {
                    "id": "fox",
                    "name": "Fig",
                    "description": "a curious young fox",
                    "visual": (
                        "small orange fox with cream chest, red-tipped ears, "
                        "and a green woolen scarf"
                    ),
                }
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


def test_legacy_svg_configs_map_to_pictures():
    assert StoryConfig.model_validate({"illustrations": "svg"}).illustrations == "pictures"
    assert (
        StoryConfig.model_validate({"illustrations": "svg-with-backgrounds"}).illustrations
        == "pictures"
    )


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
async def test_generate_picture_book_with_reference_conditioning():
    with tempfile.TemporaryDirectory() as td:
        result = await StoryCreator().generate(
            _request(
                td,
                [_story("picture-book", _linear_pages(4))],
                {"story_type": "picture-book", "num_pages": 4},
                image_role=_image_role("edits"),
            )
        )
        assert result.status == "SUCCESS", result.errors
        data = validate_artifact_data("story.v1", result.data)
        assert data.story_type == "picture-book"
        assert len(data.pages) == 4
        assert all(
            p.image_data_uri and p.image_data_uri.startswith("data:image/")
            for p in data.pages
        )
        assert all(p.svg is None for p in data.pages)
        assert data.characters[0].visual.startswith("small orange fox")

        model = _IMAGE_MODELS["edits"]
        # One plain generation (the cast sheet), then every page via edits
        # conditioned on those exact bytes.
        assert len(model.generate_prompts) == 1
        assert "model sheet" in model.generate_prompts[0].lower()
        assert "green woolen scarf" in model.generate_prompts[0]
        assert len(model.edit_prompts) == 4
        assert all(refs == [_TINY_PNG] for refs in model.edit_references)
        assert all("green woolen scarf" in p for p in model.edit_prompts)
        assert all("no text" in p.lower() for p in model.edit_prompts)

        labels = {(f.label or "") for f in result.files}
        assert "cast-sheet" in labels
        assert any(l.startswith("page:") for l in labels)
        book = open(f"{td}/story-book.html").read()
        assert "The Curious Fox" in book and "data:image/" in book


@pytest.mark.asyncio
async def test_edits_unsupported_falls_back_to_prompt_only():
    with tempfile.TemporaryDirectory() as td:
        result = await StoryCreator().generate(
            _request(
                td,
                [_story("picture-book", _linear_pages(4))],
                {"story_type": "picture-book", "num_pages": 4},
                image_role=_image_role("noedits"),
            )
        )
        assert result.status == "SUCCESS", result.errors
        data = validate_artifact_data("story.v1", result.data)
        assert all(p.image_data_uri for p in data.pages)

        model = _IMAGE_MODELS["noedits"]
        # The probe tries edits exactly once, then the whole book uses plain
        # generations: cast sheet + 4 pages.
        assert len(model.edit_prompts) == 1
        assert len(model.generate_prompts) == 5
        page_prompts = model.generate_prompts[1:]
        assert all("green woolen scarf" in p for p in page_prompts)
        assert all("Fig walks under the pines" in p for p in page_prompts)


@pytest.mark.asyncio
async def test_generate_adventure_branching():
    with tempfile.TemporaryDirectory() as td:
        result = await StoryCreator().generate(
            _request(
                td,
                [_story("adventure", _ADV_PAGES)],
                {"story_type": "adventure", "num_pages": 6, "illustrations": "none"},
            )
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
        result = await StoryCreator().generate(
            _request(
                td,
                [_story("fable", _linear_pages(4))],
                {"story_type": "fable", "num_pages": 4, "illustrations": "none"},
            )
        )
        data = validate_artifact_data("story.v1", result.data)
        assert data.moral == "Curiosity feeds the mind."


@pytest.mark.asyncio
async def test_pictures_degrade_without_image_role():
    with tempfile.TemporaryDirectory() as td:
        result = await StoryCreator().generate(
            _request(
                td,
                [_story("bedtime", _linear_pages(4))],
                {"story_type": "bedtime", "num_pages": 4},
            )
        )
        assert result.status == "SUCCESS"
        data = validate_artifact_data("story.v1", result.data)
        assert all(p.image_data_uri is None for p in data.pages)
        assert any("no image model" in w.lower() for w in result.warnings)


@pytest.mark.asyncio
async def test_pictures_degrade_when_image_provider_fails():
    with tempfile.TemporaryDirectory() as td:
        result = await StoryCreator().generate(
            _request(
                td,
                [_story("picture-book", _linear_pages(4))],
                {"story_type": "picture-book", "num_pages": 4},
                image_role=_image_role("broken"),
            )
        )
        assert result.status == "SUCCESS"
        data = validate_artifact_data("story.v1", result.data)
        assert all(p.image_data_uri is None for p in data.pages)
        assert any("cast reference sheet" in w for w in result.warnings)
        assert any("no illustration" in w for w in result.warnings)


@pytest.mark.asyncio
async def test_text_only_mode_makes_no_image_calls():
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
                models={"text": role,
                        "image": _image_role("edits")},
                output_dir=td,
                artifact_id="a",
            )
        )
        assert result.status == "SUCCESS"
        assert role.calls == 1  # only the story pass
        model = _IMAGE_MODELS["edits"]
        assert not model.generate_prompts and not model.edit_prompts
        data = validate_artifact_data("story.v1", result.data)
        assert all(p.image_data_uri is None for p in data.pages)


@pytest.mark.asyncio
async def test_adventure_without_reachable_ending_fails():
    with tempfile.TemporaryDirectory() as td:
        loop_pages = [
            {"id": "a", "text": "Loop.", "setting_id": "forest", "character_ids": [],
             "scene": "", "choices": [{"text": "on", "target_page_id": "b"}], "is_ending": False},
            {"id": "b", "text": "Loop back.", "setting_id": "forest", "character_ids": [],
             "scene": "", "choices": [{"text": "back", "target_page_id": "a"}], "is_ending": False},
        ]
        result = await StoryCreator().generate(
            _request(td, [_story("adventure", loop_pages)],
                     {"story_type": "adventure", "num_pages": 4, "illustrations": "none"})
        )
        assert result.status == "FAILURE"
        assert any("ending" in e.message for e in result.errors)
