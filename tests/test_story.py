"""Tests for StoryCreator using stubbed models (no network).

Covers full generate runs: the linear picture book, the tiered illustration
pipeline (reference-conditioned edits, prompt-only fallback), graceful
degradation when the image role is absent or broken, and the three book
editions (HTML always; EPUB and PDF when Quarto is installed).
"""

from __future__ import annotations

import json
import shutil
import tempfile

import pytest
from open_notebook_creator_sdk import ContentBundle, CreationRequest, ModelRole
from open_notebook_creator_sdk.schemas import validate_artifact_data
from open_notebook_creator_sdk.testing import assert_creator_compliant

from story_creator import StoryConfig, StoryCreator
from story_creator.exports import book_html, book_qmd

# Quarto is installed in the runtime image (and by textbook-creator's
# toolchain), but not necessarily on a contributor's machine.
_HAS_QUARTO = shutil.which("quarto") is not None
requires_quarto = pytest.mark.skipif(not _HAS_QUARTO, reason="quarto not installed")

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
        }
        for i in range(1, n + 1)
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


def _labels(result) -> list[str]:
    return [f.label or "" for f in result.files]


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


def test_retired_adventure_config_falls_back_to_short_story():
    # Saved configs from the branching era must still validate.
    assert (
        StoryConfig.model_validate({"story_type": "adventure"}).story_type
        == "short-story"
    )


def test_adventure_is_not_offered_as_a_story_type():
    schema = StoryConfig.model_json_schema()
    types = json.dumps(schema["$defs"]["StoryType"] if "$defs" in schema else schema)
    assert "adventure" not in types


def _render_story_prompt(story_type="short-story", reading_age="all-ages") -> str:
    from ai_prompter import Prompter

    from story_creator import _read_prompt

    return Prompter(template_text=_read_prompt("story.jinja")).render(
        {
            "content": "material",
            "story_type": story_type,
            "num_pages": 8,
            "reading_age": reading_age,
            "style": "paper-cutout",
            "language": None,
            "instructions": None,
        }
    ).lower()


@pytest.mark.parametrize("story_type", ["picture-book", "short-story", "fable", "bedtime"])
def test_story_prompt_demands_human_names_and_a_fixed_appearance(story_type):
    rendered = _render_story_prompt(story_type)

    assert "ordinary human first name" in rendered
    assert "skin tone as a named color" in rendered
    assert "hair color and style" in rendered
    assert "never leave skin tone or hair color unstated" in rendered
    # The deepest fix for lettering: never write a moment that needs words.
    assert "reads, writes, points at, or discusses words" in rendered
    assert "no \"scene\" may call for signs, labels, banners" in rendered
    # A fable's animals still get human first names rather than species names.
    assert ("animal characters get human first names" in rendered) == (
        story_type == "fable"
    )


def test_adult_stories_are_not_written_as_children_s_books():
    adult = _render_story_prompt(reading_age="adult")
    assert "not a children's book" in adult
    assert "no cutesy framing" in adult
    # And the picture-book registers stay put for the younger tiers.
    assert "two- to four-year-olds" in _render_story_prompt(reading_age="toddler")
    assert "teenagers" in _render_story_prompt(reading_age="young-adult")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reading_age,expected,forbidden",
    [
        ("adult", "for adult readers", "picture book"),
        ("young-adult", "young-adult novel", "picture book"),
        ("toddler", "picture book for very young children", "adult readers"),
    ],
)
async def test_art_register_follows_the_reading_age(reading_age, expected, forbidden):
    """An adult book must not be painted as a children's picture book."""
    with tempfile.TemporaryDirectory() as td:
        await StoryCreator().generate(
            _request(
                td,
                [_story("short-story", _linear_pages(4))],
                {"story_type": "short-story", "num_pages": 4,
                 "reading_age": reading_age},
                image_role=_image_role("edits"),
            )
        )
        model = _IMAGE_MODELS["edits"]
        prompts = model.generate_prompts + model.edit_prompts
        assert prompts
        for p in prompts:
            assert expected in p
            assert forbidden not in p


# --- book sources ------------------------------------------------------------


def _tiny_story():
    return validate_artifact_data(
        "story.v1",
        {
            "title": "The Curious Fox",
            "dedication": "For the curious",
            "moral": "Curiosity feeds the mind.",
            "pages": [
                {"id": "p1", "number": 1, "text": "A # hash and *stars* survive."},
                {"id": "p2", "number": 2, "text": "The last page."},
            ],
        },
    )


def test_book_html_has_no_choice_markup():
    html = book_html(_tiny_story())
    assert "The Curious Fox" in html
    assert "turn to page" not in html.lower()
    assert "choices" not in html


def test_book_qmd_escapes_markdown_and_paginates():
    qmd = book_qmd(_tiny_story(), {"p1": "page-p1.jpg"})
    assert qmd.startswith("---\ntitle: \"The Curious Fox\"")
    assert "pdf-engine: tectonic" in qmd
    assert "![](page-p1.jpg){width=100%}" in qmd
    # Prose punctuation must not become markdown structure.
    assert "A \\# hash and \\*stars\\* survive." in qmd
    # Dedication, both pages and the moral are separated by page breaks.
    assert qmd.count("{{< pagebreak >}}") == 3


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


@pytest.mark.asyncio
async def test_every_image_prompt_locks_appearance_and_bans_text():
    """The two traits that drift between pages — and lettering — are pinned."""
    with tempfile.TemporaryDirectory() as td:
        await StoryCreator().generate(
            _request(
                td,
                [_story("picture-book", _linear_pages(4))],
                {"story_type": "picture-book", "num_pages": 4},
                image_role=_image_role("edits"),
            )
        )
        model = _IMAGE_MODELS["edits"]
        prompts = model.generate_prompts + model.edit_prompts
        assert len(prompts) == 5  # cast sheet + one per page
        for p in prompts:
            low = p.lower()
            assert "same skin tone" in low
            assert "same hair color and style" in low
            assert "locked" in low
            # Leading AND trailing, since backends discount a trailing-only
            # negative in a long prompt.
            assert low.startswith("no text anywhere in the image")
            assert low.rstrip().endswith("covers are all blank.")
            # Lettering arrives on the furniture, so the furniture is named.
            for surface in ("whiteboards", "chalkboards", "posters", "screens"):
                assert surface in low
            # An empty balloon is still a balloon.
            assert "not even empty ones" in low
        # The reference-conditioned pages say to take the colors from the sheet.
        assert all(
            "copy each character's skin tone, hair, and clothing colors from it"
            in p
            for p in model.edit_prompts
        )

        book = open(f"{td}/story-book.html").read()
        assert "The Curious Fox" in book and "data:image/" in book


@pytest.mark.asyncio
async def test_only_book_editions_are_published():
    """The cast sheet and per-page art stay working files, never downloads."""
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
        labels = _labels(result)
        assert labels[0] == "HTML"
        assert set(labels) <= {"HTML", "EPUB", "PDF"}
        assert not any(label.startswith("page:") for label in labels)
        assert "cast-sheet" not in labels
        # Every published file is a real book, named after the story.
        assert all(f.filename.startswith("the-curious-fox.") for f in result.files)


@requires_quarto
@pytest.mark.asyncio
async def test_generate_renders_epub_and_pdf():
    with tempfile.TemporaryDirectory() as td:
        result = await StoryCreator().generate(
            _request(
                td,
                [_story("short-story", _linear_pages(4))],
                {"story_type": "short-story", "num_pages": 4,
                 "illustrations": "none"},
            )
        )
        assert result.status == "SUCCESS", result.errors
        by_label = {f.label: f for f in result.files}
        assert "EPUB" in by_label, result.warnings
        assert by_label["EPUB"].content_type == "application/epub+zip"
        assert (
            by_label["EPUB"].filename == "the-curious-fox.epub"
            and by_label["EPUB"].path == "story-book.epub"
        )


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
async def test_generated_stories_never_branch():
    """Choices the model volunteers are ignored, not warned about."""
    with tempfile.TemporaryDirectory() as td:
        pages = [
            dict(p, choices=[{"text": "Go left", "target_page_id": "nowhere"}],
                 is_ending=True)
            for p in _linear_pages(4)
        ]
        result = await StoryCreator().generate(
            _request(
                td,
                [_story("short-story", pages)],
                {"story_type": "short-story", "num_pages": 4,
                 "illustrations": "none"},
            )
        )
        assert result.status == "SUCCESS", result.errors
        data = validate_artifact_data("story.v1", result.data)
        assert data.start_page_id is None
        assert all(not p.choices for p in data.pages)
        assert not any("choice" in w.lower() for w in result.warnings)
        assert [p.number for p in data.pages] == [1, 2, 3, 4]


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
