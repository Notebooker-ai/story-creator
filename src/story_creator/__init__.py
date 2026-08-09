"""story-creator: an Open Notebook creator that turns notebook content into an
illustrated story (emitted as ``story.v1``).

Four narrative forms share one pipeline: a story pass writes the book
(characters with locked ``visual`` specs, settings, palette, per-page text and
scene notes); an optional art pass paints one full-page diffusion illustration
per page via the ``image`` model role. Character identity across pages is
tiered: every page prompt embeds each character's canonical visual spec
verbatim (baseline), and when the image backend supports the OpenAI-shaped
``/images/edits`` endpoint, a single cast-sheet image is painted first and
passed as a reference on every page (reference conditioning). Without an image
model the creator degrades gracefully to a text-only story.

Every run ships the same three readable editions — HTML, EPUB and PDF (see
``exports``) — and nothing else; the cast sheet and the raw page art are
working files, not downloads.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import re
from importlib import resources
from typing import Any, ClassVar, Dict, List, Literal, Optional, Set

from ai_prompter import Prompter
from loguru import logger
from open_notebook_creator_sdk import (
    BaseCreator,
    CreationError,
    CreationRequest,
    CreationResult,
    CreatorManifest,
    CreatorView,
    ModelRoleSpec,
)
from open_notebook_creator_sdk.schemas.story_v1 import (
    StoryCharacter,
    StoryPage,
    StorySetting,
    StoryV1,
)
from pydantic import BaseModel, Field, field_validator

from story_creator.exports import write_book_files

__version__ = "0.3.0"

SCHEMA_ID = "story.v1"
_MAX_CONCURRENT_PAGES = 4
_MAX_SETTINGS = 6
_IMG_SIZE = "1024x1024"

StoryType = Literal["picture-book", "short-story", "fable", "bedtime"]
ReadingAge = Literal["toddler", "early-reader", "middle-grade", "all-ages"]
Illustrations = Literal["none", "pictures"]
Style = Literal["paper-cutout", "geometric", "night-sky"]

_STYLE_PROMPTS: Dict[str, str] = {
    "paper-cutout": (
        "flat paper-cutout collage style, layered soft shapes, torn-paper "
        "texture, gentle lighting"
    ),
    "geometric": (
        "flat geometric style, simple bold shapes, clean mid-century "
        "children's book look"
    ),
    "night-sky": (
        "soft night-sky style, gentle silhouettes, stars, deep calm colors"
    ),
}


class StoryConfig(BaseModel):
    """Per-generation config; drives the host's generate form."""

    story_type: StoryType = Field(
        default="short-story",
        description=(
            "picture-book (a few sentences a page), short-story (prose), "
            "fable (animal characters and a moral), or bedtime (a calming arc)"
        ),
    )
    num_pages: int = Field(default=10, ge=4, le=24, description="How many pages")
    reading_age: ReadingAge = Field(
        default="all-ages", description="toddler, early-reader, middle-grade, or all-ages"
    )
    illustrations: Illustrations = Field(
        default="pictures",
        description=(
            "none (text only) or pictures (one AI-painted illustration per "
            "page; needs an image model)"
        ),
    )
    style: Style = Field(
        default="paper-cutout",
        description="Illustration style: paper-cutout, geometric, or night-sky",
    )

    @field_validator("illustrations", mode="before")
    @classmethod
    def _migrate_legacy_illustrations(cls, v: Any) -> Any:
        # Configs saved by the structural-SVG era map onto the diffusion mode.
        if v in ("svg", "svg-with-backgrounds"):
            return "pictures"
        return v

    @field_validator("story_type", mode="before")
    @classmethod
    def _migrate_retired_adventure(cls, v: Any) -> Any:
        # Branching "turn to page N" adventures were dropped; regenerating from
        # a saved adventure config now writes a straight-through story.
        return "short-story" if v == "adventure" else v


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def _read_prompt(name: str) -> str:
    return resources.files("story_creator.prompts").joinpath(name).read_text()


def _parse_json(raw: str) -> Optional[Any]:
    try:
        return json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        return None


def _slug(text: str, fallback: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:40]
    return s or fallback


_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def _clean_palette(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out = [str(v).strip() for v in value if _HEX_RE.match(str(v).strip())]
    return out[:6]


def _try_compress_jpeg(png_bytes: bytes) -> tuple[bytes, str]:
    """Shrink an illustration for inline storage; fall back to the original."""
    try:
        from PIL import Image  # optional but listed in deps

        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        img.thumbnail((1024, 1024))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=78, optimize=True)
        return buf.getvalue(), "image/jpeg"
    except Exception:  # noqa: BLE001
        return png_bytes, "image/png"


def _character_line(c: Dict[str, str]) -> str:
    return f"{c['name']} — {c.get('visual') or c.get('description') or ''}".strip(" —")


def _cast_sheet_prompt(characters: List[Dict[str, str]], style: str, palette: List[str]) -> str:
    return (
        f"Character model sheet, {_STYLE_PROMPTS[style]}. "
        "All characters full-body, standing side by side on a plain light "
        "background, facing forward: "
        + "; ".join(_character_line(c) for c in characters)
        + f". Color palette: {', '.join(palette)}. "
        "No text, no letters, no labels, no words."
    )


def _page_prompt(
    page: Dict[str, Any],
    setting: Optional[Dict[str, Any]],
    characters: List[Dict[str, str]],
    style: str,
    palette: List[str],
    with_reference: bool,
) -> str:
    parts: List[str] = []
    if with_reference:
        parts.append(
            "Using the reference image as the definitive character designs, "
            f"paint a children's picture-book illustration, {_STYLE_PROMPTS[style]}."
        )
    else:
        parts.append(
            f"Children's picture-book illustration, {_STYLE_PROMPTS[style]}."
        )
    parts.append(f"Scene: {page['scene'] or page['text'][:200]}")
    if setting:
        parts.append(f"Setting: {setting['description'] or setting['name']}")
    if characters:
        parts.append(
            "Characters (must look EXACTLY like this — same colors, clothing, "
            "and features on every page): "
            + "; ".join(_character_line(c) for c in characters)
        )
    parts.append(f"Color palette: {', '.join(palette)}.")
    parts.append("No text, no letters, no words, no captions.")
    return " ".join(parts)


class StoryCreator(BaseCreator):
    config_model: ClassVar[type] = StoryConfig

    @property
    def manifest(self) -> CreatorManifest:
        return self.build_manifest(
            key="stories",
            name="Story",
            version=__version__,
            description=(
                "Turn your sources into a story — a picture book, short story, "
                "fable, or bedtime story — with one AI-painted illustration "
                "per page and characters kept consistent across the whole "
                "book. Download it as HTML, EPUB or PDF."
            ),
            sdk_compat=">=0.9,<1",
            emits=[SCHEMA_ID],
            model_roles=[
                ModelRoleSpec(
                    key="text",
                    kind="language",
                    requires=["structured_json"],
                    description="LLM that writes the story and scene notes.",
                ),
                ModelRoleSpec(
                    key="image",
                    kind="image",
                    required=False,
                    description=(
                        "Image model that paints the per-page illustrations "
                        "(required for 'pictures'; without it the story is "
                        "text-only)."
                    ),
                ),
            ],
            icon="book-heart",
            view=CreatorView(entry="view/index.html"),
        )

    async def generate(self, request: CreationRequest) -> CreationResult:
        cfg = StoryConfig.model_validate(request.config)
        role = request.models.get("text")
        if role is None:
            return CreationResult(
                status="FAILURE",
                schema_id=SCHEMA_ID,
                data={},
                errors=[CreationError(phase="setup", message="missing 'text' model role")],
                user_message="No language model was provided for the story creator.",
            )
        warnings: List[str] = []

        # ---- story pass ----------------------------------------------------
        story_prompt = Prompter(template_text=_read_prompt("story.jinja")).render(
            {
                "content": request.content.text,
                "story_type": cfg.story_type,
                "num_pages": cfg.num_pages,
                "reading_age": cfg.reading_age,
                "style": cfg.style,
                "language": request.language,
                "instructions": request.instructions,
            }
        )
        llm = role.create_language(structured={"type": "json"}, max_tokens=8000)
        resp = await llm.ainvoke(story_prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        plan = _parse_json(raw)
        if not isinstance(plan, dict) or not isinstance(plan.get("pages"), list):
            return CreationResult(
                status="FAILURE",
                schema_id=SCHEMA_ID,
                data={},
                errors=[CreationError(phase="plan", message="story pass returned no pages", retryable=True)],
                user_message="The model could not write a story from this content. Please retry.",
            )

        title = str(plan.get("title") or "").strip() or "A Story"
        palette = _clean_palette(plan.get("palette")) or [
            "#2d4a3e", "#e8a04c", "#c2543a", "#f4e9d8", "#5b7fa6", "#2b2b2b"
        ]

        characters: List[Dict[str, str]] = []
        for i, c in enumerate((plan.get("characters") or [])[:6]):
            if not isinstance(c, dict):
                continue
            cid = _slug(c.get("id") or c.get("name"), f"char{i}")
            characters.append(
                {
                    "id": cid,
                    "name": str(c.get("name") or cid).strip()[:60],
                    "description": str(c.get("description") or "").strip()[:300],
                    "visual": str(c.get("visual") or "").strip()[:500],
                }
            )
        settings: List[Dict[str, Any]] = []
        for i, s in enumerate((plan.get("settings") or [])[:_MAX_SETTINGS]):
            if not isinstance(s, dict):
                continue
            sid = _slug(s.get("id") or s.get("name"), f"setting{i}")
            settings.append(
                {
                    "id": sid,
                    "name": str(s.get("name") or sid).strip()[:80],
                    "description": str(s.get("description") or "").strip()[:300],
                }
            )
        setting_ids = {s["id"] for s in settings}
        char_ids = {c["id"] for c in characters}

        pages: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()
        for i, p in enumerate(plan["pages"][: cfg.num_pages]):
            if not isinstance(p, dict):
                continue
            text = str(p.get("text") or "").strip()
            if not text:
                continue
            pid = _slug(p.get("id"), f"page-{i + 1}")
            while pid in seen_ids:
                pid = f"{pid}-{len(seen_ids)}"
            seen_ids.add(pid)
            sid = _slug(p.get("setting_id"), "")
            pages.append(
                {
                    "id": pid,
                    "number": len(pages) + 1,
                    "text": text,
                    "setting_id": sid if sid in setting_ids else None,
                    "character_ids": [
                        c for c in (p.get("character_ids") or []) if c in char_ids
                    ][:4],
                    "scene": str(p.get("scene") or "").strip()[:400],
                    "image_data_uri": None,
                }
            )
        if not pages:
            return CreationResult(
                status="FAILURE",
                schema_id=SCHEMA_ID,
                data={},
                errors=[CreationError(phase="plan", message="no valid pages")],
                user_message="No story pages could be written from this content.",
            )

        # ---- illustrations (full-page diffusion) ----------------------------
        from pathlib import Path

        out_dir = Path(request.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Page art is written to the working directory so Quarto can pick it up
        # when rendering the EPUB and PDF, but it is never published on its own
        # — the three book editions are the only downloads.
        image_names: Dict[str, str] = {}
        image_role = request.models.get("image")
        want_pictures = cfg.illustrations == "pictures"
        if want_pictures and image_role is None:
            warnings.append(
                "No image model configured — illustrations skipped (text-only story)."
            )
            want_pictures = False

        if want_pictures:
            img_model = image_role.create_image()
            chars_by_id = {c["id"]: c for c in characters}
            settings_by_id = {s["id"]: s for s in settings}

            # Cast sheet: one reference image of the whole cast, reused on
            # every page when the backend supports /images/edits.
            cast_sheet: Optional[bytes] = None
            if characters:
                try:
                    cast_sheet = await img_model.agenerate_image(
                        _cast_sheet_prompt(characters, cfg.style, palette),
                        size=_IMG_SIZE,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"story: cast sheet failed: {e}")
                    warnings.append(
                        "The cast reference sheet could not be painted; "
                        "characters may vary more between pages."
                    )
            use_edits = cast_sheet is not None
            edits_warned = False
            sem = asyncio.Semaphore(_MAX_CONCURRENT_PAGES)

            async def paint(p: Dict[str, Any]) -> Optional[bytes]:
                nonlocal use_edits, edits_warned
                page_chars = [chars_by_id[cid] for cid in p["character_ids"]]
                async with sem:
                    if use_edits:
                        prompt = _page_prompt(
                            p, settings_by_id.get(p["setting_id"] or ""),
                            page_chars, cfg.style, palette, with_reference=True,
                        )
                        try:
                            return await img_model.agenerate_image_edit(
                                prompt, [cast_sheet], size=_IMG_SIZE
                            )
                        except Exception as e:  # noqa: BLE001
                            # Most likely the gateway/model has no edits
                            # endpoint; drop to prompt-only for the whole book.
                            use_edits = False
                            if not edits_warned:
                                edits_warned = True
                                logger.warning(f"story: edits tier unavailable: {e}")
                    prompt = _page_prompt(
                        p, settings_by_id.get(p["setting_id"] or ""),
                        page_chars, cfg.style, palette, with_reference=False,
                    )
                    try:
                        return await img_model.agenerate_image(prompt, size=_IMG_SIZE)
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"story: illustration for '{p['id']}' failed: {e}")
                        return None

            # Probe with the first page alone so the edits-vs-generations
            # decision is made once, then fan out the rest concurrently.
            first_raw = await paint(pages[0])
            rest_raw = await asyncio.gather(*(paint(p) for p in pages[1:]))
            undrawn = 0
            for p, raw_img in zip(pages, [first_raw, *rest_raw]):
                if raw_img is None:
                    undrawn += 1
                    continue
                compressed, mime = _try_compress_jpeg(raw_img)
                p["image_data_uri"] = (
                    f"data:{mime};base64,{base64.b64encode(compressed).decode()}"
                )
                rel = f"page-{p['id']}.{'jpg' if mime == 'image/jpeg' else 'png'}"
                (out_dir / rel).write_bytes(compressed)
                image_names[p["id"]] = rel
            if undrawn:
                warnings.append(f"{undrawn} page(s) have no illustration.")

        # ---- assemble -------------------------------------------------------
        story = StoryV1(
            title=title,
            dedication=(str(plan.get("dedication") or "").strip() or None),
            story_type=cfg.story_type,
            reading_age=cfg.reading_age,
            style=cfg.style,
            moral=(str(plan.get("moral") or "").strip() or None),
            palette=palette,
            characters=[StoryCharacter(**c) for c in characters],
            settings=[StorySetting(**s) for s in settings],
            pages=[
                StoryPage(
                    id=p["id"],
                    number=p["number"],
                    text=p["text"],
                    setting_id=p["setting_id"],
                    character_ids=p["character_ids"],
                    image_data_uri=p["image_data_uri"],
                )
                for p in pages
            ],
        )

        # ---- editions -------------------------------------------------------
        files, book_warnings = await write_book_files(
            story, out_dir, _slug(title, "story"), image_names
        )
        warnings.extend(book_warnings)

        return CreationResult(
            status="SUCCESS",
            schema_id=SCHEMA_ID,
            data=story.model_dump(),
            files=files,
            warnings=warnings,
        )
