"""Book exports for ``story.v1``: HTML, EPUB and PDF.

Three deliverables, one source of truth. The HTML book is written directly —
a single self-contained file with the illustrations inlined as data URIs, so
it works with no toolchain at all. EPUB and PDF are rendered by Quarto (the
same toolchain the host image already installs for textbook-creator: Quarto
with Tectonic as the PDF engine) from a generated ``.qmd`` that points at the
page images already sitting in the output directory. Where Quarto is missing
or a render fails, the caller gets a warning and the HTML book still ships.
"""

from __future__ import annotations

import asyncio
import html as html_lib
import json
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger
from open_notebook_creator_sdk import CreationFile
from open_notebook_creator_sdk.schemas.story_v1 import StoryV1

# Quarto shells out to Tectonic for PDF, which may pull LaTeX packages on first
# use; EPUB is pure pandoc and much quicker.
_EPUB_TIMEOUT_SECONDS = 180
_PDF_TIMEOUT_SECONDS = 420

# Backslash-escaped in page text so prose never turns into markdown structure
# (a line starting "#" becoming a heading, "*starred*" becoming emphasis, "$"
# opening math). Pandoc reads a backslash before any ASCII punctuation as that
# literal character.
_MD_SPECIALS = set("\\`*_{}[]<>#+-!$~^|")


def _md_escape(text: str) -> str:
    return "".join("\\" + c if c in _MD_SPECIALS else c for c in text)


def book_html(story: StoryV1) -> str:
    """A standalone, print-ready HTML book with the art inlined."""
    esc = html_lib.escape
    pages_html: List[str] = []
    for p in sorted(story.pages, key=lambda x: x.number):
        art = ""
        if p.image_data_uri:
            art = f'<div class="art"><img src="{p.image_data_uri}" alt=""></div>'
        pages_html.append(
            f'<section class="page"><div class="num">{p.number}</div>'
            f'{art}<div class="txt">{esc(p.text)}</div></section>'
        )
    dedication = (
        f'<p class="dedication">{esc(story.dedication)}</p>' if story.dedication else ""
    )
    moral = f'<p class="moral">{esc(story.moral)}</p>' if story.moral else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(story.title)}</title>
<style>
  body {{ font-family: Georgia, 'Times New Roman', serif; margin: 0; color: #222; }}
  .cover, .page {{ max-width: 760px; margin: 0 auto; padding: 48px 32px; page-break-after: always; }}
  .cover h1 {{ font-size: 2.4em; text-align: center; margin-top: 20vh; }}
  .dedication {{ text-align: center; font-style: italic; color: #555; }}
  .art {{ width: 100%; aspect-ratio: 1 / 1; margin-bottom: 24px;
         border-radius: 12px; overflow: hidden; }}
  .art img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
  .txt {{ font-size: 1.15em; line-height: 1.7; white-space: pre-wrap; }}
  .num {{ text-align: right; color: #999; font-size: 0.85em; }}
  .moral {{ font-style: italic; text-align: center; }}
  @page {{ margin: 18mm; }}
  @media print {{ .cover, .page {{ padding: 0; }} }}
</style>
</head>
<body>
<section class="cover"><h1>{esc(story.title)}</h1>{dedication}</section>
{''.join(pages_html)}
{f'<section class="page">{moral}</section>' if moral else ''}
</body>
</html>
"""


def book_qmd(story: StoryV1, image_names: dict) -> str:
    """The Quarto source both EPUB and PDF render from.

    ``image_names`` maps page id -> image filename already written next to the
    ``.qmd``; pages missing from it simply render text-only.
    """
    front = json.dumps(story.title)  # JSON is valid YAML, so quoting is handled
    head = [
        "---",
        f"title: {front}",
        "format:",
        "  epub:",
        "    toc: false",
        "  pdf:",
        "    pdf-engine: tectonic",
        "    documentclass: article",
        "    geometry: margin=18mm",
        "    fontsize: 12pt",
        "---",
        "",
    ]
    body: List[str] = []
    if story.dedication:
        body.append(f"*{_md_escape(story.dedication)}*")
        body.append("")
        body.append("{{< pagebreak >}}")
        body.append("")
    for i, p in enumerate(sorted(story.pages, key=lambda x: x.number)):
        if i:
            body.append("{{< pagebreak >}}")
            body.append("")
        name = image_names.get(p.id)
        if name:
            body.append(f"![]({name}){{width=100%}}")
            body.append("")
        body.append(_md_escape(p.text))
        body.append("")
    if story.moral:
        body.append("{{< pagebreak >}}")
        body.append("")
        body.append(f"*{_md_escape(story.moral)}*")
        body.append("")
    return "\n".join(head + body)


async def _quarto_render(qmd: Path, fmt: str, timeout: int) -> bool:
    """Render ``qmd`` to one format. False (with a log line) on any failure."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "quarto",
            "render",
            qmd.name,
            "--to",
            fmt,
            cwd=str(qmd.parent),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except OSError as e:  # quarto not on PATH
        logger.warning(f"story: quarto unavailable for {fmt}: {e}")
        return False
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        logger.warning(f"story: quarto {fmt} render timed out after {timeout}s")
        return False
    if proc.returncode != 0:
        tail = (out or b"").decode("utf-8", "replace")[-2000:]
        logger.warning(f"story: quarto {fmt} render failed ({proc.returncode}): {tail}")
        return False
    return True


async def write_book_files(
    story: StoryV1,
    out_dir: Path,
    stem: str,
    image_names: Optional[dict] = None,
) -> Tuple[List[CreationFile], List[str]]:
    """Write the HTML book and render EPUB + PDF beside it.

    Returns the files to publish (HTML first, then whichever of EPUB/PDF
    rendered) and any user-facing warnings. The HTML book is self-contained,
    so it is emitted even when no toolchain is present.
    """
    files: List[CreationFile] = []
    warnings: List[str] = []

    html_rel = "story-book.html"
    (out_dir / html_rel).write_text(book_html(story), "utf-8")
    files.append(
        CreationFile(
            filename=f"{stem}.html",
            content_type="text/html",
            path=html_rel,
            label="HTML",
        )
    )

    if shutil.which("quarto") is None:
        logger.warning("story: quarto not installed — EPUB and PDF skipped")
        warnings.append(
            "Quarto is not installed on this server, so only the HTML book "
            "could be produced (no EPUB or PDF)."
        )
        return files, warnings

    qmd = out_dir / "story-book.qmd"
    qmd.write_text(book_qmd(story, image_names or {}), "utf-8")

    # Serially, not concurrently: both renders share one working directory and
    # would trample each other's intermediates. A failing PDF must not cost the
    # reader their EPUB, so each is judged on its own.
    epub_ok = await _quarto_render(qmd, "epub", _EPUB_TIMEOUT_SECONDS)
    pdf_ok = await _quarto_render(qmd, "pdf", _PDF_TIMEOUT_SECONDS)
    for ok, fmt, ctype in (
        (epub_ok, "epub", "application/epub+zip"),
        (pdf_ok, "pdf", "application/pdf"),
    ):
        rel = f"story-book.{fmt}"
        if ok and (out_dir / rel).exists():
            files.append(
                CreationFile(
                    filename=f"{stem}.{fmt}",
                    content_type=ctype,
                    path=rel,
                    label=fmt.upper(),
                )
            )
        else:
            warnings.append(f"The {fmt.upper()} edition could not be rendered.")

    return files, warnings
