"""Sanitize LLM-authored SVG so it is safe to publish and embed.

Page illustrations are composed from a vetted ``<symbol>`` library plus an
LLM-written scene fragment. Both pass through a strict whitelist: allowed
elements and attributes only, colors clamped to literal values, ``<use>``
references restricted to known symbol ids, and nothing executable (no script,
no event handlers, no external hrefs, no foreignObject, no CSS ``url()``).
The output ends up inside published, embeddable pages — treat it as hostile.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set, Tuple

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"

_SHAPE_ATTRS = {
    "x", "y", "width", "height", "rx", "ry", "cx", "cy", "r",
    "x1", "y1", "x2", "y2", "d", "points", "transform",
    "fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin",
    "stroke-dasharray", "opacity", "fill-opacity", "stroke-opacity",
    "fill-rule",
}

ALLOWED: Dict[str, Set[str]] = {
    "g": {"transform", "fill", "stroke", "opacity", "class"},
    "path": _SHAPE_ATTRS,
    "rect": _SHAPE_ATTRS,
    "circle": _SHAPE_ATTRS,
    "ellipse": _SHAPE_ATTRS,
    "line": _SHAPE_ATTRS,
    "polyline": _SHAPE_ATTRS,
    "polygon": _SHAPE_ATTRS,
    "use": {"href", "x", "y", "width", "height", "transform", "opacity"},
    "symbol": {"id", "viewBox"},
    "title": set(),
}

_COLOR_RE = re.compile(r"^(#[0-9a-fA-F]{3}|#[0-9a-fA-F]{6}|#[0-9a-fA-F]{8}|none|transparent|currentColor|white|black)$")
_NUMBERY_RE = re.compile(r"^[\w\s.,()+\-%#]*$")
_CLASS_RE = re.compile(r"^[\w\s-]{0,60}$")
_ID_RE = re.compile(r"^[A-Za-z][\w-]{0,60}$")


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _attr_ok(tag: str, name: str, value: str) -> bool:
    if name in ("fill", "stroke"):
        return bool(_COLOR_RE.match(value.strip()))
    if name == "class":
        return bool(_CLASS_RE.match(value))
    if name == "id":
        return bool(_ID_RE.match(value))
    if name == "href":
        return value.startswith("#")
    # geometry / transform / dash values: conservative charset, no url(), no js
    return bool(_NUMBERY_RE.match(value)) and "url" not in value.lower()


def _clean_element(
    el: ET.Element, symbol_ids: Set[str], inside_symbol: bool
) -> Optional[ET.Element]:
    tag = _local(el.tag)
    if tag not in ALLOWED:
        return None
    if tag == "symbol" and inside_symbol:
        return None  # no nesting

    out = ET.Element(f"{{{SVG_NS}}}{tag}")
    for raw_name, value in el.attrib.items():
        name = _local(raw_name)
        if name.startswith("on"):
            return None  # event handler anywhere -> drop the element entirely
        if name == "href" or raw_name == f"{{{XLINK_NS}}}href":
            name = "href"
        if name not in ALLOWED[tag]:
            continue
        if not _attr_ok(tag, name, value):
            continue
        out.set(name, value)

    if tag == "use":
        ref = out.get("href", "")
        if not ref.startswith("#") or ref[1:] not in symbol_ids:
            return None
    if tag == "symbol" and not out.get("id"):
        return None
    if tag == "title":
        out.text = (el.text or "")[:200]

    for child in el:
        cleaned = _clean_element(
            child, symbol_ids, inside_symbol or tag == "symbol"
        )
        if cleaned is not None:
            out.append(cleaned)
    return out


def _serialize_children(root: ET.Element) -> str:
    ET.register_namespace("", SVG_NS)
    parts = []
    for child in root:
        s = ET.tostring(child, encoding="unicode")
        # drop the redundant per-element ns declaration for compactness
        s = s.replace(f' xmlns="{SVG_NS}"', "", 1)
        parts.append(s)
    return "".join(parts)


def sanitize_fragment(markup: str, symbol_ids: Set[str]) -> Optional[str]:
    """Whitelist-clean a scene fragment (shapes + <use> refs). Returns the
    cleaned markup, or None when nothing valid survives / parsing fails."""
    if not markup or len(markup) > 400_000:
        return None
    wrapped = (
        f'<svg xmlns="{SVG_NS}" xmlns:xlink="{XLINK_NS}">{markup.strip()}</svg>'
    )
    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError:
        return None
    out = ET.Element("root")
    for child in root:
        cleaned = _clean_element(child, symbol_ids, inside_symbol=False)
        if cleaned is not None:
            out.append(cleaned)
    if len(out) == 0:
        return None
    return _serialize_children(out)


def extract_symbols(markup: str) -> List[Tuple[str, str]]:
    """Parse a <defs> (or bare list of <symbol>s) from the model and return
    sanitized ``(id, symbol_markup)`` pairs. Invalid symbols are dropped."""
    if not markup:
        return []
    body = markup.strip()
    body = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", body).strip()
    # tolerate a wrapping <svg> and/or <defs>
    wrapped = f'<svg xmlns="{SVG_NS}" xmlns:xlink="{XLINK_NS}">{body}</svg>'
    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError:
        return []
    found: List[ET.Element] = []

    def collect(el: ET.Element) -> None:
        for child in el:
            if _local(child.tag) == "symbol":
                found.append(child)
            else:
                collect(child)

    collect(root)
    out: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for sym in found:
        cleaned = _clean_element(sym, symbol_ids=set(), inside_symbol=False)
        if cleaned is None:
            continue
        sid = cleaned.get("id", "")
        if not sid or sid in seen or len(cleaned) == 0:
            continue
        seen.add(sid)
        holder = ET.Element("root")
        holder.append(cleaned)
        out.append((sid, _serialize_children(holder)))
    return out


def used_symbol_ids(fragment: str) -> Set[str]:
    return set(re.findall(r'href="#([\w-]+)"', fragment))


def compose_page(
    scene: str, symbols: Dict[str, str], aria_label: str
) -> str:
    """Assemble a standalone page SVG: defs for the symbols the scene uses,
    then the (already sanitized) scene."""
    used = used_symbol_ids(scene)
    defs = "".join(symbols[sid] for sid in sorted(used) if sid in symbols)
    label = re.sub(r"[<>&\"]", "", aria_label)[:200]
    return (
        f'<svg xmlns="{SVG_NS}" viewBox="0 0 1600 1000" role="img" '
        f'aria-label="{label}" preserveAspectRatio="xMidYMid meet">'
        f"<defs>{defs}</defs>{scene}</svg>"
    )
