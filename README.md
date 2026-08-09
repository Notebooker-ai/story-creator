# story-creator

An [Open Notebook](https://github.com/Notebooker-ai/open-notebook) creation
plugin that turns notebook content into an illustrated story (`story.v1`):
picture books, short stories, fables, bedtime stories, and branching
choose-your-own-path adventures.

Every page gets one full-page, style-locked diffusion illustration painted by
the `image` model role. Character identity across pages is tiered: the story
pass writes a locked `visual` spec per character that is embedded verbatim in
every page prompt (baseline), and when the image backend supports the
OpenAI-shaped `/images/edits` endpoint, a cast-sheet image is painted once and
passed as a reference on every page (reference conditioning — the creator
probes the first page and falls back automatically). Without an image model
the creator degrades gracefully to a text-only story. Outputs: the `story.v1`
artifact (rendered by the bundled page-turner view, with choice buttons for
adventures) and a standalone print-ready `story-book.html` (adventures print
in the classic "turn to page N" style). The view still renders legacy
SVG-era artifacts.
