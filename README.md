# story-creator

An [Open Notebook](https://github.com/Notebooker-ai/open-notebook) creation
plugin that turns notebook content into an illustrated story (`story.v1`):
picture books, short stories, fables and bedtime stories. Stories are read
straight through — there is no branching "turn to page N" form.

Every page gets one full-page, style-locked diffusion illustration painted by
the `image` model role. Character identity across pages is tiered: the story
pass writes a locked `visual` spec per character that is embedded verbatim in
every page prompt (baseline), and when the image backend supports the
OpenAI-shaped `/images/edits` endpoint, a cast-sheet image is painted once and
passed as a reference on every page (reference conditioning — the creator
probes the first page and falls back automatically). Without an image model
the creator degrades gracefully to a text-only story.

The `visual` spec must name skin tone and hair colour explicitly — those are
the traits a diffusion backend re-invents when the spec leaves them unsaid —
and every image prompt restates that they are locked for the whole book.
Characters get ordinary human first names, animals in fables included. No
prompt ever asks for lettering: the ban on text, signage and watermarks opens
*and* closes each prompt, since backends discount a trailing-only negative in
a long prompt.

Outputs: the `story.v1` artifact (rendered by the bundled page-turner view)
plus exactly three downloadable editions of the book — **HTML**, **EPUB** and
**PDF**. The HTML book is self-contained, with the art inlined as data URIs;
the EPUB and PDF are rendered by Quarto (Tectonic as the PDF engine), the same
toolchain the Open Notebook image already ships for textbook-creator. Where
Quarto is missing or a render fails, the run still succeeds with a warning and
whatever editions did render. The cast sheet and the raw page art are working
files, not downloads.

The view still renders legacy SVG-era and branching artifacts; old branching
stories are simply read in page order.
