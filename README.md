# story-creator

An [Open Notebook](https://github.com/Notebooker-ai/open-notebook) creation
plugin that turns notebook content into an illustrated story (`story.v1`):
picture books, short stories, fables, bedtime stories, and branching
choose-your-own-path adventures.

Characters are drawn once as a sanitized SVG `<symbol>` library and reused by
reference on every page, so identity is structural, not probabilistic. An
optional `image` model role paints one style-locked diffusion background per
*setting* (layered behind the vector scene); without it, the creator degrades
gracefully to pure vector scenery. Outputs: the `story.v1` artifact (rendered
by the bundled page-turner view, with choice buttons for adventures) and a
standalone print-ready `story-book.html` (adventures print in the classic
"turn to page N" style).
