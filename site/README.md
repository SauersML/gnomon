# site

The portability laws as a static page. Source of truth for every formula is
`proofs/Calibrator`; this renders them for reading.

```sh
cd site
npm install
npm run build      # -> dist/index.html
npm run dev        # local preview with reload
```

## Design constraints

**No client JavaScript.** TeX is rendered by KaTeX at build time, so `dist/`
contains markup, CSS and web fonts and nothing else. `grep -c '<script' dist/index.html`
should print `0`; if it does not, something regressed.

**KaTeX runs in strict mode** (`strict: 'error'`, `throwOnError: true` in
`astro.config.mjs`). A malformed formula fails the build rather than rendering as
red text nobody notices.

**`dist/` is generated and not committed.** A checked-in copy of a generated file
goes stale silently, which this repository has been bitten by repeatedly — see
the note at the top of `proofs/Calibrator.lean`. Build before reading.

## Gotcha worth knowing

`remark-math@6` only produces *display* math when the `$$` delimiters sit on
their own lines:

```markdown
$$
x = 1
$$
```

Written inline as `$$x = 1$$` it silently becomes *inline* math instead — no
warning, no error, just smaller glyphs mid-paragraph. The build check below
counts display blocks so a regression is visible:

```sh
grep -c 'katex-display' dist/index.html   # expect 10
```

## Layout

```
src/pages/index.md      content, with TeX in $...$ and $$...$$
src/layouts/Page.astro  shell; imports KaTeX CSS and the stylesheet
src/styles/site.css     typography and components, both themes chosen
astro.config.mjs        remark-math + rehype-katex, strict
```
