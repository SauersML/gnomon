import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// TeX is rendered to HTML by KaTeX at BUILD TIME. Nothing ships to the
// browser except markup, CSS and the KaTeX web fonts -- there is no client
// JavaScript on this site at all.
export default defineConfig({
  site: 'https://example.invalid/gnomon',
  base: '/',
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [[rehypeKatex, { strict: 'error', throwOnError: true }]],
    syntaxHighlight: 'shiki',
    shikiConfig: { theme: 'github-light', wrap: true },
  },
  // The built page must open from file:// as well as from a server. Astro
  // emits absolute asset paths (/_astro/...), which file:// cannot resolve, so
  // the stylesheet and the KaTeX fonts are inlined instead of linked.
  build: { inlineStylesheets: 'always', assets: '_astro' },
  vite: {
    build: {
      // Larger than the biggest KaTeX woff2, so every font becomes a data URI.
      assetsInlineLimit: 4 * 1024 * 1024,
    },
  },
  devToolbar: { enabled: false },
});
