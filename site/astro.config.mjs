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
  build: { inlineStylesheets: 'auto' },
  devToolbar: { enabled: false },
});
