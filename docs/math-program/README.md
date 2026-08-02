# Math program: open-problem documents

Abstract problem statements written for an external mathematics team. They are
deliberately stated without domain vocabulary, so that the mathematics is
attacked on its own terms rather than as an application. Every one of them
exists because a question in this corpus turned out to be a special case of
something more general.

Read in this order if reading for the first time: `open-problems-transfer-laws.txt`
is the current and most complete statement and supersedes the earlier drafts for
any purpose except history.

| File | What it is |
|---|---|
| `open-problems-transfer-laws.txt` | **Current.** Nine problems on transfer laws for parameterized atomic ensembles, plus the standard of what counts as a solution. |
| `open-problems-approximation.txt` | Earlier round, approximation-theoretic framing. |
| `directions-exact-laws.txt` | Earlier round, aimed at exact laws and method discovery rather than bounds. |
| `brief-degradation-calculus.txt` | Brief on the degradation cocycle and the irreducible class. |
| `response-immunity-and-harder-targets.txt` | Response arguing for harder targets, with the immunity-paragraph discipline. |
| `open-problems-abstract.txt`, `open-problems.txt` | First drafts, kept for provenance. |

## Standing conventions these documents assume

These came out of this project's own failures and are stated in the documents
themselves, but they are worth having in one place:

- **Exactness.** An inequality is an answer only with its equality cases and a
  proof of attainment. "Order-sharp" is not sharp.
- **Necessity.** A sufficient condition is half a theorem; the converse is where
  the work is.
- **Computability from inputs.** Every functional in an answer must be an
  explicit functional of the given data. A law stated in terms of quantities
  determined by the answer is circular however analytic it looks.
- **Pre-registered audit points.** An announced solution names, in the
  announcement, the two or three steps it most depends on. A proof whose author
  cannot name its weakest step has not been read carefully enough by its author.
- **Falsifiability of negative claims.** A claim that something is impossible,
  invisible, or unreachable must state what would have to exist for it to be
  false — and that falsifier must be executed and reported, not merely named. A
  named-but-unrun falsifier reads as more careful than a bare assertion while
  being exactly as unfounded.
- **Controls on nulls.** A search that finds nothing is informative only if it is
  known capable of finding something. Every computational null needs a positive
  control.

## Status of the upstream results these interact with

Recorded here because it is scope information that downstream work keeps needing
and it currently lives only in conversation.

- The **AP1 audit failed in generality.** The Insertion Lemma requires Cramér's
  condition on the log-square law. Every finitely supported law violates Cramér —
  its characteristic function is Bohr almost periodic, hence recurrent, so
  `limsup |φ| = 1` — and this is independent of whether the law is lattice.
  The blindness result survives only for the specific pair, both of which have
  smooth modulus densities. The general ladder-measurability claim is scoped to
  the smooth-modulus stratum; the non-Cramér frontier is an open annex.
- A **second, unresolved AP1 defect**: §4 applies the Insertion Lemma to a capped
  integrand, which is Lipschitz with a kink — C⁰, not C³ — so the
  third-derivative bound that buys the `b^(-3/2)` remainder is destroyed and the
  surviving remainder is `b^(-1/2)` per coordinate. No downstream formalization
  should assume §4 until this resolves.
