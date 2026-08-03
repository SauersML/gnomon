#!/usr/bin/env python3
"""Find theorems whose ENTIRE proof is a structure-field projection, on origin/main.

This is the review's defect in its mechanically checkable form, and the standard is
the coordinator's: if replacing the proof body with the field yields the same theorem,
there is no theorem. Such a proof states nothing -- the conclusion IS the hypothesis.

Precise by construction: no prose parsing, no backtick guessing. A theorem qualifies
only if its proof body reduces to `X.f` / `exact X.f` / `X.f args` where `f` is a
declared field of a structure in this corpus.

Runs against origin/main, never the worktree. THIS IS NOT A STYLE POINT. On 2026-08-03
three agents in one day reported a structure "removed from the corpus" after grepping a
worktree that carried another agent's UNCOMMITTED deletions. A worktree grep and an origin
grep answer different questions, and only the second answers "is this in the corpus".

CALIBRATION AND KNOWN LIMITS -- read before quoting a number.

  Calibrated against ground truth: it independently finds the `GenotypeChaosLimits`
  consumers in EpistaticChaos that an external review named, which is the evidence that
  it detects the real thing.

  It also found sites the review did not name. Two verified by hand:
    * PortabilityBounds.FittedSelectionLaw.magnitude_pinned -- the purest instance in the
      corpus. The field `fits` IS the theorem's statement; the proof is `F.fits fst hlo hhi`.
    * EpistaticChaos.no_moment_matching_calibration_off_temperedness := CD.divergence_phase.

  FALSE POSITIVES REMAIN and the raw count is NOT a measurement. Two modes, one fixed and
  one open:
    FIXED  -- taking the first `:=` in the declaration text picked up field names out of
              HYPOTHESIS binders (`(h : (let mu := dgp.jointMeasure) = ...)`), reporting a
              hypothesis as a proof. Now takes the last `:=`.
    OPEN   -- line-joining can absorb a following tactic or the next declaration, so
              entries whose printed proof ends in `linarith`, `simpa using h`,
              `positivity`, or `open ... in` have MORE proof than the projection and are
              probably not this defect. Inspect every hit before acting on it.

  NOT EVERY HIT IS A DEFECT. An accessor forwarding a genuine model invariant can be
  plumbing. The retired `Identification.formula_eq_observable := i.derivation` pattern,
  however, is now forbidden: it accepted the desired scientific conclusion from a caller.
  The defect is a theorem whose name and statement claim a result that a field already
  asserts.

  The standard to apply, which no tool can apply for you: if replacing the proof body with
  the field yields the same theorem, there is no theorem.
"""
import re, subprocess, collections

def sh(*a): return subprocess.run(a, capture_output=True, text=True).stdout
REF = "origin/main"
files = [f for f in sh("git","ls-tree","-r","--name-only",REF).splitlines()
         if f.endswith(".lean") and f.startswith("proofs/")]
srcs = {f: sh("git","show",f"{REF}:{f}") for f in files}

# --- structure fields declared in the corpus ---
fields = collections.defaultdict(set)
for f, s in srcs.items():
    cur = None
    for l in s.split("\n"):
        m = re.match(r'^\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|noncomputable\s+)*(structure|class)\s+([A-Za-z_][\w.\']*)', l)
        if m:
            cur = m.group(2); continue
        if cur:
            if re.match(r'^\S', l) and l.strip():
                cur = None; continue
            fm = re.match(r"\s{2,}([a-z_][\w']*)\s*:", l)
            if fm and not l.strip().startswith(("--","/-","|")):
                fields[cur].add(fm.group(1))
allfields = set()
for v in fields.values(): allfields |= v

# --- theorems and their proof bodies ---
THM = re.compile(r'^\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+)*(theorem|lemma)\s+([A-Za-z_][\w.\']*)')
hits = []
for f, s in srcs.items():
    lines = s.split("\n")
    for i, l in enumerate(lines):
        m = THM.match(l)
        if not m: continue
        # gather until the next top-level declaration
        body = []
        j = i
        while j < len(lines):
            j += 1
            if j >= len(lines): break
            nl = lines[j]
            if THM.match(nl) or re.match(r'^\s*(?:noncomputable\s+)?(def|structure|class|inductive|instance|end|namespace|/-)', nl):
                break
            body.append(nl)
        text = "\n".join(body)
        # proof body after := or `by`
        # Take the LAST top-level `:=`, not the first: the first is often inside a
        # hypothesis binder (`(h : (let mu := ...) = ...)`), which produced two false
        # positives -- a field name lifted out of a HYPOTHESIS and reported as a proof.
        idx = text.rfind(':=')
        if idx < 0: continue
        proof = re.sub(r'--.*', '', text[idx+2:]).strip()
        p = re.sub(r'^by\s+', '', proof).strip()
        p = re.sub(r'^exact\s+', '', p).strip()
        # The WHOLE proof must be the projection. A multi-step tactic block is not this
        # defect, however many `X.field` terms appear inside it.
        p = ' '.join(x.strip() for x in p.split('\n') if x.strip())
        fm = re.fullmatch(r'([A-Za-z_][\w.\']*)\.([a-z_][\w\']*)((?:\s+[\w.\'()\u25b8\u2190:\u211d\-]+)*)', p)
        if fm and fm.group(2) in allfields:
            owners = [st for st, fs in fields.items() if fm.group(2) in fs]
            hits.append((f, i+1, m.group(2), p[:70], owners[:3]))

print(f"scanned {len(files)} .lean files on {REF}")
print(f"structures with fields: {len(fields)}   distinct field names: {len(allfields)}")
print(f"THEOREMS WHOSE WHOLE PROOF IS A FIELD PROJECTION: {len(hits)}\n")
by = collections.Counter(h[0].replace("proofs/Calibrator/","") for h in hits)
for f,c in by.most_common(): print(f"  {c:3}  {f}")
print()
for f,ln,name,p,ow in sorted(hits, key=lambda r:(r[0],r[1])):
    print(f"{f.replace('proofs/Calibrator/','')}:{ln}")
    print(f"    theorem {name}")
    print(f"    proof := {p}      [field of {', '.join(ow)}]")
