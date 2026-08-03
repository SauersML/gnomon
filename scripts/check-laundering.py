#!/usr/bin/env python3
"""Ban proof laundering: a valid Lean proof of a weaker, conditional, vacuous, or
circular statement presented as the intended theorem.

WHAT THIS IS NOT.  It is not a kernel check.  The kernel is not being fooled in any
of the patterns below; every one of them typechecks.  `scripts/check-identifications.py`
guards the source text and `proofs/validation/invariants/AxiomScan.lean` guards the
transitive axiom closure, and NEITHER CAN SEE ANY OF THIS, because a laundered proof
has no `sorry`, no custom axiom, and a clean `#print axioms` report.  The defect is
that the declaration's TYPE is not the advertised mathematics.

THE STANDARD, which is the only one that matters and which no tool applies for you:

    A development has closed a theorem only when the final declaration states exactly
    the intended mathematics, has no unresolved substantive premise (explicit, implicit,
    or instance), constructs every certificate it consumes, instantiates every abstract
    parameter with a concrete object proved to satisfy it, quantifies over a domain
    proved nonempty, and has a clean transitive axiom report.

    Anything less may be a useful conditional library.  It is not the advertised proof.

A `sorry` IS PREFERRED TO EVERY PATTERN BELOW.  A `sorry` is an honest, machine-visible,
kernel-tracked hole that `AxiomScan.lean` reports as `sorryAx`.  A laundered theorem is
an invisible hole that every automated report calls green.  When the intended statement
is not proved, state the intended statement and admit it; do not restate a provable
shadow of it.  This inverts the usual repository rule -- see the LEDGER section in
`scripts/check-identifications.py` -- and it is deliberate.

FAMILIES DETECTED.  Numbering follows the audit taxonomy; `severity` decides exit code.

  FATAL -- the declaration does not prove what its name says.
    F1   hypothesis laundering: the conclusion is one of the hypotheses, verbatim.
    F1b  proof is a bare application of a hypothesis binder (`h`, `h x`, `hDeep prem`).
    F4   certificate laundering: a parameter is a structure carrying the conclusion
         (or any Prop) in a field; the theorem consumes a certificate it never builds.
    F7   conclusion-by-definition: a predicate one of whose conjuncts IS the conclusion.
    F8   definitional weakening: a target property defined as `True` or trivially.
    F9   premise strengthening: premise and conclusion are the same existential.
    F11  inconsistent instance context: a class with a `False` field, or premises
         asserting both `Nontrivial` and `Subsingleton` of one type.
    F24  trust bypass: custom `axiom`, `native_decide`, `unsafe`, custom elaborators.

  CONDITIONAL -- valid implication, but the antecedent is unproved in this corpus.
    F2   Prop alias with a theorem-like name and no inhabitant anywhere.
    F3   typeclass laundering: nonstandard class, `Fact`, `Nonempty`, `Inhabited`, or
         a `letI`/`haveI`-installed instance standing in for a proof.
    F16  wrapper chain: every Prop-valued binder in a theorem's signature.
    F19  hidden assumptions: Prop-valued *implicit* and *instance* binders, plus
         section `variable`s inherited silently.
    F23  conditional bootstrapping: `Nonempty`/`Exists` conclusion whose witness came
         in as an argument.

  FIDELITY -- statement may be right, but nothing ties it to the intended object.
    F5   existential repackaging.
    F6   `Classical.choice`/`choose` applied to an assumed existence premise.
    F10  vacuity: quantification over a domain with no inhabitant proved in-corpus.
    F12  subtype laundering: domain is `{x // DesiredProperty x}`.
    F13  a `.range`/image construction named as if it were the canonical object.
    F14  concrete-looking dead end: a concrete construction no headline theorem uses.
    F17  name inflation: `_complete`, `_proved`, `_exists`, `explicit_` on a
         declaration that still carries premises.
    F20  semantic shadowing: a corpus predicate reusing a standard name.
    F21  degenerate normalization: division by a quantity not proved nonzero.

USAGE
    scripts/check-laundering.py                  # whole corpus, human report
    scripts/check-laundering.py --severity fatal # only the fatal families
    scripts/check-laundering.py --json out.json  # machine-readable
    scripts/check-laundering.py path/to/File.lean ...

Exit status is 1 if any FATAL or CONDITIONAL finding survives.  There is deliberately
NO SUPPRESSION FILE.  A ledger of accepted laundering is how a corpus normalises it;
if a finding is wrong, fix the detector and say why in this docstring.

LIMITS, stated so a clean report is not over-read.  This is a source-text analysis: it
sees what was typed, not what the elaborator produced.  It cannot see premises
introduced by `export`ed instances from an import, a `Fact` synthesised at elaboration
time, or a definition unfolded to something other than its written form.  The
environment-level companion, `proofs/validation/invariants/LaunderingScan.lean`, walks
the fully elaborated telescope of every `Calibrator` declaration and is authoritative
where the two disagree.  Run both.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROOFS = REPO / "proofs"

# --------------------------------------------------------------------------------------
# Severity
# --------------------------------------------------------------------------------------

FATAL, CONDITIONAL, FIDELITY = "FATAL", "CONDITIONAL", "FIDELITY"
SEVERITY_ORDER = {FATAL: 0, CONDITIONAL: 1, FIDELITY: 2}

# WHAT GATES, AND WHY IT IS NOT EVERYTHING.
#
# FATAL gates.  Those families are laundering proper: the declaration does not prove
# what its name says, and no amount of context makes that acceptable.
#
# CONDITIONAL does not gate by default, and this is a deliberate line rather than a
# concession.  A theorem of the form `(hx : 0 < x) : f x < f (x + 1)` has a Prop-valued
# premise and is ordinary mathematics; so does every theorem quantified over a model
# class that the corpus proves nonempty.  Gating on those would make the guard
# permanently red, and a permanently red guard is read as broken and then ignored --
# which is how the patterns it exists to catch get in.  The CONDITIONAL count is
# printed as a LEDGER on every run: it is the honest size of what this corpus assumes,
# it is meant to be read, and it is meant to go down.
#
# `--strict` gates on CONDITIONAL as well.  That is the standard the corpus is aiming
# at, and the flag exists so the aim is checkable rather than aspirational.
EXIT_ON = {FATAL}

FAMILY_SEVERITY = {
    "F1": FATAL, "F1b": FATAL, "F4": FATAL, "F7": FATAL, "F8": FATAL,
    "F9": FATAL, "F11": FATAL, "F24": FATAL,
    "F2": CONDITIONAL, "F3": CONDITIONAL, "F16": CONDITIONAL,
    "F19": CONDITIONAL, "F23": CONDITIONAL, "F22": FIDELITY,
    "F5": FIDELITY, "F6": FIDELITY, "F10": FIDELITY, "F12": FIDELITY,
    "F13": FIDELITY, "F14": FIDELITY, "F17": FIDELITY, "F20": FIDELITY,
    "F21": FIDELITY,
}

FAMILY_TITLE = {
    "F1": "hypothesis laundering (conclusion is a hypothesis)",
    "F1b": "proof is application of a hypothesis",
    "F2": "Prop alias with theorem-like name, never inhabited",
    "F3": "typeclass laundering",
    "F4": "certificate laundering (structure parameter carrying Props)",
    "F5": "existential repackaging",
    "F6": "choice applied to an assumed existence premise",
    "F7": "conclusion-by-definition",
    "F8": "definitional weakening",
    "F9": "premise strengthening to tautology",
    "F10": "vacuity: domain with no inhabitant proved",
    "F11": "inconsistent instance context",
    "F12": "subtype laundering",
    "F13": "range/image advertised as canonical construction",
    "F14": "concrete construction no theorem consumes",
    "F16": "conditional signature (Prop-valued premises)",
    "F17": "theorem-name inflation",
    "F19": "hidden premise (implicit/instance/section variable)",
    "F20": "semantic shadowing of a standard name",
    "F22": "artificially narrow universe (bundled-premise parameter)",
    "F21": "degenerate normalization (unguarded denominator)",
    "F23": "conditional bootstrapping (witness supplied as argument)",
    "F24": "trust bypass",
}

# --------------------------------------------------------------------------------------
# Lexing: mask comments and string literals, preserving offsets
# --------------------------------------------------------------------------------------


def mask(src: str) -> str:
    """Replace comment and string content with spaces, keeping every offset and newline.

    Structural scans (delimiter depth, `:=` position) must not see a `(` inside a
    docstring.  Offsets are preserved so a match in the masked text indexes the original.
    """
    out = list(src)
    i, n = 0, len(src)
    depth = 0  # block-comment nesting; Lean's /- -/ nests
    while i < n:
        c = src[i]
        if depth:
            if src.startswith("/-", i):
                depth += 1
                out[i] = out[i + 1] = " "
                i += 2
                continue
            if src.startswith("-/", i):
                depth -= 1
                out[i] = out[i + 1] = " "
                i += 2
                continue
            if c != "\n":
                out[i] = " "
            i += 1
            continue
        if src.startswith("/-", i):
            depth = 1
            out[i] = out[i + 1] = " "
            i += 2
            continue
        if src.startswith("--", i):
            while i < n and src[i] != "\n":
                out[i] = " "
                i += 1
            continue
        if c == '"':
            out[i] = " "
            i += 1
            while i < n and src[i] != '"':
                if src[i] == "\\":
                    out[i] = " "
                    i += 1
                    if i < n:
                        out[i] = " "
                        i += 1
                    continue
                if src[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                out[i] = " "
                i += 1
            continue
        i += 1
    return "".join(out)


OPEN = {"(": ")", "{": "}", "[": "]", "⦃": "⦄", "⟨": "⟩"}
CLOSE = {v: k for k, v in OPEN.items()}


def depths(text: str) -> list[int]:
    """Delimiter depth before each character."""
    d, out = 0, []
    for c in text:
        out.append(d)
        if c in OPEN:
            d += 1
        elif c in CLOSE:
            d = max(0, d - 1)
    return out


# --------------------------------------------------------------------------------------
# Declaration model
# --------------------------------------------------------------------------------------

DECL_KINDS = (
    "theorem", "lemma", "def", "abbrev", "structure", "class", "instance",
    "inductive", "example", "opaque", "axiom",
)
DECL_RE = re.compile(
    r"^(?:@\[[^\]]*\]\s*)?"
    r"(?:(?:private|protected|noncomputable|partial|unsafe|scoped|local)\s+)*"
    r"(" + "|".join(DECL_KINDS) + r")\b[ \t]*"
    r"([A-Za-z_À-ɏͰ-Ͽ][\w.'À-ɏͰ-Ͽ]*)?",
    re.M,
)
STOP_RE = re.compile(
    r"^(?:@\[[^\]]*\]\s*)?"
    r"(?:(?:private|protected|noncomputable|partial|unsafe|scoped|local)\s+)*"
    r"(?:" + "|".join(DECL_KINDS) + r"|namespace|end|section|open|import|variable|"
    r"universe|attribute|macro|elab|syntax|notation|run_cmd|#\w+|/-)\b",
    re.M,
)


@dataclass
class Binder:
    name: str
    type: str
    kind: str          # "explicit" | "implicit" | "instance" | "strict"
    inherited: bool = False   # came from a section `variable`


@dataclass
class Decl:
    file: str
    line: int
    kind: str
    name: str
    header: str        # binders + `: conclusion`
    conclusion: str
    body: str
    doc: str
    binders: list[Binder] = dc_field(default_factory=list)
    attrs: str = ""


@dataclass
class Finding:
    family: str
    file: str
    line: int
    decl: str
    detail: str

    @property
    def severity(self) -> str:
        return FAMILY_SEVERITY[self.family]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def split_binders(header: str) -> tuple[list[Binder], str]:
    """Split a declaration header into binders and the conclusion after the top-level `:`.

    A binder is a balanced delimiter group at depth 0; the conclusion is whatever follows
    the first depth-0 `:` that is not inside such a group.  Bare `{α}`-style anonymous
    binders and `∀`-bound variables inside a binder type stay inside that binder's type.
    """
    d = depths(header)
    binders: list[Binder] = []
    i, n = 0, len(header)
    concl_start = None
    while i < n:
        c = header[i]
        if d[i] == 0 and c in OPEN and c != "⟨":
            j = i
            depth = 0
            while j < n:
                if header[j] in OPEN:
                    depth += 1
                elif header[j] in CLOSE:
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            group = header[i : j + 1]
            binders.extend(parse_binder_group(group))
            i = j + 1
            continue
        if d[i] == 0 and c == ":":
            concl_start = i + 1
            break
        i += 1
    conclusion = header[concl_start:] if concl_start is not None else ""
    return binders, norm(conclusion)


def parse_binder_group(group: str) -> list[Binder]:
    kind = {"(": "explicit", "{": "implicit", "[": "instance", "⦃": "strict"}[group[0]]
    inner = group[1:-1]
    d = depths(inner)
    colon = next((k for k, c in enumerate(inner) if c == ":" and d[k] == 0), None)
    if colon is None:
        # `[Group G]` -- anonymous instance, or `{α}` -- anonymous implicit.
        return [Binder(name="", type=norm(inner), kind=kind)]
    names = norm(inner[:colon])
    ty = norm(inner[colon + 1 :])
    return [Binder(name=nm, type=ty, kind=kind) for nm in names.split()] or [
        Binder(name="", type=ty, kind=kind)
    ]


def split_header_body(text: str) -> tuple[str, str]:
    """Split at the `:=` that opens the proof/definition body.

    Not the first `:=` in the text: a `let` inside a hypothesis binder or inside the
    conclusion has one too.  Take the first depth-0 `:=` whose line does not begin a
    `let`/`have`/`set`/`obtain`/`fun` -- those are body-internal.
    """
    d = depths(text)
    for m in re.finditer(r":=", text):
        i = m.start()
        if d[i] != 0:
            continue
        ls = text.rfind("\n", 0, i) + 1
        if re.match(r"\s*(let|have|set|obtain|fun|match|if|with)\b", text[ls:i]):
            continue
        return text[:i], text[i + 2 :]
    return text, ""


def parse_file(path: Path) -> tuple[list[Decl], list[tuple[int, str]]]:
    src = path.read_text(encoding="utf-8", errors="replace")
    m = mask(src)
    # Paths outside the repo must stay scannable: the calibration fixture writes to a
    # temp dir, and a detector that only runs on the corpus it judges cannot be tested
    # against known answers.
    try:
        rel = str(path.relative_to(REPO))
    except ValueError:
        rel = str(path)

    starts = []
    for mo in DECL_RE.finditer(m):
        starts.append((mo.start(), mo.group(1), mo.group(2) or "", mo.end()))

    # Section-scoped `variable` binders, tracked as (offset, group_text).
    variables: list[tuple[int, str]] = []
    for mo in re.finditer(r"^variable\b(.*)$", m, re.M):
        variables.append((mo.start(), src[mo.start(1) : mo.end(1)]))

    decls: list[Decl] = []
    for idx, (off, kind, name, hdr_off) in enumerate(starts):
        nxt = len(src)
        for mo in STOP_RE.finditer(m, hdr_off):
            nxt = mo.start()
            break
        # `mask` blanks comments, so STOP_RE cannot see the `/--` that opens the next
        # declaration's docstring.  Stop at a comment opener in the ORIGINAL text too, or
        # a declaration's body runs on into its neighbour's prose.
        cm = re.search(r"^\s*/-", src[hdr_off:nxt], re.M)
        if cm:
            nxt = hdr_off + cm.start()
        raw = src[off:nxt]
        raw_m = m[off:nxt]
        # docstring immediately above
        pre = src[max(0, off - 4000) : off]
        doc = ""
        dm = re.search(r"/--(.*?)-/\s*$", pre, re.S)
        if dm:
            doc = norm(dm.group(1))
        header_m, body_m = split_header_body(raw_m[hdr_off - off :])
        # Header and body come from the MASKED text.  Slicing the original at masked
        # offsets leaves comments inside the slice, and every check that compares a body
        # against an exact string then fails silently on a trailing `--` line.
        header = header_m
        body = body_m
        binders, conclusion = split_binders(header_m)
        # restore original (unmasked) text for binder types where possible
        line = src.count("\n", 0, off) + 1
        inherited = []
        for voff, vtext in variables:
            if voff < off:
                inherited.extend(
                    b for b in parse_binder_group_line(vtext)
                )
        for b in inherited:
            b.inherited = True
        attrs = ""
        am = re.match(r"^(@\[[^\]]*\])", raw)
        if am:
            attrs = am.group(1)
        decls.append(
            Decl(
                file=rel, line=line, kind=kind, name=name,
                header=norm(header), conclusion=conclusion,
                body=norm(body), doc=doc, binders=binders + inherited, attrs=attrs,
            )
        )
    return decls, variables


def parse_binder_group_line(text: str) -> list[Binder]:
    """Parse the binder groups on a `variable ...` line."""
    out: list[Binder] = []
    d = depths(text)
    i, n = 0, len(text)
    while i < n:
        if d[i] == 0 and text[i] in OPEN and text[i] != "⟨":
            j, depth = i, 0
            while j < n:
                if text[j] in OPEN:
                    depth += 1
                elif text[j] in CLOSE:
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            out.extend(parse_binder_group(text[i : j + 1]))
            i = j + 1
            continue
        i += 1
    return out


# --------------------------------------------------------------------------------------
# Prop-ness
# --------------------------------------------------------------------------------------

REL = ["=", "≠", "≤", "<", "≥", ">", "∈", "∉", "⊆", "⊂", "≡", "≈", "∼", "∣"]
LOGIC = ["∀", "∃", "¬", "∧", "∨", "↔"]

# Mathlib classes that are ordinary algebraic structure, not smuggled mathematics.
STANDARD_CLASSES = {
    "Fintype", "DecidableEq", "Decidable", "MeasurableSpace", "NormedAddCommGroup",
    "NormedSpace", "InnerProductSpace", "TopologicalSpace", "MetricSpace", "Group",
    "AddCommGroup", "CommRing", "Field", "LinearOrder", "Preorder", "PartialOrder",
    "Module", "Ring", "Monoid", "AddMonoid", "CommMonoid", "SeminormedAddCommGroup",
    "MeasureSpace", "IsProbabilityMeasure", "IsFiniteMeasure", "CompleteSpace",
    "SecondCountableTopology", "BorelSpace", "OpensMeasurableSpace", "T2Space",
    "Countable", "Encodable", "Semiring", "CommSemiring", "Algebra", "Star",
    "ContinuousMul", "ContinuousAdd", "MeasurableSingletonClass", "SFinite",
    "IsFiniteKernel", "IsMarkovKernel", "NeZero", "CharZero", "Nontrivial",
    "Subsingleton", "Unique", "FunLike", "Coe", "CoeFun", "Repr", "ToString",
    "Zero", "One", "Add", "Mul", "Neg", "Inv", "Sub", "Div", "Pow", "SMul",
    "LE", "LT", "HAdd", "HMul", "Membership", "Insert", "Singleton", "Lattice",
    "OrderedAddCommGroup", "LinearOrderedField", "RCLike", "IsROrC", "Norm",
    "Dist", "EDist", "PseudoMetricSpace", "UniformSpace", "ProperSpace",
    "FiniteDimensional", "Basis", "NormedRing", "NormedField", "NNRealAlgebra",
}
# Classes whose whole content is an assumption, regardless of who defined them.
ASSUMPTION_CLASSES = {"Fact", "Nonempty", "Inhabited"}


def is_prop_type(ty: str, prop_aliases: set[str], prop_structs: set[str]) -> bool:
    """Whether a binder type is a PROPOSITION (a hypothesis), as opposed to data.

    `(f : α → Prop)` is data -- an abstract predicate parameter, reported separately.
    `(h : ∀ x, f x)` is a proposition.  The discriminator is the head, not the presence
    of the token `Prop`.
    """
    t = norm(ty)
    if not t:
        return False
    if re.search(r"(→|->)\s*Prop$", t) or t == "Prop":
        return False          # predicate-valued data, not an assumption
    if any(t.startswith(k + " ") or t == k for k in LOGIC) or t.startswith("¬"):
        return True
    d = depths(t)
    for op in REL + LOGIC:
        for k in range(len(t) - len(op) + 1):
            if t[k : k + len(op)] == op and d[k] == 0:
                return True
    head = re.match(r"([A-Za-z_][\w.']*)", t)
    if head:
        h = head.group(1)
        base = h.split(".")[-1]
        if h in prop_aliases or base in prop_aliases:
            return True
        if base in prop_structs:
            return True
        if base in ASSUMPTION_CLASSES:
            return True
        if re.match(r"^(Is|Has)[A-Z]", base) and base not in STANDARD_CLASSES:
            return True
    return False


# --------------------------------------------------------------------------------------
# Corpus index
# --------------------------------------------------------------------------------------


@dataclass
class Corpus:
    decls: list[Decl]
    prop_aliases: set[str]                     # `def X : Prop`
    struct_fields: dict[str, list[Binder]]     # structure/class -> fields
    prop_structs: set[str]                     # structures with >=1 Prop field
    inhabited: set[str]                        # types with an in-corpus inhabitant
    used_names: dict[str, int]                 # identifier -> occurrence count


FIELD_RE = re.compile(r"^\s{2,}([a-zA-Z_][\w']*)\s*:(?!=)(.*)$")


def build_corpus(files: list[Path]) -> Corpus:
    decls: list[Decl] = []
    for f in files:
        d, _ = parse_file(f)
        decls.extend(d)

    prop_aliases = {
        d.name for d in decls
        if d.kind in ("def", "abbrev") and norm(d.conclusion) == "Prop"
    }

    struct_fields: dict[str, list[Binder]] = {}
    for f in files:
        src = mask(f.read_text(encoding="utf-8", errors="replace"))
        cur = None
        for line in src.split("\n"):
            sm = re.match(
                r"^(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|noncomputable)\s+)*"
                r"(structure|class)\s+([A-Za-z_][\w.']*)", line)
            if sm:
                cur = sm.group(2).split(".")[-1]
                struct_fields.setdefault(cur, [])
                continue
            if cur is None:
                continue
            if line.strip() and not line.startswith((" ", "\t")):
                cur = None
                continue
            fm = FIELD_RE.match(line)
            if fm and not line.strip().startswith(("--", "|", "/-")):
                struct_fields[cur].append(
                    Binder(name=fm.group(1), type=norm(fm.group(2)), kind="field"))

    prop_structs: set[str] = set()
    # Fixed point: a structure with a Prop field, or with a field whose type is a
    # structure already known to carry Props, is itself a certificate.
    for _ in range(6):
        grew = False
        for s, fs in struct_fields.items():
            if s in prop_structs:
                continue
            if any(is_prop_type(b.type, prop_aliases, prop_structs) for b in fs):
                prop_structs.add(s)
                grew = True
        if not grew:
            break

    inhabited: set[str] = set()
    for d in decls:
        c_ = norm(d.conclusion)
        m = re.match(r"Nonempty\s+\(?([A-Za-z_][\w.']*)", c_)
        if m:
            inhabited.add(m.group(1).split(".")[-1])
        # Theorems count too.  A Prop-valued structure (`IsRankAllocation k M : Prop`)
        # is inhabited by a THEOREM proving it holds of some `k` and `M`, and a data
        # structure by a theorem concluding `Nonempty S`, matched above.
        if d.kind not in ("def", "abbrev", "instance", "theorem", "lemma"):
            continue
        # A WITNESS MAY TAKE DATA AND MAY NOT TAKE HYPOTHESES.  `f (k : ℕ) (β : ℝ) : S k`
        # builds the structure for every choice of its numeric inputs, so `S` is inhabited.
        # `f (h : HardProblemSolved) : S` builds nothing: it moves the obligation to the
        # caller, which is the pattern this file exists to catch.
        if any(is_prop_type(b.type, prop_aliases, prop_structs)
               for b in d.binders if not b.inherited):
            continue
        head = re.match(r"([A-Za-z_][\w.']*)", c_)
        if head:
            inhabited.add(head.group(1).split(".")[-1])
        if d.kind == "instance":
            for tok in re.findall(r"[A-Za-z_][\w.']*", c_):
                inhabited.add(tok.split(".")[-1])

    used: dict[str, int] = defaultdict(int)
    for f in files:
        for tok in re.findall(r"[A-Za-z_][\w.']*", mask(
                f.read_text(encoding="utf-8", errors="replace"))):
            used[tok.split(".")[-1]] += 1

    return Corpus(decls, prop_aliases, struct_fields, prop_structs, inhabited, used)


# --------------------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------------------

THEOREMISH = re.compile(
    r"(?i)(theorem|conjecture|lemma|principle|law|classification|result)$")
INFLATED = re.compile(
    r"(?i)(_complete|_proved|_holds|_exists$|^explicit_|_established|_settled"
    r"|_construction$|_theorem$|_conjecture$)")
STANDARD_PREDICATES = {
    "IsSofic", "IsFinitelyPresented", "HasPropertyT", "IsAmenable", "IsCompact",
    "IsOpen", "IsClosed", "IsIntegral", "Measurable", "Continuous", "Integrable",
    "IsUnit", "IsNoetherian", "IsSeparable", "IsErgodic", "IsStationary",
    "IsProbabilityMeasure", "IsMartingale", "Convex", "Differentiable",
}


def analyse(corpus: Corpus) -> list[Finding]:
    out: list[Finding] = []
    proved_props: set[str] = set()
    for d in corpus.decls:
        if d.kind in ("theorem", "lemma"):
            head = re.match(r"([A-Za-z_][\w.']*)", d.conclusion)
            if head and not [b for b in d.binders
                             if is_prop_type(b.type, corpus.prop_aliases,
                                             corpus.prop_structs)]:
                proved_props.add(head.group(1).split(".")[-1])

    for d in corpus.decls:
        out.extend(check_decl(d, corpus, proved_props))
    out.extend(check_files(corpus))
    return out


def hypotheses(d: Decl, c: Corpus) -> list[Binder]:
    return [b for b in d.binders if is_prop_type(b.type, c.prop_aliases, c.prop_structs)]


def check_decl(d: Decl, c: Corpus, proved_props: set[str]) -> list[Finding]:
    f: list[Finding] = []
    add = lambda fam, detail: f.append(Finding(fam, d.file, d.line, d.name, detail))
    hyps = hypotheses(d, c)
    concl = norm(d.conclusion)
    body = norm(d.body)

    if d.kind in ("theorem", "lemma", "example"):
        # F1 -- the conclusion is verbatim one of the hypotheses.
        for b in hyps:
            if norm(b.type) == concl and concl:
                add("F1", f"conclusion is hypothesis `{b.name} : {b.type}`")
                break
        else:
            # F9 -- premise and conclusion are the same existential, modulo binder name.
            for b in hyps:
                if concl and _same_existential(b.type, concl):
                    add("F9", f"premise `{b.name}` is the conclusion up to renaming")
                    break

        # F1b -- the whole proof is an application of a binder.
        p = re.sub(r"^by\s+", "", body).strip()
        p = re.sub(r"^exact\s+", "", p).strip()
        p = re.sub(r"^intro[s]?\s+[\w\s]*;?\s*", "", p).strip()
        m = re.fullmatch(r"([A-Za-z_][\w']*)((?:\.[a-zA-Z_][\w']*)*)((?:\s+\S+)*)", p)
        if m:
            root = m.group(1)
            names = {b.name for b in d.binders if b.name}
            if root in names:
                b = next(x for x in d.binders if x.name == root)
                fld = m.group(2).lstrip(".").split(".")[0] if m.group(2) else ""
                owners = [s for s, fs in c.struct_fields.items()
                          if any(x.name == fld for x in fs)] if fld else []
                # Only two shapes are content-free:
                #
                #   `h`            -- the proof IS the premise.
                #   `h.field args` -- the proof is a field the caller filled in.
                #
                # `h s` is modus ponens: unfolding a definition at a point, or applying a
                # premise to a theorem the corpus proves, is ordinary mathematics.  Its
                # conditionality is real and F16 is where that is reported.
                if not fld and not m.group(3).strip() and \
                        is_prop_type(b.type, c.prop_aliases, c.prop_structs):
                    add("F1b", f"proof is the bare premise `{root}`: the theorem "
                               f"restates its own hypothesis")
                elif owners:
                    add("F4", f"proof is projection `{p[:60]}` "
                              f"[field of {', '.join(owners[:3])}]")

        # F4 -- a parameter is a certificate structure.
        for b in d.binders:
            head = re.match(r"([A-Za-z_][\w.']*)", norm(b.type))
            if not head:
                continue
            base = head.group(1).split(".")[-1]
            if base in c.prop_structs:
                carrying = [x.name for x in c.struct_fields.get(base, [])
                            if is_prop_type(x.type, c.prop_aliases, c.prop_structs)]
                if base not in c.inhabited:
                    add("F4", f"parameter `{b.name} : {base}` is a certificate "
                              f"(Prop fields: {', '.join(carrying[:4])}) and no "
                              f"inhabitant of `{base}` is constructed in-corpus")
                else:
                    add("F22", f"parameter `{b.name} : {base}` bundles the premises "
                               f"{', '.join(carrying[:4])}; the universe quantified "
                               f"over is narrower than the noun suggests")

        # F16/F19 -- any remaining Prop-valued premise.
        for b in hyps:
            head = re.match(r"([A-Za-z_][\w.']*)", norm(b.type))
            base = head.group(1).split(".")[-1] if head else ""
            if base in c.prop_structs:
                continue      # already reported as F4
            if b.kind in ("implicit", "instance", "strict") or b.inherited:
                add("F19", f"hidden premise `{b.kind}{' (section variable)' if b.inherited else ''}"
                           f" {b.name} : {_clip(b.type)}`")
            else:
                add("F16", f"premise `{b.name} : {_clip(b.type)}`")

        # F3 -- typeclass laundering.
        for b in d.binders:
            if b.kind != "instance":
                continue
            head = re.match(r"([A-Za-z_][\w.']*)", norm(b.type))
            if not head:
                continue
            base = head.group(1).split(".")[-1]
            if base in ASSUMPTION_CLASSES or (
                base in c.struct_fields and base not in STANDARD_CLASSES
            ):
                add("F3", f"instance premise `[{_clip(b.type)}]`")

        # F11 -- contradictory instance context.
        insts = [norm(b.type) for b in d.binders if b.kind == "instance"]
        for a in insts:
            if a.startswith("Nontrivial"):
                arg = a[len("Nontrivial"):].strip()
                if any(x.strip() == f"Subsingleton {arg}" for x in insts):
                    add("F11", f"premises assert both `Nontrivial {arg}` and "
                               f"`Subsingleton {arg}`: the context is empty")

        # F23 -- existential conclusion whose witness is a parameter.
        if re.match(r"(Nonempty|∃|Exists)\b", concl):
            wit = re.match(r"\s*⟨\s*([A-Za-z_][\w.']*)", re.sub(r"^by\s+", "", body))
            if wit and wit.group(1).split(".")[0] in {b.name for b in d.binders if b.name}:
                add("F23", f"existence proved by wrapping the parameter "
                           f"`{wit.group(1)}`")
            elif any(re.match(r"(Nonempty|∃|Exists)\b", norm(b.type)) for b in hyps):
                add("F5", "existential conclusion repackaging an existential premise")

        # F17 -- name inflation on a conditional statement.
        if INFLATED.search(d.name) and (hyps or any(
                b.kind == "instance" and
                re.match(r"([A-Za-z_][\w.']*)", norm(b.type)) and
                re.match(r"([A-Za-z_][\w.']*)", norm(b.type)).group(1).split(".")[-1]
                not in STANDARD_CLASSES
                for b in d.binders)):
            add("F17", f"name claims a closed result but the signature carries "
                       f"{len(hyps)} premise(s)")

        # F6 -- choice on an assumed existence premise.
        for m in re.finditer(r"Classical\.(choice|choose|arbitrary|some)\s+([A-Za-z_][\w.']*)",
                             body):
            if m.group(2).split(".")[0] in {b.name for b in hyps if b.name}:
                add("F6", f"`Classical.{m.group(1)}` applied to premise `{m.group(2)}`")

        # F10/F12 -- vacuous or self-satisfying domain.
        for b in d.binders:
            t = norm(b.type)
            if re.match(r"(Empty|PEmpty|Fin 0)\b", t):
                add("F10", f"quantifies over the empty type `{t}`")
            sm = re.match(r"\{\s*\w+\s*//\s*(.+)\}$", t)
            if sm:
                base = _head_ident(t)
                add("F12", f"domain is the subtype `{_clip(t)}`; "
                           f"no `Nonempty` for it is proved in-corpus"
                    if base not in c.inhabited else
                    f"domain is the subtype `{_clip(t)}`")

    if d.kind in ("def", "abbrev"):
        # F8 -- definitional weakening.
        if norm(d.conclusion) == "Prop" and norm(d.body) in ("True", "trivial", "⊤"):
            add("F8", "target property is defined as `True`")
        # F2 -- Prop alias with a theorem-like name and no inhabitant.
        if norm(d.conclusion) == "Prop" and THEOREMISH.search(d.name):
            if d.name.split(".")[-1] not in proved_props:
                add("F2", "Prop named like a theorem, with no proof of it in-corpus")
        # F7 -- conclusion-by-definition.
        if norm(d.conclusion) == "Prop":
            parts = _top_conjuncts(d.body)
            if len(parts) > 1:
                for p in parts:
                    if re.search(r"(?i)\b(correct|desired|conclusion|holds|valid|"
                                 r"calibrat|portab|identif|sound|complete)\w*\b", p):
                        add("F7", f"predicate `{d.name}` has the advertised conclusion "
                                  f"as a conjunct: `{_clip(p)}`")
                        break
        # F13 -- range advertised as canonical.
        if re.search(r"\.range\b|Set\.image\b", d.body) and re.search(
                r"(?i)(canonical|universal|the standard|concrete copy)", d.doc):
            add("F13", "a range/image construction is described as canonical or "
                       "universal with no isomorphism theorem cited")
        # F20 -- semantic shadowing.
        if d.name.split(".")[-1] in STANDARD_PREDICATES:
            add("F20", f"redefines the standard predicate `{d.name}` locally")
        # F21 -- unguarded denominator.
        for m in re.finditer(r"/\s*([A-Za-z_][\w.']*)", d.body):
            den = m.group(1)
            if den in {b.name for b in d.binders if b.name} and not re.search(
                    rf"{re.escape(den)}\s*(≠|>)\s*0", d.header):
                add("F21", f"divides by parameter `{den}` with no `{den} ≠ 0` premise; "
                           f"Lean makes the quotient 0 and the statement vacuous there")
                break

    if d.kind == "structure" or d.kind == "class":
        fields = c.struct_fields.get(d.name.split(".")[-1], [])
        if any(norm(x.type) == "False" for x in fields):
            add("F11", f"`{d.name}` has a field of type `False`: every theorem "
                       f"assuming it is vacuous")

    if d.kind == "axiom":
        add("F24", f"custom axiom `{d.name}` expands the trusted base")

    return f


def _clip(s: str, n: int = 70) -> str:
    s = norm(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _head_ident(t: str) -> str:
    m = re.match(r"[({\[]*\s*([A-Za-z_][\w.']*)", norm(t))
    return m.group(1).split(".")[-1] if m else ""


def _same_existential(a: str, b: str) -> bool:
    ra, rb = norm(a), norm(b)
    if not ra.startswith("∃") or not rb.startswith("∃"):
        return False
    strip = lambda s: re.sub(r"[a-zA-Z_][\w']*", "·", s)
    return strip(ra) == strip(rb)


def _top_conjuncts(s: str) -> list[str]:
    s = norm(s)
    d = depths(s)
    parts, last = [], 0
    for i, ch in enumerate(s):
        if ch == "∧" and d[i] == 0:
            parts.append(s[last:i])
            last = i + 1
    parts.append(s[last:])
    return [p.strip() for p in parts if p.strip()]


def check_files(c: Corpus) -> list[Finding]:
    """Whole-file syntax that bypasses trust, and dead concrete constructions."""
    out: list[Finding] = []
    seen_files = {d.file for d in c.decls}
    for rel in sorted(seen_files):
        src = (REPO / rel).read_text(encoding="utf-8", errors="replace")  # abs rel is a no-op
        m = mask(src)
        for pat, fam, msg in [
            (r"\bnative_decide\b", "F24", "`native_decide` moves the compiler into the "
                                          "trusted base"),
            (r"^\s*(unsafe|opaque)\s+", "F24", "unsafe/opaque declaration"),
            (r"^\s*(macro|elab|syntax|macro_rules|elab_rules)\b", "F24",
             "custom syntax or elaborator"),
            (r"@\[implemented_by", "F24", "compiled implementation substituted for the "
                                          "definition"),
            (r"^\s*(letI|haveI)\b", "F3", "instance installed locally; the obligation "
                                          "moves to whoever supplied its argument"),
            (r"#print axioms", "F24_INFO", "`#print axioms` reports global axioms only; "
                                           "it cannot see premises or wrong statements"),
        ]:
            for mo in re.finditer(pat, m, re.M):
                if fam == "F24_INFO":
                    continue
                out.append(Finding(fam, rel, m.count("\n", 0, mo.start()) + 1,
                                   "<file>", msg))
    # F14 -- concrete constructions nothing consumes.
    for d in c.decls:
        if d.kind not in ("def", "abbrev", "instance"):
            continue
        base = d.name.split(".")[-1]
        if not base or c.used_names.get(base, 0) > 1:
            continue
        if re.search(r"(?i)(concrete|explicit|witness|example|construct)", d.name + d.doc):
            out.append(Finding("F14", d.file, d.line, d.name,
                               "concrete construction with no consumer: the headline "
                               "still takes an abstract parameter"))
    return out


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*", help="files to check (default: all of proofs/)")
    ap.add_argument("--severity", choices=["fatal", "conditional", "all"],
                    default="all")
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--summary", action="store_true",
                    help="counts per family and per file only")
    ap.add_argument("--family", action="append",
                    help="restrict to one family, e.g. --family F1")
    ap.add_argument("--strict", action="store_true",
                    help="also fail on CONDITIONAL findings (the target standard: no "
                         "unresolved premise anywhere in the corpus)")
    args = ap.parse_args()

    if args.paths:
        files = [Path(p).resolve() for p in args.paths]
    else:
        files = sorted(PROOFS.rglob("*.lean"))
    files = [f for f in files if f.is_file()]

    corpus = build_corpus(files)
    findings = analyse(corpus)

    keep = {"fatal": {FATAL}, "conditional": {FATAL, CONDITIONAL},
            "all": {FATAL, CONDITIONAL, FIDELITY}}[args.severity]
    findings = [f for f in findings if f.severity in keep]
    if args.family:
        findings = [f for f in findings if f.family in set(args.family)]

    findings.sort(key=lambda f: (SEVERITY_ORDER[f.severity], f.family, f.file, f.line))

    by_family = defaultdict(int)
    by_file = defaultdict(int)
    for f in findings:
        by_family[f.family] += 1
        by_file[f.file] += 1

    print(f"scanned {len(files)} .lean files, "
          f"{sum(1 for d in corpus.decls if d.kind in ('theorem','lemma'))} theorems, "
          f"{len(corpus.struct_fields)} structures "
          f"({len(corpus.prop_structs)} carrying Props)")
    print()
    for sev in (FATAL, CONDITIONAL, FIDELITY):
        fams = sorted(x for x in by_family if FAMILY_SEVERITY[x] == sev)
        if not fams:
            continue
        print(f"{sev}")
        for fam in fams:
            print(f"  {by_family[fam]:6}  {fam:4} {FAMILY_TITLE[fam]}")
    print()
    print(f"TOTAL {len(findings)}")

    if not args.summary:
        print()
        cur = None
        for f in findings:
            if f.family != cur:
                cur = f.family
                print(f"\n=== {f.family} [{f.severity}] {FAMILY_TITLE[f.family]} "
                      f"({by_family[f.family]}) ===")
            print(f"{f.file}:{f.line}  {f.decl}\n      {f.detail}")

    if args.json:
        Path(args.json).write_text(json.dumps(
            [dict(family=f.family, severity=f.severity, file=f.file, line=f.line,
                  decl=f.decl, detail=f.detail) for f in findings], indent=1))

    gate = EXIT_ON | ({CONDITIONAL} if args.strict else set())
    n_gated = sum(1 for f in findings if f.severity in gate)
    if n_gated:
        print(f"\nFAIL: {n_gated} finding(s) at or above the gating severity "
              f"({'FATAL+CONDITIONAL' if args.strict else 'FATAL'}).")
    else:
        print(f"\nPASS: no findings at the gating severity "
              f"({'FATAL+CONDITIONAL' if args.strict else 'FATAL'}). "
              f"The CONDITIONAL ledger above is still debt, not absolution.")
    return 1 if n_gated else 0


if __name__ == "__main__":
    sys.exit(main())
