#!/usr/bin/env python3
"""Structural guards for the proof corpus.

Guards, in order of what they catch:

1. sorry ledger. `sorry` is permitted only where an Identification records an
   undischarged obligation, and every one must be listed in SORRY_LEDGER
   below. An unlisted sorry fails. This makes honest debt cheap to declare and
   impossible to accumulate silently.

2. Falsified claims. Nothing may remain marked Evidence.falsified.

3. Convention drift. Every numeric literal 2 or 4 used as a multiplier inside
   a definition is a restatement of a ploidy or coalescent-scaling convention.
   The count is pinned; adding new inline restatements without relating them
   to `ploidy` in Conventions.lean fails, so the number can only go down.

4. Equilibria with no dynamic. A definition named for a rest point or a limit
   must be derived as the fixed point of a process defined in the same file,
   not stipulated as a closed form that no theorem can contradict.

5. Duplicate bodies across files. Two definitions in different modules whose
   bodies are alpha-equivalent are one quantity written twice; unless one calls
   the other or a theorem equates them, fixing one leaves the other wrong.

6. Regimes baked into bodies. A definition whose value depends on an assumption
   about the data-generating process -- closed population, no mutation, infinite
   sites -- must name that assumption, because a formula carries no record of
   the regime it was derived in and a use site cannot discharge what it cannot
   see.

7. Validation inherited from a sibling identity. Over-determination detects
   divergence between formulas and is provably blind to a premise they share, so
   a VALIDATED tag must cite a measurement against an observable, never another
   definition. Guards 6 and 7 exist because one wrong number was certified five
   times, each time by a cross-check that could not have failed.

8. Validation with no power. A validation is evidence in proportion to the range
   its prediction spanned; a design on which the prediction is constant cannot
   reject a wrong functional form, however small the residual.

Guards 6-8 are the subject of `Calibrator.DriftRegime`, which proves that 7 and 8
are impossibilities rather than oversights.
"""
import re, sys, glob, os

ROOT = os.path.join(os.path.dirname(__file__), "..", "proofs")

SORRY_LEDGER = set()                # name -> undischarged obligation, none yet
CONVENTION_SITE_BUDGET = 0        # measured; may decrease, never increase
ISOLATED_MODULE_BUDGET = 14         # modules no theorem cross-relates to another
UNDECLARED_BUDGET = 0               # empirical defs with no status marker
UNRELATED_BUDGET = 20               # ratchets down
MISSING_ARG_BUDGET = 0              # signatures omitting a dependency of the named quantity
CONFLATION_BUDGET = 0               # one formula under names from different concept families
CONVENTION_DECL_BUDGET = 0          # composable quantities with no declared convention
OVERCLAIM_BUDGET = 0                # untested definitions whose docstring claims exactness             # measured; ratchets down as siblings get related
EQUILIBRIUM_BUDGET = 0              # equilibria stipulated as a closed form, never derived
DUPLICATE_BODY_BUDGET = 0           # one body under two names in two files, tied by nothing
REGIME_DECL_BUDGET = 0              # drift regimes baked into a body instead of a hypothesis
INHERITED_VALIDATION_BUDGET = None  # VALIDATED inherited from a sibling identity; pin on first run
VACUOUS_VALIDATION_BUDGET = None    # VALIDATED with no recorded power; pin on first run

def strip_comments(src: str) -> str:
    """Remove Lean block and line comments so prose cannot trip the guards."""
    out, i, depth = [], 0, 0
    while i < len(src):
        if src.startswith("/-", i):
            depth += 1; i += 2; continue
        if src.startswith("-/", i):
            depth = max(0, depth - 1); i += 2; continue
        if depth == 0 and src.startswith("--", i):
            j = src.find("\n", i)
            i = len(src) if j == -1 else j
            continue
        if depth == 0:
            out.append(src[i])
        elif src[i] == "\n":
            out.append("\n")
        i += 1
    return "".join(out)

def lean_files():
    return (glob.glob(os.path.join(ROOT, "Calibrator", "*.lean")) +
            glob.glob(os.path.join(ROOT, "Calibrator", "*", "*.lean")) +
            [os.path.join(ROOT, "Calibrator.lean")])

def main() -> int:
    bad = []

    for f in lean_files():
        src = strip_comments(open(f).read())
        rel = os.path.relpath(f, ROOT)

        for m in re.finditer(r'\bsorry\b', src):
            line = src[:m.start()].count("\n") + 1
            owner = None
            for d in re.finditer(r'^(?:noncomputable )?(?:def|theorem) ([A-Za-z_0-9\'.]+)', src[:m.start()], re.M):
                owner = d.group(1)
            if owner not in SORRY_LEDGER:
                bad.append(f"{rel}:{line}: sorry in `{owner}` is not in SORRY_LEDGER")

        if re.search(r'Evidence\.falsified', src):
            bad.append(f"{rel}: an Identification is still marked falsified")

    # convention drift
    defpat = re.compile(r'^(?:noncomputable )?def ([A-Za-z_0-9\'.]+)(.*?)(?=\n(?:/-|@\[|theorem |noncomputable |def |abbrev |structure |section |end |namespace ))', re.S | re.M)
    # A convention restatement is a 2 or a 4 adjacent to a population-genetic
    # parameter: 2 Ne, 4 Ne mu, 2 p (1 - p). The 2 in a Gaussian density or in
    # a quadratic expansion is not a ploidy convention, and tying it to `ploidy`
    # would be wrong, so the pattern requires the neighbouring symbol.
    POP = r"(?:Ne|N|N_b|N₀|N₁|mu|μ|m|m_rate|m_into|mig|p|p0|p₁|p₂|p_bar|maf|fst|freq|theta|θ|sigma_sq)"
    mult = re.compile(r"(?<![\^A-Za-z_0-9.])[24]\s*\*\s*(?:\([^)]*\)|[A-Za-z_0-9.]*\.)?" + POP + r"\b"
                      r"|/\s*\(\s*[24]\s*\*\s*(?:[A-Za-z_0-9.]*\.)?" + POP + r"\b"
                      r"|\b(?:[A-Za-z_0-9]+\.)?" + POP + r"\s*\*\s*[24]\b")
    # Definitions that a Conventions theorem relates back to `ploidy` or to a
    # derived primitive are not loose restatements: their constant is forced.
    tied = set()
    for f in lean_files():
        if not f.endswith("Conventions.lean"):
            continue
        conv = strip_comments(open(f).read())
        for b in re.split(r"\n(?=theorem )", conv):
            if not b.startswith("theorem"):
                continue
            stmt = b.split(":=", 1)[0]
            tied.update(re.findall(r"[A-Za-z_][A-Za-z_0-9']*", stmt))

    sites = 0
    for f in lean_files():
        if f.endswith("Conventions.lean"):
            continue
        src = strip_comments(open(f).read())
        for m in defpat.finditer(src):
            body = m.group(2)
            body = body.split(":=", 1)[1] if ":=" in body else ""
            body = re.sub(r'\^\s*[0-9]+', '', body)
            if mult.search(body) and m.group(1).split(".")[-1] not in tied:
                sites += 1
    # 3b. Undeclared empirical definitions. Every definition whose name carries
    #     domain vocabulary, or whose body contains a modelling constant, is a
    #     claim about an observable. It must declare an Empirical status, even
    #     if that status is UNTESTED. Four of the seven falsifications found so
    #     far were in definitions nobody had thought to check; the point of the
    #     marker is that the unchecked ones are enumerable rather than silent.
    DOMAIN = re.compile(r"fst|drift|selection|herit|linkage|allele|geno|migrat|coalesc|mutation|"
                        r"epistat|domin|recomb|ancestr|spike|admix|haplo|polygenic|prevalence|"
                        r"liability|penetrance|pgs|gwas|ld[A-Z_]|singleton|winners|power|ncp|effect", re.I)
    undeclared = []
    for f in lean_files():
        raw = open(f).read().split("\n")
        stripped = strip_comments(open(f).read()).split("\n")
        for i, line in enumerate(stripped):
            m = re.match(r"^(?:noncomputable )?def ([A-Za-z_0-9'.]+)", line)
            if not m:
                continue
            short = m.group(1).split(".")[-1]
            body = "\n".join(stripped[i:i + 6])
            body = body.split(":=", 1)[1] if ":=" in body else ""
            if not (DOMAIN.search(short) or mult.search(re.sub(r"\^\s*[0-9]+", "", body))):
                continue
            if "Empirical status:" not in "\n".join(raw[max(0, i - 14):i + 1]):
                undeclared.append(f"{os.path.relpath(f, ROOT)}: `{short}` has no Empirical status")
    if len(undeclared) > UNDECLARED_BUDGET:
        bad.append(f"definitions making an empirical claim without an Empirical status marker: "
                   f"{len(undeclared)}, budget {UNDECLARED_BUDGET}")
        bad.extend("    " + u for u in undeclared[:8])

    # 3c. Unrelated same-quantity definitions. Two definitions are the same
    #     quantity when their bodies agree after renaming that definition's own
    #     bound variables, and the shared body contains a constant or a named
    #     function rather than being pure operator shape: `2 p (1 - p)` counts,
    #     `a + b` does not. A group spanning two modules with no theorem
    #     mentioning two of its members is a divergence nothing can detect,
    #     which is how amInflationFactor and fstFromDrift survived.
    bodypat = re.compile(r"^(?:noncomputable )?def ([A-Za-z_0-9'.]+)(.*?):=\s*\n?\s*(.+?)"
                         r"(?=\n(?:@\[|theorem |noncomputable |def |abbrev |structure |section |end |namespace |/-))",
                         re.S | re.M)
    groups = {}
    for f in lean_files():
        src = strip_comments(open(f).read())
        mod = os.path.basename(f)[:-5]
        for m in bodypat.finditer(src):
            name, args, body = m.group(1), m.group(2), " ".join(m.group(3).split())
            if len(body) > 80:
                continue
            bound = set(re.findall(r"[A-Za-z_][A-Za-z_0-9₀-₉']*", args))
            norm = re.sub(r"[A-Za-z_][A-Za-z_0-9₀-₉'.]*",
                          lambda t: "V" if t.group(0) in bound else t.group(0), body)
            if not re.search(r"[0-9]|[A-Za-z_]{3,}", norm.replace("V", "")):
                continue
            groups.setdefault(norm, []).append((mod, name.split(".")[-1]))
    all_stmts = []
    for f in lean_files():
        for b in re.split(r"\n(?=@\[simp\]\s*\n?theorem |theorem |private theorem )",
                          strip_comments(open(f).read())):
            if re.match(r"(?:@\[simp\]\s*)?(?:private )?theorem ", b) and ":=" in b:
                all_stmts.append(b.split(":=", 1)[0])
    # A definition tied to a shared primitive in Conventions is related in the
    # stronger sense: its whole group is pinned to one object rather than to
    # each other pairwise. Credit that, or the metric penalises exactly the
    # refactor it exists to encourage.
    primitives = set()
    for f in lean_files():
        if not f.endswith("Conventions.lean"):
            continue
        for m in re.finditer(r"^(?:noncomputable )?def ([A-Za-z_0-9'.]+)",
                             strip_comments(open(f).read()), re.M):
            primitives.add(m.group(1).split(".")[-1])

    unrelated = 0
    for norm, members in groups.items():
        if len({m for m, _ in members}) < 2:
            continue
        names = [n for _, n in members]
        for n in names:
            tied = any(
                re.search(r"\b" + re.escape(n) + r"\b", st) and
                (any(re.search(r"\b" + re.escape(o) + r"\b", st) for o in names if o != n) or
                 any(re.search(r"\b" + re.escape(pr) + r"\b", st) for pr in primitives))
                for st in all_stmts)
            if not tied:
                unrelated += 1
    if unrelated > UNRELATED_BUDGET:
        bad.append(f"same-quantity definitions never related to a sibling by any theorem: "
                   f"{unrelated}, budget {UNRELATED_BUDGET}")

    # 3d. Missing-argument screen. Six of the eleven falsified definitions failed
    #     the same way: the signature omits an argument the named quantity is
    #     known to depend on. No constant repairs such a definition, and the
    #     defect is visible statically, without any simulation. Each entry is a
    #     name pattern together with the arguments that quantity must depend on.
    PREVALENCE_FREE = {"populationAUC"}   # rank definition, prevalence-free by construction
    REQUIRED_ARGS = [
        (r"power",            [r"alpha", r"z_?alpha", r"threshold", r"level"],
         "statistical power depends on the significance threshold"),
        (r"auc",              [r"prev", r"k\b", r"pi\b", r"baseRate"],
         "AUC under a threshold model depends on prevalence"),
        (r"winner|curse",     [r"alpha", r"z_?alpha", r"threshold"],
         "selection bias depends on the selection threshold"),
        (r"singleton|sfs",    [r"\bn\b", r"nsamp", r"sampleSize"],
         "site-frequency quantities depend on sample size"),
        (r"aminflation|assortativeinflation",
                              [r"h2", r"herit"],
         "assortative-mating inflation depends on heritability"),
        (r"ldamplif|amplifld|bottlenecklD",
                              [r"\br\b", r"recomb", r"\bc\b"],
         "LD amplification depends on the recombination rate"),
    ]
    missing = []
    for f in lean_files():
        body_all = strip_comments(open(f).read())
        for m in re.finditer(r"^(?:noncomputable )?def ([A-Za-z_0-9'.]+)([^:]*):", body_all, re.M):
            name, args = m.group(1).split(".")[-1], m.group(2)
            if name in PREVALENCE_FREE or re.search(r"gaussian|interval|approximation", name, re.I):
                continue   # name declares the model it is exact for, or is a wrapper
            for pat, needed, why in REQUIRED_ARGS:
                if not re.search(pat, name, re.I):
                    continue
                if not any(re.search(a, args, re.I) for a in needed):
                    missing.append(f"{os.path.relpath(f, ROOT)}: `{name}` takes no "
                                   f"{needed[0]}-like argument; {why}")
    if len(missing) > MISSING_ARG_BUDGET:
        bad.append(f"definitions omitting an argument the named quantity depends on: "
                   f"{len(missing)}, budget {MISSING_ARG_BUDGET}")
        bad.extend("    " + x for x in missing[:10])

    # 3d-bis. Overclaiming. Two of the falsified definitions carried the word
    #     "exact" in a docstring while being 26 percent wrong. A definition may
    #     claim exactness or derivation, or it may be untested, but not both:
    #     an untested definition has no standing to call itself exact.
    overclaim = []
    for f in lean_files():
        raw = open(f).read()
        for m in re.finditer(r"/--((?:(?!-/).)*)-/\s*\n(?:noncomputable )?def ([A-Za-z_0-9'.]+)", raw, re.S):
            doc, name = m.group(1), m.group(2).split(".")[-1]
            if "Empirical status: UNTESTED" not in doc:
                continue
            claim = re.search(r"\b(exact|exactly|derived from first principles|"
                              r"the true |precisely)\b", doc, re.I)
            if claim:
                overclaim.append(f"{os.path.relpath(f, ROOT)}: `{name}` is UNTESTED but its "
                                 f"docstring claims \"{claim.group(1)}\"")
    if len(overclaim) > OVERCLAIM_BUDGET:
        bad.append(f"untested definitions whose docstring claims exactness: "
                   f"{len(overclaim)}, budget {OVERCLAIM_BUDGET}")
        bad.extend("    " + x for x in overclaim[:10])

    # 3f. Convention declarations on composable quantities. A definition
    #     producing a quantity and another consuming it can disagree about its
    #     convention while both remain defensible alone, and Lean cannot object
    #     because both are real-valued. ldCorrelationSq returned r-squared over
    #     four when fed the D that admixtureLDTwoLocus produces, 350 lines apart
    #     in one file. Any definition taking an ambiguity-prone argument must
    #     state the convention it assumes.
    AMBIGUOUS = [
        (r"\bD\b", "linkage disequilibrium: haplotype D or dosage covariance (differ by ploidy)"),
        (r"\bvar_tag\b|\bvar_causal\b", "variance: allelic p(1-p) or genotypic 2p(1-p)"),
        (r"\bmaf\b|\bmaf_causal\b|\bmaf_tag\b",
         "allele frequency: of the causal variant or of the tag, which differ once r < 1"),
    ]
    undeclared_conv = []
    for f in lean_files():
        raw = open(f).read()
        for m in re.finditer(r"/--((?:(?!-/).)*)-/\s*\n(?:noncomputable )?def ([A-Za-z_0-9'.]+)([^:]*):",
                             raw, re.S):
            doc, name, args = m.group(1), m.group(2).split(".")[-1], m.group(3)
            for pat, why in AMBIGUOUS:
                if re.search(pat, args) and "Convention:" not in doc:
                    undeclared_conv.append(
                        f"{os.path.relpath(f, ROOT)}: `{name}` takes an ambiguity-prone "
                        f"argument and declares no Convention; {why}")
                    break
    if len(undeclared_conv) > CONVENTION_DECL_BUDGET:
        bad.append(f"definitions taking an ambiguity-prone quantity with no declared "
                   f"convention: {len(undeclared_conv)}, budget {CONVENTION_DECL_BUDGET}")
        bad.extend("    " + x for x in undeclared_conv[:8])


    # 3g. Naming conflation. One formula carrying names from different concept
    #     families is how allelicVariance came about: 2p(1-p) is correctly the
    #     genotype variance and correctly the HWE heterozygote frequency, and is
    #     not the allelic variance, which is p(1-p). The r-squared-over-four
    #     defect was inherited from that name, not slipped in the formula. Where
    #     one body carries names from two families, each must say what it
    #     denotes.
    FAMILY = {
        "variance": r"variance|var\b", "frequency": r"freq|maf|prop",
        "heterozygosity": r"heteroz|het\b", "rate": r"rate",
        "factor": r"factor|retention|decay", "fst": r"fst",
    }
    bodies = {}
    for f in lean_files():
        src = strip_comments(open(f).read())
        for m in re.finditer(r"^(?:noncomputable )?def ([A-Za-z_0-9'.]+)(.*?):=\s*\n?\s*(.+?)"
                             r"(?=\n(?:@\[|theorem |noncomputable |def |abbrev |structure |section |end |namespace |/-))",
                             src, re.S | re.M):
            name, args, body = m.group(1).split(".")[-1], m.group(2), " ".join(m.group(3).split())
            if len(body) > 60:
                continue
            bound = set(re.findall(r"[A-Za-z_][A-Za-z_0-9₀-₉']*", args))
            norm = re.sub(r"[A-Za-z_][A-Za-z_0-9₀-₉'.]*",
                          lambda t: "V" if t.group(0) in bound else t.group(0), body)
            if not re.search(r"[0-9]", norm):
                continue
            bodies.setdefault(norm, []).append((f, name))
    conflated = []
    for norm, members in bodies.items():
        fams = {fam for _, n in members for fam, pat in FAMILY.items() if re.search(pat, n, re.I)}
        if len(fams) < 2:
            continue
        for f, n in members:
            doc = ""
            raw = open(f).read()
            dm = re.search(r"/--((?:(?!-/).)*)-/\s*\n(?:noncomputable )?def " + re.escape(n) + r"\b", raw, re.S)
            if dm:
                doc = dm.group(1)
            if "Denotes:" not in doc:
                conflated.append(f"{os.path.relpath(f, ROOT)}: `{n}` shares a formula with names "
                                 f"from {sorted(fams)} and declares no Denotes")
    if len(conflated) > CONFLATION_BUDGET:
        bad.append(f"definitions sharing one formula across concept families with no Denotes "
                   f"declaration: {len(conflated)}, budget {CONFLATION_BUDGET}")
        bad.extend("    " + x for x in conflated[:8])


    # 3h. Equilibrium without a dynamic. `selectionMigrationEquilibrium s m =
    #     s / (s + m)` was a stipulated closed form, wrong by 4 to 14x and
    #     qualitatively wrong where the allele is lost, yet every theorem about
    #     it was true: value-guards bound a quantity into (0,1) and order it the
    #     right ways, and none of that can pin a constant. Only the process the
    #     equilibrium is an equilibrium *of* can. So a definition named for a
    #     limit or a rest point owes a theorem identifying it as the fixed point
    #     of some other definition in the same file, in the shape of
    #     `selectionMigrationEquilibrium_isFixedPoint`.
    EQUILIBRIUM_CONCEPTS = ("equilibrium", "fixedpoint", "steadystate", "stationary",
                            "limiting", "asymptotic", "balance", "equilibriumfreq")
    FIXEDPOINT_MARKERS = ("isFixedPoint", "_fixedPoint", "_isLimit", "_tendsto")

    def word_starts(name):
        """Offsets at which a camelCase or underscore-separated word begins.

        Substring matching alone reads `globalAncestry` as containing
        `balance`, so a concept counts only where a word does."""
        return {0} | {i for i in range(1, len(name))
                      if name[i - 1] in "_'" or name[i].isupper() or name[i].isdigit()}

    def names_an_equilibrium(short):
        low, starts = short.lower(), word_starts(short)
        return any(m.start() in starts
                   for c in EQUILIBRIUM_CONCEPTS
                   for m in re.finditer(re.escape(c), low))

    stipulated = []
    for f in lean_files():
        src = strip_comments(open(f).read())
        rel = os.path.relpath(f, ROOT)
        defs, bodies_here = [], {}
        for m in re.finditer(r"^(?:noncomputable )?def ([A-Za-z_0-9'.]+)(.*?)(?=\n(?:@\[|theorem |"
                             r"noncomputable |def |abbrev |structure |section |end |namespace |/-))",
                             src, re.S | re.M):
            short = m.group(1).split(".")[-1]
            defs.append((short, src[:m.start()].count("\n") + 1))
            bodies_here[short] = m.group(2).split(":=", 1)[-1]
        theorems = [(t.group(1).split(".")[-1], t.group(0).split(":=", 1)[0])
                    for t in re.finditer(r"^(?:@\[[^\]]*\]\s*\n)?(?:private )?theorem "
                                         r"([A-Za-z_0-9'.]+)(?:.*?)(?=\n(?:@\[|theorem |"
                                         r"noncomputable |def |abbrev |structure |section |end |"
                                         r"namespace |/-))", src, re.S | re.M)]
        allnames = {n for n, _ in defs}
        for short, line in defs:
            if not names_an_equilibrium(short):
                continue
            # A quantity derived from an equilibrium is not itself stipulated:
            # the obligation to derive belongs to the definition it calls.
            body = bodies_here.get(short, "")
            if any(o != short and names_an_equilibrium(o) and
                   re.search(r"\b" + re.escape(o) + r"\b", body) for o in allnames):
                continue
            ok = False
            for tname, stmt in theorems:
                if not tname.startswith(short) or not any(k in tname for k in FIXEDPOINT_MARKERS):
                    continue
                if any(o != short and re.search(r"\b" + re.escape(o) + r"\b", stmt)
                       for o in allnames):
                    ok = True
                    break
            if not ok:
                stipulated.append(f"{rel}:{line}  {short}  (no fixed-point theorem)")
    if len(stipulated) > EQUILIBRIUM_BUDGET:
        bad.append(f"equilibrium definitions with no theorem deriving them as the fixed point "
                   f"of a process in the same file: {len(stipulated)}, budget {EQUILIBRIUM_BUDGET}; "
                   f"define the one-step map and prove `<name>_isFixedPoint`")
        bad.extend("    " + x for x in stipulated[:12])

    # 3i. One body, two files. `t / (t + 2 Ne)`, `1 - (1 - 1/(2 Ne)) ^ t` and
    #     `1 - exp (-tau)` were three definitions of F_ST living in three
    #     modules, and two of them were wrong; repairing one left the other two
    #     standing, because nothing in the corpus said they were the same
    #     quantity. Alpha-equivalent bodies in different files are either one
    #     quantity, and one of them should call the other, or they are two
    #     quantities that happen to coincide, and a theorem should say so.
    dupdef = re.compile(r"^(?:noncomputable )?def ([A-Za-z_0-9'.]+)(.*?):=\s*\n?\s*(.+?)"
                        r"(?=\n(?:@\[|theorem |noncomputable |def |abbrev |structure |section |"
                        r"end |namespace |/-))", re.S | re.M)
    IDENT = r"[A-Za-z_][A-Za-z_0-9₀-₉']*"

    def alpha_normal(args, body):
        """Body with whitespace collapsed and binders renamed positionally.

        Renaming is by order of first use in the body, not by order of
        declaration, so `(m Ne)` and `(Ne m)` over the same formula normalise
        together."""
        bound, seen = set(re.findall(IDENT, args)), {}
        def rename(t):
            w = t.group(0)
            if w in bound:
                seen.setdefault(w, "V%d" % (len(seen) + 1))
                return seen[w]
            return w
        return re.sub(IDENT, rename, " ".join(body.split()))

    shapes = {}
    for f in lean_files():
        src = strip_comments(open(f).read())
        rel = os.path.relpath(f, ROOT)
        for m in dupdef.finditer(src):
            name, args, body = m.group(1).split(".")[-1], m.group(2), m.group(3)
            norm = alpha_normal(args, body)
            # Pure operator shape is not a shared quantity: `a + b` coincides
            # everywhere. Require a constant or a named function, as 3c does.
            if not re.search(r"[0-9]|[A-Za-z_]{3,}", re.sub(r"\bV[0-9]+\b", "", norm)):
                continue
            shapes.setdefault(norm, []).append((rel, src[:m.start()].count("\n") + 1, name, body))
    file_stmts = {}
    for f in lean_files():
        rel = os.path.relpath(f, ROOT)
        for b in re.split(r"\n(?=@\[simp\]\s*\n?theorem |theorem |private theorem )",
                          strip_comments(open(f).read())):
            if re.match(r"(?:@\[simp\]\s*)?(?:private )?theorem ", b) and ":=" in b:
                file_stmts.setdefault(rel, []).append(b.split(":=", 1)[0])
    duplicates = []
    for norm, members in sorted(shapes.items()):
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                (fa, la, na, ba), (fb, lb, nb, bb) = members[i], members[j]
                if fa == fb:
                    continue
                # Tied by definition: one is written in terms of the other.
                if (re.search(r"\b" + re.escape(nb) + r"\b", ba) or
                        re.search(r"\b" + re.escape(na) + r"\b", bb)):
                    continue
                # Tied by a theorem in one of the two files naming both.
                if any(re.search(r"\b" + re.escape(na) + r"\b", st) and
                       re.search(r"\b" + re.escape(nb) + r"\b", st)
                       for st in file_stmts.get(fa, []) + file_stmts.get(fb, [])):
                    continue
                duplicates.append(f"{fa}:{la} {na}  ==  {fb}:{lb} {nb}")
    duplicates.sort()
    if len(duplicates) > DUPLICATE_BODY_BUDGET:
        bad.append(f"alpha-equivalent definition bodies in different files tied by neither a "
                   f"call nor a theorem: {len(duplicates)}, budget {DUPLICATE_BODY_BUDGET}; "
                   f"make one call the other, or state the identity as a theorem")
        bad.extend("    " + x for x in duplicates[:12])

    # 3j. Regimes baked into bodies. Five definitions -- the within-population
    #     heterozygosity loss, the F_ST read off it, the target heterozygosity,
    #     the target PGS variance, and the neutral benchmark ratio -- were all
    #     functions of one number, `(1 - 1/(2 Ne))^t`, the closed-population
    #     no-mutation retention. Simulation at demographic equilibrium measures
    #     that retention as 1.02 +- 0.02 where the formula predicts e^-2 = 0.135:
    #     mutation replenishes diversity, so heterozygosity is stationary and the
    #     cluster's "F_ST" is ~0 exactly where the measurable between-population
    #     F_ST is 0.50. They are different quantities sharing a name.
    #
    #     The premise was invisible because it lived in a *body*, not in a
    #     hypothesis. A definition carrying the closed-population retention factor
    #     must therefore name its regime, so that a reader and a use site both see
    #     which data-generating process it assumes. `Calibrator.DriftRegime`
    #     exhibits the two regimes and proves they disagree at every positive time.
    regimeless = []
    for f in lean_files():
        raw = open(f).read()
        for m in re.finditer(r"/--((?:(?!-/).)*)-/\s*\n(?:noncomputable )?def "
                             r"([A-Za-z_0-9'.]+)[^:]*:[^:=]*:=((?:(?!\n/--|\ntheorem|\nend ).)*)",
                             raw, re.S):
            doc, name, body = m.group(1), m.group(2).split(".")[-1], m.group(3)
            # the closed-population retention factor, raised to a power
            if re.search(r"\(\s*1\s*-\s*1\s*/\s*\(\s*2\s*\*[^)]*\)\s*\)\s*\^", body):
                if "Regime:" not in doc:
                    regimeless.append(
                        f"{os.path.relpath(f, ROOT)}: `{name}` carries the closed-population "
                        f"retention factor in its body and declares no Regime")
    if len(regimeless) > REGIME_DECL_BUDGET:
        bad.append(f"definitions encoding a drift regime with no declared Regime: "
                   f"{len(regimeless)}, budget {REGIME_DECL_BUDGET}; name the "
                   f"data-generating assumption, see Calibrator.DriftRegime")
        bad.extend("    " + x for x in regimeless[:12])

    # 3k. Validation inherited from a sibling identity. Over-determination
    #     detects divergence between independently written formulas and is
    #     provably blind to a premise they share
    #     (`Calibrator.DriftRegime.crossChecks_blind_to_retention`): every identity
    #     among members of a cluster holds at *every* value of the shared premise,
    #     including the wrong one. So a VALIDATED tag may cite a measurement
    #     against an observable, never a sibling formula. A validation note that
    #     only names another definition is an inherited tag, and inherited tags are
    #     what let one wrong number be certified five times.
    inherited = []
    for f in lean_files():
        raw = open(f).read()
        for m in re.finditer(r"Empirical status: VALIDATED([^-]*?)-/", raw, re.S):
            note = m.group(1)
            cites_identity = re.search(r"\bthis is the identity\b|\bthe theorem\b|"
                                       r"\bby definition\b|\bdefinitionally\b|"
                                       r"\balongside\b `?[A-Za-z_0-9']+`?", note, re.I)
            cites_measurement = re.search(r"simulat|measur|against|observed|grid|"
                                          r"coalescent|SLiM|panel|out-of-sample", note, re.I)
            if cites_identity and not cites_measurement:
                inherited.append(f"{os.path.relpath(f, ROOT)}: a VALIDATED note cites a sibling "
                                 f"identity but no measurement: \"{note.strip()[:70]}\"")
    #     Reported, not enforced, until the count is measured once and pinned:
    #     these two guards are retroactive over twenty existing VALIDATED tags,
    #     and failing the build on all of them at once would be noise rather than
    #     signal. Pin INHERITED_VALIDATION_BUDGET to the first reported count and
    #     ratchet it down, exactly as CONVENTION_SITE_BUDGET was.
    if INHERITED_VALIDATION_BUDGET is None:
        for x in inherited[:10]:
            print(f"  advisory (inherited validation): {x}")
    elif len(inherited) > INHERITED_VALIDATION_BUDGET:
        bad.append(f"VALIDATED tags justified by a sibling identity rather than a measurement: "
                   f"{len(inherited)}, budget {INHERITED_VALIDATION_BUDGET}")
        bad.extend("    " + x for x in inherited[:10])

    # 3l. Validation with no power. `neutralAFBenchmarkRatio` was recorded as
    #     validated to 3.2 percent. The design was symmetric, so both sides of the
    #     ratio collapsed to ~1 and the test could not have failed;
    #     `Calibrator.DriftRegime.symmetric_design_has_no_power` proves that on any
    #     symmetric design the ratio and its *square* are indistinguishable. On
    #     asymmetric effective sizes the same formula is off by -37 to -74 percent,
    #     at nine to fifteen standard errors.
    #
    #     A validation is evidence in proportion to the range its prediction
    #     spanned. So a VALIDATED note must record at least two predicted values,
    #     and they must not all be equal: a prediction that is constant on the
    #     design tests nothing about shape.
    powerless = []
    for f in lean_files():
        raw = open(f).read()
        for m in re.finditer(r"Empirical status: VALIDATED([^-]*?)-/", raw, re.S):
            note = m.group(1)
            nums = [float(x) for x in re.findall(r"\d+\.\d+", note)]
            if len(nums) < 2:
                powerless.append(f"{os.path.relpath(f, ROOT)}: a VALIDATED note records fewer "
                                 f"than two predicted values, so its power is unstated")
                continue
            spread = max(nums) - min(nums)
            if spread <= 0.05 * max(abs(max(nums)), 1.0):
                powerless.append(f"{os.path.relpath(f, ROOT)}: a VALIDATED note's predictions "
                                 f"span only {spread:.4f}; a near-constant prediction cannot "
                                 f"reject a wrong functional form")
    if VACUOUS_VALIDATION_BUDGET is None:
        for x in powerless[:12]:
            print(f"  advisory (validation power unstated): {x}")
    elif len(powerless) > VACUOUS_VALIDATION_BUDGET:
        bad.append(f"VALIDATED tags whose design had no recorded power: {len(powerless)}, "
                   f"budget {VACUOUS_VALIDATION_BUDGET}; record the spread of the prediction "
                   f"across the design, see Calibrator.DriftRegime")
        bad.extend("    " + x for x in powerless[:12])

    # 3e. Cheap structural integrity, run before the build so that a broken
    #     rename or an unterminated comment fails in seconds rather than after a
    #     full elaboration. The "+/-" incident is the motivating case: text in a
    #     status marker contained "/-", which opened a nested comment and left a
    #     docstring unterminated.
    for f in lean_files():
        raw = open(f).read()
        rel = os.path.relpath(f, ROOT)
        if raw.count("/-") != raw.count("-/"):
            bad.append(f"{rel}: unbalanced comment delimiters "
                       f"({raw.count('/-')} open, {raw.count('-/')} close)")
        if "namespace Calibrator" in raw and not raw.rstrip().endswith("end Calibrator"):
            bad.append(f"{rel}: does not end with `end Calibrator`")
        for imp in re.findall(r"^import (Calibrator[A-Za-z.]*)", raw, re.M):
            if not os.path.exists(os.path.join(ROOT, imp.replace(".", "/") + ".lean")):
                bad.append(f"{rel}: imports {imp}, which does not exist")

    # 4. semantic isolation. A module that no theorem ever relates to another
    #    module cannot be contradicted by anything: a false definition inside it
    #    is consistent with the whole corpus. This is the condition that let two
    #    falsified identifications survive review, so the count is ratcheted.
    owner = {}
    for f in lean_files():
        mod = os.path.basename(f)[:-5]
        for m in re.finditer(r"^(?:noncomputable )?(?:def|abbrev|structure) ([A-Za-z_0-9'.]+)",
                             strip_comments(open(f).read()), re.M):
            owner[m.group(1).split(".")[-1]] = mod
    linked = {}
    for f in lean_files():
        body = strip_comments(open(f).read())
        for b in re.split(r"\n(?=@\[simp\]\s*\n?theorem |theorem |private theorem )", body):
            if not re.match(r"(?:@\[simp\]\s*)?(?:private )?theorem ", b) or ":=" not in b:
                continue
            stmt = b.split(":=", 1)[0]
            mods = {owner[t] for t in re.findall(r"[A-Za-z_][A-Za-z_0-9']*", stmt) if t in owner}
            if len(mods) > 1:
                for a in mods:
                    linked.setdefault(a, set()).update(mods - {a})
    all_mods = {os.path.basename(f)[:-5] for f in lean_files()}
    isolated = sorted(m for m in all_mods if not linked.get(m))
    if len(isolated) > ISOLATED_MODULE_BUDGET:
        bad.append(f"semantically isolated modules rose to {len(isolated)}, budget "
                   f"{ISOLATED_MODULE_BUDGET}: {', '.join(isolated[:6])}...; relate the new "
                   f"module's quantities to an existing one so it can be contradicted")

    if sites > CONVENTION_SITE_BUDGET:
        bad.append(f"convention restatement sites rose to {sites}, budget {CONVENTION_SITE_BUDGET}; "
                   f"relate the new constant to `ploidy` in Conventions.lean instead of inlining it")

    if bad:
        print("STRUCTURAL GUARD FAILURES\n")
        for b in bad:
            print("  " + b)
        return 1
    print(f"structural guards pass: convention sites {sites}/{CONVENTION_SITE_BUDGET}, "
          f"undeclared {len(undeclared)}/{UNDECLARED_BUDGET}, conventions {len(undeclared_conv)}/{CONVENTION_DECL_BUDGET}, "
          f"unrelated {unrelated}/{UNRELATED_BUDGET}, "
          f"stipulated equilibria {len(stipulated)}/{EQUILIBRIUM_BUDGET}, "
          f"duplicate bodies {len(duplicates)}/{DUPLICATE_BODY_BUDGET}, "
          f"isolated modules {len(isolated)}/{ISOLATED_MODULE_BUDGET}, "
          f"sorry ledger {len(SORRY_LEDGER)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
