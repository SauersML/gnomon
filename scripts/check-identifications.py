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
"""
import re, sys, glob, os

ROOT = os.path.join(os.path.dirname(__file__), "..", "proofs")

SORRY_LEDGER = set()                # name -> undischarged obligation, none yet
CONVENTION_SITE_BUDGET = 86        # measured; may decrease, never increase
ISOLATED_MODULE_BUDGET = 21         # modules no theorem cross-relates to another
UNDECLARED_BUDGET = 0               # empirical defs with no status marker
UNRELATED_BUDGET = 71               # measured; ratchets down as siblings get related

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
    mult = re.compile(r'(?<![\^A-Za-z_0-9.])([2-9]|[1-9][0-9]+)\s*\*|/\s*\(?\s*([2-9]|[1-9][0-9]+)\s*\*')
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

    # 3c. Unrelated same-quantity definitions. Both of the two most recent
    #     falsifications were a pair of definitions of one quantity that no
    #     theorem ever related: amInflationFactor against amEquilibriumVariance,
    #     and fstFromDrift against coalFst. Where two definitions share a domain
    #     stem, some theorem should mention both, so that a disagreement is a
    #     failed proof rather than two coexisting answers.
    STEMS = ["Fst", "Inflation", "Power", "Bias", "Overlap", "Spike", "Retention", "Variance"]
    defs_by_stem = {}
    for f in lean_files():
        body = strip_comments(open(f).read())
        for m in re.finditer(r"^(?:noncomputable )?def ([A-Za-z_0-9'.]+)", body, re.M):
            n = m.group(1).split(".")[-1]
            for st in STEMS:
                if st.lower() in n.lower():
                    defs_by_stem.setdefault(st, set()).add(n)
    all_stmts = []
    for f in lean_files():
        for b in re.split(r"\n(?=@\[simp\]\s*\n?theorem |theorem |private theorem )",
                          strip_comments(open(f).read())):
            if re.match(r"(?:@\[simp\]\s*)?(?:private )?theorem ", b) and ":=" in b:
                all_stmts.append(b.split(":=", 1)[0])
    unrelated = 0
    for st, names in defs_by_stem.items():
        if len(names) < 2:
            continue
        for n in names:
            if not any(re.search(r"\b" + re.escape(n) + r"\b", s) and
                       any(re.search(r"\b" + re.escape(o) + r"\b", s) for o in names if o != n)
                       for s in all_stmts):
                unrelated += 1
    if unrelated > UNRELATED_BUDGET:
        bad.append(f"same-quantity definitions never related to a sibling by any theorem: "
                   f"{unrelated}, budget {UNRELATED_BUDGET}")

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
          f"undeclared {len(undeclared)}/{UNDECLARED_BUDGET}, unrelated {unrelated}/{UNRELATED_BUDGET}, "
          f"isolated modules {len(isolated)}/{ISOLATED_MODULE_BUDGET}, "
          f"sorry ledger {len(SORRY_LEDGER)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
