#!/usr/bin/env python3
"""Calibration for `check-laundering.py`: a fixture of KNOWN laundering, and a fixture
of known-clean mathematics, checked in both directions.

A detector that reports nothing is indistinguishable from a corpus that is clean, and
the second is the claim the report will be read as making.  So both halves are asserted:

  POSITIVES -- every laundering pattern in the taxonomy that this tool claims to detect
               is present below and MUST be reported.  A silent detector fails here.
  NEGATIVES -- ordinary mathematics that superficially resembles laundering (modus
               ponens, a definitional unfolding lemma, a contrapositive, a parameterised
               witness) MUST NOT be reported.  A detector that flags these trains its
               readers to ignore it.

Run:  python3 scripts/test-check-laundering.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

POSITIVE = r"""
import Mathlib

namespace Fixture

def FamousConjecture : Prop := ∀ n : ℕ, n = n

-- F1: the conclusion IS the hypothesis.
theorem famous_conjecture (h : FamousConjecture) : FamousConjecture := h

-- F2: a Prop named like a theorem, with no inhabitant anywhere.
def ClassificationTheorem : Prop := ∀ n : ℕ, 0 ≤ n

-- F4: a certificate carrying the mathematics, consumed and never constructed.
structure ConstructionSetup where
  object : ℕ
  propertyA : 0 < object
  finalHardIdentity : object * object = object

theorem main_result (s : ConstructionSetup) : ∃ x : ℕ, 0 < x :=
  ⟨s.object, s.propertyA⟩

-- F4 again, in its purest form: the theorem IS the field, renamed.
theorem setup_object_pos (s : ConstructionSetup) : 0 < s.object :=
  s.propertyA

-- F8: the target property weakened to nothing.
def IsSecure (system : ℕ) : Prop := True

-- F11: a class whose field is `False`.
class MagicalStructure where
  contradiction : False

-- F24: a custom axiom.
axiom deepResult : ∀ n : ℕ, n = n

end Fixture
"""

NEGATIVE = r"""
import Mathlib

namespace Clean

-- Ordinary model class WITH a witness: not a certificate defect.
structure Model where
  rate : ℝ
  rate_pos : 0 < rate

def Model.witness : Model where
  rate := 1
  rate_pos := by norm_num

theorem Model.rate_ne_zero (m : Model) : m.rate ≠ 0 := ne_of_gt m.rate_pos

-- Definitional unfolding at a point: `h s` is modus ponens, not laundering.
def Even' (f : ℤ → ℝ) : Prop := ∀ s, f (-s) = f s

theorem even_blind (f : ℤ → ℝ) (h : Even' f) (s : ℤ) : f (-s) = f s := h s

-- A parameterised witness inhabits the class for every index.
structure Panel (n : ℕ) where
  mass : Fin n → ℝ
  mass_nonneg : ∀ j, 0 ≤ mass j

def Panel.witness (n : ℕ) : Panel n where
  mass := fun _ => 0
  mass_nonneg := fun _ => le_refl 0

-- Uses the field, but proves something the field does not say.
theorem Panel.mass_sum_nonneg {n : ℕ} (p : Panel n) (i j : Fin n) :
    0 ≤ p.mass i + p.mass j :=
  add_nonneg (p.mass_nonneg i) (p.mass_nonneg j)

end Clean
"""


def run(src: str) -> list[str]:
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "Fixture.lean"
        f.write_text(src)
        r = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "check-laundering.py"), str(f)],
            capture_output=True, text=True)
        return r.stdout.splitlines()


def families(lines: list[str]) -> set[str]:
    out = set()
    for l in lines:
        if l.strip().startswith("=== F"):
            out.add(l.strip().split()[1])
    return out


def main() -> int:
    failures = []

    got = families(run(POSITIVE))
    # `sorry` is deliberately absent from the fixture: it is the ADMITTED alternative
    # this tool prefers, not a pattern it bans.
    for want, why in [
        ("F1", "conclusion is the hypothesis"),
        ("F2", "Prop alias named like a theorem, never inhabited"),
        ("F4", "certificate structure consumed and never constructed"),
        ("F8", "target property defined as True"),
        ("F11", "class with a False field"),
        ("F24", "custom axiom"),
    ]:
        if want not in got:
            failures.append(f"MISSED {want}: {why}")

    clean = families(run(NEGATIVE))
    for bad, why in [
        ("F1", "modus ponens is not a tautology"),
        ("F1b", "`h s` is a definitional unfolding, not a restated hypothesis"),
        ("F4", "a class with a witness is not an unbuilt certificate"),
        ("F8", "no property is defined as True here"),
        ("F11", "no inconsistent context here"),
        ("F24", "no trust bypass here"),
    ]:
        if bad in clean:
            failures.append(f"FALSE POSITIVE {bad}: {why}")

    for f in failures:
        print(f"FAIL  {f}")
    if failures:
        print(f"\n{len(failures)} calibration failure(s); the detector's report is not "
              f"evidence until these pass.")
        return 1
    print("check-laundering.py calibration PASSED "
          "(6 positives detected, 6 negatives silent)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
