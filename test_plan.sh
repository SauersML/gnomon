#!/bin/bash
cat << 'PLAN'
1. *Modify `haplotypePhasePredictionError`*
   - Change `haplotypePhasePredictionError` to be non-zero when there is actual phase misspecification. However, the docstring says: "A phase-aware haplotype predictor that tracks cis/trans configuration has no structural phase-misspecification error." so it should stay 0 mathematically, but maybe it should be parameterized to show it models the configurations? Wait, if it *has* no error by definition, then its current constant `0` is mathematically correct but structurally vacuous. If we parameterize it, how does it evaluate to 0?

   Wait, let's look at what the definition says. It just says `0`. This is the definition of a "trivial witness". Let's parameterize it!
PLAN
