---
layout: ../layouts/Page.astro
title: Polygenic score transfer
subtitle: >-
  The laws taking source-population parameters to expected accuracy in a target
  population — what each is worth, what is measured, and what remains assumed.
meta: gnomon · 1219 definitions · 105 modules · zero sorry
---

## I. The chain

Five steps take source-population parameters to an accuracy you should expect
in a target population. This is the whole usable result; everything after it is
either a bound on how well this can be done, or a caveat about what it leaves out.

<div class="chain">
<div class="step">
<div class="label">1 · divergence from demography</div>

$$
F_{ST} = \frac{t}{t + 2N_e}
$$

Generations since the split over effective population size. This is `coalFst`.
It is **not** $1-\left(1-\frac{1}{2N_e}\right)^{t}$ — that expression is
*within*-population heterozygosity loss, and reads near zero where the true
between-population differentiation is one half.

</div>
<div class="step">
<div class="label">2 · signal that survives transfer</div>

$$
V_{\text{target}} = \rho^{2}\,(1 - F_{ST})\,V_A
$$

$V_A$ is the variance the score explains in the source population; $\rho^{2}$ is
the fraction of causal-variant tagging that survives. Drift and tagging enter
multiplicatively, and neither is recoverable from the other.

</div>
<div class="step">
<div class="label">3 · accuracy on the liability scale</div>

$$
R^{2}_{\text{target}} = \frac{V_{\text{target}}}{V_{\text{target}} + V_E}
$$

</div>
<div class="step">
<div class="label">4 · the threshold quantities</div>

$$
T = \Phi^{-1}(1-K), \qquad i = \frac{\varphi(T)}{K}, \qquad i_c = -\frac{iK}{1-K}
$$

$$
v_1 = 1 - R^{2}\,i\,(i - T), \qquad v_0 = 1 - R^{2}\,i_c\,(i_c - T)
$$

$K$ is trait prevalence. Cases and controls have **different** score variances
under a liability threshold, which is the entire reason the next step is not the
textbook equal-variance formula.

</div>
<div class="step" data-key>
<div class="label">5 · discrimination</div>

$$
\mathrm{AUC} = \Phi\!\left(\frac{(i - i_c)\sqrt{R^{2}}}{\sqrt{v_1 + v_0}}\right)
$$

Wray et al. 2010. **Exact at zero free parameters** — fed the true liability
$R^{2}$, the best-fit prevalence recovers the truth at 0.149 against a true 0.150
across four arms. The corpus previously used
$\Phi\!\left(\sqrt{R^{2}/2(1-R^{2})}\right)$ and called it the liability AUC;
that form is biased by $-0.068$ on binary traits.

</div>
</div>

## II. What limits it

### The closing law

$$
\mathrm{Risk} = \tilde\varepsilon^{2}\left[\frac{d_0}{2n} + \sum_i \frac{1}{2\,m\,p_i} + r_\perp^{2}\right]
$$

<div class="terms">
<span><b>d₀/2n</b> source GWAS</span>
<span><b>Σ 1/2mpᵢ</b> target diversity</span>
<span><b>r⊥²</b> the wall</span>
</div>

Three budgets the field usually conflates. $n$ is the source sample size, $m$ is
the number of **distinct** target cohorts, $p_i$ is how much each cohort's
calibration tells you. The practical consequence is unintuitive: **increasing
source sample size does nothing at all to the third term.**

### How to split calibration data

$$
n'^{*} \approx 3.29\,\tau, \qquad \mathrm{Risk}^{*} \approx \frac{16.11}{N\tau}
$$

At fixed total data, risk falls like $1/m$ in the number of distinct cohorts —
until each cohort is too small for its own calibration to saturate, and then it
turns back up. The optimum sits at about 3.3 LD decay lengths of markers per
cohort, and $n'^{*}/\tau$ stays inside $[3.27,\,3.50]$ across a thousandfold
range in $\tau$, so it is a rule rather than a fitted point.

Risk is inverse in $\tau$ as well as in $N$, so **longer LD decay is a benefit at
fixed marker count**, not a cost.

### Replication per cohort

$$
\mathrm{reliability} \geq \tau \quad\Longleftrightarrow\quad B \geq \frac{c\,\tau}{p\,(1-\tau)}
$$

An equivalence, so it states exactly when you have enough rather than merely that
more is better. The $1/(1-\tau)$ blow-up is the shape to remember: **each
additional nine of reliability costs about ten times the replicates.** A study
reaching reliability $0.153$ at $B = 16$ needed roughly 350 for $\tau = 0.8$.

## III. Two walls, and they fail differently

<div class="pair">
<div class="card">

### The support wall — a cost

$$
r_\perp = 0 \quad\Longleftrightarrow\quad \eta > 0
$$

If no marker is a deterministic function of the others, there is no
information-theoretic floor: risk decays to zero in sample size and cohort count.
The portability gap is a sample-size problem, not a permanent limit.

<div class="caution">

**It must be quoted with its cost.** A direction resolved only at coupling order
$k$ needs about $(C/\eta)^{2k}$ samples, which exceeds every bound as $k$ grows.
Only $k = 1$ gives the quadratic $1/\eta^{2}$ form. A high-order direction is a
wall in practice while remaining a sample-size problem in theory.

</div>

The forward direction is derived. The converse rests on a witness known to exist
and is carried as an explicit hypothesis.

</div>
<div class="card">

### The environmental wall — a degeneracy

If the environmental gradient is collinear with the ancestry gradient, an entire
one-parameter family of genetic and environmental splits produces **identical**
cohort shifts. Not approximately equal — equal. No cohort-level calibration
separates them at any sample size, and the level-set collapse carries that to
every threshold metric.

<div class="caution">

**Identifiability returns only off the diagonal.** Two cohorts at the same
nonzero genetic distance with different environments suffice. Cohorts strung
along a single ancestry gradient — what biobank recruitment tends to produce —
identify nothing, however many there are.

</div>

This qualifies the diversity result above: more cohorts help the drift term, and
do not help here unless they differ along a second axis.

</div>
</div>

## IV. Measured against prediction

Predictions were fixed before the runs. No fitted constants except where stated.

<div class="table-wrap">

| Claim | Predicted | Measured | Note |
|---|---:|---:|---|
| Hudson–Nei conversion $2G/(1+G)$ | exact | $3.6\times10^{-16}$ | 16 cells; textbook fact, machine-checked bridge |
| Liability AUC at true prevalence | — | RMSE $0.0126$ | zero free parameters |
| Equal-variance AUC on binary traits | — | RMSE $0.0708$ | bias $-0.068$; replaced |
| Permeability constant | $1.000$ | $1.035 / 1.013 / 1.029 / 0.984$ | constant and shape both |
| Sealing exponent $p \sim \eta^{2}$ | $2$ | $1.9999$ | converges as $\eta \to 0$ |
| Cohort optimum | $3.29\,\tau$ | matches | stable over $1000\times$ in $\tau$ |
| Curvature at the long-memory boundary | $-\frac{3}{2}\delta$ | $10^{-7}$ | geometry is flat where it matters |
| Split $F_{ST}$ against msprime | $t/(t+2N_e)$ | $\lvert z\rvert \leq 0.89$ | 200 replicates; every cell within one SE |
| Serial-founder ceiling | $0.18497$ | $0.19248$ | $3.9\%$, zero free parameters |
| Brier $=$ MSE $+$ uncertainty | identity | $R^{2} = 0.994$ | leave-one-out, nothing fitted |

</div>

## V. What was wrong

Each of these was a definition returning a well-formed number that no range check
could question.

<div class="table-wrap">

| Definition | Claimed | Computed | Error |
|---|---|---|---:|
| `founderFst` | between-population $F_{ST}$ | heterozygosity loss | ratio $2.001$ |
| `hudsonFst` | Hudson's estimator | Nei's $G_{ST}$ | up to $2\times$ |
| `effectiveSampleSizeSE` | a sample size | inverse-variance weight | $-50$ to $-98\%$ |
| `haplotypeEffectEstimationVariance` | $\sigma^{2}/(nf(1-f))$ | $\sigma^{2}/(nf)$ | $-50\%$ at $f = 0.5$ |
| `expectedSegmentLength` | tract mean | wrong arguments | map length spurious |
| AUC family | liability-threshold | equal-variance | $-0.068$ |

</div>

The last row produced a headline theorem that was $f(x) = f(x)$ under two names,
provable by `rfl`, which would have held just as well had AUC been wildly *not*
preserved. Its docstring had already been corrected and did not help, because a
theorem statement is built out of identifiers.

## VI. Coverage

A definition counts as covered only if a deliberately corrupted version is
rejected.

<div class="stats">
<div class="stat"><div class="value">21.2%</div><div class="key">gated overall</div><div class="sub">259 of 1219</div></div>
<div class="stat"><div class="value">8.9%</div><div class="key">against a proved bound</div><div class="sub">108 — the honest floor</div></div>
<div class="stat"><div class="value">30.5%</div><div class="key">among extractable</div><div class="sub">857 reachable at all</div></div>
<div class="stat"><div class="value">80%</div><div class="key">checks that discriminate</div><div class="sub">110 of 137 cross-checked</div></div>
</div>

100% is not reachable: type aliases and measure-theoretic integrals are not
checkable by any translator. The gap between 8.9% and 21.2% is coverage graded
against bounds inferred from a name rather than proved — real, but weaker than it
reads.

## VII. What this does not model

Environmental heterogeneity appears in none of the three budgets. If
socioeconomic measures explain as much individual-level variance as genetic
distance does, the largest component of what is observed sits outside every
equation here. That is now a theorem rather than an omission — see the
environmental wall — but it bounds what the rest is worth.

Also absent: admixture, assortative mating, ascertainment in the source GWAS, and
any difference in allele-frequency spectra beyond the $F_{ST}$ summary.
