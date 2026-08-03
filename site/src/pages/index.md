---
layout: ../layouts/Page.astro
title: Polygenic score transfer
subtitle: >-
  These laws change source-population parameters into an expected accuracy in a
  target population. This page gives each law, its measurement, and its limit.
meta: The corpus contains 1219 definitions in 105 modules. Every proof is complete.
---

## I. The chain

Five steps change source-population parameters into an expected accuracy. These
five steps are the complete method. The later sections give the limits.

<div class="chain">
<div class="step">
<div class="label">step 1 · divergence</div>

$$
F_{ST} = \frac{t}{t + 2N_e}
$$

$t$ counts the generations after the split. $N_e$ is the effective population
size. The corpus name for this law is `coalFst`.

The expression $1-\left(1-\frac{1}{2N_e}\right)^{t}$ is a different quantity. It
is the heterozygosity loss in one population. It is near zero when the true
$F_{ST}$ between two populations equals 0.5.

</div>
<div class="step">
<div class="label">step 2 · signal after transfer</div>

$$
V_{\text{target}} = \rho^{2}\,(1 - F_{ST})\,V_A
$$

$V_A$ is the variance that the score explains in the source population.
$\rho^{2}$ is the part of the tagging that stays in the target population.

Drift and tagging multiply. Neither quantity determines the other.

</div>
<div class="step">
<div class="label">step 3 · accuracy on the liability scale</div>

$$
R^{2}_{\text{target}} = \frac{V_{\text{target}}}{V_{\text{target}} + V_E}
$$

$V_E$ is the residual variance.

</div>
<div class="step">
<div class="label">step 4 · threshold quantities</div>

$$
T = \Phi^{-1}(1-K), \qquad i = \frac{\varphi(T)}{K}, \qquad i_c = -\frac{iK}{1-K}
$$

$$
v_1 = 1 - R^{2}\,i\,(i - T), \qquad v_0 = 1 - R^{2}\,i_c\,(i_c - T)
$$

$K$ is the trait prevalence. Cases and controls have different score variances
under a liability threshold. Step 5 therefore uses a different formula from the
equal-variance one.

</div>
<div class="step" data-key>
<div class="label">step 5 · discrimination</div>

$$
\mathrm{AUC} = \Phi\!\left(\frac{(i - i_c)\sqrt{R^{2}}}{\sqrt{v_1 + v_0}}\right)
$$

Wray et al. 2010 give this formula. It uses no free parameter.

We supplied the true liability $R^{2}$. The best-fit prevalence was 0.149. The
true prevalence is 0.150.

The corpus once used $\Phi\!\left(\sqrt{R^{2}/2(1-R^{2})}\right)$ and named it
the liability AUC. That formula carries a bias of $-0.068$ on binary traits. We
replaced it.

</div>
</div>

## II. Limits

### The closing law

$$
\mathrm{Risk} = \tilde\varepsilon^{2}\left[\frac{d_0}{2n} + \sum_i \frac{1}{2\,m\,p_i} + r_\perp^{2}\right]
$$

<div class="terms">
<span><b>d₀/2n</b> source GWAS</span>
<span><b>Σ 1/2mpᵢ</b> target diversity</span>
<span><b>r⊥²</b> the wall</span>
</div>

This law separates three budgets. $n$ is the source sample size. $m$ counts the
separate target cohorts. $p_i$ is the information from one cohort calibration.

A larger source sample size does not change the third term.

### How to divide the calibration data

$$
n'^{*} \approx 3.29\,\tau, \qquad \mathrm{Risk}^{*} \approx \frac{16.11}{N\tau}
$$

At a constant total data volume, the risk decreases as $1/m$ with each added
cohort. Each cohort needs enough markers for its own calibration. At a smaller
cohort size the risk increases again.

The best cohort contains about 3.3 LD decay lengths of markers. The ratio
$n'^{*}/\tau$ stays between 3.27 and 3.50 across a factor of 1000 in $\tau$.
This range makes the result a rule and not a fitted point.

The risk also decreases as $\tau$ increases. A longer LD decay length is an
advantage at a constant marker count.

### Replication in one cohort

$$
\mathrm{reliability} \geq \tau \quad\Longleftrightarrow\quad B \geq \frac{c\,\tau}{p\,(1-\tau)}
$$

This law is an equivalence. It gives the exact condition for sufficient
replicates.

The $1/(1-\tau)$ term controls the cost. Each further nine of reliability costs
about ten times more replicates.

One study used $B = 16$ and reached a reliability of 0.153. That study needed
about 350 replicates for a reliability of 0.8.

## III. Two walls

The two walls fail in different ways. Wall 1 sets a cost. Wall 2 sets a
degeneracy.

<div class="pair">
<div class="card">

### Wall 1 · support

$$
r_\perp = 0 \quad\Longleftrightarrow\quad \eta > 0
$$

$\eta$ is the distance from perfect LD. LD pruning removes every marker that is
a function of the other markers. No floor on the risk then remains. The risk
decreases to zero as the sample size and the cohort count increase.

<div class="caution">

**CAUTION: Always give the cost with this result.** A direction at coupling
order $k$ needs about $(C/\eta)^{2k}$ samples. This count increases to more than
every bound as $k$ increases. Only $k = 1$ gives the $1/\eta^{2}$ form.

A direction at a high coupling order is a wall in practice. It stays a
sample-size problem in theory.

</div>

We derived the forward direction. The converse needs a witness. The corpus
carries that witness as an explicit hypothesis.

</div>
<div class="card">

### Wall 2 · environment

The environmental gradient can be collinear with the ancestry gradient. A
one-parameter family of genetic and environmental splits then produces identical
cohort shifts. The shifts are equal, not approximately equal.

No cohort calibration separates the two effects at any sample size. The
level-set collapse extends this result to every threshold metric.

<div class="caution">

**CAUTION: Identifiability exists only off the diagonal.** Use two cohorts at
the same genetic distance with different environments. Cohorts along one
ancestry gradient give no identifiability. A larger cohort count does not help.

</div>

This result limits the earlier cohort rule. More cohorts are an advantage for
the drift term. They are an advantage here only when they differ along a second
axis.

</div>
</div>

## IV. Measurements

We fixed each prediction before the run. No prediction uses a fitted constant,
except where the table states one.

<div class="table-wrap">

| Claim | Prediction | Measurement | Note |
|---|---:|---:|---|
| Hudson-Nei conversion $2G/(1+G)$ | exact | $3.6\times10^{-16}$ | 16 cells |
| Liability AUC at the true prevalence | — | RMSE $0.0126$ | no free parameter |
| Equal-variance AUC on binary traits | — | RMSE $0.0708$ | bias $-0.068$; we replaced it |
| Permeability constant | $1.000$ | $1.035 / 1.013 / 1.029 / 0.984$ | constant and shape |
| Sealing exponent $p \sim \eta^{2}$ | $2$ | $1.9999$ | as $\eta$ decreases to 0 |
| Cohort optimum | $3.29\,\tau$ | agrees | across a factor of 1000 in $\tau$ |
| Curvature at the long-memory boundary | $-\frac{3}{2}\delta$ | $10^{-7}$ | the geometry is flat |
| Split $F_{ST}$ against msprime | $t/(t+2N_e)$ | $\lvert z\rvert \leq 0.89$ | 200 replicates; all cells within 1 SE |
| Serial-founder ceiling | $0.18497$ | $0.19248$ | $3.9\%$; no free parameter |
| Brier $=$ MSE $+$ uncertainty | identity | $R^{2} = 0.994$ | leave-one-out; no fit |

</div>

## V. Defects that we corrected

Each definition in this table returns a number in the correct range. A range
check cannot find these defects.

<div class="table-wrap">

| Definition | The name claims | The body computes | Error |
|---|---|---|---:|
| `founderFst` | $F_{ST}$ between populations | heterozygosity loss | ratio $2.001$ |
| `hudsonFst` | the Hudson estimator | the Nei $G_{ST}$ | up to $2\times$ |
| `effectiveSampleSizeSE` | a sample size | an inverse-variance weight | $-50$ to $-98\%$ |
| `haplotypeEffectEstimationVariance` | $\sigma^{2}/(nf(1-f))$ | $\sigma^{2}/(nf)$ | $-50\%$ at $f = 0.5$ |
| `expectedSegmentLength` | a tract mean | the wrong arguments | map length is spurious |
| the AUC family | the liability threshold | equal variance | $-0.068$ |

</div>

The last row gave one theorem the form $f(x) = f(x)$ under two names. Lean
closed that theorem in one step. The theorem is true even when the AUC changes.
Someone corrected the docstring first. That correction did not help. A theorem
statement uses identifiers and not prose.

## VI. Coverage

A definition counts as covered only when a check rejects a corrupt version of
the body.

<div class="stats">
<div class="stat"><div class="value">21.2%</div><div class="key">covered overall</div><div class="sub">259 of 1219</div></div>
<div class="stat"><div class="value">8.9%</div><div class="key">against a proved bound</div><div class="sub">108 definitions</div></div>
<div class="stat"><div class="value">30.5%</div><div class="key">of the extractable set</div><div class="sub">857 extract</div></div>
<div class="stat"><div class="value">80%</div><div class="key">checks that discriminate</div><div class="sub">110 of 137</div></div>
</div>

The corpus cannot reach 100%. No translator can check a type alias or a
measure-theoretic integral.

The difference between 8.9% and 21.2% covers bounds that come from a name.
Nobody proved those bounds. That coverage counts for less than the number
suggests.

## VII. Assumptions

The corpus carries 476 hypotheses as structure fields across 135 structures. A
structure field is different from an incomplete proof. No tool counts it. A
theorem that uses one gives a conditional result.

We now prove each field from Mathlib, or we remove the result that uses it.

## VIII. What these laws omit

The three budgets contain no environmental term. Socioeconomic measures can
explain as much variance between individuals as genetic distance explains. The
largest term is then outside every equation on this page.

Wall 2 makes this omission a theorem. The omission still limits the value of the
other results.

These laws also omit admixture, assortative mating, ascertainment in the source
GWAS, and any difference in allele frequency spectra beyond $F_{ST}$.
