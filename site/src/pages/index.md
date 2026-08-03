---
layout: ../layouts/Page.astro
title: Polygenic score transfer
subtitle: >-
  A polygenic score loses accuracy in a population that did not train it. You
  can predict the size of that loss before you run the study.
---

## I. How to predict the accuracy

Follow five steps to predict the AUC in the target population. Read sections
II and III for the limits on these five steps.

<div class="chain">
<div class="step">
<div class="label">step 1 · calculate the genetic distance</div>

$$
F_{ST} = \frac{t}{t + 2N_e}
$$

$t$ counts the generations after the two populations separate. $N_e$ is the
effective population size.

Some sources use $1-\left(1-\frac{1}{2N_e}\right)^{t}$ instead. That
expression is the heterozygosity loss inside one population. It is a different
quantity. It is near zero when the true $F_{ST}$ is 0.5.

</div>
<div class="step">
<div class="label">step 2 · calculate the surviving signal</div>

$$
V_{\text{target}} = \rho^{2}\,(1 - F_{ST})\,V_A
$$

$V_A$ is the variance that the score explains in the source population.
$\rho^{2}$ is the part of the tagging that survives in the target population.

Genetic distance and tagging each remove signal. One number does not predict
the other.

</div>
<div class="step">
<div class="label">step 3 · calculate the accuracy</div>

$$
R^{2}_{\text{target}} = \frac{V_{\text{target}}}{V_{\text{target}} + V_E}
$$

$V_E$ is the residual variance.

</div>
<div class="step">
<div class="label">step 4 · calculate four numbers from the prevalence</div>

$$
T = \Phi^{-1}(1-K), \qquad i = \frac{\varphi(T)}{K}, \qquad i_c = -\frac{iK}{1-K}
$$

$$
v_1 = 1 - R^{2}\,i\,(i - T), \qquad v_0 = 1 - R^{2}\,i_c\,(i_c - T)
$$

$K$ is the trait prevalence. $T$ is the liability threshold. $i$ and $i_c$ are
the mean liabilities in cases and in controls. $v_1$ and $v_0$ are the score
variances in cases and in controls.

The two variances differ. You need both of them in step 5.

</div>
<div class="step" data-key>
<div class="label">step 5 · calculate the AUC</div>

$$
\mathrm{AUC} = \Phi\!\left(\frac{(i - i_c)\sqrt{R^{2}}}{\sqrt{v_1 + v_0}}\right)
$$

Wray et al. 2010 published this formula. It contains no fitted number.

We supplied a true $R^{2}$ and solved for the prevalence. The answer was 0.149.
The true prevalence is 0.150.

An older formula, $\Phi\!\left(\sqrt{R^{2}/2(1-R^{2})}\right)$, treats the two
variances as equal. On binary traits it is 0.068 AUC too low. We removed it.

</div>
</div>

## II. The error has three parts

$$
\mathrm{Risk} = \tilde\varepsilon^{2}\left[\frac{d_0}{2n} + \sum_i \frac{1}{2\,m\,p_i} + r_\perp^{2}\right]
$$

<div class="terms">
<span><b>d₀/2n</b> source GWAS size</span>
<span><b>Σ 1/2mpᵢ</b> number of target cohorts</span>
<span><b>r⊥²</b> the fixed error</span>
</div>

Each part shrinks with a different kind of data. $n$ is the source sample size.
$m$ counts the separate target cohorts. $p_i$ is the information from one
cohort.

A bigger source GWAS reduces the first part only. It does nothing to the third
part.

### How many target cohorts to collect

$$
n'^{*} \approx 3.29\,\tau, \qquad \mathrm{Risk}^{*} \approx \frac{16.11}{N\tau}
$$

Take a fixed total number of genotyped markers. More separate cohorts reduce
the error, and the error drops as $1/m$. Each cohort must still contain enough
markers for its own calibration. Past that point more cohorts make the error
worse.

The best cohort contains about 3.3 LD decay lengths of markers. The ratio
$n'^{*}/\tau$ stays between 3.27 and 3.50 across a factor of 1000 in $\tau$.
You can therefore use this rule in other studies.

A longer LD decay length also reduces the error at a fixed marker count.

### How many times to fit each cohort

$$
\mathrm{reliability} \geq \tau \quad\Longleftrightarrow\quad B \geq \frac{c\,\tau}{p\,(1-\tau)}
$$

$B$ is the number of separate calibration fits for one cohort. Each fit uses a
different subsample of that cohort. $\tau$ is the reliability you need. The
condition is exact in both directions. You therefore know when you have enough
fits and when you do not.

The cost grows very fast as $\tau$ approaches 1. Each further nine costs
about ten times more fits.

One study fitted each cohort 16 times and reached a reliability of 0.153. That
study needed about 350 fits per cohort for a reliability of 0.8.

## III. Two limits

Two separate effects limit the accuracy. You can pay for the first one with
more samples. No sample size removes the second one.

<div class="pair">
<div class="card">

### Limit 1 · markers that copy each other

$$
r_\perp = 0 \quad\Longleftrightarrow\quad \eta > 0
$$

$\eta$ measures how far the markers are from perfect LD. LD pruning removes any
marker that another marker predicts exactly. After pruning, $\eta$ is more than
zero and the fixed error is zero. The whole error then shrinks as the sample
size and the cohort count grow.

<div class="caution">

**CAUTION: Always quote the cost with this result.** Some differences between
populations are cheap to measure and some are very expensive. The expensive
ones need about $(C/\eta)^{2k}$ samples, where $k$ counts how indirectly the
difference appears in the data. The count grows without limit as $k$ grows.

An expensive difference is a hard limit in a real study. Enough samples remove
it in principle only.

</div>

We proved one direction: LD pruning removes the fixed error. We did not prove
the reverse direction. That proof needs a worked example, and nobody built one
yet.

</div>
<div class="card">

### Limit 2 · environment that changes with ancestry

Environment often varies along the same axis as ancestry. When it does, a whole
range of different genetic and environmental splits produces exactly the same
result in every cohort. The results are equal, not close.

No amount of cohort data separates the genetic part from the environmental
part. This is true for AUC, sensitivity, PPV, and every other threshold
measure.

<div class="caution">

**CAUTION: Collect two cohorts that have the same genetic distance and
different environments.** That pair separates the genetic part from the
environmental part. Cohorts that differ only in genetic distance never separate
them, whatever their number.

</div>

This result limits the rule in section II. More cohorts reduce the
genetic-distance part of the error. They reduce this part only when the
environments differ.

</div>
</div>

## IV. What we measured

We fixed each prediction before the run. No prediction uses a fitted number,
except where the table states one.

<div class="table-wrap">

| What we predicted | Prediction | Measurement | Note |
|---|---:|---:|---|
| Hudson $F_{ST}$ from Nei $G_{ST}$ | exact | $3.6\times10^{-16}$ | 16 cells |
| AUC from $R^{2}$, true prevalence | — | RMSE $0.0126$ | no fitted number |
| The older AUC formula, binary traits | — | RMSE $0.0708$ | 0.068 too low, and we removed it |
| Information per cohort | $1.000$ | $1.035 / 1.013 / 1.029 / 0.984$ | four separate runs |
| Information near perfect LD | $\eta^{2}$ | exponent $1.9999$ | as $\eta$ approaches 0 |
| Best markers per cohort | $3.29\,\tau$ | agrees | across a factor of 1000 in $\tau$ |
| Curvature at the long-memory edge | $-\frac{3}{2}\delta$ | $10^{-7}$ | the geometry is flat |
| $F_{ST}$ after a split, against msprime | $t/(t+2N_e)$ | $\lvert z\rvert \leq 0.89$ | 200 replicates, all cells within 1 SE |
| Highest $F_{ST}$ a serial founder model reaches | $0.18497$ | $0.19248$ | 3.9% apart, no fitted number |
| Brier score $=$ MSE $+$ base rate term | exact | $R^{2} = 0.994$ | leave-one-out, no fit |

</div>

## V. Errors we found and corrected

Each quantity in this table returned a number in the right range. A range check
therefore finds none of these errors.

<div class="table-wrap">

| The quantity | What it computed instead | Size of the error |
|---|---|---:|
| Genetic distance between two populations | heterozygosity loss inside one population | 2.001 times out |
| The Hudson estimator of genetic distance | the Nei estimator | up to 2 times out |
| An effective sample size | an inverse-variance weight | 50% to 98% too low |
| The variance of a haplotype effect estimate | the same formula without the $(1-f)$ term | 50% too low at $f = 0.5$ |
| The mean length of an ancestry tract | a formula that reads the genetic map length | the map length does not belong |
| The AUC from $R^{2}$ | one score variance for cases and controls | 0.068 AUC |

</div>

We also found a theorem that compared a formula with itself under two names.
Lean accepted it in one step. It stays true even when the score transfers no
accuracy at all. Somebody corrected the comment above it first. That correction
changed nothing, because Lean reads the names and not the comment.

## VI. How much of the project is tested

A definition counts as tested only when a check rejects a corrupted copy of the
code.

<div class="stats">
<div class="stat"><div class="value">21.2%</div><div class="key">tested overall</div><div class="sub">259 of 1219</div></div>
<div class="stat"><div class="value">8.9%</div><div class="key">against a proved bound</div><div class="sub">108 definitions</div></div>
<div class="stat"><div class="value">30.5%</div><div class="key">of the reachable set</div><div class="sub">857 are reachable</div></div>
<div class="stat"><div class="value">80%</div><div class="key">checks that can fail</div><div class="sub">110 of 137</div></div>
</div>

The project cannot reach 100%. Some definitions are pure names, and some are
integrals. No automatic check reads either kind.

The definitions between 8.9% and 21.2% have bounds that come from a name.
Nobody proved those bounds. Those definitions are weaker evidence than the
count of 21.2% implies.

## VII. What the proofs assume

Lean reports no gap in any proof. That report is true. It is also incomplete,
because a theorem can carry a condition that nobody proved.

We counted every such condition. Most are ordinary limits on the inputs, such
as a prevalence between 0 and 1. Those are correct and they must stay. Seven
are results from outside this work that we use and do not prove.

We now prove each of those seven, or we delete the result that needs it.

## VIII. What these laws leave out

None of the three parts contains a term for environment. Socioeconomic
measures can explain as much difference between individuals as genetic distance
explains. The largest term is then outside every formula on this page.

We proved that this gap exists. Section III describes it as limit 2. The gap
still restricts the value of the other results.

These laws also leave out admixture, assortative mating, ascertainment in the
source GWAS, and any difference in allele frequency spectra beyond $F_{ST}$.
