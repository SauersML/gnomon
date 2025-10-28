
Q1. “Full likelihood” for a subdistribution hazard?

Let F₁(t) be the cumulative incidence for the event of interest and define the Fine–Gray subdistribution hazard
λ*(t) = dF₁(t)/dt ÷ {1 − F₁(t)}.
Then S*(t) := exp{−∫₀ᵗ λ*(s) ds} = 1 − F₁(t).

• Is S*(t) a valid survival?
It is a proper survivor for a defective distribution: S*(0)=1 and S*(t)↓1−π₁ as t→∞, where π₁:=F₁(∞)=P(J=1). Unless π₁=1, S*(∞)>0, so it is not a proper survival function for a fully specified event time on [0,∞); it’s the complement of the cause-1 CIF.

• Is f*(t)=λ*(t)S*(t) a valid density?
f*(t)=dF₁(t)/dt (“subdensity”). It integrates to ∫₀^∞ f*(t)dt=F₁(∞)=π₁<1 in general. So it is not a proper density on [0,∞) unless you renormalize by π₁ (which would give the conditional density of T given J=1, but that conditional does not have hazard λ*).

• Is L=∏[λ* (tᵢ)]^{dᵢ}×S*(tᵢ)/S*(t₀ᵢ) a valid full likelihood?
No. That object treats all non–cause-1 outcomes as if they were independent right censoring for the subdistribution process, which they are not. A full likelihood for the observed competing-risks data requires the joint law of (T,J); specifying only λ*(t) leaves the probabilities and timing of other causes (and their effect on the observed process) unspecified. You can obtain a proper likelihood only by modeling the full system (e.g., all cause-specific hazards, or λ* plus enough structure for the other causes). The standard Fine–Gray estimator is thus not an MLE from a full likelihood; it is an M-estimator obtained from partial/pseudo-likelihood–type estimating equations.

• Does that “likelihood” have a unique maximum; do MLE asymptotics apply?
Because it is not a true likelihood for the observed data, uniqueness and MLE asymptotics are not the right questions. For the semiparametric Fine–Gray proportional subdistribution hazards model, the usual estimating-equation machinery gives consistency and asymptotic normality under the standard regularity conditions (i.i.d. sampling, independent censoring, correct specification). Variances come from the sandwich/influence-function form. Uniqueness is as for Cox-type fits: typically yes under identifiability (no separation, full rank), but it is not a concavity guarantee from a true log-likelihood.

Q2. Hazard for Royston–Parmar (log cumulative hazard scale)

Let η(t)=log H(t)=s(u) with u=log t and s a spline. Then
H(t)=exp{η(t)} and
h(t)=dH/dt = H(t)×dη/dt.

By the chain rule,
dη/dt = (ds/du)×(du/dt) = s′(u)×(1/t).

Hence
h(t) = H(t) × s′(log t) / t.

That derivation is correct. If you include covariates via η(t|x)=s(log t)+xᵀβ (the PH version), the derivative term comes only from s′(·), so
h(t|x) = H(t|x) × s′(log t) / t
with H(t|x)=exp{s(log t)+xᵀβ}. If you allow time-dependent effects (interactions with u), their u-derivatives also appear inside s′(·).

A practical note: because h(t)≥0 requires dH/dt≥0, you need s′(u)≥0 wherever the model is evaluated; with flexible splines that’s not automatic, so in practice one uses modest degrees of freedom and checks fitted hazards.

Q3. “Jacobian” term in the likelihood

For an observed event time t, the contribution is f(t)=h(t)S(t). Using the expression above,

log f(t) = log h(t) + log S(t)
= log[H(t) s′(log t)/t] − H(t)
= η(t) + log s′(log t) − log t − H(t).

The −log t is exactly the Jacobian from the transformation u=log t (since |du/dt|=1/t). If you bundle the chain-rule factor into a single derivative, ∂η/∂t := s′(log t)/t, then

log f(t) = log[H(t) × (∂η/∂t)] − H(t)
= η(t) + log(∂η/∂t) − H(t),

which is algebraically identical—the −log t has simply been absorbed in ∂η/∂t. There is no “extra” Jacobian beyond this chain-rule term.

Bottom line

• Subdistribution hazards yield a defective survival S*(t)=1−F₁(t) and a subdensity integrating to π₁. By themselves they do not furnish a full likelihood for competing-risks data; Fine–Gray’s estimator is based on estimating equations with standard large-sample properties, not on MLE from a proper full likelihood in λ* alone.

• For Royston–Parmar on the log-cumulative-hazard scale, h(t)=H(t)s′(log t)/t, and the −log t term you see in log f(t) is exactly the Jacobian from u=log t (or, equivalently, it’s included when you write ∂η/∂t).

# Q1 — “Full likelihood” for subdistribution hazards (Fine–Gray)

**Is (S^*(t)) a proper survival function?**
Not in the usual sense. For a chosen cause (k), the *subdistribution* (F_k(t)=\Pr(T\le t,\ \text{cause}=k)) is a **defective** (a.k.a. *improper*) cdf: (\lim_{t\to\infty}F_k(t)=\pi_k\le 1). Fine & Gray formalize this by introducing an improper random variable (T^*=\mathbf 1{\text{cause}=k}T+(1-\mathbf 1{\text{cause}=k})\cdot\infty). Then (S_k^*(t)=1-F_k(t)=\Pr(T^*>t)) is decreasing with (S_k^*(\infty)=1-\pi_k>0) whenever some competing events occur. So (S^*(t)) is the survivor of a *defective* distribution (a subprobability measure), not a proper survival that converges to 0. ([JSTOR][1])

**Is (f^*(t)=\lambda^*(t),S^*(t)) a valid density?**
Yes, but of the *improper* distribution above: (f^*(t)=\frac{d}{dt}F_k(t)) is non-negative and integrates to (\pi_k) (not 1). It is therefore often called a *subdensity*. ([PMC][2])

**Can you do maximum likelihood with a parametric baseline, or must you use a partial likelihood?**

* The *classic* Fine–Gray model is **semiparametric**: it specifies proportional subdistribution hazards with an **unspecified** baseline and is estimated by a **modified/weighted partial likelihood** using an altered risk set. That estimating objective is a partial (or “pseudo-”) likelihood, not a true full likelihood, and it is **not** the route we pursue in this project. ([JSTOR][1])
* If you **fully specify** the baseline subdistribution hazard (or equivalently parameterize the CIF directly), then full-likelihood (or Bayesian) estimation is available and has been developed in several settings. Examples:

  * **Parametric CIF** via an *improper* baseline such as Gompertz (natural here because of the mass at (+\infty)); estimation proceeds by full likelihood. ([Carolina Digital Repository][3])
  * **Bayesian fully specified subdistribution hazards** with an improper baseline prior/parametric form. ([PMC][4])
  * **Sieve semiparametric MLE** for Fine–Gray under interval censoring (baseline approximated by splines/B-splines), with asymptotics for both baseline and regression parameters. ([PMC][5])
  * Alternative **parametric** approaches for competing risks (mixtures; direct CIF modeling) that bypass the FG partial likelihood entirely. ([PMC][6])

**Any caveats about “full likelihood” in subdistribution models?**
Two conceptual issues are worth keeping in mind when treating subdistribution hazards as if they underwrote a conventional likelihood:

1. The Fine–Gray *risk set* includes individuals who have already failed from other causes, with time-dependent weights; this is precisely why the classic estimator is partial/pseudo-likelihood rather than a straightforward full likelihood with ordinary risk sets. ([SAS Support][7])
2. Fitting **multiple** Fine–Gray models (one per cause) and then summing predicted CIFs can be problematic; it can exceed 1 in finite samples and, more importantly, reflects internal incoherence of simultaneous proportional subdistribution hazards across causes. Use with care, or favor cause-specific hazard modeling when absolute risk for *multiple* causes is the target. ([PubMed][8])

**Bottom line (Q1):** (S^*(t)) is a survivor of a defective distribution; (f^*(t)=\lambda^*(t)S^*(t)) is a subdensity integrating to (\pi_k). With a fully specified baseline (or parametric CIF), **MLE is valid** and used in the literature; we therefore adopt the full-likelihood formulation. The ubiquitous historical Fine–Gray estimator remains a **partial likelihood** because its baseline is unspecified. ([JSTOR][1])

# Q2 — Royston–Parmar (RP) hazard: the derivative

In the proportional-hazards RP model you write
[
\log H(t\mid x);=; s(\log t) ;+; x^\top\beta,
]
with (s(\cdot)) a restricted cubic spline in (u=\log t). Then
[
H(t\mid x)=\exp{s(\log t)+x^\top\beta}.
]
The hazard is (h(t\mid x)=\frac{d}{dt}H(t\mid x)). By the chain rule,
[
\boxed{;h(t\mid x);=;H(t\mid x),\frac{d}{dt}{s(\log t)+x^\top\beta}
;=;H(t\mid x),\frac{s'(\log t)}{t};}
]
since (d(\log t)/dt=1/t) and (x^\top\beta) is constant in (t).

Royston & Parmar write the same relationship explicitly: with (s=u\mapsto s(u)) and (u=\log t),
[
S(t)=\exp{-\exp s},\quad f(t)=\frac{ds}{dt},\exp{s-\exp s},\quad
h(t)=\frac{ds}{dt},\exp{s},
]
so (ds/dt=s'(\log t)/t) must appear. That factor is not optional; it’s the Jacobian from the log-time reparameterization. ([ResearchGate][9])

**Bottom line (Q2):** Yes—(h(t)=dH/dt) **must** include the chain-rule term (\big(s'(\log t)\big)/t). Any formula omitting (1/t) is mathematically incorrect. ([ResearchGate][9])

# Q3 — Change of variables in survival likelihoods

Suppose you work on the transformed time (u=\log t). General change-of-variables says for a monotone (u=g(t)),
[
f_T(t)=f_U(u),\bigg|\frac{du}{dt}\bigg|.
]
With (u=\log t), (\frac{du}{dt}=1/t), so (f_T(t)=f_U(\log t)/t). This is the same ubiquitous (1/t) factor seen in log-normal densities (a concrete instance of the Jacobian). ([PennState: Statistics Online Courses][10])

How does this play out in the **survival** likelihood? The individual contribution is
[
L_i ;=; f_T(t_i)^{\delta_i}; S_T(t_i)^{1-\delta_i}
;=; \big(h_T(t_i),S_T(t_i)\big)^{\delta_i} S_T(t_i)^{1-\delta_i},
]
and when your model is specified *on the (u)-scale* (e.g., (\eta(u)=\log H(t)) with (u=\log t)), the Jacobian is already embedded when you map back to (t): (h_T(t)=\frac{d}{dt}H(t)=H(t),\eta'(u),(du/dt)=H(t),\eta'(u)/t). You do **not** tack an extra (-\log t) term onto the log-likelihood by hand; it appears automatically through (h(t)) (or (f(t))) once you differentiate correctly. This is exactly what the RP derivation in Q2 shows. ([ResearchGate][9])

**Bottom line (Q3):** Modeling on (u=\log t) does *mathematically* introduce a Jacobian. In survival likelihoods that Jacobian is absorbed through the derivative step that turns your modeled cumulative hazard (or cdf) into a hazard (or pdf). If you compute (h(t)) correctly (via chain rule), the likelihood is already on the right scale; no extra, ad hoc (-\log t) term is needed. ([PennState: Statistics Online Courses][10])

---

## Pointers to the literature (by question)

**Q1 (subdistribution hazards & likelihoods)**
• Fine & Gray’s original formulation and the “improper” (T^*). ([JSTOR][1])
• Subdistribution = defective distribution; formal definitions and notation. ([CRAN][11])
• Conceptual notes on censoring and subdistribution hazards. ([PMC][2])
• The standard estimator is a modified/weighted partial likelihood; SAS tech note. ([SAS Support][7])
• Fully parametric/Bayesian subdistribution hazards. ([PMC][4])
• Parametric CIFs using improper baselines (e.g., Gompertz). ([Carolina Digital Repository][3])
• Sieve semiparametric MLE for FG under interval censoring. ([PMC][5])
• Cautionary results on multiple FG fits / incoherent risk sums. ([PubMed][8])

**Q2 (Royston–Parmar mathematics)**
• Royston–Parmar (2002) and an explicit formula showing (f(t)) and (h(t)) contain (ds/dt). ([ResearchGate][9])
• Tutorials and software docs that restate (\log H(t)=s(\log t)) (context for the derivative you take). ([PubMed][12])

**Q3 (change of variables/Jacobians in survival)**
• General Jacobian/change-of-variables rule (textbook lecture notes). ([PennState: Statistics Online Courses][10])
• Concrete example: the log-normal pdf’s (1/t) factor arises from (u=\log t). ([Wikipedia][13])
• RP derivation again shows the same (1/t) term via (ds/dt). ([ResearchGate][9])


[1]: https://www.jstor.org/stable/2670170?utm_source=chatgpt.com "A Proportional Hazards Model for the Subdistribution"
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5379776/?utm_source=chatgpt.com "The importance of censoring in competing risks analysis ..."
[3]: https://cdr.lib.unc.edu/downloads/n296x675h?utm_source=chatgpt.com "NIH Public Access"
[4]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3374158/?utm_source=chatgpt.com "Bayesian Inference of the Fully Specified Subdistribution ..."
[5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4629270/?utm_source=chatgpt.com "The Fine–Gray Model Under Interval Censored Competing ..."
[6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3069508/?utm_source=chatgpt.com "Parametric mixture models to evaluate and summarize ..."
[7]: https://support.sas.com/resources/papers/proceedings18/2159-2018.pdf?utm_source=chatgpt.com "Cause-Specific Analysis of Competing Risks Using the ..."
[8]: https://pubmed.ncbi.nlm.nih.gov/33969508/?utm_source=chatgpt.com "Fine-Gray subdistribution hazard models to simultaneously ..."
[9]: https://www.researchgate.net/publication/11177715_Flexible_Parametric_Proportional-Hazards_and_Proportional-Odds_Models_for_Censored_Survival_Data_with_Application_to_Prognostic_Modelling_and_Estimation_of_Treatment_Effects?utm_source=chatgpt.com "(PDF) Flexible Parametric Proportional-Hazards and ..."
[10]: https://online.stat.psu.edu/stat414/lesson/23/23.1?utm_source=chatgpt.com "23.1 - Change-of-Variables Technique | STAT 414"
[11]: https://cran.r-project.org/web/packages/survival/vignettes/compete.pdf?utm_source=chatgpt.com "Multi-state models and competing risks"
[12]: https://pubmed.ncbi.nlm.nih.gov/12210632/?utm_source=chatgpt.com "Flexible parametric proportional-hazards and ..."
[13]: https://en.wikipedia.org/wiki/Log-normal_distribution?utm_source=chatgpt.com "Log-normal distribution"


Fine–Gray is a proportional **subdistribution** hazards (PSH) model. The classic estimator uses **weighted risk-set** estimating equations—equivalently, a *pseudo* partial likelihood formed on Fine–Gray risk sets that retain subjects who have experienced competing events, with inverse-probability-of-censoring weights (IPCW). ([PMC][1]) Our implementation, however, parameterizes the baseline subdistribution hazard with Royston–Parmar splines, which turns the objective into a bona fide full likelihood of the form described in Q1.

Option A—per-subject likelihood contributions `ℓ_i = d_i \log λ_i^*(t_i) - (H_i(t_i) - H_i(a_{entry,i}))`—becomes valid once the baseline is fully specified. This is precisely the “direct likelihood” perspective used in stpm2cr and related flexible parametric models; it **is** compatible with the Fine–Gray scale when the baseline is parametric. ([PubMed][2])

Two practical notes when adopting the full-likelihood view:

* Left truncation and time-varying entry are handled by subtracting `H_i(a_{entry,i})` rather than by pruning risk sets. ([PMC][1])
* The CIF follows from the subdistribution hazard as (F_1(t)=1-\exp{-H^*(t)}); because we model `H^*` directly, this identity now sits inside the likelihood and yields coherent probabilities. ([PMC][1])

So, for a Royston–Parmar model **on the Fine–Gray scale** with a parameterized baseline, the statistically consistent estimation framework is the **full likelihood** assembled per subject, not the historical risk-set partial likelihood.

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7216972/?utm_source=chatgpt.com "On the relation between the cause‐specific hazard and ..."
[2]: https://pubmed.ncbi.nlm.nih.gov/30305806/?utm_source=chatgpt.com "stpm2cr: A flexible parametric competing risks model using ..."

# Q2 — What is the hazard (h(t)) under Royston–Parmar on log-time?

Royston–Parmar (RP) models set (\eta(t)=\log H(t)=s(\log t)) (plus covariates). Differentiating (H(t)=\exp{\eta(t)}) gives

[
h(t)=\frac{dH}{dt}
= \exp{\eta(t)},\frac{d\eta}{dt}
= \exp{s(\log t)};\frac{ds(\log t)}{dt}.
]

By the chain rule, (\frac{ds(\log t)}{dt}=\frac{ds}{du}\cdot\frac{du}{dt}=\frac{s'(\log t)}{t}). Thus

[
\boxed{,h(t)=\exp{s(\log t)},\frac{s'(\log t)}{t},}
\qquad\text{(Option A).}
]

Exactly this derivative-of-the-spline times (\exp{\eta}) appears in the RP literature and in the `rstpm2` documentation for hazard predictions. See, e.g., Mozumder et al. (their Eq. 17) and Clements’ notes: “(\lambda(t|x)=\Lambda(t|x),s'(\log t)/t)”. ([SciSpace][1])

# Q3 — Does the log-likelihood include a Jacobian from (u=\log t)?

For an observed event at time (t),

[
\log f(t)=\log h(t)+\log S(t)=\eta(t)+\log!\Big(\frac{d\eta}{dt}\Big)-H(t).
]

If you write the derivative **with respect to (u=\log t)**, then (\frac{d\eta}{dt}=\frac{d\eta}{du}\cdot\frac{1}{t}), so

[
\log f(t)=\eta(t)+\log!\Big(\frac{d\eta}{du}\Big) ;-; \log t ;-; H(t).
]

So:

* If you keep (\frac{d\eta}{dt}) as a single object, there is **no separate Jacobian term**—it’s already **absorbed**. **(Option B)**
* If you rewrite via (u=\log t), an explicit **(-\log t)** appears. **(Option A)**

They are **algebraically identical**; there is no “double counting.” This exact equivalence is why some software reports a log-likelihood that differs by a constant: Stata traditionally **adds (\sum \log t_i)** (over uncensored events) to make the reported log-likelihood invariant to the choice of time scale, whereas many R packages leave the raw term in. The `flexsurv` manual notes this explicitly (and the same logic applies to RP models): “Stata … adds the sum of the log uncensored survival times to the log-likelihood to remove dependency on the time scale.” The constant does **not** affect MLEs. ([CRAN][3])

**Does `rstpm2`’s log-likelihood include a term depending on the observed (t_i)?** Yes—through (\log(d\eta/dt)). Written on the log-time scale that is (\log s'(\log t_i)-\log t_i). The (-\log t_i) is **data-only** (no parameters), so packages may add or drop an offset for reporting; the MLEs are unchanged. `rstpm2` fits by full MLE for right-censored and left-truncated data on the RP scale, consistent with the derivation above. ([CRAN][4])

---

[1]: https://scispace.com/pdf/direct-likelihood-inference-on-the-cause-specific-cumulative-15nwx4rb08.pdf "Direct likelihood inference on the cause-specific cumulative incidence function: A flexible parametric regression modelling approach."
[2]: https://journals.sagepub.com/doi/pdf/10.1177/1536867X1701700212?utm_source=chatgpt.com "A flexible parametric competing-risks model using a direct ..."
[3]: https://cran.r-project.org/web/packages/flexsurv/flexsurv.pdf?utm_source=chatgpt.com "flexsurv: Flexible Parametric Survival and Multi-State Models"
[4]: https://cran.r-project.org/web/packages/rstpm2/vignettes/SimpleGuide.Rnw?utm_source=chatgpt.com "rstpm2: a simple guide"


---

### 1) How RP (Royston–Parmar) models handle competing risks in practice

* **Cause-specific hazards route (most common):** Fit a separate RP model for each cause on the **log cumulative hazard** (or log hazard) scale, then derive cause-specific cumulative incidence functions (CIFs) from the fitted hazards. This is a **full-likelihood** approach—the RP model is parametric and estimated by ML—not a Cox-style partial likelihood. See the BMC Methods paper advocating RP for competing risks via cause-specific hazards, and the original/extended RP work. ([BioMed Central][1])

* **Direct CIF route (subdistribution scale):** Use **stpm2cr** to model the CIF *directly* via a smooth parametric subdistribution hazard; this is a **direct likelihood** (full-likelihood) formulation designed to avoid numerical integration and to simplify prediction. ([CRAN][2])

Practically: when analysts say they used “Royston–Parmar competing risks,” they mean one of the two full-likelihood paths above (cause-specific hazards via **stpm2/strcs/stpm3** or subdistribution/CIF via **stpm2cr**). They are **not** using Cox partial likelihood (unless they explicitly fit Cox models). ([pclambert.net][3])

Contrast with **Fine–Gray** in standard software: those are typically semi-parametric with estimating-equation or partial-likelihood style risk sets (e.g., **cmprsk** in R), i.e., baseline not parameterized. ([CRAN][4])

---

### 2) “Fine–Gray with a parametric baseline (RP splines)” — best approach

If you want a **parametric** subdistribution model using RP splines, use **stpm2cr**. It implements a **direct likelihood** for the cause-specific CIF via a flexible (restricted cubic spline) specification of the *log cumulative subdistribution hazard*, supports delayed entry (left truncation) and time-varying effects, and was written precisely for this purpose. Seminal references and software docs: ([CRAN][2])

By contrast, classic Fine–Gray implementations (R **cmprsk**, many texts) are semi-parametric with weighted score/partial-likelihood style estimation and do **not** parameterize the baseline with RP splines. (Stata’s **stcrreg** follows Fine–Gray; its manual describes ML under that framework but still does not give you a flexible parametric baseline.) ([CRAN][4])

**Bottom line:** if you truly want “Fine–Gray + parametric baseline,” use **stpm2cr** (Stata). In R, there isn’t a turnkey exact analog; people either (i) fit cause-specific RP models and derive CIFs, or (ii) use generalized survival modeling in **rstpm2** on the CIF scale with custom code. ([BioMed Central][1])

---

### 3) Time-varying effects and the ∂η/∂t term in the hazard

In RP models on the **log cumulative hazard** scale, the linear predictor η(t, x) defines **H(t|x)=exp{η(t, x)}**. The hazard is
[
h(t|x)=\frac{d}{dt}H(t|x)=\exp{\eta(t,x)},\frac{\partial\eta(t,x)}{\partial t}.
]
That **derivative term is intrinsic**—it comes straight from the chain rule. When you include time-varying effects (interactions of covariates with spline functions of time), **∂η/∂t must be included** to get the hazard (and any hazard-based contrasts) right. This is exactly how the methodology is presented, and how **stpm2/strcs/stpm3** compute hazards. ([Stata][5])

Are there published demonstrations of bias **from omitting** ∂η/∂t? I can’t find an example where a paper purposely drops it; modern software computes it automatically. What *is* published are (i) the formula itself (as above), (ii) notes that modeling on the log-cumulative-hazard scale can complicate **interpretation** of hazard-ratio curves when you have multiple time-dependent effects (another sign the derivative is essential), and (iii) simulation work showing RP spline baselines accurately approximate complex hazards (which rely on the correct derivative). ([pclambert.net][6])

---

### 4) Computational trade-offs: full likelihood vs partial likelihood with 100k+ subjects and left truncation

* **Full-likelihood RP (stpm2/strcs/stpm3; rstpm2):**

  * **Scales well** in large samples because evaluation is essentially **O(n × K)** where *K* is the (small) number of spline basis functions (plus any TVE bases). On the standard RP scale (log cumulative hazard), **no numerical time-integration** is needed; predictions are fast and smooth. ([pclambert.net][3])
  * **Left truncation** (delayed entry) is natively supported and simply modifies the individual likelihood contributions—well documented in Stata and R manuals. ([SAGE Journals][7])
  * Newer Stata code (**stpm3**, **standsurv**) emphasizes **analytic derivatives** and speed; the authors explicitly discuss computation and have worked on accelerating tricky cases (log-hazard scale being more numerically delicate). ([EconPapers][8])

* **Partial-likelihood Cox/Fine–Gray (for contrast only):**

  * **coxph** and kin use efficient risk-set updates and are extremely optimized for **very large** n, with straightforward support for left truncation via (start, stop] risk intervals. They remain relevant when the goal is *only* relative effects and no parametric baseline is required, but they fall outside the scope of the full-likelihood approach we are implementing. ([CRAN][9])
  * Classic Fine–Gray (e.g., **cmprsk**) uses weighted estimating equations/partial-likelihood logic; it scales well but does **not** give you a parameterized baseline for extrapolation or smooth hazard/CIF shapes without post-hoc smoothing, which is why we avoid it here. ([CRAN][4])

**Practical heuristics for 100k+ with left truncation:**

* If you need **extrapolation, absolute risks, smooth hazards/CIFs, or standardized contrasts**, RP full-likelihood is often preferable despite a somewhat heavier per-observation cost. Use a modest number of baseline and TVE spline degrees of freedom and the log-cumulative-hazard scale for stability. ([pclambert.net][3])
* If you only need **relative effects** or plan to combine models in a prediction tool with heavy resampling, partial likelihood (Cox/cause-specific + Aalen–Johansen; or semi-parametric Fine–Gray) can be faster and memory-lighter, but it forfeits the calibrated absolute-risk outputs we require. ([CRAN][9])

---

## Citations (key sources)

* RP competing-risks via **direct likelihood** on the CIF scale (introduces **stpm2cr**): Lambert et al., *Stat Med* 2017; overview and software notes. ([CRAN][2])
* **stpm2cr** documentation (includes left truncation): Mozumder et al., *Stata J* 2017. ([PMC][10])
* RP for **cause-specific hazards** in competing risks: Hinchliffe & Lambert, *BMC Med Res Methodol* 2013. ([BioMed Central][1])
* RP methodology & implementation (full likelihood): Royston & Parmar 2002; Lambert & Royston 2009; Stata/R docs. ([Wiley Online Library][11])
* Hazard on RP scale requires the **derivative** term (explicit formula): Lambert slides (Stata Nordic/Baltic Users’ Meeting 2018). ([Stata][5])
* Fine–Gray is semi-parametric in standard toolchains (estimating-equations/partial-likelihood flavor): **cmprsk** manual; original Fine–Gray paper. (Stata **stcrreg** follows Fine–Gray and does not give a flexible parametric baseline.) ([CRAN][4])
* Large-N/left-truncation support and computational notes: **rstpm2** manual (left truncation supported); Stata **stpm3** release notes & slides on speed/analytic derivatives. ([CRAN][12])
* RP spline baselines approximate complex hazards well (simulation): Rutherford, Crowther & Lambert 2015. ([Taylor & Francis Online][13])

---

### TL;DR answers

1. RP models for competing risks are implemented with **full likelihood**, either by modeling **cause-specific hazards** or by **directly modeling the CIF** (stpm2cr). Not partial likelihood. ([BioMed Central][1])
2. To fit a “Fine–Gray with RP baseline,” use **stpm2cr** (direct-likelihood subdistribution model with RP splines). Standard Fine–Gray toolchains are semi-parametric and don’t parameterize the baseline. ([CRAN][2])
3. The **∂η/∂t** term is **non-negotiable**—it’s how hazards are obtained from the RP linear predictor. I’m not aware of published case studies that *omit* it and quantify the bias; software computes it for you. ([Stata][5])
4. With 100k+ and left truncation, **full-likelihood RP** remains feasible (especially on the log-cumulative-hazard scale) and is our chosen path because it delivers smooth baselines, extrapolation, and standardized absolute risks; partial-likelihood Cox/Fine–Gray remains an alternative only when relative effects suffice. ([CRAN][9])

[1]: https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/1471-2288-13-13?utm_source=chatgpt.com "Flexible parametric modelling of cause-specific hazards to ..."
[2]: https://cran.r-project.org/web/packages/rstpm2/rstpm2.pdf "rstpm2: Smooth Survival Models, Including Generalized Survival Models"
[3]: https://pclambert.net/pdf/Stata2024_Germany_Paul_Lambert.pdf?utm_source=chatgpt.com "Recent developments in the fitting and assessment of flexible ..."
[4]: https://cran.r-project.org/web/packages/cmprsk/cmprsk.pdf?utm_source=chatgpt.com "cmprsk: Subdistribution Analysis of Competing Risks - CRAN"
[5]: https://www.stata.com/meeting/nordic-and-baltic18/slides/nordic-and-baltic18_Lambert.pdf "Standardized survival curves and related measures using flexible parametric survival models"
[6]: https://pclambert.net/courses/stpm3course_Stockholm_27Sept2024.pdf?utm_source=chatgpt.com "Modelling survival data using flexible parametric survival ..."
[7]: https://journals.sagepub.com/doi/pdf/10.1177/1536867x0900900206?utm_source=chatgpt.com "Further Development of Flexible Parametric Models for ..."
[8]: https://econpapers.repec.org/RePEc%3Aboc%3Abocode%3As459207?utm_source=chatgpt.com "STPM3: Stata module to fit flexible parametric survival ..."
[9]: https://cran.r-project.org/web/packages/survival/survival.pdf?utm_source=chatgpt.com "Survival Analysis"
[10]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6175038/?utm_source=chatgpt.com "stpm2cr: A flexible parametric competing risks model using ..."
[11]: https://onlinelibrary.wiley.com/doi/10.1002/sim.1203?utm_source=chatgpt.com "Flexible parametric proportional‐hazards and ..."
[12]: https://cran.r-project.org/web/packages/rstpm2/rstpm2.pdf?utm_source=chatgpt.com "rstpm2: Smooth Survival Models, Including Generalized ..."
[13]: https://www.tandfonline.com/doi/pdf/10.1080/00949655.2013.845890?utm_source=chatgpt.com "The use of restricted cubic splines to approximate complex ..."


