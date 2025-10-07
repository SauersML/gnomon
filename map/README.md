INPUTS
  Biobank summary:
    D                         // number of SNPs used in reference PCA
    N                         // number of individuals used in reference PCA
    K                         // number of PCs kept
    ε₀ ∈ (0,1)               // small floor for imputation quality (e.g., 0.05)
    SNPs_ref[ D ]:            // per-SNP records keyed by a harmonized ID (CHR:POS:REF:ALT and counted allele)
      id
      counted_allele          // allele used for genotype coding in reference (usually ALT)
      μⱼ                      // mean genotype in reference
      σⱼ > 0                  // std genotype in reference
      Lⱼ[ K ]                 // loadings for PCs 1..K (unit columns: ∑ⱼ Lⱼk² = 1)
    λ[ K ]                    // sample eigenvalues of reference PCA (for standardized X′)
  New individual (imputed VCF or equivalent):
    SNPs_new:                 // subset (may exceed or fall short of SNPs_ref; allele representations may differ)
      id, REF, ALT, strand_info
      dosage g_est, quality Q ∈ ℝ (expected in [0,1], but will be clamped)
      counted_allele_new      // allele count basis for dosage if explicit (else infer from INFO/ALT)
  Options / thresholds:
    drop_ambiguous_strands ∈ {true,false}             // e.g., drop A/T or C/G unless resolvable by frequency
    cos_min_reliable ∈ (0,1)                          // e.g., 0.90 (flag if √R_k² < this)
    min_overlap_frac ∈ (0,1)                          // e.g., 0.20 (flag if |S_overlap|/D below this)
    allow_freq_flip ∈ {true,false}                    // if true, use allele freq to resolve strand flips
    use_AP_for_spikes_only ∈ {true,false}             // apply AP only if λ_k is a spike
    safe_eps > 0                                       // tiny number to avoid division by zero in practice

DERIVED CONSTANTS
  γ ← D / N
  MP_edge ← (1 + √γ)²                                  // Marchenko–Pastur upper edge (for standardized noise=1)

HELPERS
  clamp01(x)          → min(max(x, 0), 1)
  nonneg(x)           → max(x, 0)
  dot(a[·], b[·])     → ∑ aᵢ bᵢ
  norm2(a[·])         → √(∑ aᵢ²)
  is_spike(λ_k, γ)    → λ_k > MP_edge
  harmonize_dosage(snp_new, snp_ref):
      // returns dosage aligned to ref counted allele, or ⊥ if cannot be reconciled
      if allele sets match and counted alleles align:
          return g_est (possibly unchanged)
      if REF/ALT swapped but counted allele mirrored:
          return 2 − g_est                             // diploid flip
      if strand flip resolvable (and allow_freq_flip):
          apply complement + (optional) 2−g_est
      if ambiguous A/T or C/G and not confidently resolvable:
          return ⊥
  safe_std(σ)         → max(σ, safe_eps)

CORE PROCEDURE: project_one_individual(…)
  Initialize accumulators:
    P_proj[ K ]     ← 0
    R2[ K ]         ← 0
    used_count      ← 0
    excluded_counts ← {no_ref_match:0, allele_conflict:0, ambiguous:0, missing_Q:0, other:0}

  For each snp_new in SNPs_new:
    if snp_new.id ∉ SNPs_ref: 
        excluded_counts.no_ref_match += 1
        continue
    snp_ref ← SNPs_ref[ snp_new.id ]

    g_aligned ← harmonize_dosage(snp_new, snp_ref)
    if g_aligned = ⊥:
        excluded_counts.allele_conflict += 1   // or .ambiguous as appropriate
        continue

    Q ← clamp01(snp_new.Q)                     // tolerate out-of-range inputs
    Q_eff ← ε₀ + Q(1−ε₀)                       // effective quality
    μ ← snp_ref.μⱼ
    σ ← safe_std(snp_ref.σⱼ)

    x′ ← (g_aligned − μ) / (σ √Q_eff)          // quality-aware standardization

    // Update all PCs in one pass (streaming-friendly)
    for k in 1..K:
      ℓ ← snp_ref.Lⱼ[k]
      P_proj[k] += x′ · ℓ
      R2[k]     += ℓ²

    used_count += 1

  // Geometry fix (unit-axis projection) and reliability
  P_prime[ K ] ← 0
  cos_align[ K ] ← 0
  for k in 1..K:
    if R2[k] = 0:
      P_prime[k]  ← 0
      cos_align[k]← 0
    else:
      r ← √R2[k]
      P_prime[k]  ← P_proj[k] / r
      cos_align[k]← r                               // √R_k² is cosine similarity with original axis

  // High-dimensional shrinkage correction (AP)
  ρ[ K ] ← 1
  for k in 1..K:
    apply_AP ← true
    if use_AP_for_spikes_only and not is_spike(λ[k], γ):
      apply_AP ← false

    if apply_AP:
      Δ ← (λ[k] + γ − 1)² − 4γ
      if Δ < 0:
        apply_AP ← false                             // no real spike separation ⇒ skip AP
      else:
        λ_pop ← ( (λ[k] + γ − 1) + √Δ ) / 2
        ρ[k]   ← λ_pop / λ[k]
        if ρ[k] ≤ 0 or not finite(ρ[k]): ρ[k] ← 1   // guardrails

  P_dblprime[ K ] ← 0
  for k in 1..K:
    P_dblprime[k] ← P_prime[k] / ρ[k]

  // QC flags
  overlap_frac ← used_count / D
  warn_overlap ← (overlap_frac < min_overlap_frac)
  warn_cos     ← [ for k in 1..K: (cos_align[k] < cos_min_reliable) ]

  RETURN:
    scores:
      P_proj        // raw dot onto truncated loading
      P_prime       // unit-axis, geometry-consistent
      P_dblprime    // final, AP-corrected (recommended)
    reliability:
      cos_align     // √R_k² per PC (cosine similarity to original L_k)
      overlap_frac
      used_count, D
    AP:
      ρ             // shrinkage factors (≈1 if little correction)
      γ, MP_edge
    bookkeeping:
      excluded_counts
      per-PC flags: {applied_AP: bool, is_spike: bool, low_cos: bool}

BULK MODE (multiple individuals)
  For each individual:
    project_one_individual(…)                       // identical logic; results are per-person
  Note: u_k (the effective unit axis) differs per person if S_overlap differs; PCs across people
        are on comparable scale but bases are not strictly identical ⇒ be cautious with distances.
