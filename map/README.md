Because the biobank and the single individual are standardized on the same reference, and placed on the same per-axis scale, the directional geometry is preserved. Fitting on the projected biobank means residual magnitude shrinkage is just a shared, axis-wise rescaling, so both a single new datapoint and the biobank data inhabit the same commensurately shrunken space and distances. Consequently, de-shrinkage or OADP/AP rotations would merely re-inflate coordinates and risk needless perturbation.


```
INPUTS
  Biobank summary:
    D, N, K
    ε₀ ∈ (0,1)                         // e.g., 0.05
    SNPs_ref[ D ]:
      id, counted_allele, μⱼ, σⱼ>0, Lⱼ[ K ]   // columns of L are unit: ∑ⱼ Lⱼk² = 1
    λ[ K ]                               // sample eigenvalues from standardized X′
  New individual (imputed):
    SNPs_new:
      id, REF, ALT, strand_info, dosage g_est, quality Q (expected ∈ [0,1]), counted_allele_new?
  Options:
    drop_ambiguous_strands ∈ {t,f}
    cos_min_reliable ∈ (0,1)             // e.g., 0.90
    min_overlap_frac ∈ (0,1)             // e.g., 0.20
    allow_freq_flip ∈ {t,f}
    use_AP_for_spikes_only ∈ {t,f}
    safe_eps > 0

DERIVED
  γ ← D / N
  MP_edge ← (1 + √γ)²

HELPERS
  clamp01(x)       → min(max(x,0),1)
  nonneg(x)        → max(x,0)
  dot(a,b)         → ∑ᵢ aᵢ bᵢ
  norm2(a)         → √(∑ᵢ aᵢ²)
  is_spike(λₖ,γ)   → λₖ > MP_edge
  safe_std(σ)      → max(σ, safe_eps)
  harmonize_dosage(snp_new, snp_ref):
    if allele sets + counted allele align: return g_est
    if REF/ALT swapped with mirrored count: return (2 − g_est)
    if strand flip resolvable (and allow_freq_flip): apply complement ± (2 − g_est)
    if ambiguous A/T or C/G and unresolved (or drop_ambiguous_strands): return ⊥
    else return ⊥

CORE: project_one_individual
  P_proj[ K ] ← 0
  R2[ K ]     ← 0
  used_count  ← 0
  excluded ← {no_ref_match:0, allele_conflict:0, ambiguous:0, missing_Q:0, other:0}

  FOR each snp_new IN SNPs_new:
    if snp_new.id ∉ SNPs_ref: excluded.no_ref_match++; continue
    snp_ref ← SNPs_ref[snp_new.id]

    g_aligned ← harmonize_dosage(snp_new, snp_ref)
    if g_aligned = ⊥: excluded.allele_conflict++; continue

    Q ← clamp01(snp_new.Q)
    Q_eff ← ε₀ + Q(1 − ε₀)
    μ ← snp_ref.μⱼ
    σ ← safe_std(snp_ref.σⱼ)

    x′ ← (g_aligned − μ) / (σ √Q_eff)

    FOR k=1..K:
      ℓ ← snp_ref.Lⱼ[k]
      P_proj[k] += x′ · ℓ
      R2[k]     += ℓ²

    used_count++

  // Geometry fix (unit-axis)
  P′[ K ] ← 0
  cos_align[ K ] ← 0
  FOR k=1..K:
    if R2[k]=0: P′[k]=0; cos_align[k]=0
    else:
      r ← √R2[k]
      P′[k] ← P_proj[k] / r
      cos_align[k] ← r                         // √Rₖ² = cos(angle(Lₖ,uₖ))

  // High-D shrinkage correction (AP)
  ρ[ K ] ← 1
  FOR k=1..K:
    apply_AP ← true
    if use_AP_for_spikes_only ∧ ¬is_spike(λ[k],γ): apply_AP ← false
    if apply_AP:
      Δ ← (λ[k] + γ − 1)² − 4γ
      if Δ < 0: apply_AP ← false
      else:
        λ_pop ← ( (λ[k] + γ − 1) + √Δ ) / 2
        ρ[k]  ← λ_pop / λ[k]
        if (ρ[k] ≤ 0) ∨ (¬finite(ρ[k])): ρ[k] ← 1

  P″[ K ] ← 0
  FOR k=1..K: P″[k] ← P′[k] / ρ[k]

  overlap_frac ← used_count / D
  warn_overlap ← (overlap_frac < min_overlap_frac)
  warn_cos     ← [ for k: cos_align[k] < cos_min_reliable ]

  RETURN
    scores:
      P_proj       // truncated-loading dot product
      P′           // geometry-consistent (unit-axis)
      P″           // final AP-corrected (recommended)
    reliability:
      cos_align    // √Rₖ² per PC (cosine similarity to original Lₖ)
      overlap_frac, used_count, D
    AP:
      ρ, γ, MP_edge
    bookkeeping:
      excluded
      per_PC_flags: {applied_AP:bool, is_spike:bool, low_cos:bool}
```
