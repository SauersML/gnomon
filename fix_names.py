import re

with open('proofs/Calibrator/StatisticalGeneticsMethodology.lean', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('theorem overlap_bias_structural', 'theorem overlap_bias')
content = content.replace('theorem blocked_cv_less_biased_expected', 'theorem blocked_cv_less_biased')
content = content.replace('theorem constrained_intercept_more_powerful_expected', 'theorem constrained_intercept_more_powerful')
content = content.replace('theorem method_disagreement_increases_uncertainty_expected', 'theorem method_disagreement_increases_uncertainty')
content = content.replace('theorem common_variants_higher_correlation_structural', 'theorem common_variants_higher_correlation')

with open('proofs/Calibrator/StatisticalGeneticsMethodology.lean', 'w', encoding='utf-8') as f:
    f.write(content)
