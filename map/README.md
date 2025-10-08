### High-dimensionality projection
Because the biobank and the single individual are standardized on the same reference, and placed on the same per-axis scale, the directional geometry is preserved. Fitting on the projected biobank means residual magnitude shrinkage is just a shared, axis-wise rescaling, so both a single new datapoint and the biobank data inhabit the same commensurately shrunken space and distances. Consequently, de-shrinkage or OADP/AP rotations would merely re-inflate coordinates and risk needless perturbation.

### Missing SNVs
If we project onto a unit vector made only from the SNVs we have, missing SNVs don’t contribute signal or variance; their loading mass is subtracted from the denominator and the axis is renormalized.
