# calibrate/ PGS and PC Structure Findings

## Defaults and Shared Infrastructure
- `ModelConfig::external` and the in-tree configs default the spline difference penalty to order 2, so null spaces correspond to degree-1 polynomials (constant and linear trends).【F:calibrate/model.rs†L102-L125】
- During training, the constructor generates B-spline bases for PGS and each PC, applies weighted sum-to-zero constraints to PGS, and decomposes PC penalties into null and range components using `null_range_whiten`. The corresponding transforms and knot vectors are saved in the model config for reuse at prediction time.【F:calibrate/construction.rs†L402-L465】【F:calibrate/model.rs†L503-L616】
- The canonical design order is: intercept, each PC’s null columns (if any), each PC’s range columns, the constrained PGS main basis, then any interaction blocks. Flattening and prediction reuse this exact ordering.【F:calibrate/construction.rs†L537-L621】【F:calibrate/model.rs†L625-L784】

## PGS Main Effect
- Layout construction explicitly advances past the PGS main-effect columns without attaching any penalty, keeping the spline smooth completely unpenalized once the sum-to-zero constraint is applied.【F:calibrate/construction.rs†L238-L241】【F:calibrate/construction.rs†L498-L511】
- Training builds the PGS main basis by dropping the intercept column, enforcing the weighted sum-to-zero constraint, and storing the resulting transform; the whitened range transform is retained only for interactions.【F:calibrate/construction.rs†L402-L415】
- Prediction mirrors this by reconstructing the unconstrained basis, multiplying by the saved constraint, and inserting the constrained columns after the intercept and PC blocks; the range transform is not used for PGS main effects.【F:calibrate/model.rs†L516-L652】

## PC Main Effects
- For each PC, training slices off the intercept column, applies the order-2 difference penalty, and uses `null_range_whiten` to obtain a whitened penalized range basis plus an explicit null-space basis. Both bases are inserted into the design matrix (null first) and recorded in the layout.【F:calibrate/construction.rs†L442-L569】
- `ModelLayout::new` assigns **two** penalized blocks per PC: one for the null-space columns and one for the range-space columns. The null block receives its own penalty index (“select=TRUE” style) before the range block, ensuring the intercept/slope directions extracted from the penalty null space are also ridge-penalized. This is the double-penalty mechanism for PC mains.【F:calibrate/construction.rs†L200-L235】
- Because the default penalty order is 2, the null block captures the constant and linear trend directions of the spline. They appear in the design immediately after the global intercept and are therefore subject to the dedicated ridge penalty described above.【F:calibrate/model.rs†L102-L125】【F:calibrate/construction.rs†L537-L569】
- Prediction rebuilds both the null and range bases using the stored transforms, stitches them into the design in the same order, and therefore continues to penalize both components during scoring.【F:calibrate/model.rs†L560-L737】

## PGS×PC Interaction Effects
- When at least one PC is present, the layout adds a tensor-product block for each PC, sized as (PGS range columns × PC range columns) and tied to a single penalty index because both margins are whitened.【F:calibrate/construction.rs†L243-L259】
- Training forms these interactions by multiplying the whitened (range-only) PGS and PC bases row-wise, then orthogonalizes the tensor block against the intercept, PGS main effect, and that PC’s null+range main effects using a weighted projection. The resulting pure interaction columns are stored in the design, while the projection matrix (`interaction_orth_alpha`) is persisted for prediction.【F:calibrate/construction.rs†L623-L758】
- Prediction repeats the whitened tensor construction, retrieves the saved projection, and subtracts the main-effect space so that the interaction contributes no intercept or marginal trend. Only a single penalty matrix (identity on the interaction block) is attached to each interaction, so there is no double penalty in this pathway.【F:calibrate/model.rs†L649-L745】【F:calibrate/construction.rs†L498-L520】【F:calibrate/construction.rs†L1714-L1778】

## Double-Penalty Inventory
- PC main effects are the only place where a double penalty is enforced: every PC null-space block receives its own ridge penalty in addition to the standard range penalty, and tests verify the three-penalty layout (PC null, PC range, interaction).【F:calibrate/construction.rs†L200-L235】【F:calibrate/construction.rs†L1714-L1816】
- PGS main effects remain unpenalized after enforcing identifiability, and the intercept column is also unpenalized because it is never added to the penalty map.【F:calibrate/construction.rs†L238-L241】【F:calibrate/construction.rs†L498-L511】
- PGS×PC interactions carry a single Frobenius-normalized identity penalty derived from their whitened bases—no secondary penalty is applied because the orthogonalization removes null-space overlap instead.【F:calibrate/construction.rs†L243-L259】【F:calibrate/construction.rs†L623-L758】【F:calibrate/construction.rs†L1714-L1778】

## Direct Answers
- **Does the PC main effect have a double penalty?** Yes. Each PC’s null-space columns (containing the intercept/slope directions of the spline) get their own penalty index in addition to the range penalty, so both subspaces are ridge-penalized.【F:calibrate/construction.rs†L200-L235】【F:calibrate/construction.rs†L1714-L1816】
- **Are intercept and slope present, and are they penalized via ridge?** With the default order-2 difference penalty, the null-space basis isolates constant and linear trends; those columns are inserted immediately after the global intercept and receive the dedicated null-space penalty, so both intercept- and slope-like directions are ridge-penalized rather than left free.【F:calibrate/model.rs†L102-L125】【F:calibrate/construction.rs†L537-L569】【F:calibrate/construction.rs†L200-L235】
- **What about PC×PGS interactions?** They are built from whitened range bases, orthogonalized to remove intercept/linear components, and tied to a single penalty matrix. There is no double penalty on the interaction because the null directions are removed by construction instead.【F:calibrate/construction.rs†L623-L758】【F:calibrate/model.rs†L649-L745】【F:calibrate/construction.rs†L1714-L1778】

