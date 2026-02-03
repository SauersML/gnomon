import Mathlib.Analysis.InnerProductSpace.Projection
import Mathlib.LinearAlgebra.Matrix.ToLin
import Mathlib.LinearAlgebra.Matrix.Rank
import Mathlib.Topology.Algebra.Module.FiniteDimension

open Matrix
open InnerProductSpace

-- Check orthogonalProjection
#check orthogonalProjection
#check orthogonalProjection_eq_of_dist_le

-- Check Submodule.Closed
#check Submodule.Closed
#check Submodule.isClosed_of_finiteDimensional

-- Check Matrix.rank_eq...
#check Matrix.rank_eq_finrank_range_toLin
#check Matrix.rank_eq_finrank_range_toLin'
#check Matrix.toLin'

-- Check toLin' definition
variable (n m : Type) [Fintype n] [Fintype m] [DecidableEq n] [DecidableEq m]
variable (A : Matrix m n ℝ)
#check toLin' A
