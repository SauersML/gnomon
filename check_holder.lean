import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.Analysis.SpecialFunctions.Pow.Real

open MeasureTheory

example : ENNReal.HolderTriple 2 2 1 := by infer_instance

#check MemLp.integrable_mul
