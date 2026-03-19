#!/bin/bash
awk '
  /^theorem/ {
    in_theorem = 1
    theorem_name = $2
    body = ""
    lines_count = 0
  }
  in_theorem {
    body = body "\n" $0
    lines_count++
  }
  in_theorem && /:= by$/ {
    in_proof = 1
  }
  in_proof && /^[ \t]*linarith[ \t]*$/ {
    if (lines_count <= 6) print theorem_name
  }
  in_proof && /^[ \t]*ring[ \t]*$/ {
    if (lines_count <= 6) print theorem_name
  }
  in_proof && /^[ \t]*nlinarith[ \t]*$/ {
    if (lines_count <= 6) print theorem_name
  }
' proofs/Calibrator/*.lean
