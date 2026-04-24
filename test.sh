#!/bin/bash
export PATH="$HOME/.elan/bin:$PATH"
cd proofs
lake exe cache get
lake build Calibrator.HaplotypeTheory
