import re

with open('proofs/Calibrator.lean', 'r') as f:
    content = f.read()

# I messed up `simpa using` replacements causing errors like `unexpected token 'using'`
# Let's revert by using git reset
