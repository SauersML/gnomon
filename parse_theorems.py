import re

with open('proofs/Calibrator.lean', 'r') as f:
    content = f.read()

theorems = re.findall(r'theorem\s+(\w+)\s*(.*?)(?:\s*:=|\s*by)', content, re.DOTALL)

for name, args in theorems:
    if 'h_' in args or 'hS_' in args or 'hlam' in args or 'H' in args or 'h' in args:
        print(f"--- Theorem: {name} ---")
        print(args.strip())
        print()
