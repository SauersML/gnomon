import re

with open('proofs/Calibrator.lean', 'r') as f:
    content = f.read()

theorems = re.findall(r'theorem\s+(\w+)\s*(.*?)(?:\s*:=|\s*by)', content, re.DOTALL)

for name, args in theorems:
    print(f"--- Theorem: {name} ---")
    print(args.strip())
    print()
