#!/usr/bin/env python3
"""Generate a PLINK1 fileset with strong LD so plink2 emits LD-compressed
(type 2/3) PGEN records, then report the record-type histogram of the pgen."""
import random, struct, sys, os

random.seed(7)
N = 200          # samples
M = 1500         # variants

# PLINK1 2-bit codes: 0b00 homALT, 0b01 missing, 0b10 het, 0b11 homREF
def pack(row):
    out = bytearray((N + 3) // 4)
    for i, g in enumerate(row):
        out[i >> 2] |= (g & 3) << (2 * (i & 3))
    return bytes(out)

rows = []
base = None
for v in range(M):
    if base is None or v % 40 == 0:
        # fresh haplotype block
        p = random.uniform(0.02, 0.5)
        base = []
        for _ in range(N):
            a = (random.random() < p) + (random.random() < p)
            base.append([3, 2, 0][a])
        row = list(base)
    else:
        # near-copy of block anchor -> strong LD -> type 2/3 records
        row = list(base)
        for _ in range(random.randint(0, 4)):
            row[random.randrange(N)] = random.choice([0, 2, 3])
        if v % 7 == 0:
            # flip REF/ALT sense -> exercises type 3 (inverted LD)
            row = [{0: 3, 3: 0, 2: 2, 1: 1}[g] for g in row]
    # sprinkle missing calls
    for _ in range(random.randint(0, 3)):
        row[random.randrange(N)] = 1
    rows.append(row)

d = os.path.dirname(os.path.abspath(__file__))
with open(f"{d}/ld.bed", "wb") as f:
    f.write(bytes([0x6c, 0x1b, 0x01]))
    for r in rows:
        f.write(pack(r))

with open(f"{d}/ld.bim", "w") as f:
    for v in range(M):
        f.write(f"1\trs{v}\t0\t{(v + 1) * 100}\tA\tG\n")

with open(f"{d}/ld.fam", "w") as f:
    for s in range(N):
        f.write(f"FAM{s}\tIID{s}\t0\t0\t{1 + s % 2}\t-9\n")
print(f"wrote {M} variants x {N} samples")
