"""Small random PLINK fileset and score files for the gscore smoke test.

  make_fixture.py <dir>

Writes panel.{bed,bim,fam} (1,003 people, so the last byte has padding bits; about 2%
missing calls; loci with two or three .bim rows), native.tsv (five scores with empty
cells, flipped alleles, unmatched pairs and duplicate lines, weights from 1e-9 to 2.5),
pgs.txt (PGS Catalog format) and catalog/ (80 small PGS Catalog files, which take the
many-score catalog kernel).
"""
import os
import random
import sys

out = sys.argv[1]
rng = random.Random(20260914)
os.makedirs(os.path.join(out, "catalog"), exist_ok=True)
people = 1003
bases = "ACGT"

variants = []
pos = 1000
for i in range(700):
    pos += rng.randint(1, 5000)
    chrom = 1 if i < 400 else 2
    a1, a2 = rng.sample(bases, 2)
    variants.append((chrom, pos, a1, a2))
    if i % 50 == 7:
        variants.append((chrom, pos, a1, a2))
    if i % 90 == 11:
        b1, b2 = rng.sample([b for b in bases if b != a1], 2)
        variants.append((chrom, pos, b1, b2))

with open(os.path.join(out, "panel.fam"), "w") as f:
    for p in range(people):
        f.write(f"F{p} I{p} 0 0 0 -9\n")
with open(os.path.join(out, "panel.bim"), "w") as f:
    for i, (chrom, p, a1, a2) in enumerate(variants):
        f.write(f"{chrom}\tv{i}\t0\t{p}\t{a1}\t{a2}\n")
with open(os.path.join(out, "panel.bed"), "wb") as f:
    f.write(bytes([0x6C, 0x1B, 0x01]))
    for _ in variants:
        row = bytearray((people + 3) // 4)
        for p in range(people):
            code = 1 if rng.random() < 0.02 else rng.choice((0, 2, 3))
            row[p // 4] |= code << (2 * (p % 4))
        f.write(bytes(row))


def weight():
    magnitude = 10 ** rng.uniform(-9, 0.4)
    return f"{rng.choice((-1, 1)) * magnitude:.9g}"


with open(os.path.join(out, "native.tsv"), "w") as f:
    f.write("variant_id\teffect_allele\tother_allele\tS1\tS2\tS3\tS4\tS5\n")
    for chrom, p, a1, a2 in variants:
        if rng.random() < 0.15:
            continue
        effect, other = (a1, a2) if rng.random() < 0.5 else (a2, a1)
        if rng.random() < 0.03:
            other = "T" if other != "T" else "C"
        cells = [weight() if rng.random() < 0.8 else "" for _ in range(5)]
        if not any(cells):
            cells[0] = weight()
        f.write(f"{chrom}:{p}\t{effect}\t{other}\t" + "\t".join(cells) + "\n")
        if rng.random() < 0.02:
            f.write(f"{chrom}:{p}\t{effect}\t{other}\t{weight()}\t\t\t\t\n")


def pgs_file(path, name, picks):
    with open(path, "w") as f:
        f.write(f"#pgs_id={name}\n#genome_build=GRCh38\n")
        f.write("chr_name\tchr_position\teffect_allele\tother_allele\teffect_weight\n")
        for chrom, p, a1, a2 in picks:
            effect, other = (a1, a2) if rng.random() < 0.5 else (a2, a1)
            f.write(f"{chrom}\t{p}\t{effect}\t{other}\t{weight()}\n")


pgs_file(os.path.join(out, "pgs.txt"), "PGSSMOKE", sorted(rng.sample(variants, 500)))
for k in range(80):
    pgs_file(os.path.join(out, "catalog", f"C{k:03d}.txt"), f"C{k:03d}", sorted(rng.sample(variants, rng.randint(5, 40))))
print(f"fixture: {people} people, {len(variants)} .bim rows, in {out}")
