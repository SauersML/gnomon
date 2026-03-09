#!/usr/bin/env bash
set -euo pipefail

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

plink_zip_url="https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20250819.zip"

cd "$workdir"

curl -fsSL -o plink.zip "$plink_zip_url"
unzip -q plink.zip

echo "PLINK version: $(./plink --version | head -n 1)"

cat > old.vcf <<'VCF'
##fileformat=VCFv4.2
##contig=<ID=1,length=1000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	ind_100001	ind_100002
1	1	rs1	A	G	.	PASS	.	GT	0/1	1/1
VCF

cat > fixed.vcf <<'VCF'
##fileformat=VCFv4.2
##contig=<ID=1,length=1000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	ind100001	ind100002
1	1	rs1	A	G	.	PASS	.	GT	0/1	1/1
VCF

./plink --vcf old.vcf --allow-extra-chr --biallelic-only strict --make-bed --out old --silent
./plink --vcf fixed.vcf --allow-extra-chr --biallelic-only strict --make-bed --out fixed --silent

echo "Generated old FAM:"
cat old.fam
echo "Generated fixed FAM:"
cat fixed.fam

python3 <<'PY'
import csv


def parse_fam(path: str) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle, delimiter=" ") if row]
    return [[field for field in row if field] for row in rows]


def iid_to_fid(path: str) -> dict[str, str]:
    rows = parse_fam(path)
    return {row[1]: row[0] for row in rows}


old_rows = parse_fam("old.fam")
fixed_rows = parse_fam("fixed.fam")
print("Parsed old FAM IDs:", [{"FID": row[0], "IID": row[1]} for row in old_rows])
print("Parsed fixed FAM IDs:", [{"FID": row[0], "IID": row[1]} for row in fixed_rows])

old_map = iid_to_fid("old.fam")
fixed_map = iid_to_fid("fixed.fam")

old_requested = ["ind_100001"]
fixed_requested = ["ind100001"]

missing_old = [iid for iid in old_requested if iid not in old_map]
if not missing_old:
    raise SystemExit("Expected old underscore-delimited IDs to fail, but they did not.")

missing_fixed = [iid for iid in fixed_requested if iid not in fixed_map]
if missing_fixed:
    raise SystemExit(f"Fixed PLINK-safe IDs still failed lookup: {missing_fixed}")

print(f"Replicated root cause: missing {missing_old[0]!r} in old FAM IID column")
print("Verified fix end-to-end: fixed IDs survive PLINK import and lookup succeeds.")
PY
