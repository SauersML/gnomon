# Test fixtures

`ld.{bed,bim,fam}` / `ld_p.{pgen,pvar,psam}` — 200 samples x 1500 variants,
generated with strong linkage disequilibrium so that `plink2 --make-pgen`
emits mostly LD-compressed PGEN records (record types 2 and 3 account for
~97% of the file). Those record types are the ones that decode incorrectly
if a reader assumes a strictly sequential pass, so they are what the
`score/tests/pgen_parity.rs` parity tests need in order to be meaningful.

Regenerate with `scripts/gen_pgen_fixture.py`, then:

    plink2 --bfile ld --make-pgen --out ld_p

`xy_sex.{vcf.gz,pgen,pvar,psam,bed,bim,fam}` — 8 1000 Genomes samples, 4 recorded
male and 4 female, and every 200th biallelic SNP of the 30x chrX/Y SNP VCF. The
`terms/cross_format_tests.rs` test runs sex inference on the .bed, the .pgen and
the .vcf.gz, and again with one female recorded as male in the .psam and the .fam.
All five tables must be identical: a sex check must not read the label it infers.
Regenerate from the chrX/Y SNP VCF and its .psam (`xysnp` on MSI):

    awk -F'\t' 'NR > 1 && $4 == 1 { print $1 }' xysnp.psam | head -4 > samples
    awk -F'\t' 'NR > 1 && $4 == 2 { print $1 }' xysnp.psam | head -4 >> samples
    awk -F'\t' 'NR == 1 { print "#IID\tSEX"; next } { print $1 "\t" $4 }' xysnp.psam > sex.txt
    bcftools view -S samples -m2 -M2 -v snps xysnp.vcf.gz -Ov \
      | awk '/^#/ { print; next } { n++; if (n % 200 == 1) print }' \
      | bcftools view -Oz -o xy_sex.vcf.gz
    plink2 --vcf xy_sex.vcf.gz --update-sex sex.txt --make-pgen --out xy_sex
    plink2 --pfile xy_sex --make-bed --out xy_sex
