# Test fixtures

`ld.{bed,bim,fam}` / `ld_p.{pgen,pvar,psam}` — 200 samples x 1500 variants,
generated with strong linkage disequilibrium so that `plink2 --make-pgen`
emits mostly LD-compressed PGEN records (record types 2 and 3 account for
~97% of the file). Those record types are the ones that decode incorrectly
if a reader assumes a strictly sequential pass, so they are what the
`score/tests/pgen_parity.rs` parity tests need in order to be meaningful.

Regenerate with `scripts/gen_pgen_fixture.py`, then:

    plink2 --bfile ld --make-pgen --out ld_p
