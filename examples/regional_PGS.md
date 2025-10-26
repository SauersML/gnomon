The 17q21 inversion is on chr17 from 45585159 to 46292045 on hg38. Let's add 50 kb to each side so the region becomes 45535159 to 46342045.

We'll calculate the following scores.

**Alzheimer’s disease:**
- PGS004146
- PGS004898

**Breast cancer**
- PGS000007
- PGS000317

**Obesity**
- PGS005198
- PGS004378

Let's use the array data (though we could also stream the srWGS):
```
./gnomon/target/release/gnomon score "PGS004146 | chr17:45535159-46342045, PGS004898 | chr17:45535159-46342045, PGS000007 | chr17:45535159-46342045, PGS000317 | chr17:45535159-46342045, PGS005198 | chr17:45535159-46342045, PGS004378 | chr17:45535159-46342045" arrays
```
