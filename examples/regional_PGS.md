### Local polygenic scores

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

This should complete in a few seconds.

Let's check missingness per score:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

d = pd.read_csv('../../arrays.sscore', sep='\t')
cols = d.columns[d.columns.str.endswith('_MISSING_PCT')].tolist()

vals = d[cols]
vmin = np.nanmin(vals.to_numpy())
vmax = np.nanmax(vals.to_numpy())
bins = np.linspace(vmin, vmax, 51)

n = len(cols)
ncols = 2
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.4 * nrows), constrained_layout=True)
axes = np.atleast_2d(axes).ravel()

for i, c in enumerate(cols):
    ax = axes[i]
    ax.hist(d[c].dropna(), bins=bins)
    ax.set_title(c, fontsize=10)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

for ax in axes[n:]:
    ax.set_visible(False)

fig.suptitle("Distributions of *_MISSING_PCT columns", fontsize=12)
fig.supxlabel("value")
fig.supylabel("count")
plt.show()
```

<img width="1011" height="1031" alt="image" src="https://github.com/user-attachments/assets/28ba9cd0-e8e7-4a2a-a270-5501117cca3e" />

Pretty good.

Let's check the score distributions:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

df = pd.read_csv('../../arrays.sscore', sep='\t')
cols = df.filter(regex=r'_AVG$').columns.tolist()

n = len(cols); ncols = 2; nrows = -(-n // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.2*nrows), constrained_layout=True)
axes = np.ravel(axes)

for i, c in enumerate(cols):
    ax = axes[i]
    x = df[c].to_numpy()
    x = x[np.isfinite(x)]
    m, M = x.min(), x.max()
    span = np.maximum(M - m, 1e-12)
    ax.hist(x, bins=np.linspace(m, m + span, 51))
    ax.axvline(np.median(x), linestyle='--', linewidth=1)
    ax.set_title(c, fontsize=10)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.tick_params(axis='x', labelsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

for ax in axes[n:]:
    ax.set_visible(False)

fig.supxlabel("value")
fig.supylabel("count")
fig.suptitle("Histograms of *_AVG columns", fontsize=12)
plt.show()
```

<img width="1011" height="971" alt="image" src="https://github.com/user-attachments/assets/8dff839d-a9c1-4494-9e14-b8644b810c57" />

We can see the scores with no overlap in the region are simply all zero. We need one extra score for Alzheimer's disease and two extra scores for breast cancer.

For breast cancer, we can try:
- PGS000335
- PGS000015
- PGS000508
- PGS000344
- PGS000007
- PGS000507
- PGS004869

For Alzheimer's disease, we can try:
- PGS004146
- PGS003957
- PGS003334
- PGS004589
- PGS004898
- PGS004229

These extra scores should complete in under 10 seconds:
```
./gnomon/target/release/gnomon score "PGS004146 | chr17:45535159-46342045, PGS004898 | chr17:45535159-46342045, PGS000007 | chr17:45535159-46342045, PGS000317 | chr17:45535159-46342045, PGS005198 | chr17:45535159-46342045, PGS004378 | chr17:45535159-46342045, PGS003957 | chr17:45535159-46342045, PGS003334 | chr17:45535159-46342045, PGS004589 | chr17:45535159-46342045, PGS004229 | chr17:45535159-46342045, PGS000335 | chr17:45535159-46342045, PGS000015 | chr17:45535159-46342045, PGS000508 | chr17:45535159-46342045, PGS000344 | chr17:45535159-46342045, PGS000507 | chr17:45535159-46342045, PGS004869 | chr17:45535159-46342045" arrays
```

<img width="1011" height="2731" alt="image" src="https://github.com/user-attachments/assets/e418ce20-45e2-424c-9e11-9da3d54b5df3" />

Low missingness.

Let's check which have variance:

<img width="1011" height="2571" alt="image" src="https://github.com/user-attachments/assets/529d3046-fef4-4920-90e8-c583942ee75d" />

Our options are: PGS000015, PGS000335, PGS000507, PGS000508, PGS004146, PGS004229, PGS004378, PGS004869, and PGS005198.

Obesity: PGS004378, PGS005198

Alzheimer’s: PGS004146, PGS004229

Breast Cancer: PGS000015, PGS000335, PGS000507, PGS000508, PGS004869

For breast cancer, we'll choose these two (best AUROC of the full score):
- PGS004869
- PGS000507
