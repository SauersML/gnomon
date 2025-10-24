Start RStudio first.

This will assume you have a file ending in .sscore already. You might want to upload it e.g.:
```
import os, subprocess

bucket = os.getenv("WORKSPACE_BUCKET")
src = "arrays.sscore"
subprocess.run(["gsutil", "-m", "cp", src, f"{bucket}/arrays.sscore"], check=True)
```

Or download:
```
bucket <- Sys.getenv("WORKSPACE_BUCKET")
remote <- sprintf("%s/arrays.sscore", bucket)
local  <- "arrays.sscore"
system2("gsutil", c("-m", "cp", remote, local))
```

We want to compare a few models.

1.a. Prediction from linear PCs alone
1.b. Prediction from spline PCs alone
2.a  Prediction from linear scores alone
2.b  Prediction from spline scores alone
3.a  Predicting accuracy with linear PC
3.b  Predicting accuracy with PC splines
4.a  Predicting with linear scores and linear PCs
4.b  Predicting with spline scores and spline PCs

