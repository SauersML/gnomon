Create a workspace and then open a terminal with a fast disk since we will be doing i/o.

Download microarray data:
```
gsutil -u "$GOOGLE_PROJECT" -m cp -r gs://fc-aou-datasets-controlled/v8/microarray/plink/* .
```

