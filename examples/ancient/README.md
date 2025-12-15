Run: `python3 create_scorefiles.py`

Move the hg38 scorefiles to a directory:
```
mkdir -p ancient_hg38_scores
cp gnomon/examples/ancient/ancient_dna_*_selection_hg38.txt ancient_hg38_scores/
```

Run:
```
./gnomon/target/release/gnomon score ancient_hg38_scores arrays
```
