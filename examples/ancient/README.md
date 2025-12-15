Run: `python3 create_scorefiles.py`

Move the hg38 scorefiles to a directory:
```
mkdir -p ancient_hg38_scores
cp gnomon/examples/ancient/ancient_dna_*_selection_hg38.txt ancient_hg38_scores/
```

Get gnomon:
```
curl -fsSL https://raw.githubusercontent.com/SauersML/gnomon/main/install.sh | bash
```

Run:
```
gnomon score ancient_hg38_scores arrays
```
