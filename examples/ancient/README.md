Run: `python3 create_scorefiles.py`

Run:
```
# Run negative selection score
./gnomon/target/release/gnomon score \
  ancient_hg38_scores/ancient_dna_negative_selection_hg38.txt \
  arrays

# Run positive selection score
./gnomon/target/release/gnomon score \
  ancient_hg38_scores/ancient_dna_positive_selection_hg38.txt \
  arrays
```
