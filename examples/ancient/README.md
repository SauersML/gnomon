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

Analyze scores:
```
python -c "import pandas as pd; df = pd.read_csv('arrays_ancient_hg38_scores.sscore', sep='\s+'); scores = df[['#IID', 'ancient_hg38_scores_ancient_dna_positive_selection_hg38_AVG', 'ancient_hg38_scores_ancient_dna_negative_selection_hg38_AVG']].copy(); scores.columns = ['SampleID', 'positive_score', 'negative_score']; scores[['positive_score', 'negative_score']] = (scores[['positive_score', 'negative_score']] - scores[['positive_score', 'negative_score']].mean()) / scores[['positive_score', 'negative_score']].std(); scores.to_csv('ancient_scores_znormed.tsv', sep='\t', index=False); print('Z-normed scores written to ancient_scores_znormed.tsv')"
```
