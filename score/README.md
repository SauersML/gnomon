#### Example biobank run:
Get gnomon ready:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && { for f in ~/.bashrc ~/.profile; do [ -f "$f" ] || touch "$f"; grep -qxF 'source "$HOME/.cargo/env"' "$f" || printf '\n# Rust / Cargo\nsource "$HOME/.cargo/env"\n' >> "$f"; done; } && source "$HOME/.cargo/env" && git clone https://github.com/SauersML/gnomon.git && cd gnomon && rustup override set nightly && cargo build --release && cd ~
```

Or to update gnomon:
```
cd ~/gnomon && git remote set-url origin https://github.com/SauersML/gnomon.git && git fetch --prune origin && br=$(git remote show origin | sed -n 's/.*HEAD branch: //p') && git checkout "$br" && git reset --hard "origin/$br" && git submodule update --init --recursive && cargo build --release && cd ~
```

Download data:
```
gsutil -u "$GOOGLE_PROJECT" -m cp -r gs://fc-aou-datasets-controlled/v8/microarray/plink/* .
```

Run a local score:
```
./gnomon/target/release/gnomon score "score.txt" arrays
```

Or use a score from PGS catalog:
```
./gnomon/target/release/gnomon score "PGS003725" arrays
```

Or stream the data:
```
./gnomon/target/release/gnomon score "PGS003725" gs://fc-aou-datasets-controlled/v8/microarray/plink/*
```

Optionally, use a keep file:
```
awk '{print $2}' arrays.fam | head -n 256 > keep.txt
```

```
./gnomon/target/release/gnomon score "PGS003725" arrays --keep keep.txt
```

#### Choosing where outputs go

By default the results land beside the genotypes as `<genotypes>_<score stem>.sscore`
(`arrays_score.sscore` for the first run above), or in the working directory for
`gs://` and `https://` inputs. The converted and sorted score-file caches land beside
the score file. gnomon refuses to overwrite an existing `.sscore`.

`--out PREFIX` writes the results to `PREFIX.sscore` instead. Everything else the run
keeps goes under PREFIX's directory: the resume checkpoint
(`PREFIX.sscore.gnomon-checkpoint.bin`, removed once the run succeeds), downloaded PGS
Catalog files, and the score-file caches (`gnomon_score_cache/`).
```
./gnomon/target/release/gnomon score "PGS003725" arrays --out results/arrays_pgs003725
```

Nothing is written beside a PLINK or VCF input or beside the score files, so the inputs
may sit in a read-only or shared directory. Concurrent runs on the same inputs are safe
when their prefixes differ. Two genotype-side intermediates still go beside the
genotypes:
- the PLINK conversion cache for BCF, DTC and `--panel` inputs (`<stem>.gnomon_cache/`)
- the re-sorted fileset written when a `.bim` is out of order
  (`<prefix>.sorted.{bed,bim,fam}`)

Each score-file cache entry under `gnomon_score_cache/` is named with a key over the
gnomon build and the score file's path, size and modification time, so an edited score
file is converted again rather than served from a stale entry. Without `--out`, a score
file in a directory gnomon cannot write is cached under the user cache directory
instead: `$XDG_CACHE_HOME/gnomon/score_cache`, which defaults to
`~/.cache/gnomon/score_cache` on Linux.

Results and checkpoints are written to a temporary file in their destination directory
and renamed into place once complete, so another process never reads a partial
`.sscore`.

#### Example:
```
./target/release/gnomon score ./ci_workdir/PGS004696_hmPOS_GRCh38.txt ./ci_workdir/gnomon_native_data
```

#### Debug:
```
./target/debug/gnomon score ./ci_workdir/PGS004696_hmPOS_GRCh38.txt ./ci_workdir/gnomon_native_data
```
