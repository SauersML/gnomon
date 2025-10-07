#### Example biobank run:
Get gnomon ready:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && { for f in ~/.bashrc ~/.profile; do [ -f "$f" ] || touch "$f"; grep -qxF 'source "$HOME/.cargo/env"' "$f" || printf '\n# Rust / Cargo\nsource "$HOME/.cargo/env"\n' >> "$f"; done; } && source "$HOME/.cargo/env" && git clone https://github.com/SauersML/gnomon.git && cd gnomon && rustup override set nightly && cargo build --release && cd ~
```

Download data:
```
gsutil -u "$GOOGLE_PROJECT" -m cp -r gs://fc-aou-datasets-controlled/v8/microarray/plink/* .
```

Run a local score:
```
./gnomon/target/release/gnomon score --score "score.txt" arrays
```

Or use a score from PGS catalog:
```
./gnomon/target/release/gnomon score --score "PGS003725" arrays
```

Optionally, use a keep file:
```
awk '{print $2}' arrays.fam | head -n 256 > keep.txt
```

```
./gnomon/target/release/gnomon score --score "PGS003725" --keep keep.txt arrays
```

#### Example:
```
./target/release/gnomon score --score ./ci_workdir/PGS004696_hmPOS_GRCh38.txt ./ci_workdir/gnomon_native_data
```

#### Debug:
```
./target/debug/gnomon --score ./ci_workdir/PGS004696_hmPOS_GRCh38.txt ./ci_workdir/gnomon_native_data
```
