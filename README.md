# gnomon

Wiktionary gives this etymology:
>Borrowed from French gnomon, or directly from its etymon Latin gnōmōn, or directly from its etymon Ancient Greek γνώμων (gnṓmōn, “discerner, interpreter; carpenter’s square; gnomon of a sundial; (geometry) gnomon”), from γιγνώσκω (gignṓskō, “to be aware of; to perceive; to know”), ultimately from Proto-Indo-European *ǵneh₃- (“to know”); the word is thus related to know.

The word "gnomon" shares a root with the Ancient Greek γνώμη (gnṓmē), meaning means of knowing or judgement, gnôma, meaning "sign" or "symptom," the Finnish word "kone," meaning "machine," "kunją," meaning "omen," Sanskrit ज्ञा (jñā), meaning "to know," Jñāna (knowledge, in Indian philosophy), and the English word "know."

## Overview

Gnomon is a high-performance Rust engine for computing and calibrating polygenic scores at biobank scale. It combines streaming genotype processing with penalized generalized additive models to produce calibrated risk predictions that account for population structure and sex-specific effects.

## Architecture

- **[`cli/`](cli/)** – Run polygenic score calculations, fit ancestry models, and train calibration models from the command line. See [`cli/README.md`](cli/README.md) for usage.
- **[`score/`](score/)** – Calculate raw polygenic scores for individuals from genotype data and published score files. See [`score/README.md`](score/README.md) for examples.
- **[`map/`](map/)** – Infer genetic ancestry by fitting and projecting samples onto principal components that capture population structure. See [`map/README.md`](map/README.md) for details.
- **[`calibrate/`](calibrate/)** – Transform raw polygenic scores into calibrated risk predictions that account for ancestry and sex. See [`calibrate/README.md`](calibrate/README.md) for statistical model and implementation.
- **[`examples/`](examples/)** – Reproduce published polygenic score analyses and validate calibration performance.

## Quick Start

```bash
# Install Rust nightly
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build gnomon
git clone https://github.com/SauersML/gnomon.git
cd gnomon
rustup override set nightly
cargo build --release

# Compute a polygenic score
./target/release/gnomon score PGS003725 path/to/genotypes

# Fit a PCA model
./target/release/gnomon fit path/to/genotypes --components 10

# Train a calibration model
./target/release/gnomon train training_data.tsv --num-pcs 10

# Apply calibration to new samples
./target/release/gnomon infer test_data.tsv --model model.toml
```

Each subcommand writes outputs to the current directory or alongside the input data. Run `gnomon --help` or `gnomon <subcommand> --help` for detailed options.
