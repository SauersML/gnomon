use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct VariantKey {
    pub chromosome: String,
    pub position: u64,
}

impl VariantKey {
    pub fn new(chromosome: &str, position: u64) -> Self {
        let chromosome = normalize_chromosome(chromosome);
        Self {
            chromosome,
            position,
        }
    }
}

fn normalize_chromosome(chromosome: &str) -> String {
    let mut normalized = chromosome.trim().to_string();
    if normalized.len() >= 3 {
        let prefix = &normalized[..3];
        if prefix.eq_ignore_ascii_case("chr") {
            normalized = normalized[3..].to_string();
        }
    }
    normalized.to_ascii_uppercase()
}

#[derive(Debug)]
pub enum VariantListError {
    Io(io::Error),
    Parse { line: usize, message: String },
}

impl From<io::Error> for VariantListError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

#[derive(Clone, Debug)]
pub struct VariantFilter {
    unique: HashSet<VariantKey>,
}

impl VariantFilter {
    pub fn from_file(path: &Path) -> Result<Self, VariantListError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut unique = HashSet::new();
        let mut header_skipped = false;

        for (idx, line) in reader.lines().enumerate() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let mut fields = trimmed.split_whitespace();
            let chrom = match fields.next() {
                Some(value) => value,
                None => continue,
            };
            let pos_text = match fields.next() {
                Some(value) => value,
                None => {
                    return Err(VariantListError::Parse {
                        line: idx + 1,
                        message: "expected a position column".into(),
                    });
                }
            };

            if !header_skipped
                && (chrom.eq_ignore_ascii_case("chrom")
                    || chrom.eq_ignore_ascii_case("chr")
                    || pos_text.eq_ignore_ascii_case("pos"))
            {
                header_skipped = true;
                continue;
            }

            let position: u64 = match pos_text.parse() {
                Ok(value) if value > 0 => value,
                Ok(_) => {
                    return Err(VariantListError::Parse {
                        line: idx + 1,
                        message: "position must be positive".into(),
                    });
                }
                Err(_) if !header_skipped => {
                    header_skipped = true;
                    continue;
                }
                Err(_) => {
                    return Err(VariantListError::Parse {
                        line: idx + 1,
                        message: format!("invalid position value: {pos_text}"),
                    });
                }
            };

            let key = VariantKey::new(chrom, position);
            unique.insert(key);
        }

        if unique.is_empty() {
            return Err(VariantListError::Parse {
                line: 0,
                message: "variant list did not contain any usable records".into(),
            });
        }

        Ok(Self { unique })
    }

    pub fn from_keys(keys: impl IntoIterator<Item = VariantKey>) -> Self {
        let mut unique = HashSet::new();
        for key in keys {
            unique.insert(key);
        }
        Self { unique }
    }

    pub fn contains(&self, key: &VariantKey) -> bool {
        self.unique.contains(key)
    }

    pub fn requested_unique(&self) -> usize {
        self.unique.len()
    }

    pub fn missing_keys(&self, matched: &HashSet<VariantKey>) -> Vec<VariantKey> {
        self.unique
            .iter()
            .filter(|key| !matched.contains(*key))
            .cloned()
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct VariantSelection {
    pub indices: Vec<usize>,
    pub keys: Vec<VariantKey>,
    pub missing: Vec<VariantKey>,
    pub requested_unique: usize,
}

impl VariantSelection {
    pub fn matched_unique(&self) -> usize {
        self.keys.len()
    }
}
