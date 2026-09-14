//! One cohort as a PLINK 1 .bed, a .pgen and a .vcf.gz gives one sex table, whatever
//! the .psam or .fam records as a sample's sex. A sex check must never read the label
//! it infers, and PGEN must decode like the .bed plink2 writes from it.

use super::sex::infer_sex_to_tsv_at;
use infer_sex::GenomeBuild;
use std::fs;
use std::path::{Path, PathBuf};

fn fixture(extension: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data/testdata")
        .join(format!("xy_sex.{extension}"))
}

/// The label files with sample `iid` recorded as male: the SEX column of a .psam,
/// found by name, or the fifth column of a .fam.
fn relabel_male(extension: &str, text: &str, iid: &str) -> String {
    let mut lines = text.lines();
    let mut out = String::new();
    let (iid_column, sex_column) = if extension == "psam" {
        let header = lines.next().expect("psam header");
        out.push_str(header);
        out.push('\n');
        let columns: Vec<&str> = header.split('\t').collect();
        let find = |names: &[&str]| {
            columns
                .iter()
                .position(|column| names.contains(column))
                .expect("psam column")
        };
        (find(&["#IID", "IID"]), find(&["SEX"]))
    } else {
        (1, 4)
    };
    for line in lines {
        let mut fields: Vec<&str> = line.split_whitespace().collect();
        if fields.get(iid_column) == Some(&iid) {
            fields[sex_column] = "1";
        }
        out.push_str(&fields.join("\t"));
        out.push('\n');
    }
    out
}

#[test]
fn sex_table_is_the_same_for_every_format_and_any_recorded_sex() {
    let psam = fs::read_to_string(fixture("psam")).unwrap();
    let female = relabel_female_candidate(&psam);
    let dir = tempfile::tempdir().unwrap();
    let infer = |name: &str, extensions: &[&str], relabel: bool| -> String {
        let stage = dir.path().join(name);
        fs::create_dir_all(&stage).unwrap();
        for &extension in extensions {
            let target = stage.join(format!("in.{extension}"));
            if relabel && matches!(extension, "psam" | "fam") {
                let text = fs::read_to_string(fixture(extension)).unwrap();
                fs::write(&target, relabel_male(extension, &text, &female)).unwrap();
            } else {
                fs::copy(fixture(extension), &target).unwrap();
            }
        }
        let output = stage.join("sex.tsv");
        infer_sex_to_tsv_at(
            &stage.join(format!("in.{}", extensions[0])),
            Some(GenomeBuild::Build38),
            &output,
        )
        .unwrap_or_else(|err| panic!("{name}: {err}"));
        fs::read_to_string(output).unwrap()
    };

    let bed = infer("bed", &["bed", "bim", "fam"], false);
    assert_eq!(bed.lines().count(), 9, "a header and eight samples");
    assert_eq!(infer("pgen", &["pgen", "pvar", "psam"], false), bed, "pgen");
    assert_eq!(infer("vcf", &["vcf.gz"], false), bed, "vcf.gz");
    assert_eq!(
        infer("pgen_relabeled", &["pgen", "pvar", "psam"], true),
        bed,
        "pgen with {female} recorded as male"
    );
    assert_eq!(
        infer("bed_relabeled", &["bed", "bim", "fam"], true),
        bed,
        "bed with {female} recorded as male"
    );
}

/// The first sample the .psam records as female, whose X heterozygous calls a
/// label-driven haploid rule would erase.
fn relabel_female_candidate(psam: &str) -> String {
    let mut lines = psam.lines();
    let columns: Vec<&str> = lines.next().expect("psam header").split('\t').collect();
    let iid = columns
        .iter()
        .position(|column| *column == "#IID" || *column == "IID")
        .expect("IID column");
    let sex = columns
        .iter()
        .position(|column| *column == "SEX")
        .expect("SEX column");
    lines
        .map(|line| line.split('\t').collect::<Vec<_>>())
        .find(|fields| fields[sex] == "2")
        .map(|fields| fields[iid].to_string())
        .expect("a female sample")
}
