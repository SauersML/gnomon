//! One cohort as a PLINK 1 .bed, a .pgen, a .vcf.gz and a .bcf gives one sex table,
//! whatever the .psam or .fam records as a sample's sex. A sex check must never read
//! the label it infers, and every format must decode like the .bed plink2 writes from
//! it. The fixture's samples are 1000 Genomes males and females, so each table is
//! also held to their recorded sex.

use super::sex::infer_sex_to_tsv_at;
use infer_sex::GenomeBuild;
use noodles_vcf::variant::RecordBuf;
use noodles_vcf::variant::io::Write as _;
use noodles_vcf::variant::record_buf::samples::{Keys, Samples, sample::Value};
use std::fs;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

fn fixture(extension: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data/testdata")
        .join(format!("xy_sex.{extension}"))
}

/// The sex 1000 Genomes records for the fixture's samples, in file order.
const RECORDED_SEX: [(&str, &str); 8] = [
    ("HG00096", "male"),
    ("HG00101", "male"),
    ("HG00103", "male"),
    ("HG00105", "male"),
    ("HG00097", "female"),
    ("HG00099", "female"),
    ("HG00100", "female"),
    ("HG00102", "female"),
];

/// The fixture's X and Y rows, all outside the PAR.
const X_ROWS: usize = 2039;
const Y_ROWS: usize = 844;

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

/// Infers the sex table of the input at `stage/in.<extension>` as GRCh38, the
/// fixture's build.
fn infer(stage: &Path, extension: &str) -> String {
    infer_as(stage, extension, Some(GenomeBuild::Build38))
}

/// [`infer`] with `build` forced, or inferred when `None`.
fn infer_as(stage: &Path, extension: &str, build: Option<GenomeBuild>) -> String {
    let output = stage.join(format!("sex.{extension}.{build:?}.tsv"));
    infer_sex_to_tsv_at(&stage.join(format!("in.{extension}")), build, &output)
        .unwrap_or_else(|err| panic!("{}: {err}", stage.display()));
    fs::read_to_string(output).unwrap()
}

/// One column of a sex table, by header name.
fn column(table: &str, name: &str) -> Vec<String> {
    let mut lines = table.lines();
    let index = lines
        .next()
        .expect("header")
        .split('\t')
        .position(|header| header == name)
        .unwrap_or_else(|| panic!("no {name} column"));
    lines
        .map(|line| line.split('\t').nth(index).expect("field").to_string())
        .collect()
}

/// Every sample called as 1000 Genomes records it.
fn assert_recorded_sex(table: &str, context: &str) {
    let calls: Vec<(String, String)> = column(table, "IID")
        .into_iter()
        .zip(column(table, "Sex"))
        .collect();
    let recorded: Vec<(String, String)> = RECORDED_SEX
        .iter()
        .map(|&(iid, sex)| (iid.to_string(), sex.to_string()))
        .collect();
    assert_eq!(calls, recorded, "{context}");
}

fn fixture_vcf_text() -> String {
    let mut text = String::new();
    flate2::read::MultiGzDecoder::new(fs::File::open(fixture("vcf.gz")).unwrap())
        .read_to_string(&mut text)
        .unwrap();
    text
}

/// The fixture's VCF as its header lines, the column header last, and its
/// records split into tab-separated fields.
fn fixture_vcf() -> (Vec<String>, Vec<Vec<String>>) {
    let (mut header, mut records) = (Vec::new(), Vec::new());
    for line in fixture_vcf_text().lines() {
        if line.starts_with('#') {
            header.push(line.to_string());
        } else {
            records.push(line.split('\t').map(str::to_string).collect());
        }
    }
    (header, records)
}

fn vcf_text(header: &[String], records: &[Vec<String>]) -> String {
    let mut out = String::new();
    for line in header {
        out.push_str(line);
        out.push('\n');
    }
    for fields in records {
        out.push_str(&fields.join("\t"));
        out.push('\n');
    }
    out
}

/// The fixture's VCF with `edit` applied to the fields of every record, dropping
/// the records it returns false for.
fn edit_vcf(edit: impl FnMut(&mut Vec<String>) -> bool) -> String {
    let (header, mut records) = fixture_vcf();
    records.retain_mut(edit);
    vcf_text(&header, &records)
}

/// Writes `vcf` as `stage/in.vcf`, and as the BCF and PLINK 1 fileset plink2
/// would write from it: `in.bcf`, and `in.bed` with A1 the ALT allele, a haploid
/// call as the homozygous one, and no recorded sex in `in.fam`.
fn stage_vcf(stage: &Path, vcf: &str) {
    fs::create_dir_all(stage).unwrap();
    let vcf_path = stage.join("in.vcf");
    fs::write(&vcf_path, vcf).unwrap();

    let mut reader =
        noodles_vcf::io::Reader::new(BufReader::new(fs::File::open(&vcf_path).unwrap()));
    let header = reader.read_header().unwrap();
    let mut writer = noodles_bcf::io::Writer::new(fs::File::create(stage.join("in.bcf")).unwrap());
    writer.write_header(&header).unwrap();
    let mut record = RecordBuf::default();
    while reader.read_record_buf(&header, &mut record).unwrap() != 0 {
        // noodles reads a missing sample as no values, and its BCF writer refuses
        // a missing call and a field that every sample is missing. A missing call
        // is written as the string ".", as bcftools writes a missing one-allele
        // call, and a field no sample has is dropped.
        let samples = record.samples();
        let keys: Vec<String> = samples.keys().as_ref().iter().cloned().collect();
        let values: Vec<Vec<Option<Value>>> = samples
            .values()
            .map(|sample| {
                let mut values = sample.values().to_vec();
                values.resize(keys.len(), None);
                values
            })
            .collect();
        let kept: Vec<usize> = (0..keys.len())
            .filter(|&index| {
                keys[index] == "GT" || values.iter().any(|sample| sample[index].is_some())
            })
            .collect();
        let values = values
            .iter()
            .map(|sample| {
                kept.iter()
                    .map(|&index| match &sample[index] {
                        None if keys[index] == "GT" => Some(Value::String(".".to_string())),
                        value => value.clone(),
                    })
                    .collect()
            })
            .collect();
        let keys: Keys = kept.iter().map(|&index| keys[index].clone()).collect();
        *record.samples_mut() = Samples::new(keys, values);
        writer.write_variant_record(&header, &record).unwrap();
    }
    writer.try_finish().unwrap();

    let mut lines = vcf.lines().filter(|line| !line.starts_with("##"));
    let samples: Vec<&str> = lines
        .next()
        .expect("column header")
        .split('\t')
        .skip(9)
        .collect();
    let fam: String = samples
        .iter()
        .map(|iid| format!("0\t{iid}\t0\t0\t0\t-9\n"))
        .collect();
    fs::write(stage.join("in.fam"), fam).unwrap();
    let mut bim = String::new();
    let mut bed = vec![0x6c, 0x1b, 0x01];
    for line in lines {
        let fields: Vec<&str> = line.split('\t').collect();
        assert!(!fields[4].contains(','), "the fixture is biallelic");
        bim.push_str(&format!(
            "{}\t{}\t0\t{}\t{}\t{}\n",
            fields[0], fields[2], fields[1], fields[4], fields[3]
        ));
        let mut row = vec![0u8; samples.len().div_ceil(4)];
        for (slot, sample) in fields[9..].iter().enumerate() {
            let gt = sample.split(':').next().unwrap();
            let alleles: Vec<&str> = gt.split(['/', '|']).collect();
            let code = if alleles.contains(&".") {
                0b01
            } else if alleles.iter().all(|&allele| allele == "1") {
                0b00
            } else if alleles.iter().all(|&allele| allele == "0") {
                0b11
            } else {
                0b10
            };
            row[slot / 4] |= code << (2 * (slot % 4));
        }
        bed.extend_from_slice(&row);
    }
    fs::write(stage.join("in.bim"), bim).unwrap();
    fs::write(stage.join("in.bed"), bed).unwrap();
}

/// The sex tables of a staged panel's VCF, BCF and .bed, which must agree.
fn staged_table(stage: &Path) -> String {
    let bed = infer(stage, "bed");
    assert_eq!(infer(stage, "vcf"), bed, "{}: vcf", stage.display());
    assert_eq!(infer(stage, "bcf"), bed, "{}: bcf", stage.display());
    bed
}

/// Copies the fixture's `extensions` to `stage/in.<extension>`, passing each
/// through `edit`.
fn stage_fixture(stage: &Path, extensions: &[&str], edit: impl Fn(&str, Vec<u8>) -> Vec<u8>) {
    fs::create_dir_all(stage).unwrap();
    for &extension in extensions {
        let bytes = fs::read(fixture(extension)).unwrap();
        fs::write(
            stage.join(format!("in.{extension}")),
            edit(extension, bytes),
        )
        .unwrap();
    }
}

#[test]
fn sex_table_is_the_same_for_every_format_and_any_recorded_sex() {
    let psam = fs::read_to_string(fixture("psam")).unwrap();
    let female = relabel_female_candidate(&psam);
    let dir = tempfile::tempdir().unwrap();
    let table = |name: &str, extensions: &[&str], relabel: bool| -> String {
        let stage = dir.path().join(name);
        stage_fixture(&stage, extensions, |extension, bytes| {
            if relabel && matches!(extension, "psam" | "fam") {
                let text = String::from_utf8(bytes).unwrap();
                relabel_male(extension, &text, &female).into_bytes()
            } else {
                bytes
            }
        });
        infer(&stage, extensions[0])
    };

    let bed = table("bed", &["bed", "bim", "fam"], false);
    assert_eq!(bed.lines().count(), 9, "a header and eight samples");
    assert_recorded_sex(&bed, "bed");
    assert_eq!(table("pgen", &["pgen", "pvar", "psam"], false), bed, "pgen");
    assert_eq!(table("vcf", &["vcf.gz"], false), bed, "vcf.gz");
    assert_eq!(
        table("pgen_relabeled", &["pgen", "pvar", "psam"], true),
        bed,
        "pgen with {female} recorded as male"
    );
    assert_eq!(
        table("bed_relabeled", &["bed", "bim", "fam"], true),
        bed,
        "bed with {female} recorded as male"
    );

    // The same cohort through the in-test writers, and as a BCF.
    let stage = dir.path().join("written");
    stage_vcf(&stage, &fixture_vcf_text());
    assert_eq!(staged_table(&stage), bed, "written from the VCF");
}

/// The fixture's males carry only haploid X calls, 232 of them ALT. A haploid
/// ALT call is the homozygous call plink2 imports, never a heterozygous one, so
/// every male shows no X heterozygosity in any format, and every female some.
#[test]
fn a_haploid_alt_call_is_never_heterozygous() {
    let vcf = fixture_vcf_text();
    let male_x_calls: Vec<&str> = vcf
        .lines()
        .filter(|line| line.starts_with("X\t"))
        .flat_map(|line| line.split('\t').skip(9).take(4))
        .collect();
    assert_eq!(male_x_calls.len(), 4 * X_ROWS);
    assert!(male_x_calls.iter().all(|call| matches!(*call, "0" | "1")));
    assert!(male_x_calls.contains(&"1"), "haploid ALT calls");

    let dir = tempfile::tempdir().unwrap();
    let stage = dir.path().join("haploid");
    stage_vcf(&stage, &vcf);
    let table = staged_table(&stage);
    assert_recorded_sex(&table, "haploid males");
    let het = column(&table, "X_NonPAR_Het");
    assert!(het[..4].iter().all(|count| count == "0"), "{het:?}");
    assert!(het[4..].iter().all(|count| count != "0"), "{het:?}");
}

/// Without a Y non-PAR locus the Y density is missing, which infer_sex reads as
/// zero, calling every sample female. The calls must come from X instead.
#[test]
fn a_panel_without_y_is_called_on_x_in_every_format() {
    let dir = tempfile::tempdir().unwrap();
    let stage = dir.path().join("no_y");
    stage_vcf(&stage, &edit_vcf(|fields| fields[0] != "Y"));
    let table = staged_table(&stage);
    assert_recorded_sex(&table, "no Y");
    assert!(
        column(&table, "Y_NonPAR_Valid")
            .iter()
            .all(|count| count == "0")
    );
    assert!(
        column(&table, "Y_Density")
            .iter()
            .all(|density| density == "NA")
    );

    // A .pgen cannot drop rows here, so its Y rows are relabelled as MT, which
    // sex inference does not read.
    let pgen = dir.path().join("no_y_pgen");
    stage_fixture(&pgen, &["pgen", "pvar", "psam"], |extension, bytes| {
        if extension != "pvar" {
            return bytes;
        }
        let text = String::from_utf8(bytes).unwrap();
        let mut out = String::new();
        for line in text.lines() {
            match line.strip_prefix("Y\t") {
                Some(rest) => out.push_str(&format!("MT\t{rest}\n")),
                None => out.push_str(&format!("{line}\n")),
            }
        }
        out.into_bytes()
    });
    assert_eq!(infer(&pgen, "pgen"), table, "pgen without Y");
}

/// The fixture's males miss 18 to 28 of the 844 Y loci and its females every
/// one. A male who calls every locus is no less male: his Y density is the
/// highest a sample can have, and the females beside him still missed chrY.
/// Alone, he is a panel without Y dropout, and his call comes from his X.
#[test]
fn a_male_who_calls_every_locus_is_called_male_in_every_format() {
    let fill_male_calls = |fields: &mut Vec<String>| {
        for call in &mut fields[9..13] {
            if call == "." {
                *call = "0".to_string();
            }
        }
        true
    };
    let dir = tempfile::tempdir().unwrap();
    let stage = dir.path().join("complete_males");
    stage_vcf(&stage, &edit_vcf(fill_male_calls));
    let table = staged_table(&stage);
    assert_recorded_sex(&table, "complete males");
    let y_valid = column(&table, "Y_NonPAR_Valid");
    assert!(
        y_valid[..4]
            .iter()
            .all(|count| *count == Y_ROWS.to_string()),
        "{y_valid:?}"
    );

    let (mut header, mut records) = fixture_vcf();
    let columns = header.last_mut().expect("column header");
    *columns = columns.split('\t').take(10).collect::<Vec<_>>().join("\t");
    for fields in &mut records {
        fill_male_calls(fields);
        fields.truncate(10);
    }
    let alone = dir.path().join("one_complete_male");
    stage_vcf(&alone, &vcf_text(&header, &records));
    let table = staged_table(&alone);
    assert_eq!(column(&table, "IID"), [RECORDED_SEX[0].0]);
    assert_eq!(column(&table, "Sex"), ["male"]);
    assert_eq!(column(&table, "Y_NonPAR_Valid"), [Y_ROWS.to_string()]);
}

/// A GRCh38 array whose last X probe lies between 154.9 and 155.7 Mb was read as
/// GRCh37 from that probe alone. A probe past the end of GRCh37's chrX
/// (155,270,560) proves GRCh38, under which the tail of PAR1, 2,699,521 to
/// 2,781,479, is pseudoautosomal: a male's diploid calls there are not X
/// heterozygosity, as they would be under GRCh37.
#[test]
fn a_grch38_panel_whose_last_x_probe_is_under_155_7_mb_is_read_as_grch38() {
    const PAR1_TAIL_ROWS: u64 = 40;
    let (header, records) = fixture_vcf();
    let x_row = |position: u64, male: &str, female: &str| -> Vec<String> {
        let mut fields = records
            .iter()
            .find(|fields| fields[0] == "X")
            .expect("an X row")
            .clone();
        fields[1] = position.to_string();
        fields[2] = format!("X:{position}");
        fields[7] = ".".to_string();
        for (slot, call) in fields[9..].iter_mut().enumerate() {
            *call = if slot < 4 { male } else { female }.to_string();
        }
        fields
    };
    let mut panel = Vec::new();
    for fields in &records {
        if fields[0] == "X"
            && panel
                .last()
                .is_some_and(|last: &Vec<String>| last[0] != "X")
        {
            panel.extend((0..PAR1_TAIL_ROWS).map(|i| x_row(2_700_000 + 2_000 * i, "0|1", "0|1")));
        }
        if fields[0] == "Y"
            && panel
                .last()
                .is_some_and(|last: &Vec<String>| last[0] == "X")
        {
            panel.push(x_row(155_300_000, "0", "0|0"));
        }
        panel.push(fields.clone());
    }
    assert_eq!(panel.len(), records.len() + PAR1_TAIL_ROWS as usize + 1);

    let dir = tempfile::tempdir().unwrap();
    let stage = dir.path().join("grch38_ending_at_155_3_mb");
    stage_vcf(&stage, &vcf_text(&header, &panel));
    let table = infer_as(&stage, "bed", None);
    assert_eq!(infer_as(&stage, "vcf", None), table, "vcf");
    assert_eq!(infer_as(&stage, "bcf", None), table, "bcf");
    assert!(
        column(&table, "Build")
            .iter()
            .all(|build| build == "Build38"),
        "{table}"
    );
    assert_recorded_sex(&table, "inferred build");
    let het = column(&table, "X_NonPAR_Het");
    assert!(het[..4].iter().all(|count| count == "0"), "{het:?}");

    let as_grch37 = infer_as(&stage, "bed", Some(GenomeBuild::Build37));
    let het = column(&as_grch37, "X_NonPAR_Het");
    assert!(
        het[..4]
            .iter()
            .all(|count| *count == PAR1_TAIL_ROWS.to_string()),
        "{het:?}"
    );
}

/// Female chrY calls that no genotyper should have made, such as no-calls filled
/// as reference, leave the panel without Y dropout. Every sample's Y density is
/// then the male one, and the calls must come from X.
#[test]
fn a_panel_where_every_sample_calls_chry_is_called_on_x_in_every_format() {
    let dir = tempfile::tempdir().unwrap();
    let stage = dir.path().join("filled_y");
    stage_vcf(
        &stage,
        &edit_vcf(|fields| {
            if fields[0] == "Y" {
                for call in &mut fields[9..] {
                    if call == "." {
                        *call = "0".to_string();
                    }
                }
            }
            true
        }),
    );
    let table = staged_table(&stage);
    assert_recorded_sex(&table, "filled Y");
    assert!(
        column(&table, "Y_NonPAR_Valid")
            .iter()
            .all(|count| *count == Y_ROWS.to_string())
    );
}

const DS_HEADER: &str =
    r#"##FORMAT=<ID=DS,Number=A,Type=Float,Description="Alternate allele dosage">"#;

/// The dosages an imputation server writes beside its calls: near them, and
/// for a heterozygote rarely exactly 1.
fn imputed_dosage(call: &str) -> &'static str {
    match call {
        "." => ".",
        "0" => "0.01",
        "1" => "0.98",
        "0|0" => "0.02",
        "0|1" | "1|0" => "0.97",
        "1|1" => "1.95",
        other => panic!("unexpected call {other}"),
    }
}

/// A dosage too far from every allele count to import as any call.
fn uncertain_dosage(call: &str) -> &'static str {
    if call == "." { "." } else { "0.5" }
}

/// Where a record has calls, a sex check counts them, as plink2's VCF import
/// reads them, whatever dosage sits beside them: a DS field must not change the
/// table, with or without chrY.
#[test]
fn a_dosage_beside_the_call_does_not_change_the_table() {
    let dir = tempfile::tempdir().unwrap();
    for keep_y in [true, false] {
        let calls_only = dir.path().join(format!("calls_{keep_y}"));
        stage_vcf(&calls_only, &edit_vcf(|fields| keep_y || fields[0] != "Y"));
        let expected = infer(&calls_only, "bed");
        assert_recorded_sex(&expected, "calls");

        let dosages: [(&str, fn(&str) -> &'static str); 2] =
            [("imputed", imputed_dosage), ("uncertain", uncertain_dosage)];
        for (name, dosage) in dosages {
            let (mut header, mut records) = fixture_vcf();
            header.insert(header.len() - 1, DS_HEADER.to_string());
            records.retain(|fields| keep_y || fields[0] != "Y");
            for fields in &mut records {
                fields[8] = "GT:DS".to_string();
                for call in &mut fields[9..] {
                    *call = format!("{call}:{}", dosage(call));
                }
            }
            let stage = dir.path().join(format!("{name}_{keep_y}"));
            stage_vcf(&stage, &vcf_text(&header, &records));
            let context = format!("{name} DS beside the calls, Y kept: {keep_y}");
            assert_eq!(infer(&stage, "vcf"), expected, "vcf, {context}");
            assert_eq!(infer(&stage, "bcf"), expected, "bcf, {context}");
        }
    }
}

/// A record without GT has only its dosages, which import as plink2 imports
/// them by default: each as the nearest allele count within 0.1, and otherwise
/// as a missing call. plink2 2.0.0-a.7.5 imports DS 0.05, 0.09, 0.91, 1.09 and
/// 1.91 as 0/0, 0/0, 0/1, 0/1 and 1/1, and 0.11, 0.5, 0.89, 1.11 and 1.89 as
/// missing. On X, Y and MT such a record cannot say whether a dosage is on the
/// 0..1 or the 0..2 scale, and plink2 refuses it; so does sex inference.
#[test]
fn a_dosage_without_a_call_imports_as_the_nearest_call() {
    let (mut header, records) = fixture_vcf();
    let uncertain = |row: usize, slot: usize| (row + slot) % 7 == 0;
    let dir = tempfile::tempdir().unwrap();

    // The calls, missing where the autosomal dosage below is uncertain.
    let mut calls = records.clone();
    for (row, fields) in calls.iter_mut().enumerate() {
        if fields[0] == "22" {
            for (slot, call) in fields[9..].iter_mut().enumerate() {
                if uncertain(row, slot) {
                    *call = ".".to_string();
                }
            }
        }
    }
    let calls_only = dir.path().join("calls");
    stage_vcf(&calls_only, &vcf_text(&header, &calls));
    let expected = infer(&calls_only, "bed");
    assert_recorded_sex(&expected, "calls");

    header.insert(header.len() - 1, DS_HEADER.to_string());
    let only_dosages = |chromosome: &str| -> String {
        let mut dosages = records.clone();
        for (row, fields) in dosages.iter_mut().enumerate() {
            if fields[0] != chromosome {
                continue;
            }
            fields[8] = "DS".to_string();
            for (slot, call) in fields[9..].iter_mut().enumerate() {
                let dosage = match call.as_str() {
                    "." => ".",
                    _ if uncertain(row, slot) => "0.5",
                    "0" | "0|0" => "0.08",
                    "0|1" | "1|0" => "1.09",
                    "1" | "1|1" => "1.91",
                    other => panic!("unexpected call {other}"),
                };
                *call = dosage.to_string();
            }
        }
        vcf_text(&header, &dosages)
    };

    let stage = dir.path().join("autosomal_dosages");
    stage_vcf(&stage, &only_dosages("22"));
    assert_eq!(infer(&stage, "vcf"), expected, "vcf");
    assert_eq!(infer(&stage, "bcf"), expected, "bcf");

    let stage = dir.path().join("x_dosages");
    stage_vcf(&stage, &only_dosages("X"));
    for extension in ["vcf", "bcf"] {
        let err = infer_sex_to_tsv_at(
            &stage.join(format!("in.{extension}")),
            Some(GenomeBuild::Build38),
            &stage.join(format!("sex.{extension}.tsv")),
        )
        .expect_err("X dosages without GT");
        assert!(err.to_string().contains("without GT"), "{extension}: {err}");
    }
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

/// A GRCh38 panel of GT-only calls in every shape a GT takes here: phased and
/// unphased diploid calls, `.` in either allele or both, haploid calls, and, with
/// `max_ploidy` 3, triploid calls with and without a `.`. Odd samples look male.
fn panel_vcf(seed: u64, n_samples: usize, max_ploidy: usize) -> String {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ seed;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut text = String::from("##fileformat=VCFv4.3\n");
    for contig in ["1", "2", "22", "X", "Y"] {
        text.push_str(&format!("##contig=<ID={contig}>\n"));
    }
    text.push_str("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n");
    text.push_str("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT");
    for sample in 0..n_samples {
        text.push_str(&format!("\tS{sample}"));
    }
    text.push('\n');
    let mut rows: Vec<(&str, u64)> = Vec::new();
    for (contig, count) in [("1", 900u64), ("2", 700), ("22", 500)] {
        rows.extend((0..count).map(|i| (contig, 10_000 + i * 97)));
    }
    rows.extend((0..400u64).map(|i| ("X", 5_000 + i * 390_000)));
    rows.extend((0..120u64).map(|i| ("Y", 5_000 + i * 470_000)));
    for (contig, position) in rows {
        text.push_str(&format!("{contig}\t{position}\t.\tA\tG\t.\tPASS\t.\tGT"));
        for sample in 0..n_samples {
            let roll = next();
            let allele = |shift: u32| match (roll >> shift) % 16 {
                0 => ".",
                code if code % 2 == 0 => "0",
                _ => "1",
            };
            let separator = |shift: u32| if (roll >> shift) % 2 == 0 { '/' } else { '|' };
            let haploid =
                (sample % 2 == 1 && matches!(contig, "X" | "Y")) || (roll >> 56) % 13 == 0;
            let gt = if haploid {
                allele(8).to_string()
            } else if contig == "Y" {
                "./.".to_string()
            } else if max_ploidy == 3 && (roll >> 48) % 11 == 0 {
                format!(
                    "{}{}{}{}{}",
                    allele(8),
                    separator(4),
                    allele(16),
                    separator(5),
                    allele(24)
                )
            } else {
                format!("{}{}{}", allele(8), separator(4), allele(16))
            };
            text.push('\t');
            text.push_str(&gt);
        }
        text.push('\n');
    }
    text
}

/// The int8 codes htslib writes for one VCF GT of one-digit alleles: an allele
/// as `(index + 1) << 1` and `.` as 0, each with the phase bit of the `|` before
/// it.
fn htslib_gt_codes(gt: &str) -> Vec<i8> {
    let mut codes = Vec::new();
    let mut phased = 0i8;
    let mut rest = gt;
    loop {
        let end = rest.find(['/', '|']).unwrap_or(rest.len());
        let code = match &rest[..end] {
            "." => 0,
            allele => (allele.parse::<i8>().expect("a one-digit allele") + 1) << 1,
        };
        codes.push(code | phased);
        match rest[end..].chars().next() {
            Some(separator) => {
                phased = i8::from(separator == '|');
                rest = &rest[end + 1..];
            }
            None => return codes,
        }
    }
}

/// Writes the BCF htslib writes from the GT-only VCF at `vcf`, which noodles'
/// writer does not: after the header noodles writes, each record is built by
/// hand, a `.` after a `|` keeps its phase bit (code 1), and a call shorter than
/// the record's widest is padded once, with int8's end-of-vector sentinel.
fn write_htslib_bcf(vcf: &Path, bcf: &Path) {
    use noodles_vcf::header::StringMaps;
    use std::io::Write as _;

    let mut reader = noodles_vcf::io::Reader::new(BufReader::new(fs::File::open(vcf).unwrap()));
    let header = reader.read_header().unwrap();
    let string_maps = StringMaps::try_from(&header).unwrap();
    let gt_key = string_maps
        .strings()
        .get_index_of("GT")
        .expect("GT in the header");
    let mut writer = noodles_bcf::io::Writer::new(fs::File::create(bcf).unwrap());
    writer.write_header(&header).unwrap();
    let typed_string = |out: &mut Vec<u8>, value: &str| {
        assert!(value.len() < 15, "a short string");
        out.push(u8::try_from(value.len()).unwrap() << 4 | 7);
        out.extend_from_slice(value.as_bytes());
    };
    let text = fs::read_to_string(vcf).unwrap();
    for line in text.lines().filter(|line| !line.starts_with('#')) {
        let fields: Vec<&str> = line.split('\t').collect();
        let alleles: Vec<&str> = std::iter::once(fields[3])
            .chain(fields[4].split(',').filter(|alt| *alt != "."))
            .collect();
        let calls: Vec<Vec<i8>> = fields[9..].iter().map(|gt| htslib_gt_codes(gt)).collect();
        let ploidy = calls.iter().map(Vec::len).max().expect("samples");
        let contig = string_maps
            .contigs()
            .get_index_of(fields[0])
            .expect("contig in the header");
        let mut site = Vec::new();
        site.extend_from_slice(&i32::try_from(contig).unwrap().to_le_bytes());
        site.extend_from_slice(&(fields[1].parse::<i32>().unwrap() - 1).to_le_bytes());
        site.extend_from_slice(&i32::try_from(fields[3].len()).unwrap().to_le_bytes());
        // A missing QUAL, the allele count over no INFO, and one FORMAT series
        // over the sample count.
        site.extend_from_slice(&0x7f80_0001_u32.to_le_bytes());
        site.extend_from_slice(&(u32::try_from(alleles.len()).unwrap() << 16).to_le_bytes());
        site.extend_from_slice(&(1u32 << 24 | u32::try_from(calls.len()).unwrap()).to_le_bytes());
        typed_string(&mut site, fields[2]);
        for allele in &alleles {
            typed_string(&mut site, allele);
        }
        // An empty FILTER.
        site.push(0x00);
        let mut samples = vec![
            0x11,
            u8::try_from(gt_key).unwrap(),
            u8::try_from(ploidy).unwrap() << 4 | 1,
        ];
        for mut call in calls {
            call.resize(ploidy, i8::MIN + 1);
            samples.extend(call.iter().map(|&code| code as u8));
        }
        let out = writer.get_mut();
        out.write_all(&u32::try_from(site.len()).unwrap().to_le_bytes())
            .unwrap();
        out.write_all(&u32::try_from(samples.len()).unwrap().to_le_bytes())
            .unwrap();
        out.write_all(&site).unwrap();
        out.write_all(&samples).unwrap();
    }
    writer.try_finish().unwrap();
}

/// The dosages score and map read from `a` and from `b` must be the same bit for
/// bit, with a missing call NaN in both.
fn assert_same_dosages(a: &Path, b: &Path) {
    use crate::map::fit::VariantBlockSource as _;

    let read = |path: &Path| -> (usize, Vec<u64>) {
        let dataset = crate::map::io::GenotypeDataset::open(path, None).unwrap();
        let mut source = dataset.block_source().unwrap();
        let n_samples = source.n_samples();
        let mut storage = vec![0.0; 256 * n_samples];
        let mut bits = Vec::new();
        loop {
            let filled = source.next_block_into(256, &mut storage).unwrap();
            if filled == 0 {
                return (n_samples, bits);
            }
            bits.extend(storage[..filled * n_samples].iter().map(|value| {
                if value.is_nan() {
                    u64::MAX
                } else {
                    value.to_bits()
                }
            }));
        }
    };
    let ((n_samples, left), (_, right)) = (read(a), read(b));
    assert_eq!(
        left.len(),
        right.len(),
        "{} and {}",
        a.display(),
        b.display()
    );
    if let Some(at) = left.iter().zip(&right).position(|(l, r)| l != r) {
        panic!(
            "{} and {} first differ at variant {}, sample {}: {:#x} and {:#x}",
            a.display(),
            b.display(),
            at / n_samples,
            at % n_samples,
            left[at],
            right[at]
        );
    }
}

/// score reads a BCF through its own genotype decoder, not the one map and terms
/// share, so the same panel must also score alike from its VCF and its htslib BCF:
/// the same people, missing counts and matched variants, and sums bit for bit.
fn assert_same_scores(stage: &Path, vcf: &str) {
    use crate::score::native_vcf::score_vcf_streaming;

    let score = stage.join("panel.score.tsv");
    let mut text = String::from("variant_id\teffect_allele\tother_allele\tpanel\n");
    let records = vcf.lines().filter(|line| !line.starts_with('#'));
    for (index, line) in records.enumerate().filter(|(index, _)| index % 3 == 0) {
        let fields: Vec<&str> = line.splitn(3, '\t').collect();
        text.push_str(&format!(
            "{}:{}\tG\tA\t{:.2}\n",
            fields[0],
            fields[1],
            (index % 7) as f64 / 4.0 - 0.75
        ));
    }
    fs::write(&score, text).unwrap();
    let run = |input: &str| {
        score_vcf_streaming(&stage.join(input), std::slice::from_ref(&score), None, None)
            .map_err(|err| err.to_string())
    };
    match (run("in.vcf"), run("in.htslib.bcf")) {
        (Ok(from_vcf), Ok(from_bcf)) => {
            assert!(
                from_vcf.matched_variants > 0,
                "{}: nothing scored",
                stage.display()
            );
            assert_eq!(
                (
                    &from_bcf.person_iids,
                    &from_bcf.missing_counts,
                    from_bcf.matched_variants
                ),
                (
                    &from_vcf.person_iids,
                    &from_vcf.missing_counts,
                    from_vcf.matched_variants
                ),
                "{}: score",
                stage.display()
            );
            let bits = |sums: &[f64]| sums.iter().map(|sum| sum.to_bits()).collect::<Vec<_>>();
            assert_eq!(
                bits(&from_bcf.sum_scores),
                bits(&from_vcf.sum_scores),
                "{}: score sums",
                stage.display()
            );
        }
        (from_vcf, from_bcf) => panic!(
            "{}: scoring the VCF gave {from_vcf:?} and the htslib BCF {from_bcf:?}",
            stage.display()
        ),
    }
}

/// A panel of partly missing calls, phased and unphased, and haploid calls gives
/// one sex table as a VCF, as the BCF htslib writes from it, as the BCF noodles
/// writes from it and as the .bed plink2 imports. Its VCF and htslib BCF decode to
/// the same dosages bit for bit through the decoder map and terms share, and score
/// alike through score's own. With triploid calls, which a .bed cannot hold, the
/// VCF and the htslib BCF still agree on all of it. htslib writes the `.` of `1|.`
/// as code 1, which the shared BCF decoder once read as an allele (#2373).
#[test]
fn partly_missing_haploid_and_triploid_calls_read_alike_from_a_vcf_and_its_htslib_bcf() {
    let dir = tempfile::tempdir().unwrap();
    for seed in [1u64, 2, 3] {
        for max_ploidy in [2usize, 3] {
            let stage = dir.path().join(format!("panel_{seed}_{max_ploidy}"));
            let vcf = panel_vcf(seed, 12, max_ploidy);
            assert!(
                vcf.contains("\t1|.") && vcf.contains("\t.|1"),
                "the panel holds phased missing alleles"
            );
            stage_vcf(&stage, &vcf);
            write_htslib_bcf(&stage.join("in.vcf"), &stage.join("in.htslib.bcf"));
            let table = infer(&stage, "vcf");
            assert_eq!(
                infer(&stage, "htslib.bcf"),
                table,
                "{}: htslib bcf",
                stage.display()
            );
            if max_ploidy == 2 {
                assert_eq!(staged_table(&stage), table, "{}: bed", stage.display());
            }
            assert_same_dosages(&stage.join("in.vcf"), &stage.join("in.htslib.bcf"));
            assert_same_scores(&stage, &vcf);
        }
    }
}
