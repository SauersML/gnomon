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
