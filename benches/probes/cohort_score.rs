//! Full-marker cohort/keep validation. Inputs are existing normalized PGS files.
use gnomon::score;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
    sync::Arc,
    time::Instant,
};

fn main() {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    assert!(
        (3..=5).contains(&args.len()),
        "prefix score_directory output [keep|-] [reference]"
    );
    let mut files: Vec<_> = std::fs::read_dir(&args[1])
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with("_hmPOS_GRCh38.gnomon.sorted.gnomon.tsv")
        })
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no normalized PGS inputs");
    let keep = args
        .get(3)
        .filter(|path| path.as_os_str() != "-")
        .map(|path| path.as_path());
    let start = Instant::now();
    let prep = score::prepare::prepare_for_computation(&args[..1], &files, keep, None).unwrap();
    let preparation = start.elapsed();
    let context = score::pipeline::PipelineContext::new(Arc::new(prep));
    let start = Instant::now();
    let (scores, counts) = score::pipeline::run(&context).unwrap();
    let compute = start.elapsed();
    let prep = &context.prep_result;
    let columns = prep.score_names.len();
    let mut header = "IID".to_owned();
    for name in &prep.score_names {
        header.push_str(&format!("\t{name}\t{name}_MISSING_COUNT"));
    }
    let mut reference = HashMap::new();
    if let Some(path) = args.get(4) {
        let mut lines = BufReader::new(File::open(path).unwrap()).lines();
        assert_eq!(lines.next().unwrap().unwrap(), header);
        for line in lines {
            let line = line.unwrap();
            let mut fields = line.split('\t');
            let id = fields.next().unwrap().to_owned();
            let values: Vec<_> = fields.map(|v| v.parse::<f64>().unwrap()).collect();
            assert_eq!(values.len(), columns * 2);
            assert!(reference.insert(id, values).is_none());
        }
    }
    let mut output = BufWriter::new(File::create(&args[2]).unwrap());
    writeln!(output, "{header}").unwrap();
    let mut max_abs = 0.0f64;
    let mut max_relative = 0.0f64;
    for (person, id) in prep.final_person_iids.iter().enumerate() {
        let expected = if reference.is_empty() {
            None
        } else {
            let base = id
                .rsplit_once("_t")
                .filter(|(_, suffix)| suffix.parse::<usize>().is_ok())
                .map_or(id.as_str(), |(base, _)| base);
            Some(
                reference
                    .get(base)
                    .unwrap_or_else(|| panic!("no reference for {id}")),
            )
        };
        write!(output, "{id}").unwrap();
        for column in 0..columns {
            let cell = person * columns + column;
            let value = scores[cell];
            assert!(value.is_finite());
            if let Some(expected) = expected {
                let error = (value - expected[column * 2]).abs();
                let relative = error / value.abs().max(expected[column * 2].abs()).max(1e-6);
                max_abs = max_abs.max(error);
                max_relative = max_relative.max(relative);
                assert!(
                    relative <= 2e-6,
                    "{id} {} relative={relative}",
                    prep.score_names[column]
                );
                assert_eq!(counts[cell] as f64, expected[column * 2 + 1]);
            }
            write!(output, "\t{value:.17e}\t{}", counts[cell]).unwrap();
        }
        writeln!(output).unwrap();
    }
    output.flush().unwrap();
    println!(
        "people={} scores={} matched={} nnz={} prep_ms={:.3} compute_ms={:.3} reference_rows={} max_abs={max_abs:.12e} max_relative={max_relative:.12e}",
        prep.num_people_to_score,
        columns,
        prep.num_reconciled_variants,
        prep.sparse_weights().len(),
        preparation.as_secs_f64() * 1000.0,
        compute.as_secs_f64() * 1000.0,
        reference.len()
    );
}
