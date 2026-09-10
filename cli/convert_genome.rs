use std::process::ExitCode;

fn main() -> ExitCode {
    match convert_genome::cli::run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {error:#}");
            ExitCode::FAILURE
        }
    }
}
