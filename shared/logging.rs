//! Log records for the command-line programs.
//!
//! Library code reports recoverable trouble through the `log` facade: a remote read
//! falling back or retrying, a progress-tracking inconsistency. Nothing prints those
//! records until a program installs a logger, so the CLI installs this one at startup.
//! It writes to stderr, so standard output carries results only. The level is warn
//! unless `GNOMON_LOG_LEVEL` names another: off, error, warn, info, debug or trace.

use log::{LevelFilter, Log, Metadata, Record};
use std::io::{self, Write};

/// The environment variable that selects the log level.
pub const LOG_LEVEL_ENV: &str = "GNOMON_LOG_LEVEL";

struct StderrLogger;

impl Log for StderrLogger {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &Record<'_>) {
        if self.enabled(record.metadata()) {
            // One write per record, so records from several threads never interleave.
            let _ = io::stderr()
                .lock()
                .write_all(format_record(record).as_bytes());
        }
    }

    fn flush(&self) {
        let _ = io::stderr().flush();
    }
}

static LOGGER: StderrLogger = StderrLogger;

/// Installs the stderr logger. Call once, at program startup. If a logger is already
/// installed, it stays in place.
pub fn install_stderr_logger() {
    let requested = std::env::var(LOG_LEVEL_ENV).ok();
    let level = level_from(requested.as_deref());
    if log::set_logger(&LOGGER).is_err() {
        return;
    }
    log::set_max_level(level.unwrap_or(LevelFilter::Warn));
    if let Err(name) = level {
        log::warn!(
            "{LOG_LEVEL_ENV}={name:?} is not a log level (off, error, warn, info, debug, trace); using warn"
        );
    }
}

/// The level named by `value`, case-insensitively. An unset or empty value means warn.
/// A value that names no level is returned as the error.
fn level_from(value: Option<&str>) -> Result<LevelFilter, &str> {
    match value.map(str::trim) {
        None | Some("") => Ok(LevelFilter::Warn),
        Some(name) => name.parse::<LevelFilter>().map_err(|_| name),
    }
}

/// One line per record: its level, then its message.
fn format_record(record: &Record<'_>) -> String {
    format!("[{}] {}\n", record.level(), record.args())
}

#[cfg(test)]
mod tests {
    use super::{format_record, level_from};
    use log::{Level, LevelFilter, Record};

    #[test]
    fn levels_are_named_case_insensitively_and_default_to_warn() {
        assert_eq!(level_from(None), Ok(LevelFilter::Warn));
        assert_eq!(level_from(Some("")), Ok(LevelFilter::Warn));
        assert_eq!(level_from(Some("debug")), Ok(LevelFilter::Debug));
        assert_eq!(level_from(Some(" TRACE ")), Ok(LevelFilter::Trace));
        assert_eq!(level_from(Some("off")), Ok(LevelFilter::Off));
        assert_eq!(level_from(Some("loud")), Err("loud"));
    }

    #[test]
    fn a_record_is_one_line_with_its_level() {
        assert_eq!(
            format_record(
                &Record::builder()
                    .level(Level::Warn)
                    .args(format_args!("falling back to HTTPS"))
                    .build()
            ),
            "[WARN] falling back to HTTPS\n"
        );
    }
}
