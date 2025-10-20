use grep::regex::RegexMatcher;
use grep::searcher::{Searcher, Sink, SinkMatch};
use std::ffi::{OsStr, OsString};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use walkdir::WalkDir;

// A custom "Sink" for the grep searcher. It collects all matching lines
// from a single file to build a comprehensive error message.
struct ViolationCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

// A collector for forbidden comment content
struct ForbiddenCommentCollector {
    violations: Vec<String>,
    file_path: PathBuf,
    check_stars_in_doc_comments: bool,
}

// A custom collector for checking if comments have an excessive ratio of uppercase characters
struct CustomUppercaseCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

// A custom collector for checking if comments are primarily composed of dashes
struct DashHeavyCommentCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

// A custom collector for #[allow(dead_code)] attribute violations
struct DeadCodeCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

// A custom collector for #[ignore] test attribute violations
struct IgnoredTestCollector {
    violations: Vec<String>,
    file_path: PathBuf,
}

static CURRENT_STAGE: OnceLock<Mutex<String>> = OnceLock::new();

fn update_stage(label: &str) {
    let tracker = CURRENT_STAGE.get_or_init(|| Mutex::new(String::new()));
    if let Ok(mut guard) = tracker.lock() {
        guard.clear();
        guard.push_str(label);
    }

    println!("cargo:warning=gnomon build stage: {label}");
    let _ = io::stdout().flush();
}

fn emit_stage_detail(detail: &str) {
    println!("cargo:warning=gnomon build detail: {detail}");
    let _ = io::stdout().flush();
}

fn install_stage_panic_hook() {
    let tracker: &'static Mutex<String> = CURRENT_STAGE.get_or_init(|| Mutex::new(String::new()));
    std::panic::set_hook(Box::new(move |info| {
        let stage_name = tracker
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_else(|_| String::from("<stage lock poisoned>"));
        eprintln!("\n⚠️ build script panic while processing stage: {stage_name}");
        eprintln!("{info}");
    }));
}

impl ViolationCollector {
    fn new(file_path: &Path) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
        }
    }

    // After searching, this method checks if any violations were found.
    // If so, it formats a detailed error message and returns it.
    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let file_name = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} underscore-prefixed variables in {}:\n",
            self.violations.len(),
            file_name
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg
            .push_str("\n⚠️ Underscore-prefixed variable names are not allowed in this project.\n");
        error_msg.push_str(
            "   Either use the variable (removing the underscore) or remove it completely.\n",
        );

        Some(error_msg)
    }
}

impl ForbiddenCommentCollector {
    fn new(file_path: &Path, check_stars_in_doc_comments: bool) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
            check_stars_in_doc_comments,
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let file_name = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} forbidden comment patterns in {}:\n",
            self.violations.len(),
            file_name
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg.push_str("\n⚠️ Comments containing 'FIXED', 'CORRECTED', 'FIX', 'FIXES', 'NEW', 'CHANGED', 'CHANGES', 'CHANGE', 'MODIFIED', 'MODIFIES', 'MODIFY', 'UPDATED', 'UPDATES', or 'UPDATE' are STRICTLY FORBIDDEN in this project.\n");
        error_msg.push_str("   These comments will cause compilation to fail. Remove them completely rather than commenting them out.\n");
        error_msg.push_str("   The '**' pattern is not allowed in regular comments (but is allowed in doc comments).\n");
        error_msg.push_str(
            "   Comments where over 80% of alphabetic characters are uppercase are not allowed.\n",
        );
        error_msg.push_str("   Please remove these patterns before committing.\n");

        Some(error_msg)
    }
}

impl CustomUppercaseCollector {
    fn new(file_path: &Path) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let file_name = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} comments with excessive uppercase alphabetic characters in {}:\n",
            self.violations.len(),
            file_name
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg.push_str(
            "\n⚠️ Comments where over 80% of alphabetic characters are uppercase are STRICTLY FORBIDDEN in this project.\n",
        );
        error_msg.push_str("   STRONGLY CONSIDER deleting the comment completely.\n");

        Some(error_msg)
    }
}

impl DashHeavyCommentCollector {
    fn new(file_path: &Path) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let file_name = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} comments composed primarily of dashes in {}:\n",
            self.violations.len(),
            file_name
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg.push_str(
            "\n⚠️ Comments where over 80% of non-whitespace characters are dashes are STRICTLY FORBIDDEN in this project.\n",
        );
        error_msg.push_str("   Remove decorative dash-only comments completely.\n");

        Some(error_msg)
    }
}

impl DeadCodeCollector {
    fn new(file_path: &Path) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let file_name = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} #[allow(dead_code)] attributes in {}:\n",
            self.violations.len(),
            file_name
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg.push_str(
            "\n⚠️ #[allow(dead_code)] attributes are STRICTLY FORBIDDEN in this project.\n",
        );
        error_msg
            .push_str("   Either use the code (removing the attribute) or remove it completely.\n");

        Some(error_msg)
    }
}

impl IgnoredTestCollector {
    fn new(file_path: &Path) -> Self {
        Self {
            violations: Vec::new(),
            file_path: file_path.to_path_buf(),
        }
    }

    fn check_and_get_error_message(&self) -> Option<String> {
        if self.violations.is_empty() {
            return None;
        }

        let file_name = self.file_path.to_str().unwrap_or("?");
        let mut error_msg = format!(
            "\n❌ ERROR: Found {} #[ignore] test attributes in {}:\n",
            self.violations.len(),
            file_name
        );

        for violation in &self.violations {
            error_msg.push_str(&format!("   {violation}\n"));
        }

        error_msg
            .push_str("\n⚠️ #[ignore] TEST ATTRIBUTES ARE STRICTLY FORBIDDEN IN THIS PROJECT!\n");
        error_msg.push_str("   IGNORING TESTS IS NEVER ALLOWED FOR ANY REASON.\n");
        error_msg.push_str("   Fix the test so it can run properly without being ignored.\n");

        Some(error_msg)
    }
}

// Implement the `Sink` trait for our collector.
// The `matched` method is called by the searcher for every line that matches the regex.
impl Sink for ViolationCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        // Get the line number and the content of the matched line.
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        // Skip matches in comments and string literals to avoid false positives
        // But make sure we don't miss underscore variables in code

        // Check if this line is purely a comment
        let is_pure_comment = line_text.trim_start().starts_with("//")
            || (line_text.contains("/*")
                && !line_text.contains("*/match")
                && !line_text.contains("*/let"));

        // Check if the match is in a string literal and not part of code
        let mut is_in_string = false;
        if line_text.contains("\"") {
            // More careful string detection logic
            let parts: Vec<&str> = line_text.split('\"').collect();
            // If the underscore variable is between quotes, it's in a string
            for (i, part) in parts.iter().enumerate() {
                if i % 2 == 1 && part.contains("_") {
                    // Inside quotes
                    is_in_string = true;
                    break;
                }
            }
        }

        if is_pure_comment || is_in_string {
            return Ok(true); // Skip this match and continue searching
        }

        // Format the violation string exactly as the `rg -n` command would.
        self.violations.push(format!("{line_number}:{line_text}"));

        // Return `Ok(true)` to continue searching for more matches in the same file.
        Ok(true)
    }
}

// Implement the Sink trait for the forbidden comment collector
impl Sink for ForbiddenCommentCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        // Skip ** in doc comments if not checking for them
        // But NEVER skip any line containing FIXED, CORRECTED, or FIX
        if !self.check_stars_in_doc_comments
            && is_doc_comment(line_text)
            && line_text.contains("**")
            && !line_text.contains("FIXED")
            && !line_text.contains("CORRECTED")
            && !line_text.contains("FIX")
            && !line_text.contains("FIXES")
            && !line_text.contains("NEW")
            && !line_text.contains("CHANGED")
            && !line_text.contains("CHANGES")
            && !line_text.contains("CHANGE")
            && !line_text.contains("MODIFIED")
            && !line_text.contains("MODIFIES")
            && !line_text.contains("MODIFY")
            && !line_text.contains("UPDATED")
            && !line_text.contains("UPDATES")
            && !line_text.contains("UPDATE")
        {
            // Skip this match, it's just ** in a doc comment
            return Ok(true);
        }

        // Format the violation string
        self.violations.push(format!("{line_number}:{line_text}"));

        Ok(true)
    }
}

// Implement the Sink trait for the uppercase character collector
impl Sink for CustomUppercaseCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        // Check if it's a comment line
        if !line_text.trim_start().starts_with("//")
            && !line_text.contains("/*")
            && !line_text.starts_with("///")
        {
            return Ok(true); // Not a comment, skip
        }

        // Extract just the comment part (remove the // or /* prefix)
        let comment_text = if line_text.trim_start().starts_with("///") {
            line_text.trim_start()[3..].trim()
        } else if line_text.trim_start().starts_with("//") {
            line_text.trim_start()[2..].trim()
        } else if let Some(idx) = line_text.find("/*") {
            match line_text[idx + 2..].find("*/") {
                Some(end) => line_text[idx + 2..idx + 2 + end].trim(),
                None => line_text[idx + 2..].trim(),
            }
        } else {
            return Ok(true); // Not a comment we can parse, skip
        };

        // Find all alphabetic characters
        let alpha_chars: Vec<char> = comment_text.chars().filter(|c| c.is_alphabetic()).collect();

        if !alpha_chars.is_empty() {
            let uppercase_count = alpha_chars.iter().filter(|c| c.is_uppercase()).count();
            let uppercase_ratio = uppercase_count as f64 / alpha_chars.len() as f64;

            if uppercase_ratio > 0.8 {
                self.violations.push(format!("{line_number}:{line_text}"));
            }
        }

        Ok(true)
    }
}

impl Sink for DashHeavyCommentCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        if !line_text.trim_start().starts_with("//")
            && !line_text.contains("/*")
            && !line_text.starts_with("///")
        {
            return Ok(true);
        }

        let comment_text = if line_text.trim_start().starts_with("///") {
            line_text.trim_start()[3..].trim()
        } else if line_text.trim_start().starts_with("//") {
            line_text.trim_start()[2..].trim()
        } else if let Some(idx) = line_text.find("/*") {
            match line_text[idx + 2..].find("*/") {
                Some(end) => line_text[idx + 2..idx + 2 + end].trim(),
                None => line_text[idx + 2..].trim(),
            }
        } else {
            return Ok(true);
        };

        let non_whitespace_chars: Vec<char> = comment_text
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();

        if !non_whitespace_chars.is_empty() {
            let dash_count = non_whitespace_chars.iter().filter(|c| **c == '-').count();
            let dash_ratio = dash_count as f64 / non_whitespace_chars.len() as f64;

            if dash_ratio > 0.8 {
                self.violations.push(format!("{line_number}:{line_text}"));
            }
        }

        Ok(true)
    }
}

impl Sink for DeadCodeCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        // Get the line number and the content of the matched line.
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        // Format the violation string
        self.violations.push(format!("{line_number}:{line_text}"));

        // Return `Ok(true)` to continue searching for more matches in the same file.
        Ok(true)
    }
}

impl Sink for IgnoredTestCollector {
    type Error = std::io::Error;

    fn matched(&mut self, _: &Searcher, mat: &SinkMatch) -> Result<bool, Self::Error> {
        // Get the line number and the content of the matched line.
        let line_number = mat.line_number().unwrap_or(0);
        let line_text = std::str::from_utf8(mat.bytes()).unwrap_or("").trim_end();

        // Format the violation string
        self.violations.push(format!("{line_number}:{line_text}"));

        // Return `Ok(true)` to continue searching for more matches in the same file.
        Ok(true)
    }
}

fn main() {
    install_stage_panic_hook();

    // Always rerun this script if the build script itself changes.
    update_stage("initialization");
    println!("cargo:rerun-if-changed=build.rs");

    // Manually check for unused variables in the build script
    update_stage("manual lint self-check");
    manually_check_for_unused_variables();

    // Collect all violations from all checks
    let mut all_violations = Vec::new();

    // Scan Rust source files for underscore prefixed variables
    update_stage("scan underscore-prefixed bindings");
    let underscore_violations = scan_for_underscore_prefixes();
    let underscore_report = format!(
        "underscore scan identified {} violation groups",
        underscore_violations.len()
    );
    emit_stage_detail(&underscore_report);
    all_violations.extend(underscore_violations);

    // Scan Rust source files for forbidden comment patterns
    update_stage("scan forbidden comment patterns");
    let comment_violations = scan_for_forbidden_comment_patterns();
    let comment_report = format!(
        "forbidden comment scan identified {} violation groups",
        comment_violations.len()
    );
    emit_stage_detail(&comment_report);
    all_violations.extend(comment_violations);

    // Scan Rust source files for #[allow(dead_code)] attributes
    update_stage("scan allow(dead_code) attributes");
    let dead_code_violations = scan_for_allow_dead_code();
    let dead_code_report = format!(
        "allow(dead_code) scan identified {} violation groups",
        dead_code_violations.len()
    );
    emit_stage_detail(&dead_code_report);
    all_violations.extend(dead_code_violations);

    // Scan Rust source files for #[ignore] test attributes
    update_stage("scan #[ignore] test annotations");
    let ignored_test_violations = scan_for_ignored_tests();
    let ignored_report = format!(
        "ignored test scan identified {} violation groups",
        ignored_test_violations.len()
    );
    emit_stage_detail(&ignored_report);
    all_violations.extend(ignored_test_violations);

    // If any violations were found, print them all and exit with error
    if !all_violations.is_empty() {
        update_stage("report validation errors");
        eprintln!("\n❌ VALIDATION ERRORS");
        eprintln!("====================");

        let violation_count = all_violations.len();

        for violation in all_violations {
            eprintln!("{violation}");
            eprintln!("--------------------");
        }

        eprintln!(
            "\n⚠️ Found {} total code quality violations. Fix all issues before committing.",
            violation_count
        );
        std::process::exit(1);
    }

    update_stage("build script completed");
    emit_stage_detail("Validation checks completed without errors");
}

// This function manually checks for unused variables in the current file
fn manually_check_for_unused_variables() {
    // Force compilation to fail with unused_variables, dead_code, and unused_imports lint
    // This ensures build.rs itself follows the strict coding policy
    let build_path = Path::new("build.rs");
    let rustc_binary = std::env::var_os("RUSTC").unwrap_or_else(|| OsString::from("rustc"));
    let manual_lint_args = manual_lint_arguments(build_path);

    update_stage("manual lint self-check: running rustc");
    emit_stage_detail(&format!(
        "manual lint self-check: selected rustc executable: {:?}",
        rustc_binary
    ));

    if let Some(host) = std::env::var_os("HOST") {
        emit_stage_detail(&format!(
            "manual lint self-check: HOST environment: {:?}",
            host
        ));
    }

    if let Some(target) = std::env::var_os("TARGET") {
        emit_stage_detail(&format!(
            "manual lint self-check: TARGET environment: {:?}",
            target
        ));
    }

    if let Some(triple) = std::env::var_os("CARGO_CFG_TARGET_ARCH") {
        emit_stage_detail(&format!(
            "manual lint self-check: cfg target arch: {:?}",
            triple
        ));
    }

    emit_stage_detail(&format!(
        "manual lint self-check: build context arch/os: {} / {}",
        std::env::consts::ARCH,
        std::env::consts::OS
    ));

    update_stage("manual lint self-check: preparing rustc command");
    emit_stage_detail(&format!(
        "manual lint self-check: command preview: {}",
        command_preview(&rustc_binary, &manual_lint_args)
    ));

    if let Some(cwd) = std::env::current_dir().ok() {
        emit_stage_detail(&format!(
            "manual lint self-check: current dir before spawn: {:?}",
            cwd
        ));
    }

    let mut command = std::process::Command::new(&rustc_binary);
    command.args(&manual_lint_args);

    update_stage("manual lint self-check: invoking rustc");
    emit_stage_detail("manual lint self-check: calling Command::output() for rustc self-lint");
    let status = command.output();

    update_stage("manual lint self-check: rustc invocation returned");

    match status {
        Ok(output) => {
            emit_stage_detail(&format!(
                "manual lint self-check: rustc exit status: {:?}",
                output.status.code()
            ));
            emit_stage_detail(&format!(
                "manual lint self-check: rustc stdout bytes: {} / stderr bytes: {}",
                output.stdout.len(),
                output.stderr.len()
            ));

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                if stderr.contains("unused variable") {
                    eprintln!("\n❌ ERROR: Unused variables detected in build.rs!");
                    eprintln!("{stderr}");
                    eprintln!("\n⚠️ Unused variables are STRICTLY FORBIDDEN in this project.");
                    eprintln!(
                        "   Either use the variable or remove it completely. Underscore prefixes are NOT allowed."
                    );
                    std::process::exit(1);
                } else if stderr.contains("function is never used") {
                    eprintln!("\n❌ ERROR: Unused functions detected in build.rs!");
                    eprintln!("{stderr}");
                    eprintln!("\n⚠️ Unused functions are STRICTLY FORBIDDEN in this project.");
                    eprintln!("   Either use the function or remove it completely.");
                    std::process::exit(1);
                } else if stderr.contains("unused import") {
                    eprintln!("\n❌ ERROR: Unused imports detected in build.rs!");
                    eprintln!("{stderr}");
                    eprintln!("\n⚠️ Unused imports are STRICTLY FORBIDDEN in this project.");
                    eprintln!("   Either use the imported item or remove the import completely.");
                    std::process::exit(1);
                }
            } else {
                emit_stage_detail("Completed rustc self-lint for build.rs");
            }
        }
        Err(err) => {
            emit_stage_detail(&format!(
                "manual lint self-check: failed to start rustc self-lint command: {err}"
            ));
            eprintln!(
                "cargo:warning=Could not check for unused variables/functions/imports in build.rs"
            );
        }
    }
}

fn manual_lint_arguments(build_path: &Path) -> Vec<OsString> {
    vec![
        OsString::from("--edition"),
        OsString::from("2021"),
        OsString::from("-D"),
        OsString::from("unused_variables"),
        OsString::from("-D"),
        OsString::from("dead_code"),
        OsString::from("-D"),
        OsString::from("unused_imports"),
        OsString::from("--crate-type"),
        OsString::from("bin"),
        OsString::from("--error-format"),
        OsString::from("human"),
        build_path.as_os_str().to_os_string(),
    ]
}

fn command_preview(program: &OsStr, args: &[OsString]) -> String {
    let mut parts = Vec::with_capacity(args.len() + 1);
    parts.push(format!("{program:?}"));
    for arg in args {
        parts.push(format!("{arg:?}"));
    }
    parts.join(" ")
}

fn scan_for_underscore_prefixes() -> Vec<String> {
    // Regex pattern to find underscore prefixed variable names.
    // This pattern needs to be more generalized to catch all underscore-prefixed variables,
    // especially in match statements and destructuring patterns
    let pattern = r"\b(_[a-zA-Z0-9_]+)\b";
    let mut all_violations = Vec::new();

    match RegexMatcher::new_line_matcher(pattern) {
        Ok(matcher) => {
            let mut searcher = Searcher::new();

            // Use `walkdir` to find all Rust files, replacing the `find` command.
            // This is more portable and robust.
            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e| e.ok()) // Ignore any errors during directory traversal.
                .filter(|e| !e.path().starts_with("./target")) // Exclude the target directory.
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
            // Keep only .rs files.
            {
                let path = entry.path();

                // Check if we can read the file
                match std::fs::read_to_string(path) {
                    Ok(_) => {}         // File exists and can be read
                    Err(_) => continue, // Skip files we can't read
                };

                // Add debug info for estimate.rs to help diagnose the underscore variable detection
                let is_estimate_rs = path
                    .to_str()
                    .is_some_and(|p| p.ends_with("calibrate/estimate.rs"));
                if is_estimate_rs {
                    println!(
                        "cargo:warning=Analyzing estimate.rs for underscore-prefixed variables"
                    );
                }

                // Create a new collector for each file.
                let mut collector = ViolationCollector::new(path);

                // Search the file using our regex matcher and collector sink.
                if searcher
                    .search_path(&matcher, path, &mut collector)
                    .is_err()
                {
                    // Handle search errors gracefully
                    continue;
                }

                // Process results
                if let Some(error_message) = collector.check_and_get_error_message() {
                    // Add this error to our collection instead of returning immediately
                    all_violations.push(error_message);
                }
            }
        }
        Err(e) => {
            // If there's an error creating the matcher, report it but don't return early
            all_violations.push(format!(
                "Error creating regex matcher for underscore prefixes: {}",
                e
            ));
        }
    }

    // Return all violations found
    all_violations
}

fn is_doc_comment(line: &str) -> bool {
    line.trim_start().starts_with("///")
}

fn scan_for_forbidden_comment_patterns() -> Vec<String> {
    // Regex patterns to find forbidden comment patterns
    // Note: We specifically target comments by looking for // or /* */ patterns
    // This ensures we don't flag these terms in actual code
    let mut all_violations = Vec::new();

    // Split into separate patterns for clarity and reliability
    // 1. Pattern to catch forbidden words in comments
    let forbidden_words_pattern = r"(//|/\*|///).*(?:FIXED|CORRECTED|FIX|FIXES|NEW|CHANGED|CHANGES|CHANGE|MODIFIED|MODIFIES|MODIFY|UPDATED|UPDATES|UPDATE)";
    // 2. Pattern to catch ** in comments (excluding doc comments)
    let stars_pattern = r"(//|/\*).*\*\*";
    // 3. Pattern to catch comments for uppercase ratio enforcement
    let all_caps_pattern = r"(//|/\*|///).*";
    // 4. Pattern to catch comments that might be composed primarily of dashes
    let dash_heavy_pattern = r"(//|/\*|///).*";

    // Check for forbidden words
    match RegexMatcher::new_line_matcher(forbidden_words_pattern) {
        Ok(forbidden_matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| !e.path().starts_with("./target")) // Exclude target directory
                .filter(|e| e.file_name() != "build.rs") // Exclude the build script itself
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                // Use a collector that doesn't filter out doc comments for forbidden words
                let mut collector = ForbiddenCommentCollector::new(path, true);
                if searcher
                    .search_path(&forbidden_matcher, path, &mut collector)
                    .is_err()
                {
                    continue;
                }

                if let Some(error_message) = collector.check_and_get_error_message() {
                    all_violations.push(error_message);
                }
            }
        }
        Err(e) => {
            // Record the error but continue checking other patterns
            all_violations.push(format!("Error creating forbidden words regex: {}", e));
        }
    }

    // Check for stars in non-doc comments
    match RegexMatcher::new_line_matcher(stars_pattern) {
        Ok(stars_matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| !e.path().starts_with("./target")) // Exclude target directory
                .filter(|e| e.file_name() != "build.rs") // Exclude the build script itself
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                // Use a single collector with custom filtering logic
                // false means don't check for ** in doc comments
                let mut collector = ForbiddenCommentCollector::new(path, false);
                if searcher
                    .search_path(&stars_matcher, path, &mut collector)
                    .is_err()
                {
                    continue;
                }

                if let Some(error_message) = collector.check_and_get_error_message() {
                    all_violations.push(error_message);
                }
            }
        }
        Err(e) => {
            // Record the error but continue checking other patterns
            all_violations.push(format!("Error creating stars pattern regex: {}", e));
        }
    }

    // Check for comments where the uppercase ratio exceeds the threshold
    match RegexMatcher::new_line_matcher(all_caps_pattern) {
        Ok(all_caps_matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| !e.path().starts_with("./target"))
                .filter(|e| e.file_name() != "build.rs")
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                let mut custom_collector = CustomUppercaseCollector::new(path);
                if searcher
                    .search_path(&all_caps_matcher, path, &mut custom_collector)
                    .is_err()
                {
                    continue;
                }

                if let Some(error_message) = custom_collector.check_and_get_error_message() {
                    all_violations.push(error_message);
                }
            }
        }
        Err(e) => {
            // Record the error but don't return early
            all_violations.push(format!("Error creating uppercase pattern regex: {}", e));
        }
    }

    // Check for comments composed primarily of dashes
    match RegexMatcher::new_line_matcher(dash_heavy_pattern) {
        Ok(dash_matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| !e.path().starts_with("./target"))
                .filter(|e| e.file_name() != "build.rs")
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                let mut dash_collector = DashHeavyCommentCollector::new(path);
                if searcher
                    .search_path(&dash_matcher, path, &mut dash_collector)
                    .is_err()
                {
                    continue;
                }

                if let Some(error_message) = dash_collector.check_and_get_error_message() {
                    all_violations.push(error_message);
                }
            }
        }
        Err(e) => {
            all_violations.push(format!("Error creating dash-heavy pattern regex: {}", e));
        }
    }

    all_violations
}

fn scan_for_allow_dead_code() -> Vec<String> {
    // Regex pattern to find #[allow(dead_code)] attributes
    let pattern = r"#\s*\[\s*allow\s*\(\s*dead_code\s*\)\s*\]";
    let mut all_violations = Vec::new();

    match RegexMatcher::new_line_matcher(pattern) {
        Ok(matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| !e.path().starts_with("./target")) // Exclude target directory
                .filter(|e| e.file_name() != "build.rs") // Exclude the build script itself
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                // Check if we can read the file
                match std::fs::read_to_string(path) {
                    Ok(_) => {}         // File exists and can be read
                    Err(_) => continue, // Skip files we can't read
                };

                // Create a collector for each file
                let mut collector = DeadCodeCollector::new(path);

                // Search the file using our regex matcher and collector sink
                if searcher
                    .search_path(&matcher, path, &mut collector)
                    .is_err()
                {
                    // Handle search errors gracefully
                    continue;
                }

                // Process results
                if let Some(error_message) = collector.check_and_get_error_message() {
                    // Add this error to our collection instead of returning immediately
                    all_violations.push(error_message);
                }
            }
        }
        Err(e) => {
            // If there's an error creating the matcher, report it but don't return early
            all_violations.push(format!("Error creating dead code regex matcher: {}", e));
        }
    }

    // Return all violations found
    all_violations
}

fn scan_for_ignored_tests() -> Vec<String> {
    // Regex pattern to find #[ignore] test attributes
    let pattern = r"#\s*\[\s*ignore\s*\]";
    let mut all_violations = Vec::new();

    match RegexMatcher::new_line_matcher(pattern) {
        Ok(matcher) => {
            let mut searcher = Searcher::new();

            for entry in WalkDir::new(".")
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| !e.path().starts_with("./target")) // Exclude target directory
                .filter(|e| e.file_name() != "build.rs") // Exclude the build script itself
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
            {
                let path = entry.path();

                // Check if we can read the file
                match std::fs::read_to_string(path) {
                    Ok(_) => {}         // File exists and can be read
                    Err(_) => continue, // Skip files we can't read
                };

                // Create a collector for each file
                let mut collector = IgnoredTestCollector::new(path);

                // Search the file using our regex matcher and collector sink
                if searcher
                    .search_path(&matcher, path, &mut collector)
                    .is_err()
                {
                    // Handle search errors gracefully
                    continue;
                }

                // Process results
                if let Some(error_message) = collector.check_and_get_error_message() {
                    // Add this error to our collection instead of returning immediately
                    all_violations.push(error_message);
                }
            }
        }
        Err(e) => {
            // If there's an error creating the matcher, report it but don't return early
            all_violations.push(format!("Error creating ignored tests regex matcher: {}", e));
        }
    }

    // Return all violations found
    all_violations
}
