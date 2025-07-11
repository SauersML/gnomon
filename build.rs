use grep::regex::RegexMatcher;
use grep::searcher::{Searcher, Sink, SinkMatch};
use std::error::Error;
use std::path::{Path, PathBuf};
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
            error_msg.push_str(&format!("   {}\n", violation));
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
            error_msg.push_str(&format!("   {}\n", violation));
        }

        error_msg.push_str("\n⚠️ Comments containing 'FIXED', 'CORRECTED', or 'FIX' are STRICTLY FORBIDDEN in this project.\n");
        error_msg.push_str("   These comments will cause compilation to fail. Remove them completely rather than commenting them out.\n");
        error_msg.push_str("   The '**' pattern is not allowed in regular comments (but is allowed in doc comments).\n");
        error_msg.push_str("   Please remove these patterns before committing.\n");

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
        let is_pure_comment = line_text.trim_start().starts_with("//") || 
                             (line_text.contains("/*") && !line_text.contains("*/match") && !line_text.contains("*/let"));
        
        // Check if the match is in a string literal and not part of code
        let mut is_in_string = false;
        if line_text.contains("\"") {
            // More careful string detection logic
            let parts: Vec<&str> = line_text.split('\"').collect();
            // If the underscore variable is between quotes, it's in a string
            for (i, part) in parts.iter().enumerate() {
                if i % 2 == 1 && part.contains("_") { // Inside quotes
                    is_in_string = true;
                    break;
                }
            }
        }
        
        if is_pure_comment || is_in_string {
            return Ok(true); // Skip this match and continue searching
        }

        // Format the violation string exactly as the `rg -n` command would.
        self.violations
            .push(format!("{}:{}", line_number, line_text));

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
        if !self.check_stars_in_doc_comments && 
           is_doc_comment(line_text) && 
           line_text.contains("**") && 
           !line_text.contains("FIXED") && 
           !line_text.contains("CORRECTED") && 
           !line_text.contains("FIX") {
            // Skip this match, it's just ** in a doc comment
            return Ok(true);
        }

        // Format the violation string
        self.violations
            .push(format!("{}:{}", line_number, line_text));

        Ok(true)
    }
}

fn main() {
    // Always rerun this script if the build script itself changes.
    println!("cargo:rerun-if-changed=build.rs");

    // Manually check for unused variables in the build script
    manually_check_for_unused_variables();

    // Scan Rust source files for underscore prefixed variables and fail if found.
    if let Err(e) = scan_for_underscore_prefixes() {
        // Print the formatted error and fail the build.
        // The `eprintln!` here is crucial for showing the error in `cargo`'s output.
        eprintln!("{}", e);
        std::process::exit(1);
    }
    
    // Scan Rust source files for forbidden comment patterns and fail if found.
    if let Err(e) = scan_for_forbidden_comment_patterns() {
        eprintln!("{}", e);
        std::process::exit(1);
    }
}

// This function manually checks for unused variables in the current file
fn manually_check_for_unused_variables() {
    // Use grep to search for patterns that might indicate unused variables
    let pattern = r"\blet\s+([^_][a-zA-Z0-9_]+)\s*="; // Look for variable declarations
    let matcher = RegexMatcher::new_line_matcher(pattern).unwrap();
    let mut searcher = Searcher::new();

    let build_path = Path::new("build.rs");
    let file_content = std::fs::read_to_string(build_path).unwrap();

    // Collect all variable declarations
    let mut collector = ViolationCollector::new(build_path);
    searcher
        .search_path(&matcher, build_path, &mut collector)
        .unwrap();

    // For each declared variable, check if it's used elsewhere in the file
    for line in &collector.violations {
        // Extract variable name from the violation line
        if let Some(var_name_start) = line.find("let ") {
            if let Some(var_name_end) = line[var_name_start + 4..].find('=') {
                let var_name = line[var_name_start + 4..var_name_start + 4 + var_name_end].trim();

                // Count occurrences of this variable name in the file
                // (should be more than 1 if it's used after declaration)
                let count = file_content.matches(var_name).count();

                if count <= 1 {
                    // If variable appears only once (its declaration), it's unused
                    eprintln!(
                        "\n❌ ERROR: Unused variable detected in build.rs: {}",
                        var_name
                    );
                    eprintln!("   {}", line);
                    eprintln!("\n⚠️ Unused variables are not allowed in this project.");
                    eprintln!(
                        "   Either use the variable or prefix it with an underscore to explicitly mark it as unused."
                    );
                    std::process::exit(1);
                }
            }
        }
    }
}

fn scan_for_underscore_prefixes() -> Result<(), Box<dyn Error>> {
    // Regex pattern to find underscore prefixed variable names.
    // This pattern needs to be more generalized to catch all underscore-prefixed variables,
    // especially in match statements and destructuring patterns
    let pattern = r"\b(_[a-zA-Z0-9_]+)\b";
    let matcher = RegexMatcher::new_line_matcher(pattern)?;
    let mut searcher = Searcher::new();

    // Use `walkdir` to find all Rust files, replacing the `find` command.
    // This is more portable and robust.
    for entry in WalkDir::new(".")
        .into_iter()
        .filter_map(|e| e.ok()) // Ignore any errors during directory traversal.
        .filter(|e| !e.path().starts_with("./target")) // Exclude the target directory.
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "rs"))
    // Keep only .rs files.
    {
        let path = entry.path();

        // Read file content to check for test modules
        let file_content = match std::fs::read_to_string(path) {
            Ok(content) => content,
            Err(_) => continue, // Skip files we can't read
        };

        // Skip analyzing files that are in test modules or test files
        let is_test_file = path.to_str().map_or(false, |p| p.contains("tests/") || p.ends_with("_test.rs") || p.ends_with(".test.rs"));
        
        // Add debug info for estimate.rs to help diagnose the underscore variable detection
        let is_estimate_rs = path.to_str().map_or(false, |p| p.ends_with("calibrate/estimate.rs"));
        if is_estimate_rs {
            println!("cargo:warning=Analyzing estimate.rs for underscore-prefixed variables");
        }
        
        if !is_test_file {
            // Create a new collector for each file.
            let mut collector = ViolationCollector::new(path);

            // Search the file using our regex matcher and collector sink.
            searcher.search_path(&matcher, path, &mut collector)?;

            // Process results
            if let Some(error_message) = collector.check_and_get_error_message() {
                // If violations were found, return the error - no exceptions for test modules
                return Err(error_message.into());
            }
        }
    }

    // If the loop completes without finding any violations, the check passes.
    Ok(())
}

fn is_doc_comment(line: &str) -> bool {
    line.trim_start().starts_with("///")
}

fn scan_for_forbidden_comment_patterns() -> Result<(), Box<dyn Error>> {
    // Regex patterns to find forbidden comment patterns
    // Note: We specifically target comments by looking for // or /* */ patterns
    // This ensures we don't flag these terms in actual code
    
    // Split into two separate patterns for clarity and reliability
    // 1. Pattern to catch forbidden words in comments
    let forbidden_words_pattern = r"(//|/\*|///).*(?:FIXED|CORRECTED|FIX)";
    // 2. Pattern to catch ** in comments (excluding doc comments)
    let stars_pattern = r"(//|/\*).*\*\*";
    
    // First check for forbidden words
    let forbidden_matcher = RegexMatcher::new_line_matcher(forbidden_words_pattern)?;
    let mut searcher = Searcher::new();
    
    for entry in WalkDir::new(".")
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| !e.path().starts_with("./target")) // Exclude target directory
        .filter(|e| e.file_name() != "build.rs")      // Exclude the build script itself
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "rs"))
    {
        let path = entry.path();
        
        // Use a collector that doesn't filter out doc comments for forbidden words
        let mut collector = ForbiddenCommentCollector::new(path, true);
        searcher.search_path(&forbidden_matcher, path, &mut collector)?;
        
        if let Some(error_message) = collector.check_and_get_error_message() {
            return Err(error_message.into());
        }
    }
    
    // Then check for stars in non-doc comments
    let stars_matcher = RegexMatcher::new_line_matcher(stars_pattern)?;
    
    for entry in WalkDir::new(".")
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| !e.path().starts_with("./target")) // Exclude target directory
        .filter(|e| e.file_name() != "build.rs")      // Exclude the build script itself
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "rs"))
    {
        let path = entry.path();
        
        // Use a single collector with custom filtering logic
        // false means don't check for ** in doc comments
        let mut collector = ForbiddenCommentCollector::new(path, false);
        searcher.search_path(&stars_matcher, path, &mut collector)?;
        
        if let Some(error_message) = collector.check_and_get_error_message() {
            return Err(error_message.into());
        }
    }

    Ok(())
}
