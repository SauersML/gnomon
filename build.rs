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

        error_msg.push_str("\n⚠️ Comments containing 'FIXED', 'CORRECTED', or 'FIX' are not allowed in this project.\n");
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
    let pattern = r"\b(let|for|(mut|ref)\s+|[,(]\s*|fn\s+\w+\s*\([^)]*?)\s+(_[a-zA-Z0-9_]+)\b";
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

        // Create a new collector for each file.
        let mut collector = ViolationCollector::new(path);

        // Search the file using our regex matcher and collector sink.
        searcher.search_path(&matcher, path, &mut collector)?;

        // After searching the file, check if our collector found anything.
        if let Some(error_message) = collector.check_and_get_error_message() {
            // If violations were found, return the formatted error message immediately.
            // This will cause the build to fail.
            return Err(error_message.into());
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
    
    // Single pattern to check for all forbidden patterns in all types of comments
    // We'll filter doc comments with ** in the Sink implementation
    let comment_pattern = r"(//|/\*).*\b(FIXED|CORRECTED|FIX)\b|(//|/\*).*\*\*";
    
    let mut searcher = Searcher::new();
    
    // Compile the pattern
    let comment_matcher = RegexMatcher::new_line_matcher(comment_pattern)?;

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
        searcher.search_path(&comment_matcher, path, &mut collector)?;
        
        if let Some(error_message) = collector.check_and_get_error_message() {
            return Err(error_message.into());
        }
    }

    Ok(())
}
