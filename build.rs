use std::path::Path;
use std::process::Command;

fn main() {
    // Always rerun this script if the build script itself changes
    println!("cargo:rerun-if-changed=build.rs");

    // Scan Rust source files for underscore prefixed variables and fail if found
    if let Err(e) = scan_for_underscore_prefixes() {
        // Print error and fail the build
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn scan_for_underscore_prefixes() -> Result<(), String> {
    // Find all Rust files in the project
    let find_cmd = Command::new("find")
        .args([
            ".", // Current directory
            "-name",
            "*.rs", // Find Rust files
            "-not",
            "-path",
            "./target/*", // Exclude target directory
        ])
        .output()
        .map_err(|e| format!("Failed to run find command: {}", e))?;

    if !find_cmd.status.success() {
        return Err(format!(
            "Find command failed: {}",
            String::from_utf8_lossy(&find_cmd.stderr)
        ));
    }

    let file_list = String::from_utf8_lossy(&find_cmd.stdout);
    let files: Vec<&str> = file_list.lines().collect();

    // Regex pattern to find underscore prefixed variable names
    // This matches: `let var_with_underscore_prefix`, `fn func(var_with_underscore_prefix`, `for var_with_underscore_prefix in`, etc.
    // We use ripgrep (rg) for better performance and features
    let pattern = r"\b(let|for|(mut|ref)\s+|[,(]\s*|fn\s+\w+\s*\([^)]*?)\s+(_[a-zA-Z0-9_]+)\b";

    for file_path in files {
        let file_path = Path::new(file_path.trim());

        // Skip if file doesn't exist (should not happen normally)
        if !file_path.exists() {
            continue;
        }

        // Run ripgrep to find underscore prefixed vars
        let grep_cmd = Command::new("rg")
            .args([
                "-n",                        // Show line numbers
                "--context=0",               // No context
                "--no-heading",              // No file headers
                pattern,                     // The regex pattern
                file_path.to_str().unwrap(), // The file to check
            ])
            .output()
            .map_err(|e| format!("Failed to run ripgrep: {}", e))?;

        // If matches found, report them
        if grep_cmd.status.success() && !grep_cmd.stdout.is_empty() {
            let matches = String::from_utf8_lossy(&grep_cmd.stdout);
            let violations: Vec<_> = matches.lines().collect();

            if !violations.is_empty() {
                // Build a helpful error message with line numbers and context
                let file_name = file_path.to_str().unwrap();
                let mut error_msg = format!(
                    "\n❌ ERROR: Found {} underscore-prefixed variables in {}:\n",
                    violations.len(),
                    file_name
                );

                // Add each violation to the error message
                for violation in violations {
                    error_msg.push_str(&format!("   {}\n", violation));
                }

                error_msg.push_str(
                    "\n⚠️ Underscore-prefixed variable names are not allowed in this project.\n",
                );
                error_msg.push_str("   Either use the variable (removing the underscore) or remove it completely.\n");

                return Err(error_msg);
            }
        }
    }

    Ok(())
}
