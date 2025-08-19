from pathlib import Path
import shutil
import subprocess
import sys
import re

INPUT = Path("gam_model_fit.rds")
OUTPUT = Path("gam_model_fit.R")
DECIMAL_PLACES = 3

def main():
    """Converts an RDS file to a rounded, sourceable R script."""
    if not INPUT.exists():
        sys.stderr.write(f"Error: Input file '{INPUT}' not found in {Path('.').resolve()}\n")
        sys.exit(1)

    rscript = shutil.which("Rscript")
    if not rscript:
        sys.stderr.write(
            "Error: Rscript not found. Please install R (https://cran.r-project.org/) "
            "and ensure `Rscript` is on your PATH.\n"
        )
        sys.exit(2)

    # Step 1: Generate the full-precision R file using Rscript.
    # -----------------------------------------------------------
    print(f"Generating full-precision R code from '{INPUT}'...")

    # R code to read the RDS and write a single assignment using dput().
    r_code = f"""
options(digits = 17, width = 10000)
obj <- readRDS("{INPUT}")
sink("{OUTPUT}")
cat("gam_model_fit <- ")
dput(obj)
cat("\\n")
sink()
"""

    try:
        # Pass the R code directly to Rscript and capture its output.
        subprocess.run(
            [rscript, "-e", r_code],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        sys.stderr.write("Error: Rscript failed to produce dput output.\n")
        sys.stderr.write(f"--- Rscript stderr ---\n{e.stderr}\n")
        sys.exit(3)

    if not OUTPUT.exists() or OUTPUT.stat().st_size == 0:
        sys.stderr.write(f"Error: Rscript failed to write '{OUTPUT}'.\n")
        sys.exit(4)

    print(f"Successfully generated intermediate file '{OUTPUT}'.")

    # Step 2: Post-process the generated file to round the floats.
    # -------------------------------------------------------------
    print(f"Rounding all floating-point numbers in '{OUTPUT}' to {DECIMAL_PLACES} decimal places...")

    try:
        content = OUTPUT.read_text()

        # This function will be called for every number found by the regex.
        def round_match(match: re.Match) -> str:
            """Converts a matched float string, rounds it, and formats it."""
            number_str = match.group(0)
            rounded_num = round(float(number_str), DECIMAL_PLACES)
            # Format to ensure trailing zeros (e.g., 2.5 -> "2.500")
            return f"{rounded_num:.{DECIMAL_PLACES}f}"

        # Regex to find floating point numbers. It looks for:
        # - An optional sign (+ or -)
        # - Digits, a decimal point, and more digits
        # - Optional scientific notation (e.g., e-05)
        float_pattern = re.compile(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?")

        # Find all floats and replace them with their rounded versions.
        rounded_content = float_pattern.sub(round_match, content)

        OUTPUT.write_text(rounded_content)

    except Exception as e:
        sys.stderr.write(f"Error: Failed during the rounding process.\n{e}\n")
        sys.exit(5)


    print("\n--- Success! ---")
    print(f"Wrote rounded R code to '{OUTPUT}'.")
    print("You can open it in a text editor or run `source('gam_model_fit.R')` in R.")

if __name__ == "__main__":
    main()
