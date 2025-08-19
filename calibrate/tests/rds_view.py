from pathlib import Path
import shutil
import subprocess
import sys

INPUT = Path("gam_model_fit.rds")
OUTPUT = Path("gam_model_fit.R")

def main():
    """Converts an RDS file to a sourceable R script using R's dput()."""
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

    # R code: read the RDS, and write a single assignment using dput().
    # The 'width' option is set to its maximum allowed value (10000) to minimize
    # unwanted line breaks in the dput() output.
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
        # check=True will raise CalledProcessError if Rscript returns a non-zero exit code.
        result = subprocess.run(
            [rscript, "-e", r_code],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        # If Rscript fails, print its detailed error message for easier debugging.
        sys.stderr.write("Error: Rscript failed to produce dput output.\n")
        sys.stderr.write(f"--- Rscript stderr ---\n{e.stderr}\n")
        sys.exit(3)

    if not OUTPUT.exists() or OUTPUT.stat().st_size == 0:
        sys.stderr.write(f"Error: Failed to write '{OUTPUT}' (file is empty or missing).\n")
        sys.exit(4)

    print(f"Successfully wrote {OUTPUT} (plain-text R code).")
    print("You can open it in a text editor or run `source('gam_model_fit.R')` in R.")

if __name__ == "__main__":
    main()
