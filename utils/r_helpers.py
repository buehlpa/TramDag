# utils/r_helpers.py


import subprocess
import tempfile
import textwrap

def fit_r_model_subprocess(target, dtype, data_path, verbose=False):
    
    # make sure that a valid R version is installed and Rscript is in PATH
    # make sure that the R packages MASS, tram, readr are installed
    # You can install them in R with:
    # install.packages(c("MASS", "tram", "readr"))
    
    # normalize dtype spelling
    dtype = dtype.lower().strip()
    if dtype in ["continous", "continuous"]:  # handle both spellings
        dtype = "continuous"
    elif dtype != "ordinal":
        raise ValueError(f"Unknown dtype: {dtype}")

    if dtype == "ordinal":
        r_code = textwrap.dedent(f"""
        library(MASS)
        library(tram)
        library(readr)

        data <- read_csv("{data_path}")
        data${target} <- factor(data${target}, ordered=TRUE)
        model <- polr({target} ~ 1, data=data, method="logistic")
        cat(model$zeta, sep="\\n")
        """)
    else:  # continuous
        r_code = textwrap.dedent(f"""
        library(MASS)
        library(tram)
        library(readr)

        data <- read_csv("{data_path}")
        model <- Colr({target} ~ 1, data=data, order=19)
        cat(model$theta, sep="\\n")
        """)

    # Write temporary R script
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(r_code)
        script_path = f.name

    if verbose:
        print("[DEBUG] R script written to:", script_path)
        print("[DEBUG] R code:\n", r_code)

    # Run Rscript
    result = subprocess.run(
        ["Rscript", script_path],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("[DEBUG] Rscript failed")
        print("[DEBUG] STDOUT:\n", result.stdout)
        print("[DEBUG] STDERR:\n", result.stderr)
        raise RuntimeError("Rscript execution failed, see STDERR above")

    if verbose:
        print("[DEBUG] Rscript succeeded")
        print("[DEBUG] STDOUT:\n", result.stdout)

    # Parse numeric output
    try:
        values = [float(x) for x in result.stdout.strip().split("\n") if x]
    except ValueError:
        print("[DEBUG] Could not parse R output as floats")
        print("[DEBUG] Raw output:", result.stdout)
        raise

    return values
