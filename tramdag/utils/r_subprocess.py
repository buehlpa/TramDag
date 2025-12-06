
"""
Copyright 2025 Zurich University of Applied Sciences (ZHAW)
Pascal Buehler, Beate Sick, Oliver Duerr

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# This file contains helper functions for R model fitting via subprocess calls.

import subprocess
import tempfile
import textwrap
import shutil
import os

def check_r_setup():
    """
    Verify that Rscript is available and that required R packages are installed.
    Raises RuntimeError if any component is missing.
    """

    if shutil.which("Rscript") is None:
        raise RuntimeError("Rscript not found. Please install R >= 4.0 and ensure it’s on PATH.")

    required_packages = ["tram", "ordinal", "readr", "MASS"]
    missing = []

    for pkg in required_packages:
        cmd = [
            "Rscript",
            "-e",
            f"if(!requireNamespace('{pkg}', quietly=TRUE)) quit(status=1)"
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            missing.append(pkg)

    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            f"Missing R packages: {missing_str}. "
            f"Run in R:\ninstall.packages(c({', '.join(repr(x) for x in missing)}))"
        )

def fit_r_model_subprocess(target, dtype,theta_count, data_path,ls_shift_covariates=None, debug=False):
    
    """
    Fit a simple R model (POLR for ordinal or COLR for continuous outcomes)
    using a subprocess call to Rscript and return the estimated intercept/theta
    parameters as floats.

    This function:
    - Generates an R script dynamically for the specified target variable.
    - Writes the script to a temporary file.
    - Executes the script via `Rscript` as a subprocess.
    - Parses the resulting coefficients printed by the R model into Python floats.

    Parameters
    ----------
    target : str
        Name of the target column in the CSV file.
    dtype : str
        Type of the target variable. Must be one of:
          - "ordinal" (fits an ordered logistic regression using `MASS::polr`)
          - "continuous" or "continous" (fits a COLR model using `tram::Colr`)
    data_path : str
        Path to the CSV file containing the dataset. Must be readable by
        `readr::read_csv` in R.
    debug : bool, optional (default=False)
        If True, prints debug information including the generated R script,
        the Rscript command output, and intermediate states.

    Returns
    -------
    list of float
        Extracted numeric parameters (intercepts/thetas) from the fitted R model.

    Raises
    ------
    ValueError
        If `dtype` is not recognized.
    RuntimeError
        If the R subprocess fails to execute successfully.
    ValueError
        If the R output cannot be parsed into floats.

    Notes
    -----
    - Requires a valid R installation with `Rscript` available in the PATH.
    - The following R packages must be installed:
        * MASS
        * tram
        * readr
    - For ordinal targets, the function fits a logistic regression using
      `MASS::polr` and extracts `zeta`.
    - For continuous targets, the function fits a `tram::Colr` model and
      extracts `theta`.

    Examples
    --------
    >>> values = fit_r_model_subprocess(
    ...     target="y",
    ...     dtype="ordinal",
    ...     data_path="data/train.csv",
    ...     debug=True
    ... )
    >>> print(values)
    [-0.42, 0.15, 0.87]
    """
    check_r_setup()
    
    data_path = os.path.abspath(data_path)
    # R is fine with forward slashes even on Windows
    data_path = data_path.replace("\\", "/")
    
    
    dtype = dtype.lower().strip()
    if dtype in ["continous", "continuous"]:  # handle both spellings
        dtype = "continuous"
    elif dtype != "ordinal":
        raise ValueError(f"Unknown dtype: {dtype}")

    if ls_shift_covariates:
        # assume shift_covariates is a list of valid column names in the CSV
        rhs = " + ".join(ls_shift_covariates)
    else:
        rhs = "1"

    if dtype == "ordinal":
        if theta_count < 2:
            # glm fallback: intercept + slopes in coef(model)
            r_code = textwrap.dedent(f"""
            library(MASS)
            library(tram)
            library(readr)
            data <- read_csv("{data_path}")
            data${target} <- as.numeric(as.factor(data${target})) - 1
            form <- as.formula(paste0("`{target}` ~ {rhs}"))
            model <- glm(form, data=data, family=binomial(link="logit"))

            coefs <- coef(model)
            intercept <- coefs[1]
            shifts <- if (length(coefs) > 1) coefs[-1] else numeric(0)

            cat("INTERCEPTS\\n")
            cat(intercept, sep="\\n")
            cat("\\nSHIFTS\\n")
            if (length(shifts) > 0) cat(shifts, sep="\\n")
            """)
        
        else:
            # polr: zeta = intercept(s), coefficients = slopes
            r_code = textwrap.dedent(f"""
            library(MASS)
            library(tram)
            library(readr)
            data <- read_csv("{data_path}")
            data${target} <- factor(data${target}, ordered=TRUE)
            form <- as.formula(paste0("`{target}` ~ {rhs}"))
            model <- polr(form, data=data, method="logistic")

            zeta <- model$zeta
            shifts <- model$coefficients

            cat("INTERCEPTS\\n")
            cat(zeta, sep="\\n")
            cat("\\nSHIFTS\\n")
            if (length(shifts) > 0) cat(shifts, sep="\\n")
            """)
            
    else:  # continuous
        r_code = textwrap.dedent(f"""
        library(MASS)
        library(tram)
        library(readr)

        data <- read_csv("{data_path}")
        form <- as.formula(paste0("`{target}` ~ {rhs}"))
        model <- Colr(form, data=data, order={theta_count-1})

        theta_all <- model$theta
        if (length(theta_all) < {theta_count}) {{
            stop("model$theta shorter than theta_count")
        }}

        intercepts <- theta_all[seq_len({theta_count})]
        shifts <- if (length(theta_all) > {theta_count}) theta_all[-seq_len({theta_count})] else numeric(0)

        cat("INTERCEPTS\\n")
        cat(intercepts, sep="\\n")
        cat("\\nSHIFTS\\n")
        if (length(shifts) > 0) cat(shifts, sep="\\n")
        """)

    # Write temporary R script
    with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
        f.write(r_code)
        script_path = f.name

    if debug:
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

    if debug:
        print("[DEBUG] Rscript succeeded")
        print("[DEBUG] STDOUT:\n", result.stdout)

    # Parse numeric output into intercepts and shifts
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    if debug:
        print("[DEBUG] Parsed R output lines:", lines)

    intercepts = []
    shifts = []

    if "INTERCEPTS" in lines:
        idx_int = lines.index("INTERCEPTS")
        idx_shifts = lines.index("SHIFTS") if "SHIFTS" in lines else len(lines)

        intercept_lines = lines[idx_int + 1:idx_shifts]
        shift_lines = lines[idx_shifts + 1:] if "SHIFTS" in lines else []

        try:
            intercepts = [float(x) for x in intercept_lines]
            shifts = [float(x) for x in shift_lines] if shift_lines else []
        except ValueError:
            print("[DEBUG] Could not parse R output as floats")
            print("[DEBUG] Raw output:", result.stdout)
            raise
    else:
        # backward compatibility: everything is treated as intercepts
        try:
            intercepts = [float(x) for x in lines]
            shifts = []
        except ValueError:
            print("[DEBUG] Could not parse R output as floats")
            print("[DEBUG] Raw output:", result.stdout)
            raise

    return intercepts, shifts
