
import subprocess
import tempfile
import textwrap

def fit_r_model_subprocess(target, dtype,theta_count, data_path, debug=False):
    
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
        model <- Colr({target} ~ 1, data=data, order={theta_count-1})
        cat(model$theta, sep="\\n")
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

    # Parse numeric output
    try:
        values = [float(x) for x in result.stdout.strip().split("\n") if x]
    except ValueError:
        print("[DEBUG] Could not parse R output as floats")
        print("[DEBUG] Raw output:", result.stdout)
        raise

    return values
