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



import torch
import warnings
from tqdm import trange


def contram_nll(outputs, targets, min_max,return_h=False):
    """
    Args:
        outputs: dict with keys 'int_out' and 'shift_out'
        targets: shape (n,)
        min_max: tuple of two floats or tensors (min, max)
    Returns:
        scalar NLL
    """
    # Ensure min and max are not part of graph
    min_val = min_max[0].clone().detach() if isinstance(min_max[0], torch.Tensor) else torch.tensor(min_max[0], dtype=targets.dtype, device=targets.device)
    max_val = min_max[1].clone().detach() if isinstance(min_max[1], torch.Tensor) else torch.tensor(min_max[1], dtype=targets.dtype, device=targets.device)


    thetas_tilde = outputs['int_out']  # shape (n, b)
    print('thetas_tilde',thetas_tilde) # TODO remove later
    thetas = transform_intercepts_continous(thetas_tilde)
    print('thetas',thetas) # TODO remove later
    if outputs['shift_out'] is None:
        # If no shift model is provided, set Shifts to zero
        Shifts = torch.zeros_like(thetas[:, 0])
    else:
        # Shifts = torch.stack(outputs['shift_out'])  # shape (n,) old
        Shifts = torch.stack(outputs['shift_out'], dim=0)        # (n_shifts, B, 1)


    # Compute h
    h_I = h_extrapolated(thetas, targets, min_val, max_val)  # shape (n,)
    print('h_I',h_I) # TODO remove later
    #h = h_I + torch.sum(Shifts)  # shape (n,) old
    
    Shifts_sum = torch.sum(Shifts, dim=0).squeeze(-1)            # (B,)
    print('Shifts_sum',Shifts_sum) # TODO remove later
    h = h_I + Shifts_sum 
    
    # Latent logistic density log-prob
    log_latent_density = -h - 2 * torch.nn.functional.softplus(-h)  # shape (n,)

    # Derivative term (log |h'| - log(scale))
    h_dash = h_dash_extrapolated(thetas, targets, min_val, max_val)  # shape (n,)
    log_hdash = torch.log(torch.abs(h_dash)) - torch.log(max_val - min_val)  # shape (n,)

    # Final NLL
    nll = -torch.mean(log_latent_density + log_hdash)
    
    if return_h:
        return h,nll
    else:
        return nll







def transform_intercepts_continous(theta_tilde:torch.Tensor) -> torch.Tensor:
    # TODO alter docstring for continous
    # TODO assertion error for shape of theta_tilde
    """
    Transforms the unordered theta_tilde to ordered theta values for the bernstein polynomial
    E.G: 
    theta_1 = theta_tilde_1
    theta_2 = theta_tilde_1 + exp(theta_tilde_2)
    ..
    :param theta_tilde: The unordered theta_tilde values
    :return: The ordered theta values
    """

    # Compute the shift based on the last dimension size
    last_dim_size = theta_tilde.shape[-1]
    shift = torch.log(torch.tensor(2.0)) * last_dim_size / 2

    # Get the width values by applying softplus from the second position onward
    widths = torch.nn.functional.softplus(theta_tilde[..., 1:])

    # Concatenate the first value (raw) with the softplus-transformed widths
    widths = torch.cat([theta_tilde[..., [0]], widths], dim=-1)

    # Return the cumulative sum minus the shift
    return torch.cumsum(widths, dim=-1) - shift

def inverse_transform_intercepts_continous(theta: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Inverse of transform_intercepts_continous.
    Recovers theta_tilde from theta with numerical stability.
    Uses float64 internally, but returns the same dtype as input.
    """
    # Store original dtype and cast to float64 for precision
    orig_dtype = theta.dtype
    theta = theta.to(torch.float64)

    last_dim_size = theta.shape[-1]
    shift = torch.log(torch.tensor(2.0, dtype=torch.float64)) * last_dim_size / 2

    # Undo shift
    theta_shifted = theta + shift

    # Recover widths
    widths = torch.empty_like(theta_shifted)
    widths[..., 0] = theta_shifted[..., 0]
    widths[..., 1:] = theta_shifted[..., 1:] - theta_shifted[..., :-1]

    # Invert softplus (with clamp for numerical stability)
    theta_tilde = torch.empty_like(theta_shifted)
    theta_tilde[..., 0] = widths[..., 0]
    theta_tilde[..., 1:] = torch.log(torch.expm1(torch.clamp(widths[..., 1:], min=eps)))

    return theta_tilde.to(orig_dtype)


def bernstein_basis(tensor, M):
    """
    Compute the Bernstein basis polynomials for a given input tensor.
    Args:
        tensor (torch.Tensor): Input tensor of shape (n_samples).
        M (int): Degree of the Bernstein polynomial.
    Returns:
        torch.Tensor: Tensor of shape (B, Nodes, M+1) with the Bernstein basis.
    """
    tensor = torch.as_tensor(tensor)
    dtype = tensor.dtype
    M = torch.tensor(M, dtype=dtype, device=tensor.device)

    # Expand dims to allow broadcasting
    tensor_expanded = tensor.unsqueeze(-1)  # shape (B, Nodes, 1)

    # Clip values to avoid log(0)
    eps = torch.finfo(dtype).eps
    tensor_expanded = torch.clamp(tensor_expanded, min=eps, max=1 - eps)

    k_values = torch.arange(M + 1, dtype=dtype, device=tensor.device)  # shape (M+1,)
    
    # Log binomial coefficient: log(M choose k)
    log_binomial_coeff = (
        torch.lgamma(M + 1) 
        - torch.lgamma(k_values + 1) 
        - torch.lgamma(M - k_values + 1)
    )

    # Log powers
    log_powers = (
        k_values * torch.log(tensor_expanded)
        + (M - k_values) * torch.log(1 - tensor_expanded)
    )

    # Bernstein basis in log space
    log_bernstein = log_binomial_coeff + log_powers  # shape (B, Nodes, M+1)

    return torch.exp(log_bernstein)


def h_dag(targets: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
    """
    Args:
        targets: shape (n,)
        thetas: shape (n, b)
    Returns:
        Tensor: shape (n,)
    """
    _, b = thetas.shape
    B = bernstein_basis(targets, b - 1)  # shape (n, b)
    return torch.mean(B * thetas, dim=1)


def h_dag_dash(targets: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
    """
    Args:
        targets: shape (n,)
        thetas: shape (n, b)
    Returns:
        Tensor: shape (n,)
    """
    _, b = thetas.shape
    dtheta = thetas[:, 1:] - thetas[:, :-1]         # shape (n, b-1)
    B_dash = bernstein_basis(targets, b - 2)        # shape (n, b-1)
    return torch.sum(B_dash * dtheta, dim=1)


def h_extrapolated(thetas: torch.Tensor, targets: torch.Tensor, k_min: float, k_max: float) -> torch.Tensor:
    """
    Args:
        thetas: shape (n, b)
        targets: shape (n,)
        k_min: float, lower bound of scaling (not tracked in graph)
        k_max: float, upper bound of scaling (not tracked in graph)
    Returns:
        Tensor of shape (n,)
    """
    # Constants (not part of the graph)
    L_START = 0.0001
    R_START = 1.0 - L_START

    # Detach constants from graph
    L_tensor = torch.tensor(L_START, dtype=targets.dtype, device=targets.device)
    R_tensor = torch.tensor(R_START, dtype=targets.dtype, device=targets.device)

    # Scale targets
    t_i = (targets - k_min) / (k_max - k_min)  # shape (n,)
    t_i_exp = t_i.unsqueeze(-1)  # shape (n, 1)

    # Left extrapolation (t_i < L_START)
    b0 = h_dag(L_tensor.expand_as(targets), thetas).unsqueeze(-1)  # (n, 1)
    slope0 = h_dag_dash(L_tensor.expand_as(targets), thetas).unsqueeze(-1)  # (n, 1)
    h_left = slope0 * (t_i_exp - L_tensor) + b0

    # Right extrapolation (t_i > R_START)
    b1 = h_dag(R_tensor.expand_as(targets), thetas).unsqueeze(-1)  # (n, 1)
    slope1 = h_dag_dash(R_tensor.expand_as(targets), thetas).unsqueeze(-1)  # (n, 1)
    h_right = slope1 * (t_i_exp - R_tensor) + b1

    # In-domain interpolation (L_START <= t_i <= R_START)
    h_center = h_dag(t_i, thetas).unsqueeze(-1)

    # Compose the output
    h = torch.where(t_i_exp < L_tensor, h_left, h_center)
    h = torch.where(t_i_exp > R_tensor, h_right, h)

    return h.squeeze(-1)



## sampling

def h_dash_extrapolated(thetas: torch.Tensor, targets: torch.Tensor, k_min: float, k_max: float) -> torch.Tensor:
    """
    Extrapolated version of h_dag_dash for out-of-domain values.

    Args:
        thetas: shape (n, b)
        targets: shape (n,)
        k_min: float (not tracked by autograd)
        k_max: float (not tracked by autograd)

    Returns:
        Tensor: shape (n,)
    """
    # Constants
    L_START = 0.0001
    R_START = 1.0 - L_START

    L_tensor = torch.tensor(L_START, dtype=targets.dtype, device=targets.device)
    R_tensor = torch.tensor(R_START, dtype=targets.dtype, device=targets.device)

    # Scale input
    t_scaled = (targets - k_min) / (k_max - k_min)  # (n,)
    t_exp = t_scaled.unsqueeze(-1)  # (n, 1)

    # Slopes
    slope0 = h_dag_dash(L_tensor.expand_as(targets), thetas).unsqueeze(-1)  # (n, 1)
    slope1 = h_dag_dash(R_tensor.expand_as(targets), thetas).unsqueeze(-1)  # (n, 1)

    # In-domain
    h_center = h_dag_dash(t_scaled, thetas).unsqueeze(-1)  # (n, 1)

    # Initialize with center
    h_dash = h_center

    # Left extrapolation
    mask0 = t_exp < L_tensor
    h_dash = torch.where(mask0, slope0, h_dash)

    # Right extrapolation
    mask1 = t_exp > R_tensor
    h_dash = torch.where(mask1, slope1, h_dash)

    return h_dash.squeeze(-1)


def h_extrapolated_with_shift(
    thetas: torch.Tensor,
    targets: torch.Tensor,
    shifts: torch.Tensor,
    k_min: float,
    k_max: float
) -> torch.Tensor:
    """
    Args:
        thetas: shape (n, b)
        targets: shape (n,)
        shifts: shape (n,)
        k_min: float
        k_max: float
    Returns:
        Tensor of shape (n,)fit and ffit and
    """
    # Use the device of targets as the reference
    device = targets.device
    dtype = targets.dtype

    # Move inputs to the same device
    thetas = thetas.to(device)
    if shifts is None:
        shifts = torch.zeros_like(targets, dtype=dtype, device=device)
    else:
        shifts = shifts.to(device)

    # Constants and scaling
    L_START = 0.0001
    R_START = 1.0 - L_START

    L_tensor = torch.tensor(L_START, dtype=dtype, device=device)
    R_tensor = torch.tensor(R_START, dtype=dtype, device=device)
    
    if isinstance(k_min, torch.Tensor):
        k_min_tensor = k_min.clone().detach().to(dtype=dtype, device=device)
    else:
        k_min_tensor = torch.tensor(k_min, dtype=dtype, device=device)

    if isinstance(k_max, torch.Tensor):
        k_max_tensor = k_max.clone().detach().to(dtype=dtype, device=device)
    else:
        k_max_tensor = torch.tensor(k_max, dtype=dtype, device=device)


    t_i = (targets - k_min_tensor) / (k_max_tensor - k_min_tensor)  # (n,)
    t_i_exp = t_i.unsqueeze(-1)        # (n, 1)
    shifts_exp = shifts.unsqueeze(-1)  # (n, 1)

    # Left extrapolation
    b0 = h_dag(L_tensor.expand_as(targets), thetas).unsqueeze(-1) + shifts_exp
    slope0 = h_dag_dash(L_tensor.expand_as(targets), thetas).unsqueeze(-1)
    h_left = slope0 * (t_i_exp - L_tensor) + b0

    h = h_left.clone()
    mask0 = t_i_exp < L_tensor
    h = torch.where(mask0, h_left, t_i_exp)  # placeholder

    # Right extrapolation
    b1 = h_dag(R_tensor.expand_as(targets), thetas).unsqueeze(-1) + shifts_exp
    slope1 = h_dag_dash(R_tensor.expand_as(targets), thetas).unsqueeze(-1)
    h_right = slope1 * (t_i_exp - R_tensor) + b1

    mask1 = t_i_exp > R_tensor
    h = torch.where(mask1, h_right, h)

    # In-domain
    mask_mid = (t_i_exp >= L_tensor) & (t_i_exp <= R_tensor)
    h_center = h_dag(t_i, thetas).unsqueeze(-1) + shifts_exp
    h = torch.where(mask_mid, h_center, h)

    return h.squeeze(-1)


def sample_standard_logistic(shape, epsilon=1e-7, device=None):
    """
    Samples from a standard logistic distribution using inverse CDF.

    Args:
        shape (tuple or int): Shape of the output tensor.
        epsilon (float): Clipping value to avoid log(0).
        device (torch.device or str): Device for the tensor.

    Returns:
        torch.Tensor: Logistic-distributed samples with no NaNs.
    """
    uniform_samples = torch.rand(shape, device=device)
    clipped = torch.clamp(uniform_samples, epsilon, 1 - epsilon)
    logistic_samples = torch.log(clipped / (1 - clipped))

    if torch.isnan(logistic_samples).any():
        n_nan = torch.isnan(logistic_samples).sum().item()
        warnings.warn(f"{n_nan} NaNs found in logistic samples — removing them.")
        logistic_samples = logistic_samples[~torch.isnan(logistic_samples)]

    return logistic_samples

def chandrupatla_root_finder(f, low, high, max_iter=10_000, tol=1e-8):
    """
    Fully vectorized PyTorch implementation of Chandrupatla's method.
    Works on batched inputs (low/high shape: (n,))
    """
    a = low.clone()
    b = high.clone()
    c = a.clone()

    fa = f(a)
    fb = f(b)
    fc = fa.clone()

    for _ in trange(max_iter, desc="[INFO] Chandrupatla root finding ->  iterations:", leave=True):
        cond = (fb != fc) & (fa != fc)
        s_iqi = (
            (a * fb * fc) / ((fa - fb) * (fa - fc)) +
            (b * fa * fc) / ((fb - fa) * (fb - fc)) +
            (c * fa * fb) / ((fc - fa) * (fc - fb))
        )
        s_secant = b - fb * (b - a) / (fb - fa)
        s_bisect = (a + b) / 2

        s = torch.where(cond, s_iqi, s_secant)

        # Conditions to fall back to bisection
        fallback = (
            (s < torch.minimum(a, b)) |
            (s > torch.maximum(a, b)) |
            (torch.abs(s - b) >= torch.abs(b - c) / 2) |
            (torch.abs(b - c) < tol)
        )
        s = torch.where(fallback, s_bisect, s)

        fs = f(s)
        c, fc = b.clone(), fb.clone()

        use_lower = (fa * fs < 0)
        b = torch.where(use_lower, s, b)
        fb = torch.where(use_lower, fs, fb)
        a = torch.where(use_lower, a, s)
        fa = torch.where(use_lower, fa, fs)

        converged = (torch.abs(b - a) < tol) | (torch.abs(fs) < tol)
        if converged.all():
            break

    return (a + b) / 2


def vectorized_object_function( thetas: torch.Tensor,targets: torch.Tensor, shifts: torch.Tensor,
                               latent_sample: torch.Tensor, k_min: float, k_max: float) -> torch.Tensor:
    # h(xj)-latent_sample=0 , solve for xj
    return h_extrapolated_with_shift(thetas, targets, shifts, k_min, k_max) - latent_sample


def bisection_root_finder(f, low, high, max_iter=1000, tol=1e-7):
    """
    Pure PyTorch scalar root-finder (element-wise bisection).
    f: function that accepts tensor of shape (n,)
    low, high: tensors of shape (n,)
    Returns: tensor of shape (n,)
    
    
    https://en.wikipedia.org/wiki/Bisection_method
    Iteration tasks
    The input for the method is a continuous function f, an interval [a, b], and the function values f(a) and f(b). The function values are of opposite sign (there is at least one zero crossing within the interval). Each iteration performs these steps:

    1.Calculate c, the midpoint of the interval, c = ⁠/a + b)/2
    
    2.Calculate the function value at the midpoint, f(c).
    3. If convergence is satisfactory (that is, c - a is sufficiently small, or |f(c)| is sufficiently small), return c and stop iterating.
    Examine the sign of f(c) and replace either (a, f(a)) or (b, f(b)) with (c, f(c)) so that there is a zero crossing within the new interval.
        
        
        
    
    e.g.:
    
    root = bisection_root_finder(
    lambda targets: vectorized_object_function(
        thetas_expanded,
        targets,
        shifts,
        latent_sample,
        k_min=min_max[0],
        k_max=min_max[1]
    ),
    low,
    high
    )

    
    """
    for _ in trange(max_iter, desc="Bisection root finding", leave=True):
        mid = (low + high) / 2.0
        f_mid = f(mid)
        f_low = f(low)

        # Where sign changes, root is between low and mid; else mid and high
        sign_change = f_low * f_mid < 0
        high = torch.where(sign_change, mid, high)
        low = torch.where(sign_change, low, mid)

        if torch.max(torch.abs(high - low)) < tol:
            break

    return (low + high) / 2.0


