import torch


def contram_nll(outputs, targets, min_max):
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
    thetas = transform_intercepts_continous(thetas_tilde)

    if outputs['shift_out'] is None:
        # If no shift model is provided, set Shifts to zero
        Shifts = torch.zeros_like(thetas[:, 0])
    else:
        Shifts = outputs['shift_out']  # shape (n,)
    

    # Compute h
    h_I = h_extrapolated(thetas, targets, min_val, max_val)  # shape (n,)
    h = h_I + torch.sum(Shifts)  # shape (n,)

    # Latent logistic density log-prob
    log_latent_density = -h - 2 * torch.nn.functional.softplus(-h)  # shape (n,)

    # Derivative term (log |h'| - log(scale))
    h_dash = h_dash_extrapolated(thetas, targets, min_val, max_val)  # shape (n,)
    log_hdash = torch.log(torch.abs(h_dash)) - torch.log(max_val - min_val)  # shape (n,)

    # Final NLL
    nll = -torch.mean(log_latent_density + log_hdash)

    return nll



def transform_intercepts_continous(theta_tilde:torch.Tensor) -> torch.Tensor:
    
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
    R_START = 1.0-L_START


    # Detach constants from graph
    L_tensor = torch.tensor(L_START, dtype=targets.dtype, device=targets.device)
    R_tensor = torch.tensor(R_START, dtype=targets.dtype, device=targets.device)

    # Scale targets
    t_i = (targets - k_min) / (k_max - k_min)  # shape (n,)
    t_i_exp = t_i.unsqueeze(-1)  # shape (n, 1)

    # Extrapolation at left (t_i < 0)
    b0 = h_dag(L_tensor.expand_as(targets), thetas).unsqueeze(-1)     # (n, 1)
    slope0 = h_dag_dash(L_tensor.expand_as(targets), thetas).unsqueeze(-1)  # (n, 1)
    h_left = slope0 * (t_i_exp - L_tensor) + b0

    # Start with placeholder
    h = h_left.clone()

    # Mask for left extrapolation
    mask0 = t_i_exp < L_tensor
    h = torch.where(mask0, h_left, t_i_exp)  # placeholder fill

    # Extrapolation at right (t_i > 1)
    b1 = h_dag(R_tensor.expand_as(targets), thetas).unsqueeze(-1)
    slope1 = h_dag_dash(R_tensor.expand_as(targets), thetas).unsqueeze(-1)
    h_right = slope1 * (t_i_exp - R_tensor) + b1

    mask1 = t_i_exp > R_tensor
    h = torch.where(mask1, h_right, h)

    # In-domain: t_i ∈ [0,1]
    mask_mid = (t_i_exp >= L_tensor) & (t_i_exp <= R_tensor)
    h_center = h_dag(t_i, thetas).unsqueeze(-1)
    h = torch.where(mask_mid, h_center, h)

    return h.squeeze(-1)


def h_dash_extrapolated(thetas: torch.Tensor, targets: torch.Tensor, k_min: float, k_max: float) -> torch.Tensor:
    """
    Extrapolated version of h_dag_dash for out-of-domain values.
    
    Args:
        t_targetsi: shape (n,)
        thetas: shape (n, b)
        k_min: float (not tracked by autograd)
        k_max: float (not tracked by autograd)
    
    Returns:
        Tensor: shape (n,)
    """
    # Constants
    L_START = 0.0001
    R_START = 1.0-L_START

    # Detach constants from graph
    L_tensor = torch.tensor(L_START, dtype=targets.dtype, device=targets.device)
    R_tensor = torch.tensor(R_START, dtype=targets.dtype, device=targets.device)

    # Scale input
    t_scaled = (targets - k_min) / (k_max - k_min)
    t_exp = t_scaled.unsqueeze(-1)  # shape (n, 1)

    # Left extrapolation: constant slope at L_START
    slope0 = h_dag_dash(L_tensor.expand_as(targets), thetas).unsqueeze(-1)  # (n, 1)
    mask0 = t_exp < L_tensor
    h_dash = torch.where(mask0, slope0, t_exp)  # placeholder init

    # Right extrapolation: constant slope at R_START
    slope1 = h_dag_dash(R_tensor.expand_as(targets), thetas).unsqueeze(-1)  # (n, 1)
    mask1 = t_exp > R_tensor
    h_dash = torch.where(mask1, slope1, h_dash)

    # In-domain interpolation
    mask_mid = (t_exp >= L_tensor) & (t_exp <= R_tensor)
    h_center = h_dag_dash(t_scaled, thetas).unsqueeze(-1)  # (n, 1)
    h_dash = torch.where(mask_mid, h_center, h_dash)

    return h_dash.squeeze(-1)  # shape (n,)




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
    k_min_tensor = torch.tensor(k_min, dtype=dtype, device=device)
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


