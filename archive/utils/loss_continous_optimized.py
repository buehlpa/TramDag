import torch
import torch.nn.functional as F


# ---- Cached Bernstein ----
class BernsteinCache:
    def __init__(self, max_degree: int, device: torch.device):
        self.cache = {}
        self.device = device
        self.max_degree = max_degree

    def get(self, M: int):
        if M in self.cache:
            return self.cache[M]

        k = torch.arange(M + 1, dtype=torch.float32, device=self.device)
        log_binom = torch.lgamma(torch.tensor(M + 1.0, device=self.device)) \
                   - torch.lgamma(k + 1) - torch.lgamma(M - k + 1)
        self.cache[M] = log_binom
        return log_binom


def bernstein_basis_fast(tensor, M: int, log_binom):
    tensor = tensor.unsqueeze(-1)
    eps = torch.finfo(tensor.dtype).eps
    tensor = torch.clamp(tensor, min=eps, max=1 - eps)

    k = torch.arange(M + 1, dtype=tensor.dtype, device=tensor.device)
    log_powers = k * torch.log(tensor) + (M - k) * torch.log(1 - tensor)

    return torch.exp(log_binom + log_powers)


def transform_intercepts_continous(theta_tilde):
    shift = torch.log(torch.tensor(2.0)) * theta_tilde.shape[-1] / 2
    widths = F.softplus(theta_tilde[..., 1:])
    widths = torch.cat([theta_tilde[..., [0]], widths], dim=-1)
    return torch.cumsum(widths, dim=-1) - shift


def h_dag_fast(targets, thetas, log_binom):
    B = bernstein_basis_fast(targets, thetas.shape[1] - 1, log_binom)
    return torch.sum(B * thetas, dim=1)


def h_dag_dash_fast(targets, thetas, log_binom):
    dtheta = thetas[:, 1:] - thetas[:, :-1]
    B_dash = bernstein_basis_fast(targets, thetas.shape[1] - 2, log_binom)
    return torch.sum(B_dash * dtheta, dim=1)


def h_extrapolated_fast(thetas, targets, k_min, k_max, log_binom, log_binom_dash):
    L = 0.0001
    R = 1.0 - L

    t = (targets - k_min) / (k_max - k_min)
    t_exp = t.unsqueeze(-1)

    L_t = torch.full_like(targets, L)
    R_t = torch.full_like(targets, R)

    b0 = h_dag_fast(L_t, thetas, log_binom).unsqueeze(-1)
    s0 = h_dag_dash_fast(L_t, thetas, log_binom_dash).unsqueeze(-1)
    h_left = s0 * (t_exp - L) + b0

    b1 = h_dag_fast(R_t, thetas, log_binom).unsqueeze(-1)
    s1 = h_dag_dash_fast(R_t, thetas, log_binom_dash).unsqueeze(-1)
    h_right = s1 * (t_exp - R) + b1

    h_center = h_dag_fast(t, thetas, log_binom).unsqueeze(-1)

    h = torch.where(t_exp < L, h_left, h_center)
    h = torch.where(t_exp > R, h_right, h)

    return h.squeeze(-1)


def h_dash_extrapolated_fast(thetas, targets, k_min, k_max, log_binom_dash):
    L = 0.0001
    R = 1.0 - L

    t = (targets - k_min) / (k_max - k_min)
    t_exp = t.unsqueeze(-1)

    L_t = torch.full_like(targets, L)
    R_t = torch.full_like(targets, R)

    s0 = h_dag_dash_fast(L_t, thetas, log_binom_dash).unsqueeze(-1)
    s1 = h_dag_dash_fast(R_t, thetas, log_binom_dash).unsqueeze(-1)
    s_mid = h_dag_dash_fast(t, thetas, log_binom_dash).unsqueeze(-1)

    h_dash = torch.where(t_exp < L, s0, s_mid)
    h_dash = torch.where(t_exp > R, s1, h_dash)

    return h_dash.squeeze(-1)


def contram_nll_optimized(outputs, targets, min_max, return_h=False):
    device = targets.device
    min_val = min_max[0].detach() if isinstance(min_max[0], torch.Tensor) else torch.tensor(min_max[0], device=device)
    max_val = min_max[1].detach() if isinstance(min_max[1], torch.Tensor) else torch.tensor(min_max[1], device=device)

    thetas_tilde = outputs['int_out']
    thetas = transform_intercepts_continous(thetas_tilde)

    if outputs['shift_out'] is None:
        shifts = torch.zeros_like(thetas[:, 0])
    else:
        shifts = torch.stack(outputs['shift_out'], dim=0).sum(dim=0).squeeze(-1)

    cache = BernsteinCache(max_degree=thetas.shape[1], device=device)
    log_binom = cache.get(thetas.shape[1] - 1)
    log_binom_dash = cache.get(thetas.shape[1] - 2)

    h_I = h_extrapolated_fast(thetas, targets, min_val, max_val, log_binom, log_binom_dash)
    h = h_I + shifts

    log_latent = -h - 2 * F.softplus(-h)

    h_dash = h_dash_extrapolated_fast(thetas, targets, min_val, max_val, log_binom_dash)
    log_hdash = torch.log(torch.abs(h_dash)) - torch.log(max_val - min_val)

    nll = -torch.mean(log_latent + log_hdash)
    return (h, nll) if return_h else nll
