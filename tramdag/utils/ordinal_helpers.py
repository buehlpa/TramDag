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

# utils functions: ensures that intercept is increasing
def transform_intercepts_ordinal(int_in):
    # get batch size
    bs = int_in.shape[0]

    # Initialize class 0 and K as constants (on same device as input)
    int0 = torch.full((bs, 1), -float('inf'), device=int_in.device)
    intK = torch.full((bs, 1), float('inf'), device=int_in.device)

    # Reshape to match the batch size
    int1 = int_in[:, 0].reshape(bs, 1)

    # Exponentiate and accumulate the values for the transformation
    intk = torch.cumsum(torch.exp(int_in[:, 1:]), dim=1)
    # intk = torch.cumsum(torch.square(int_in[:, 1:]), dim=1)

    # Concatenate intercepts along the second axis (columns)
    int_out = torch.cat([int0, int1, int1 + intk, intK], dim=1)

    return int_out

def inverse_transform_intercepts_ordinal(int_out: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Inverse of transform_intercepts_ordinal (numerically stable).
    Recovers int_in from int_out.

    Parameters
    ----------
    int_out : torch.Tensor
        Tensor of shape (B, K+1) with -inf at [:,0] and +inf at [:,-1],
        or shape (K+1,) for a single sample.
    eps : float, optional
        Small value to clamp diffs, avoids log(0) -> -inf.

    Returns
    -------
    int_in : torch.Tensor
        Tensor of shape (B, K-1) or (K-1,) if input was 1D.
    """
    squeezed = False
    if int_out.ndim == 1:
        int_out = int_out.unsqueeze(0)  # make it (1, K+1)
        squeezed = True

    first = int_out[:, 1:2]
    diffs = int_out[:, 2:-1] - int_out[:, 1:-2]
    rest = torch.log(torch.clamp(diffs, min=eps))

    out = torch.cat([first, rest], dim=1)
    return out.squeeze(0) if squeezed else out

def ontram_nll(outputs, targets):
    int_in = outputs['int_out']
    shift_in = outputs['shift_out']
    target_class_low = torch.argmax(targets, dim=1)
    target_class_up = target_class_low + 1

    int_trans = transform_intercepts_ordinal(int_in)

    if shift_in is not None:
        shift = torch.stack(shift_in, dim=1).sum(dim=1).view(-1)
        upper = int_trans[torch.arange(int_trans.size(0)), target_class_up] - shift
        lower = int_trans[torch.arange(int_trans.size(0)), target_class_low] - shift
    else:
        upper = int_trans[torch.arange(int_trans.size(0)), target_class_up]
        lower = int_trans[torch.arange(int_trans.size(0)), target_class_low]

    p = torch.sigmoid(upper) - torch.sigmoid(lower)

    if torch.any(torch.isnan(p)):
        print(f"\n--- DEBUG ---")
        print("int_in:\n", int_in)
        print("int_trans:\n", int_trans)
        print("shift:\n", shift if shift_in is not None else "None")
        print("upper:\n", upper)
        print("lower:\n", lower)
        print("sigmoid(upper):\n", torch.sigmoid(upper))
        print("sigmoid(lower):\n", torch.sigmoid(lower))
        print("p (prob diff):\n", p)
        print("targets:\n", targets)
        print("Contains NaNs in p!", torch.isnan(p).any())
        print("--- END DEBUG ---\n")

    nll = -torch.mean(torch.log(p + 1e-16))  # tiny offset to prevent log(0)
    return nll


