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




def get_cdf_ordinal(outputs):
    """"
    Get cumulative distribution function
    
    Args:
        outputs: output of a model of class OntramModel
    """
    int_in = outputs['int_out']
    shift_in = outputs['shift_out']
    
    # transform intercepts
    int = transform_intercepts_ordinal(int_in)

    if shift_in is not None:
        shift = torch.stack(shift_in, dim=1).sum(dim=1)    
        cdf = torch.sigmoid(torch.sub(int, shift))
    else:
        cdf = torch.sigmoid(int)
    return cdf

def get_pdf_ordinal(cdf):
    """"
    Get probability density function
    
    Args:
        cdf: cumulative distirbution function returning from get_cdf
    """
    return torch.sub(cdf[:,1:], cdf[:,:-1])