import torch

# utils functions: ensures that intercept is increasing
def transform_intercepts(int_in):
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
    # intercepts and shift terms
    int_in = outputs['int_out']
    shift_in = outputs['shift_out']
    target_class_low = torch.argmax(targets, dim=1)
    target_class_up = target_class_low+1
    #print("target class: ", target_class_up)
    # transform intercepts
    int = transform_intercepts(int_in)
    
    # likelihood contribution for each batch sample
    if shift_in is not None:
        # sum up shift terms and flatten
        shift = torch.stack(shift_in, dim=1).sum(dim=1).view(-1)
        # target_class+1 because we start with -inf when transforming tensors
        # print("up: ", int[torch.arange(int.size(0)), target_class_up])
        # print("low: ", int[torch.arange(int.size(0)), target_class_low])
        # print("shift: ", shift)
        # print("diff: ", int[torch.arange(int.size(0)), target_class_up]-shift)
        # print("class prob: ", torch.sigmoid(int[torch.arange(int.size(0)), target_class_up]-shift))
        lli = torch.sigmoid(int[torch.arange(int.size(0)), target_class_up]-shift) - torch.sigmoid(int[torch.arange(int.size(0)), target_class_low]-shift)
    else:
        lli = torch.sigmoid(int[torch.arange(int.size(0)), target_class_up]) - torch.sigmoid(int[torch.arange(int.size(0)), target_class_low])
    nll = -torch.mean(torch.log(lli + 1e-16))
    return nll