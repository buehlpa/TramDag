
from loss_ordinal import transform_intercepts
import torch


def get_cdf(outputs):
    """"
    Get cumulative distribution function
    
    Args:
        outputs: output of a model of class OntramModel
    """
    int_in = outputs['int_out']
    shift_in = outputs['shift_out']
    
    # transform intercepts
    int = transform_intercepts(int_in)

    if shift_in is not None:
        shift = torch.stack(shift_in, dim=1).sum(dim=1)    
        cdf = torch.sigmoid(torch.sub(int, shift))
    else:
        cdf = torch.sigmoid(int)
    return cdf

def get_pdf(cdf):
    """"
    Get probability density function
    
    Args:
        cdf: cumulative distirbution function returning from get_cdf
    """
    return torch.sub(cdf[:,1:], cdf[:,:-1])

def pred_proba(pdf, targets):
    """"
    Get probability for the true class
    
    Args:
        pdf: probability density function returning from get_pdf
        targets: Outcome classes, one hot encoded
    """
    target_class = torch.argmax(targets, dim=1)
    proba = pdf[torch.arange(pdf.shape[0]), target_class]
    return proba

def pred_class(pdf):
    """"
    Get the predicted class.
    
    Args:
        pdf: probability density function returning from get_pdf
    """
    return torch.argmax(pdf, dim=1)