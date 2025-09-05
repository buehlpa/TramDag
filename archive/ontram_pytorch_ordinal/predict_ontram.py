import torch
import numpy as np

from .helper_predict import get_cdf, get_pdf, pred_proba, pred_class, get_parameters_si

def predict_ontram(model, test_loader, checkpoint_path=None, device='cuda', output=None, si=True):
    """
    Fit function uniting simple intercept and complex intercept terms 
    
    Args:
        model: model of class OntramModel
        test_loader: torch test loader containing either (tabular_data, outcome) or (tabular_data, image_data, outcome)
        checkpoint_path: Path from where to load the model weights
        device: "cpu" or "cuda" 
        output: prob: predicted probability for the true class, 
                class: predicted class
                all: probability, class, PDF, CDF and parameter estimates
    """

    model = model
    test_loader = test_loader
    checkpoint_path = checkpoint_path
    device = device
    if output is None:
        output = 'all'
    else:
        output = output

    def predict_ontram_si(model, test_loader, checkpoint_path, device, output):
        """
        Function for predicting with a trained simple intercept OntramModel with weights stored under checkpoint_path. 

        Args:
            model: model of class OntramModel
            test_loader: torch test loader containing either (tabular_data, outcome) or (tabular_data, image_data, outcome)
            checkpoint_path: Path from where to load the model weights
            device: "cpu" or "cuda" 
            output: prob: predicted probability for the true class, 
                    class: predicted class
                    all: probability, class, PDF, CDF and parameter estimates
        """

        # model to device
        model.to(device)

        # load model if we have saved a checkpoint
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # init lists to store results
        cdf_all = []
        pdf_all = []
        proba_all = []
        class_all = []
        params_all = []

        # No shift term -----------------------------------------------------------------------
        if model.nn_shift is None:
            with torch.no_grad(): 
                for y_te in test_loader:
                    y_te = y_te[0]
                    y_te = y_te.to(device)

                    # Forward pass
                    int_te = torch.from_numpy(np.ones(shape=[len(y_te),1])).float()
                    int_te = int_te.to(device)
                    out = model(int_te)

                    cdf = get_cdf(out)
                    pdf = get_pdf(cdf)
                    cdf_all.append(cdf)
                    pdf_all.append(pdf)
                    proba_all.append(pred_proba(pdf, y_te))
                    class_all.append(pred_class(pdf))

                # Concatenate all predictions into a single tensor
                cdf_all = torch.cat(cdf_all, dim=0)
                pdf_all = torch.cat(pdf_all, dim=0)
                proba_all = torch.cat(proba_all, dim=0)
                class_all = torch.cat(class_all, dim=0)
                params_all = get_parameters_si(model)

        # One shift term -----------------------------------------------------------------------
        elif len(model.nn_shift)==1:
            with torch.no_grad(): 
                for x_te, y_te in test_loader:
                    x_te, y_te = x_te.to(device), y_te.to(device)

                    # Forward pass
                    int_te = torch.from_numpy(np.ones(shape=[len(y_te),1])).float()
                    int_te = int_te.to(device)
                    out = model(int_te, [x_te])

                    cdf = get_cdf(out)
                    pdf = get_pdf(cdf)
                    cdf_all.append(cdf)
                    pdf_all.append(pdf)
                    proba_all.append(pred_proba(pdf, y_te))
                    class_all.append(pred_class(pdf))

                # Concatenate all predictions into a single tensor
                cdf_all = torch.cat(cdf_all, dim=0)
                pdf_all = torch.cat(pdf_all, dim=0)
                proba_all = torch.cat(proba_all, dim=0)
                class_all = torch.cat(class_all, dim=0)
                params_all = get_parameters_si(model)

        # Two shift terms -----------------------------------------------------------------------
        elif len(model.nn_shift)==2:
            with torch.no_grad(): 
                for x_te1, x_te2, y_te in test_loader:
                    x_te1, x_te2, y_te = x_te1.to(device), x_te2.to(device), y_te.to(device)

                    # Forward pass
                    int_te = torch.from_numpy(np.ones(shape=[len(y_te),1])).float()
                    int_te = int_te.to(device)
                    out = model(int_te, [x_te1, x_te2])  

                    cdf = get_cdf(out)
                    pdf = get_pdf(cdf)
                    cdf_all.append(cdf)
                    pdf_all.append(pdf)
                    proba_all.append(pred_proba(pdf, y_te))
                    class_all.append(pred_class(pdf))

                # Concatenate all predictions into a single tensor
                cdf_all = torch.cat(cdf_all, dim=0)
                pdf_all = torch.cat(pdf_all, dim=0)
                proba_all = torch.cat(proba_all, dim=0)
                class_all = torch.cat(class_all, dim=0)
                params_all = get_parameters_si(model)

        # Three shift terms -----------------------------------------------------------------------
        elif len(model.nn_shift)==3:
            with torch.no_grad(): 
                for x_te1, x_te2, x_te3, y_te in test_loader:
                    x_te1, x_te2, x_te3, y_te = x_te1.to(device), x_te2.to(device), x_te3.to(device), y_te.to(device)

                    # Forward pass
                    int_te = torch.from_numpy(np.ones(shape=[len(y_te),1])).float()
                    int_te = int_te.to(device)
                    out = model(int_te, [x_te1, x_te2, x_te3])  

                    cdf = get_cdf(out)
                    pdf = get_pdf(cdf)
                    cdf_all.append(cdf)
                    pdf_all.append(pdf)
                    proba_all.append(pred_proba(pdf, y_te))
                    class_all.append(pred_class(pdf))

                # Concatenate all predictions into a single tensor
                cdf_all = torch.cat(cdf_all, dim=0)
                pdf_all = torch.cat(pdf_all, dim=0)
                proba_all = torch.cat(proba_all, dim=0)
                class_all = torch.cat(class_all, dim=0)
                params_all = get_parameters_si(model)

        # More than three data modalities ----------------------------------------------------
        elif len(model.nn_shift)>3:
            raise ValueError("More than three data modalities provided.")

        # Output -----------------------------------------------------------------------------

        # Transform tensors into numpy
        proba_all = proba_all.cpu().numpy()
        class_all = class_all.cpu().numpy()
        pdf_all = pdf_all.cpu().numpy()
        cdf_all = cdf_all.cpu().numpy()
        # params_all = params_all.cpu().numpy()

        if output=='prob':
            return proba_all
        if output=='class':
            return class_all
        if output=='all':
            return {'prob': proba_all, 'class': class_all, 'pdf': pdf_all, 'cdf': cdf_all, 'params': params_all}



    def predict_ontram_ci(model, test_loader, checkpoint_path, device, output):
        """
        Function for predicting with a trained simple intercept OntramModel with weights stored under checkpoint_path. 

        Args:
            model: model of class OntramModel
            test_loader: torch test loader containing either (tabular_data, outcome) or (tabular_data, image_data, outcome)
            checkpoint_path: Path from where to load the model weights
            device: "cpu" or "cuda" 
            output: prob: predicted probability for the true class, 
                    class: predicted class
                    all: probability, class, PDF, CDF and parameter estimates
        """

        # model to device
        model.to(device)

        # load model if we have saved a checkpoint
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # init lists to store results
        cdf_all = []
        pdf_all = []
        proba_all = []
        class_all = []
        params_all = []

        # No shift term -----------------------------------------------------------------------
        if model.nn_shift is None:
            with torch.no_grad(): 
                for x_te, y_te in test_loader:
                    x_te, y_te = x_te.to(device), y_te.to(device)

                    # Forward pass
                    out = model(x_te)

                    cdf = get_cdf(out)
                    pdf = get_pdf(cdf)
                    cdf_all.append(cdf)
                    pdf_all.append(pdf)
                    proba_all.append(pred_proba(pdf, y_te))
                    class_all.append(pred_class(pdf))

                # Concatenate all predictions into a single tensor
                cdf_all = torch.cat(cdf_all, dim=0)
                pdf_all = torch.cat(pdf_all, dim=0)
                proba_all = torch.cat(proba_all, dim=0)
                class_all = torch.cat(class_all, dim=0)
                params_all = get_parameters_si(model)

        # One shift term -----------------------------------------------------------------------
        elif len(model.nn_shift)==1:
            with torch.no_grad(): 
                for x_te1, x_te2, y_te in test_loader:
                    x_te1, x_te2, y_te = x_te1.to(device), x_te2.to(device), y_te.to(device)

                    # Forward pass
                    out = model(x_te1, [x_te2])

                    cdf = get_cdf(out)
                    pdf = get_pdf(cdf)
                    cdf_all.append(cdf)
                    pdf_all.append(pdf)
                    proba_all.append(pred_proba(pdf, y_te))
                    class_all.append(pred_class(pdf))

                # Concatenate all predictions into a single tensor
                cdf_all = torch.cat(cdf_all, dim=0)
                pdf_all = torch.cat(pdf_all, dim=0)
                proba_all = torch.cat(proba_all, dim=0)
                class_all = torch.cat(class_all, dim=0)
                params_all = get_parameters_si(model)

        # Two shift terms -----------------------------------------------------------------------
        elif len(model.nn_shift)==2:
            with torch.no_grad(): 
                for x_te1, x_te2, x_te3, y_te in test_loader:
                    x_te1, x_te2, x_te3, y_te = x_te1.to(device), x_te2.to(device), x_te3.to(device), y_te.to(device)

                    # Forward pass
                    out = model(x_te1, [x_te2, x_te3])

                    cdf = get_cdf(out)
                    pdf = get_pdf(cdf)
                    cdf_all.append(cdf)
                    pdf_all.append(pdf)
                    proba_all.append(pred_proba(pdf, y_te))
                    class_all.append(pred_class(pdf))

                # Concatenate all predictions into a single tensor
                cdf_all = torch.cat(cdf_all, dim=0)
                pdf_all = torch.cat(pdf_all, dim=0)
                proba_all = torch.cat(proba_all, dim=0)
                class_all = torch.cat(class_all, dim=0)
                params_all = get_parameters_si(model)

        # More than three data modalities ----------------------------------------------------
        elif len(model.nn_shift)>3:
            raise ValueError("More than three data modalities provided.")

        # Output -----------------------------------------------------------------------------
        
        # Transform tensors into numpy
        proba_all = proba_all.cpu().numpy()
        class_all = class_all.cpu().numpy()
        pdf_all = pdf_all.cpu().numpy()
        cdf_all = cdf_all.cpu().numpy()
        # params_all = params_all.cpu().numpy()

        if output=='prob':
            return proba_all
        if output=='class':
            return class_all
        if output=='all':
            return {'prob': proba_all, 'class': class_all, 'pdf': pdf_all, 'cdf': cdf_all, 'params': params_all}


    if si:
        return predict_ontram_si(model, test_loader, checkpoint_path, device, output)
    else:
        return predict_ontram_ci(model, test_loader, checkpoint_path, device, output)

