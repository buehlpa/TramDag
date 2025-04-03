from utils.tram_models import *
from utils.graph_utils import *



def get_fully_specified_tram_model(node,conf_dict,verbose=True):
    

    ### iF node is a source -> no deep nn is needed
    if conf_dict[node]['node_type'] == 'source':
        nn_int = SimpleIntercept()
        tram_model = TramModel(nn_int, None)  
        if verbose:
            print('>>>>>>>>>>>>  source node --> only  modelled only  by si')
            print(tram_model)
        
    ### if node is not a source node 
    else:
        # read terms and model names form the config
        terms_dict=conf_dict[node]['transformation_terms_in_h()']
        model_names_dict=conf_dict[node]['transformation_term_nn_models_in_h()']
        
        # Combine terms and model names and divide in intercept and shift terms
        model_dict=merge_transformation_dicts(terms_dict, model_names_dict)
        intercepts_dict = {k: v for k, v in model_dict.items() if "ci" in v['h_term'] or 'si' in v['h_term']}        
        shifts_dict = {k: v for k, v in model_dict.items() if "ci" not in v['h_term'] and  'si' not in v['h_term']}        
        
        # make sure that nns are correctly defined afterwards
        nn_int, nn_shifts_list = None, None
        
        # intercept term
        if not np.any(np.array([True for diction in intercepts_dict.values() if 'ci' in diction['h_term']]) == True):
            print('>>>>>>>>>>>> No ci detected --> intercept defaults to si') if verbose else None
            nn_int = SimpleIntercept()
        
        else:
            
            # intercept term -> model
            nn_int_name = list(intercepts_dict.items())[0][1]['class_name'] # TODO this doesnt work for multi inpout CI's
            nn_int = globals()[nn_int_name]()
        
        # shift term -> lsit of models         
        nn_shift_names=[v["class_name"] for v in shifts_dict.values() if "class_name" in v]
        nn_shifts_list = [globals()[name]() for name in nn_shift_names]
        
        # ontram model
        tram_model = TramModel(nn_int, nn_shifts_list)    
        
        print('>>> TRAM MODEL:\n',tram_model) if verbose else None
    return tram_model
