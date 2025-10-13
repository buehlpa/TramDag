def sample_full_dag(configuration_dict,
                    EXPERIMENT_DIR,
                    device,
                    do_interventions={},
                    predefined_latent_samples_df: pd.DataFrame = None,
                    number_of_samples: int = 10_000,
                    batch_size: int = 256,
                    delete_all_previously_sampled: bool = True,
                    verbose: bool = True,
                    debug: bool = False,
                    minmax_dict=None):
    """
    Sample values for all nodes in a DAG given trained TRAM models, respecting
    parental ordering. Supports both generative sampling (new U's) and
    reconstruction sampling (predefined U's), as well as node-level interventions.

    Parameters
    ----------
    configuration_dict : dict
        Full experiment configuration. Must contain a "nodes" entry where each node
        has metadata including:
            - 'node_type': str, either 'source' or other
            - 'parents': list of parent node names
            - 'min': float, minimum allowed value for the node
            - 'max': float, maximum allowed value for the node
            - 'data_type': str, e.g. "continuous" or "ordinal"
    EXPERIMENT_DIR : str
        Base directory where per-node models and sampling results are stored.
    device : torch.device
        Device for model evaluation (e.g., 'cuda' or 'cpu').
    do_interventions : dict, optional
        Mapping of node names to fixed values. For intervened nodes, the model is
        bypassed and all samples are set to the specified value. Example:
        {'x1': 1.0}.
    predefined_latent_samples_df : pd.DataFrame, optional
        DataFrame of predefined latent variables (U's) for reconstruction. Must
        contain one column per node in the form "{node}_U". If provided, the number
        of samples is set to the number of rows in this DataFrame.
    number_of_samples : int, default=10_000
        Number of samples to draw per node when no predefined latent samples are given.
    batch_size : int, default=32
        Batch size for DataLoader evaluation during sampling.
    delete_all_previously_sampled : bool, default=True
        If True, deletes all existing sampled.pt/latents.pt files before starting.
    verbose : bool, default=True
        If True, print progress information.
    debug : bool, default=False
        If True, print detailed debug information for troubleshooting.

    Returns
    -------
    sampled_by_node : dict
        Mapping from node name to a tensor of sampled values (on CPU).
    latents_by_node : dict
        Mapping from node name to the latent variables (U's) used to generate
        those samples (on CPU).

    Notes
    -----
    - The function respects DAG ordering by ensuring parents are sampled before
      their children.
    - In generative mode (no predefined_latent_samples_df), latent U's are sampled
      from a standard logistic distribution and parents are taken from sampled.pt.
    - In reconstruction mode (with predefined_latent_samples_df), latent U's are
      read from the DataFrame, but parent values are still loaded from sampled.pt
      unless explicitly overridden upstream.
    - Models are loaded from "best_model.pt" in each node's directory and applied
      in eval mode with no gradient tracking.
    - Continuous outcomes are sampled via vectorized root finding
      (Chandrupatla's algorithm), while ordinal outcomes use categorical sampling.
    """
    if verbose or debug:
        print(f"[INFO] Starting full DAG sampling with {number_of_samples} samples per node.")
        if do_interventions:
            print(f"[INFO] Interventions specified for nodes: {list(do_interventions.keys())}")
            
    if debug:
        print('[DEBUG] sample_full_dag: device:', device)
        
        
    if predefined_latent_samples_df is not None:
        number_of_samples = len(predefined_latent_samples_df)
        if verbose or debug:
            print(f'[INFO] Using predefined latents samples from dataframe -> therefore n_samples is set to the number of rows in the dataframe: {len(predefined_latent_samples_df)}')
    
    
    
    
    
    
    target_nodes_dict=configuration_dict["nodes"]

    # Collect results for direct use in notebooks
    sampled_by_node = {}
    latents_by_node = {}


    if delete_all_previously_sampled:
        if verbose or debug:
            print("[INFO] Deleting all previously sampled data.")
        delete_all_samplings(target_nodes_dict, EXPERIMENT_DIR)
    
    
    processed_nodes=[] # log the processed nodes in this list

    # repeat process until all nodes are sampled
    while set(processed_nodes) != set(target_nodes_dict.keys()): 
        for node in target_nodes_dict: # for each node in the target_nodes_dict
            if node in processed_nodes:
                if verbose :
                    print(f"[INFO] Node '{node}' already processed.")
                continue
                        
            print(f'\n----*----------*-------------*--------Sample Node: {node} ------------*-----------------*-------------------*--') 
            
            ## 1. Paths 
            NODE_DIR = os.path.join(EXPERIMENT_DIR, f'{node}')
            SAMPLING_DIR = os.path.join(NODE_DIR, 'sampling')
            os.makedirs(SAMPLING_DIR, exist_ok=True)
            SAMPLED_PATH = os.path.join(SAMPLING_DIR, "sampled.pt")
            LATENTS_PATH = os.path.join(SAMPLING_DIR, "latents.pt")
            
            ## 2. Check if sampled and latents already exist 
            try:
                if check_sampled_and_latents(NODE_DIR, debug=debug):
                    processed_nodes.append(node)
                    continue
            except FileNotFoundError:
                pass
            
            ## 3. logic to make sure parents are always sampled first
            ### TODO : throw error if dict not in causal order..
            skipping_node = False
            if target_nodes_dict[node]['node_type'] != 'source':
                for parent in target_nodes_dict[node]['parents']:
                    parent_dir = os.path.join(EXPERIMENT_DIR, parent)
                    try:
                        check_sampled_and_latents(parent_dir, debug=debug)
                    except FileNotFoundError:
                        skipping_node = True
                        if verbose or debug:
                            print(f"[INFO] Skipping {node} as parent '{parent}' is not sampled yet.")
                        break

            if skipping_node:
                continue
            
            ## INTERVENTION, if node is to be intervened on , data is just saved
            if do_interventions and node in do_interventions.keys():
                    # For interventions make all the values the same for 
                    intervention_value = do_interventions[node]
                    if verbose or debug:
                        print(f"[INFO] Applying intervention for node '{node}' with value {intervention_value}")
                    intervention_vals = torch.full((number_of_samples,), intervention_value)
                    torch.save(intervention_vals, SAMPLED_PATH)
                    
                    ### dummy latents jsut for the check , not needed
                    dummy_latents = torch.full((number_of_samples,), float('nan'))  
                    torch.save(dummy_latents, LATENTS_PATH)
                    
                    # Store for immediate use
                    sampled_by_node[node] = intervention_vals
                    latents_by_node[node] = dummy_latents
                    
                    processed_nodes.append(node)
                    print(f'Interventional data for node {node} is saved')
                    continue  
                
            ##### NO INTERVENTION, based on the sampled data from the parents the latents for each node the observational distribution is generated    
            else:
                # if latents are predefined use them
                if predefined_latent_samples_df is not None:
                    predefinded_sample_name = node + "_U" ## TODO In config add in validator exception for _U

                    if predefinded_sample_name not in predefined_latent_samples_df.columns:
                        raise ValueError(
                            f"Predefined latent samples for node '{node}' not found in dataframe columns. "
                            f"Must be named '{predefinded_sample_name}'.")
                        
                    predefinded_sample = predefined_latent_samples_df[predefinded_sample_name].values
                    print(f'[INFO] Using predefined latents samples for node {node} from dataframe column: {predefinded_sample_name}')
                    
                    latent_sample = torch.tensor(predefinded_sample, dtype=torch.float32).to(device)
                    
                    ### sampling new latents
                else:
                    print(f'[INFO] Sampling new latents for node {node} from standard logistic distribution')
                    latent_sample = torch.tensor(logistic.rvs(size=number_of_samples), dtype=torch.float32).to(device)
                    
                
                ### load modelweights
                MODEL_PATH = os.path.join(NODE_DIR, "best_model.pt")
                tram_model = get_fully_specified_tram_model(node, configuration_dict, debug=debug, device=device,verbose=verbose).to(device)
                tram_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                
                # isntead of sample loader use Generic Dataset but the df is just to sampled data from befor -> create df for each node
                sampled_df=create_df_from_sampled(node, target_nodes_dict, number_of_samples, EXPERIMENT_DIR)
                
                sample_dataset = GenericDataset(sampled_df,target_col=node,
                                                    target_nodes=target_nodes_dict,
                                                    return_intercept_shift=True,
                                                    return_y=False,
                                                    debug=debug)
                
                sample_loader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=False,num_workers=4, pin_memory=True)
                
                ###*************************************************** Continous Modelled Outcome ************************************************
                
                if is_outcome_modelled_continous(node,target_nodes_dict):
                    sampled=sample_continous_modelled_target(node,target_nodes_dict,sample_loader,tram_model,latent_sample,device=device, debug=debug,minmax_dict=minmax_dict)
                    
                ###*************************************************** Ordinal Modelled Outcome ************************************************
                
                elif is_outcome_modelled_ordinal(node,target_nodes_dict):
                    sampled=sample_ordinal_modelled_target(sample_loader,tram_model,device=device, debug=debug)
                
                else:
                    raise ValueError(f"Unsupported data_type '{target_nodes_dict[node]['data_type']}' for node '{node}' in sampling.")
                    
                ###*************************************************** Saving the latenst and sampled  ************************************************
                if torch.isnan(sampled).any():
                    print(f"[WARNING] NaNs detected in sampled output for node '{node}'")
                    
                torch.save(sampled, SAMPLED_PATH)
                torch.save(latent_sample, LATENTS_PATH)
                
                # Store CPU copies for immediate use
                sampled_by_node[node] = sampled.detach().cpu()
                latents_by_node[node] = latent_sample.detach().cpu()
                
                if verbose:
                    print(f"[INFO] Completed sampling for node '{node}'")
                
                processed_nodes.append(node)          
        
    if verbose:
        print("[INFO] DAG sampling completed successfully for all nodes.")

    return sampled_by_node, latents_by_node