
import numpy as np
import seaborn as sns
import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import colorsys

import ipywidgets as widgets
from IPython.display import display, clear_output
import re
import sys
import json
from datetime import datetime



# Utility: Generate N shades of green
def generate_green_shades(n):
    return [
        mcolors.to_hex(colorsys.hsv_to_rgb(0.33, 0.4 + 0.5 * (i / max(n - 1, 1)), 0.7 + 0.3 * (i / max(n - 1, 1))))
        for i in range(n)
    ]

def plot_dag(adj_matrix, data_type, seed=42, use_spring=True):
    """
    Plot the DAG with Source, Sink, and Intermediate nodes.

    Parameters:
    - adj_matrix: square upper‐triangular numpy array of edge labels (strings)
    - data_type: dict mapping node labels to types (keys are node names), 
                 length must match adj_matrix.shape[0]
    - seed: int, random seed for layout
    - use_spring: bool, if True use networkx.spring_layout; 
                  if False try Graphviz “dot” (falls back to spring)
    """

    # assume validate_adj_matrix and create_nx_graph are defined elsewhere
    if not validate_adj_matrix(adj_matrix):
        raise ValueError("Invalid adjacency matrix.")
    if len(data_type) != adj_matrix.shape[0]:
        raise ValueError("data_type must match adjacency matrix size.")

    node_labels = list(data_type.keys())
    G, edge_labels = create_nx_graph(adj_matrix, node_labels)

    # classify nodes
    sources       = {n for n in G.nodes if G.in_degree(n) == 0}
    sinks         = {n for n in G.nodes if G.out_degree(n) == 0}
    intermediates = set(G.nodes) - sources - sinks

    # assign node colors
    node_colors = [
        "green" if n in sources
        else "red" if n in sinks
        else "lightblue"
        for n in G.nodes
    ]

    # choose layout
    if use_spring:
        pos = nx.spring_layout(G, seed=seed, k=1.5, iterations=100)
    else:
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        except (ImportError, nx.NetworkXException):
            pos = nx.spring_layout(G, seed=seed, k=1.5, iterations=100)

    # draw nodes and edges
    plt.figure(figsize=(8, 6))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        edge_color="gray",
        node_size=2500,
        arrowsize=20
    )

    # draw edge labels colored by prefix
    for (u, v), lbl in edge_labels.items():
        color = (
            "blue"  if lbl.startswith("ci")
            else "red"   if lbl.startswith("ls")
            else "green" if lbl.startswith("cs")
            else "black"
        )
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels={(u, v): lbl},
            font_color=color,
            font_size=12
        )

    # build legend
    legend_items = [
        Patch(facecolor="green",     edgecolor="black", label="Source"),
        Patch(facecolor="red",       edgecolor="black", label="Sink"),
        Patch(facecolor="lightblue", edgecolor="black", label="Intermediate")
    ]
    plt.legend(handles=legend_items, loc="upper right", frameon=True)

    plt.title("TRAM DAG")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_nn_names_matrix(nn_names_matrix,data_type):
    """
    plots the nn_names_matrix more nicely in a matrix
    """
    
    fig, ax = plt.subplots(figsize=(10, 8))

    nn_names_matrix = np.array(nn_names_matrix, dtype=object)
    colors = np.vectorize(lambda x: 0 if x == 0 else 1)(nn_names_matrix)

    sns.heatmap(colors, annot=nn_names_matrix, fmt="", cmap="Blues", cbar=False,
                linewidths=0.5, linecolor='gray', annot_kws={"size": 10}, ax=ax)

    ax.set_title("Neural Network Model Mapping", fontsize=14)
    ax.set_xticks(np.arange(nn_names_matrix.shape[1]) + 0.5)
    ax.set_yticks(np.arange(nn_names_matrix.shape[0]) + 0.5)
    
    node_labels = list(data_type.keys())
    
    if node_labels is not None:
        ax.set_xticklabels(node_labels, fontsize=12)
        ax.set_yticklabels(node_labels, fontsize=12)
    else:
        ax.set_xticklabels([f'X{j}' for j in range(nn_names_matrix.shape[1])], fontsize=12)
        ax.set_yticklabels([f'X{i}' for i in range(nn_names_matrix.shape[0])], fontsize=12)

    plt.show()

def create_nx_graph(adj_matrix, node_labels=None):
    
    """
    This function takes an adjacency matrix and returns a networkx DiGraph object and a dictionary of edge labels.
    """
    
    # labels to the vars
    if node_labels is None:
        node_labels = {i: f'X{i}' for i in range(adj_matrix.shape[0])} # all with X_i
    else:
        node_labels = {i: node_labels[i] for i in range(len(node_labels))}
        
        if len(node_labels) != adj_matrix.shape[0]:
            raise ValueError("Number of node labels should match the number of nodes in the adjacency matrix.")
    
    G = nx.DiGraph()
    G.add_nodes_from(node_labels.values())
    
    edge_labels = {}
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] != "0":  # Ignore "0" (no edge)
                G.add_edge(node_labels[i], node_labels[j])
                edge_labels[(node_labels[i], node_labels[j])] = adj_matrix[i, j]
    return G, edge_labels




def interactive_adj_matrix(CONF_DICT_PATH ,seed=5):
    
    data_type =  load_configuration_dict(CONF_DICT_PATH)['data_type']
    n = len(data_type.keys())
    variables = list(data_type.keys())
    adj_matrix=read_adj_matrix_from_configuration(CONF_DICT_PATH)
    
    if adj_matrix is not None:
        plot_dag(adj_matrix, data_type, seed=seed)
        return None
        
    else:
        print("No matrix found. Please fill out the DAG and click 'Generate'.")

        output = widgets.Output()
        cells = {}

        def create_grid():
            input_grid = []
            header_widgets = [widgets.Label(value='')] + [widgets.Label(value=name) for name in variables]
            input_grid.extend(header_widgets)

            for i in range(n):
                input_grid.append(widgets.Label(value=variables[i]))
                for j in range(n):
                    if i >= j:
                        cell = widgets.Label(value="0")
                    else:
                        cell = widgets.Text(value="", placeholder="e.g., ls", layout=widgets.Layout(width='70px'))
                    cells[(i, j)] = cell
                    input_grid.append(cell)

            return widgets.GridBox(
                children=input_grid,
                layout=widgets.Layout(
                    grid_template_columns=("80px " * (n + 1)).strip(),
                    overflow='auto'
                )
            )

        def on_generate_clicked(b):
            with output:
                clear_output()
                adj_matrix = np.empty((n, n), dtype=object)
                for i in range(n):
                    for j in range(n):
                        if i >= j:
                            adj_matrix[i, j] = "0"
                        else:
                            adj_matrix[i, j] = cells[(i, j)].value.strip() or "0"

                try:
                    if not validate_adj_matrix(adj_matrix):
                        raise ValueError("Invalid adjacency matrix. Please check the criteria.")
                    write_adj_matrix_to_configuration(adj_matrix, CONF_DICT_PATH)
                    plot_dag(adj_matrix, data_type, seed=seed)
                    return None
                except Exception as e:
                    print(f"Error saving or plotting DAG: {e}")

        generate_btn = widgets.Button(description="Generate Matrix + Plot DAG", button_style='success')
        generate_btn.on_click(on_generate_clicked)

        gridbox = create_grid()
        ui = widgets.VBox([
            widgets.Label("Fill in the adjacency matrix (upper triangle only). Use 'ls', 'cs', etc."),
            gridbox,
            generate_btn,
            output
        ])

        display(ui)
        return None

def interactive_nn_names_matrix(CONF_DICT_PATH, seed=5):
    """
    If a saved NN-names matrix exists in configuration, load & display it.
    Otherwise, generate defaults from the adjacency matrix, show them
    in an editable grid (only for non-zero entries), and let the user overwrite before saving & plotting.
    """
    # Load config, types, matrices
    cfg = load_configuration_dict(CONF_DICT_PATH)
    data_type = cfg['data_type']
    adj_matrix = read_adj_matrix_from_configuration(CONF_DICT_PATH)
    nn_names_matrix = read_nn_names_matrix_from_configuration(CONF_DICT_PATH)
    var_names = list(data_type.keys())
    n = len(var_names)

    # If already saved, just plot and exit
    if nn_names_matrix is not None:
        plot_nn_names_matrix(nn_names_matrix, data_type)
        return

    # No saved NN-names → build defaults
    default_nn = create_nn_model_names(adj_matrix, data_type)

    output = widgets.Output()
    cells = {}

    def create_grid():
        # Build header row
        header_widgets = [widgets.Label(value="")] + [widgets.Label(value=v) for v in var_names]
        grid = header_widgets.copy()

        for i, vi in enumerate(var_names):
            grid.append(widgets.Label(value=vi))
            for j, vj in enumerate(var_names):
                if i >= j:
                    # Diagonal & lower triangle: non-editable blank
                    cell = widgets.Label(value="")
                else:
                    default = default_nn[i, j]
                    if default == "0":
                        # Do not display zeros
                        cell = widgets.Label(value="")
                    else:
                        # Editable for prefilled entries only
                        cell = widgets.Text(
                            value=default,
                            placeholder="",
                            layout=widgets.Layout(width="100px")
                        )
                        cells[(i, j)] = cell
                grid.append(cell)

        return widgets.GridBox(
            children=grid,
            layout=widgets.Layout(
                grid_template_columns=("100px " * (n + 1)).strip(),
                overflow="auto"
            )
        )

    def on_generate_clicked(b):
        with output:
            clear_output()
            # Build final nn_names_matrix
            nm = np.empty((n, n), dtype=object)
            for i in range(n):
                for j in range(n):
                    if i >= j:
                        nm[i, j] = "0"
                    else:
                        if (i, j) in cells:
                            val = cells[(i, j)].value.strip()
                            nm[i, j] = val if val else default_nn[i, j]
                        else:
                            nm[i, j] = "0"
            try:
                write_nn_names_matrix_to_configuration(nm, CONF_DICT_PATH)
                plot_nn_names_matrix(nm, data_type)
            except Exception as e:
                print(f"Error saving or plotting NN-names matrix: {e}")

    # Button to save and plot
    btn = widgets.Button(description="Generate NN-Names + Plot", button_style="success")
    btn.on_click(on_generate_clicked)

    # Layout UI
    grid = create_grid()
    ui = widgets.VBox([
        widgets.Label("Edit only the existing model names (non-zero entries)."),
        grid,
        btn,
        output
    ])
    display(ui)

# configuration dicitonary utils


def new_conf_dict(experiment_name,EXPERIMENT_DIR,DATA_PATH,LOG_DIR):
    """
    creates the empty_configuration_file for the experiment
    
    Structure:
    
    json / dictionary like
    
    {
        date_of_creation: '1.1.2024'
        experiment_name: "example_1"
        PATHS:{
                    DATA_PATH:
                    LOG_DIR:
                    EXPERIMENT_DIR:
                }  
        data_type: {'x1':'cont','x2':'cont','x3':'cont','x4':'cont','x5':'cont','x6':'cont','x7':'cont','x8':'cont'},  # continous , images , ordinal
        adj_matrix:   [[0,cs],[0,0]],
        model_names:  [[0,ComplexInterceptDefaultTabular],[0,0]],
        
        seed:42, 
        
        nodes: {'x1': { 'Modelnr': 0,
                        'data_type': 'cont',
                        'node_type': 'source',
                        'parents': [],
                        'parents_datatype': {},
                        'transformation_terms_in_h()': {},
                        'transformation_term_nn_models_in_h()': {},
                        'min': 0.1023280003906143,
                        'max': 1.895380529733125},
                'x2': { 'Modelnr': 1,
                        'data_type': 'cont',
                        'node_type': 'sink',
                        'parents': ['x1'],
                        'parents_datatype': {},
                        'transformation_terms_in_h()': {'x1':'cs'},
                        'transformation_term_nn_models_in_h()': {'x1':'ComplexInterceptDefaultTabular'},
                        'min': 0.09848326169895154,
                        'max': 1.9048444463462053},
                }
    
    """
    configuration_dict=    {
                            'date_of_creation':          None,
                            'experiment_name' :          None,
                            'PATHS':{
                                    'DATA_PATH':         None,
                                    'LOG_DIR':           None,
                                    'EXPERIMENT_DIR':    None,
                                    }, 
                            'data_type':                 None,
                            'adj_matrix':                None,
                            'model_names':               None,
                            'seed':                      None, 
                            'nodes':                     None,
                            }
    
    configuration_dict['date_of_creation']=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    configuration_dict['experiment_name']=experiment_name
    configuration_dict['PATHS']['DATA_PATH']=DATA_PATH
    configuration_dict['PATHS']['LOG_DIR']=LOG_DIR
    configuration_dict['PATHS']['EXPERIMENT_DIR']=EXPERIMENT_DIR
    
    return configuration_dict


def write_configuration_dict(configuration_dict, CONF_DICT_PATH):
    """
    Write out the configuration dict as JSON. 
    Catches filesystem and serialization errors.
    """
    try:
        with open(CONF_DICT_PATH, 'w', encoding='utf-8') as f:
            json.dump(configuration_dict, f, indent=4)
    except (OSError, TypeError) as e:
        # OSError covers file I/O errors, TypeError covers JSON serialization issues
        print(f"Error writing config to {CONF_DICT_PATH}: {e}", file=sys.stderr)
        raise   

def load_configuration_dict(CONF_DICT_PATH):
    """
    Load configuration dictionary from a JSON file.

    :param CONF_DICT_PATH: Path to the JSON configuration file.
    :return: The configuration dictionary.
    :raises:
        OSError if the file can’t be read,
        json.JSONDecodeError if the file isn’t valid JSON.
    """
    try:
        with open(CONF_DICT_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error loading config from {CONF_DICT_PATH}: {e}", file=sys.stderr)
        raise

def create_and_write_new_configuration_dict(experiment_name,CONF_DICT_PATH,EXPERIMENT_DIR,DATA_PATH,LOG_DIR):
    
    """
    Create a new configuration dictionary for the experiment and write it to the specified path.
    :param experiment_name: Name of the experiment.
    :param CONF_DICT_PATH: Path where the configuration dictionary should be saved.
    :param EXPERIMENT_DIR: Directory for the experiment.
    :param DATA_PATH: Path to the data.
    :param LOG_DIR: Directory for logs.
    :return: The created configuration dictionary.
    :raises:
        OSError if the file can’t be written,
        json.JSONDecodeError if the file isn’t valid JSON.
        Exception if there is an error creating the configuration dictionary.
    """
    try:
        configuration_dict=new_conf_dict(experiment_name,EXPERIMENT_DIR,DATA_PATH,LOG_DIR)
    except:
        print(f"Error creating configuration dictionary for {experiment_name}.", file=sys.stderr)
        raise
    try:
        write_configuration_dict(configuration_dict, CONF_DICT_PATH)
    except:
        print(f"Error writing configuration dictionary to {CONF_DICT_PATH}.", file=sys.stderr)
        raise
    return configuration_dict

def read_adj_matrix_from_configuration(CONF_DICT_PATH):
    """
    Read the adjacency matrix from the configuration dictionary.
    
    :param CONF_DICT_PATH: Path to the configuration dictionary.
    :return: The adjacency matrix as a numpy array.
    """
    configuration_dict = load_configuration_dict(CONF_DICT_PATH)
    if configuration_dict['adj_matrix'] is None:
        return None
    else:
        adj_matrix = configuration_dict['adj_matrix']
        if isinstance(adj_matrix, list):
            adj_matrix = np.array(configuration_dict['adj_matrix'])
        return adj_matrix

def write_adj_matrix_to_configuration(adj_matrix, CONF_DICT_PATH):
    """
    Write the adjacency matrix to the configuration dictionary.
    
    :param adj_matrix: The adjacency matrix to write.
    :param CONF_DICT_PATH: Path to the configuration dictionary.
    """
    configuration_dict = load_configuration_dict(CONF_DICT_PATH)
    configuration_dict['adj_matrix'] = adj_matrix.tolist()  # Convert to list for JSON serialization
    write_configuration_dict(configuration_dict, CONF_DICT_PATH)


def write_data_type_to_configuration(data_type: dict, CONF_DICT_PATH: str) -> None:
    """
    Write the data type information to the configuration dictionary.
    Prints a success message if it completes without error, otherwise prints the exception.
    
    :param data_type: Dictionary containing variable names and their data types.
    :param CONF_DICT_PATH: Path to the configuration dictionary (JSON file).
    """
    try:
        configuration_dict = load_configuration_dict(CONF_DICT_PATH)
        configuration_dict['data_type'] = data_type
        write_configuration_dict(configuration_dict, CONF_DICT_PATH)
    except Exception as e:
        print(f"Failed to update configuration: {e}")
    else:
        print("Configuration updated successfully.")

def read_nn_names_matrix_from_configuration(CONF_DICT_PATH):
    """
    Read the neural network names matrix from the configuration dictionary.
    
    :param CONF_DICT_PATH: Path to the configuration dictionary.
    :return: The neural network names matrix as a numpy array.
    """
    configuration_dict = load_configuration_dict(CONF_DICT_PATH)
    if configuration_dict['model_names'] is None:
        return None
    else:
        nn_names_matrix = configuration_dict['model_names']
        if isinstance(nn_names_matrix, list):
            nn_names_matrix = np.array(configuration_dict['model_names'])
        return nn_names_matrix
    
def write_nn_names_matrix_to_configuration(nn_names_matrix, CONF_DICT_PATH):
    """
    Write the neural network names matrix to the configuration dictionary.
    
    :param nn_names_matrix: The neural network names matrix to write.
    :param CONF_DICT_PATH: Path to the configuration dictionary.
    """
    configuration_dict = load_configuration_dict(CONF_DICT_PATH)
    configuration_dict['model_names'] = nn_names_matrix.tolist()  # Convert to list for JSON serialization
    write_configuration_dict(configuration_dict, CONF_DICT_PATH)
    
    


def write_nodes_information_to_configuration(CONF_DICT_PATH, min_vals, max_vals):  
    """
    Write the nodes information to the configuration dictionary.
    
    :param CONF_DICT_PATH: Path to the configuration dictionary.
    """
    try:
        adj_matrix = read_adj_matrix_from_configuration(CONF_DICT_PATH)
        nn_names_matrix = read_nn_names_matrix_from_configuration(CONF_DICT_PATH)
        data_type = load_configuration_dict(CONF_DICT_PATH)['data_type']
        
        configuration_dict = create_node_dict(adj_matrix, nn_names_matrix, data_type, min_vals, max_vals)
        
        conf = load_configuration_dict(CONF_DICT_PATH)
        conf['nodes'] = configuration_dict
    
        write_configuration_dict(conf, CONF_DICT_PATH)
        
    except Exception as e:
        print("Failed to update configuration:", e)
    else:
        print("Configuration updated successfully.")


def write_nodes_information_to_configuration_v2(CONF_DICT_PATH, min_vals, max_vals,levels_dict=None):  
    """
    Write the nodes information to the configuration dictionary.
    
    :param CONF_DICT_PATH: Path to the configuration dictionary.
    """
    try:
        adj_matrix = read_adj_matrix_from_configuration(CONF_DICT_PATH)
        nn_names_matrix = read_nn_names_matrix_from_configuration(CONF_DICT_PATH)
        data_type = load_configuration_dict(CONF_DICT_PATH)['data_type']
        
        configuration_dict = create_node_dict_v2(adj_matrix, nn_names_matrix, data_type, min_vals, max_vals,levels_dict=levels_dict)
        print(configuration_dict)
        conf = load_configuration_dict(CONF_DICT_PATH)
        conf['nodes'] = configuration_dict
    
        write_configuration_dict(conf, CONF_DICT_PATH)
        
    except Exception as e:
        print("Failed to update configuration:", e)
    else:
        print("Configuration updated successfully.")




def create_node_dict(adj_matrix, nn_names_matrix, data_type, min_vals, max_vals):
    """
    Creates a configuration dictionary for TRAMADAG based on an adjacency matrix,
    a neural network names matrix, and a data type dictionary.
    """
    if not validate_adj_matrix(adj_matrix):
        raise ValueError("Invalid adjacency matrix. Please check the criteria.")
    
    if len(data_type) != adj_matrix.shape[0]:
        raise ValueError("Data type dictionary should have the same length as the adjacency matrix.")
    
    nodes_dict = {}
    G, edge_labels = create_nx_graph(adj_matrix, node_labels=list(data_type.keys()))
    
    sources = [node for node in G.nodes if G.in_degree(node) == 0]
    sinks = [node for node in G.nodes if G.out_degree(node) == 0]
    
    for i, node in enumerate(G.nodes):
        parents = list(G.predecessors(node))
        nodes_dict[node] = {}
        nodes_dict[node]['Modelnr'] = i
        nodes_dict[node]['data_type'] = data_type[node]
        nodes_dict[node]['node_type'] = "source" if node in sources else "sink" if node in sinks else "internal"
        nodes_dict[node]['parents'] = parents
        nodes_dict[node]['parents_datatype'] = {parent:data_type[parent] for parent in parents}
        nodes_dict[node]['transformation_terms_in_h()'] = {parent: edge_labels[(parent, node)] for parent in parents if (parent, node) in edge_labels}
        nodes_dict[node]['min']=min_vals[i].tolist()   
        nodes_dict[node]['max']=max_vals[i].tolist()
        
        transformation_term_nn_models = {}
        for parent in parents:
            parent_idx = list(data_type.keys()).index(parent)  
            child_idx = list(data_type.keys()).index(node) 
            
            if nn_names_matrix[parent_idx, child_idx] != "0":
                transformation_term_nn_models[parent] = nn_names_matrix[parent_idx, child_idx]
        nodes_dict[node]['transformation_term_nn_models_in_h()'] = transformation_term_nn_models
    return nodes_dict

def create_node_dict_v2(adj_matrix, nn_names_matrix, data_type, min_vals, max_vals,levels_dict=None):
    """
    Creates a configuration dictionary for TRAMADAG based on an adjacency matrix,
    a neural network names matrix, and a data type dictionary.
    """
    if not validate_adj_matrix(adj_matrix):
        raise ValueError("Invalid adjacency matrix. Please check the criteria.")
    
    if len(data_type) != adj_matrix.shape[0]:
        raise ValueError("Data type dictionary should have the same length as the adjacency matrix.")
    
    target_nodes = {}
    G, edge_labels = create_nx_graph(adj_matrix, node_labels=list(data_type.keys()))
    
    sources = [node for node in G.nodes if G.in_degree(node) == 0]
    sinks = [node for node in G.nodes if G.out_degree(node) == 0]
    
    for i, node in enumerate(G.nodes):
        parents = list(G.predecessors(node))
        target_nodes[node] = {}
        target_nodes[node]['Modelnr'] = i
        target_nodes[node]['data_type'] = data_type[node]
        
        # write the levels of the ordinal outcome
        if data_type[node] == 'ord':
            if levels_dict is None:
                raise ValueError(
                    "levels_dict must be provided for ordinal nodes; "
                    "e.g. levels_dict={'x3': 3}"
                )
            if node not in levels_dict:
                raise KeyError(
                    f"levels_dict is missing an entry for node '{node}'. "
                    f"Expected something like levels_dict['{node}'] = <num_levels>"
                )
            target_nodes[node]['levels'] = levels_dict[node]
    
        target_nodes[node]['node_type'] = "source" if node in sources else "sink" if node in sinks else "internal"
        target_nodes[node]['parents'] = parents
        target_nodes[node]['parents_datatype'] = {parent:data_type[parent] for parent in parents}
        target_nodes[node]['transformation_terms_in_h()'] = {parent: edge_labels[(parent, node)] for parent in parents if (parent, node) in edge_labels}
        target_nodes[node]['min'] = min_vals.iloc[i].tolist()   
        target_nodes[node]['max'] = max_vals.iloc[i].tolist()

        
        transformation_term_nn_models = {}
        for parent in parents:
            parent_idx = list(data_type.keys()).index(parent)  
            child_idx = list(data_type.keys()).index(node) 
            
            if nn_names_matrix[parent_idx, child_idx] != "0":
                transformation_term_nn_models[parent] = nn_names_matrix[parent_idx, child_idx]
        target_nodes[node]['transformation_term_nn_models_in_h()'] = transformation_term_nn_models
    return target_nodes


def create_nn_model_names(adj_matrix, data_type):
    """
    Returns the model names for the NN models based on the adj_matrix and data_type.
    Supports extended codes like ci11, cs21 etc.
    """

    # Warn if cs or ci appear anywhere
    if np.any(np.char.startswith(adj_matrix.astype(str), 'cs')) or \
       np.any(np.char.startswith(adj_matrix.astype(str), 'ci')):
        print('*************\n Model has Complex intercepts and Complex shifts, please add your Model to the modelzoo \n*************')

    # Base model name mappings
    full_model_mappings = {
        'cont': {
            'cs': 'ComplexShiftDefaultTabular',
            'ci': 'ComplexInterceptDefaultTabular',
            'ls': 'LinearShift',
            'si': 'SimpleIntercept'
        },
        'ord': {
            'cs': 'ComplexShiftDefaultTabular',
            'ci': 'ComplexInterceptDefaultTabular',
            'ls': 'LinearShift',
            'si': 'SimpleIntercept'
        },
        'other': {
            'cs': 'ComplexShiftDefaultImage',
            'ci': 'ComplexInterceptDefaultImage',
            'ls': 'LinearShift',
            'si': 'SimpleIntercept'
        }
    }

    # Create new matrix to store names
    nn_names_matrix = np.empty_like(adj_matrix, dtype=object)
    var_types = list(data_type.values())

    for i in range(adj_matrix.shape[0]):
        variable_type = var_types[i]
        for j in range(adj_matrix.shape[1]):
            code = str(adj_matrix[i, j])
            if code == '0':
                nn_names_matrix[i, j] = '0'
                continue
            match = re.fullmatch(r'(cs|ci|ls|si)(\d*)', code)
            if match:
                base_code, suffix = match.groups()
                base_name = full_model_mappings[variable_type][base_code]
                nn_names_matrix[i, j] = base_name + suffix
            else:
                nn_names_matrix[i, j] = code  # Leave unrecognized codes unchanged

    return nn_names_matrix



## Adjcency matrix funcions
import re



def is_valid_column(col, col_idx):
    ci_pattern = re.compile(r"^ci(\d+)$")
    cs_pattern = re.compile(r"^cs(\d+)$")

    has_ci = False
    ci_numbered_ids = []

    has_cs = False
    cs_numbered_ids = []

    for item in col:
        if item == "0":
            continue
        elif item in ("si", "ls"):
            continue
        elif item == "ci":
            if has_ci:
                print(f"Column {col_idx} invalid: 'ci' can only be used once in a column.")
                return False
            else:
                has_ci = True
        elif m := ci_pattern.match(item):
            ci_numbered_ids.append(m.group(1))
        elif item == "cs":
            has_cs = True
        elif m := cs_pattern.match(item):
            cs_numbered_ids.append(m.group(1))
        else:
            print(f"Column {col_idx} invalid: Unknown token '{item}' found.")
            return False

    if has_ci and ci_numbered_ids:
        print(f"Column {col_idx} invalid: Cannot mix 'ci' and 'ciXX' in same column. ")
        return False

    if len(ci_numbered_ids) == 1:
        print(f"Column {col_idx} invalid: Only one 'ciXX' present. Need at least two.")
        return False

    if len(ci_numbered_ids) != len(set(ci_numbered_ids)):
        print(f"Column {col_idx} invalid: Duplicate 'ciXX' entries found.")
        return False

    if len(cs_numbered_ids) == 1:
        print(f"Column {col_idx} invalid: Only one 'csXX' present. Need at least two. ")
        return False

    if len(cs_numbered_ids) != len(set(cs_numbered_ids)):
        print(f"Column {col_idx} invalid: Duplicate 'csXX' entries found. ")
        return False

    return True


def validate_matrix_columns(adj_matrix):
    all_valid = True
    for i in range(adj_matrix.shape[1]):
        col = adj_matrix[:, i]
        if not is_valid_column(col, i):
            all_valid = False
    return all_valid




def validate_adj_matrix(adj_matrix):
    """
    Validate if the adjacency matrix follows the given criteria:
    1. Contains only allowed elements: "0", "ls", "cs", "ci"
    2. Is upper triangular (no edges in the lower part)
    3. Diagonal must be "0" (no self-loops)
    
    Parameters:
    - adj_matrix: 2D numpy array (object dtype)
    
    Returns:
    - bool: True if the adjacency matrix satisfies all conditions, False otherwise
    """

    num_nodes = adj_matrix.shape[0]

    #1.  Check allowed elements
    if not validate_matrix_columns(adj_matrix):
        return False

    #2. Check upper triangular property
    for i in range(num_nodes):
        for j in range(i):  # Lower triangle check
            if adj_matrix[i, j] != "0":
                return False

    # 3. Check diagonal is all "0"
    if not np.all(np.diag(adj_matrix) == "0"):
        return False

    return True


def get_binary_matrix_from_adjmatrix(adj_matrix):
    """
    Convert the adjacency matrix to a binary matrix.
    params:
    adj_matrix: 2D numpy array, adjacency matrix of the DAG trainglular upper
    e.g. adj_matrix = np.array([
                                ["0", "cs", "ls", "0"],  # A -> B (cs), A -> C (ls)
                                ["0", "0", "0", "ls"],  # B -> D (ls)
                                ["0", "0", "0", "cs"],  # C -> D (cs)
                                ["0", "0", "0", "0"]    # No outgoing edges from D
                            ], dtype=object)
    return:
    binary_matrix: 2D numpy array, binary matrix of the adjacency matrix
    """
    return (adj_matrix != "0").astype(int)


def merge_transformation_dicts(transformation_terms_in_h, transformation_term_nn_models_in_h):
    """
    Merges two dictionaries by key, creating a unified structure where each key maps to a dictionary 
    containing both 'h' (from transformation_terms_in_h) and 'modelname' (from transformation_term_nn_models_in_h).

    Args:
        transformation_terms_in_h (dict): Dictionary with transformation terms.
        transformation_term_nn_models_in_h (dict): Dictionary with corresponding model names.

    Returns:
        dict: A merged dictionary where each key maps to {'h': value from transformation_terms_in_h, 
              'modelname': value from transformation_term_nn_models_in_h}.
    """
    merged_dict = {
        key: {
            "h_term": transformation_terms_in_h.get(key, None),  # Get value from first dict, default to None if missing
            "class_name": transformation_term_nn_models_in_h.get(key, None)  # Get value from second dict
        }
        for key in transformation_terms_in_h.keys()  # Iterate over keys from the first dict
    }
    return merged_dict





def sort_dict_by_value_contains_i(model_dict):
    """
    Sorts a dictionary based on whether the values contain the letter 'i'.

    Args:
        model_dict (dict): Dictionary to sort (key-value pairs).

    Returns:
        dict: Sorted dictionary where values containing 'i' appear first.
    """
    return dict(sorted(model_dict.items(), key=lambda x: 'i' not in x[1]))


def sort_second_dict_by_first_dict_keys(sort_by, to_sort):
    """
    Sorts the second dictionary based on the order of keys in the first dictionary.

    Args:
        sort_by (dict): The reference dictionary whose key order is used for sorting.
        to_sort (dict): The dictionary to be sorted.

    Returns:
        dict: A sorted dictionary with keys appearing in the same order as in first_dict.
    """
    return {key: to_sort[key] for key in sort_by.keys() if key in to_sort}



############### graveyarded code, not used anymore, but might be useful in the future


# def get_configuration_dict(adj_matrix, nn_names_matrix, data_type):
#     """
#     Creates a configuration dictionary for TRAMADAG based on an adjacency matrix,
#     a neural network names matrix, and a data type dictionary.
#     """
#     if not validate_adj_matrix(adj_matrix):
#         raise ValueError("Invalid adjacency matrix. Please check the criteria.")
    
#     if len(data_type) != adj_matrix.shape[0]:
#         raise ValueError("Data type dictionary should have the same length as the adjacency matrix.")
    
#     configuration_dict = {}
#     G, edge_labels = create_nx_graph(adj_matrix, node_labels=list(data_type.keys()))
    
#     sources = [node for node in G.nodes if G.in_degree(node) == 0]
#     sinks = [node for node in G.nodes if G.out_degree(node) == 0]
    
#     for i, node in enumerate(G.nodes):
#         parents = list(G.predecessors(node))
        
#         configuration_dict[node] = {}
#         configuration_dict[node]['Modelnr'] = i
#         configuration_dict[node]['data_type'] = data_type[node]
#         configuration_dict[node]['node_type'] = "source" if node in sources else "sink" if node in sinks else "internal"
#         configuration_dict[node]['parents'] = parents
#         configuration_dict[node]['parents_datatype'] = {parent:data_type[parent] for parent in parents}
#         configuration_dict[node]['transformation_terms_in_h()'] = {parent: edge_labels[(parent, node)] for parent in parents if (parent, node) in edge_labels}
        
#         transformation_term_nn_models = {}
#         for parent in parents:
#             parent_idx = list(data_type.keys()).index(parent)  
#             child_idx = list(data_type.keys()).index(node) 
            
#             if nn_names_matrix[parent_idx, child_idx] != "0":
#                 transformation_term_nn_models[parent] = nn_names_matrix[parent_idx, child_idx]
#         configuration_dict[node]['transformation_term_nn_models_in_h()'] = transformation_term_nn_models
    
#     return configuration_dict