
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import ipywidgets as widgets
from IPython.display import display, clear_output
import os

### plot the DAG
def plot_dag(adj_matrix, data_type, seed=42):
    """
    Plot the Directed Acyclic Graph (DAG) with Source and Sink nodes.
    params:
    adj_matrix: 2D numpy array, adjacency matrix of the DAG trainglular upper 
    e.g. adj_matrix = np.array([
                                ["0", "cs", "ls", "0"],  # A -> B (cs), A -> C (ls)
                                ["0", "0", "0", "ls"],  # B -> D (ls)
                                ["0", "0", "0", "cs"],  # C -> D (cs)
                                ["0", "0", "0", "0"]    # No outgoing edges from D
                            ], dtype=object)
    seed: int, seed for the random layout , change to get different layout
    """
    
    #validate adj_matrix
    if not validate_adj_matrix(adj_matrix):
        raise ValueError("Invalid adjacency matrix. Please check the criteria.")
    
    if len(data_type) != adj_matrix.shape[0]:
        raise ValueError("Data type dictionary should have the same length as the adjacency matrix.")
    
    
    node_labels=list(data_type.keys())
    
    #create a nx graph object
    G, edge_labels=create_nx_graph(adj_matrix,node_labels=node_labels)
    
    # sources and sinks
    sources = [node for node in G.nodes if G.in_degree(node) == 0]  # No incoming edges
    sinks = [node for node in G.nodes if G.out_degree(node) == 0]   # No outgoing edges
    
    node_colors = []
    for node in G.nodes:
        if node in sources:
            node_colors.append("green")  # Source nodes in green
        elif node in sinks:
            node_colors.append("red")    # Sink nodes in red
        else:
            node_colors.append("lightblue")  # Intermediate nodes in light blue

    # Draw the Graph
    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(G, seed=seed)  # Layout for positioning
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray',
            node_size=3000, arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=12)

    # Add a legend 
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Source'),
        Patch(facecolor='red', edgecolor='black', label='Sink')]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=True)
    plt.title("TRAM DAG")
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



def interactive_adj_matrix(variable_names, data_type, seed, experiment_dir="./", filename="adj_matrix.npy"):
    n = len(variable_names)
    FILENAME = os.path.join(experiment_dir, filename)
    adj_matrix_previous = None
    adj_matrix_return = None
    cells = {}

    output = widgets.Output()
    save_checkbox = widgets.Checkbox(value=False, description='Save matrix to file')
    filename_text = widgets.Text(value=filename, layout=widgets.Layout(width='200px'))

    def create_grid(initial_values=None):
        input_grid = []
        header_widgets = [widgets.Label(value='')] + [widgets.Label(value=name) for name in variable_names]
        input_grid.extend(header_widgets)

        for i in range(n):
            input_grid.append(widgets.Label(value=variable_names[i]))
            for j in range(n):
                if i >= j:
                    cell = widgets.Label(value="0")
                else:
                    val = ""
                    if initial_values is not None:
                        val = initial_values[i, j]
                    cell = widgets.Text(value=val if val != "0" else "", layout=widgets.Layout(width='70px'))
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
        nonlocal adj_matrix_previous, adj_matrix_return
        with output:
            clear_output()
            adj_matrix = np.empty((n, n), dtype=object)
            for i in range(n):
                for j in range(n):
                    if i >= j:
                        adj_matrix[i, j] = "0"
                    else:
                        val = cells[(i, j)].value.strip() or "0"
                        adj_matrix[i, j] = val

            adj_matrix_previous = adj_matrix.copy()
            adj_matrix_return = adj_matrix.copy()
            print("Adjacency Matrix stored internally.")

            try:
                plot_dag(adj_matrix, data_type, seed=seed)
            except Exception as e:
                print(f"Error plotting DAG: {e}")

            if save_checkbox.value:
                fname = os.path.join(experiment_dir, filename_text.value.strip())
                if fname.endswith('.npy'):
                    np.save(fname, adj_matrix)
                    print(f"Saved to {fname}")
                elif fname.endswith('.csv'):
                    np.savetxt(fname, adj_matrix, fmt="%s", delimiter=",")
                    print(f"Saved to {fname}")
                else:
                    print("Unsupported file type. Use .npy or .csv.")

    generate_btn = widgets.Button(description="Generate Matrix + Plot DAG", button_style='success')
    generate_btn.on_click(on_generate_clicked)

    if os.path.exists(FILENAME):
        print(f"Found existing matrix at {FILENAME}. Loading...")
        try:
            adj_matrix_previous = np.load(FILENAME, allow_pickle=True)
            adj_matrix_return = adj_matrix_previous.copy()
            gridbox = create_grid(initial_values=adj_matrix_previous)

            display(widgets.VBox([
                widgets.Label("Loaded existing matrix. You can edit or regenerate it."),
                gridbox,
                widgets.HBox([save_checkbox, filename_text]),
                generate_btn,
                output
            ]))

            with output:
                try:
                    plot_dag(adj_matrix_previous, data_type, seed=seed)
                except Exception as e:
                    print(f"Error plotting DAG: {e}")
        except Exception as e:
            print(f"Error loading matrix: {e}")
    else:
        print(f"No matrix found in {FILENAME}. Please create one.")
        gridbox = create_grid()
        display(widgets.VBox([
            widgets.Label("Fill in the adjacency matrix (upper triangle only). Use 'ls', 'cs', etc."),
            gridbox,
            widgets.HBox([save_checkbox, filename_text]),
            generate_btn,
            output
        ]))

    return adj_matrix_return




def get_configuration_dict(adj_matrix, nn_names_matrix, data_type):
    """
    Creates a configuration dictionary for TRAMADAG based on an adjacency matrix,
    a neural network names matrix, and a data type dictionary.
    """
    if not validate_adj_matrix(adj_matrix):
        raise ValueError("Invalid adjacency matrix. Please check the criteria.")
    
    if len(data_type) != adj_matrix.shape[0]:
        raise ValueError("Data type dictionary should have the same length as the adjacency matrix.")
    
    configuration_dict = {}
    G, edge_labels = create_nx_graph(adj_matrix, node_labels=list(data_type.keys()))
    
    sources = [node for node in G.nodes if G.in_degree(node) == 0]
    sinks = [node for node in G.nodes if G.out_degree(node) == 0]
    
    for i, node in enumerate(G.nodes):
        parents = list(G.predecessors(node))
        
        configuration_dict[node] = {}
        configuration_dict[node]['Modelnr'] = i
        configuration_dict[node]['data_type'] = data_type[node]
        configuration_dict[node]['node_type'] = "source" if node in sources else "sink" if node in sinks else "internal"
        configuration_dict[node]['parents'] = parents
        configuration_dict[node]['parents_datatype'] = {parent:data_type[parent] for parent in parents}
        configuration_dict[node]['transformation_terms_in_h()'] = {parent: edge_labels[(parent, node)] for parent in parents if (parent, node) in edge_labels}
        
        transformation_term_nn_models = {}
        for parent in parents:
            parent_idx = list(data_type.keys()).index(parent)  
            child_idx = list(data_type.keys()).index(node) 
            
            if nn_names_matrix[parent_idx, child_idx] != "0":
                transformation_term_nn_models[parent] = nn_names_matrix[parent_idx, child_idx]
        configuration_dict[node]['transformation_term_nn_models_in_h()'] = transformation_term_nn_models
    
    return configuration_dict



def create_nn_model_names(adj_matrix,data_type):
    
    """
    retunrns the model names for the nn models based on the adj_matrix and data_type
    """
    
    
    if 'cs' in adj_matrix or 'ci' in adj_matrix:
        print('************* \n Model has Complex intercepts and Coomplex shifts , please add your Model to the modelzoo \n************')
    
    # Default class model mappings
    full_model_mappings = {
        'cont': {'cs': 'ComplexShiftDefaultTabular',
                'ci': 'ComplexInterceptDefaultTabular', 
                'ls': 'LinearShift',
                'si': 'SimpleIntercept'},
        'ord': {'cs': 'ComplexShiftDefaultTabular',
                'ci': 'ComplexInterceptDefaultTabular', 
                'ls': 'LinearShift',
                'si': 'SimpleIntercept'},
        'other': {'cs': 'ComplexShiftDefaultImage',
                'ci': 'ComplexInterceptDefaultImage', 
                'ls': 'LinearShift',
                'si': 'SimpleIntercept'}
    }

    nn_names_matrix = adj_matrix.copy()
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] in ['cs', 'ci', 'ls', 'si']:
                variable_type = list(data_type.values())[i]
                nn_names_matrix[i, j] = full_model_mappings[variable_type][adj_matrix[i, j]]

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