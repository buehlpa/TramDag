
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


### plot the DAG
def plot_dag(adj_matrix, seed=42):
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
    
    #create a nx graph object
    G, edge_labels=create_nx_graph(adj_matrix)
    
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



def create_nx_graph(adj_matrix):
    
    """
    This function takes an adjacency matrix and returns a networkx DiGraph object and a dictionary of edge labels.
    """
    
    # labels to the vars
    node_labels = {i: f'X{i}' for i in range(adj_matrix.shape[0])}
    G = nx.DiGraph()
    G.add_nodes_from(node_labels.values())
    
    edge_labels = {}
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] != "0":  # Ignore "0" (no edge)
                G.add_edge(node_labels[i], node_labels[j])
                edge_labels[(node_labels[i], node_labels[j])] = adj_matrix[i, j]
    return G, edge_labels

def get_configuration_dict(adj_matrix, data_type):
    
    """
    args:
    adj_matrix: 2D numpy array, adjacency matrix of the DAG trainglular upper
    data_type: dictionary, data type of the nodes
  
    e.g. adj_matrix = np.array([
                                ["0", "cs", "ls", "0"],  # A -> B (cs), A -> C (ls)
                                ["0", "0", "0", "ls"],  # B -> D (ls)
                                ["0", "0", "0", "cs"],  # C -> D (cs)
                                ["0", "0", "0", "0"]    # No outgoing edges from D
                            ], dtype=object)
    data_type = {"X0": "other", "X1": "cont", "X2": "cont", "X3": "ord"}      
  
    This function takes an adjacency matrix and a dictionary of data types and returns a configuration dictionary
    that stores the nececassry information for the TRAMADAG
    
    """
    # check if the adj_matrix is valid
    if not validate_adj_matrix(adj_matrix):
        raise ValueError("Invalid adjacency matrix. Please check the criteria.")
    
    #check if dataype has same leng as the adj_matrix
    if len(data_type) != adj_matrix.shape[0]:
        raise ValueError("Data type dictionary should have the same length as the adjacency matrix.")
    
    configuration_dict = {}
    G, edge_labels = create_nx_graph(adj_matrix)

    sources = [node for node in G.nodes if G.in_degree(node) == 0]  # No incoming edges
    sinks = [node for node in G.nodes if G.out_degree(node) == 0]   # No outgoing edges

    for i,node in enumerate(G.nodes):
        parents = list(G.predecessors(node))
        
        configuration_dict[node] = {}
        configuration_dict[node]['Modelnr'] = i
        configuration_dict[node]['data_type']=data_type[node]
        configuration_dict[node]['node_type'] = "source" if node in sources else "sink" if node in sinks else "internal"
        configuration_dict[node]['parents'] = list(G.predecessors(node))
        configuration_dict[node]['transformation_terms_in_h()'] = {parent: edge_labels[(parent, node)] for parent in parents if (parent, node) in edge_labels}
        # configuration_dict[node]['children'] = list(G.successors(node))
        # configuration_dict[node]['child_edges'] = {child: edge_labels[(node, child)] for child in G.successors(node) if (node, child) in edge_labels}
        
    return configuration_dict








## Adjcency matrix funcions
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
    allowed_values = {"0", "ls", "cs", "ci"}
    num_nodes = adj_matrix.shape[0]

    #1.  Check allowed elements
    if not np.all(np.isin(adj_matrix, list(allowed_values))):
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