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

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.patches import Patch
import pandas as pd
import os

from .utils.configuration import *


# renamme set _meta_adj amtrix
class TramDagConfig:
    def __init__(self, conf_dict: dict = None, CONF_DICT_PATH: str = None,  _verify: bool = False,**kwargs):
        """
        Initialize TramDagConfig.

        Args:
            conf_dict: optional dict with configuration. If None, starts empty.
            CONF_DICT_PATH: optional path to config file.
        """

        self.debug = False
        self.verbose = False
        
        for key, value in kwargs.items():
            if key in ['conf_dict', 'CONF_DICT_PATH']:
                raise ValueError(f"Cannot override '{key}' via kwargs.")
            setattr(self, key, value)
        
        self.conf_dict = conf_dict or {}
        self.CONF_DICT_PATH = CONF_DICT_PATH
        
        # verification 
        if _verify:
            self._verify_completeness()

    @classmethod
    def load_json(cls, CONF_DICT_PATH: str,debug: bool = False):
        """Load a configuration from file and initialize a new instance.

        Reads a configuration dictionary from the specified path and returns
        a class instance initialized with that configuration.

        Args:
            CONF_DICT_PATH: Path to the configuration file to load.
            debug: Whether to enable debug mode during initialization.

        Returns:
            An instance of the class initialized with the loaded configuration.
        """
        conf = load_configuration_dict(CONF_DICT_PATH)
        return cls(conf, CONF_DICT_PATH=CONF_DICT_PATH, debug=debug, _verify=True)

    def update(self):
        """
        Reload the latest configuration from disk into the current instance.
        """
        if not hasattr(self, "CONF_DICT_PATH") or self.CONF_DICT_PATH is None:
            raise ValueError("CONF_DICT_PATH not set — cannot update configuration.")
        self.conf_dict = load_configuration_dict(self.CONF_DICT_PATH)


    def save(self, CONF_DICT_PATH: str = None):
        """Save the current configuration to a file.

        If no path is provided, the method uses the stored `self.CONF_DICT_PATH`. 
        Raises an error if neither is available.

        Args:
            CONF_DICT_PATH: Optional. Path to the configuration file to write.

        Raises:
            ValueError: If no valid path is provided.
        """
        path = CONF_DICT_PATH or self.CONF_DICT_PATH
        if path is None:
            raise ValueError("No CONF_DICT_PATH provided to save config.")
        write_configuration_dict(self.conf_dict, path)

    def _verify_completeness(self):
        """
        Verify that the configuration is complete and consistent:
        - All mandatory keys exist
        - Mandatory keys have valid values
        - Optional keys (if present) are valid
        """
        mandatory_keys = ["experiment_name","PATHS", "nodes", "data_type", "adj_matrix","nodes","model_names"]
        optional_keys = ["date_of_creation", "seed"]

        # ---- 1. Check mandatory keys exist
        missing = [k for k in mandatory_keys if k not in self.conf_dict]
        if missing:
            print(f"[WARNING] Missing mandatory keys in configuration: {missing}"
                "\n Please add them to the configuration dict and reload.")
            
        # --- 2. Check  if mandatory keys in nodesdict are present
        mandatory_keys_nodes = ['data_type', 'node_type','parents','parents_datatype','transformation_terms_in_h()','transformation_term_nn_models_in_h()']
        optional_keys_nodes = ["levels"]
        for node, node_dict in self.conf_dict.get("nodes", {}).items():
            # check missing mandatory keys
            missing_node_keys = [k for k in mandatory_keys_nodes if k not in node_dict]
            if missing_node_keys:
                print(f"[WARNING] Node '{node}' is missing mandatory keys: {missing_node_keys}")
                

        
        if self._verify_levels_dict():
            if self.debug:
                print("[DEBUG] levels are present for all ordinal variables in configuration dict.")
            pass
        else:
            print("[WARNING]  levels are missing for some ordinal variables in configuration dict. THIS will FAIL in model training later!\n"
                " Please provide levels manually to config and reload or compute levels from data using the method compute_levels().\n"
                " e.g. cfg.compute_levels(train_df) # computes levels from training data and writes to cfg")

        if self._verify_experiment_name():
            if self.debug:
                print("[DEBUG] experiment_name is valid in configuration dict.")
            pass
        
        if self._verify_adj_matrix():
            if self.debug:
                print("[DEBUG] adj_matrix is valid in configuration dict.")
            pass

    def _verify_levels_dict(self):
        """
        Verify that levels_dict is present for all categorical variables.
        """
        data_type = self.conf_dict.get('data_type', {})
        nodes = self.conf_dict.get('nodes', {})
        for var, dtype in data_type.items():
            if 'ordinal' in dtype:
                if var not in nodes or 'levels' not in nodes[var]:
                    return False
        return True

    def _verify_experiment_name(self):
        experiment_name = self.conf_dict.get("experiment_name")
        if experiment_name is None or str(experiment_name).strip() == "":
            return False
        return True
        
    def _verify_adj_matrix(self):
        adj_matrix = self.conf_dict['adj_matrix']
        if isinstance(adj_matrix, list):
            adj_matrix = np.array(self.conf_dict['adj_matrix'])
        if validate_adj_matrix(adj_matrix):
            return True
        else:
            return False
        
    def compute_levels(self, df: pd.DataFrame, write: bool = True):
        """Compute and update variable levels in the configuration.

        Derives level information for each variable from the provided DataFrame
        and updates the `nodes` section of the configuration dictionary accordingly.
        Optionally writes the updated configuration back to disk.

        Args:
            df: Input DataFrame containing data used to infer levels.
            write: Whether to save the updated configuration to file. Defaults to True.

        Raises:
            Exception: If saving the configuration fails when `write=True`.

        Notes:
            - Variables not found in the configuration's `nodes` dictionary trigger a warning.
            - Updates are persisted if `write=True` and a valid `CONF_DICT_PATH` is set.
        """
        levels_dict = create_levels_dict(df, self.conf_dict['data_type'])
        
        # update nodes dict with levels
        for var, levels in levels_dict.items():
            if var in self.conf_dict['nodes']:
                self.conf_dict['nodes'][var]['levels'] = levels
            else:
                print(f"[WARNING] Variable '{var}' not found in nodes dict. Cannot add levels.")
        
        if write and self.CONF_DICT_PATH is not None:
            try:
                self.save(self.CONF_DICT_PATH)
                if self.verbose or self.debug:
                    print(f'[INFO] Configuration with updated levels saved to {self.CONF_DICT_PATH}')
            except Exception as e:
                print(f'[ERROR] Failed to save configuration: {e}')

    def plot_dag(self, seed: int = 42, causal_order: bool = False):
        """
        Plot the DAG with Source, Sink, and Intermediate nodes.

        Parameters
        ----------
        seed : int, default=42
            Random seed for layout stability.
        causal_order : bool, default=True
            If True, use Graphviz 'dot' layout to preserve causal order.
            If False, use networkx.spring_layout for a force-directed layout.
        """
        adj_matrix = self.conf_dict.get("adj_matrix")
        data_type  = self.conf_dict.get("data_type")

        if adj_matrix is None or data_type is None:
            raise ValueError("Configuration must include 'adj_matrix' and 'data_type'.")

        if isinstance(adj_matrix, list):
            adj_matrix = np.array(adj_matrix)

        if not validate_adj_matrix(adj_matrix):
            raise ValueError("Invalid adjacency matrix.")
        if len(data_type) != adj_matrix.shape[0]:
            raise ValueError("data_type must match adjacency matrix size.")

        node_labels = list(data_type.keys())
        G, edge_labels = create_nx_graph(adj_matrix, node_labels)

        sources       = {n for n in G.nodes if G.in_degree(n) == 0}
        sinks         = {n for n in G.nodes if G.out_degree(n) == 0}
        intermediates = set(G.nodes) - sources - sinks

        node_colors = [
            "green" if n in sources
            else "red" if n in sinks
            else "lightblue"
            for n in G.nodes
        ]

        if causal_order:
            try:
                pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
            except (ImportError, nx.NetworkXException):
                pos = nx.spring_layout(G, seed=seed, k=1.5, iterations=100)
        else:
            pos = nx.spring_layout(G, seed=seed, k=1.5, iterations=100)

        plt.figure(figsize=(8, 6))
        nx.draw(
            G, pos,
            with_labels=True,
            node_color=node_colors,
            edge_color="gray",
            node_size=2500,
            arrowsize=20
        )

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

        legend_items = [
            Patch(facecolor="green",     edgecolor="black", label="Source"),
            Patch(facecolor="red",       edgecolor="black", label="Sink"),
            Patch(facecolor="lightblue", edgecolor="black", label="Intermediate")
        ]
        plt.legend(handles=legend_items, loc="upper right", frameon=True)

        plt.title(f"TRAM DAG")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

  
    def setup_configuration(self, experiment_name=None, EXPERIMENT_DIR=None, debug=False, _verify=False):
        """
        Create or reuse a configuration for an experiment.

        This method is dual-mode:
            1. If called on the class (`TramDagConfig.setup_configuration(...)`):
               - Creates or loads a configuration from disk.
               - Returns a new `TramDagConfig` instance initialized with that configuration.
            2. If called on an existing instance (`config.setup_configuration(...)`):
               - Updates the current instance's `conf_dict` and `CONF_DICT_PATH` in place.
               - Does not return a new object.

        Defaults:
            - `experiment_name`: defaults to "experiment_1" if None.
            - `EXPERIMENT_DIR`: defaults to `<cwd>/<experiment_name>` if None.
            - A directory is created if it does not exist.
            - If a configuration file already exists, it is reused; otherwise, a new one is created.

        Args:
            experiment_name (str, optional): Name of the experiment.
            EXPERIMENT_DIR (str, optional): Directory path for the experiment.
            debug (bool, optional): Enables debug mode during initialization. Default is False.
            _verify (bool, optional): Whether to verify configuration completeness. Default is False.

        Returns:
            TramDagConfig or None:
                - Returns a new instance if called on the class.
                - Returns None if called on an instance (in-place update).

        Side Effects:
            - Creates directories and configuration files as needed.
            - Prints the configuration status.
        """
        is_class_call = isinstance(self, type)
        cls = self if is_class_call else self.__class__

        if experiment_name is None:
            experiment_name = "experiment_1"
        if EXPERIMENT_DIR is None:
            EXPERIMENT_DIR = os.path.join(os.getcwd(), experiment_name)

        CONF_DICT_PATH = os.path.join(EXPERIMENT_DIR, "configuration.json")
        DATA_PATH = EXPERIMENT_DIR

        os.makedirs(EXPERIMENT_DIR, exist_ok=True)

        if os.path.exists(CONF_DICT_PATH):
            print(f"Configuration already exists: {CONF_DICT_PATH}")
        else:
            _ = create_and_write_new_configuration_dict(
                experiment_name, CONF_DICT_PATH, EXPERIMENT_DIR, DATA_PATH, None
            )
            print(f"Created new configuration file at {CONF_DICT_PATH}")

        conf = load_configuration_dict(CONF_DICT_PATH)

        if is_class_call:
            return cls(conf, CONF_DICT_PATH=CONF_DICT_PATH, debug=debug, _verify=_verify)
        else:
            self.conf_dict = conf
            self.CONF_DICT_PATH = CONF_DICT_PATH
            if _verify:
                self._verify_completeness()
                
    def set_data_type(self, data_type: dict, CONF_DICT_PATH: str = None) -> None:
        """
        Update or write the data type section of a configuration file.
        Works for both class-level and instance-level calls.
        """
        is_class_call = isinstance(self, type)
        cls = self if is_class_call else self.__class__

        # resolve path
        if CONF_DICT_PATH is None:
            if not is_class_call and getattr(self, "CONF_DICT_PATH", None):
                CONF_DICT_PATH = self.CONF_DICT_PATH
            elif not is_class_call:
                CONF_DICT_PATH = os.path.join(os.getcwd(), "configuration.json")
            else:
                raise ValueError("CONF_DICT_PATH must be provided when called on the class.")

        try:
            # load existing or create empty configuration
            configuration_dict = (
                load_configuration_dict(CONF_DICT_PATH)
                if os.path.exists(CONF_DICT_PATH)
                else {}
            )

            validate_variable_names(data_type.keys())
            if not validate_data_types(data_type):
                raise ValueError("Invalid data types in the provided dictionary.")

            configuration_dict["data_type"] = data_type
            write_configuration_dict(configuration_dict, CONF_DICT_PATH)

            # safe printing
            try:
                print_data_type_modeling_setting(data_type or {})
            except Exception as e:
                print(f"[WARNING] Could not print data type modeling settings: {e}")

            if not is_class_call:
                self.conf_dict = configuration_dict
                self.CONF_DICT_PATH = CONF_DICT_PATH


        except Exception as e:
            print(f"Failed to update configuration: {e}")
        else:
            print(f"Configuration updated successfully at {CONF_DICT_PATH}.")
            
    def set_meta_adj_matrix(self, CONF_DICT_PATH: str = None, seed: int = 5):
        """
        Launch the interactive TRAMDAG term editor.

        Dual-mode:
            1. Class call (`TramDagConfig.interactive_tramdag_terms(CONF_DICT_PATH=...)`)
               - Requires `CONF_DICT_PATH`.
            2. Instance call (`cfg.interactive_tramdag_terms()`)
               - Uses `self.CONF_DICT_PATH` if available.

        Args:
            CONF_DICT_PATH (str, optional): Path to the configuration file.
            seed (int, optional): Random seed for reproducibility.
        """
        is_class_call = isinstance(self, type)

        # resolve path
        if CONF_DICT_PATH is None:
            if not is_class_call and getattr(self, "CONF_DICT_PATH", None):
                CONF_DICT_PATH = self.CONF_DICT_PATH
            else:
                raise ValueError("CONF_DICT_PATH must be provided when called on the class.")

        # launch interactive editor
        interactive_adj_matrix(CONF_DICT_PATH, seed=seed)

        # reload config if instance
        if not is_class_call:
            self.conf_dict = load_configuration_dict(CONF_DICT_PATH)
            self.CONF_DICT_PATH = CONF_DICT_PATH


    def set_tramdag_nn_models(self, CONF_DICT_PATH: str = None):
        is_class_call = isinstance(self, type)
        if CONF_DICT_PATH is None:
            if not is_class_call and getattr(self, "CONF_DICT_PATH", None):
                CONF_DICT_PATH = self.CONF_DICT_PATH
            else:
                raise ValueError("CONF_DICT_PATH must be provided when called on the class.")

        updated_conf = interactive_nn_names_matrix(CONF_DICT_PATH)
        if updated_conf is not None and not is_class_call:
            self.conf_dict = updated_conf
            self.CONF_DICT_PATH = CONF_DICT_PATH


