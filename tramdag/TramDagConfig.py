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
    """
    Configuration manager for TRAM-DAG experiments.

    This class encapsulates:

    - The experiment configuration dictionary (`conf_dict`).
    - Its backing file path (`CONF_DICT_PATH`).
    - Utilities to load, validate, modify, and persist configuration.
    - DAG visualization and interactive editing helpers.

    Typical usage
    -------------
    - Load existing configuration from disk via `TramDagConfig.load_json`.
    - Or create/reuse experiment setup via `TramDagConfig().setup_configuration`.
    - Update sections such as `data_type`, adjacency matrix, and neural network
    model names using the provided methods.
    """
        
    def __init__(self, conf_dict: dict = None, CONF_DICT_PATH: str = None,  _verify: bool = False,**kwargs):
        """
        Initialize a TramDagConfig instance.

        Parameters
        ----------
        conf_dict : dict or None, optional
            Configuration dictionary. If None, an empty dict is used and can
            be populated later. Default is None.
        CONF_DICT_PATH : str or None, optional
            Path to the configuration file on disk. Default is None.
        _verify : bool, optional
            If True, run `_verify_completeness()` after initialization.
            Default is False.
        **kwargs
            Additional attributes to be set on the instance. Keys "conf_dict"
            and "CONF_DICT_PATH" are forbidden and raise a ValueError.

        Raises
        ------
        ValueError
            If any key in `kwargs` is "conf_dict" or "CONF_DICT_PATH".

        Notes
        -----
        By default, `debug` and `verbose` are set to False. They can be
        overridden via `kwargs`.
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
        """
        Load a configuration from a JSON file and construct a TramDagConfig.

        Parameters
        ----------
        CONF_DICT_PATH : str
            Path to the configuration JSON file.
        debug : bool, optional
            If True, initialize the instance with `debug=True`. Default is False.

        Returns
        -------
        TramDagConfig
            Newly created configuration instance with `conf_dict` loaded from
            `CONF_DICT_PATH` and `_verify_completeness()` executed.

        Raises
        ------
        FileNotFoundError
            If the configuration file cannot be found (propagated by
            `load_configuration_dict`).
        """

        conf = load_configuration_dict(CONF_DICT_PATH)
        return cls(conf, CONF_DICT_PATH=CONF_DICT_PATH, debug=debug, _verify=True)

    def update(self):
        """
        Reload the latest configuration from disk into this instance.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `CONF_DICT_PATH` is not set on the instance.

        Notes
        -----
        The current in-memory `conf_dict` is overwritten by the contents
        loaded from `CONF_DICT_PATH`.
        """

        if not hasattr(self, "CONF_DICT_PATH") or self.CONF_DICT_PATH is None:
            raise ValueError("CONF_DICT_PATH not set — cannot update configuration.")
        
        
        self.conf_dict = load_configuration_dict(self.CONF_DICT_PATH)


    def save(self, CONF_DICT_PATH: str = None):
        """
        Persist the current configuration dictionary to disk.

        Parameters
        ----------
        CONF_DICT_PATH : str or None, optional
            Target path for the configuration file. If None, uses
            `self.CONF_DICT_PATH`. Default is None.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If neither the function argument nor `self.CONF_DICT_PATH`
            provides a valid path.

        Notes
        -----
        The resulting file is written via `write_configuration_dict`.
        """
        path = CONF_DICT_PATH or self.CONF_DICT_PATH
        if path is None:
            raise ValueError("No CONF_DICT_PATH provided to save config.")
        write_configuration_dict(self.conf_dict, path)

    def _verify_completeness(self):
        """
        Check that the configuration is structurally complete and consistent.

        The following checks are performed:

        1. Top-level mandatory keys:
        - "experiment_name"
        - "PATHS"
        - "nodes"
        - "data_type"
        - "adj_matrix"
        - "model_names"

        2. Per-node mandatory keys:
        - "data_type"
        - "node_type"
        - "parents"
        - "parents_datatype"
        - "transformation_terms_in_h()"
        - "transformation_term_nn_models_in_h()"

        3. Ordinal / categorical levels:
        - All ordinal variables must have a corresponding "levels" entry
            under `conf_dict["nodes"][var]`.

        4. Experiment name:
        - Must be non-empty.

        5. Adjacency matrix:
        - Must be valid under `validate_adj_matrix`.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        - Missing or invalid components are reported via printed warnings.
        - Detailed debug messages are printed when `self.debug=True`.
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
        Verify that all ordinal variables have levels specified in the config.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if all variables declared as ordinal in ``conf_dict["data_type"]``
            have a "levels" entry in ``conf_dict["nodes"][var]``.
            False otherwise.

        Notes
        -----
        This method does not modify the configuration; it only checks presence
        of level information.
        """
        data_type = self.conf_dict.get('data_type', {})
        nodes = self.conf_dict.get('nodes', {})
        for var, dtype in data_type.items():
            if 'ordinal' in dtype:
                if var not in nodes or 'levels' not in nodes[var]:
                    return False
        return True

    def _verify_experiment_name(self):
        """
        Check whether the experiment name in the configuration is valid.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if ``conf_dict["experiment_name"]`` exists and is non-empty
            after stripping whitespace. False otherwise.
        """
        experiment_name = self.conf_dict.get("experiment_name")
        if experiment_name is None or str(experiment_name).strip() == "":
            return False
        return True
        
    def _verify_adj_matrix(self):
        """
        Validate the adjacency matrix stored in the configuration.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if the adjacency matrix passes `validate_adj_matrix`, False otherwise.

        Notes
        -----
        If the adjacency matrix is stored as a list, it is converted to a
        NumPy array before validation.
        """
        adj_matrix = self.conf_dict['adj_matrix']
        if isinstance(adj_matrix, list):
            adj_matrix = np.array(self.conf_dict['adj_matrix'])
        if validate_adj_matrix(adj_matrix):
            return True
        else:
            return False
        
    def compute_levels(self, df: pd.DataFrame, write: bool = True):
        """
        Infer and update ordinal/categorical levels from data.

        For each variable in the configuration's `data_type` section, this
        method uses the provided DataFrame to construct a levels dictionary
        and injects the corresponding "levels" entry into `conf_dict["nodes"]`.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame used to infer levels for configured variables.
        write : bool, optional
            If True and `CONF_DICT_PATH` is set, the updated configuration is
            written back to disk. Default is True.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If saving the configuration fails when `write=True`.

        Notes
        -----
        - Variables present in `levels_dict` but not in `conf_dict["nodes"]`
        trigger a warning and are skipped.
        - If `self.verbose` or `self.debug` is True, a success message is printed
        when the configuration is saved.
        """
        self.update()
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
        Visualize the DAG defined by the configuration.

        Nodes are categorized and colored as:
        - Source nodes (no incoming edges): green.
        - Sink nodes (no outgoing edges): red.
        - Intermediate nodes: light blue.

        Parameters
        ----------
        seed : int, optional
            Random seed for layout stability in the spring layout fallback.
            Default is 42.
        causal_order : bool, optional
            If True, attempt to use Graphviz 'dot' layout via
            `networkx.nx_agraph.graphviz_layout` to preserve causal ordering.
            If False or if Graphviz is unavailable, use `spring_layout`.
            Default is False.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `adj_matrix` or `data_type` is missing or inconsistent with each other,
            or if the adjacency matrix fails validation.

        Notes
        -----
        Edge labels are colored by prefix:
        - "ci": blue
        - "ls": red
        - "cs": green
        - other: black
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

        This method behaves differently depending on how it is called:

        1. Class call (e.g. `TramDagConfig.setup_configuration(...)`):
        - Creates or loads a configuration at the resolved path.
        - Returns a new `TramDagConfig` instance.

        2. Instance call (e.g. `cfg.setup_configuration(...)`):
        - Updates `self.conf_dict` and `self.CONF_DICT_PATH` in place.
        - Optionally verifies completeness.
        - Returns None.

        Parameters
        ----------
        experiment_name : str or None, optional
            Name of the experiment. If None, defaults to "experiment_1".
        EXPERIMENT_DIR : str or None, optional
            Directory for the experiment. If None, defaults to
            `<cwd>/<experiment_name>`.
        debug : bool, optional
            If True, initialize / update with `debug=True`. Default is False.
        _verify : bool, optional
            If True, call `_verify_completeness()` after loading. Default is False.

        Returns
        -------
        TramDagConfig or None
            - A new instance when called on the class.
            - None when called on an existing instance.

        Notes
        -----
        - A configuration file named "configuration.json" is created if it does
        not exist yet.
        - Underlying creation uses `create_and_write_new_configuration_dict`
        and `load_configuration_dict`.
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
        Update or write the `data_type` section of a configuration file.

        Supports both class-level and instance-level usage:

        - Class call:
        - Requires `CONF_DICT_PATH` argument.
        - Reads the file if it exists, or starts from an empty dict.
        - Writes updated configuration to `CONF_DICT_PATH`.

        - Instance call:
        - Uses `self.CONF_DICT_PATH` if available, otherwise defaults to
            `<cwd>/configuration.json` if no path is provided.
        - Updates `self.conf_dict` and `self.CONF_DICT_PATH` after writing.

        Parameters
        ----------
        data_type : dict
            Mapping `{variable_name: type_spec}`, where `type_spec` encodes
            modeling types (e.g. continuous, ordinal, etc.).
        CONF_DICT_PATH : str or None, optional
            Path to the configuration file. Must be provided for class calls.
            For instance calls, defaults as described above.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `CONF_DICT_PATH` is missing when called on the class, or if
            validation of data types fails.

        Notes
        -----
        - Variable names are validated via `validate_variable_names`.
        - Data type values are validated via `validate_data_types`.
        - A textual summary of modeling settings is printed via
        `print_data_type_modeling_setting`, if possible.
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
        Launch the interactive editor to set or modify the adjacency matrix.

        This method:

        1. Resolves the configuration path either from the argument or, for
        instances, from `self.CONF_DICT_PATH`.
        2. Invokes `interactive_adj_matrix` to edit the adjacency matrix.
        3. For instances, reloads the updated configuration into `self.conf_dict`.

        Parameters
        ----------
        CONF_DICT_PATH : str or None, optional
            Path to the configuration file. Must be provided when called
            on the class. For instance calls, defaults to `self.CONF_DICT_PATH`.
        seed : int, optional
            Random seed for any layout or stochastic behavior in the interactive
            editor. Default is 5.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `CONF_DICT_PATH` is not provided and cannot be inferred
            (e.g. in a class call without path).

        Notes
        -----
        `self.update()` is called at the start to ensure the in-memory config
        is in sync with the file before launching the editor.
        """
        self.update()
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
        """
        Launch the interactive editor to set TRAM-DAG neural network model names.

        Depending on call context:

        - Class call:
        - Requires `CONF_DICT_PATH` argument.
        - Returns nothing and does not modify a specific instance.

        - Instance call:
        - Resolves `CONF_DICT_PATH` from the argument or `self.CONF_DICT_PATH`.
        - Updates `self.conf_dict` and `self.CONF_DICT_PATH` if the editor
            returns an updated configuration.

        Parameters
        ----------
        CONF_DICT_PATH : str or None, optional
            Path to the configuration file. Must be provided when called on
            the class. For instance calls, defaults to `self.CONF_DICT_PATH`.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `CONF_DICT_PATH` is not provided and cannot be inferred
            (e.g. in a class call without path).

        Notes
        -----
        The interactive editor is invoked via `interactive_nn_names_matrix`.
        If it returns `None`, the instance configuration is left unchanged.
        """
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


