
import os
import sys
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.abspath(os.path.join(script_dir, os.pardir))

if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import torch
from utils.tram_model_helpers import *


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

## minimal test




#### --- Modelloader Test Code ---


def load_and_write_test_dict_to_json(input:dict, test_name,file_path = None):
    """
    input has to bea dictionary with the following structure:
                                "input":{
                                                'x1': {
                                                'data_type': 'cont',
                                                'node_type': 'source',
                                                'parents': [],
                                                'parents_datatype': {},
                                                'transformation_terms_in_h()': {},
                                                'transformation_term_nn_models_in_h()': {}},
                                                
                                For n nodes
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
    except Exception as e:
        print(f"Error loading JSON file: {e}")
    
    data[test_name]={}
    data[test_name].setdefault("input", {})
    data[test_name].setdefault("output", {})

    
    data[test_name]["input"] = input
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for node_name in input:
        model = get_fully_specified_tram_model(node_name, input, verbose=False).to(device)
        data[test_name]["output"][node_name] = repr(model)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Updated outputs written to {file_path}")
    except Exception as e:
        print(f"Error writing to JSON file: {e}")
        
        
def run_model_loader_test(test_name: str, testdict_path: str, device: torch.device = None):
    """
    General test for fully_specified_tram_model loader based on ground-truth JSON.

    Args:
        test_name: Key in the JSON file identifying the test case.
        testdict_path: Path to the JSON file containing input and expected output.
        device: Torch device to move models to; defaults to CUDA if available, else CPU.

    Raises:
        AssertionError: If any model's repr() does not match the expected output.
        ValueError: If the test_name is not found in the JSON file.
    """
    # Load ground-truth data
    with open(testdict_path, 'r') as f:
        test_data = json.load(f)

    if test_name not in test_data:
        raise ValueError(f"Test name '{test_name}' not found in {testdict_path}")
    

    inputs = test_data[test_name]['input']
    expected_outputs = test_data[test_name].get('output', {})

    # Determine device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Iterate and compare
    for node_name in inputs:
        model = get_fully_specified_tram_model(node_name, inputs, verbose=False).to(device)
        actual_repr = repr(model)
        expected_repr = expected_outputs.get(node_name)

        assert expected_repr is not None, (
            f"No expected output found for node '{node_name}' "
            f"in '{testdict_path}'."
        )
        assert actual_repr == expected_repr, (
            f"Mismatch for node '{node_name}':\n"
            f"  Expected: {expected_repr}\n"
            f"  Actual:   {actual_repr}"
        )
        
if __name__ == "__main__":
    # Define the directory for test files
    TEST_DIR = os.path.join(os.path.dirname(__file__), 'model_tests')
    os.makedirs(TEST_DIR, exist_ok=True)

    # Run tests for two variables
    for test in ['test_1', 'test_2', 'test_3']:
        run_model_loader_test(test, os.path.join(TEST_DIR, 'twovars_model_loader_test_dict.json'))

    # Run tests for three variables
    for test in ['test_1', 'test_2', 'test_3', 'test_4', 'test_5', 'test_6']:
        run_model_loader_test(test, os.path.join(TEST_DIR, 'threevars_model_loader_test_dict.json'))
        
    # Run tests for ten variables
    for test in ['test_1']:
        run_model_loader_test(test, os.path.join(TEST_DIR, 'threevars_model_loader_test_dict.json'))