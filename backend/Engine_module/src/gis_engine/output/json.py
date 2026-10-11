import os
import json
from typing import Dict, Any

def write_pair_metadata_json(output_path: str, metadata_dict: Dict[str, Any]):
    """
    Saves the pair inference log metadata to a JSON file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata_dict, f, indent=4)
