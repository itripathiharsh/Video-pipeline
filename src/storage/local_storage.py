import os
import json
from typing import List, Dict


class LocalStorage:
    def __init__(self, base_output_dir: str):
        self.base_dir = base_output_dir

        self.meta_dir = os.path.join(self.base_dir, "metadata")
        os.makedirs(self.meta_dir, exist_ok=True)

    def save_metadata(self, metadata: List[Dict], filename: str = "metadata.json") -> str:
        path = os.path.join(self.meta_dir, filename)

        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)

        return path
