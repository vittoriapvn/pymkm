import os
import json
import importlib.util
from typing import List, Dict
from pathlib import Path


def get_default_txt_path(source: str, filename: str) -> str:
    try:
        # Attempt to locate installed package data
        spec = importlib.util.find_spec(f"pymkm.data.defaults.{source}")
        if spec is not None and spec.origin is not None:
            base_dir = os.path.dirname(spec.origin)
            full_path = os.path.join(base_dir, filename)
            if os.path.exists(full_path):
                return full_path
    except Exception:
        pass

    # Local fallback (e.g. during development)
    local = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "defaults", source, filename)
    )
    if os.path.exists(local):
        return local

    raise FileNotFoundError(f"Cannot find file '{filename}' for source '{source}'")


def get_available_sources() -> List[str]:
    try:
        from importlib.resources import files
        base = files("pymkm.data.defaults")
        return [f.name for f in base.iterdir() if f.is_dir()]
    except Exception:
        spec = importlib.util.find_spec("pymkm.data.defaults")
        if spec and spec.origin:
            folder_path = Path(spec.origin).parent
            return [f.name for f in folder_path.iterdir() if f.is_dir()]
    raise FileNotFoundError("Could not locate default sources.")


def list_available_defaults(source: str) -> List[str]:
    try:
        from importlib.resources import files
        folder = files(f"pymkm.data.defaults.{source}")
        return [f.name for f in folder.iterdir() if f.suffix == ".txt"]
    except Exception:
        spec = importlib.util.find_spec(f"pymkm.data.defaults.{source}")
        if spec and spec.origin:
            folder_path = Path(spec.origin).parent
            return [f.name for f in folder_path.iterdir() if f.suffix == ".txt"]
    raise FileNotFoundError(f"Could not locate txt files for source: {source}")


def load_lookup_table() -> Dict[str, Dict]:
    try:
        from importlib.resources import files
        path = files("pymkm.data").joinpath("elements.json")
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        # Fallback for local use
        local_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data", "elements.json")
        )
        with open(local_path, "r") as f:
            return json.load(f)