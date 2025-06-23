import pandas as pd

def load_validation_file(filepath):
    """
    Load a validation file containing a metadata header followed by numeric data.

    Args:
        filepath (str or Path): Path to the validation data file.

    Returns:
        metadata (dict): Dictionary containing metadata fields.
        data (pd.DataFrame): DataFrame with numerical data (columns: x, y).
    """
    metadata = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("data:"):
            data_start = i + 1
            break
        if ':' in line:
            key, val = line.strip().split(':', 1)
            metadata[key.strip()] = val.strip()

    data = pd.read_csv(filepath, skiprows=data_start, sep=r"\s+", names=["x", "y"])
    return metadata, data