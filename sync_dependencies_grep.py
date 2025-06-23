
import ast
from pathlib import Path
import toml
import sys
import importlib.metadata

# Extended list of standard library modules
STANDARD_LIBS = {
    "os", "sys", "math", "re", "json", "time", "typing", "pathlib",
    "itertools", "collections", "dataclasses", "functools", "subprocess",
    "logging", "shutil", "copy", "datetime", "builtins", "warnings",
    "pickle", "importlib", "concurrent"
}

def get_installed_version(pkg_name):
    try:
        version = importlib.metadata.version(pkg_name)
        return f"{pkg_name} >= {version}"
    except importlib.metadata.PackageNotFoundError:
        return pkg_name  # fallback if version not found

def find_imports(source_dir="pymkm", internal_package="pymkm"):
    """
    Recursively parse .py files and collect top-level import statements.
    Filters out standard libraries and internal project modules.
    """
    packages = set()
    for py_file in Path(source_dir).rglob("*.py"):
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(py_file))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            packages.add(alias.name.split(".")[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and not node.level:
                            packages.add(node.module.split(".")[0])
        except Exception as e:
            print(f"⚠️ Skipping {py_file}: {e}", file=sys.stderr)

    # Filter and retrieve versioned dependencies
    return sorted(
        get_installed_version(p)
        for p in packages
        if p not in STANDARD_LIBS and p != internal_package
    )

def sync_requirements_and_pyproject(req_file="requirements.txt", pyproject_file="pyproject.toml", source_dir="pymkm"):
    """
    Writes the found dependencies with versions into requirements.txt and updates pyproject.toml.
    """
    pkgs = find_imports(source_dir)

    # Write requirements.txt
    Path(req_file).write_text("\n".join(pkgs) + "\n")
    print(f"✅ Generated {req_file} with {len(pkgs)} packages (with versions)")

    # Update pyproject.toml
    py_path = Path(pyproject_file)
    if not py_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found: {pyproject_file}")

    pyproject = toml.load(py_path)
    pyproject.setdefault("project", {})["dependencies"] = pkgs

    with open(py_path, "w") as f:
        toml.dump(pyproject, f)

    print(f"✅ Updated {pyproject_file} with versioned dependencies from imports")

if __name__ == "__main__":
    sync_requirements_and_pyproject()
