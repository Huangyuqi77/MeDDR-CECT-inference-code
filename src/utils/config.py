"""Unified configuration loader for the PLC radiomics pipeline."""
import sys
from pathlib import Path
import yaml


def load_config(config_name: str, config_dir: str = None) -> dict:
    """Load a YAML config file by name (without extension)."""
    if config_dir is None:
        config_dir = Path(__file__).resolve().parents[2] / "configs"
    else:
        config_dir = Path(config_dir)
    path = config_dir / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_configs(config_dir: str = None) -> dict:
    """Load all standard configs into a single dict keyed by config name."""
    names = ["paths", "radiomics", "model_search", "plotting"]
    cfg = {}
    for name in names:
        cfg[name] = load_config(name, config_dir)
    return cfg


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[2]


def resolve_path(rel_path: str, base: Path = None) -> Path:
    """Resolve a path that may be relative to project root."""
    p = Path(rel_path)
    if p.is_absolute():
        return p
    if base is None:
        base = get_project_root()
    return (base / p).resolve()


def setup_encoding():
    """Ensure UTF-8 stdout/stderr on Windows."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
