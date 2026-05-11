"""Logging and run-history utilities."""
import json
import datetime
from pathlib import Path
from src.utils.config import get_project_root


def get_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_timestamp_compact() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def append_run_history(script_name: str, inputs: str, outputs: str,
                       status: str, metrics: str = ""):
    """Append a run entry to worklog/RUN_HISTORY.md."""
    root = get_project_root()
    path = root / "worklog" / "RUN_HISTORY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = (
        f"\n## {get_timestamp()} - {script_name}\n"
        f"- **Input**: {inputs}\n"
        f"- **Output**: {outputs}\n"
        f"- **Status**: {status}\n"
    )
    if metrics:
        entry += f"- **Metrics**: {metrics}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)


def update_artifact_manifest(file_path: str, file_type: str,
                             source_step: str, description: str,
                             is_final: bool = False):
    """Add or update an entry in worklog/artifact_manifest.json."""
    root = get_project_root()
    manifest_path = root / "worklog" / "artifact_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8-sig") as f:
            manifest = json.load(f)
    else:
        manifest = []

    entry = {
        "file_path": str(file_path),
        "file_type": file_type,
        "source_step": source_step,
        "timestamp": get_timestamp(),
        "description": description,
        "is_final": is_final,
    }

    # Update existing or append
    for i, e in enumerate(manifest):
        if e["file_path"] == str(file_path):
            manifest[i] = entry
            break
    else:
        manifest.append(entry)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def update_status(current_step: str, next_step: str,
                  blockers: str = "None", artifacts: list = None):
    """Overwrite worklog/STATUS.md with current status."""
    root = get_project_root()
    path = root / "worklog" / "STATUS.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"# Pipeline Status\n\n"
        f"**Last updated**: {get_timestamp()}\n\n"
        f"## Current Step\n{current_step}\n\n"
        f"## Next Step\n{next_step}\n\n"
        f"## Blockers\n{blockers}\n\n"
    )
    if artifacts:
        content += "## Artifacts Produced\n"
        for a in artifacts:
            content += f"- {a}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
