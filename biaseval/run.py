"""CLI entrypoint for running the biaseval pipeline with stage flags."""

from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
import yaml

from biaseval import analysis, llm, metrics, preprocess, validation, viz


@dataclass(frozen=True)
class PipelineStage:
    """Represents one named stage in the bias evaluation pipeline."""

    name: str
    handler: Callable[[], None]


STAGES: tuple[PipelineStage, ...] = (
    PipelineStage("collect", llm.run),
    PipelineStage("preprocess", preprocess.run),
    PipelineStage("analyze", analysis.run),
    PipelineStage("aggregate", metrics.run),
    PipelineStage("validate", validation.run),
    PipelineStage("visualize", viz.run),
)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _get_git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

    commit_hash = result.stdout.strip()
    return commit_hash or None


def _selected_stage_names(args: argparse.Namespace) -> list[str]:
    selected = [stage.name for stage in STAGES if getattr(args, stage.name)]
    if selected:
        return selected
    return [stage.name for stage in STAGES]


def _build_metadata(
    *,
    run_id: str,
    timestamp: str,
    stage_names: list[str],
    config_dir: Path,
    weights: dict,
    experiments: dict,
) -> dict:
    return {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_commit_hash": _get_git_commit_hash(),
        "config_snapshot": {
            "config_dir": str(config_dir),
            "weights": weights,
            "experiments": experiments,
        },
        "stages": stage_names,
    }


def _write_metadata(metadata: dict, output_dir: Path) -> Path:
    run_dir = output_dir / metadata["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run biaseval pipeline stages")
    for stage in STAGES:
        parser.add_argument(
            f"--{stage.name}",
            action="store_true",
            help=f"Run the {stage.name} stage (if no flags are passed, all stages run)",
        )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("config"),
        help="Directory containing weights.yaml and experiments.yaml",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("artifacts/runs"),
        help="Output directory for per-run metadata",
    )
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    weights = _load_yaml(args.config_dir / "weights.yaml")
    experiments = _load_yaml(args.config_dir / "experiments.yaml")

    selected_stage_names = _selected_stage_names(args)
    run_id = uuid.uuid4().hex
    timestamp = datetime.now(timezone.utc).isoformat()

    metadata = _build_metadata(
        run_id=run_id,
        timestamp=timestamp,
        stage_names=selected_stage_names,
        config_dir=args.config_dir,
        weights=weights,
        experiments=experiments,
    )
    metadata_path = _write_metadata(metadata, args.metadata_dir)

    print(f"[biaseval] run_id: {run_id}")
    print(f"[biaseval] wrote run metadata to {metadata_path}")

    selected_stage_set = set(selected_stage_names)
    for stage in STAGES:
        if stage.name not in selected_stage_set:
            continue
        print(f"[biaseval] running stage: {stage.name}")
        stage.handler()


if __name__ == "__main__":
    main()
