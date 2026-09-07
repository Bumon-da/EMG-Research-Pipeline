"""
Experiment Manager

Gives every pipeline run its own isolated output folder under
`output/experiments/<experiment_name>/`, and records a manifest describing
what config/dataset the run used, so results from different runs are never
silently overwritten and can be compared later.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from config import settings
from config.settings import EXPERIMENTS_DIR


class ExperimentManager:

    def __init__(self, name: str | None = None) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.experiment_name = name or f"experiment_{timestamp}"

        self.path = Path(EXPERIMENTS_DIR) / self.experiment_name

        self.path.mkdir(parents=True, exist_ok=True)

    def folder(self, name: str) -> Path:
        directory = self.path / name
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def save_manifest(self, extra: dict[str, Any] | None = None) -> Path:
        """
        Save a JSON manifest recording the config and environment this run
        used. `extra` can carry run-specific facts (e.g. dataset stats)
        that aren't known until after the run has started.
        """

        config_snapshot = {
            "RANDOM_SEED": settings.RANDOM_SEED,
            "SAMPLING_RATE": settings.SAMPLING_RATE,
            "NUM_SUBJECTS": settings.NUM_SUBJECTS,
            "NUM_CHANNELS": settings.NUM_CHANNELS,
            "WINDOW_SIZE_MS": settings.WINDOW_SIZE_MS,
            "WINDOW_OVERLAP": settings.WINDOW_OVERLAP,
            "LOWCUT": settings.LOWCUT,
            "HIGHCUT": settings.HIGHCUT,
            "NOTCH_FREQ": settings.NOTCH_FREQ,
            "DEFAULT_TEST_REPETITIONS": settings.DEFAULT_TEST_REPETITIONS,
            "RAW_DATA_DIR": str(settings.RAW_DATA_DIR),
            "EXCLUDE_REST": settings.EXCLUDE_REST,
            "EXERCISE_NUM_GESTURES": settings.EXERCISE_NUM_GESTURES,
            "MCR_THRESHOLD": settings.MCR_THRESHOLD,
            "SSC_THRESHOLD": settings.SSC_THRESHOLD,
            "HIST_BIN_EDGES": settings.HIST_BIN_EDGES,
            "WAVELET": settings.WAVELET,
            "WAVELET_LEVEL": settings.WAVELET_LEVEL,
        }

        manifest = {
            "experiment_name": self.experiment_name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "python_version": sys.version,
            "platform": platform.platform(),
            "config": config_snapshot,
        }

        if extra:
            manifest.update(extra)

        filepath = self.path / "manifest.json"

        filepath.write_text(
            json.dumps(manifest, indent=2, default=str),
            encoding="utf-8",
        )

        return filepath
