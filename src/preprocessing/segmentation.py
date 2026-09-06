"""
Signal Segmentation (Windowing)

Slides a fixed-size window (`WINDOW_SIZE_MS` at `WINDOW_OVERLAP`, from
config.settings) over each trial's samples and decides, per candidate
window, whether it is usable for a downstream classifier:

- The label must be uniform across the window - a window spanning a
  gesture-label transition is dropped rather than assigned a majority
  label, since a stitched-together label at a transition would be
  ambiguous.
- The window must fall entirely on one side of a `TrialSplit`'s
  train/test masks (src.data.splitter) - a window straddling the split
  boundary, or overlapping a `repetition_split(exclude_rest=True)` gap,
  belongs to neither and is dropped. This is exactly the rule
  src/data/splitter.py's own module docstring anticipates ("a window is
  only valid if every sample inside it falls on the same side of the
  split").

Windows carry index metadata only (start/end/label/split), never a copy
of the underlying signal - `trial.emg[window.start:window.end]` is a
zero-copy view, which is what a future feature-extraction stage will
actually slice, on demand, per window. Materializing every window's array
for the whole dataset (~1.2M windows) up front would cost several GB for
no current consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np

from config.settings import SAMPLING_RATE, WINDOW_OVERLAP, WINDOW_SIZE_MS
from src.data.datamodels import Trial
from src.data.splitter import TrialSplit

Split = Literal["train", "test"]


def window_samples(window_size_ms: float = WINDOW_SIZE_MS, sampling_rate: int = SAMPLING_RATE) -> int:
    """Window length in samples, e.g. 200ms @ 100Hz -> 20."""
    return round(window_size_ms / 1000 * sampling_rate)


def step_samples(
    window_size_ms: float = WINDOW_SIZE_MS,
    window_overlap: float = WINDOW_OVERLAP,
    sampling_rate: int = SAMPLING_RATE,
) -> int:
    """Stride between window starts, e.g. a 20-sample window @ 50% overlap -> 10."""
    return max(1, round(window_samples(window_size_ms, sampling_rate) * (1 - window_overlap)))


@dataclass
class Window:
    start: int  # inclusive sample index into the trial's own arrays
    end: int  # exclusive; end - start == window_samples(...)
    label: int  # the single label uniform across [start, end)
    split: Split


@dataclass
class WindowCounts:
    total_candidates: int
    kept_train: int
    kept_test: int
    dropped_transition: int
    dropped_boundary: int  # also covers exclude_rest=True "gap" windows


def _iter_candidate_windows(
    trial: Trial,
    trial_split: TrialSplit,
    window: int,
    step: int,
) -> Iterator[tuple[Window | None, str | None]]:
    """
    Single pass over every candidate window position for this trial.
    Yields (window_or_none, drop_reason_or_none) - exactly one is not
    None. `trial.labels` is reshaped to 1D before use since real .mat
    files store restimulus/rerepetition as (n, 1) arrays, matching
    src/data/splitter.py's own handling of the same fields.
    """

    labels = np.asarray(trial.labels).reshape(-1)
    n_samples = labels.shape[0]

    for start in range(0, n_samples - window + 1, step):
        end = start + window

        label_slice = labels[start:end]
        if not np.all(label_slice == label_slice[0]):
            yield None, "transition"
            continue

        label = int(label_slice[0])

        if trial_split.train_mask[start:end].all():
            yield Window(start=start, end=end, label=label, split="train"), None
        elif trial_split.test_mask[start:end].all():
            yield Window(start=start, end=end, label=label, split="test"), None
        else:
            yield None, "boundary"


def segment_trial_detailed(
    trial: Trial,
    trial_split: TrialSplit,
    window_size_ms: float = WINDOW_SIZE_MS,
    window_overlap: float = WINDOW_OVERLAP,
    sampling_rate: int = SAMPLING_RATE,
) -> tuple[list[Window], WindowCounts]:
    """
    Segment one trial into valid windows, returning both the kept windows
    and a full breakdown of how many candidates were dropped and why -
    the orchestrator uses this so it never scans a trial twice.
    """

    window = window_samples(window_size_ms, sampling_rate)
    step = step_samples(window_size_ms, window_overlap, sampling_rate)

    windows: list[Window] = []
    total = 0
    kept_train = 0
    kept_test = 0
    dropped_transition = 0
    dropped_boundary = 0

    for win, reason in _iter_candidate_windows(trial, trial_split, window, step):
        total += 1
        if win is not None:
            windows.append(win)
            if win.split == "train":
                kept_train += 1
            else:
                kept_test += 1
        elif reason == "transition":
            dropped_transition += 1
        else:
            dropped_boundary += 1

    counts = WindowCounts(
        total_candidates=total,
        kept_train=kept_train,
        kept_test=kept_test,
        dropped_transition=dropped_transition,
        dropped_boundary=dropped_boundary,
    )

    return windows, counts


def segment_trial(
    trial: Trial,
    trial_split: TrialSplit,
    window_size_ms: float = WINDOW_SIZE_MS,
    window_overlap: float = WINDOW_OVERLAP,
    sampling_rate: int = SAMPLING_RATE,
) -> list[Window]:
    """
    The lean, reusable primitive: valid windows only, no diagnostics. This
    is what a future feature-extraction stage calls itself, per trial, on
    demand - v0.5.0's own orchestrator uses `segment_trial_detailed`
    instead, since it also needs the drop-reason counts for reporting.
    """

    windows, _ = segment_trial_detailed(
        trial, trial_split, window_size_ms, window_overlap, sampling_rate
    )
    return windows
