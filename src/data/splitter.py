"""
Train / Test Splitting

Two split strategies are provided because they answer different research
questions and are NOT interchangeable:

- Subject-dependent (`repetition_split`): every subject appears in both
  train and test, split by repetition. This measures how well the model
  recognizes gestures for a subject it has already seen other repetitions
  from - the standard NinaPro benchmarking protocol.

- Subject-independent (`subject_split`): entire subjects are held out for
  test. This measures generalization to a new, unseen user - the more
  relevant question for a system meant to work on a new patient without
  per-patient calibration, but a much harder task and typically yields
  lower accuracy.

Both split on `Trial.reps` / `Trial.labels` (the refined rerepetition /
restimulus fields when available), not the raw stimulus/repetition fields.

Splits are returned as a per-trial boolean mask (train/test) over each
trial's sample rows, not as a flattened list of per-sample records - with
~12.5M total samples in this dataset, materializing one Python object per
sample would mean tens of millions of objects and a slow, memory-heavy
result. A mask is also what the windowing/feature-extraction stage will
actually need: a window is only valid if every sample inside it falls on
the same side of the split.

Whichever strategy is used, state it explicitly in any report/paper that
depends on it - the two are not comparable to each other or to numbers
from other papers unless the split is the same.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from config.logging_config import logger
from config.settings import DEFAULT_TEST_REPETITIONS, RANDOM_SEED
from src.data.datamodels import Subject


@dataclass
class TrialSplit:
    subject_id: str
    trial_filename: str
    train_mask: np.ndarray  # bool, shape (samples,)
    test_mask: np.ndarray  # bool, shape (samples,)

    @property
    def train_size(self) -> int:
        return int(self.train_mask.sum())

    @property
    def test_size(self) -> int:
        return int(self.test_mask.sum())


@dataclass
class SplitResult:
    strategy: str
    trials: list[TrialSplit] = field(default_factory=list)
    _index: dict[tuple[str, str], TrialSplit] = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def train_size(self) -> int:
        return sum(t.train_size for t in self.trials)

    @property
    def test_size(self) -> int:
        return sum(t.test_size for t in self.trials)

    def get(self, subject_id: str, trial_filename: str) -> TrialSplit | None:
        """
        O(1) lookup via a lazily-built (subject_id, trial_filename) index,
        rebuilt whenever `self.trials` has grown since the index was last
        built - callers are free to append to `trials` (as `repetition_split`/
        `subject_split` do while constructing a result) without going
        through a setter.
        """
        if len(self._index) != len(self.trials):
            self._index = {
                (t.subject_id, t.trial_filename): t for t in self.trials
            }
        return self._index.get((subject_id, trial_filename))


def repetition_split(
    subjects: list[Subject],
    test_repetitions: list[int] | None = None,
    exclude_rest: bool = False,
) -> SplitResult:
    """
    Subject-dependent split: every subject contributes to both train and
    test, partitioned by repetition index.

    Args:
        subjects: Loaded dataset.
        test_repetitions: Repetition numbers held out for test. Defaults
            to `config.settings.DEFAULT_TEST_REPETITIONS`.
        exclude_rest: If True, samples labeled 0 (rest) are excluded from
            both masks. NinaPro's rest periods dominate the sample count
            (they are longer than any single gesture), so many published
            results exclude or subsample rest - decide deliberately rather
            than by default.
    """

    test_reps = set(test_repetitions or DEFAULT_TEST_REPETITIONS)

    result = SplitResult(strategy="repetition_split")

    for subject in subjects:
        for trial in subject.trials:
            reps = np.asarray(trial.reps).reshape(-1)
            labels = np.asarray(trial.labels).reshape(-1)

            in_test = np.isin(reps, list(test_reps))

            if exclude_rest:
                keep = labels != 0
                in_test = in_test & keep
                in_train = (~np.isin(reps, list(test_reps))) & keep
            else:
                in_train = ~in_test

            result.trials.append(
                TrialSplit(
                    subject_id=subject.subject_id,
                    trial_filename=trial.filename,
                    train_mask=in_train,
                    test_mask=in_test,
                )
            )

    logger.info(
        f"repetition_split: train={result.train_size:,} test={result.test_size:,} "
        f"(test repetitions={sorted(test_reps)})"
    )

    return result


def subject_split(
    subjects: list[Subject],
    test_subject_ids: list[str] | None = None,
    n_test_subjects: int = 5,
    seed: int = RANDOM_SEED,
    exclude_rest: bool = False,
) -> SplitResult:
    """
    Subject-independent split: whole subjects are held out for test.

    Args:
        subjects: Loaded dataset.
        test_subject_ids: Explicit subject IDs to hold out. If omitted,
            `n_test_subjects` are chosen at random using `seed`.
        n_test_subjects: How many subjects to hold out when
            `test_subject_ids` is not given.
        seed: Random seed for the subject sample (ignored if
            `test_subject_ids` is given). Defaults to the project-wide
            `RANDOM_SEED` so this is reproducible by default.
        exclude_rest: See `repetition_split`.
    """

    all_ids = [s.subject_id for s in subjects]

    if test_subject_ids is None:
        rng = random.Random(seed)
        test_subject_ids = rng.sample(all_ids, k=min(n_test_subjects, len(all_ids)))

    test_ids = set(test_subject_ids)

    result = SplitResult(strategy="subject_split")

    for subject in subjects:
        is_test_subject = subject.subject_id in test_ids

        for trial in subject.trials:
            labels = np.asarray(trial.labels).reshape(-1)
            keep = labels != 0 if exclude_rest else np.ones_like(labels, dtype=bool)

            train_mask = keep & (not is_test_subject)
            test_mask = keep & is_test_subject

            result.trials.append(
                TrialSplit(
                    subject_id=subject.subject_id,
                    trial_filename=trial.filename,
                    train_mask=train_mask,
                    test_mask=test_mask,
                )
            )

    logger.info(
        f"subject_split: train={result.train_size:,} test={result.test_size:,} "
        f"(held-out subjects={sorted(test_ids)})"
    )

    return result
