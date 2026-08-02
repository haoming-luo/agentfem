"""Small, evidence-preserving surrogate training workflow."""

from __future__ import annotations

from dataclasses import dataclass

from ..datasets import DatasetSplit, ScientificDataset
from .domain import BoxApplicabilityDomain, GuardedSurrogate
from .linear import RidgeSurrogate


@dataclass(frozen=True)
class SurrogateTrainingRun:
    """A trained model together with its independent validation evidence."""

    model: object
    split: DatasetSplit
    validation: object

    @property
    def accepted(self) -> bool | None:
        return self.validation.accepted

    def guard(self, *, fallback=None, applicability=None) -> GuardedSurrogate:
        """Attach an applicability domain and optional high-fidelity fallback."""

        domain = applicability or BoxApplicabilityDomain.from_dataset(self.split.train)
        return GuardedSurrogate(self.model, domain, fallback=fallback)

    def summary(self) -> dict[str, object]:
        return {
            "kind": "surrogate_training_run",
            "model": (
                self.model.summary()
                if hasattr(self.model, "summary")
                else type(self.model).__name__
            ),
            "split": {
                "seed": self.split.seed,
                "validation_fraction": self.split.validation_fraction,
                "training_samples": len(self.split.train.samples),
                "validation_samples": len(self.split.validation.samples),
            },
            "validation": self.validation.summary(),
        }


def train(
    dataset: ScientificDataset,
    *,
    estimator=None,
    validation_fraction: float = 0.2,
    seed: int = 0,
    thresholds=None,
) -> SurrogateTrainingRun:
    """Split, fit, and independently validate one surrogate estimator.

    The default is the transparent ridge baseline. PyTorch or other estimators
    participate through the same small ``fit``/``validate`` protocol; AgentFEM
    continues to own scientific schema, evidence, and applicability rather
    than the training implementation.
    """

    if len(dataset.samples) < 3:
        raise ValueError(
            "Surrogate training requires at least three successful samples "
            "so training and validation evidence are not the same data."
        )
    selected = estimator or RidgeSurrogate()
    if not hasattr(selected, "fit"):
        raise TypeError("estimator must provide fit(ScientificDataset).")
    split = dataset.split(
        validation_fraction=validation_fraction,
        seed=seed,
    )
    trained = selected.fit(split.train)
    if not hasattr(trained, "validate"):
        raise TypeError("A fitted estimator must provide validate(dataset, thresholds=...).")
    validation = trained.validate(split.validation, thresholds=thresholds)
    return SurrogateTrainingRun(trained, split, validation)


__all__ = ["SurrogateTrainingRun", "train"]
