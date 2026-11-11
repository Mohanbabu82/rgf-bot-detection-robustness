"""Run configuration.

Each field carries a tag in its comment:
  [M] stated in the manuscript, [R] recalled by co-author (unverified),
  [C] reconstruction choice, [V] unresolved, run as a variant axis.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, replace
from typing import Iterator

from . import provenance as P

DATASETS = ("cresci2017", "cresci2015", "twibot20")   # twibot20: library support only, not in the plan
FAMILIES = ("literal", "corrected")
FOLLOW_GRAPH = ("cresci2015", "twibot20")   # datasets that ship follower/friend edges
CROSS_PARTNER = {"cresci2017": "cresci2015", "cresci2015": "cresci2017", "twibot20": "cresci2017"}


@dataclass(frozen=True)
class RunConfig:
    dataset: str                          # cresci2017 | cresci2015 | twibot20
    family: str = "corrected"             # literal | corrected

    # ---- variant axes [V] ----
    fusion: str = "P3"                    # P3: GCN output only [M Alg.1, 4.2; R]; P1: meta||text||GCN [M 3.1]
    tap: str = "hidden"                   # hidden: 64-d after ReLU | prob: 2-class softmax
    text: str = "B"                       # A: bert-base-uncased CLS on tweets [M 3.1]; B: SBERT on bio [M Alg.1, 4.1]
    cresci_label_rule: str = "explicit"   # explicit [C] | literal_bot_substring [M Alg.1]
    cresci_knn_features: str = "meta"     # meta [M 3.1] | all [M Alg.1 line 24]
    twibot_split: str = "official"        # official [M 4.1] | random [M 4.2]
    scaler_fit: str = "train"             # train [C] | all [M Alg.1 order]

    # ---- protocol [C] ----
    gcn_selection: str = "val_loss"       # fixed (100 epochs) [M] | val_loss [C]
    stacking: str = "blend"               # insample [M Alg.1] | blend [C] | oof [C, prob tap only]
    es_on: str = "val"                    # val [C] | test (DIAGNOSTIC-INVALID only)
    setting: str = "transductive"         # transductive [M] | inductive [C]
    feature_set: str = "core4"            # core4 [M 3.1 "exact set"] | extended6 [C]

    # ---- hyperparameters [M] ----
    k: int = 10
    gcn_hidden: int = 64
    gcn_dropout: float = 0.5
    gcn_lr: float = 0.01
    gcn_weight_decay: float = 5e-4
    gcn_epochs: int = 100
    lgbm_num_leaves: int = 31
    lgbm_lr: float = 0.05
    lgbm_n_estimators: int = 1000
    lgbm_early_stopping: int = 50
    max_tweets: int = 20
    max_tokens: int = 128

    # text checkpoints: A is [M]; B is a [C] choice (manuscript names no checkpoint)
    text_model_a: str = "bert-base-uncased"
    text_model_b: str = "sentence-transformers/all-MiniLM-L6-v2"

    smoke_test: bool = False              # synthetic data / test encoder; never a result

    def validate(self) -> None:
        assert self.dataset in DATASETS, self.dataset
        assert self.family in FAMILIES, self.family
        assert self.fusion in ("P1", "P3")
        assert self.tap in ("hidden", "prob")
        assert self.text in ("A", "B")
        assert self.cresci_label_rule in ("explicit", "literal_bot_substring")
        assert self.cresci_knn_features in ("meta", "all")
        assert self.twibot_split in ("official", "random")
        assert self.scaler_fit in ("train", "all")
        assert self.gcn_selection in ("fixed", "val_loss")
        assert self.stacking in ("insample", "blend", "oof")
        assert self.es_on in ("val", "test")
        assert self.setting in ("transductive", "inductive")
        assert self.feature_set in ("core4", "extended6")
        if self.stacking == "oof" and self.tap != "prob":
            raise ValueError(
                "Out-of-fold stacking needs the probability tap: hidden units of "
                "independently trained GCNs are not aligned across folds."
            )
        if self.setting == "inductive" and self.family != "corrected":
            raise ValueError("Inductive evaluation is defined for the corrected family only.")

    @property
    def provenance(self) -> str:
        if self.smoke_test:
            return P.SMOKE_TEST
        if self.es_on == "test":
            return P.DIAGNOSTIC_INVALID
        return P.REPRODUCED_LITERAL if self.family == "literal" else P.REPRODUCED_CORRECTED

    def to_dict(self) -> dict:
        d = asdict(self)
        d["provenance"] = self.provenance
        return d


def literal(dataset: str, **kw) -> RunConfig:
    """Manuscript-faithful protocol, including its flaws."""
    base = dict(
        dataset=dataset, family="literal", scaler_fit="all",
        gcn_selection="fixed", stacking="insample", es_on="val",
        setting="transductive",
    )
    base.update(kw)
    cfg = RunConfig(**base)
    cfg.validate()
    return cfg


def corrected(dataset: str, **kw) -> RunConfig:
    base = dict(
        dataset=dataset, family="corrected", scaler_fit="train",
        gcn_selection="val_loss", stacking="blend", es_on="val",
        cresci_label_rule="explicit", twibot_split="official",
    )
    base.update(kw)
    cfg = RunConfig(**base)
    cfg.validate()
    return cfg


def literal_grid(dataset: str, **kw) -> Iterator[RunConfig]:
    """Cresci-2017: 32 configs; TwiBot-20: 16 configs."""
    axes = [("fusion", ("P1", "P3")), ("tap", ("hidden", "prob")), ("text", ("A", "B"))]
    if dataset == "cresci2017":
        axes += [("cresci_knn_features", ("meta", "all")),
                 ("cresci_label_rule", ("explicit", "literal_bot_substring"))]
    else:
        axes += [("twibot_split", ("official", "random"))]
    names = [a for a, _ in axes]
    for combo in itertools.product(*[v for _, v in axes]):
        yield literal(dataset, **dict(zip(names, combo)), **kw)


def corrected_grid(dataset: str, **kw) -> Iterator[RunConfig]:
    """8 configs per dataset."""
    for fusion, tap, text in itertools.product(("P1", "P3"), ("hidden", "prob"), ("A", "B")):
        yield corrected(dataset, fusion=fusion, tap=tap, text=text, **kw)


def diagnostic(cfg: RunConfig) -> RunConfig:
    """Same config, LightGBM early-stopped on the TEST set. Never a result."""
    return replace(cfg, es_on="test")
