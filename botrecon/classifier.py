"""LightGBM classifier [M 3.1]: gbdt, 31 leaves, lr 0.05, <= 1,000 trees,
early stopping after 50 rounds without improvement, binary objective.

Not executed in the environment where this package was written (no LightGBM).
"""
from __future__ import annotations

import numpy as np


class LGBMStage:
    def __init__(self, cfg, seed: int):
        import lightgbm as lgb
        self.lgb = lgb
        self.cfg = cfg
        self.model = lgb.LGBMClassifier(
            boosting_type="gbdt", num_leaves=cfg.lgbm_num_leaves, learning_rate=cfg.lgbm_lr,
            n_estimators=cfg.lgbm_n_estimators, objective="binary", random_state=seed,
            verbose=-1,
        )

    def fit(self, X, y, X_es, y_es):
        self.model.fit(X, y, eval_set=[(X_es, y_es)], eval_metric="binary_logloss",
                       callbacks=[self.lgb.early_stopping(self.cfg.lgbm_early_stopping, verbose=False)])
        self.best_iteration = int(self.model.best_iteration_ or self.cfg.lgbm_n_estimators)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
