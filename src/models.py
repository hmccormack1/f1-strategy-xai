"""Models: a tyre-age baseline, XGBoost (black box) and an EBM with
interactions off (a pure GAM).

Split by season. The last 25% of training races pick the decision threshold,
then each model is refit on all training races.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             precision_score, recall_score, f1_score)

from .features import FEATURE_COLS

DEFAULT_SPLIT = {"train": [2023, 2024], "test": 2025, "shift": 2026}


def make_xgb(seed=0):
    return xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        eval_metric="aucpr", random_state=seed, n_jobs=-1)


def make_ebm(seed=0):
    return ExplainableBoostingClassifier(interactions=0, random_state=seed, n_jobs=-1)


class RateBaseline:
    """Pit probability = historical pit rate for this compound and tyre-age bucket."""
    def __init__(self, bucket=3, prior=20):
        self.bucket, self.prior = bucket, prior

    def _keys(self, X):
        return list(zip(X["compound"], (X["tyre_life"] // self.bucket).clip(upper=20)))

    def fit(self, X, y):
        self.rate_ = float(np.mean(y))
        tab = pd.DataFrame({"k": self._keys(X), "y": np.asarray(y)}).groupby("k")["y"].agg(["sum", "count"])
        # shrink small groups toward the overall rate
        self.table_ = ((tab["sum"] + self.prior * self.rate_) / (tab["count"] + self.prior)).to_dict()
        return self

    def predict_proba(self, X):
        p = np.array([self.table_.get(k, self.rate_) for k in self._keys(X)])
        return np.column_stack([1 - p, p])


def best_threshold(y, p):
    """Threshold that maximizes F1 (balances precision and recall)."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[np.argmax(f1[:-1])])


def scores(y, p, threshold):
    yhat = (p >= threshold).astype(int)
    return {
        "n_rows": int(len(y)), "pit_rate": round(float(np.mean(y)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4) if len(set(y)) > 1 else None,
        "precision": round(float(precision_score(y, yhat, zero_division=0)), 4),
        "recall": round(float(recall_score(y, yhat, zero_division=0)), 4),
        "f1": round(float(f1_score(y, yhat, zero_division=0)), 4),
    }


def split(feat, cfg=DEFAULT_SPLIT):
    tr = feat[feat["Year"].isin(cfg["train"])]
    te = feat[feat["Year"] == cfg["test"]]
    sh = feat[feat["Year"] == cfg["shift"]] if cfg.get("shift") else feat.iloc[0:0]
    return tr, te, sh


def train_all(feat, cfg=DEFAULT_SPLIT, seed=0, log=print):
    """Fit baseline, XGBoost and EBM; choose thresholds; return models + metrics."""
    tr, te, sh = split(feat, cfg)
    races = sorted(tr["race_id"].unique())  # "YYYY_RR" sorts by date
    val_races = set(races[int(len(races) * 0.75):])
    fit, val = tr[~tr["race_id"].isin(val_races)], tr[tr["race_id"].isin(val_races)]
    makers = {"baseline": RateBaseline, "xgboost": lambda: make_xgb(seed), "ebm": lambda: make_ebm(seed)}

    models, thresholds, metrics = {}, {}, {}
    for name, make in makers.items():
        log(f"  {name}: choosing threshold on the last {len(val_races)} training races ...")
        m = make().fit(fit[FEATURE_COLS], fit["label"])
        thresholds[name] = best_threshold(val["label"], m.predict_proba(val[FEATURE_COLS])[:, 1])
        log(f"  {name}: fitting on {cfg['train']} ...")
        m = make().fit(tr[FEATURE_COLS], tr["label"])
        models[name] = m
        metrics[name] = {}
        for label, part in [(str(cfg["test"]), te), (str(cfg.get("shift")), sh)]:
            if len(part):
                p = m.predict_proba(part[FEATURE_COLS])[:, 1]
                metrics[name][label] = scores(part["label"].values, p, thresholds[name])
        metrics[name]["threshold"] = round(thresholds[name], 4)
    return models, thresholds, metrics
