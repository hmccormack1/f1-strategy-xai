"""Explanations (SHAP for XGBoost, EBM term contributions) and the experiments
that test them: agreement, stability, faithfulness, distribution shift.

Both are in log-odds of pitting, so they can be compared directly.
"""
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import average_precision_score

from .features import FEATURE_COLS


def shap_values(xgb_model, X):
    return shap.TreeExplainer(xgb_model).shap_values(X[FEATURE_COLS])


def ebm_contribs(ebm, X):
    names = list(ebm.term_names_)
    c = ebm.eval_terms(X[FEATURE_COLS])
    return pd.DataFrame(c, columns=names)[FEATURE_COLS].values


def global_importance(contribs):
    return pd.Series(np.abs(contribs).mean(axis=0), index=FEATURE_COLS)


def top_k_sets(contribs, k=3):
    idx = np.argsort(-np.abs(contribs), axis=1)[:, :k]
    return [set(r) for r in idx]


def jaccard(a_sets, b_sets):
    return float(np.mean([len(a & b) / len(a | b) for a, b in zip(a_sets, b_sets)]))


# agreement
def shape_curves(X, contribs, feature, bins=12):
    """Mean contribution of `feature` per value bin."""
    x = X[feature].values
    uniq = np.unique(x)
    if len(uniq) <= bins:
        groups = pd.Series(x)
    else:
        groups = pd.qcut(pd.Series(x).rank(method="first"), bins, labels=False)
    df = pd.DataFrame({"g": groups.values, "x": x, "c": contribs[:, FEATURE_COLS.index(feature)]})
    out = df.groupby("g").agg(x=("x", "mean"), c=("c", "mean"), n=("c", "size"))
    return out.sort_values("x")


def agreement(X, shap_c, ebm_c, k=3):
    gi_s, gi_e = global_importance(shap_c), global_importance(ebm_c)
    rho = spearmanr(gi_s, gi_e).correlation
    shape_r = {}
    for f in FEATURE_COLS:
        a, b = shape_curves(X, shap_c, f), shape_curves(X, ebm_c, f)
        if len(a) > 2 and a["c"].std() > 1e-9 and b["c"].std() > 1e-9:
            shape_r[f] = round(float(pearsonr(a["c"], b["c"])[0]), 3)
    return {
        "global_rank_spearman": round(float(rho), 3),
        f"local_top{k}_jaccard": round(jaccard(top_k_sets(shap_c, k), top_k_sets(ebm_c, k)), 3),
        "shape_correlation": shape_r,
        "importance_shap": gi_s.round(4).to_dict(),
        "importance_ebm": gi_e.round(4).to_dict(),
    }


# stability
def stability(make_model, X_train, y_train, X_test, seeds=(0, 1, 2, 3, 4), k=3, log=print):
    imps, tops = [], []
    for s in seeds:
        log(f"    seed {s}")
        m = make_model(s).fit(X_train[FEATURE_COLS], y_train)
        sv = shap_values(m, X_test)
        imps.append(global_importance(sv))
        tops.append(top_k_sets(sv, k))
    imp = pd.DataFrame(imps)
    pairs = [(i, j) for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
    rank_rho = [spearmanr(imp.iloc[i], imp.iloc[j]).correlation for i, j in pairs]
    local = [jaccard(tops[i], tops[j]) for i, j in pairs]
    return {
        "mean_pairwise_rank_spearman": round(float(np.mean(rank_rho)), 3),
        f"mean_pairwise_local_top{k}_jaccard": round(float(np.mean(local)), 3),
        "importance_mean": imp.mean().round(4).to_dict(),
        "importance_std": imp.std().round(4).to_dict(),
    }


# faithfulness (deletion test)
def deletion_curve(model, X, ranking, background, ks=(0, 1, 2, 3, 4, 5), n_draws=10, seed=0):
    """Replace each row's top-k ranked features with values from random background
    rows and return the mean predicted probability for each k."""
    rng = np.random.default_rng(seed)
    Xv = X[FEATURE_COLS].values.astype(float)
    bg = background[FEATURE_COLS].values.astype(float)
    order = np.argsort(-ranking, axis=1)
    out = []
    for k in ks:
        probs = []
        for _ in range(n_draws if k else 1):
            Xk = Xv.copy()
            if k:
                donors = bg[rng.integers(0, len(bg), len(Xk))]
                cols = order[:, :k]
                rows = np.arange(len(Xk))[:, None]
                Xk[rows, cols] = donors[rows, cols]
            probs.append(model.predict_proba(pd.DataFrame(Xk, columns=FEATURE_COLS))[:, 1].mean())
        out.append(float(np.mean(probs)))
    return out


def faithfulness(xgb_model, X, shap_c, ebm_c, background, seed=0):
    """Run on laps where XGBoost predicts a pit (positive contribution = toward pitting)."""
    rng = np.random.default_rng(seed)
    ks = (0, 1, 2, 3, 4, 5)
    curves = {
        "SHAP ranking": deletion_curve(xgb_model, X, shap_c, background, ks),
        "EBM ranking (applied to XGBoost)": deletion_curve(xgb_model, X, ebm_c, background, ks),
        "Random ranking": deletion_curve(xgb_model, X, rng.random(shap_c.shape), background, ks),
    }
    return {"k": list(ks), "curves": {k: [round(v, 4) for v in c] for k, c in curves.items()}}


# distribution shift
def shift_report(xgb_model, X_test, X_shift):
    s_te, s_sh = shap_values(xgb_model, X_test), shap_values(xgb_model, X_shift)
    gi_te, gi_sh = global_importance(s_te), global_importance(s_sh)
    drift = {}
    for f in FEATURE_COLS:
        a, b = X_test[f].astype(float), X_shift[f].astype(float)
        drift[f] = round(float((b.mean() - a.mean()) / (a.std() + 1e-9)), 3)  # mean shift in SDs
    return {
        "importance_test": gi_te.round(4).to_dict(),
        "importance_shift": gi_sh.round(4).to_dict(),
        "importance_rank_spearman": round(float(spearmanr(gi_te, gi_sh).correlation), 3),
        "feature_mean_shift_sd": drift,
    }


# design iterations
def design_iterations(make_model, train, test, seeds=(0, 1, 2, 3, 4)):
    """Test-season PR-AUC for each version of the feature set.

    v1: original (absolute temp, no SC-upcoming)
    v2: + SC-upcoming, after SHAP showed the model ignored the safety car
    v3: temp change instead of absolute temp (final)
    plus a no-temperature control
    """
    final = list(FEATURE_COLS)
    base = [c for c in final if c not in ("sc_upcoming", "temp_change")]
    variants = {
        "v1 original (absolute temp, no SC-upcoming)": base + ["track_temp"],
        "v2 + SC/VSC deployed feature": base + ["track_temp", "sc_upcoming"],
        "No temperature (control)": base + ["sc_upcoming"],
        "v3 final (temp change instead of absolute)": final,
    }
    out = {}
    for name, cols in variants.items():
        r = [average_precision_score(test["label"], make_model(s).fit(train[cols], train["label"])
                                     .predict_proba(test[cols])[:, 1]) for s in seeds]
        out[name] = {"pr_auc_mean": round(float(np.mean(r)), 4), "pr_auc_sd": round(float(np.std(r)), 4),
                     "features": cols}
    return out
