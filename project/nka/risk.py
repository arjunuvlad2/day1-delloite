"""Synthetic loan-default model with SHAP explanations and a fairness check (Day 5 lab 5A)."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_k, "1")
import warnings
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
WORDS = {"income": "income level", "debt_ratio": "debt-to-income ratio", "credit_history_years": "length of credit history",
         "loan_amount": "requested loan amount", "employment_years": "time in employment", "postcode_risk": "postcode risk score"}


def build_risk_model(seed: int = 42):
    """Return (model, X, y, group): X, y, group are the TEST split; model.auc_ holds the test AUC."""
    rng = np.random.default_rng(seed); n = 4000
    isB = rng.random(n) < 0.4
    group = np.where(isB, "B", "A")
    df = pd.DataFrame({
        "income": np.clip(rng.lognormal(10.9, 0.4, n) * np.where(isB, 0.9, 1), 15000, None).round(-2),
        "debt_ratio": np.clip(rng.normal(0.35, 0.12, n) + np.where(isB, 0.03, 0), 0.02, 0.9),
        "credit_history_years": np.clip(rng.gamma(3, 3, n), 0, 40),
        "loan_amount": np.clip(rng.lognormal(9.8, 0.5, n), 2000, None).round(-2),
        "employment_years": np.clip(rng.gamma(2.5, 2.5, n), 0, 35),
        "postcode_risk": np.clip(0.2 + 0.5 * isB + rng.normal(0, 0.15, n), 0, 1),    # proxy for group
    })
    z = (-1.2 + 4 * (df.debt_ratio - 0.35) - 1.8 * (np.log(df.income) - 10.9) + 0.9 * (np.log(df.loan_amount) - 9.8)
         - 0.06 * df.credit_history_years - 0.05 * df.employment_years + 0.8 * (df.postcode_risk - 0.35))
    default = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    Xtr, Xte, ytr, yte, _, gte = train_test_split(df, default, group, test_size=0.3, random_state=seed, stratify=default)
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=seed).fit(Xtr, ytr)
    model.auc_ = float(roc_auc_score(yte, model.predict_proba(Xte)[:, 1]))
    model.train_data_ = (Xtr, ytr)
    return model, Xte, pd.Series(yte, index=Xte.index), pd.Series(gte, index=Xte.index)


def _shap(model, X):
    import shap
    return shap.TreeExplainer(model).shap_values(X)


def explain_global(model, X, rows: int = 500) -> pd.DataFrame:
    """Mean |SHAP| per feature over the first `rows` rows, sorted high to low."""
    S = _shap(model, X.iloc[:rows])
    return (pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(S).mean(0)})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def explain_local(model, X, row_idx: int) -> pd.DataFrame:
    """One applicant (positional row index): value and SHAP contribution per feature, largest first."""
    S = _shap(model, X.iloc[[row_idx]])[0]
    df = pd.DataFrame({"feature": X.columns, "value": X.iloc[row_idx].values, "contribution": S})
    return df.reindex(df.contribution.abs().sort_values(ascending=False).index).reset_index(drop=True)


def reason_text(contributions: pd.DataFrame, min_push: float = 0.1) -> str:
    """Plain-English adverse-action reason from the features that push default risk up."""
    top = [WORDS.get(f, f) for f, c in zip(contributions.feature, contributions.contribution) if c > min_push][:3]
    if not top: return "No single factor pushed the decision towards decline."
    return "Your application was declined mainly because of your " + ", ".join(top[:-1]) + (" and " if len(top) > 1 else "") + top[-1] + "."


def fairness_report(model, X, group, threshold: float = 0.20, drop=None, y=None) -> pd.DataFrame:
    """Approval rate per group (approve if P(default) < threshold) and the disparate impact ratio.

    drop: a feature name (or list) to remove; the model is retrained without it on the stored training data.
    """
    drop = [drop] if isinstance(drop, str) else list(drop or [])
    cols = [c for c in X.columns if c not in drop]
    m = model
    if drop:
        Xtr, ytr = model.train_data_
        m = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42).fit(Xtr[cols], ytr)
    p = m.predict_proba(X[cols])[:, 1]
    appr = pd.Series(p < threshold, index=X.index).groupby(np.asarray(group)).mean()
    ratio = appr.min() / appr.max()
    out = pd.DataFrame({"features": ["drop " + ",".join(drop) if drop else "all"], "approval_A": appr["A"], "approval_B": appr["B"],
                        "di_ratio": ratio, "flag": "FLAG (<0.8)" if ratio < 0.8 else "ok"})
    if y is not None: out["auc"] = roc_auc_score(y, p)
    return out
