# github/basic-scripts/pipeline/rfe_survival_cv.py
import argparse, json, os, sys, time, hashlib, math, random
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import yaml

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.pipeline import make_pipeline

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored

# ----------------------------
# Helpers (kept local to file for simplicity; you can move to src/ later)
# ----------------------------

def load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_table(path: Path) -> pd.DataFrame:
    # auto-detect CSV vs Parquet
    if path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    elif path.suffix.lower() in [".csv", ".tsv"]:
        sep = "," if path.suffix.lower() == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported file type: {path}")

def load_feature_list(feature_dir: Path, name: str) -> list:
    file = feature_dir / f"{name}.txt"
    if not file.exists():
        raise FileNotFoundError(f"Feature set file not found: {file}")
    feats = [line.strip() for line in file.read_text().splitlines() if line.strip()]
    return feats

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_structured_surv(event_bool: np.ndarray, time_vals: np.ndarray,
                       event_col="event", time_col="time"):
    # sksurv expects a structured array with dtype [('event', '?'), ('time', '<f8')]
    arr = np.array(list(zip(event_bool.astype(bool), time_vals.astype(float))),
                   dtype=[(event_col, '?'), (time_col, '<f8')])
    return arr

def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # replace common non-numeric tokens, then coerce
    df = df.replace({"No Data": np.nan, "#VALUE!": np.nan, "": np.nan})
    return df.apply(pd.to_numeric, errors="coerce")

def filter_slice(df: pd.DataFrame, schema: dict, arm, group, timepoint):
    # Copy to avoid SettingWithCopy warnings
    out = df.copy()

    # Arm filter (if not "all")
    arm_col = schema.get("arm")
    if arm_col and isinstance(arm, str) and arm.lower() != "all":
        out = out[out[arm_col] == arm]

    # Group filter (if not "all")
    group_col = schema.get("group")
    if group_col and isinstance(group, str) and group.lower() != "all":
        out = out[out[group_col] == group]

    # Timepoint filter: we keep it simple—if a single timepoint value is given,
    # filter on that value. If your dataset encodes baseline/week2 differently,
    # adapt here (or pre-derive change variables in a data prep step).
    tp_col = schema.get("timepoint")
    if tp_col and isinstance(timepoint, str) and timepoint.lower() not in ["", "all", "twotimes"]:
        out = out[out[tp_col] == timepoint]

    return out

def timestamp_slug():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ----------------------------
# Core RFE + CV for survival
# ----------------------------

def run_rfe_survival_cv(df: pd.DataFrame,
                        feature_names: list,
                        schema_cols: dict,
                        model_alpha: float,
                        n_repeats: int,
                        n_folds_list: list,
                        max_features: int,
                        rfe_step: int,
                        seed: int,
                        out_dir: Path):
    ensure_dir(out_dir / "fit")
    ensure_dir(out_dir / "cv")

    # Build y (structured array) and X (numeric)
    event_col = schema_cols["survival_event"]
    time_col = schema_cols["survival_time"]

    # Keep only necessary columns: features + event + time
    keep_cols = [c for c in feature_names if c in df.columns] + [event_col, time_col]
    missing_feats = [c for c in feature_names if c not in df.columns]
    if missing_feats:
        print(f"[WARN] {len(missing_feats)} features not found in data; ignoring e.g. {missing_feats[:5]}")

    sub = df[keep_cols].copy()
    sub = safe_numeric(sub)
    sub = sub.dropna(subset=[event_col, time_col])  # require survival fields

    # Build X, y
    X = sub[[c for c in feature_names if c in sub.columns]].copy()
    y = to_structured_surv(event_bool=sub[event_col].astype(bool).values,
                           time_vals=sub[time_col].values,
                           event_col=event_col, time_col=time_col)

    # Drop rows with any NA in features (simple approach)
    mask_complete = ~X.isna().any(axis=1)
    X = X.loc[mask_complete]
    y = y[mask_complete.values]

    # Standardize
    scaler = StandardScaler()

    # Outputs
    metrics_rows = []
    folds_json = {}
    rng = np.random.RandomState(seed)

    for n_folds in n_folds_list:
        for rep in range(1, n_repeats + 1):
            rep_seed = int(rng.randint(0, 2**32 - 1))
            # Stratify by event indicator for more balanced folds
            strat = pd.Series(y[event_col].astype(int))
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rep_seed)

            fold_assign = []
            for fold_ix, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(strat)), strat), start=1):
                fold_assign.append({
                    "fold": fold_ix,
                    "train_idx": train_idx.tolist(),
                    "test_idx": test_idx.tolist()
                })

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
                X_test  = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)

                # Try 1..max_features
                for k in range(1, min(max_features, X.shape[1]) + 1):
                    # Base estimator (CoxPH)
                    base = CoxPHSurvivalAnalysis(alpha=model_alpha)
                    # RFE selects exactly k features (by recursive elimination)
                    selector = RFE(estimator=base, n_features_to_select=k, step=rfe_step)
                    selector.fit(X_train.values, y_train)

                    support_mask = selector.support_
                    selected_cols = X.columns[support_mask].tolist()

                    # Fit Cox on the reduced feature set to compute C-index
                    X_train_sel = X_train[selected_cols].values
                    X_test_sel  = X_test[selected_cols].values

                    # Refit a fresh CoxPH (safety) and evaluate
                    cox = CoxPHSurvivalAnalysis(alpha=model_alpha)
                    cox.fit(X_train_sel, y_train)
                    risk_scores = cox.predict(X_test_sel)
                    cindex, _, _ = concordance_index_censored(
                        y_test[event_col], y_test[time_col], risk_scores
                    )

                    # Store row
                    metrics_rows.append({
                        "folds": n_folds,
                        "repeat": rep,
                        "fold": fold_ix,
                        "num_features": k,
                        "c_index": float(cindex),
                        "selected_features": ";".join(selected_cols)
                    })

            folds_json[f"{n_folds}f_rep{rep}"] = fold_assign

    # Save artifacts
    pd.DataFrame(metrics_rows).to_csv(out_dir / "cv" / "metrics.csv", index=False)

    # Summary (mean/sd C-index per k and per folds)
    summary = (pd.DataFrame(metrics_rows)
               .groupby(["folds", "num_features"])["c_index"]
               .agg(["mean", "std"])
               .reset_index())
    summary.to_json(out_dir / "cv" / "summary.json", orient="records", indent=2)

    # Save fold maps for reproducibility
    (out_dir / "cv" / "folds.json").write_text(json.dumps(folds_json, indent=2))

    # Minimal model info
    model_info = {
        "estimator": "CoxPHSurvivalAnalysis",
        "alpha": model_alpha,
        "rfe_step": rfe_step,
        "max_features": max_features,
        "cv": {"repeats": n_repeats, "folds": n_folds_list},
        "seed": seed,
        "n_samples": int(X.shape[0]),
        "n_features_total": int(X.shape[1]),
    }
    (out_dir / "fit" / "model_info.json").write_text(json.dumps(model_info, indent=2))

    # Preprocess summary
    prep = {
        "columns_used": feature_names,
        "columns_present": [c for c in feature_names if c in df.columns],
        "event_col": event_col, "time_col": time_col,
        "n_after_dropna": int(X.shape[0]),
    }
    (out_dir / "preprocess_summary.json").write_text(json.dumps(prep, indent=2))

    print(f"[OK] Wrote artifacts to {out_dir}")

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="RFE Survival CV (YAML-driven)")
    ap.add_argument("--settings", required=True, help="Path to private/settings.yaml")
    ap.add_argument("--data", required=True, help="Path to private/data.yaml")
    ap.add_argument("--matrix", required=True, help="Path to private/matrix.csv")
    ap.add_argument("--feature_dir", required=True, help="Path to private/feature_sets/")
    ap.add_argument("--runs_root", default="../runs", help="Where to write outputs")
    args = ap.parse_args()

    settings = load_yaml(Path(args.settings))
    data_cfg = load_yaml(Path(args.data))
    matrix = pd.read_csv(args.matrix)

    # Seed everything
    seed = int(settings.get("seed", 42))
    np.random.seed(seed)
    random.seed(seed)

    # Schema columns (names in your dataset)
    cols = settings.get("columns", {})
    required = ["survival_event", "survival_time"]
    for r in required:
        if r not in cols:
            raise ValueError(f"Missing required column mapping: columns.{r} in settings.yaml")

    # CV + model params
    folds_list = settings.get("cv", {}).get("folds", [3, 5])
    repeats = int(settings.get("cv", {}).get("repeats", 10))
    max_feats = int(settings.get("rfe", {}).get("max_features", 5))
    rfe_step = int(settings.get("rfe", {}).get("step", 1))
    alpha = float(settings.get("model", {}).get("alpha", 0.001))

    runs_root = Path(args.runs_root)
    stamp = timestamp_slug()

    # Load each dataset once and cache
    data_paths = {k: Path(v["path"]) for k, v in data_cfg.get("datasets", {}).items()}
    cache = {}

    for _, row in matrix.iterrows():
        dataset = str(row["dataset"])
        timepoint = str(row["timepoint"])
        featureset = str(row["featureset"])
        arm = str(row.get("arm", "all"))
        group = str(row.get("group", "all"))

        # Load data (from private path)
        if dataset not in cache:
            if dataset not in data_paths:
                raise ValueError(f"Dataset '{dataset}' not found in data.yaml")
            df = load_table(data_paths[dataset])
            cache[dataset] = df
        else:
            df = cache[dataset]

        # Read feature list
        feats = load_feature_list(Path(args.feature_dir), featureset)

        # Apply filters (arm/group/timepoint)
        df_slice = filter_slice(df, cols, arm, group, timepoint)

        # Choose output directory
        out_dir = (runs_root / stamp / "artifacts" / "rfe_survival" /
                   dataset / timepoint / featureset / f"arm={arm}__group={group}")
        ensure_dir(out_dir)

        # Run RFE + CV
        run_rfe_survival_cv(
            df=df_slice,
            feature_names=feats,
            schema_cols=cols,
            model_alpha=alpha,
            n_repeats=repeats,
            n_folds_list=folds_list,
            max_features=max_feats,
            rfe_step=rfe_step,
            seed=seed,
            out_dir=out_dir
        )

if __name__ == "__main__":
    main()
