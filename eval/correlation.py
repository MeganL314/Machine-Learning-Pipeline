import argparse, yaml
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

def load_yaml(p): 
    with open(p, "r") as f: 
        return yaml.safe_load(f)

def load_table(p: Path) -> pd.DataFrame:
    if p.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(p)
    sep = "," if p.suffix.lower()==".csv" else "\t"
    return pd.read_csv(p, sep=sep)


def load_feature_list(dir_: Path, name: str):
    txt = (dir_ / f"{name}.txt").read_text().splitlines()
    return [x.strip() for x in txt if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--feature_dir", required=True)
    ap.add_argument("--out_root", default="../runs")
    args = ap.parse_args()

    settings = load_yaml(args.settings)
    data_cfg = load_yaml(args.data)
    matrix = pd.read_csv(args.matrix)

    # take just the first lesson row for now
    for each in matrix.iterrows():
        dataset = str(each["dataset"])
        featureset = str(each["featureset"])

        data_path = Path(data_cfg["datasets"][dataset]["path"])
        df = load_table(data_path)

        feats = load_feature_list(Path(args.feature_dir), featureset)
        feats_present = [c for c in feats if c in df.columns]

        if not feats_present:
            raise SystemExit("None of the requested features were found in the data.")

        # make numeric & drop rows with any NA in these columns
        X = df[feats_present].replace({"No Data": np.nan, "#VALUE!": np.nan})
        X = X.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

        # Spearman correlation (robust for ranks)
        corr = X.corr(method="spearman")

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = Path(args.out_root) / stamp / "correlations" / f"{dataset}__{featureset}"
        out_dir.mkdir(parents=True, exist_ok=True)

        corr.to_csv(out_dir / "corr_spearman.csv")
        print(f"saved: {out_dir/'corr_spearman.csv'}")

        # also print the top absolute correlations as a quick peek
        tri = corr.where(~np.tril(np.ones(corr.shape), k=0).astype(bool))
        pairs = (
            tri.stack()
            .abs()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"level_0":"feat1","level_1":"feat2",0:"abs_rho"})
        )
        print("top correlations:")
        print(pairs.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
