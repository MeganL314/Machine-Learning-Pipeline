import argparse, yaml
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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

    for idx, each in matrix.iterrows():
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

        print("rows, cols before:", X.shape)
        X_num = X.apply(pd.to_numeric, errors="coerce")
        na_per_col = X_num.isna().sum().sort_values(ascending=False)
        print("top NA columns:\n", na_per_col.head(10))
        print("# rows with any NA across features:", X_num.isna().any(axis=1).sum())

        X = X.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

        print(len(feats_present))
        print("\n\n")

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = Path(args.out_root) / "distribution" / stamp  / f"{dataset}__{featureset}"
        out_dir.mkdir(parents=True, exist_ok=True)


        with PdfPages(out_dir / f"{featureset}_histograms.pdf") as pdf:
            for feature in feats_present:
                #print(feature)
                #print(X[feature])
                fig, ax = plt.subplots(figsize=(6,4))
                ax.hist(X[feature], bins=30, edgecolor="black")
                ax.set_title(f"Histogram: {feature}")
                ax.set_xlabel(feature); ax.set_ylabel("Count")
                fig.tight_layout()
                pdf.savefig(fig)                         # append this page
                plt.close(fig) 



if __name__ == "__main__":
    main()
