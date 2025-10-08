import argparse, yaml
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


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
        print(each)
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

        stamp = datetime.now().strftime("%Y-%m-%d_%H")
        out_dir = Path(args.out_root) / "correlations" / stamp  / f"{dataset}__{featureset}"
        out_dir.mkdir(parents=True, exist_ok=True)

        corr.to_csv(out_dir / "corr_spearman.csv", float_format="%.5f")
        print(f"saved: {out_dir/'corr_spearman.csv'}")

        # also print the top absolute correlations
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


        ## Print corr plots for each combination of features
        combinations = [] # list of combination pairs
        with PdfPages(out_dir / f"{featureset}_CorrelationPlots.pdf") as pdf:
            for each_feature in feats_present:
                for second_feature in feats_present:
                    if each_feature != second_feature:
                        new_combo = [each_feature, second_feature]

                        if new_combo not in combinations and new_combo[::-1] not in combinations:
                            ## plot X[each_feature] vs. X[second_feature]
                            rho, pval = spearmanr(X[each_feature], X[second_feature])
                            plt.figure(figsize=(5, 4))
                            plt.scatter(X[each_feature], X[second_feature], alpha=0.7)
                            plt.title(f"{each_feature} vs {second_feature}")
                            plt.xlabel(each_feature)
                            plt.ylabel(second_feature)
                            plt.text(0.05, 0.95,
                                     f"Spearman r = {rho:.2f}\np = {pval:.3e}",
                                     transform=plt.gca().transAxes,
                                     ha="left", va="top",
                                     fontsize=9,
                                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))

                            # tidy layout and save to PDF
                            
                            plt.tight_layout()
                            pdf.savefig()
                            plt.close()


                            ## add each_feature and second_feature to the combinations list
                            combinations.append(new_combo)


if __name__ == "__main__":
    main()
