# github/basic-scripts/pipeline/rfe_survival_cv.py
import argparse, json, os, sys, time, hashlib, math, random
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sksurv.linear_model import CoxPHSurvivalAnalysis  # (left to minimize diffs; unused now)
from sksurv.metrics import concordance_index_censored  # (left to minimize diffs; unused now)
import ast
import csv
from feature_engine.selection import DropCorrelatedFeatures
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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





def create_XY_response(Main_df, feature_index, event_col=None, time_col=None, response_col=None):  
    ## if response_col does not equal NA
    if response_col is not None:
        # join response and features
        response_and_features = list(set(feature_index) | {response_col})
        # 'clean df by removing any rows with NA
        Main_df_cleaned_response = Main_df.dropna(subset=response_and_features)
        # return clean target array and features_array
        target_array = Main_df_cleaned_response[response_col].values
        features_array_response = Main_df_cleaned_response[feature_index]
    
    # if event and time cols do not equal NA
    if event_col is not None and time_col is not None:
        # define event and type
        event_time = [event_col, time_col]
        event_type = [(event_col, '?'), (time_col, '<f8')]
        # join features and outcomes
        survival_and_features = list(set(feature_index) | set(event_time))
        # remove rows with NA
        Main_df_cleaned_survival = Main_df.dropna(subset=survival_and_features)
        # create time array and features array
        time_array = Main_df_cleaned_survival[event_time].to_numpy()
        time_array_new = np.array([tuple(row) for row in time_array], dtype=event_type)
        event_indicator = time_array_new[event_col]
        time_to_event = time_array_new[time_col]
        features_array_survival = Main_df_cleaned_survival[feature_index]
        
    # Combine features only if both response and survival are requested
    if event_col and time_col and response_col:
        common_features = features_array_response.columns.intersection(features_array_survival.columns)
        features_array = Main_df_cleaned_response[common_features]
    elif response_col:
        features_array = features_array_response
    elif event_col and time_col:
        features_array = features_array_survival
        target_array = None

    return event_indicator, time_to_event, features_array, target_array


# def remove_correlated():
# easier pre-built function:
tr = DropCorrelatedFeatures(variables=None,
                            method='spearman',
                            threshold=0.9)


# def transformations
def log_transforms(df, featureset, suffix="_log"):
    # Adds log-transformed versions of selected features
    # keeps original features
    log_cols = {}

    for each in featureset:
        x = df[each]

        # fix if negative
        min_val = x.min()
        if min_val <= 0:
            shift = abs(min_val) + 1e-6
            x = x + shift

        log_cols[each + suffix] = np.log(x)

    out_df = pd.concat([df, pd.DataFrame(log_cols, index=df.index)], axis=1)
    return out_df


# def cross_validation_RFE():
def cross_validation_RFE(X_full, y_full, X_HoldOut, y_HoldOut,
    output_file, splits, mandatory_feature=None, num_features_max=3):

    # Initialize Logistic Regression model
    logistic_model = LogisticRegression(penalty="l2", 
                                        solver="liblinear", max_iter=1000)

    # Open CSV file for writing results
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "NumFeatures", "Validation_AUC", "Validation_Accuracy",
            "Holdout_AUC", "Holdout_Accuracy", "OR > 1 Features", "OR < 1 Features"
        ])

        for random_state in range(1, 11):
            print(f"Running cross-validation with random state: {random_state}")

            for num_features in range(1, num_features_max + 1):
                rfe = RFE(logistic_model, n_features_to_select=num_features)
                cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)

                scaler = StandardScaler()

                for fold, (train_index, test_index) in enumerate(cv.split(X_full, y_full)):
                    X_train, X_test = X_full.iloc[train_index], X_full.iloc[test_index]
                    y_train, y_test = y_full[train_index], y_full[test_index]

                    vt = VarianceThreshold(threshold=1e-8)
                    vt.fit(X_train)
                    keep_vt = vt.get_support()
                    X_train_v = X_train.loc[:, keep_vt]
                    X_test_v  = X_test.loc[:,  keep_vt]
                    X_hold_v  = X_HoldOut.loc[:, keep_vt]

                    current_cols = list(X_train_v.columns)

                    # Cap extreme values before scaling
                    q_low  = X_train_v.quantile(0.005)
                    q_high = X_train_v.quantile(0.995)
                    X_train_v = X_train_v.clip(lower=q_low, upper=q_high, axis=1)
                    X_test_v  = X_test_v.clip(lower=q_low, upper=q_high, axis=1)
                    X_hold_v  = X_hold_v.clip(lower=q_low, upper=q_high, axis=1)

                    # Scale features
                    scaler.fit(X_train_v)
                    X_train_scaled = scaler.transform(X_train_v)
                    X_test_scaled = scaler.transform(X_test_v)
                    X_HoldOut_scaled = scaler.transform(X_hold_v)

                    colnames = current_cols

                    # --- Conditional logic: if mandatory_feature is provided ---
                    if mandatory_feature is not None:
                        if mandatory_feature not in colnames:
                            raise ValueError(
                                f"Mandatory feature '{mandatory_feature}' not found in X_full.columns"
                            )
                        m_idx = colnames.index(mandatory_feature)
                        rest_idx = [i for i in range(len(colnames)) if i != m_idx]
                        n_rest = max(0, num_features - 1)

                        # iterative conditional RFE
                        selected_rest = rest_idx.copy()
                        while len(selected_rest) > n_rest:
                            cols_order = [m_idx] + selected_rest
                            Xtr_sel = X_train_scaled[:, cols_order]
                            logistic_model.fit(Xtr_sel, y_train)
                            coefs = np.asarray(logistic_model.coef_).ravel()
                            drop_local = int(np.argmin(np.abs(coefs[1:])))
                            del selected_rest[drop_local]

                        cols_order = [m_idx] + selected_rest
                        selected_feature_names = [colnames[i] for i in cols_order]

                    else:
                        # --- Normal RFE if no mandatory feature ---
                        rfe.fit(X_train_scaled, y_train)
                        cols_order = np.where(rfe.support_)[0].tolist()
                        selected_feature_names = [colnames[i] for i in cols_order]

                    # select matrices
                    X_train_rfe   = X_train_scaled[:, cols_order]
                    X_test_rfe    = X_test_scaled[:,  cols_order]
                    X_HoldOut_rfe = X_HoldOut_scaled[:, cols_order]

                    # Fit logistic model
                    logistic_model.fit(X_train_rfe, y_train)
                    odds_ratios = np.exp(logistic_model.coef_).ravel()

                    hr_greater_than_1 = [f for f, orv in zip(selected_feature_names, odds_ratios) if orv > 1]
                    hr_less_than_1    = [f for f, orv in zip(selected_feature_names, odds_ratios) if orv <= 1]

                    hr_greater_than_1_str = ", ".join(hr_greater_than_1)
                    hr_less_than_1_str = ", ".join(hr_less_than_1)

                    # Validation metrics
                    y_score = logistic_model.decision_function(X_test_rfe)
                    val_auc = roc_auc_score(y_test, y_score)
                    y_pred_cls = logistic_model.predict(X_test_rfe)
                    val_acc = accuracy_score(y_test, y_pred_cls)

                    # Holdout metrics
                    y_score_ho = logistic_model.decision_function(X_HoldOut_rfe)
                    ho_auc = roc_auc_score(y_HoldOut, y_score_ho)
                    y_pred_ho = logistic_model.predict(X_HoldOut_rfe)
                    ho_acc = accuracy_score(y_HoldOut, y_pred_ho)

                    # Write results to CSV
                    writer.writerow([
                        num_features, val_auc, val_acc, ho_auc, ho_acc,
                        hr_greater_than_1_str, hr_less_than_1_str
                    ])


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


    print("Matrix shape:", matrix.shape)
    print("Matrix columns:", list(matrix.columns))
    print(matrix.head(2).to_string(index=False))

    for idx, row in matrix.iterrows():

        dataset = str(row["dataset"])
        featureset = str(row["featureset"])
        response_col = str(row["response"])

        data_path = Path(data_cfg["datasets"][dataset]["path"])
        df = load_table(data_path)

        feats = load_feature_list(Path(args.feature_dir), featureset)
        feats_present = [c for c in feats if c in df.columns]

        cols_to_balance = ast.literal_eval(row["cols_to_balance"])
        df["strat_col"] = df[cols_to_balance].astype(str).agg("_".join, axis=1)

        # --- Drop rows with NA in response or any selected feature columns ---
        response_and_features = list(set(feats_present) | {response_col})
        # print(response_and_features)
        # 'clean df by removing any rows with NA
        df_clean = df.dropna(subset=response_and_features)

        # finalize y
        y_series = df_clean[response_col].loc[df_clean.index].astype(int)
        # --------------------------------------------------------------------

        ## Split data into test and train based
        train_df, test_df, y_train, y_test = train_test_split(
            df_clean, y_series, test_size=0.25, stratify=df_clean["strat_col"], random_state=row["random_state"]
        )
        print(train_df["strat_col"].value_counts().head())
        print(test_df["strat_col"].value_counts().head())

        # Build X matrices
        X_train = train_df[feats_present].copy()
        X_test  = test_df[feats_present].copy()
        
       ## Remove correlated features
        ## Make sure to keep 'transf HPV16/18 copies per ml of plasma D1'
        no_corr = ['transf HPV16/18 copies per ml of plasma D1', 'HPV16/18 copies per ml of plasma D1']
        cols_for_corr = [c for c in X_train.columns if c not in no_corr]

        # X_train_drop_corr = tr.fit_transform(X_train[cols_for_corr])
        drop = ['LAP TGF-beta-1', 'Eosinophil Abs  D1', 'WBC D1', 'sCD27/IL8 D1', 
                'transf HPV16/18 copies per ml of plasma D1', 'Hemoglobin D1', 'Lymphocytes % D1', 'Neutrophil % D1']
        X_train_drop_corr = X_train.drop(columns=drop, errors='ignore')

        # add the protected column back in
        X_train_drop = pd.concat([X_train['transf HPV16/18 copies per ml of plasma D1'], X_train_drop_corr], axis=1)

        dropped_features = list(set(X_train.columns) - set(X_train_drop.columns))
        print("Dropped features:", dropped_features)

        corr = X_train.corr(method="spearman")

        # Print the top absolute correlations
        tri = corr.where(~np.tril(np.ones(corr.shape), k=0).astype(bool))
        pairs = (
            tri.stack()
            .abs()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"level_0":"feat1","level_1":"feat2",0:"abs_rho"})
        )
        print("top correlations:")
        print(pairs.head(20).to_string(index=False))


        X_test_drop = X_test[X_train_drop.columns]
        #cols_for_transform = [c for c in X_train_drop.columns if c not in {"Sex_num", "ICI_num",
        #                                                               "cancer_num", "transf HPV16/18 copies per ml of plasma D1",
        #                                                               "HPV16/18 copies per ml of plasma D1"}]
        cols_for_transform = ["sPDL1 D1", "TGFb1 D1", "sPD1 D1", "IL8 D1",
                              "Granzyme B D1", "sCD27 D1", "sCD40L D1",
                              "Ratio sCD27/sCD40L D1", "sCD73 D1", "sCTLA4 D1",
                              "GZMB/IL8 D1", "GZMB/TGFb1 D1", "GZMB/sCD73 D1",
                              "GZMB/sCD40L D1", "sCD27/TGFb1 D1",
                              "sCD27/sCD73 D1", "TGFb1/GZMB D1", "TGFb1/sCD27 D1"]

        ## Transformation??
        X_train_with_logs = log_transforms(X_train_drop, cols_for_transform)
        #print(X_train_with_logs.filter(like="_log").head())
        print(X_train_with_logs.columns.tolist())

        X_test_with_logs = log_transforms(X_test_drop, cols_for_transform)
        #print(X_test_with_logs.filter(like="_log").head())


        # y now is already numeric
        print(y_train.head())
        print(y_test.head())

        stamp = datetime.now().strftime("%Y-%m-%d_%H")
        out_dir = Path(args.out_root) / stamp  / f"RFE_LogReg_{dataset}__{featureset}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{row['folds']}fold_CV_RS{row['random_state']}.csv"


        ## RFE-based Logistic Regression
        folds = int(row["folds"])

 #       cross_validation_RFE(X_train_with_logs, y_train.values, X_test_with_logs, y_test.values, 
  #                  out_file, folds, 'transf HPV16/18 copies per ml of plasma D1')

        cross_validation_RFE(X_train_with_logs, y_train.values, X_test_with_logs, y_test.values, 
                    out_file, folds)

        save_dir = Path("../../../data-wrangle/TrainTestSets")     
        save_dir.mkdir(parents=True, exist_ok=True)

        X_train_with_logs.to_csv(save_dir / f"{featureset}_X_train_soluble_logs_LR.csv", index=False)
        pd.DataFrame(y_train).to_csv(save_dir / f"{featureset}_y_train_LR.csv", index=False)
        X_test_with_logs.to_csv(save_dir / f"{featureset}_X_test_soluble_logs_LR.csv", index=False)
        pd.DataFrame(y_test).to_csv(save_dir / f"{featureset}_y_test_LR.csv", index=False)



if __name__ == "__main__":
    main()
