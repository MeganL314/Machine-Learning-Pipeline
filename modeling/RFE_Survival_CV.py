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





def create_XY_response(Main_df, feature_index, response='no', survival='yes', 
                       event_col='dead', time_col='survival'):
    
    if response == 'yes':
        # Handling Response1
        Main_df['Response1_Num'] = Main_df['Response1'].map({'CB': 1, 'NCB': 0})
        response_features = feature_index.union(['Response1'])
        Main_df_cleaned_response = Main_df.dropna(subset=response_features)
        target_array = Main_df_cleaned_response['Response1_Num'].values
        features_array_response = Main_df_cleaned_response[feature_index]
    
    if survival == 'yes':
        event_time = [event_col, time_col]
        event_type = [(event_col, '?'), (time_col, '<f8')]
        survival_features = feature_index.union(event_time)
        Main_df_cleaned_survival = Main_df.dropna(subset=survival_features)
        time_array = Main_df_cleaned_survival[event_time].to_numpy()
        time_array_new = np.array([tuple(row) for row in time_array], dtype=event_type)
        event_indicator = time_array_new[event_col]
        time_to_event = time_array_new[time_col]
        features_array_survival = Main_df_cleaned_survival[feature_index]
        
    # Combine features only if both response and survival are requested
    if response == 'yes' and survival == 'yes':
        common_features = features_array_response.columns.intersection(features_array_survival.columns)
        features_array = Main_df_cleaned_response[common_features]
    elif response == 'yes':
        features_array = features_array_response
    elif survival == 'yes':
        features_array = features_array_survival

    return event_indicator, time_to_event, features_array, target_array


# def remove_correlated():




# def cross_validation_RFE():



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


    for row in matrix.iterrows():
        dataset = str(row["dataset"])
        featureset = str(row["featureset"])

        data_path = Path(data_cfg["datasets"][dataset]["path"])
        df = load_table(data_path)

        feats = load_feature_list(Path(args.feature_dir), featureset)
        feats_present = [c for c in feats if c in df.columns]

        # make numeric & drop rows with any NA in these columns
        X = df[feats_present].replace({"No Data": np.nan, "#VALUE!": np.nan})
        X = X.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

        event_PDS, time_PDS, X_PDS, y_ = create_XY_response(X, featureset, response='no', survival='yes', 
                       event_col=row["event"], time_col=row["time"])



