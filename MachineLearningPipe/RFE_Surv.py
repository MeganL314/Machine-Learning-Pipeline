import pandas as pd; pd.set_option('display.max_rows', None)
import re
import numpy as np
from lifelines import CoxPHFitter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
import csv
pd.set_option('display.max_columns', None); pd.set_option('display.width', 1000)
from sklearn.feature_selection import RFE
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

### Load data
PDS = pd.read_csv("/Path/To/Files/filenames.csv")
Quicc = pd.read_csv("/Path/To/Files/filenames.csv")

################## For Quicc, filter by Arm ########################
# Arm2 = Quicc[(Quicc['Arm'] != 'Arm 1')]
Quicc.columns = [ re.sub(r"\.", "-", col) for col in Quicc.columns]
# print(Arm2.columns.tolist())

### Specify lists of analytes
# Olink Baseline
Olink_Baseline = PDS.columns[24:115].tolist()
# Olink + CBCs Baseline
Olink_CBC_Baseline = PDS.columns[13:115].tolist()
# Olink 15 days
Olink_15days_V1 = [re.sub(r"_Baseline", "_Ch15d", text) for text in Olink_Baseline]
remove = ['IL12_Ch15d','IL15_Ch15d', 'PD-L1_Ch15d'] ### Make sure to remove  PD-L1, IL-15, and IL-12 Change
Olink_15days = [item for item in Olink_15days_V1 if item not in remove]
# Olink + CBCs 15 days
Olink_CBC_15days_V1 = [re.sub(r"_Baseline", "_Ch15d", text) for text in Olink_CBC_Baseline]
Olink_CBC_15days = [item for item in Olink_CBC_15days_V1 if item not in remove]
# Olink Two Times
Olink_TwoTimes = Olink_15days + Olink_Baseline
# Olink + CBCs Two Times
Olink_CBC_TwoTimes = Olink_CBC_15days + Olink_CBC_Baseline


def create_XY_response(Main_df, feature_list, response='no', survival='yes', 
                       event_col='dead', time_col='survival'):
    target_array = event_indicator = time_to_event = features_array = None

    if response == 'yes':
        Main_df['Response1_Num'] = Main_df['Response1'].map({'CB': 1, 'NCB': 0})
        response_features = set(feature_list).union(['Response1'])
        Main_df_cleaned_response = Main_df.dropna(subset=response_features)
        target_array = Main_df_cleaned_response['Response1_Num'].values
        features_array_response = Main_df_cleaned_response[feature_list]

    if survival == 'yes':
        event_time = [event_col, time_col]
        event_type = [(event_col, '?'), (time_col, '<f8')]
        survival_features = list(set(feature_list).union(event_time))
        Main_df_cleaned_survival = Main_df.dropna(subset=survival_features)
        time_array = Main_df_cleaned_survival[event_time].to_numpy()
        time_array_new = np.array([tuple(row) for row in time_array], dtype=event_type)
        event_indicator = time_array_new[event_col]
        time_to_event = time_array_new[time_col]
        features_array_survival = Main_df_cleaned_survival[feature_list]

    if response == 'yes' and survival == 'yes':
        common_features = features_array_response.columns.intersection(features_array_survival.columns)
        features_array = Main_df_cleaned_response[common_features]
    elif response == 'yes':
        features_array = features_array_response
    elif survival == 'yes':
        features_array = features_array_survival

    return event_indicator, time_to_event, features_array, target_array


## SPLIT DATA FIRST
train_df, test_df = train_test_split(Quicc, test_size=0.25, random_state=8)


event_train_D1, time_train_D1, X_train_D1, y_train_D1 = create_XY_response(train_df, Olink_Baseline)
event_test_D1, time_test_D1, X_test_D1, y_test_D1 = create_XY_response(test_df, Olink_Baseline)

event_train_Ch15, time_train_Ch15, X_train_Ch15, y_train_Ch15 = create_XY_response(train_df, Olink_15days)
event_test_Ch15, time_test_Ch15, X_test_Ch15, y_test_Ch15 = create_XY_response(test_df, Olink_15days)

event_train_TT, time_train_TT, X_train_TT, y_train_Ch15 = create_XY_response(train_df, Olink_TwoTimes)
event_test_TT, time_test_TT, X_test_TT, y_test_Ch15 = create_XY_response(test_df, Olink_TwoTimes)

event_type = [('dead', '?'), ('survival', '<f8')]
## Baseline
y_train_D1 = np.array(list(zip(event_train_D1, time_train_D1)), dtype=event_type)
y_test_D1 = np.array(list(zip(event_test_D1, time_test_D1)), dtype=event_type)

## Change at 15 days
y_train_Ch15 = np.array(list(zip(event_train_Ch15, time_train_Ch15)), dtype=event_type)
y_test_Ch15 = np.array(list(zip(event_test_Ch15, time_test_Ch15)), dtype=event_type)

## Two time points
y_train_TT = np.array(list(zip(event_train_TT, time_train_TT)), dtype=event_type)
y_test_TT = np.array(list(zip(event_test_TT, time_test_TT)), dtype=event_type)



def drop_features( check_corr_df, subset_columns, outname=None):
    # Replace common non-numeric placeholders with NaN and convert to numeric
    cleaned_df = check_corr_df.replace({'No Data': np.nan, '#VALUE!': np.nan, '': np.nan})
    cleaned_df = cleaned_df.apply(pd.to_numeric, errors='coerce')

    # Report missing values
    missing = cleaned_df.isnull().sum()
    if missing.any():
        print("Missing values detected:")
        print(missing[missing > 0])

    # Subset columns and compute Spearman correlation
    subset_df = cleaned_df[subset_columns]
    corr_matrix = subset_df.corr(method='spearman').abs()

    # Save correlation matrix to CSV if outname is provided
    if outname:
        corr_matrix.to_csv(outname)
        print(f"Correlation matrix saved to: {outname}")

    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Identify highly correlated columns
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.9)]

    # Print correlated pairs
    correlated_pairs = [(row, col) for col in upper_tri.columns for row in upper_tri.index if upper_tri.loc[row, col] > 0.9]
    if correlated_pairs:
        print("Correlated pairs:")
        for row, col in correlated_pairs:
            print(f"{col} / {row}")

    print(f"Columns to drop: {to_drop}")
    print(f"Number of columns to drop: {len(to_drop)}")

    # Return remaining features
    return [col for col in subset_columns if col not in to_drop]

Baseline_drop = drop_features(X_train_D1, Olink_Baseline, outname="Baseline_Olink_BothArms_Spearman.csv")
Change15_drop = drop_features(X_train_Ch15, Olink_15days, outname="Change15_Olink_BothArms_Spearman.csv")
TwoTimes_drop = drop_features(X_train_TT, Olink_TwoTimes, outname="BothTimePoints_Olink_BothArms_Spearman.csv")




def cross_validation_RFE(X_full, y_full, X_HoldOut, y_HoldOut, output_file, splits, num_features_max=3):
    # Initialize Cox Proportional Hazards model
    cph_model = CoxPHSurvivalAnalysis(alpha=0.001)

    # Store results
    output_csv = output_file + "_CrossValidation_CPH.csv"

    # Open CSV file for writing results
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['NumFeatures', 'Validation_Concordance', 
                         'Holdout_Concordance', 'HR > 1 Features', 'HR < 1 Features'])

        for random_state in range(1, 11):
            print(f'Running cross-validation with random state: {random_state}')

            # Loop through the number of features to select
            for num_features in range(1, num_features_max + 1):
                rfe = RFE(cph_model, n_features_to_select=num_features)
                cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)

                fold_concordances = []
                fold_concordances_holdout = []

                scaler = StandardScaler()
                for fold, (train_index, test_index) in enumerate(cv.split(X_full, y_full['dead'])):
                    X_train, X_test = X_full.iloc[train_index], X_full.iloc[test_index]
                    y_train, y_test = y_full[train_index], y_full[test_index]

                    # Scale the features for train and test data
                    scaler.fit(X_train)
                    X_train_scaled = scaler.transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    X_HoldOut_scaled = scaler.transform(X_HoldOut)

                    X_train_scaled_df = pd.DataFrame(X_train_scaled)
                    ### Check why there is an error:
                    nan_columns = X_train_scaled_df.columns[X_train_scaled_df.isna().any()].tolist()
                   # print("Columns with NaN values:", nan_columns)

                    inf_columns = X_train_scaled_df.columns[np.isinf(X_train_scaled_df).any(axis=0)].tolist()

                    # print("Columns with infinite values:", inf_columns)
                    
                    # Fit the RFE model
                    rfe.fit(X_train_scaled, y_train)
                    selected_features = rfe.support_

                    # Select the RFE features (by index, not column names)
                    X_train_rfe = X_train_scaled[:, selected_features]
                    X_test_rfe = X_test_scaled[:, selected_features]

                    # Fit the model on selected features to calculate hazard ratios
                    cph_model.fit(X_train_rfe, y_train)
                    hazard_ratios = np.exp(cph_model.coef_)  # Exponentiate the coefficients to get hazard ratios
                    
                    # Separate features based on HR > 1 or HR < 1
                    hr_greater_than_1 = []
                    hr_less_than_1 = []

                    for feature, hr in zip(X_full.columns[selected_features], hazard_ratios):
                        if hr > 1:
                            hr_greater_than_1.append(feature)
                        else:
                            hr_less_than_1.append(feature)

                    # Join feature names for HR > 1 and HR < 1
                    hr_greater_than_1_str = ", ".join(hr_greater_than_1)
                    hr_less_than_1_str = ", ".join(hr_less_than_1)

                    # Make predictions using the RFE model
                    y_pred = rfe.estimator_.predict(X_test_rfe)
                    
                    # Calculate C-index
                    status_values = y_test['dead']  # Ensure it's a Series/Array for status
                    time_values = y_test['survival'] # Ensure it's a Series/Array for time
                    cindex = concordance_index_censored(status_values, time_values, y_pred)
                    fold_concordances.append(cindex[0])

                    # Evaluate on holdout set (scale holdout set using the same scaler)
                    X_HoldOut_rfe = X_HoldOut_scaled[:, selected_features]
                    y_HoldOut_pred = rfe.estimator_.predict(X_HoldOut_rfe)

                    # Holdout set concordance index
                    holdout_concordance = concordance_index_censored(y_HoldOut['dead'], y_HoldOut['survival'], y_HoldOut_pred)
                    fold_concordances_holdout.append(holdout_concordance[0])

                    selected_feature_names = ", ".join(X_full.columns[selected_features])

                    # Write results to CSV for this fold
                    writer.writerow([num_features, cindex[0], holdout_concordance[0], hr_greater_than_1_str, hr_less_than_1_str])

#cross_validation_RFE(X_train_D1[Baseline_drop], y_train_D1, X_test_D1[Baseline_drop], y_test_D1, 
#                     "./RFE_Surv/Baselinefilename", 3)

