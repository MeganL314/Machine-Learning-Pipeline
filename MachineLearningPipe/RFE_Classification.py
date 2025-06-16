
import pandas as pd; pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None); pd.set_option('display.width', 1000)
import numpy as np
from numpy import set_printoptions
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
pd.set_option('mode.chained_assignment', None)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from numpy import mean, std
### How to choose the number of features to select: 
### https://machinelearningmastery.com/rfe-feature-selection-in-python/
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import csv
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc, roc_auc_score






Data = pd.read_csv("/path/to/SerumAnalyteData.csv")

features_list = Data.columns[22:len(Data.columns)].tolist()


def create_XY_response(Main_df, feature_df):
    # Handling Response1
    Main_df['Response1_Num'] = Main_df['Response1'].map({'CB': 1, 'NCB': 0})
    response_features = feature_df.columns.union(['Response1'])
    Main_df_cleaned_response = Main_df.dropna(subset=response_features)
    
    # Extract Response1 target and feature matrices
    target_array = Main_df_cleaned_response['Response1_Num'].values
    features_array_response = Main_df_cleaned_response[feature_df.columns]
    
    # Handling Survival Data
    event_time = ['dead', 'followup']
    event_type = [('dead', '?'), ('followup', '<f8')]
    survival_features = feature_df.columns.union(event_time)
    Main_df_cleaned_survival = Main_df.dropna(subset=survival_features)
    
    # Extract survival target and feature matrices
    time_array = Main_df_cleaned_survival[event_time].to_numpy()
    time_array_new = np.array([(e1, e2) for e1, e2 in time_array], dtype=event_type)
    event_indicator = time_array_new['dead']
    time_to_event = time_array_new['followup']
    features_array_survival = Main_df_cleaned_survival[feature_df.columns]
    
    # Ensure consistency in features
    common_features = features_array_response.columns.intersection(features_array_survival.columns)
    features_array = Main_df_cleaned_response[common_features]
    
    print("Event Indicator:", event_indicator)
    print("Time to Event:", time_to_event)
    print("Target Array:", target_array)

    return event_indicator, time_to_event, features_array, target_array


event_indicator, time_to_event, X_ctDNA, y_ctDNA = create_XY_response(Data, 
                                                                      Data.loc[:, features_list])


X_train_clean, X_test, y_train_clean, y_test, event_train, event_test = train_test_split(
    X_ctDNA, y_ctDNA, event_indicator, test_size=0.27, random_state=8, stratify=y_ctDNA)



def Drop_Features(features_df, check_corr_df, subset_columns, outname):
    # Replace known non-numeric placeholders with NaN
    check_corr_df = check_corr_df.replace({'No Data': np.nan, '#VALUE!': np.nan, '': np.nan})

    # Convert all columns to numeric, coerce errors to NaN
    check_corr_df = check_corr_df.apply(pd.to_numeric, errors='coerce')

    # Check for missing values and print if there are any
    missing_counts = check_corr_df.isnull().sum()
    if missing_counts.any():
        print("Missing values detected:")
        print(missing_counts[missing_counts > 0])
    
    # Subset the DataFrame to only include the specified columns
    check_corr_df_subset = check_corr_df[subset_columns]

    # Compute the full Spearman correlation matrix
    spearman_corr = check_corr_df_subset.corr(method='spearman').abs()

    # Compute the upper triangle of the Spearman correlation matrix
    upper = spearman_corr.where(np.triu(np.ones(spearman_corr.shape), k=1).astype(bool))
    
    # Find features with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    
    # Identify and print correlated pairs
    correlated = [(row, column) for column in upper.columns for row in upper.index if upper.loc[row, column] > 0.9]
    if correlated:
        print("Correlated pairs:")
        for row, column in correlated:
            print(f"{column} / {row}")
    
    print("Columns to drop:", to_drop)
    print("Number of columns to drop:", len(to_drop))

    columns_left = [col for col in subset_columns if col not in to_drop]
    
    # Drop correlated features
    # output = features_df.copy()
    # output.drop(to_drop, axis=1, inplace=True)
    
    # Save the full correlation matrix to a CSV file
    spearman_corr.to_csv("Spearman_" + outname + '.csv')  
    
    return columns_left

def scaler(data):
    return StandardScaler().fit_transform(data)


smote = SMOTE(random_state=42)


X_train_ctDNA_res, y_train_ctDNA_res = smote.fit_resample(X_train_clean, y_train_clean)


def cross_validation_RFE(X_full, y_full, X_HoldOut, y_HoldOut, num_features_max, output_file, splits):
    # Initialize logistic regression model
    logistic_model = LogisticRegression(max_iter=1000)

    # Store results
    output_csv = output_file + "_CrossValidation.csv"

    # Open CSV file for writing results
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['RandomState', 'NumFeatures', 'Fold', 'Validation_Accuracy', 'Validation_AUC', 
                         'Holdout_Accuracy', 'Holdout_AUC', 'OR>1 Features', 'OR<1 Features'])

        # Loop through random states
        for random_state in range(1, 11):
            print(f'Running cross-validation with random state: {random_state}')

            # Loop through the number of features to select
            for num_features in range(1, num_features_max + 1):
                rfe = RFE(estimator=logistic_model, n_features_to_select=num_features)
                cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)

                fold_accuracies = []
                fold_auc_scores = []
                fold_accuracies_holdout = []
                fold_auc_scores_holdout = []

                scaler = StandardScaler()
                for fold, (train_index, test_index) in enumerate(cv.split(X_full, y_full)):
                    X_train, X_test = X_full.iloc[train_index], X_full.iloc[test_index]
                    y_train, y_test = y_full[train_index], y_full[test_index]

                    # Scale the features for train and test data
                    scaler.fit(X_train)
                    X_train_scaled = scaler.transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    X_HoldOut_scaled = scaler.transform(X_HoldOut)
                    
                    # Fit the RFE model
                    rfe.fit(X_train_scaled, y_train)
                    selected_features = rfe.support_
                    X_train_rfe = X_train_scaled[:, selected_features]
                    X_test_rfe = X_test_scaled[:, selected_features]

                    # Fit the logistic regression model
                    logistic_model.fit(X_train_rfe, y_train)

                    # Calculate validation fold accuracy and AUC
                    accuracy = logistic_model.score(X_test_rfe, y_test)
                    fold_accuracies.append(accuracy)

                    # Get predicted probabilities for AUC calculation
                    y_pred_prob = logistic_model.predict_proba(X_test_rfe)[:, 1]  # Get probabilities for the positive class
                    auc = roc_auc_score(y_test, y_pred_prob)  # Calculate AUC
                    fold_auc_scores.append(auc)

                    selected_feature_names = X_full.columns[selected_features].tolist()
                    # print(selected_feature_names)
                    
                    # Compute the odds ratios
                    odds_ratios = np.exp(logistic_model.coef_[0])
                    # print(odds_ratios)
                    
                    # Classify the features based on OR > 1 or OR < 1
                    or_greater_than_1 = []
                    or_less_than_1 = []

                    for feature, or_value in zip(selected_feature_names, odds_ratios):
                        if or_value > 1:
                            or_greater_than_1.append(feature)
                        else:
                            # print(or_value)
                            or_less_than_1.append(feature)
                            # print(feature)


                    
                    # Evaluate on holdout set (scale holdout set using the same scaler)
                    X_HoldOut_rfe = X_HoldOut_scaled[:, selected_features]

                    # Holdout set accuracy and AUC
                    holdout_accuracy = logistic_model.score(X_HoldOut_rfe, y_HoldOut)
                    holdout_auc = roc_auc_score(y_HoldOut, logistic_model.predict_proba(X_HoldOut_rfe)[:, 1])

                    fold_accuracies_holdout.append(holdout_accuracy)
                    fold_auc_scores_holdout.append(holdout_auc)

                    # Write results to CSV for this fold
                    writer.writerow([random_state, num_features, fold + 1, accuracy, auc,  
                                     holdout_accuracy, holdout_auc, ", ".join(or_greater_than_1), ", ".join(or_less_than_1)])













