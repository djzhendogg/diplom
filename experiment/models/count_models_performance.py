import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef, roc_auc_score, f1_score
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from process_one_hot import process_dataset, one_hot_encode


def process_file(file, path, save_path, verbose=False):
    try:
        df = pd.read_csv(os.path.join(path, file))

        one_hot_df = process_dataset(df, one_hot_encode, pad_value=0)

        # Prepare features and target
        feature_columns = [col for col in one_hot_df.columns if col not in ['sequence', 'label']]
        X = one_hot_df[feature_columns].values
        y = one_hot_df['label'].values
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': SVC(probability=True, random_state=42),  # probability=True for ROC-AUC
            'KNN': KNeighborsClassifier(),
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
        }
        scoring = {
            'mcc': make_scorer(matthews_corrcoef),
            'roc_auc': make_scorer(roc_auc_score),
            'f1': make_scorer(f1_score)
        }

        results = []
        print(f"Файл {file} содержит {X.shape[0]} строк, {len(feature_columns)} колонок")
        for model_name, model in models.items():
            if verbose:
                print(f"Training {model_name}...")

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_results = cross_validate(
                model, X, y,
                cv=cv,
                scoring=scoring,
                return_train_score=False,
                n_jobs=-1
            )

            # Store results
            results.append({
                'Model': model_name,
                'MCC_mean': cv_results['test_mcc'].mean(),
                'MCC_std': cv_results['test_mcc'].std(),
                'AUC-ROC_mean': cv_results['test_roc_auc'].mean(),
                'AUC-ROC_std': cv_results['test_roc_auc'].std(),
                'F1_mean': cv_results['test_f1'].mean(),
                'F1_std': cv_results['test_f1'].std()
            })
            if verbose:
                print(f"  MCC: {cv_results['test_mcc'].mean():.3f} (+/- {cv_results['test_mcc'].std() * 2:.3f})")
                print(
                    f"  AUC-ROC: {cv_results['test_roc_auc'].mean():.3f} (+/- {cv_results['test_roc_auc'].std() * 2:.3f})")
                print(f"  F1: {cv_results['test_f1'].mean():.3f} (+/- {cv_results['test_f1'].std() * 2:.3f})")
                print()

        summary_df = pd.DataFrame(results)
        if verbose:
            print("\n=== Summary of Results ===")
            print(summary_df.to_string(index=False, float_format="%.3f"))

        # Optionally, save results to CSV
        out_path = os.path.join(save_path, file.split('.')[0] + '.csv')
        summary_df.to_csv(out_path, index=False)
        return {
            'file': file,
            'status': 'success'
        }

    except Exception as e:
        return {
            'file': file,
            'status': 'error',
            'message': str(e),
        }
