import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef, roc_auc_score, f1_score
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from experiment.utils.process_one_hot import process_dataset, one_hot_encode
from experiment.utils.runner import run_processing


def process_file_with_subsampling(file, path, save_path, n_runs=100, frac=0.8):
    try:
        df = pd.read_csv(os.path.join(path, file))

        reports = {
            'mcc': [],
            'roc_auc': [],
            'f1': [],
        }

        # Prepare features and target
        for i in range(n_runs):
            sample_df = df.sample(frac=frac, replace=False, random_state=i)

            one_hot_df = process_dataset(sample_df, one_hot_encode, pad_value=0)
            X = one_hot_df.drop(['sequence', 'label'], axis=1).astype(float)
            y = one_hot_df['label']
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'SVM': SVC(probability=True, random_state=42),
                'KNN': KNeighborsClassifier(),
                'Random Forest': RandomForestClassifier(random_state=42),
                'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
            }
            scoring = {
                'mcc': make_scorer(matthews_corrcoef),
                'roc_auc': make_scorer(roc_auc_score),
                'f1': make_scorer(f1_score)
            }

            model_results = {
                'mcc': [],
                'roc_auc': [],
                'f1': [],
            }
            for model_name, model in models.items():
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_results = cross_validate(
                    model, X, y,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=False,
                    n_jobs=-1
                )

                model_results['mcc'].append(cv_results['test_mcc'].mean())
                model_results['roc_auc'].append(cv_results['test_roc_auc'].mean())
                model_results['f1'].append(cv_results['test_f1'].mean())

            reports['mcc'].append(float(np.mean(model_results['mcc'])))
            reports['roc_auc'].append(float(np.mean(model_results['roc_auc'])))
            reports['f1'].append(float(np.mean(model_results['f1'])))

        out_path = os.path.join(save_path, file.split('.')[0] + '_subsampled.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(reports, f, ensure_ascii=False, indent=4)

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


if __name__ == "__main__":
    run_processing(
        path="../../data",
        save_path="../results/raw_subsampling",
        process_func=process_file_with_subsampling,
        max_processes=10
    )
