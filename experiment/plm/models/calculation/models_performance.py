import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef, roc_auc_score, f1_score
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from experiment.utils.runner import run_processing


def process_file(file, path, save_path, verbose=False):
    try:
        df = pd.read_pickle(os.path.join(path, file))

        # Prepare features and target
        feature_columns = [col for col in df.columns if col not in ['sequence', 'label']]
        X = df[feature_columns].values
        y = df['label'].values
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
            pipeline = Pipeline([('scaler', StandardScaler()), (model_name, model)])
            cv_results = cross_validate(
                pipeline, X, y,
                cv=cv,
                scoring=scoring,
                return_train_score=False,
                n_jobs=-1
            )

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


if __name__ == "__main__":
    embedding_types = ['ankh_em', 'esm_c', 'protbert', 'prott5']
    for embedding_type in embedding_types:
        run_processing(
            path=f"../../plm_embeddings/results/{embedding_type}",
            save_path=f"../results/{embedding_type}",
            process_func=process_file,
            max_processes=20,
            file_extension=".pkl"
        )
