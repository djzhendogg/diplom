import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

target_type = 'auc_roc_mean'


ridge = joblib.load(f'ridge/{target_type}/model.joblib')
X_train_sc = np.load(f'scaled_train/x/{target_type}.npy')
with open(f'scaled_train/columns/{target_type}.txt', 'r', encoding='utf-8') as f:
    feature_names = [line.strip() for line in f.readlines()]

explainer = shap.Explainer(ridge, X_train_sc)
shap_values = explainer(X_train_sc)

# Создаем и сохраняем summary plot
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values,
    X_train_sc,
    feature_names=feature_names,
    show=False,
    max_display=10
)
plt.tight_layout()
plt.savefig(f'images/shap/{target_type}_top10.png', dpi=300, bbox_inches='tight')
plt.close()