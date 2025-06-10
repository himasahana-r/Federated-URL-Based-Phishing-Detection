from model import create_model
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd
import numpy as np

# Load test set
X_test = pd.read_csv("X_test.csv").values
y_test = pd.read_csv("y_test.csv").values.flatten()

# Load trained global model (optional: save/load weights)
model = create_model(X_test.shape[1])
model.load_weights("global_model_clustered.weights.h5")  # If you saved it during training

y_pred = model.predict(X_test).flatten()
y_pred_label = (y_pred > 0.5).astype(int)

print(classification_report(y_test, y_pred_label))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))
