import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from model import create_model

# Load the full dataset (use original df before splitting to clients)
df = pd.read_csv("new_data_urls.csv")  # or use your cleaned version
df = df.drop_duplicates()
df['url'] = df['url'].str.lower()

# Feature extraction (reuse your lexical feature code)
from urllib.parse import urlparse
import re

def extract_features(url):
    parsed = urlparse(url)
    hostname = parsed.netloc
    path = parsed.path
    return pd.Series({
        "url_length": len(url),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": len(re.findall(r'[^a-zA-Z0-9]', url)),
        "has_https": int(url.startswith("https")),
        "num_dots": url.count("."),
        "has_ip": int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}", url))),
        "path_length": len(path),
        "hostname_length": len(hostname),
    })

features = df['url'].apply(extract_features)
features['label'] = df['status']

X = features.drop("label", axis=1).values
y = features["label"].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Build & train the model
model = create_model(X.shape[1])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate
y_pred = model.predict(X_test).flatten()
y_pred_label = (y_pred > 0.5).astype(int)

print(classification_report(y_test, y_pred_label))
print("ROC-AUC:", roc_auc_score(y_test, y_pred))
