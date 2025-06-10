import pandas as pd
import numpy as np
from urllib.parse import urlparse
import re
from sklearn.model_selection import train_test_split
import os

# 1. Load and clean the dataset
df = pd.read_csv("new_data_urls.csv")  # Update path if needed
df = df.drop_duplicates()
df['url'] = df['url'].str.lower()

# 2. Define feature extraction function
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

# 3. Extract features
features = df['url'].apply(extract_features)
features['label'] = df['status']

X = features.drop("label", axis=1)
y = features["label"]

# 4. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Save test set for later evaluation
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# 5. Non-IID Partitioning (10 clients)
def non_iid_partition(X, y, num_clients=10):
    clients = {}
    phishing_idx = y[y == 1].index.to_numpy()
    legit_idx = y[y == 0].index.to_numpy()
    
    np.random.shuffle(phishing_idx)
    np.random.shuffle(legit_idx)
    
    phishing_splits = np.array_split(phishing_idx, num_clients)
    legit_splits = np.array_split(legit_idx, num_clients)
    
    for i in range(num_clients):
        indices = np.concatenate((phishing_splits[i], legit_splits[i]))
        np.random.shuffle(indices)
        client_X = X.loc[indices]
        client_y = y.loc[indices]
        client_data = client_X.copy()
        client_data['label'] = client_y
        clients[f"client_{i+1}"] = client_data
    
    return clients

# Generate and save client datasets
clients_data = non_iid_partition(X_train, y_train, num_clients=10)
for i, (client_name, client_df) in enumerate(clients_data.items(), start=1):
    client_df.to_csv(f"client_{i}.csv", index=False)
