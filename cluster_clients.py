import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Load all client files
client_ratios = []
client_ids = []

for i in range(1, 11):
    df = pd.read_csv(f"client_{i}.csv")
    total = len(df)
    phishing = df['label'].sum()
    ratio = phishing / total
    client_ratios.append([ratio])  # list of [ratio]
    client_ids.append(f"client_{i}")

# Convert to NumPy array for clustering
X = np.array(client_ratios)

# Perform hierarchical clustering
linked = linkage(X, 'ward')  # you can try 'single' or 'complete' as well

# Plot dendrogram to visualize grouping
plt.figure(figsize=(10, 5))
dendrogram(linked, labels=client_ids)
plt.title("Client Clustering Based on Phishing Ratio")
plt.xlabel("Clients")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("client_clustering_dendrogram.png")
plt.show()

# Form clusters (e.g., 3 groups)
group_labels = fcluster(linked, t=3, criterion='maxclust')

# Print groupings
for i, label in enumerate(group_labels):
    print(f"{client_ids[i]} --> Group {label}")
