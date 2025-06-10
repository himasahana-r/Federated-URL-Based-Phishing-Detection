# Federated URL-Based Phishing Detection

A privacy-preserving phishing URL detection system built using Federated Learning with character-level CNNs, FedAvg, FedProx, Clustered Aggregation, and Secure Aggregation.

## Key Highlights
- Detects phishing URLs without sharing raw data
- Simulates non-IID clients using a real-world URL dataset
- Implements FedAvg, FedProx, and hierarchical clustering
- Applies Secure Aggregation to protect client updates

## Tools & Tech
- Python, TensorFlow, Flower (FL Framework)
- Character-level CNNs
- Secure Aggregation Protocols

## Results
Achieved strong accuracy and privacy trade-offs across three models:
- Baseline: 85% accuracy
- Clustered: 80% accuracy
- Secure Aggregation: 76% accuracy
