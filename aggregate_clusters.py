from model import create_model
import numpy as np
import pandas as pd

# 🔹 Step 1: Detect input_dim from sample CSV
sample_input = pd.read_csv("X_test.csv")  # or use any client CSV file
input_dim = sample_input.shape[1]         # Number of features

# 🔹 Step 2: Create model instances with correct input size
model_1 = create_model(input_dim=input_dim)
model_2 = create_model(input_dim=input_dim)
model_3 = create_model(input_dim=input_dim)

model_1.load_weights("group_1_model.weights.h5")
model_2.load_weights("group_2_model.weights.h5")
model_3.load_weights("group_3_model.weights.h5")

weights_1 = model_1.get_weights()
weights_2 = model_2.get_weights()
weights_3 = model_3.get_weights()

# Average weights
avg_weights = []
for w1, w2, w3 in zip(weights_1, weights_2, weights_3):
    avg = (w1 * 8 + w2 * 1 + w3 * 1) / 10
    avg_weights.append(avg)

# Save global model
final_model = create_model(input_dim=input_dim)
final_model.set_weights(avg_weights)
final_model.save_weights("global_model_clustered.weights.h5")

print("✅ Aggregated global model saved as global_model_clustered.weights.h5")
