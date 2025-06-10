# save_initial_model.py
import flwr as fl
from model import create_model
import pickle

# Step 1: Create the model
input_dim = 18  # Adjust if your input shape is different
model = create_model(input_dim)

# Step 2: Get initial model weights
weights = fl.common.ndarrays_to_parameters(model.get_weights())

# Step 3: Save to file
with open("initial_parameters.pkl", "wb") as f:
    pickle.dump(weights, f)

print("✅ Initial parameters saved to initial_parameters.pkl")
