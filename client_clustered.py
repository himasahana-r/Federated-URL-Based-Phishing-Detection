import flwr as fl
import pandas as pd
import numpy as np
from model import create_model
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import sys
import os

class FLClient(fl.client.NumPyClient):
    def __init__(self, X, y, client_id):
        self.client_id = client_id
        self.model = create_model(X.shape[1])
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2
        )
        self._masks = None  # will hold our random masks

    def get_parameters(self, config):
        # Round A: server asks for parameters → we return masked weights
        return self.model.get_weights()

    def fit(self, parameters, config):
        # 1. Receive & set the global model parameters
        self.model.set_weights(parameters)
        early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

        # 2. Train locally with class weights
        class_weights = {0: 2.7, 1: 1.0}
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=10,
            batch_size=32,
            verbose=0,
            class_weight=class_weights
        )

        # 3. Extract raw weights
        raw_weights = self.model.get_weights()

        # 4. Generate a reproducible random mask per tensor
        rng = np.random.seed(42 + self.client_id)
        masks = [np.random.normal(0, 0.005, w.shape) for w in raw_weights]
        self._masks = masks

        # 5. Apply the mask
        masked = [w + m for w, m in zip(raw_weights, masks)]

        # 6. Save cluster model (only one client per cluster writes to disk)
        if self.client_id == 3:
            self.model.save_weights("group_1_model.weights.h5")
        elif self.client_id == 1:
            self.model.save_weights("group_2_model.weights.h5")
        elif self.client_id == 2:
            self.model.save_weights("group_3_model.weights.h5")

        # 7. Return masked weights instead of raw
        return masked, len(self.X_train), {}

    def evaluate(self, parameters, config):
        # Not used by secure-agg during fit; but Flower still calls it if configured
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        return loss, len(self.X_val), {"accuracy": accuracy}

    def send_masks(self):
        """Round B RPC: server calls this to collect back our masks."""
        # Flwr expects signature (loss, num_examples, metrics)
        # We’ll send our masks in place of metrics
        return 0.0, len(self.X_train), self._masks

# Helper to load a client’s data from its cluster folder
def load_client_data(cluster_folder, client_id):
    df = pd.read_csv(os.path.join(cluster_folder, f"client_{client_id}.csv"))
    X = df.drop(columns=['label']).values
    y = df['label'].values
    return X, y

if __name__ == "__main__":
    cluster_folder = sys.argv[1]  # e.g. "cluster_1"
    client_id = int(sys.argv[2])  # e.g. 3
    X, y = load_client_data(cluster_folder, client_id)
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FLClient(X, y, client_id)
    )
