import flwr as fl
import pandas as pd
import numpy as np
from model import create_model
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys

# Load specific client data
def load_client_data(client_id):
    df = pd.read_csv(f"client_{client_id}.csv")
    X = df.drop(columns=['label']).values
    y = df['label'].values
    return X, y

# FedProx hyperparameter
mu = 0.01  # regularization strength (tune this!)

class FLClient(fl.client.NumPyClient):
    def __init__(self, X, y, client_id):
        self.model = create_model(X.shape[1])
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=0.2)
        self.client_id = client_id
        self.prev_weights = None  # to store global weights

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.prev_weights = parameters  # store the global weights before training

        # Custom training loop with FedProx-style regularization
        optimizer = tf.keras.optimizers.Adam()
        loss_fn = tf.keras.losses.BinaryCrossentropy()

        batch_size = 32
        epochs = 1
        dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train)).batch(batch_size)

        for epoch in range(epochs):
            for step, (x_batch, y_batch) in enumerate(dataset):
                with tf.GradientTape() as tape:
                    logits = self.model(x_batch, training=True)
                    loss = loss_fn(y_batch, logits)

                    # Add FedProx regularization term
                    prox_loss = 0.0
                    for w, w0 in zip(self.model.trainable_weights, self.prev_weights):
                        prox_loss += tf.reduce_sum(tf.square(w - tf.convert_to_tensor(w0)))
                    loss += (mu / 2) * prox_loss

                grads = tape.gradient(loss, self.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        # Save global model (from one client only)
        if self.client_id == 1:
            self.model.save_weights("global_model_fedprox.weights.h5")

        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        return loss, len(self.X_val), {"accuracy": accuracy}


if __name__ == "__main__":
    client_id = int(sys.argv[1])
    X, y = load_client_data(client_id)
    fl.client.start_numpy_client(server_address="localhost:8080", client=FLClient(X, y, client_id))
