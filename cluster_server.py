# server_secure.py
import flwr as fl
from secure_agg_strategy import SecureAggregate

def main():
    strategy = SecureAggregate(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=1,
        min_available_clients=1,
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
