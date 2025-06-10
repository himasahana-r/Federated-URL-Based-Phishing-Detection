import flwr as fl

if __name__ == "__main__":
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=fl.server.strategy.FedAvg(min_available_clients=10)
    )
