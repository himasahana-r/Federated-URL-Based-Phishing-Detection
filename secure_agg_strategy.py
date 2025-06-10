# secure_agg_strategy.py
import time
import flwr as fl
import numpy as np

class SecureAggregate(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._masked_aggregate = None  # Holds masked model weights

    def configure_fit(self, server_round, parameters, client_manager):
        """Decide whether to request training (Round A) or mask sending (Round B)."""
        while len(client_manager.all()) < self.min_available_clients:
            print(f"🌙 Waiting for {self.min_available_clients} clients, currently {len(client_manager.all())}")
            time.sleep(1)

        clients = list(client_manager.all().values())

        if self._masked_aggregate is None:
            # Round A: normal training
            fit_ins = fl.common.FitIns(parameters, {})  
            return [(c, fit_ins) for c in clients]
        else:
            # Round B: send masks (no training, reuse FitIns trick)
            fit_ins = fl.common.FitIns(parameters, {"collect_masks": True})
            return [(c, fit_ins) for c in clients]

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate masked model updates after training (Round A)."""
        masked_agg, _ = super().aggregate_fit(server_round, results, failures)
        self._masked_aggregate = masked_agg
        return None, {}

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate masks collected during Round B, and unmask the true weights."""
        mask_lists = [metrics for _, (_, _, metrics) in results]
        avg_masks = [
            sum(mask_tensor_list) / len(mask_lists)
            for mask_tensor_list in zip(*mask_lists)
        ]
        # Unmask the masked aggregate
        true_weights = [
            mw - am for mw, am in zip(self._masked_aggregate, avg_masks)
        ]

        # Create new Parameters object
        parameters = fl.common.ndarrays_to_parameters(true_weights)

        # Clear stored masked aggregate for next full training
        self._masked_aggregate = None

        return parameters, {}

    def evaluate(self, server_round, parameters):
        # Skip centralized evaluation
        return None, {}
