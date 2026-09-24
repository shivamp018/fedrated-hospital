import flwr as fl
import numpy as np
import json
from typing import List, Tuple, Optional, Dict, Union
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, FitRes, Scalar

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None and server_round == 20:
            # Save aggregated_parameters at the final round
            np.savez("global_model_weights.npz", *[val for val in fl.common.parameters_to_ndarrays(aggregated_parameters)])
            print(f"Saved aggregated weights for final round {server_round}")
            
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        if loss_aggregated is not None:
            round_metrics = {"round": server_round, "loss": loss_aggregated}
            
            # Aggregate accuracy, auc, f1 from clients
            if results:
                total_examples = sum([res.num_examples for _, res in results])
                accuracies = [res.metrics["accuracy"] * res.num_examples for _, res in results]
                aucs = [res.metrics["auc"] * res.num_examples for _, res in results]
                
                round_metrics["accuracy"] = sum(accuracies) / total_examples
                round_metrics["auc"] = sum(aucs) / total_examples
                
            self.history.append(round_metrics)
            
            with open('metrics.json', 'w') as f:
                json.dump(self.history, f)
                
        return loss_aggregated, metrics_aggregated

strategy = SaveModelStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
)

if __name__ == "__main__":
    print("Starting Federated Learning Server (Orchestrator)...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )
