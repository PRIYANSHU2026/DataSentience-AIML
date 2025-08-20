"""
Federated Learning Implementation

This module implements federated learning with distributed training,
model aggregation, and privacy-preserving techniques.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import random
from typing import List, Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class FederatedConfig:
    """Configuration for federated learning"""
    num_clients: int = 10
    num_rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    client_fraction: float = 0.5  # Fraction of clients to select per round
    aggregation_method: str = 'fedavg'  # 'fedavg', 'fedprox', 'scaffold'
    secure_aggregation: bool = False
    differential_privacy: bool = False
    noise_scale: float = 1.0


class ModelAggregator:
    """Base class for model aggregation strategies"""
    
    def __init__(self, method: str = 'fedavg'):
        self.method = method
    
    def aggregate(self, global_model: nn.Module, client_models: List[nn.Module], 
                 client_weights: List[float] = None) -> nn.Module:
        """
        Aggregate client models into global model
        
        Args:
            global_model: Current global model
            client_models: List of client models
            client_weights: Weights for each client (optional)
        
        Returns:
            Updated global model
        """
        if self.method == 'fedavg':
            return self._federated_averaging(global_model, client_models, client_weights)
        elif self.method == 'fedprox':
            return self._federated_proximal(global_model, client_models, client_weights)
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")
    
    def _federated_averaging(self, global_model: nn.Module, client_models: List[nn.Module],
                           client_weights: List[float] = None) -> nn.Module:
        """Federated averaging aggregation"""
        # Use equal weights if not provided
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)
        
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # Create new global model
        new_global_model = copy.deepcopy(global_model)
        
        # Aggregate parameters
        with torch.no_grad():
            for param_name, param in new_global_model.named_parameters():
                aggregated_param = torch.zeros_like(param)
                
                for i, client_model in enumerate(client_models):
                    client_param = dict(client_model.named_parameters())[param_name]
                    aggregated_param += client_weights[i] * client_param.data
                
                param.data = aggregated_param
        
        return new_global_model
    
    def _federated_proximal(self, global_model: nn.Module, client_models: List[nn.Module],
                          client_weights: List[float] = None, mu: float = 0.01) -> nn.Module:
        """Federated proximal aggregation with regularization"""
        # Use equal weights if not provided
        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)
        
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # Create new global model
        new_global_model = copy.deepcopy(global_model)
        
        # Aggregate parameters with proximal term
        with torch.no_grad():
            for param_name, param in new_global_model.named_parameters():
                global_param = param.data
                aggregated_param = torch.zeros_like(param)
                
                for i, client_model in enumerate(client_models):
                    client_param = dict(client_model.named_parameters())[param_name]
                    # Add proximal regularization
                    proximal_term = mu * (client_param.data - global_param)
                    aggregated_param += client_weights[i] * (client_param.data - proximal_term)
                
                param.data = aggregated_param
        
        return new_global_model


class FederatedClient:
    """Federated learning client"""
    
    def __init__(self, client_id: int, local_data: torch.utils.data.DataLoader,
                 model: nn.Module, config: FederatedConfig):
        self.client_id = client_id
        self.local_data = local_data
        self.model = copy.deepcopy(model)
        self.config = config
        
        # Local optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate)
        
        # Training history
        self.training_history = []
    
    def train_local(self, global_model: nn.Module) -> Tuple[nn.Module, Dict]:
        """
        Train model on local data
        
        Args:
            global_model: Current global model
        
        Returns:
            Updated local model and training statistics
        """
        # Copy global model to local
        self.model.load_state_dict(global_model.state_dict())
        
        # Training statistics
        epoch_losses = []
        epoch_accuracies = []
        
        # Local training epochs
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            self.model.train()
            
            for batch_idx, (data, target) in enumerate(self.local_data):
                # Forward pass
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            # Calculate epoch statistics
            avg_loss = epoch_loss / len(self.local_data)
            accuracy = correct / total
            
            epoch_losses.append(avg_loss)
            epoch_accuracies.append(accuracy)
        
        # Training statistics
        stats = {
            'client_id': self.client_id,
            'epoch_losses': epoch_losses,
            'epoch_accuracies': epoch_accuracies,
            'final_loss': epoch_losses[-1],
            'final_accuracy': epoch_accuracies[-1],
            'data_size': len(self.local_data.dataset)
        }
        
        self.training_history.append(stats)
        
        return copy.deepcopy(self.model), stats
    
    def get_model_update(self, global_model: nn.Module) -> Dict[str, torch.Tensor]:
        """Get model update (difference from global model)"""
        self.model.load_state_dict(global_model.state_dict())
        
        update = {}
        for name, param in self.model.named_parameters():
            update[name] = param.data.clone()
        
        return update
    
    def apply_model_update(self, update: Dict[str, torch.Tensor]):
        """Apply model update to local model"""
        for name, param in self.model.named_parameters():
            if name in update:
                param.data = update[name].clone()


class FederatedServer:
    """Federated learning server"""
    
    def __init__(self, global_model: nn.Module, config: FederatedConfig):
        self.global_model = global_model
        self.config = config
        self.aggregator = ModelAggregator(config.aggregation_method)
        
        # Training history
        self.training_history = []
        self.global_metrics = []
        
        # Client selection history
        self.client_selection_history = []
    
    def select_clients(self, clients: List[FederatedClient]) -> List[FederatedClient]:
        """Select subset of clients for current round"""
        num_selected = max(1, int(self.config.client_fraction * len(clients)))
        selected_clients = random.sample(clients, num_selected)
        
        self.client_selection_history.append([c.client_id for c in selected_clients])
        return selected_clients
    
    def aggregate_models(self, client_models: List[nn.Module], 
                        client_weights: List[float] = None) -> nn.Module:
        """Aggregate client models"""
        return self.aggregator.aggregate(self.global_model, client_models, client_weights)
    
    def evaluate_global_model(self, test_data: torch.utils.data.DataLoader) -> Dict:
        """Evaluate global model on test data"""
        self.global_model.eval()
        
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_data:
                output = self.global_model(data)
                test_loss += F.cross_entropy(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        test_loss /= len(test_data)
        accuracy = correct / total
        
        return {
            'test_loss': test_loss,
            'test_accuracy': accuracy
        }
    
    def train(self, clients: List[FederatedClient], 
              test_data: Optional[torch.utils.data.DataLoader] = None) -> nn.Module:
        """
        Run federated learning training
        
        Args:
            clients: List of federated clients
            test_data: Test data for evaluation (optional)
        
        Returns:
            Trained global model
        """
        print(f"Starting federated learning with {len(clients)} clients")
        print(f"Configuration: {self.config}")
        
        for round_idx in range(self.config.num_rounds):
            print(f"\nRound {round_idx + 1}/{self.config.num_rounds}")
            
            # Select clients for this round
            selected_clients = self.select_clients(clients)
            print(f"Selected clients: {[c.client_id for c in selected_clients]}")
            
            # Train clients locally
            client_models = []
            client_weights = []
            round_stats = []
            
            for client in selected_clients:
                # Local training
                local_model, stats = client.train_local(self.global_model)
                client_models.append(local_model)
                
                # Weight based on data size
                weight = stats['data_size']
                client_weights.append(weight)
                round_stats.append(stats)
                
                print(f"Client {client.client_id}: Loss={stats['final_loss']:.4f}, "
                      f"Accuracy={stats['final_accuracy']:.4f}")
            
            # Aggregate models
            self.global_model = self.aggregate_models(client_models, client_weights)
            
            # Evaluate global model
            if test_data is not None:
                eval_stats = self.evaluate_global_model(test_data)
                self.global_metrics.append(eval_stats)
                
                print(f"Global model: Test Loss={eval_stats['test_loss']:.4f}, "
                      f"Test Accuracy={eval_stats['test_accuracy']:.4f}")
            
            # Record round statistics
            round_summary = {
                'round': round_idx + 1,
                'selected_clients': [c.client_id for c in selected_clients],
                'client_stats': round_stats,
                'global_metrics': eval_stats if test_data is not None else None
            }
            self.training_history.append(round_summary)
        
        print(f"\nFederated learning completed!")
        return self.global_model
    
    def get_training_summary(self) -> Dict:
        """Get training summary statistics"""
        if not self.training_history:
            return {}
        
        # Extract metrics
        rounds = [h['round'] for h in self.training_history]
        test_accuracies = [h['global_metrics']['test_accuracy'] 
                          for h in self.training_history if h['global_metrics']]
        test_losses = [h['global_metrics']['test_loss'] 
                      for h in self.training_history if h['global_metrics']]
        
        # Client participation
        all_participating_clients = set()
        for h in self.training_history:
            all_participating_clients.update(h['selected_clients'])
        
        return {
            'total_rounds': len(rounds),
            'final_test_accuracy': test_accuracies[-1] if test_accuracies else None,
            'final_test_loss': test_losses[-1] if test_losses else None,
            'unique_participating_clients': len(all_participating_clients),
            'average_clients_per_round': np.mean([len(h['selected_clients']) 
                                                for h in self.training_history])
        }
    
    def visualize_training_progress(self, save_path: str = 'federated_training_progress.png'):
        """Visualize federated learning training progress"""
        if not self.global_metrics:
            print("No global metrics available for visualization")
            return
        
        # Extract metrics
        rounds = range(1, len(self.global_metrics) + 1)
        test_accuracies = [m['test_accuracy'] for m in self.global_metrics]
        test_losses = [m['test_loss'] for m in self.global_metrics]
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Test accuracy over rounds
        axes[0, 0].plot(rounds, test_accuracies, 'b-', marker='o')
        axes[0, 0].set_title('Global Model Test Accuracy')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True)
        
        # Test loss over rounds
        axes[0, 1].plot(rounds, test_losses, 'r-', marker='o')
        axes[0, 1].set_title('Global Model Test Loss')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Client participation heatmap
        if self.client_selection_history:
            client_ids = sorted(set([c for round_clients in self.client_selection_history 
                                   for c in round_clients]))
            participation_matrix = np.zeros((len(client_ids), len(self.client_selection_history)))
            
            for round_idx, round_clients in enumerate(self.client_selection_history):
                for client_id in round_clients:
                    client_idx = client_ids.index(client_id)
                    participation_matrix[client_idx, round_idx] = 1
            
            axes[1, 0].imshow(participation_matrix, cmap='Blues', aspect='auto')
            axes[1, 0].set_title('Client Participation Heatmap')
            axes[1, 0].set_xlabel('Round')
            axes[1, 0].set_ylabel('Client ID')
            axes[1, 0].set_yticks(range(len(client_ids)))
            axes[1, 0].set_yticklabels(client_ids)
        
        # Average client accuracy per round
        avg_client_accuracies = []
        for round_stats in self.training_history:
            if round_stats['client_stats']:
                avg_acc = np.mean([s['final_accuracy'] for s in round_stats['client_stats']])
                avg_client_accuracies.append(avg_acc)
        
        if avg_client_accuracies:
            axes[1, 1].plot(range(1, len(avg_client_accuracies) + 1), avg_client_accuracies, 'g-', marker='s')
            axes[1, 1].set_title('Average Client Accuracy per Round')
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('Average Accuracy')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training progress visualization saved to {save_path}")


class HeterogeneousFederatedLearning:
    """Federated learning with heterogeneous data distributions"""
    
    def __init__(self, server: FederatedServer):
        self.server = server
        self.client_distributions = {}
    
    def create_heterogeneous_clients(self, base_data: torch.utils.data.Dataset,
                                   num_clients: int, distribution_type: str = 'iid') -> List[FederatedClient]:
        """Create clients with different data distributions"""
        clients = []
        
        if distribution_type == 'iid':
            # IID distribution (homogeneous)
            data_per_client = len(base_data) // num_clients
            for i in range(num_clients):
                start_idx = i * data_per_client
                end_idx = start_idx + data_per_client if i < num_clients - 1 else len(base_data)
                
                client_data = torch.utils.data.Subset(base_data, range(start_idx, end_idx))
                client_loader = torch.utils.data.DataLoader(client_data, batch_size=self.server.config.batch_size)
                
                client = FederatedClient(i, client_loader, copy.deepcopy(self.server.global_model), self.server.config)
                clients.append(client)
        
        elif distribution_type == 'non_iid':
            # Non-IID distribution (heterogeneous)
            # Sort data by labels
            sorted_indices = torch.argsort(torch.tensor([base_data[i][1] for i in range(len(base_data))]))
            
            # Distribute data non-uniformly
            for i in range(num_clients):
                # Each client gets data from specific label ranges
                start_label = (i * 10) // num_clients
                end_label = ((i + 1) * 10) // num_clients
                
                # Find indices for this label range
                client_indices = []
                for idx in sorted_indices:
                    label = base_data[idx][1]
                    if start_label <= label < end_label:
                        client_indices.append(idx.item())
                
                # Sample from available indices
                if len(client_indices) > 0:
                    num_samples = min(len(client_indices), len(base_data) // num_clients)
                    selected_indices = random.sample(client_indices, num_samples)
                    
                    client_data = torch.utils.data.Subset(base_data, selected_indices)
                    client_loader = torch.utils.data.DataLoader(client_data, batch_size=self.server.config.batch_size)
                    
                    client = FederatedClient(i, client_loader, copy.deepcopy(self.server.global_model), self.server.config)
                    clients.append(client)
                    
                    # Record distribution
                    labels = [base_data[idx][1] for idx in selected_indices]
                    self.client_distributions[i] = {
                        'labels': labels,
                        'label_counts': np.bincount(labels, minlength=10)
                    }
        
        return clients
    
    def analyze_heterogeneity(self) -> Dict:
        """Analyze data heterogeneity across clients"""
        if not self.client_distributions:
            return {}
        
        # Calculate distribution statistics
        all_labels = []
        for client_id, dist in self.client_distributions.items():
            all_labels.extend(dist['labels'])
        
        # Global distribution
        global_distribution = np.bincount(all_labels, minlength=10)
        
        # Client-specific distributions
        client_distributions = {}
        for client_id, dist in self.client_distributions.items():
            client_distributions[client_id] = dist['label_counts']
        
        # Calculate heterogeneity metrics
        heterogeneity_metrics = {
            'global_distribution': global_distribution,
            'client_distributions': client_distributions,
            'num_clients': len(self.client_distributions),
            'total_samples': len(all_labels)
        }
        
        return heterogeneity_metrics
    
    def visualize_heterogeneity(self, save_path: str = 'heterogeneity_analysis.png'):
        """Visualize data heterogeneity across clients"""
        metrics = self.analyze_heterogeneity()
        if not metrics:
            print("No heterogeneity data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Global distribution
        axes[0, 0].bar(range(10), metrics['global_distribution'])
        axes[0, 0].set_title('Global Data Distribution')
        axes[0, 0].set_xlabel('Class Label')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(True)
        
        # Client distributions heatmap
        client_ids = sorted(metrics['client_distributions'].keys())
        distribution_matrix = np.array([metrics['client_distributions'][cid] for cid in client_ids])
        
        im = axes[0, 1].imshow(distribution_matrix, cmap='viridis', aspect='auto')
        axes[0, 1].set_title('Client Data Distributions')
        axes[0, 1].set_xlabel('Class Label')
        axes[0, 1].set_ylabel('Client ID')
        axes[0, 1].set_yticks(range(len(client_ids)))
        axes[0, 1].set_yticklabels(client_ids)
        plt.colorbar(im, ax=axes[0, 1])
        
        # Distribution variance across clients
        distribution_variance = np.var(distribution_matrix, axis=0)
        axes[1, 0].bar(range(10), distribution_variance)
        axes[1, 0].set_title('Distribution Variance Across Clients')
        axes[1, 0].set_xlabel('Class Label')
        axes[1, 0].set_ylabel('Variance')
        axes[1, 0].grid(True)
        
        # Client data sizes
        client_sizes = [np.sum(metrics['client_distributions'][cid]) for cid in client_ids]
        axes[1, 1].bar(client_ids, client_sizes)
        axes[1, 1].set_title('Data Size per Client')
        axes[1, 1].set_xlabel('Client ID')
        axes[1, 1].set_ylabel('Number of Samples')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Heterogeneity analysis saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    print("Federated Learning Demo")
    print("=" * 50)
    
    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self, input_size=784, hidden_size=128, num_classes=10):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            x = x.view(-1, 784)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # Create dummy data
    num_clients = 5
    data_per_client = 100
    
    # Create federated configuration
    config = FederatedConfig(
        num_clients=num_clients,
        num_rounds=10,
        local_epochs=3,
        batch_size=16,
        learning_rate=0.01,
        client_fraction=0.6
    )
    
    # Create global model
    global_model = SimpleModel()
    
    # Create federated server
    server = FederatedServer(global_model, config)
    
    # Create dummy clients
    clients = []
    for i in range(num_clients):
        # Create dummy data for each client
        dummy_data = torch.randn(data_per_client, 784)
        dummy_labels = torch.randint(0, 10, (data_per_client,))
        dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size)
        
        client = FederatedClient(i, dataloader, copy.deepcopy(global_model), config)
        clients.append(client)
    
    # Create test data
    test_data = torch.randn(200, 784)
    test_labels = torch.randint(0, 10, (200,))
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size)
    
    # Run federated learning
    trained_model = server.train(clients, test_loader)
    
    # Get training summary
    summary = server.get_training_summary()
    print(f"\nTraining Summary: {summary}")
    
    # Visualize training progress
    server.visualize_training_progress()
    
    # Test heterogeneous federated learning
    print("\nTesting Heterogeneous Federated Learning...")
    heterogeneous_fl = HeterogeneousFederatedLearning(server)
    
    # Create heterogeneous clients
    base_dataset = torch.utils.data.TensorDataset(
        torch.randn(1000, 784),
        torch.randint(0, 10, (1000,))
    )
    
    heterogeneous_clients = heterogeneous_fl.create_heterogeneous_clients(
        base_dataset, num_clients=5, distribution_type='non_iid'
    )
    
    # Analyze heterogeneity
    heterogeneity_metrics = heterogeneous_fl.analyze_heterogeneity()
    print(f"Heterogeneity metrics: {heterogeneity_metrics}")
    
    # Visualize heterogeneity
    heterogeneous_fl.visualize_heterogeneity()
    
    print("Federated learning demo completed!")
