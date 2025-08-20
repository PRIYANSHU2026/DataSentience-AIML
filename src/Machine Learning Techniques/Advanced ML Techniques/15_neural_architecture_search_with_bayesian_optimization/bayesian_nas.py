"""
Neural Architecture Search with Bayesian Optimization

This module implements Bayesian Optimization for Neural Architecture Search,
using Gaussian Processes as surrogate models and various acquisition functions
to guide the search for optimal neural network architectures.

Key Components:
- BayesianNAS: Main search algorithm
- ArchitectureEncoder: Encodes architectures for BO
- ArchitectureEvaluator: Evaluates architecture performance
- BayesianOptimizer: Manages the BO loop
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import json
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ArchitectureConfig:
    """Configuration for architecture search space"""
    # Layer types
    layer_types: List[str] = field(default_factory=lambda: [
        'conv1x1', 'conv3x3', 'conv5x5', 'maxpool3x3', 'avgpool3x3',
        'sep_conv3x3', 'sep_conv5x5', 'dil_conv3x3', 'dil_conv5x5'
    ])
    
    # Architecture parameters
    max_layers: int = 10
    min_layers: int = 3
    max_channels: int = 512
    min_channels: int = 16
    max_kernel_size: int = 7
    min_kernel_size: int = 1
    
    # Search parameters
    max_iterations: int = 100
    n_initial_points: int = 10
    acquisition_function: str = 'ei'  # 'ei', 'ucb', 'pi', 'es'
    kernel_type: str = 'rbf'  # 'rbf', 'matern'
    
    # Training parameters
    epochs_per_evaluation: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.max_layers >= self.min_layers
        assert self.max_channels >= self.min_channels
        assert self.max_kernel_size >= self.min_kernel_size
        assert self.acquisition_function in ['ei', 'ucb', 'pi', 'es']
        assert self.kernel_type in ['rbf', 'matern']


class ArchitectureEncoder:
    """Encodes neural architectures into feature vectors for Bayesian Optimization"""
    
    def __init__(self, config: ArchitectureConfig):
        self.config = config
        self.feature_dim = self._calculate_feature_dim()
        
    def _calculate_feature_dim(self) -> int:
        """Calculate the dimensionality of the encoded architecture"""
        # Features per layer: layer_type, channels, kernel_size, activation
        features_per_layer = 4
        return self.config.max_layers * features_per_layer
    
    def encode_architecture(self, architecture: Dict) -> np.ndarray:
        """Encode architecture into a feature vector"""
        features = np.zeros(self.feature_dim)
        
        layers = architecture.get('layers', [])
        for i, layer in enumerate(layers[:self.config.max_layers]):
            base_idx = i * 4
            
            # Layer type (one-hot encoding)
            layer_type = layer.get('type', 'conv3x3')
            type_idx = self.config.layer_types.index(layer_type)
            features[base_idx] = type_idx / len(self.config.layer_types)
            
            # Number of channels (normalized)
            channels = layer.get('channels', 64)
            features[base_idx + 1] = (channels - self.config.min_channels) / \
                                   (self.config.max_channels - self.config.min_channels)
            
            # Kernel size (normalized)
            kernel_size = layer.get('kernel_size', 3)
            features[base_idx + 2] = (kernel_size - self.config.min_kernel_size) / \
                                   (self.config.max_kernel_size - self.config.min_kernel_size)
            
            # Activation function
            activation = layer.get('activation', 'relu')
            activation_map = {'relu': 0.0, 'tanh': 0.5, 'sigmoid': 1.0}
            features[base_idx + 3] = activation_map.get(activation, 0.0)
        
        return features
    
    def decode_architecture(self, features: np.ndarray) -> Dict:
        """Decode feature vector back to architecture"""
        architecture = {'layers': []}
        
        for i in range(self.config.max_layers):
            base_idx = i * 4
            
            # Layer type
            type_val = features[base_idx] * len(self.config.layer_types)
            type_idx = int(np.clip(type_val, 0, len(self.config.layer_types) - 1))
            layer_type = self.config.layer_types[type_idx]
            
            # Channels
            channels_val = features[base_idx + 1] * \
                          (self.config.max_channels - self.config.min_channels) + \
                          self.config.min_channels
            channels = int(np.clip(channels_val, self.config.min_channels, self.config.max_channels))
            
            # Kernel size
            kernel_val = features[base_idx + 2] * \
                        (self.config.max_kernel_size - self.config.min_kernel_size) + \
                        self.config.min_kernel_size
            kernel_size = int(np.clip(kernel_val, self.config.min_kernel_size, self.config.max_kernel_size))
            
            # Activation
            activation_val = features[base_idx + 3]
            if activation_val < 0.33:
                activation = 'relu'
            elif activation_val < 0.66:
                activation = 'tanh'
            else:
                activation = 'sigmoid'
            
            architecture['layers'].append({
                'type': layer_type,
                'channels': channels,
                'kernel_size': kernel_size,
                'activation': activation
            })
        
        return architecture


class ArchitectureEvaluator:
    """Evaluates neural architectures by training and testing them"""
    
    def __init__(self, config: ArchitectureConfig, data_loaders: Dict):
        self.config = config
        self.data_loaders = data_loaders
        self.encoder = ArchitectureEncoder(config)
        
    def evaluate_architecture(self, architecture: Dict) -> Dict:
        """Evaluate a single architecture"""
        try:
            # Create model from architecture
            model = self._create_model(architecture)
            
            # Train the model
            train_loss, train_acc = self._train_model(model)
            
            # Evaluate on validation set
            val_loss, val_acc = self._evaluate_model(model)
            
            # Calculate complexity metrics
            num_params = sum(p.numel() for p in model.parameters())
            model_size = num_params * 4 / (1024 * 1024)  # MB
            
            return {
                'architecture': architecture,
                'val_accuracy': val_acc,
                'val_loss': val_loss,
                'train_accuracy': train_acc,
                'train_loss': train_loss,
                'num_parameters': num_params,
                'model_size_mb': model_size,
                'success': True
            }
            
        except Exception as e:
            return {
                'architecture': architecture,
                'val_accuracy': 0.0,
                'val_loss': float('inf'),
                'train_accuracy': 0.0,
                'train_loss': float('inf'),
                'num_parameters': 0,
                'model_size_mb': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def _create_model(self, architecture: Dict) -> nn.Module:
        """Create a PyTorch model from architecture specification"""
        layers = []
        in_channels = 3  # RGB input
        
        for layer_spec in architecture['layers']:
            layer_type = layer_spec['type']
            channels = layer_spec['channels']
            kernel_size = layer_spec['kernel_size']
            activation = layer_spec['activation']
            
            # Create layer based on type
            if 'conv' in layer_type:
                if 'sep' in layer_type:
                    # Separable convolution
                    layers.extend([
                        nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2, groups=in_channels),
                        nn.Conv2d(in_channels, channels, 1)
                    ])
                elif 'dil' in layer_type:
                    # Dilated convolution
                    layers.append(nn.Conv2d(in_channels, channels, kernel_size, 
                                          padding=kernel_size//2, dilation=2))
                else:
                    # Regular convolution
                    layers.append(nn.Conv2d(in_channels, channels, kernel_size, 
                                          padding=kernel_size//2))
                
                in_channels = channels
                
            elif 'pool' in layer_type:
                if 'max' in layer_type:
                    layers.append(nn.MaxPool2d(kernel_size, padding=kernel_size//2))
                else:
                    layers.append(nn.AvgPool2d(kernel_size, padding=kernel_size//2))
            
            # Add activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            
            # Add batch normalization
            if 'conv' in layer_type:
                layers.append(nn.BatchNorm2d(channels))
        
        # Add final layers
        layers.extend([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, 10)  # 10 classes for CIFAR-10
        ])
        
        return nn.Sequential(*layers)
    
    def _train_model(self, model: nn.Module) -> Tuple[float, float]:
        """Train the model for a few epochs"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(self.config.epochs_per_evaluation):
            for batch_idx, (data, target) in enumerate(self.data_loaders['train']):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Early stopping if we have enough batches
                if batch_idx >= 10:  # Limit training for speed
                    break
        
        avg_loss = total_loss / (epoch + 1)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def _evaluate_model(self, model: nn.Module) -> Tuple[float, float]:
        """Evaluate the model on validation set"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.data_loaders['val']:
                data, target = data.to(device), target.to(device)
                output = model(data)
                total_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                # Limit evaluation for speed
                if total >= 1000:
                    break
        
        avg_loss = total_loss / (total // self.config.batch_size)
        accuracy = 100. * correct / total
        return avg_loss, accuracy


class BayesianOptimizer:
    """Bayesian Optimization for Neural Architecture Search"""
    
    def __init__(self, config: ArchitectureConfig, evaluator: ArchitectureEvaluator):
        self.config = config
        self.evaluator = evaluator
        self.encoder = evaluator.encoder
        
        # Initialize Gaussian Process
        if config.kernel_type == 'rbf':
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        else:
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=42
        )
        
        # Storage for observations
        self.X_observed = []
        self.y_observed = []
        self.architectures_observed = []
        self.evaluation_history = []
        
    def suggest_next_architecture(self) -> Dict:
        """Suggest the next architecture to evaluate using acquisition function"""
        if len(self.X_observed) < self.config.n_initial_points:
            # Random sampling for initial points
            return self._random_architecture()
        
        # Update GP with current observations
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        self.gp.fit(X, y)
        
        # Generate candidate architectures
        candidates = self._generate_candidates()
        candidate_features = np.array([self.encoder.encode_architecture(arch) for arch in candidates])
        
        # Calculate acquisition function values
        if self.config.acquisition_function == 'ei':
            acq_values = self._expected_improvement(candidate_features)
        elif self.config.acquisition_function == 'ucb':
            acq_values = self._upper_confidence_bound(candidate_features)
        elif self.config.acquisition_function == 'pi':
            acq_values = self._probability_improvement(candidate_features)
        else:  # 'es'
            acq_values = self._entropy_search(candidate_features)
        
        # Select best candidate
        best_idx = np.argmax(acq_values)
        return candidates[best_idx]
    
    def _random_architecture(self) -> Dict:
        """Generate a random architecture"""
        num_layers = np.random.randint(self.config.min_layers, self.config.max_layers + 1)
        architecture = {'layers': []}
        
        for _ in range(num_layers):
            layer = {
                'type': np.random.choice(self.config.layer_types),
                'channels': np.random.randint(self.config.min_channels, self.config.max_channels + 1),
                'kernel_size': np.random.randint(self.config.min_kernel_size, self.config.max_kernel_size + 1),
                'activation': np.random.choice(['relu', 'tanh', 'sigmoid'])
            }
            architecture['layers'].append(layer)
        
        return architecture
    
    def _generate_candidates(self, n_candidates: int = 100) -> List[Dict]:
        """Generate candidate architectures for acquisition function evaluation"""
        candidates = []
        
        # Random candidates
        for _ in range(n_candidates // 2):
            candidates.append(self._random_architecture())
        
        # Mutated candidates from best observed
        if self.y_observed:
            best_idx = np.argmax(self.y_observed)
            best_arch = self.architectures_observed[best_idx]
            
            for _ in range(n_candidates // 2):
                mutated = self._mutate_architecture(best_arch)
                candidates.append(mutated)
        
        return candidates
    
    def _mutate_architecture(self, architecture: Dict) -> Dict:
        """Mutate an architecture by randomly changing some parameters"""
        mutated = {'layers': []}
        
        for layer in architecture['layers']:
            if np.random.random() < 0.3:  # 30% mutation probability
                new_layer = layer.copy()
                
                # Randomly change one parameter
                param = np.random.choice(['type', 'channels', 'kernel_size', 'activation'])
                if param == 'type':
                    new_layer['type'] = np.random.choice(self.config.layer_types)
                elif param == 'channels':
                    new_layer['channels'] = np.random.randint(
                        self.config.min_channels, self.config.max_channels + 1)
                elif param == 'kernel_size':
                    new_layer['kernel_size'] = np.random.randint(
                        self.config.min_kernel_size, self.config.max_kernel_size + 1)
                else:
                    new_layer['activation'] = np.random.choice(['relu', 'tanh', 'sigmoid'])
                
                mutated['layers'].append(new_layer)
            else:
                mutated['layers'].append(layer.copy())
        
        return mutated
    
    def _expected_improvement(self, X: np.ndarray) -> np.ndarray:
        """Expected Improvement acquisition function"""
        y_best = np.max(self.y_observed)
        mean, std = self.gp.predict(X, return_std=True)
        
        # Avoid division by zero
        std = np.maximum(std, 1e-9)
        
        # Calculate EI
        z = (mean - y_best) / std
        ei = (mean - y_best) * norm.cdf(z) + std * norm.pdf(z)
        
        return ei
    
    def _upper_confidence_bound(self, X: np.ndarray, beta: float = 2.0) -> np.ndarray:
        """Upper Confidence Bound acquisition function"""
        mean, std = self.gp.predict(X, return_std=True)
        ucb = mean + beta * std
        return ucb
    
    def _probability_improvement(self, X: np.ndarray) -> np.ndarray:
        """Probability of Improvement acquisition function"""
        y_best = np.max(self.y_observed)
        mean, std = self.gp.predict(X, return_std=True)
        
        # Avoid division by zero
        std = np.maximum(std, 1e-9)
        
        z = (mean - y_best) / std
        pi = norm.cdf(z)
        return pi
    
    def _entropy_search(self, X: np.ndarray) -> np.ndarray:
        """Entropy Search acquisition function (simplified)"""
        mean, std = self.gp.predict(X, return_std=True)
        # Simplified entropy search: prefer points with high uncertainty
        return std
    
    def add_observation(self, architecture: Dict, performance: float):
        """Add a new observation to the optimization history"""
        features = self.encoder.encode_architecture(architecture)
        
        self.X_observed.append(features)
        self.y_observed.append(performance)
        self.architectures_observed.append(architecture)
    
    def get_best_architecture(self) -> Tuple[Dict, float]:
        """Get the best architecture found so far"""
        if not self.y_observed:
            return None, 0.0
        
        best_idx = np.argmax(self.y_observed)
        return self.architectures_observed[best_idx], self.y_observed[best_idx]


class BayesianNAS:
    """Main class for Neural Architecture Search with Bayesian Optimization"""
    
    def __init__(self, config: ArchitectureConfig, data_loaders: Dict):
        self.config = config
        self.data_loaders = data_loaders
        self.evaluator = ArchitectureEvaluator(config, data_loaders)
        self.optimizer = BayesianOptimizer(config, self.evaluator)
        
        # Results storage
        self.search_history = []
        self.best_architecture = None
        self.best_performance = 0.0
        
    def search(self) -> Dict:
        """Perform the Bayesian Optimization search"""
        print(f"Starting Bayesian NAS with {self.config.max_iterations} iterations...")
        print(f"Acquisition function: {self.config.acquisition_function}")
        print(f"Kernel: {self.config.kernel_type}")
        
        start_time = time.time()
        
        for iteration in range(self.config.max_iterations):
            print(f"\nIteration {iteration + 1}/{self.config.max_iterations}")
            
            # Suggest next architecture
            architecture = self.optimizer.suggest_next_architecture()
            print(f"Suggested architecture: {len(architecture['layers'])} layers")
            
            # Evaluate architecture
            print("Evaluating architecture...")
            result = self.evaluator.evaluate_architecture(architecture)
            
            # Add to optimization history
            performance = result['val_accuracy']
            self.optimizer.add_observation(architecture, performance)
            
            # Update best
            if performance > self.best_performance:
                self.best_performance = performance
                self.best_architecture = architecture
                print(f"New best! Accuracy: {performance:.2f}%")
            
            # Store result
            result['iteration'] = iteration
            result['best_so_far'] = self.best_performance
            self.search_history.append(result)
            
            print(f"Validation accuracy: {performance:.2f}%")
            print(f"Best so far: {self.best_performance:.2f}%")
            print(f"Model parameters: {result['num_parameters']:,}")
        
        search_time = time.time() - start_time
        
        return {
            'best_architecture': self.best_architecture,
            'best_performance': self.best_performance,
            'search_history': self.search_history,
            'search_time': search_time,
            'total_evaluations': len(self.search_history)
        }
    
    def plot_search_progress(self):
        """Plot the search progress"""
        if not self.search_history:
            print("No search history to plot")
            return
        
        iterations = [r['iteration'] for r in self.search_history]
        accuracies = [r['val_accuracy'] for r in self.search_history]
        best_so_far = [r['best_so_far'] for r in self.search_history]
        
        plt.figure(figsize=(12, 8))
        
        # Plot individual accuracies
        plt.subplot(2, 2, 1)
        plt.scatter(iterations, accuracies, alpha=0.6, label='Individual')
        plt.plot(iterations, best_so_far, 'r-', linewidth=2, label='Best so far')
        plt.xlabel('Iteration')
        plt.ylabel('Validation Accuracy (%)')
        plt.title('Search Progress')
        plt.legend()
        plt.grid(True)
        
        # Plot parameter count vs accuracy
        plt.subplot(2, 2, 2)
        param_counts = [r['num_parameters'] for r in self.search_history]
        plt.scatter(param_counts, accuracies, alpha=0.6)
        plt.xlabel('Number of Parameters')
        plt.ylabel('Validation Accuracy (%)')
        plt.title('Accuracy vs Model Size')
        plt.grid(True)
        
        # Plot model size distribution
        plt.subplot(2, 2, 3)
        model_sizes = [r['model_size_mb'] for r in self.search_history]
        plt.hist(model_sizes, bins=20, alpha=0.7)
        plt.xlabel('Model Size (MB)')
        plt.ylabel('Frequency')
        plt.title('Model Size Distribution')
        plt.grid(True)
        
        # Plot training vs validation accuracy
        plt.subplot(2, 2, 4)
        train_acc = [r['train_accuracy'] for r in self.search_history]
        plt.scatter(train_acc, accuracies, alpha=0.6)
        plt.plot([0, 100], [0, 100], 'r--', alpha=0.5)
        plt.xlabel('Training Accuracy (%)')
        plt.ylabel('Validation Accuracy (%)')
        plt.title('Training vs Validation Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def get_statistics(self) -> Dict:
        """Get statistics about the search"""
        if not self.search_history:
            return {}
        
        accuracies = [r['val_accuracy'] for r in self.search_history]
        param_counts = [r['num_parameters'] for r in self.search_history]
        model_sizes = [r['model_size_mb'] for r in self.search_history]
        
        return {
            'best_accuracy': max(accuracies),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'best_architecture': self.best_architecture,
            'total_architectures': len(self.search_history),
            'mean_parameters': np.mean(param_counts),
            'mean_model_size': np.mean(model_sizes),
            'successful_evaluations': sum(1 for r in self.search_history if r['success'])
        }


def create_dummy_data_loaders():
    """Create dummy data loaders for demonstration"""
    # Create dummy data
    dummy_data = torch.randn(100, 3, 32, 32)
    dummy_labels = torch.randint(0, 10, (100,))
    
    # Create dummy dataset
    class DummyDataset:
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]
    
    dataset = DummyDataset(dummy_data, dummy_labels)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False
    )
    
    return {
        'train': train_loader,
        'val': val_loader
    }


if __name__ == "__main__":
    # Example usage
    print("Neural Architecture Search with Bayesian Optimization")
    print("=" * 60)
    
    # Create configuration
    config = ArchitectureConfig(
        max_iterations=20,  # Reduced for demo
        n_initial_points=5,
        acquisition_function='ei',
        kernel_type='rbf',
        epochs_per_evaluation=5  # Reduced for demo
    )
    
    # Create data loaders
    data_loaders = create_dummy_data_loaders()
    
    # Create Bayesian NAS
    bayesian_nas = BayesianNAS(config, data_loaders)
    
    # Perform search
    results = bayesian_nas.search()
    
    # Print results
    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)
    print(f"Best accuracy: {results['best_performance']:.2f}%")
    print(f"Search time: {results['search_time']:.2f} seconds")
    print(f"Total evaluations: {results['total_evaluations']}")
    
    # Get statistics
    stats = bayesian_nas.get_statistics()
    print(f"\nMean accuracy: {stats['mean_accuracy']:.2f}% ± {stats['std_accuracy']:.2f}%")
    print(f"Mean parameters: {stats['mean_parameters']:,.0f}")
    print(f"Successful evaluations: {stats['successful_evaluations']}/{stats['total_architectures']}")
    
    # Plot results
    bayesian_nas.plot_search_progress()
    
    print("\nSearch completed!")
