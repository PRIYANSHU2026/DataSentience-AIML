"""
Surrogate Models for Bayesian Neural Architecture Search

This module implements various surrogate models for Bayesian Optimization,
including different Gaussian Process kernels and advanced surrogate models
that can better capture the complex relationships in neural architecture search.

Key Components:
- GaussianProcessSurrogate: Base GP surrogate with various kernels
- NeuralArchitectureKernel: Custom kernel for architecture similarity
- MultiTaskGaussianProcess: Multi-task GP for transfer learning
- DeepKernelGaussianProcess: GP with deep kernel learning
- EnsembleSurrogate: Ensemble of multiple surrogate models
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union, Any
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, ConstantKernel, WhiteKernel, 
    RationalQuadratic, ExpSineSquared, DotProduct
)
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class GaussianProcessSurrogate:
    """Base Gaussian Process surrogate model with various kernel options"""
    
    def __init__(self, kernel_type: str = 'rbf', noise_level: float = 1e-6, 
                 normalize_y: bool = True, random_state: int = 42):
        self.kernel_type = kernel_type
        self.noise_level = noise_level
        self.normalize_y = normalize_y
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Initialize kernel
        self.kernel = self._create_kernel()
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=noise_level,
            normalize_y=normalize_y,
            random_state=random_state
        )
    
    def _create_kernel(self):
        """Create kernel based on kernel_type"""
        if self.kernel_type == 'rbf':
            return ConstantKernel(1.0) * RBF(length_scale=1.0)
        elif self.kernel_type == 'matern':
            return ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
        elif self.kernel_type == 'rational_quadratic':
            return ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0)
        elif self.kernel_type == 'exp_sine_squared':
            return ConstantKernel(1.0) * ExpSineSquared(length_scale=1.0, periodicity=1.0)
        elif self.kernel_type == 'dot_product':
            return ConstantKernel(1.0) + DotProduct(sigma_0=1.0)
        elif self.kernel_type == 'white':
            return ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the surrogate model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit GP
        self.gp.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict using the surrogate model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        if return_std:
            return self.gp.predict(X_scaled, return_std=True)
        else:
            return self.gp.predict(X_scaled)
    
    def get_kernel_params(self) -> Dict:
        """Get kernel parameters"""
        if not self.is_fitted:
            return {}
        
        kernel_params = self.gp.kernel_.get_params()
        return kernel_params
    
    def log_marginal_likelihood(self) -> float:
        """Get log marginal likelihood"""
        if not self.is_fitted:
            return 0.0
        
        return self.gp.log_marginal_likelihood()


class NeuralArchitectureKernel:
    """Custom kernel for neural architecture similarity"""
    
    def __init__(self, layer_types: List[str], feature_weights: Optional[Dict] = None):
        self.layer_types = layer_types
        self.feature_weights = feature_weights or {
            'layer_type': 1.0,
            'channels': 0.5,
            'kernel_size': 0.3,
            'activation': 0.2
        }
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between two sets of architectures"""
        if X1.ndim == 1:
            X1 = X1.reshape(1, -1)
        if X2.ndim == 1:
            X2 = X2.reshape(1, -1)
        
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._compute_similarity(X1[i], X2[j])
        
        return K
    
    def _compute_similarity(self, arch1: np.ndarray, arch2: np.ndarray) -> float:
        """Compute similarity between two architectures"""
        max_layers = len(arch1) // 4  # 4 features per layer
        
        total_similarity = 0.0
        total_weight = 0.0
        
        for layer_idx in range(max_layers):
            base_idx = layer_idx * 4
            
            # Layer type similarity
            type1 = arch1[base_idx]
            type2 = arch2[base_idx]
            type_sim = 1.0 - abs(type1 - type2)
            total_similarity += self.feature_weights['layer_type'] * type_sim
            
            # Channels similarity (normalized)
            channels1 = arch1[base_idx + 1]
            channels2 = arch2[base_idx + 1]
            channels_sim = 1.0 - abs(channels1 - channels2)
            total_similarity += self.feature_weights['channels'] * channels_sim
            
            # Kernel size similarity
            kernel1 = arch1[base_idx + 2]
            kernel2 = arch2[base_idx + 2]
            kernel_sim = 1.0 - abs(kernel1 - kernel2)
            total_similarity += self.feature_weights['kernel_size'] * kernel_sim
            
            # Activation similarity
            act1 = arch1[base_idx + 3]
            act2 = arch2[base_idx + 3]
            act_sim = 1.0 - abs(act1 - act2)
            total_similarity += self.feature_weights['activation'] * act_sim
            
            total_weight += sum(self.feature_weights.values())
        
        return total_similarity / total_weight if total_weight > 0 else 0.0


class MultiTaskGaussianProcess:
    """Multi-task Gaussian Process for transfer learning in NAS"""
    
    def __init__(self, n_tasks: int, kernel_type: str = 'rbf', 
                 task_correlation: Optional[np.ndarray] = None):
        self.n_tasks = n_tasks
        self.kernel_type = kernel_type
        self.task_correlation = task_correlation or np.eye(n_tasks)
        self.gps = []
        self.scalers = []
        
        # Initialize GP for each task
        for _ in range(n_tasks):
            kernel = self._create_kernel()
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                random_state=42
            )
            self.gps.append(gp)
            self.scalers.append(StandardScaler())
    
    def _create_kernel(self):
        """Create kernel for individual tasks"""
        if self.kernel_type == 'rbf':
            return ConstantKernel(1.0) * RBF(length_scale=1.0)
        elif self.kernel_type == 'matern':
            return ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
        else:
            return ConstantKernel(1.0) * RBF(length_scale=1.0)
    
    def fit(self, X_list: List[np.ndarray], y_list: List[np.ndarray]):
        """Fit multi-task GP"""
        if len(X_list) != self.n_tasks or len(y_list) != self.n_tasks:
            raise ValueError("Number of tasks must match n_tasks")
        
        for i in range(self.n_tasks):
            X_scaled = self.scalers[i].fit_transform(X_list[i])
            self.gps[i].fit(X_scaled, y_list[i])
        
        return self
    
    def predict(self, X: np.ndarray, task_idx: int, return_std: bool = True):
        """Predict for a specific task"""
        if task_idx >= self.n_tasks:
            raise ValueError(f"Task index {task_idx} out of range")
        
        X_scaled = self.scalers[task_idx].transform(X)
        
        if return_std:
            return self.gps[task_idx].predict(X_scaled, return_std=True)
        else:
            return self.gps[task_idx].predict(X_scaled)
    
    def predict_all_tasks(self, X: np.ndarray, return_std: bool = True):
        """Predict for all tasks"""
        predictions = []
        stds = []
        
        for i in range(self.n_tasks):
            if return_std:
                pred, std = self.predict(X, i, return_std=True)
                predictions.append(pred)
                stds.append(std)
            else:
                pred = self.predict(X, i, return_std=False)
                predictions.append(pred)
        
        if return_std:
            return np.array(predictions), np.array(stds)
        else:
            return np.array(predictions)


class DeepKernelNetwork(nn.Module):
    """Deep neural network for learning kernel features"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], 
                 output_dim: int = 16):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.feature_net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to extract features"""
        return self.feature_net(x)


class DeepKernelGaussianProcess:
    """Gaussian Process with deep kernel learning"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], 
                 feature_dim: int = 16, kernel_type: str = 'rbf'):
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.kernel_type = kernel_type
        
        # Deep feature network
        self.feature_net = DeepKernelNetwork(input_dim, hidden_dims, feature_dim)
        
        # GP with RBF kernel on learned features
        self.kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract deep features from input"""
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            features = self.feature_net(X_tensor).numpy()
        return features
    
    def fit(self, X: np.ndarray, y: np.ndarray, n_epochs: int = 100, 
            lr: float = 0.001):
        """Fit the deep kernel GP"""
        # First, train the feature network
        self._train_feature_network(X, y, n_epochs, lr)
        
        # Extract features
        X_features = self._extract_features(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Fit GP on learned features
        self.gp.fit(X_scaled, y)
        self.is_fitted = True
        
        return self
    
    def _train_feature_network(self, X: np.ndarray, y: np.ndarray, 
                              n_epochs: int, lr: float):
        """Train the deep feature network"""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        optimizer = torch.optim.Adam(self.feature_net.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        self.feature_net.train()
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            features = self.feature_net(X_tensor)
            
            # Simple regression loss (can be improved)
            loss = criterion(features.mean(dim=1), y_tensor)
            
            loss.backward()
            optimizer.step()
    
    def predict(self, X: np.ndarray, return_std: bool = True):
        """Predict using deep kernel GP"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Extract features
        X_features = self._extract_features(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_features)
        
        if return_std:
            return self.gp.predict(X_scaled, return_std=True)
        else:
            return self.gp.predict(X_scaled)


class EnsembleSurrogate:
    """Ensemble of multiple surrogate models"""
    
    def __init__(self, surrogate_types: List[str] = None, 
                 weights: Optional[List[float]] = None):
        self.surrogate_types = surrogate_types or ['rbf', 'matern', 'rational_quadratic']
        self.weights = weights or [1.0] * len(self.surrogate_types)
        
        if len(self.weights) != len(self.surrogate_types):
            raise ValueError("Number of weights must match number of surrogate types")
        
        # Initialize surrogates
        self.surrogates = []
        for surrogate_type in self.surrogate_types:
            surrogate = GaussianProcessSurrogate(kernel_type=surrogate_type)
            self.surrogates.append(surrogate)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit all surrogate models"""
        for surrogate in self.surrogates:
            surrogate.fit(X, y)
        
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = True):
        """Ensemble prediction"""
        predictions = []
        stds = []
        
        for surrogate in self.surrogates:
            if return_std:
                pred, std = surrogate.predict(X, return_std=True)
                predictions.append(pred)
                stds.append(std)
            else:
                pred = surrogate.predict(X, return_std=False)
                predictions.append(pred)
        
        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(self.weights).reshape(-1, 1)
        
        weighted_pred = np.average(predictions, weights=weights, axis=0)
        
        if return_std:
            stds = np.array(stds)
            # Weighted average of uncertainties
            weighted_std = np.sqrt(np.average(stds**2, weights=weights, axis=0))
            return weighted_pred, weighted_std
        else:
            return weighted_pred
    
    def get_individual_predictions(self, X: np.ndarray, return_std: bool = True):
        """Get predictions from individual surrogates"""
        results = []
        
        for i, surrogate in enumerate(self.surrogates):
            if return_std:
                pred, std = surrogate.predict(X, return_std=True)
                results.append({
                    'type': self.surrogate_types[i],
                    'prediction': pred,
                    'std': std,
                    'weight': self.weights[i]
                })
            else:
                pred = surrogate.predict(X, return_std=False)
                results.append({
                    'type': self.surrogate_types[i],
                    'prediction': pred,
                    'weight': self.weights[i]
                })
        
        return results


class SurrogateAnalyzer:
    """Analyzer for surrogate model performance and behavior"""
    
    def __init__(self, surrogate):
        self.surrogate = surrogate
    
    def analyze_uncertainty(self, X: np.ndarray) -> Dict:
        """Analyze uncertainty patterns in the surrogate"""
        if hasattr(self.surrogate, 'predict'):
            mean, std = self.surrogate.predict(X, return_std=True)
            
            return {
                'mean_uncertainty': np.mean(std),
                'std_uncertainty': np.std(std),
                'min_uncertainty': np.min(std),
                'max_uncertainty': np.max(std),
                'uncertainty_distribution': std
            }
        else:
            return {}
    
    def plot_uncertainty_landscape(self, X: np.ndarray, y_true: Optional[np.ndarray] = None):
        """Plot uncertainty landscape"""
        if not hasattr(self.surrogate, 'predict'):
            print("Surrogate does not support uncertainty prediction")
            return
        
        mean, std = self.surrogate.predict(X, return_std=True)
        
        plt.figure(figsize=(15, 5))
        
        # Plot predictions vs uncertainty
        plt.subplot(1, 3, 1)
        plt.scatter(mean, std, alpha=0.6)
        plt.xlabel('Predicted Value')
        plt.ylabel('Uncertainty (Std)')
        plt.title('Prediction vs Uncertainty')
        plt.grid(True)
        
        # Plot uncertainty distribution
        plt.subplot(1, 3, 2)
        plt.hist(std, bins=20, alpha=0.7)
        plt.xlabel('Uncertainty (Std)')
        plt.ylabel('Frequency')
        plt.title('Uncertainty Distribution')
        plt.grid(True)
        
        # Plot predictions vs true values (if available)
        if y_true is not None:
            plt.subplot(1, 3, 3)
            plt.scatter(y_true, mean, alpha=0.6)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
            plt.xlabel('True Value')
            plt.ylabel('Predicted Value')
            plt.title('Prediction vs True Value')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def compare_surrogates(self, surrogates: List, X: np.ndarray, y_true: np.ndarray):
        """Compare multiple surrogate models"""
        results = []
        
        for surrogate in surrogates:
            if hasattr(surrogate, 'predict'):
                mean, std = surrogate.predict(X, return_std=True)
                
                # Calculate metrics
                mse = np.mean((mean - y_true) ** 2)
                mae = np.mean(np.abs(mean - y_true))
                mean_uncertainty = np.mean(std)
                
                results.append({
                    'surrogate': type(surrogate).__name__,
                    'mse': mse,
                    'mae': mae,
                    'mean_uncertainty': mean_uncertainty,
                    'predictions': mean,
                    'uncertainty': std
                })
        
        # Plot comparison
        plt.figure(figsize=(15, 5))
        
        # MSE comparison
        plt.subplot(1, 3, 1)
        names = [r['surrogate'] for r in results]
        mses = [r['mse'] for r in results]
        plt.bar(names, mses)
        plt.ylabel('MSE')
        plt.title('Mean Squared Error Comparison')
        plt.xticks(rotation=45)
        
        # MAE comparison
        plt.subplot(1, 3, 2)
        maes = [r['mae'] for r in results]
        plt.bar(names, maes)
        plt.ylabel('MAE')
        plt.title('Mean Absolute Error Comparison')
        plt.xticks(rotation=45)
        
        # Uncertainty comparison
        plt.subplot(1, 3, 3)
        uncertainties = [r['mean_uncertainty'] for r in results]
        plt.bar(names, uncertainties)
        plt.ylabel('Mean Uncertainty')
        plt.title('Uncertainty Comparison')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return results


if __name__ == "__main__":
    # Example usage of surrogate models
    print("Surrogate Models for Bayesian NAS")
    print("=" * 50)
    
    # Generate dummy data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.sin(X[:, 0]) + 0.1 * np.random.randn(100)
    
    # Test different surrogate models
    surrogates = [
        GaussianProcessSurrogate(kernel_type='rbf'),
        GaussianProcessSurrogate(kernel_type='matern'),
        GaussianProcessSurrogate(kernel_type='rational_quadratic'),
        DeepKernelGaussianProcess(input_dim=10, hidden_dims=[32, 16], feature_dim=8)
    ]
    
    # Fit and compare surrogates
    for surrogate in surrogates:
        print(f"\nTesting {type(surrogate).__name__}...")
        surrogate.fit(X, y)
        
        # Test prediction
        X_test = np.random.randn(20, 10)
        if hasattr(surrogate, 'predict'):
            mean, std = surrogate.predict(X_test, return_std=True)
            print(f"  Mean prediction: {np.mean(mean):.4f}")
            print(f"  Mean uncertainty: {np.mean(std):.4f}")
    
    # Test ensemble surrogate
    print("\nTesting Ensemble Surrogate...")
    ensemble = EnsembleSurrogate(['rbf', 'matern', 'rational_quadratic'])
    ensemble.fit(X, y)
    
    mean, std = ensemble.predict(X_test, return_std=True)
    print(f"  Ensemble mean prediction: {np.mean(mean):.4f}")
    print(f"  Ensemble mean uncertainty: {np.mean(std):.4f}")
    
    # Test multi-task GP
    print("\nTesting Multi-Task GP...")
    X_task1 = np.random.randn(50, 10)
    y_task1 = np.sin(X_task1[:, 0]) + 0.1 * np.random.randn(50)
    X_task2 = np.random.randn(50, 10)
    y_task2 = np.cos(X_task2[:, 0]) + 0.1 * np.random.randn(50)
    
    mtgp = MultiTaskGaussianProcess(n_tasks=2)
    mtgp.fit([X_task1, X_task2], [y_task1, y_task2])
    
    pred1, std1 = mtgp.predict(X_test, task_idx=0, return_std=True)
    pred2, std2 = mtgp.predict(X_test, task_idx=1, return_std=True)
    
    print(f"  Task 1 mean prediction: {np.mean(pred1):.4f}")
    print(f"  Task 2 mean prediction: {np.mean(pred2):.4f}")
    
    print("\nSurrogate models testing completed!")
