"""
Acquisition Functions for Bayesian Neural Architecture Search

This module implements various acquisition functions for Bayesian Optimization,
including standard functions and advanced variants specifically designed for
neural architecture search scenarios.

Key Components:
- ExpectedImprovement: Standard Expected Improvement
- UpperConfidenceBound: UCB with adaptive exploration
- ProbabilityImprovement: Probability of Improvement
- EntropySearch: Information-theoretic acquisition
- MultiObjectiveAcquisition: Multi-objective optimization
- KnowledgeGradient: Knowledge gradient acquisition
- ThompsonSampling: Thompson sampling for BO
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import warnings
warnings.filterwarnings('ignore')


class BaseAcquisitionFunction:
    """Base class for acquisition functions"""
    
    def __init__(self, surrogate_model, **kwargs):
        self.surrogate_model = surrogate_model
        self.kwargs = kwargs
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Evaluate acquisition function at points X"""
        raise NotImplementedError
    
    def optimize(self, bounds: List[Tuple[float, float]], n_candidates: int = 1000) -> Tuple[np.ndarray, float]:
        """Optimize acquisition function to find next point"""
        # Generate random candidates
        candidates = self._generate_candidates(bounds, n_candidates)
        
        # Evaluate acquisition function
        acq_values = self(candidates)
        
        # Find best candidate
        best_idx = np.argmax(acq_values)
        best_point = candidates[best_idx]
        best_value = acq_values[best_idx]
        
        return best_point, best_value
    
    def _generate_candidates(self, bounds: List[Tuple[float, float]], n_candidates: int) -> np.ndarray:
        """Generate random candidates within bounds"""
        candidates = []
        for _ in range(n_candidates):
            candidate = []
            for lower, upper in bounds:
                candidate.append(np.random.uniform(lower, upper))
            candidates.append(candidate)
        return np.array(candidates)


class ExpectedImprovement(BaseAcquisitionFunction):
    """Expected Improvement acquisition function"""
    
    def __init__(self, surrogate_model, xi: float = 0.01):
        super().__init__(surrogate_model)
        self.xi = xi  # Exploration-exploitation trade-off
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Compute Expected Improvement"""
        if not hasattr(self.surrogate_model, 'predict'):
            raise ValueError("Surrogate model must have predict method")
        
        # Get predictions
        mean, std = self.surrogate_model.predict(X, return_std=True)
        
        # Get best observed value
        if hasattr(self.surrogate_model, 'y_observed'):
            y_best = np.max(self.surrogate_model.y_observed)
        else:
            y_best = 0.0  # Default if no observations available
        
        # Avoid division by zero
        std = np.maximum(std, 1e-9)
        
        # Calculate improvement
        improvement = mean - y_best - self.xi
        
        # Calculate EI
        z = improvement / std
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        
        # Set negative values to zero
        ei = np.maximum(ei, 0.0)
        
        return ei


class UpperConfidenceBound(BaseAcquisitionFunction):
    """Upper Confidence Bound acquisition function"""
    
    def __init__(self, surrogate_model, beta: float = 2.0, adaptive: bool = True):
        super().__init__(surrogate_model)
        self.beta = beta
        self.adaptive = adaptive
        self.iteration = 0
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Compute Upper Confidence Bound"""
        if not hasattr(self.surrogate_model, 'predict'):
            raise ValueError("Surrogate model must have predict method")
        
        # Get predictions
        mean, std = self.surrogate_model.predict(X, return_std=True)
        
        # Adaptive beta
        if self.adaptive:
            # GP-UCB adaptive beta
            d = X.shape[1] if X.ndim > 1 else 1
            delta = 0.1
            self.beta = np.sqrt(2 * np.log((self.iteration + 1)**(d/2 + 2) * np.pi**2 / (3 * delta)))
        
        # Calculate UCB
        ucb = mean + self.beta * std
        
        return ucb
    
    def update_iteration(self, iteration: int):
        """Update iteration counter for adaptive beta"""
        self.iteration = iteration


class ProbabilityImprovement(BaseAcquisitionFunction):
    """Probability of Improvement acquisition function"""
    
    def __init__(self, surrogate_model, xi: float = 0.01):
        super().__init__(surrogate_model)
        self.xi = xi
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Compute Probability of Improvement"""
        if not hasattr(self.surrogate_model, 'predict'):
            raise ValueError("Surrogate model must have predict method")
        
        # Get predictions
        mean, std = self.surrogate_model.predict(X, return_std=True)
        
        # Get best observed value
        if hasattr(self.surrogate_model, 'y_observed'):
            y_best = np.max(self.surrogate_model.y_observed)
        else:
            y_best = 0.0
        
        # Avoid division by zero
        std = np.maximum(std, 1e-9)
        
        # Calculate PI
        z = (mean - y_best - self.xi) / std
        pi = norm.cdf(z)
        
        return pi


class EntropySearch(BaseAcquisitionFunction):
    """Entropy Search acquisition function (simplified version)"""
    
    def __init__(self, surrogate_model, n_samples: int = 100):
        super().__init__(surrogate_model)
        self.n_samples = n_samples
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Compute Entropy Search acquisition values"""
        if not hasattr(self.surrogate_model, 'predict'):
            raise ValueError("Surrogate model must have predict method")
        
        # Get predictions
        mean, std = self.surrogate_model.predict(X, return_std=True)
        
        # Simplified entropy search: prefer points with high uncertainty
        # In a full implementation, this would compute the reduction in entropy
        # of the posterior distribution over the global optimum
        
        # For now, we use uncertainty as a proxy
        entropy_search = std
        
        return entropy_search


class MultiObjectiveAcquisition(BaseAcquisitionFunction):
    """Multi-objective acquisition function for Pareto optimization"""
    
    def __init__(self, surrogate_models: List, weights: Optional[List[float]] = None,
                 method: str = 'weighted_sum'):
        super().__init__(surrogate_models[0])  # Use first model as base
        self.surrogate_models = surrogate_models
        self.weights = weights or [1.0] * len(surrogate_models)
        self.method = method
        
        if len(self.weights) != len(self.surrogate_models):
            raise ValueError("Number of weights must match number of surrogate models")
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Compute multi-objective acquisition values"""
        if self.method == 'weighted_sum':
            return self._weighted_sum(X)
        elif self.method == 'pareto_front':
            return self._pareto_front(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _weighted_sum(self, X: np.ndarray) -> np.ndarray:
        """Weighted sum of individual acquisition functions"""
        total_acq = np.zeros(X.shape[0])
        
        for i, (surrogate, weight) in enumerate(zip(self.surrogate_models, self.weights)):
            # Use Expected Improvement for each objective
            ei = ExpectedImprovement(surrogate)
            acq_values = ei(X)
            total_acq += weight * acq_values
        
        return total_acq
    
    def _pareto_front(self, X: np.ndarray) -> np.ndarray:
        """Pareto front-based acquisition function"""
        # Get predictions from all surrogates
        predictions = []
        for surrogate in self.surrogate_models:
            mean, _ = surrogate.predict(X, return_std=True)
            predictions.append(mean)
        
        predictions = np.array(predictions)
        
        # Calculate hypervolume contribution (simplified)
        # In practice, this would use proper hypervolume calculation
        pareto_scores = np.sum(predictions, axis=0)
        
        return pareto_scores


class KnowledgeGradient(BaseAcquisitionFunction):
    """Knowledge Gradient acquisition function"""
    
    def __init__(self, surrogate_model, n_samples: int = 50):
        super().__init__(surrogate_model)
        self.n_samples = n_samples
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Compute Knowledge Gradient acquisition values"""
        if not hasattr(self.surrogate_model, 'predict'):
            raise ValueError("Surrogate model must have predict method")
        
        # Get current best value
        if hasattr(self.surrogate_model, 'y_observed'):
            current_best = np.max(self.surrogate_model.y_observed)
        else:
            current_best = 0.0
        
        kg_values = np.zeros(X.shape[0])
        
        for i, x in enumerate(X):
            # Sample from posterior at x
            mean, std = self.surrogate_model.predict(x.reshape(1, -1), return_std=True)
            
            # Generate samples
            samples = np.random.normal(mean[0], std[0], self.n_samples)
            
            # Calculate expected improvement
            improvements = np.maximum(samples - current_best, 0)
            kg_values[i] = np.mean(improvements)
        
        return kg_values


class ThompsonSampling(BaseAcquisitionFunction):
    """Thompson Sampling for Bayesian Optimization"""
    
    def __init__(self, surrogate_model, n_samples: int = 10):
        super().__init__(surrogate_model)
        self.n_samples = n_samples
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Compute Thompson Sampling acquisition values"""
        if not hasattr(self.surrogate_model, 'predict'):
            raise ValueError("Surrogate model must have predict method")
        
        # Get predictions
        mean, std = self.surrogate_model.predict(X, return_std=True)
        
        # Generate samples from posterior
        samples = np.random.normal(mean, std, (self.n_samples, len(mean)))
        
        # Take maximum across samples
        ts_values = np.max(samples, axis=0)
        
        return ts_values


class AdaptiveAcquisitionFunction(BaseAcquisitionFunction):
    """Adaptive acquisition function that switches between different strategies"""
    
    def __init__(self, surrogate_model, strategies: List[str] = None,
                 switch_frequency: int = 10):
        super().__init__(surrogate_model)
        self.strategies = strategies or ['ei', 'ucb', 'pi']
        self.switch_frequency = switch_frequency
        self.current_strategy_idx = 0
        self.iteration = 0
        
        # Initialize strategy functions
        self.strategy_functions = {
            'ei': ExpectedImprovement(surrogate_model),
            'ucb': UpperConfidenceBound(surrogate_model),
            'pi': ProbabilityImprovement(surrogate_model),
            'es': EntropySearch(surrogate_model),
            'ts': ThompsonSampling(surrogate_model)
        }
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Compute adaptive acquisition values"""
        # Switch strategy periodically
        if self.iteration % self.switch_frequency == 0:
            self.current_strategy_idx = (self.current_strategy_idx + 1) % len(self.strategies)
        
        current_strategy = self.strategies[self.current_strategy_idx]
        strategy_func = self.strategy_functions[current_strategy]
        
        # Update iteration for UCB if needed
        if current_strategy == 'ucb' and hasattr(strategy_func, 'update_iteration'):
            strategy_func.update_iteration(self.iteration)
        
        return strategy_func(X)
    
    def update_iteration(self, iteration: int):
        """Update iteration counter"""
        self.iteration = iteration


class ConstrainedAcquisitionFunction(BaseAcquisitionFunction):
    """Acquisition function with constraint handling"""
    
    def __init__(self, surrogate_model, constraint_models: List,
                 constraint_bounds: List[Tuple[float, float]],
                 penalty_factor: float = 1e6):
        super().__init__(surrogate_model)
        self.constraint_models = constraint_models
        self.constraint_bounds = constraint_bounds
        self.penalty_factor = penalty_factor
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Compute constrained acquisition values"""
        # Get main acquisition values
        main_acq = ExpectedImprovement(self.surrogate_model)(X)
        
        # Check constraints
        constraint_violations = np.zeros(X.shape[0])
        
        for i, (constraint_model, (lower, upper)) in enumerate(zip(self.constraint_models, self.constraint_bounds)):
            if hasattr(constraint_model, 'predict'):
                mean, std = constraint_model.predict(X, return_std=True)
                
                # Calculate violation probability
                if upper is not None:
                    violation_upper = norm.cdf((mean - upper) / std)
                else:
                    violation_upper = 0.0
                
                if lower is not None:
                    violation_lower = norm.cdf((lower - mean) / std)
                else:
                    violation_lower = 0.0
                
                constraint_violations += violation_upper + violation_lower
        
        # Apply penalty
        constrained_acq = main_acq - self.penalty_factor * constraint_violations
        
        return constrained_acq


class AcquisitionFunctionAnalyzer:
    """Analyzer for acquisition function behavior and performance"""
    
    def __init__(self, acquisition_functions: List[BaseAcquisitionFunction]):
        self.acquisition_functions = acquisition_functions
    
    def compare_acquisition_functions(self, X: np.ndarray, 
                                    labels: Optional[List[str]] = None) -> Dict:
        """Compare different acquisition functions"""
        if labels is None:
            labels = [f"AF_{i}" for i in range(len(self.acquisition_functions))]
        
        results = {}
        
        plt.figure(figsize=(15, 10))
        
        for i, (af, label) in enumerate(zip(self.acquisition_functions, labels)):
            try:
                acq_values = af(X)
                results[label] = {
                    'mean': np.mean(acq_values),
                    'std': np.std(acq_values),
                    'min': np.min(acq_values),
                    'max': np.max(acq_values),
                    'values': acq_values
                }
                
                # Plot distribution
                plt.subplot(2, 3, i + 1)
                plt.hist(acq_values, bins=20, alpha=0.7, label=label)
                plt.xlabel('Acquisition Value')
                plt.ylabel('Frequency')
                plt.title(f'{label} Distribution')
                plt.legend()
                plt.grid(True)
                
            except Exception as e:
                print(f"Error evaluating {label}: {e}")
                results[label] = {'error': str(e)}
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    def plot_acquisition_landscape(self, acquisition_function: BaseAcquisitionFunction,
                                 bounds: List[Tuple[float, float]], 
                                 n_points: int = 100):
        """Plot acquisition function landscape"""
        # Generate grid
        x1 = np.linspace(bounds[0][0], bounds[0][1], n_points)
        x2 = np.linspace(bounds[1][0], bounds[1][1], n_points)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.column_stack([X1.ravel(), X2.ravel()])
        
        # Evaluate acquisition function
        try:
            acq_values = acquisition_function(X)
            acq_values = acq_values.reshape(X1.shape)
            
            # Plot
            plt.figure(figsize=(10, 8))
            contour = plt.contourf(X1, X2, acq_values, levels=20, cmap='viridis')
            plt.colorbar(contour, label='Acquisition Value')
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.title('Acquisition Function Landscape')
            plt.grid(True)
            plt.show()
            
        except Exception as e:
            print(f"Error plotting acquisition landscape: {e}")
    
    def analyze_convergence(self, acquisition_function: BaseAcquisitionFunction,
                           X_history: List[np.ndarray], y_history: List[float]) -> Dict:
        """Analyze convergence behavior of acquisition function"""
        if len(X_history) < 2:
            return {}
        
        # Calculate acquisition values over iterations
        acq_values_history = []
        
        for X in X_history:
            try:
                acq_values = acquisition_function(X)
                acq_values_history.append(np.max(acq_values))
            except:
                acq_values_history.append(0.0)
        
        # Plot convergence
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(acq_values_history, 'b-', label='Max Acquisition Value')
        plt.xlabel('Iteration')
        plt.ylabel('Acquisition Value')
        plt.title('Acquisition Function Convergence')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(y_history, 'r-', label='Best Objective Value')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title('Objective Function Convergence')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'acq_values_history': acq_values_history,
            'final_acq_value': acq_values_history[-1] if acq_values_history else 0.0,
            'convergence_rate': np.mean(np.diff(acq_values_history[-10:])) if len(acq_values_history) >= 10 else 0.0
        }


if __name__ == "__main__":
    # Example usage of acquisition functions
    print("Acquisition Functions for Bayesian NAS")
    print("=" * 50)
    
    # Create dummy surrogate model
    class DummySurrogate:
        def __init__(self):
            self.y_observed = [0.5, 0.8, 0.3, 0.9]
        
        def predict(self, X, return_std=True):
            # Dummy predictions
            mean = np.sin(X[:, 0]) + 0.1 * np.random.randn(X.shape[0])
            std = 0.1 * np.ones(X.shape[0])
            return mean, std
    
    surrogate = DummySurrogate()
    
    # Test different acquisition functions
    acquisition_functions = [
        ExpectedImprovement(surrogate),
        UpperConfidenceBound(surrogate),
        ProbabilityImprovement(surrogate),
        EntropySearch(surrogate),
        ThompsonSampling(surrogate)
    ]
    
    labels = ['EI', 'UCB', 'PI', 'ES', 'TS']
    
    # Generate test points
    X_test = np.random.randn(100, 2)
    
    # Compare acquisition functions
    analyzer = AcquisitionFunctionAnalyzer(acquisition_functions)
    results = analyzer.compare_acquisition_functions(X_test, labels)
    
    # Print results
    for label, result in results.items():
        if 'error' not in result:
            print(f"\n{label}:")
            print(f"  Mean: {result['mean']:.4f}")
            print(f"  Std:  {result['std']:.4f}")
            print(f"  Min:  {result['min']:.4f}")
            print(f"  Max:  {result['max']:.4f}")
        else:
            print(f"\n{label}: Error - {result['error']}")
    
    # Test adaptive acquisition function
    print("\nTesting Adaptive Acquisition Function...")
    adaptive_af = AdaptiveAcquisitionFunction(surrogate, ['ei', 'ucb', 'pi'])
    
    for i in range(5):
        adaptive_af.update_iteration(i)
        acq_values = adaptive_af(X_test[:10])
        print(f"  Iteration {i}: Max acquisition value = {np.max(acq_values):.4f}")
    
    print("\nAcquisition functions testing completed!")
