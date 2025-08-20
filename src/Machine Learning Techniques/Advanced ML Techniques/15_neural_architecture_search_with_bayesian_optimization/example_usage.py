"""
Example Usage for Neural Architecture Search with Bayesian Optimization

This module provides comprehensive examples demonstrating various aspects of
Bayesian NAS, including different acquisition functions, surrogate models,
and advanced optimization strategies.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

# Import our Bayesian NAS components
from bayesian_nas import (
    ArchitectureConfig, ArchitectureEncoder, ArchitectureEvaluator,
    BayesianOptimizer, BayesianNAS
)
from surrogate_models import (
    GaussianProcessSurrogate, MultiTaskGaussianProcess,
    DeepKernelGaussianProcess, EnsembleSurrogate
)
from acquisition_functions import (
    ExpectedImprovement, UpperConfidenceBound, ProbabilityImprovement,
    EntropySearch, MultiObjectiveAcquisition, KnowledgeGradient,
    ThompsonSampling, AdaptiveAcquisitionFunction, ConstrainedAcquisitionFunction
)


def create_dummy_data_loaders():
    """Create dummy data loaders for demonstration"""
    dummy_data = torch.randn(200, 3, 32, 32)
    dummy_labels = torch.randint(0, 10, (200,))

    class DummyDataset:
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    dataset = DummyDataset(dummy_data, dummy_labels)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    return {'train': train_loader, 'val': val_loader}


def example_1_basic_bayesian_nas():
    """Basic Bayesian NAS with Expected Improvement"""
    print("Example 1: Basic Bayesian NAS")
    print("=" * 50)

    config = ArchitectureConfig(
        max_iterations=10,
        n_initial_points=3,
        acquisition_function='ei',
        kernel_type='rbf',
        epochs_per_evaluation=3
    )

    data_loaders = create_dummy_data_loaders()
    bayesian_nas = BayesianNAS(config, data_loaders)
    results = bayesian_nas.search()

    print(f"Best accuracy: {results['best_performance']:.2f}%")
    print(f"Search time: {results['search_time']:.2f} seconds")
    
    bayesian_nas.plot_search_progress()
    return results


def example_2_multi_objective_optimization():
    """Multi-objective Bayesian NAS optimizing accuracy and model size"""
    print("\nExample 2: Multi-Objective Bayesian NAS")
    print("=" * 50)

    class MultiObjectiveBayesianNAS:
        def __init__(self, config, data_loaders):
            self.config = config
            self.data_loaders = data_loaders
            self.evaluator = ArchitectureEvaluator(config, data_loaders)
            self.accuracy_surrogate = GaussianProcessSurrogate(kernel_type='rbf')
            self.size_surrogate = GaussianProcessSurrogate(kernel_type='rbf')
            self.acquisition = MultiObjectiveAcquisition(
                [self.accuracy_surrogate, self.size_surrogate],
                weights=[1.0, -0.1]
            )
            self.search_history = []

        def search(self, max_iterations=8):
            print("Starting multi-objective Bayesian NAS...")
            
            for iteration in range(max_iterations):
                architecture = self._random_architecture()
                result = self.evaluator.evaluate_architecture(architecture)
                
                accuracy = result['val_accuracy']
                model_size = result['model_size_mb']
                features = self.evaluator.encoder.encode_architecture(architecture)
                
                if iteration == 0:
                    self.accuracy_surrogate.fit(features.reshape(1, -1), [accuracy])
                    self.size_surrogate.fit(features.reshape(1, -1), [model_size])
                else:
                    X_acc = np.vstack([self.accuracy_surrogate.X_observed, features.reshape(1, -1)])
                    y_acc = self.accuracy_surrogate.y_observed + [accuracy]
                    self.accuracy_surrogate.fit(X_acc, y_acc)
                    
                    X_size = np.vstack([self.size_surrogate.X_observed, features.reshape(1, -1)])
                    y_size = self.size_surrogate.y_observed + [model_size]
                    self.size_surrogate.fit(X_size, y_size)
                
                result['iteration'] = iteration
                result['objectives'] = [accuracy, model_size]
                self.search_history.append(result)
                
                print(f"Iteration {iteration + 1}: Accuracy: {accuracy:.2f}%, Size: {model_size:.2f} MB")

            return self.search_history

        def _random_architecture(self):
            num_layers = np.random.randint(3, 6)
            architecture = {'layers': []}
            for _ in range(num_layers):
                layer = {
                    'type': np.random.choice(['conv3x3', 'conv1x1', 'maxpool3x3']),
                    'channels': np.random.randint(16, 128),
                    'kernel_size': np.random.randint(1, 4),
                    'activation': 'relu'
                }
                architecture['layers'].append(layer)
            return architecture

        def plot_pareto_front(self):
            if not self.search_history:
                return
            
            accuracies = [r['objectives'][0] for r in self.search_history]
            sizes = [r['objectives'][1] for r in self.search_history]
            
            plt.figure(figsize=(10, 6))
            plt.scatter(sizes, accuracies, alpha=0.7, s=100)
            plt.xlabel('Model Size (MB)')
            plt.ylabel('Validation Accuracy (%)')
            plt.title('Pareto Front: Accuracy vs Model Size')
            plt.grid(True)
            plt.show()

    config = ArchitectureConfig(max_iterations=8, epochs_per_evaluation=3)
    data_loaders = create_dummy_data_loaders()
    
    mo_nas = MultiObjectiveBayesianNAS(config, data_loaders)
    results = mo_nas.search()
    mo_nas.plot_pareto_front()
    
    return results


def example_3_acquisition_function_comparison():
    """Comparison of different acquisition functions"""
    print("\nExample 3: Acquisition Function Comparison")
    print("=" * 50)

    class AcquisitionFunctionComparison:
        def __init__(self, config, data_loaders):
            self.config = config
            self.data_loaders = data_loaders
            self.evaluator = ArchitectureEvaluator(config, data_loaders)
            self.results = {}

        def compare_acquisition_functions(self, max_iterations=6):
            print("Comparing acquisition functions...")
            
            acquisition_functions = {
                'Expected Improvement': ExpectedImprovement,
                'Upper Confidence Bound': UpperConfidenceBound,
                'Probability Improvement': ProbabilityImprovement,
                'Entropy Search': EntropySearch,
                'Thompson Sampling': ThompsonSampling
            }
            
            for af_name, af_class in acquisition_functions.items():
                print(f"\nTesting {af_name}...")
                
                surrogate = GaussianProcessSurrogate(kernel_type='rbf')
                af = af_class(surrogate)
                
                best_performance = 0.0
                performance_history = []
                
                for iteration in range(max_iterations):
                    architecture = self._random_architecture()
                    features = self.evaluator.encoder.encode_architecture(architecture)
                    result = self.evaluator.evaluate_architecture(architecture)
                    performance = result['val_accuracy']
                    
                    if iteration == 0:
                        surrogate.fit(features.reshape(1, -1), [performance])
                    else:
                        X = np.vstack([surrogate.X_observed, features.reshape(1, -1)])
                        y = surrogate.y_observed + [performance]
                        surrogate.fit(X, y)
                    
                    best_performance = max(best_performance, performance)
                    performance_history.append(best_performance)
                    
                    print(f"  Iteration {iteration + 1}: {performance:.2f}% (Best: {best_performance:.2f}%)")
                
                self.results[af_name] = performance_history

        def _random_architecture(self):
            num_layers = np.random.randint(3, 5)
            architecture = {'layers': []}
            for _ in range(num_layers):
                layer = {
                    'type': np.random.choice(['conv3x3', 'conv1x1', 'maxpool3x3']),
                    'channels': np.random.randint(16, 64),
                    'kernel_size': np.random.randint(1, 3),
                    'activation': 'relu'
                }
                architecture['layers'].append(layer)
            return architecture

        def plot_comparison(self):
            plt.figure(figsize=(12, 8))
            
            for af_name, performance_history in self.results.items():
                plt.plot(performance_history, 'o-', label=af_name, linewidth=2, markersize=6)
            
            plt.xlabel('Iteration')
            plt.ylabel('Best Performance (%)')
            plt.title('Acquisition Function Comparison')
            plt.legend()
            plt.grid(True)
            plt.show()

    config = ArchitectureConfig(max_iterations=6, epochs_per_evaluation=3)
    data_loaders = create_dummy_data_loaders()
    
    af_comparison = AcquisitionFunctionComparison(config, data_loaders)
    af_comparison.compare_acquisition_functions()
    af_comparison.plot_comparison()
    
    return af_comparison.results


def example_4_constrained_optimization():
    """Bayesian NAS with constraints (e.g., model size)"""
    print("\nExample 4: Constrained Bayesian NAS")
    print("=" * 50)

    class ConstrainedBayesianNAS:
        def __init__(self, config, data_loaders):
            self.config = config
            self.data_loaders = data_loaders
            self.evaluator = ArchitectureEvaluator(config, data_loaders)
            self.performance_surrogate = GaussianProcessSurrogate(kernel_type='rbf')
            self.size_surrogate = GaussianProcessSurrogate(kernel_type='rbf')
            self.acquisition = ConstrainedAcquisitionFunction(
                self.performance_surrogate,
                [self.size_surrogate],
                [(0, 5.0)]  # Model size constraint: max 5MB
            )
            self.search_history = []

        def search(self, max_iterations=8):
            print("Starting constrained Bayesian NAS...")
            print("Constraint: Model size <= 5MB")
            
            for iteration in range(max_iterations):
                architecture = self._random_architecture()
                features = self.evaluator.encoder.encode_architecture(architecture)
                result = self.evaluator.evaluate_architecture(architecture)
                
                performance = result['val_accuracy']
                model_size = result['model_size_mb']
                constraint_violation = max(0, model_size - 5.0)
                is_feasible = constraint_violation == 0
                
                if iteration == 0:
                    self.performance_surrogate.fit(features.reshape(1, -1), [performance])
                    self.size_surrogate.fit(features.reshape(1, -1), [model_size])
                else:
                    X_perf = np.vstack([self.performance_surrogate.X_observed, features.reshape(1, -1)])
                    y_perf = self.performance_surrogate.y_observed + [performance]
                    self.performance_surrogate.fit(X_perf, y_perf)
                    
                    X_size = np.vstack([self.size_surrogate.X_observed, features.reshape(1, -1)])
                    y_size = self.size_surrogate.y_observed + [model_size]
                    self.size_surrogate.fit(X_size, y_size)
                
                result['iteration'] = iteration
                result['constraint_violation'] = constraint_violation
                result['is_feasible'] = is_feasible
                self.search_history.append(result)
                
                status = "✓" if is_feasible else "✗"
                print(f"{status} Performance: {performance:.2f}%, Size: {model_size:.2f}MB")

            return self.search_history

        def _random_architecture(self):
            num_layers = np.random.randint(3, 6)
            architecture = {'layers': []}
            for _ in range(num_layers):
                layer = {
                    'type': np.random.choice(['conv3x3', 'conv1x1', 'maxpool3x3']),
                    'channels': np.random.randint(16, 128),
                    'kernel_size': np.random.randint(1, 4),
                    'activation': 'relu'
                }
                architecture['layers'].append(layer)
            return architecture

        def plot_constrained_results(self):
            if not self.search_history:
                return
            
            iterations = [r['iteration'] for r in self.search_history]
            performances = [r['val_accuracy'] for r in self.search_history]
            sizes = [r['model_size_mb'] for r in self.search_history]
            feasible = [r['is_feasible'] for r in self.search_history]
            
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            colors = ['green' if f else 'red' for f in feasible]
            plt.scatter(iterations, performances, c=colors, alpha=0.7, s=100)
            plt.xlabel('Iteration')
            plt.ylabel('Performance (%)')
            plt.title('Performance Over Iterations')
            plt.grid(True)
            
            plt.subplot(1, 3, 2)
            plt.scatter(iterations, sizes, c=colors, alpha=0.7, s=100)
            plt.axhline(y=5, color='red', linestyle='--', label='Constraint (5MB)')
            plt.xlabel('Iteration')
            plt.ylabel('Model Size (MB)')
            plt.title('Model Size Over Iterations')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 3, 3)
            plt.scatter(sizes, performances, c=colors, alpha=0.7, s=100)
            plt.axvline(x=5, color='red', linestyle='--', label='Constraint (5MB)')
            plt.xlabel('Model Size (MB)')
            plt.ylabel('Performance (%)')
            plt.title('Performance vs Model Size')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()

    config = ArchitectureConfig(max_iterations=8, epochs_per_evaluation=3)
    data_loaders = create_dummy_data_loaders()
    
    constrained_nas = ConstrainedBayesianNAS(config, data_loaders)
    results = constrained_nas.search()
    constrained_nas.plot_constrained_results()
    
    return results


def example_5_surrogate_model_comparison():
    """Comparison of different surrogate models"""
    print("\nExample 5: Surrogate Model Comparison")
    print("=" * 50)

    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(50, 10)
    y = np.sin(X[:, 0]) + 0.1 * np.random.randn(50)

    # Test different surrogate models
    surrogate_models = {
        'RBF GP': GaussianProcessSurrogate(kernel_type='rbf'),
        'Matern GP': GaussianProcessSurrogate(kernel_type='matern'),
        'Rational Quadratic GP': GaussianProcessSurrogate(kernel_type='rational_quadratic'),
        'Deep Kernel GP': DeepKernelGaussianProcess(input_dim=10, hidden_dims=[32, 16], feature_dim=8),
        'Ensemble': EnsembleSurrogate(['rbf', 'matern', 'rational_quadratic'])
    }

    results = {}
    X_test = np.random.randn(20, 10)

    for name, surrogate in surrogate_models.items():
        print(f"\nTesting {name}...")
        
        try:
            surrogate.fit(X, y)
            mean, std = surrogate.predict(X_test, return_std=True)
            
            results[name] = {
                'mean_prediction': np.mean(mean),
                'mean_uncertainty': np.mean(std),
                'predictions': mean,
                'uncertainty': std
            }
            
            print(f"  Mean prediction: {np.mean(mean):.4f}")
            print(f"  Mean uncertainty: {np.mean(std):.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[name] = {'error': str(e)}

    # Plot comparison
    plt.figure(figsize=(15, 5))
    
    # Mean predictions
    plt.subplot(1, 3, 1)
    names = [name for name, result in results.items() if 'error' not in result]
    means = [result['mean_prediction'] for name, result in results.items() if 'error' not in result]
    plt.bar(names, means)
    plt.ylabel('Mean Prediction')
    plt.title('Mean Predictions Comparison')
    plt.xticks(rotation=45)
    
    # Mean uncertainties
    plt.subplot(1, 3, 2)
    uncertainties = [result['mean_uncertainty'] for name, result in results.items() if 'error' not in result]
    plt.bar(names, uncertainties)
    plt.ylabel('Mean Uncertainty')
    plt.title('Mean Uncertainty Comparison')
    plt.xticks(rotation=45)
    
    # Prediction vs Uncertainty
    plt.subplot(1, 3, 3)
    for name, result in results.items():
        if 'error' not in result:
            plt.scatter(result['mean_prediction'], result['mean_uncertainty'], 
                       label=name, s=100, alpha=0.7)
    plt.xlabel('Mean Prediction')
    plt.ylabel('Mean Uncertainty')
    plt.title('Prediction vs Uncertainty')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    return results


def main():
    """Run all examples"""
    print("Bayesian Neural Architecture Search - Examples")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run examples
    examples = [
        ("Basic Bayesian NAS", example_1_basic_bayesian_nas),
        ("Multi-Objective Optimization", example_2_multi_objective_optimization),
        ("Acquisition Function Comparison", example_3_acquisition_function_comparison),
        ("Constrained Optimization", example_4_constrained_optimization),
        ("Surrogate Model Comparison", example_5_surrogate_model_comparison)
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            print(f"\n{'='*20} {name} {'='*20}")
            result = example_func()
            results[name] = result
            print(f"✓ {name} completed successfully")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            results[name] = None
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    successful_examples = sum(1 for result in results.values() if result is not None)
    total_examples = len(examples)
    
    print(f"Successful examples: {successful_examples}/{total_examples}")
    
    for name, result in results.items():
        status = "✓" if result is not None else "✗"
        print(f"{status} {name}")
    
    print(f"\nAll examples completed!")
    return results


if __name__ == "__main__":
    # Run all examples
    results = main()
