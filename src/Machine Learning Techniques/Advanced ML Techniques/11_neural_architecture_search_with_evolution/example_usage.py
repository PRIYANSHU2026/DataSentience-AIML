"""
Example Usage: Neural Architecture Search with Evolution

This file demonstrates practical usage of evolutionary NAS including:
- Basic evolutionary search
- Multi-objective optimization
- Advanced genetic operations
- Result analysis and visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from evolutionary_nas import (
    ArchitectureConfig, ArchitectureIndividual, Population,
    GeneticOperations, EvolutionaryNAS
)


def example_1_basic_evolutionary_search():
    """Example 1: Basic evolutionary architecture search"""
    print("=" * 60)
    print("Example 1: Basic Evolutionary Architecture Search")
    print("=" * 60)
    
    # Create configuration
    config = ArchitectureConfig(
        max_layers=6,
        min_layers=3,
        max_filters=128,
        min_filters=16,
        layer_types=['conv', 'maxpool', 'dropout', 'batch_norm']
    )
    
    # Initialize evolutionary NAS
    nas = EvolutionaryNAS(
        config=config,
        population_size=30,
        mutation_rate=0.15,
        crossover_rate=0.7,
        elite_size=3,
        tournament_size=3
    )
    
    # Create dummy data loaders
    train_loader, val_loader = create_dummy_data_loaders()
    
    # Run evolution
    best_architecture = nas.evolve(
        train_loader=train_loader,
        val_loader=val_loader,
        generations=15,
        max_epochs=5,
        device='cpu',
        verbose=True
    )
    
    print(f"\nBest architecture found:")
    print(f"Fitness: {best_architecture.fitness:.4f}")
    print(f"Number of layers: {len(best_architecture.genes)}")
    
    # Print architecture details
    for i, layer in enumerate(best_architecture.genes):
        print(f"Layer {i+1}: {layer}")
    
    return nas, best_architecture


def example_2_multi_objective_optimization():
    """Example 2: Multi-objective optimization (accuracy vs efficiency)"""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Objective Optimization")
    print("=" * 60)
    
    class MultiObjectiveNAS(EvolutionaryNAS):
        def fitness_function(self, individual, train_loader, val_loader, max_epochs=10, device='cpu'):
            """Multi-objective fitness function"""
            try:
                # Build and train model
                model = individual.get_model((3, 32, 32), 10)
                model.to(device)
                
                # Training setup
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                # Training loop
                best_val_acc = 0.0
                for epoch in range(max_epochs):
                    model.train()
                    for data, target in train_loader:
                        data, target = data.to(device), target.to(device)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                    
                    # Validation
                    model.eval()
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for data, target in val_loader:
                            data, target = data.to(device), target.to(device)
                            output = model(data)
                            _, predicted = torch.max(output.data, 1)
                            total += target.size(0)
                            correct += (predicted == target).sum().item()
                    
                    val_acc = correct / total
                    best_val_acc = max(best_val_acc, val_acc)
                
                # Calculate efficiency metrics
                num_params = sum(p.numel() for p in model.parameters())
                model_size_mb = num_params * 4 / (1024 * 1024)  # Assuming float32
                
                # Multi-objective fitness (Pareto front approach)
                # Weighted sum of accuracy and efficiency
                accuracy_weight = 0.7
                efficiency_weight = 0.3
                
                # Normalize efficiency (smaller is better)
                max_model_size = 50.0  # MB
                normalized_efficiency = 1.0 - (model_size_mb / max_model_size)
                
                fitness = (accuracy_weight * best_val_acc + 
                          efficiency_weight * normalized_efficiency)
                
                # Store additional metrics
                individual.accuracy = best_val_acc
                individual.model_size = model_size_mb
                individual.num_params = num_params
                
                return fitness
            
            except Exception as e:
                return 0.0
    
    # Create configuration for efficient architectures
    config = ArchitectureConfig(
        max_layers=5,
        min_layers=2,
        max_filters=64,
        min_filters=8,
        layer_types=['conv', 'maxpool', 'dropout']
    )
    
    # Initialize multi-objective NAS
    nas = MultiObjectiveNAS(
        config=config,
        population_size=25,
        mutation_rate=0.2,
        crossover_rate=0.6,
        elite_size=2
    )
    
    # Create data loaders
    train_loader, val_loader = create_dummy_data_loaders()
    
    # Run evolution
    best_architecture = nas.evolve(
        train_loader=train_loader,
        val_loader=val_loader,
        generations=12,
        max_epochs=3,
        device='cpu',
        verbose=True
    )
    
    print(f"\nMulti-objective results:")
    print(f"Fitness: {best_architecture.fitness:.4f}")
    print(f"Accuracy: {best_architecture.accuracy:.4f}")
    print(f"Model size: {best_architecture.model_size:.2f} MB")
    print(f"Parameters: {best_architecture.num_params:,}")
    
    return nas, best_architecture


def example_3_advanced_genetic_operations():
    """Example 3: Advanced genetic operations and selection strategies"""
    print("\n" + "=" * 60)
    print("Example 3: Advanced Genetic Operations")
    print("=" * 60)
    
    class AdvancedGeneticOperations(GeneticOperations):
        @staticmethod
        def rank_based_selection(population, selection_pressure=1.5):
            """Rank-based selection"""
            # Sort population by fitness
            sorted_pop = sorted(population, key=lambda x: x.fitness if x.fitness is not None else -float('inf'), reverse=True)
            
            # Calculate selection probabilities
            n = len(sorted_pop)
            probabilities = []
            for i in range(n):
                prob = (2 - selection_pressure) / n + (2 * selection_pressure * (n - i - 1)) / (n * (n - 1))
                probabilities.append(prob)
            
            # Select based on probabilities
            r = random.random()
            cumulative = 0.0
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    return sorted_pop[i]
            
            return sorted_pop[-1]
        
        @staticmethod
        def arithmetic_crossover(parent1, parent2, alpha=0.5):
            """Arithmetic crossover for continuous parameters"""
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)
            
            child1.age = 0
            child2.age = 0
            child1.fitness = None
            child2.fitness = None
            
            # Arithmetic crossover for layer parameters
            min_layers = min(len(parent1.genes), len(parent2.genes))
            
            for i in range(min_layers):
                if parent1.genes[i]['type'] == parent2.genes[i]['type'] == 'conv':
                    # Crossover filter counts
                    f1 = parent1.genes[i]['filters']
                    f2 = parent2.genes[i]['filters']
                    child1.genes[i]['filters'] = int(alpha * f1 + (1 - alpha) * f2)
                    child2.genes[i]['filters'] = int(alpha * f2 + (1 - alpha) * f1)
                    
                    # Crossover kernel sizes
                    k1 = parent1.genes[i]['kernel_size']
                    k2 = parent2.genes[i]['kernel_size']
                    child1.genes[i]['kernel_size'] = int(alpha * k1 + (1 - alpha) * k2)
                    child2.genes[i]['kernel_size'] = int(alpha * k2 + (1 - alpha) * k1)
            
            return child1, child2
    
    # Test advanced genetic operations
    config = ArchitectureConfig(max_layers=4, min_layers=2)
    
    # Create test individuals
    individual1 = ArchitectureIndividual(config)
    individual2 = ArchitectureIndividual(config)
    
    # Set fitness values for testing
    individual1.fitness = 0.8
    individual2.fitness = 0.6
    
    print("Testing advanced genetic operations:")
    
    # Test arithmetic crossover
    child1, child2 = AdvancedGeneticOperations.arithmetic_crossover(individual1, individual2)
    print(f"Arithmetic crossover created children with {len(child1.genes)} and {len(child2.genes)} layers")
    
    # Test adaptive mutation
    mutated = AdvancedGeneticOperations.adaptive_mutation(individual1, generation=5, max_generations=20)
    print(f"Adaptive mutation created individual with {len(mutated.genes)} layers")
    
    return AdvancedGeneticOperations


def example_4_population_diversity_analysis():
    """Example 4: Population diversity analysis and maintenance"""
    print("\n" + "=" * 60)
    print("Example 4: Population Diversity Analysis")
    print("=" * 60)
    
    class DiversityAwarePopulation(Population):
        def __init__(self, config, size=100):
            super().__init__(config, size)
            self.diversity_history = []
        
        def calculate_genotypic_diversity(self):
            """Calculate genotypic diversity based on architecture differences"""
            if len(self.individuals) < 2:
                return 0.0
            
            # Calculate pairwise differences
            differences = []
            for i in range(len(self.individuals)):
                for j in range(i + 1, len(self.individuals)):
                    diff = self._calculate_architecture_difference(
                        self.individuals[i], self.individuals[j]
                    )
                    differences.append(diff)
            
            return np.mean(differences) if differences else 0.0
        
        def _calculate_architecture_difference(self, ind1, ind2):
            """Calculate difference between two architectures"""
            # Compare number of layers
            layer_diff = abs(len(ind1.genes) - len(ind2.genes))
            
            # Compare layer types
            min_layers = min(len(ind1.genes), len(ind2.genes))
            type_diff = sum(1 for i in range(min_layers) 
                          if ind1.genes[i]['type'] != ind2.genes[i]['type'])
            
            # Compare parameters for same layer types
            param_diff = 0
            for i in range(min_layers):
                if ind1.genes[i]['type'] == ind2.genes[i]['type']:
                    layer_type = ind1.genes[i]['type']
                    if layer_type == 'conv':
                        param_diff += abs(ind1.genes[i].get('filters', 0) - ind2.genes[i].get('filters', 0))
                        param_diff += abs(ind1.genes[i].get('kernel_size', 0) - ind2.genes[i].get('kernel_size', 0))
            
            return layer_diff + type_diff + param_diff * 0.1
        
        def maintain_diversity(self, min_diversity=0.5):
            """Maintain population diversity by replacing similar individuals"""
            current_diversity = self.calculate_genotypic_diversity()
            
            if current_diversity < min_diversity:
                # Find and replace similar individuals
                for i in range(len(self.individuals)):
                    for j in range(i + 1, len(self.individuals)):
                        diff = self._calculate_architecture_difference(
                            self.individuals[i], self.individuals[j]
                        )
                        if diff < 2:  # Very similar individuals
                            # Replace one with a random individual
                            self.individuals[j] = ArchitectureIndividual(self.config)
        
        def evolve(self, mutation_rate, crossover_rate, elite_size=5):
            """Enhanced evolution with diversity maintenance"""
            # Record diversity before evolution
            self.diversity_history.append(self.calculate_genotypic_diversity())
            
            # Perform standard evolution
            super().evolve(mutation_rate, crossover_rate, elite_size)
            
            # Maintain diversity
            self.maintain_diversity()
    
    # Test diversity-aware population
    config = ArchitectureConfig(max_layers=5, min_layers=2)
    population = DiversityAwarePopulation(config, size=20)
    
    print("Population diversity analysis:")
    print(f"Initial diversity: {population.calculate_genotypic_diversity():.3f}")
    
    # Evolve population
    population.evolve(mutation_rate=0.2, crossover_rate=0.7, elite_size=2)
    print(f"Diversity after evolution: {population.calculate_genotypic_diversity():.3f}")
    print(f"Diversity history: {population.diversity_history}")
    
    return population


def example_5_evolutionary_analysis():
    """Example 5: Comprehensive evolutionary analysis and visualization"""
    print("\n" + "=" * 60)
    print("Example 5: Evolutionary Analysis")
    print("=" * 60)
    
    # Run a complete evolutionary search
    config = ArchitectureConfig(
        max_layers=6,
        min_layers=3,
        max_filters=128,
        min_filters=16
    )
    
    nas = EvolutionaryNAS(
        config=config,
        population_size=40,
        mutation_rate=0.12,
        crossover_rate=0.75,
        elite_size=4
    )
    
    # Create data loaders
    train_loader, val_loader = create_dummy_data_loaders()
    
    # Run evolution
    best_architecture = nas.evolve(
        train_loader=train_loader,
        val_loader=val_loader,
        generations=20,
        max_epochs=3,
        device='cpu',
        verbose=True
    )
    
    # Comprehensive analysis
    print("\nEvolutionary Analysis Results:")
    
    # Plot evolution progress
    nas.plot_evolution('evolution_analysis.png')
    
    # Get statistics
    stats = nas.get_statistics()
    print(f"Best fitness achieved: {stats['best_fitness']:.4f}")
    print(f"Fitness improvement: {stats['improvement']:.4f}")
    print(f"Final diversity: {stats['final_diversity']:.3f}")
    
    # Analyze best architecture
    print(f"\nBest Architecture Analysis:")
    print(f"Number of layers: {len(best_architecture.genes)}")
    print(f"Layer types: {[layer['type'] for layer in best_architecture.genes]}")
    
    # Count layer types
    layer_counts = {}
    for layer in best_architecture.genes:
        layer_type = layer['type']
        layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
    
    print(f"Layer distribution: {layer_counts}")
    
    # Analyze parameter distribution
    conv_layers = [layer for layer in best_architecture.genes if layer['type'] == 'conv']
    if conv_layers:
        filters = [layer['filters'] for layer in conv_layers]
        kernels = [layer['kernel_size'] for layer in conv_layers]
        print(f"Filter sizes: {filters}")
        print(f"Kernel sizes: {kernels}")
        print(f"Average filters: {np.mean(filters):.1f}")
        print(f"Average kernel size: {np.mean(kernels):.1f}")
    
    # Create detailed visualization
    create_evolutionary_visualization(nas)
    
    return nas, best_architecture


def create_evolutionary_visualization(nas):
    """Create detailed evolutionary visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data
    generations = [h['generation'] for h in nas.evolution_history]
    best_fitness = [h['best_fitness'] for h in nas.evolution_history]
    avg_fitness = [h['avg_fitness'] for h in nas.evolution_history]
    diversity = [h['diversity'] for h in nas.evolution_history]
    
    # Plot 1: Fitness evolution
    axes[0, 0].plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
    axes[0, 0].plot(generations, avg_fitness, 'r--', label='Average Fitness', linewidth=2)
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Fitness')
    axes[0, 0].set_title('Fitness Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Diversity evolution
    axes[0, 1].plot(generations, diversity, 'g-', label='Population Diversity', linewidth=2)
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Diversity')
    axes[0, 1].set_title('Population Diversity')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Fitness distribution over generations
    fitness_distributions = []
    for h in nas.evolution_history:
        fitnesses = [ind.fitness for ind in nas.population.individuals if ind.fitness is not None]
        fitness_distributions.append(fitnesses)
    
    axes[1, 0].boxplot(fitness_distributions, positions=generations)
    axes[1, 0].set_xlabel('Generation')
    axes[1, 0].set_ylabel('Fitness')
    axes[1, 0].set_title('Fitness Distribution')
    axes[1, 0].grid(True)
    
    # Plot 4: Architecture complexity evolution
    complexity_history = []
    for h in nas.evolution_history:
        best_arch = h['best_architecture']
        complexity = len(best_arch.genes)
        complexity_history.append(complexity)
    
    axes[1, 1].plot(generations, complexity_history, 'purple', label='Best Architecture Complexity', linewidth=2)
    axes[1, 1].set_xlabel('Generation')
    axes[1, 1].set_ylabel('Number of Layers')
    axes[1, 1].set_title('Architecture Complexity Evolution')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('detailed_evolution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Detailed evolutionary analysis saved to 'detailed_evolution_analysis.png'")


def create_dummy_data_loaders():
    """Create dummy data loaders for testing"""
    # Create dummy dataset
    X = torch.randn(800, 3, 32, 32)
    y = torch.randint(0, 10, (800,))
    
    # Split into train/val
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader


def main():
    """Run all evolutionary NAS examples"""
    print("Evolutionary Neural Architecture Search Examples")
    print("=" * 80)
    
    # Run all examples
    nas1, best1 = example_1_basic_evolutionary_search()
    example_2_multi_objective_optimization()
    example_3_advanced_genetic_operations()
    example_4_population_diversity_analysis()
    nas5, best5 = example_5_evolutionary_analysis()
    
    print("\n" + "=" * 80)
    print("All evolutionary NAS examples completed!")
    print("=" * 80)
    
    # Summary of results
    print(f"\nSummary:")
    print(f"Basic search best fitness: {best1.fitness:.4f}")
    print(f"Comprehensive search best fitness: {best5.fitness:.4f}")
    print(f"Best architecture complexity: {len(best5.genes)} layers")


if __name__ == "__main__":
    main()
