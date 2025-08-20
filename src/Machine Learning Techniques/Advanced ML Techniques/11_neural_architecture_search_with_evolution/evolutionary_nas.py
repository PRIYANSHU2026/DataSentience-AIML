"""
Neural Architecture Search with Evolution

This module implements evolutionary algorithms for neural architecture search,
including genetic operations, population management, and fitness evaluation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class ArchitectureConfig:
    """Configuration for neural network architecture"""
    layer_types: List[str] = None
    max_layers: int = 10
    min_layers: int = 3
    max_filters: int = 512
    min_filters: int = 16
    max_kernel_size: int = 7
    min_kernel_size: int = 1
    dropout_rates: List[float] = None
    activation_functions: List[str] = None
    
    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ['conv', 'maxpool', 'avgpool', 'dropout', 'batch_norm']
        if self.dropout_rates is None:
            self.dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
        if self.activation_functions is None:
            self.activation_functions = ['relu', 'leaky_relu', 'elu', 'swish']


class ArchitectureIndividual:
    """Represents a single neural network architecture as a genetic individual"""
    
    def __init__(self, config: ArchitectureConfig):
        self.config = config
        self.genes = self._initialize_random_architecture()
        self.fitness = None
        self.age = 0
        self.evaluation_history = []
    
    def _initialize_random_architecture(self) -> List[Dict]:
        """Initialize a random neural network architecture"""
        num_layers = random.randint(self.config.min_layers, self.config.max_layers)
        architecture = []
        
        for i in range(num_layers):
            layer = self._create_random_layer()
            architecture.append(layer)
        
        return architecture
    
    def _create_random_layer(self) -> Dict:
        """Create a random layer configuration"""
        layer_type = random.choice(self.config.layer_types)
        layer_config = {'type': layer_type}
        
        if layer_type == 'conv':
            layer_config.update({
                'filters': random.randint(self.config.min_filters, self.config.max_filters),
                'kernel_size': random.randint(self.config.min_kernel_size, self.config.max_kernel_size),
                'activation': random.choice(self.config.activation_functions),
                'batch_norm': random.choice([True, False])
            })
        elif layer_type in ['maxpool', 'avgpool']:
            layer_config.update({
                'pool_size': random.randint(2, 4),
                'stride': random.randint(1, 2)
            })
        elif layer_type == 'dropout':
            layer_config.update({
                'rate': random.choice(self.config.dropout_rates)
            })
        elif layer_type == 'batch_norm':
            layer_config.update({
                'momentum': random.uniform(0.1, 0.9)
            })
        
        return layer_config
    
    def mutate(self, mutation_rate: float = 0.1) -> 'ArchitectureIndividual':
        """Create a mutated copy of this individual"""
        mutated = copy.deepcopy(self)
        mutated.age = 0
        mutated.fitness = None
        
        for i, layer in enumerate(mutated.genes):
            if random.random() < mutation_rate:
                # Mutate layer type
                if random.random() < 0.3:
                    layer['type'] = random.choice(self.config.layer_types)
                
                # Mutate layer parameters
                if layer['type'] == 'conv':
                    if random.random() < 0.5:
                        layer['filters'] = random.randint(self.config.min_filters, self.config.max_filters)
                    if random.random() < 0.5:
                        layer['kernel_size'] = random.randint(self.config.min_kernel_size, self.config.max_kernel_size)
                    if random.random() < 0.3:
                        layer['activation'] = random.choice(self.config.activation_functions)
                    if random.random() < 0.2:
                        layer['batch_norm'] = not layer.get('batch_norm', False)
                
                elif layer['type'] in ['maxpool', 'avgpool']:
                    if random.random() < 0.5:
                        layer['pool_size'] = random.randint(2, 4)
                    if random.random() < 0.3:
                        layer['stride'] = random.randint(1, 2)
                
                elif layer['type'] == 'dropout':
                    if random.random() < 0.5:
                        layer['rate'] = random.choice(self.config.dropout_rates)
        
        # Add or remove layers
        if random.random() < mutation_rate * 0.5:
            if len(mutated.genes) < self.config.max_layers and random.random() < 0.5:
                # Add a layer
                insert_pos = random.randint(0, len(mutated.genes))
                mutated.genes.insert(insert_pos, self._create_random_layer())
            elif len(mutated.genes) > self.config.min_layers:
                # Remove a layer
                remove_pos = random.randint(0, len(mutated.genes) - 1)
                mutated.genes.pop(remove_pos)
        
        return mutated
    
    def crossover(self, other: 'ArchitectureIndividual') -> Tuple['ArchitectureIndividual', 'ArchitectureIndividual']:
        """Perform crossover with another individual to create offspring"""
        child1 = copy.deepcopy(self)
        child2 = copy.deepcopy(other)
        
        child1.age = 0
        child2.age = 0
        child1.fitness = None
        child2.fitness = None
        
        # Single-point crossover
        if len(self.genes) > 1 and len(other.genes) > 1:
            crossover_point1 = random.randint(1, len(self.genes) - 1)
            crossover_point2 = random.randint(1, len(other.genes) - 1)
            
            # Swap parts of the architectures
            child1.genes = self.genes[:crossover_point1] + other.genes[crossover_point2:]
            child2.genes = other.genes[:crossover_point2] + self.genes[crossover_point1:]
        
        return child1, child2
    
    def get_model(self, input_shape: Tuple[int, ...], num_classes: int) -> nn.Module:
        """Convert architecture genes to a PyTorch model"""
        return self._build_model(input_shape, num_classes)
    
    def _build_model(self, input_shape: Tuple[int, ...], num_classes: int) -> nn.Module:
        """Build PyTorch model from architecture genes"""
        layers = []
        current_channels = input_shape[0] if len(input_shape) == 3 else 1
        
        for layer_config in self.genes:
            layer_type = layer_config['type']
            
            if layer_type == 'conv':
                filters = layer_config['filters']
                kernel_size = layer_config['kernel_size']
                activation = layer_config['activation']
                use_batch_norm = layer_config.get('batch_norm', False)
                
                layers.append(nn.Conv2d(current_channels, filters, kernel_size, padding=kernel_size//2))
                current_channels = filters
                
                if use_batch_norm:
                    layers.append(nn.BatchNorm2d(filters))
                
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'leaky_relu':
                    layers.append(nn.LeakyReLU())
                elif activation == 'elu':
                    layers.append(nn.ELU())
                elif activation == 'swish':
                    layers.append(nn.SiLU())
            
            elif layer_type == 'maxpool':
                pool_size = layer_config['pool_size']
                stride = layer_config.get('stride', pool_size)
                layers.append(nn.MaxPool2d(pool_size, stride))
            
            elif layer_type == 'avgpool':
                pool_size = layer_config['pool_size']
                stride = layer_config.get('stride', pool_size)
                layers.append(nn.AvgPool2d(pool_size, stride))
            
            elif layer_type == 'dropout':
                rate = layer_config['rate']
                layers.append(nn.Dropout(rate))
            
            elif layer_type == 'batch_norm':
                momentum = layer_config['momentum']
                layers.append(nn.BatchNorm2d(current_channels, momentum=momentum))
        
        # Add global average pooling and classification head
        layers.extend([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(current_channels, num_classes)
        ])
        
        return nn.Sequential(*layers)
    
    def __str__(self):
        return f"Architecture(fitness={self.fitness:.4f}, age={self.age}, layers={len(self.genes)})"


class Population:
    """Manages a population of architecture individuals"""
    
    def __init__(self, config: ArchitectureConfig, size: int = 100):
        self.config = config
        self.size = size
        self.individuals = [ArchitectureIndividual(config) for _ in range(size)]
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def evaluate_fitness(self, fitness_function: Callable, *args, **kwargs):
        """Evaluate fitness for all individuals in the population"""
        for individual in self.individuals:
            if individual.fitness is None:
                individual.fitness = fitness_function(individual, *args, **kwargs)
                individual.evaluation_history.append(individual.fitness)
    
    def select_parents(self, tournament_size: int = 3) -> List[ArchitectureIndividual]:
        """Select parents using tournament selection"""
        parents = []
        
        for _ in range(2):
            # Tournament selection
            tournament = random.sample(self.individuals, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness if x.fitness is not None else -float('inf'))
            parents.append(winner)
        
        return parents
    
    def evolve(self, mutation_rate: float, crossover_rate: float, elite_size: int = 5):
        """Evolve the population for one generation"""
        # Sort by fitness
        self.individuals.sort(key=lambda x: x.fitness if x.fitness is not None else -float('inf'), reverse=True)
        
        # Keep elite individuals
        new_population = self.individuals[:elite_size].copy()
        
        # Generate offspring
        while len(new_population) < self.size:
            if random.random() < crossover_rate:
                # Crossover
                parents = self.select_parents()
                child1, child2 = parents[0].crossover(parents[1])
                
                # Mutate children
                if random.random() < mutation_rate:
                    child1 = child1.mutate(mutation_rate)
                if random.random() < mutation_rate:
                    child2 = child2.mutate(mutation_rate)
                
                new_population.extend([child1, child2])
            else:
                # Mutation only
                parent = self.select_parents()[0]
                child = parent.mutate(mutation_rate)
                new_population.append(child)
        
        # Trim to population size
        self.individuals = new_population[:self.size]
        
        # Age individuals
        for individual in self.individuals:
            individual.age += 1
        
        self.generation += 1
        
        # Record statistics
        fitnesses = [ind.fitness for ind in self.individuals if ind.fitness is not None]
        if fitnesses:
            self.best_fitness_history.append(max(fitnesses))
            self.avg_fitness_history.append(np.mean(fitnesses))
    
    def get_best_individual(self) -> ArchitectureIndividual:
        """Get the best individual in the population"""
        return max(self.individuals, key=lambda x: x.fitness if x.fitness is not None else -float('inf'))
    
    def get_diversity(self) -> float:
        """Calculate population diversity"""
        if len(self.individuals) < 2:
            return 0.0
        
        # Simple diversity metric based on architecture lengths
        lengths = [len(ind.genes) for ind in self.individuals]
        return np.std(lengths)


class GeneticOperations:
    """Collection of genetic operations for architecture evolution"""
    
    @staticmethod
    def tournament_selection(population: List[ArchitectureIndividual], tournament_size: int = 3) -> ArchitectureIndividual:
        """Tournament selection"""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness if x.fitness is not None else -float('inf'))
    
    @staticmethod
    def roulette_wheel_selection(population: List[ArchitectureIndividual]) -> ArchitectureIndividual:
        """Roulette wheel selection"""
        fitnesses = [ind.fitness if ind.fitness is not None else 0.0 for ind in population]
        total_fitness = sum(fitnesses)
        
        if total_fitness == 0:
            return random.choice(population)
        
        # Normalize fitnesses
        normalized_fitnesses = [f / total_fitness for f in fitnesses]
        
        # Roulette wheel selection
        r = random.random()
        cumulative = 0.0
        
        for i, fitness in enumerate(normalized_fitnesses):
            cumulative += fitness
            if r <= cumulative:
                return population[i]
        
        return population[-1]
    
    @staticmethod
    def uniform_crossover(parent1: ArchitectureIndividual, parent2: ArchitectureIndividual) -> Tuple[ArchitectureIndividual, ArchitectureIndividual]:
        """Uniform crossover between two parents"""
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        child1.age = 0
        child2.age = 0
        child1.fitness = None
        child2.fitness = None
        
        # Uniform crossover for each layer
        min_layers = min(len(parent1.genes), len(parent2.genes))
        
        for i in range(min_layers):
            if random.random() < 0.5:
                # Swap layer configurations
                child1.genes[i], child2.genes[i] = child2.genes[i], child1.genes[i]
        
        return child1, child2
    
    @staticmethod
    def adaptive_mutation(individual: ArchitectureIndividual, generation: int, max_generations: int) -> ArchitectureIndividual:
        """Adaptive mutation that changes rate based on generation"""
        # Decrease mutation rate over time
        base_rate = 0.1
        adaptive_rate = base_rate * (1 - generation / max_generations)
        
        return individual.mutate(adaptive_rate)


class EvolutionaryNAS:
    """Main evolutionary neural architecture search algorithm"""
    
    def __init__(self, config: ArchitectureConfig, population_size: int = 100, 
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                 elite_size: int = 5, tournament_size: int = 3):
        self.config = config
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        
        self.population = Population(config, population_size)
        self.best_architecture = None
        self.evolution_history = []
    
    def fitness_function(self, individual: ArchitectureIndividual, train_loader, val_loader, 
                        max_epochs: int = 10, device: str = 'cpu') -> float:
        """Fitness function that evaluates an architecture"""
        try:
            # Build model
            model = individual.get_model((3, 32, 32), 10)  # CIFAR-10 default
            model.to(device)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            best_val_acc = 0.0
            for epoch in range(max_epochs):
                # Training
                model.train()
                for batch_idx, (data, target) in enumerate(train_loader):
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
            
            # Return fitness (validation accuracy)
            return best_val_acc
        
        except Exception as e:
            # Return low fitness for invalid architectures
            return 0.0
    
    def evolve(self, train_loader, val_loader, generations: int = 50, 
               max_epochs: int = 10, device: str = 'cpu', verbose: bool = True) -> ArchitectureIndividual:
        """Run evolutionary search"""
        print(f"Starting evolutionary NAS with {generations} generations...")
        
        for generation in range(generations):
            # Evaluate fitness
            self.population.evaluate_fitness(
                self.fitness_function, train_loader, val_loader, max_epochs, device
            )
            
            # Get best individual
            best_individual = self.population.get_best_individual()
            
            if self.best_architecture is None or best_individual.fitness > self.best_architecture.fitness:
                self.best_architecture = copy.deepcopy(best_individual)
            
            # Record statistics
            self.evolution_history.append({
                'generation': generation,
                'best_fitness': best_individual.fitness,
                'avg_fitness': np.mean([ind.fitness for ind in self.population.individuals if ind.fitness is not None]),
                'diversity': self.population.get_diversity(),
                'best_architecture': copy.deepcopy(best_individual)
            })
            
            if verbose and generation % 5 == 0:
                print(f"Generation {generation}: Best Fitness = {best_individual.fitness:.4f}, "
                      f"Avg Fitness = {self.evolution_history[-1]['avg_fitness']:.4f}, "
                      f"Diversity = {self.evolution_history[-1]['diversity']:.2f}")
            
            # Evolve population
            if generation < generations - 1:  # Don't evolve after last generation
                self.population.evolve(self.mutation_rate, self.crossover_rate, self.elite_size)
        
        print(f"Evolution completed! Best fitness: {self.best_architecture.fitness:.4f}")
        return self.best_architecture
    
    def plot_evolution(self, save_path: str = 'evolution_progress.png'):
        """Plot evolution progress"""
        generations = [h['generation'] for h in self.evolution_history]
        best_fitness = [h['best_fitness'] for h in self.evolution_history]
        avg_fitness = [h['avg_fitness'] for h in self.evolution_history]
        diversity = [h['diversity'] for h in self.evolution_history]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Fitness plot
        ax1.plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
        ax1.plot(generations, avg_fitness, 'r--', label='Average Fitness', linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Evolution Progress')
        ax1.legend()
        ax1.grid(True)
        
        # Diversity plot
        ax2.plot(generations, diversity, 'g-', label='Population Diversity', linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Diversity')
        ax2.set_title('Population Diversity')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_statistics(self) -> Dict:
        """Get evolution statistics"""
        if not self.evolution_history:
            return {}
        
        fitnesses = [h['best_fitness'] for h in self.evolution_history]
        
        return {
            'best_fitness': max(fitnesses),
            'final_fitness': fitnesses[-1],
            'improvement': fitnesses[-1] - fitnesses[0],
            'generations': len(self.evolution_history),
            'final_diversity': self.evolution_history[-1]['diversity']
        }


def create_dummy_data_loaders():
    """Create dummy data loaders for testing"""
    # Create dummy dataset
    X = torch.randn(1000, 3, 32, 32)
    y = torch.randint(0, 10, (1000,))
    
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


if __name__ == "__main__":
    # Example usage
    print("Evolutionary Neural Architecture Search Demo")
    print("=" * 50)
    
    # Create configuration
    config = ArchitectureConfig(
        max_layers=8,
        min_layers=4,
        max_filters=256,
        min_filters=32
    )
    
    # Initialize evolutionary NAS
    nas = EvolutionaryNAS(
        config=config,
        population_size=20,  # Small population for demo
        mutation_rate=0.1,
        crossover_rate=0.8,
        generations=10  # Few generations for demo
    )
    
    # Create dummy data
    train_loader, val_loader = create_dummy_data_loaders()
    
    # Run evolution
    best_architecture = nas.evolve(
        train_loader=train_loader,
        val_loader=val_loader,
        generations=10,
        max_epochs=3,  # Few epochs for demo
        device='cpu',
        verbose=True
    )
    
    print(f"\nBest architecture found:")
    print(f"Fitness: {best_architecture.fitness:.4f}")
    print(f"Number of layers: {len(best_architecture.genes)}")
    print(f"Architecture: {best_architecture.genes}")
    
    # Plot evolution progress
    nas.plot_evolution()
    
    # Print statistics
    stats = nas.get_statistics()
    print(f"\nEvolution Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
