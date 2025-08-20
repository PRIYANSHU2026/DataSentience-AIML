# Federated Learning

## Overview
Federated Learning represents a revolutionary paradigm in machine learning that enables collaborative model training across distributed devices while preserving data privacy. Unlike traditional centralized learning approaches that require data to be collected and stored in a central location, federated learning allows models to be trained on decentralized data sources without ever sharing the raw data. This approach creates a distributed intelligence system where knowledge is learned collectively while maintaining the privacy and security of individual data sources, making it ideal for applications involving sensitive data, edge computing, and privacy-preserving machine learning.

## Core Concepts

### Federated Learning Framework
The core innovation lies in the distributed training paradigm:

```
Federated Learning = Local Training + Secure Communication + Model Aggregation
```

**Key Components:**
- **Global Model**: Shared model maintained by the central server
- **Local Models**: Models trained on individual client devices
- **Aggregation Strategy**: Method for combining local model updates
- **Communication Protocol**: Secure protocol for model exchange
- **Privacy Mechanisms**: Techniques for preserving data privacy

### Federated Averaging (FedAvg)
The foundational algorithm for federated learning:

```
w_{t+1} = Σᵢ (n_i / N) * w_{t+1}^i
```

Where:
- `w_{t+1}` is the global model at round t+1
- `w_{t+1}^i` is the local model of client i
- `n_i` is the number of samples for client i
- `N` is the total number of samples across all clients

### Local Training Process
Each client performs local training on their data:

```
w_{t+1}^i = w_t - η * ∇L_i(w_t)
```

Where:
- `w_t` is the global model received from server
- `η` is the learning rate
- `∇L_i(w_t)` is the gradient of loss on client i's data

### Communication Protocol
```python
class FederatedProtocol:
    def __init__(self, num_clients, aggregation_rounds):
        self.num_clients = num_clients
        self.aggregation_rounds = aggregation_rounds
        self.global_model = None
        self.client_models = {}
    
    def initialize_global_model(self, model):
        """Initialize the global model"""
        self.global_model = copy.deepcopy(model)
    
    def distribute_model(self, client_id):
        """Distribute global model to client"""
        return copy.deepcopy(self.global_model)
    
    def collect_updates(self, client_id, local_model, num_samples):
        """Collect model updates from client"""
        self.client_models[client_id] = {
            'model': copy.deepcopy(local_model),
            'num_samples': num_samples
        }
    
    def aggregate_models(self):
        """Aggregate local models into global model"""
        total_samples = sum(client['num_samples'] for client in self.client_models.values())
        
        # Initialize aggregated model
        aggregated_model = copy.deepcopy(self.global_model)
        
        # Weighted average of model parameters
        for param_name, param in aggregated_model.named_parameters():
            aggregated_param = torch.zeros_like(param)
            
            for client_data in self.client_models.values():
                client_model = client_data['model']
                client_samples = client_data['num_samples']
                weight = client_samples / total_samples
                
                client_param = dict(client_model.named_parameters())[param_name]
                aggregated_param += weight * client_param
            
            param.data = aggregated_param.data
        
        self.global_model = aggregated_model
        return self.global_model
```

## Bizarre and Advanced Aspects

### 1. Distributed Consciousness
Federated learning creates a form of distributed intelligence where knowledge emerges from the collective training of multiple isolated devices, similar to how consciousness might emerge from distributed neural networks.

### 2. Privacy-Preserving Intelligence
The system learns from data it never directly observes, creating a form of "blind learning" where intelligence is developed through indirect observation of model updates.

### 3. Collective Learning
Knowledge is distributed across a network of devices, with each device contributing to the collective intelligence while maintaining its own local knowledge.

### 4. Federated Memory
The global model serves as a form of distributed memory that stores knowledge learned from all participating devices without storing the actual data.

### 5. Edge Intelligence
Learning happens at the boundaries of the network, enabling intelligent decision-making at the edge while maintaining privacy and reducing communication overhead.

### 6. Adaptive Federation
Advanced variants can adapt the federation process based on device capabilities, data distributions, and network conditions.

## Technical Architecture

### Federated Server
```python
import torch
import torch.nn as nn
import copy
import numpy as np
from typing import List, Dict, Any

class FederatedServer:
    def __init__(self, global_model, aggregator, num_rounds=100, 
                 client_fraction=1.0, min_clients=1):
        self.global_model = global_model
        self.aggregator = aggregator
        self.num_rounds = num_rounds
        self.client_fraction = client_fraction
        self.min_clients = min_clients
        self.round_history = []
    
    def train(self, clients: List['FederatedClient']):
        """Run federated learning training"""
        print(f"Starting federated learning with {len(clients)} clients")
        
        for round_idx in range(self.num_rounds):
            print(f"Round {round_idx + 1}/{self.num_rounds}")
            
            # Select clients for this round
            selected_clients = self._select_clients(clients)
            
            # Distribute global model to selected clients
            self._distribute_model(selected_clients)
            
            # Collect updates from clients
            client_updates = self._collect_updates(selected_clients)
            
            # Aggregate updates
            self._aggregate_updates(client_updates)
            
            # Evaluate global model
            if round_idx % 10 == 0:
                self._evaluate_global_model(clients)
            
            # Record round information
            self.round_history.append({
                'round': round_idx,
                'num_clients': len(selected_clients),
                'global_loss': self._compute_global_loss(clients)
            })
        
        return self.global_model
    
    def _select_clients(self, clients):
        """Select clients for the current round"""
        num_selected = max(self.min_clients, 
                          int(len(clients) * self.client_fraction))
        selected_indices = np.random.choice(
            len(clients), num_selected, replace=False
        )
        return [clients[i] for i in selected_indices]
    
    def _distribute_model(self, selected_clients):
        """Distribute global model to selected clients"""
        for client in selected_clients:
            client.receive_global_model(copy.deepcopy(self.global_model))
    
    def _collect_updates(self, selected_clients):
        """Collect model updates from clients"""
        updates = []
        for client in selected_clients:
            update = client.train_and_send_update()
            updates.append(update)
        return updates
    
    def _aggregate_updates(self, client_updates):
        """Aggregate client updates using the aggregator"""
        self.global_model = self.aggregator.aggregate(
            self.global_model, client_updates
        )
    
    def _evaluate_global_model(self, clients):
        """Evaluate global model on a subset of clients"""
        total_loss = 0
        total_samples = 0
        
        for client in clients[:5]:  # Evaluate on first 5 clients
            loss, num_samples = client.evaluate_model(self.global_model)
            total_loss += loss * num_samples
            total_samples += num_samples
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        print(f"Global model loss: {avg_loss:.4f}")
    
    def _compute_global_loss(self, clients):
        """Compute average loss across all clients"""
        total_loss = 0
        total_samples = 0
        
        for client in clients:
            loss, num_samples = client.evaluate_model(self.global_model)
            total_loss += loss * num_samples
            total_samples += num_samples
        
        return total_loss / total_samples if total_samples > 0 else 0

class FederatedClient:
    def __init__(self, client_id, local_data, model, local_epochs=5, 
                 learning_rate=0.01, batch_size=32):
        self.client_id = client_id
        self.local_data = local_data
        self.local_model = copy.deepcopy(model)
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.local_model.parameters(), lr=learning_rate
        )
    
    def receive_global_model(self, global_model):
        """Receive global model from server"""
        self.local_model.load_state_dict(global_model.state_dict())
    
    def train_and_send_update(self):
        """Train local model and send update to server"""
        # Train local model
        self._train_local_model()
        
        # Create update
        update = {
            'client_id': self.client_id,
            'model_state': copy.deepcopy(self.local_model.state_dict()),
            'num_samples': len(self.local_data),
            'local_loss': self._compute_local_loss()
        }
        
        return update
    
    def _train_local_model(self):
        """Train local model for specified number of epochs"""
        self.local_model.train()
        
        for epoch in range(self.local_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx in range(0, len(self.local_data), self.batch_size):
                batch_data = self.local_data[batch_idx:batch_idx + self.batch_size]
                
                # Prepare batch
                inputs, targets = self._prepare_batch(batch_data)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.local_model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Client {self.client_id}, Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
    
    def _prepare_batch(self, batch_data):
        """Prepare batch data for training"""
        # This is a simplified implementation
        # In practice, you'd have proper data loading and preprocessing
        inputs = torch.randn(len(batch_data), 784)  # Example for MNIST
        targets = torch.randint(0, 10, (len(batch_data),))
        return inputs, targets
    
    def _compute_local_loss(self):
        """Compute loss on local data"""
        self.local_model.eval()
        total_loss = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch_idx in range(0, len(self.local_data), self.batch_size):
                batch_data = self.local_data[batch_idx:batch_idx + self.batch_size]
                inputs, targets = self._prepare_batch(batch_data)
                
                outputs = self.local_model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item() * len(batch_data)
                num_samples += len(batch_data)
        
        return total_loss / num_samples if num_samples > 0 else 0
    
    def evaluate_model(self, model):
        """Evaluate a model on local data"""
        model.eval()
        total_loss = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch_idx in range(0, len(self.local_data), self.batch_size):
                batch_data = self.local_data[batch_idx:batch_idx + self.batch_size]
                inputs, targets = self._prepare_batch(batch_data)
                
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item() * len(batch_data)
                num_samples += len(batch_data)
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0
        return avg_loss, num_samples
```

### Model Aggregators
```python
class ModelAggregator:
    def __init__(self):
        pass
    
    def aggregate(self, global_model, client_updates):
        """Aggregate client updates into global model"""
        raise NotImplementedError

class FedAvgAggregator(ModelAggregator):
    def __init__(self):
        super().__init__()
    
    def aggregate(self, global_model, client_updates):
        """Federated Averaging aggregation"""
        # Calculate total samples
        total_samples = sum(update['num_samples'] for update in client_updates)
        
        # Initialize aggregated state dict
        aggregated_state = {}
        
        # Get parameter names from global model
        param_names = global_model.state_dict().keys()
        
        for param_name in param_names:
            aggregated_param = torch.zeros_like(global_model.state_dict()[param_name])
            
            for update in client_updates:
                client_state = update['model_state']
                client_samples = update['num_samples']
                weight = client_samples / total_samples
                
                aggregated_param += weight * client_state[param_name]
            
            aggregated_state[param_name] = aggregated_param
        
        # Update global model
        global_model.load_state_dict(aggregated_state)
        return global_model

class FedProxAggregator(ModelAggregator):
    def __init__(self, mu=0.01):
        super().__init__()
        self.mu = mu  # Proximal term coefficient
    
    def aggregate(self, global_model, client_updates):
        """FedProx aggregation with proximal term"""
        # Similar to FedAvg but with proximal regularization
        total_samples = sum(update['num_samples'] for update in client_updates)
        
        aggregated_state = {}
        param_names = global_model.state_dict().keys()
        
        for param_name in param_names:
            aggregated_param = torch.zeros_like(global_model.state_dict()[param_name])
            
            for update in client_updates:
                client_state = update['model_state']
                client_samples = update['num_samples']
                weight = client_samples / total_samples
                
                # Add proximal term
                global_param = global_model.state_dict()[param_name]
                proximal_term = self.mu * (client_state[param_name] - global_param)
                
                aggregated_param += weight * (client_state[param_name] - proximal_term)
            
            aggregated_state[param_name] = aggregated_param
        
        global_model.load_state_dict(aggregated_state)
        return global_model

class SecureAggregator(ModelAggregator):
    def __init__(self, noise_scale=1.0):
        super().__init__()
        self.noise_scale = noise_scale
    
    def aggregate(self, global_model, client_updates):
        """Secure aggregation with differential privacy"""
        # Add noise for differential privacy
        total_samples = sum(update['num_samples'] for update in client_updates)
        
        aggregated_state = {}
        param_names = global_model.state_dict().keys()
        
        for param_name in param_names:
            aggregated_param = torch.zeros_like(global_model.state_dict()[param_name])
            
            for update in client_updates:
                client_state = update['model_state']
                client_samples = update['num_samples']
                weight = client_samples / total_samples
                
                # Add noise for privacy
                noise = torch.randn_like(client_state[param_name]) * self.noise_scale
                noisy_param = client_state[param_name] + noise
                
                aggregated_param += weight * noisy_param
            
            aggregated_state[param_name] = aggregated_param
        
        global_model.load_state_dict(aggregated_state)
        return global_model
```

## Implementation Details

### Secure Aggregation
```python
import cryptography
from cryptography.fernet import Fernet
import hashlib

class SecureFederatedLearning:
    def __init__(self, encryption_key=None):
        if encryption_key is None:
            self.encryption_key = Fernet.generate_key()
        else:
            self.encryption_key = encryption_key
        
        self.cipher = Fernet(self.encryption_key)
    
    def encrypt_model_update(self, model_update):
        """Encrypt model update for secure transmission"""
        # Serialize model state
        serialized_update = self._serialize_model_update(model_update)
        
        # Encrypt serialized update
        encrypted_update = self.cipher.encrypt(serialized_update)
        
        return encrypted_update
    
    def decrypt_model_update(self, encrypted_update):
        """Decrypt model update"""
        # Decrypt update
        decrypted_update = self.cipher.decrypt(encrypted_update)
        
        # Deserialize model update
        model_update = self._deserialize_model_update(decrypted_update)
        
        return model_update
    
    def _serialize_model_update(self, model_update):
        """Serialize model update to bytes"""
        # This is a simplified implementation
        # In practice, you'd use proper serialization
        import pickle
        return pickle.dumps(model_update)
    
    def _deserialize_model_update(self, serialized_update):
        """Deserialize model update from bytes"""
        import pickle
        return pickle.loads(serialized_update)

class DifferentialPrivacy:
    def __init__(self, epsilon=1.0, delta=1e-5):
        self.epsilon = epsilon
        self.delta = delta
    
    def add_noise_to_gradients(self, gradients, sensitivity):
        """Add noise to gradients for differential privacy"""
        # Calculate noise scale
        noise_scale = self._calculate_noise_scale(sensitivity)
        
        # Add Gaussian noise
        noise = torch.randn_like(gradients) * noise_scale
        noisy_gradients = gradients + noise
        
        return noisy_gradients
    
    def _calculate_noise_scale(self, sensitivity):
        """Calculate noise scale for differential privacy"""
        # Gaussian mechanism noise scale
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        return noise_scale
```

### Heterogeneous Federated Learning
```python
class HeterogeneousFederatedLearning:
    def __init__(self):
        self.client_models = {}
        self.knowledge_distillation = KnowledgeDistillation()
    
    def handle_heterogeneous_clients(self, clients):
        """Handle clients with different model architectures"""
        for client in clients:
            if client.model_architecture != self.global_model_architecture:
                # Use knowledge distillation
                distilled_model = self.knowledge_distillation.distill(
                    teacher_model=self.global_model,
                    student_model=client.model,
                    client_data=client.data
                )
                client.model = distilled_model
    
    def aggregate_heterogeneous_models(self, client_updates):
        """Aggregate models with different architectures"""
        # Use knowledge distillation for aggregation
        aggregated_model = self.knowledge_distillation.aggregate_models(
            client_updates, self.global_model
        )
        return aggregated_model

class KnowledgeDistillation:
    def __init__(self, temperature=3.0, alpha=0.7):
        self.temperature = temperature
        self.alpha = alpha
    
    def distill(self, teacher_model, student_model, client_data):
        """Distill knowledge from teacher to student model"""
        teacher_model.eval()
        student_model.train()
        
        optimizer = torch.optim.Adam(student_model.parameters())
        criterion = nn.KLDivLoss(reduction='batchmean')
        
        for epoch in range(10):
            for batch in client_data:
                inputs, targets = batch
                
                # Teacher predictions
                with torch.no_grad():
                    teacher_logits = teacher_model(inputs)
                    teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
                
                # Student predictions
                student_logits = student_model(inputs)
                student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
                
                # Distillation loss
                distillation_loss = criterion(student_probs, teacher_probs)
                
                # Task loss
                task_loss = F.cross_entropy(student_logits, targets)
                
                # Combined loss
                total_loss = self.alpha * distillation_loss + (1 - self.alpha) * task_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
        
        return student_model
    
    def aggregate_models(self, client_models, global_model):
        """Aggregate models using knowledge distillation"""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated aggregation
        aggregated_model = copy.deepcopy(global_model)
        
        # Average parameters (simplified)
        for param_name, param in aggregated_model.named_parameters():
            aggregated_param = torch.zeros_like(param)
            
            for client_model in client_models:
                client_param = dict(client_model.named_parameters())[param_name]
                aggregated_param += client_param
            
            aggregated_param /= len(client_models)
            param.data = aggregated_param.data
        
        return aggregated_model
```

## Advanced Variants

### 1. Federated Meta-Learning
```python
class FederatedMetaLearning:
    def __init__(self, meta_learning_rate=0.01):
        self.meta_learning_rate = meta_learning_rate
    
    def meta_train(self, clients, num_meta_rounds=10):
        """Meta-train federated learning"""
        for meta_round in range(num_meta_rounds):
            # Meta-training on a subset of clients
            meta_train_clients = clients[:len(clients)//2]
            meta_test_clients = clients[len(clients)//2:]
            
            # Meta-training
            meta_loss = self._meta_training_step(meta_train_clients)
            
            # Meta-testing
            meta_test_loss = self._meta_testing_step(meta_test_clients)
            
            print(f"Meta-round {meta_round}: Train loss: {meta_loss:.4f}, Test loss: {meta_test_loss:.4f}")
    
    def _meta_training_step(self, clients):
        """Single meta-training step"""
        # This is a simplified implementation
        total_loss = 0
        for client in clients:
            loss = client.train_meta_step()
            total_loss += loss
        return total_loss / len(clients)
    
    def _meta_testing_step(self, clients):
        """Single meta-testing step"""
        total_loss = 0
        for client in clients:
            loss = client.test_meta_step()
            total_loss += loss
        return total_loss / len(clients)
```

### 2. Federated Transfer Learning
```python
class FederatedTransferLearning:
    def __init__(self):
        self.source_domain_clients = []
        self.target_domain_clients = []
    
    def transfer_learning(self, source_clients, target_clients):
        """Perform federated transfer learning"""
        # Train on source domain
        source_model = self._train_source_domain(source_clients)
        
        # Adapt to target domain
        target_model = self._adapt_to_target_domain(source_model, target_clients)
        
        return target_model
    
    def _train_source_domain(self, source_clients):
        """Train model on source domain"""
        # Standard federated learning on source domain
        federated_server = FederatedServer(
            global_model=source_clients[0].model,
            aggregator=FedAvgAggregator()
        )
        return federated_server.train(source_clients)
    
    def _adapt_to_target_domain(self, source_model, target_clients):
        """Adapt source model to target domain"""
        # Fine-tune source model on target domain
        for client in target_clients:
            client.model.load_state_dict(source_model.state_dict())
        
        federated_server = FederatedServer(
            global_model=source_model,
            aggregator=FedAvgAggregator()
        )
        return federated_server.train(target_clients)
```

### 3. Federated Reinforcement Learning
```python
class FederatedReinforcementLearning:
    def __init__(self):
        self.global_policy = None
        self.client_policies = {}
    
    def federated_rl_training(self, rl_clients, num_rounds=100):
        """Train reinforcement learning agents in federated setting"""
        for round_idx in range(num_rounds):
            # Distribute global policy
            self._distribute_policy(rl_clients)
            
            # Local RL training
            client_updates = []
            for client in rl_clients:
                update = client.train_local_policy()
                client_updates.append(update)
            
            # Aggregate policies
            self._aggregate_policies(client_updates)
            
            print(f"Federated RL Round {round_idx + 1} completed")
    
    def _distribute_policy(self, rl_clients):
        """Distribute global policy to clients"""
        for client in rl_clients:
            client.receive_global_policy(self.global_policy)
    
    def _aggregate_policies(self, client_updates):
        """Aggregate client policy updates"""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated aggregation for RL policies
        aggregated_policy = copy.deepcopy(self.global_policy)
        
        for param_name, param in aggregated_policy.named_parameters():
            aggregated_param = torch.zeros_like(param)
            
            for update in client_updates:
                client_param = update['policy_state'][param_name]
                aggregated_param += client_param
            
            aggregated_param /= len(client_updates)
            param.data = aggregated_param.data
        
        self.global_policy = aggregated_policy
```

## Performance Metrics

### 1. Training Metrics
- **Convergence rate**: Speed of model convergence across rounds
- **Communication efficiency**: Number of rounds needed for convergence
- **Local training efficiency**: Time and resources for local training
- **Aggregation quality**: Quality of model aggregation

### 2. Privacy Metrics
- **Differential privacy**: Privacy guarantees provided
- **Information leakage**: Amount of information leaked during training
- **Privacy budget**: Privacy budget consumption
- **Attack resistance**: Resistance to privacy attacks

### 3. System Metrics
- **Scalability**: Performance with increasing number of clients
- **Communication overhead**: Bandwidth and latency requirements
- **Resource utilization**: CPU, memory, and energy usage
- **Fault tolerance**: Resilience to client failures

### 4. Model Quality Metrics
- **Global model accuracy**: Accuracy of the aggregated model
- **Personalization**: Quality of personalized models
- **Fairness**: Fairness across different client groups
- **Robustness**: Robustness to adversarial attacks

## Applications

### 1. Mobile Computing
- **On-device learning**: Training models on mobile devices
- **Personalized recommendations**: Personalized content recommendations
- **Mobile health**: Health monitoring and prediction
- **Mobile security**: Fraud detection and security

### 2. Healthcare
- **Medical imaging**: Collaborative medical image analysis
- **Drug discovery**: Collaborative drug discovery
- **Patient monitoring**: Privacy-preserving patient monitoring
- **Clinical trials**: Distributed clinical trial analysis

### 3. Finance
- **Fraud detection**: Collaborative fraud detection across banks
- **Credit scoring**: Privacy-preserving credit scoring
- **Risk assessment**: Distributed risk assessment
- **Trading algorithms**: Collaborative trading strategies

### 4. Internet of Things (IoT)
- **Edge intelligence**: Intelligence at the edge
- **Sensor networks**: Collaborative sensor data analysis
- **Smart homes**: Privacy-preserving smart home systems
- **Industrial IoT**: Collaborative industrial monitoring

### 5. Privacy-Sensitive Applications
- **Social media**: Privacy-preserving social media analysis
- **E-commerce**: Collaborative recommendation systems
- **Education**: Privacy-preserving educational analytics
- **Government**: Secure government data analysis

## Research Frontiers

### 1. Advanced Privacy Techniques
- **Homomorphic encryption**: Fully homomorphic encryption for federated learning
- **Secure multi-party computation**: MPC-based federated learning
- **Zero-knowledge proofs**: Privacy-preserving verification
- **Blockchain integration**: Decentralized federated learning

### 2. Heterogeneous Federated Learning
- **Model heterogeneity**: Handling different model architectures
- **Data heterogeneity**: Handling different data distributions
- **System heterogeneity**: Handling different device capabilities
- **Temporal heterogeneity**: Handling time-varying data

### 3. Communication Efficiency
- **Compression techniques**: Model compression for efficient communication
- **Quantization**: Quantized federated learning
- **Sparsification**: Sparse federated learning
- **Adaptive communication**: Adaptive communication strategies

### 4. Robustness and Security
- **Byzantine robustness**: Robustness to malicious clients
- **Adversarial attacks**: Defense against adversarial attacks
- **Poisoning attacks**: Defense against data poisoning
- **Backdoor attacks**: Defense against backdoor attacks

## Usage Examples

### Basic Federated Learning
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize federated learning
global_model = SimpleModel()
aggregator = FedAvgAggregator()

# Create federated server
server = FederatedServer(
    global_model=global_model,
    aggregator=aggregator,
    num_rounds=50,
    client_fraction=0.5
)

# Create federated clients
clients = []
for i in range(10):
    # Simulate local data for each client
    local_data = torch.randn(100, 784)  # 100 samples per client
    client = FederatedClient(
        client_id=i,
        local_data=local_data,
        model=SimpleModel(),
        local_epochs=3
    )
    clients.append(client)

# Run federated learning
trained_model = server.train(clients)
print("Federated learning completed!")
```

### Secure Federated Learning
```python
# Initialize secure federated learning
secure_fl = SecureFederatedLearning()

# Create secure aggregator
secure_aggregator = SecureAggregator(noise_scale=0.1)

# Create federated server with secure aggregation
server = FederatedServer(
    global_model=SimpleModel(),
    aggregator=secure_aggregator,
    num_rounds=30
)

# Run secure federated learning
secure_model = server.train(clients)
print("Secure federated learning completed!")
```

### Heterogeneous Federated Learning
```python
# Initialize heterogeneous federated learning
heterogeneous_fl = HeterogeneousFederatedLearning()

# Create clients with different model architectures
heterogeneous_clients = []
for i in range(5):
    # Different model architectures for different clients
    if i % 2 == 0:
        model = SimpleModel()
    else:
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    client = FederatedClient(
        client_id=i,
        local_data=torch.randn(100, 784),
        model=model
    )
    heterogeneous_clients.append(client)

# Handle heterogeneous clients
heterogeneous_fl.handle_heterogeneous_clients(heterogeneous_clients)

# Run heterogeneous federated learning
heterogeneous_model = server.train(heterogeneous_clients)
print("Heterogeneous federated learning completed!")
```

### Federated Meta-Learning
```python
# Initialize federated meta-learning
federated_meta = FederatedMetaLearning(meta_learning_rate=0.01)

# Run federated meta-learning
federated_meta.meta_train(clients, num_meta_rounds=20)
print("Federated meta-learning completed!")
```

### Federated Transfer Learning
```python
# Initialize federated transfer learning
federated_transfer = FederatedTransferLearning()

# Create source and target domain clients
source_clients = clients[:5]  # First 5 clients as source domain
target_clients = clients[5:]  # Last 5 clients as target domain

# Perform federated transfer learning
transferred_model = federated_transfer.transfer_learning(
    source_clients, target_clients
)
print("Federated transfer learning completed!")
```

### Performance Analysis
```python
import matplotlib.pyplot as plt

def analyze_federated_performance(server):
    """Analyze federated learning performance"""
    rounds = [h['round'] for h in server.round_history]
    losses = [h['global_loss'] for h in server.round_history]
    num_clients = [h['num_clients'] for h in server.round_history]
    
    # Plot training progress
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(rounds, losses)
    plt.title('Global Model Loss')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(rounds, num_clients)
    plt.title('Number of Participating Clients')
    plt.xlabel('Round')
    plt.ylabel('Number of Clients')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.hist(num_clients, bins=10)
    plt.title('Distribution of Participating Clients')
    plt.xlabel('Number of Clients')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Analyze performance
analyze_federated_performance(server)
```

## Files in this Directory
- `federated_learning.py`: Core federated learning implementation
- `secure_aggregation.py`: Privacy-preserving aggregation methods
- `federated_optimization.py`: Advanced federated optimization
- `example_usage.py`: Working examples
