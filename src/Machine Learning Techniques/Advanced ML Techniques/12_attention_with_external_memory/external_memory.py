"""
Attention with External Memory

This module implements attention mechanisms with external memory banks,
including memory controllers, attention variants, and complete memory-augmented networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import math


class MemoryBank:
    """External memory bank for storing and retrieving information"""
    
    def __init__(self, memory_size: int, memory_dim: int, device: str = 'cpu'):
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.device = device
        
        # Initialize memory matrix
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim) * 0.1)
        self.usage_vector = torch.zeros(memory_size, device=device)
        self.link_matrix = torch.zeros(memory_size, memory_size, device=device)
        self.precedence_vector = torch.zeros(memory_size, device=device)
        
        # Memory access history
        self.read_history = []
        self.write_history = []
    
    def read(self, read_weights: torch.Tensor) -> torch.Tensor:
        """Read from memory using attention weights"""
        # read_weights: (batch_size, num_read_heads, memory_size)
        batch_size, num_read_heads, _ = read_weights.shape
        
        # Expand memory for batch processing
        if self.memory.dim() == 2:
            memory_expanded = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        elif self.memory.dim() == 3:
            memory_expanded = self.memory
        else:
            # Create a default memory if dimensions are wrong
            memory_expanded = torch.randn(batch_size, self.memory_size, self.memory_dim)
        
        # Read operation: weighted sum of memory locations
        read_vectors = torch.bmm(read_weights, memory_expanded)
        # read_vectors: (batch_size, num_read_heads, memory_dim)
        
        # Record read history
        self.read_history.append(read_weights.detach().cpu())
        
        return read_vectors
    
    def write(self, write_weights: torch.Tensor, write_vectors: torch.Tensor, 
              erase_vectors: torch.Tensor) -> None:
        """Write to memory using attention weights"""
        # write_weights: (batch_size, num_write_heads, memory_size)
        # write_vectors: (batch_size, num_write_heads, memory_dim)
        # erase_vectors: (batch_size, num_write_heads, memory_dim)
        
        batch_size, num_write_heads, memory_size = write_weights.shape
        
        # Update memory matrix
        for b in range(batch_size):
            for h in range(num_write_heads):
                # Erase operation
                erase_weights = write_weights[b, h].unsqueeze(1)  # (memory_size, 1)
                erase_operation = erase_weights * erase_vectors[b, h].unsqueeze(0)  # (memory_size, memory_dim)
                self.memory.data = self.memory.data * (1 - erase_operation)
                
                # Add operation
                add_weights = write_weights[b, h].unsqueeze(1)  # (memory_size, 1)
                add_operation = add_weights * write_vectors[b, h].unsqueeze(0)  # (memory_size, memory_dim)
                self.memory.data = self.memory.data + add_operation
        
        # Update usage vector
        usage_update = torch.max(write_weights, dim=1)[0]  # (batch_size, memory_size)
        self.usage_vector = torch.max(self.usage_vector, usage_update.mean(dim=0))
        
        # Record write history
        self.write_history.append(write_weights.detach().cpu())
    
    def get_memory_state(self) -> Dict:
        """Get current memory state"""
        return {
            'memory': self.memory.data.clone(),
            'usage': self.usage_vector.clone(),
            'link_matrix': self.link_matrix.clone(),
            'precedence': self.precedence_vector.clone()
        }
    
    def reset_memory(self):
        """Reset memory to initial state"""
        self.memory.data = torch.randn(self.memory_size, self.memory_dim, device=self.device) * 0.1
        self.usage_vector = torch.zeros(self.memory_size, device=self.device)
        self.link_matrix = torch.zeros(self.memory_size, self.memory_size, device=self.device)
        self.precedence_vector = torch.zeros(self.memory_size, device=self.device)
        self.read_history = []
        self.write_history = []


class ContentBasedAttention(nn.Module):
    """Content-based attention mechanism for memory access"""
    
    def __init__(self, key_dim: int, memory_dim: int, temperature: float = 1.0):
        super().__init__()
        self.key_dim = key_dim
        self.memory_dim = memory_dim
        self.temperature = temperature
        
        # Key transformation
        self.key_transform = nn.Linear(key_dim, memory_dim)
        
    def forward(self, query: torch.Tensor, memory: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute content-based attention weights
        
        Args:
            query: (batch_size, key_dim) - Query vector
            memory: (memory_size, memory_dim) - Memory matrix
            mask: (memory_size,) - Optional mask for memory locations
        
        Returns:
            attention_weights: (batch_size, memory_size) - Attention weights
        """
        batch_size = query.shape[0]
        memory_size = memory.shape[0]
        
        # Transform query to memory space
        query_transformed = self.key_transform(query)  # (batch_size, memory_dim)
        
        # Compute similarity between query and memory
        # Expand for broadcasting
        query_expanded = query_transformed.unsqueeze(1)  # (batch_size, 1, memory_dim)
        memory_expanded = memory.unsqueeze(0)  # (1, memory_size, memory_dim)
        
        # Cosine similarity
        similarity = F.cosine_similarity(query_expanded, memory_expanded, dim=2)  # (batch_size, memory_size)
        
        # Apply temperature scaling
        attention_weights = similarity / self.temperature
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(0).expand(batch_size, -1)
            attention_weights = attention_weights.masked_fill(mask_expanded == 0, -float('inf'))
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_weights, dim=1)
        
        return attention_weights


class LocationBasedAttention(nn.Module):
    """Location-based attention mechanism for memory access"""
    
    def __init__(self, memory_size: int, num_heads: int = 1):
        super().__init__()
        self.memory_size = memory_size
        self.num_heads = num_heads
        
        # Location-based parameters
        self.interpolation_gate = nn.Linear(1, 1)
        self.shift_weights = nn.Linear(1, 3)  # -1, 0, 1 shifts
        self.sharpen_weights = nn.Linear(1, 1)
        
    def forward(self, previous_weights: torch.Tensor, 
                interpolation_gate: torch.Tensor,
                shift_weights: torch.Tensor,
                sharpen_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute location-based attention weights
        
        Args:
            previous_weights: (batch_size, num_heads, memory_size) - Previous attention weights
            interpolation_gate: (batch_size, num_heads, 1) - Interpolation gate
            shift_weights: (batch_size, num_heads, 3) - Shift weights for -1, 0, 1
            sharpen_weights: (batch_size, num_heads, 1) - Sharpening weights
        
        Returns:
            attention_weights: (batch_size, num_heads, memory_size) - New attention weights
        """
        batch_size, num_heads, memory_size = previous_weights.shape
        
        # Circular convolution for shifting
        shifted_weights = []
        for shift in [-1, 0, 1]:
            shifted = torch.roll(previous_weights, shifts=shift, dims=2)
            shifted_weights.append(shifted)
        
        shifted_weights = torch.stack(shifted_weights, dim=3)  # (batch_size, num_heads, memory_size, 3)
        
        # Apply shift weights
        shift_weights_expanded = shift_weights.unsqueeze(2)  # (batch_size, num_heads, 1, 3)
        convolved_weights = torch.sum(shifted_weights * shift_weights_expanded, dim=3)
        
        # Interpolation between previous and convolved weights
        interpolation_gate_expanded = interpolation_gate.expand(-1, -1, memory_size)
        interpolated_weights = (interpolation_gate_expanded * convolved_weights + 
                              (1 - interpolation_gate_expanded) * previous_weights)
        
        # Sharpening
        sharpen_weights_expanded = sharpen_weights.expand(-1, -1, memory_size)
        sharpened_weights = interpolated_weights ** sharpen_weights_expanded
        
        # Normalize
        attention_weights = sharpened_weights / (torch.sum(sharpened_weights, dim=2, keepdim=True) + 1e-8)
        
        return attention_weights


class MemoryController(nn.Module):
    """Neural controller for memory operations"""
    
    def __init__(self, input_size: int, memory_dim: int, controller_size: int, 
                 num_read_heads: int, num_write_heads: int):
        super().__init__()
        self.input_size = input_size
        self.memory_dim = memory_dim
        self.controller_size = controller_size
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        
        # Controller network (LSTM)
        self.controller = nn.LSTM(
            input_size=input_size + num_read_heads * memory_dim,
            hidden_size=controller_size,
            num_layers=1,
            batch_first=True
        )
        
        # Output projections
        self.read_keys = nn.Linear(controller_size, num_read_heads * memory_dim)
        self.read_strengths = nn.Linear(controller_size, num_read_heads)
        self.read_modes = nn.Linear(controller_size, num_read_heads * 3)  # 3 read modes
        
        self.write_keys = nn.Linear(controller_size, num_write_heads * memory_dim)
        self.write_strengths = nn.Linear(controller_size, num_write_heads)
        self.write_vectors = nn.Linear(controller_size, num_write_heads * memory_dim)
        self.erase_vectors = nn.Linear(controller_size, num_write_heads * memory_dim)
        
        # Location-based attention parameters
        self.interpolation_gates = nn.Linear(controller_size, num_read_heads + num_write_heads)
        self.shift_weights = nn.Linear(controller_size, (num_read_heads + num_write_heads) * 3)
        self.sharpen_weights = nn.Linear(controller_size, num_read_heads + num_write_heads)
        
        # Attention mechanisms
        self.content_attention = ContentBasedAttention(controller_size, memory_dim)
        self.location_attention = LocationBasedAttention(128)  # Assuming memory_size=128
        
        # Previous attention weights
        self.prev_read_weights = None
        self.prev_write_weights = None
    
    def forward(self, inputs: torch.Tensor, memory_bank: MemoryBank) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through memory controller
        
        Args:
            inputs: (batch_size, seq_len, input_size) - Input sequence
            memory_bank: MemoryBank instance
        
        Returns:
            outputs: (batch_size, seq_len, controller_size) - Controller outputs
            attention_info: Dict with attention weights and memory operations
        """
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device
        
        # Initialize hidden state
        h0 = torch.zeros(1, batch_size, self.controller_size, device=device)
        c0 = torch.zeros(1, batch_size, self.controller_size, device=device)
        
        # Initialize previous attention weights
        if self.prev_read_weights is None:
            self.prev_read_weights = torch.zeros(batch_size, self.num_read_heads, 
                                               memory_bank.memory_size, device=device)
            self.prev_write_weights = torch.zeros(batch_size, self.num_write_heads, 
                                                memory_bank.memory_size, device=device)
        
        outputs = []
        attention_info = {
            'read_weights': [],
            'write_weights': [],
            'read_vectors': [],
            'write_vectors': []
        }
        
        for t in range(seq_len):
            # Read from memory
            read_vectors = memory_bank.read(self.prev_read_weights)
            read_vectors_flat = read_vectors.view(batch_size, -1)
            
            # Prepare controller input
            controller_input = torch.cat([inputs[:, t], read_vectors_flat], dim=1)
            controller_input = controller_input.unsqueeze(1)  # Add sequence dimension
            
            # Controller forward pass
            controller_output, (h0, c0) = self.controller(controller_input, (h0, c0))
            controller_output = controller_output.squeeze(1)  # Remove sequence dimension
            
            # Generate read parameters
            read_keys = self.read_keys(controller_output).view(batch_size, self.num_read_heads, self.memory_dim)
            read_strengths = self.read_strengths(controller_output)
            read_modes = self.read_modes(controller_output).view(batch_size, self.num_read_heads, 3)
            
            # Generate write parameters
            write_keys = self.write_keys(controller_output).view(batch_size, self.num_write_heads, self.memory_dim)
            write_strengths = self.write_strengths(controller_output)
            write_vectors = self.write_vectors(controller_output).view(batch_size, self.num_write_heads, self.memory_dim)
            erase_vectors = self.erase_vectors(controller_output).view(batch_size, self.num_write_heads, self.memory_dim)
            
            # Generate location-based attention parameters
            interpolation_gates = torch.sigmoid(self.interpolation_gates(controller_output))
            shift_weights = F.softmax(self.shift_weights(controller_output).view(batch_size, -1, 3), dim=2)
            sharpen_weights = 1 + torch.tanh(self.sharpen_weights(controller_output))
            
            # Compute read attention weights
            read_weights = []
            for h in range(self.num_read_heads):
                # Content-based attention
                content_weights = self.content_attention(
                    controller_output, memory_bank.memory.data
                )
                
                # Location-based attention
                location_weights = self.location_attention(
                    self.prev_read_weights[:, h:h+1],
                    interpolation_gates[:, h:h+1],
                    shift_weights[:, h:h+1],
                    sharpen_weights[:, h:h+1]
                ).squeeze(1)
                
                # Combine content and location attention
                combined_weights = (read_modes[:, h, 0:1] * content_weights + 
                                  read_modes[:, h, 1:2] * location_weights + 
                                  read_modes[:, h, 2:3] * self.prev_read_weights[:, h])
                
                read_weights.append(combined_weights)
            
            read_weights = torch.stack(read_weights, dim=1)  # (batch_size, num_read_heads, memory_size)
            
            # Compute write attention weights
            write_weights = []
            for h in range(self.num_write_heads):
                # Content-based attention for writing
                content_weights = self.content_attention(
                    controller_output, memory_bank.memory.data
                )
                
                # Location-based attention for writing
                location_weights = self.location_attention(
                    self.prev_write_weights[:, h:h+1],
                    interpolation_gates[:, self.num_read_heads + h:self.num_read_heads + h + 1],
                    shift_weights[:, self.num_read_heads + h:self.num_read_heads + h + 1],
                    sharpen_weights[:, self.num_read_heads + h:self.num_read_heads + h + 1]
                ).squeeze(1)
                
                # Combine for writing (simplified)
                combined_weights = 0.5 * content_weights + 0.5 * location_weights
                write_weights.append(combined_weights)
            
            write_weights = torch.stack(write_weights, dim=1)  # (batch_size, num_write_heads, memory_size)
            
            # Write to memory
            memory_bank.write(write_weights, write_vectors, erase_vectors)
            
            # Read from memory with new weights
            read_vectors = memory_bank.read(read_weights)
            
            # Update previous weights
            self.prev_read_weights = read_weights
            self.prev_write_weights = write_weights
            
            # Store outputs and attention info
            outputs.append(controller_output)
            attention_info['read_weights'].append(read_weights.detach().cpu())
            attention_info['write_weights'].append(write_weights.detach().cpu())
            attention_info['read_vectors'].append(read_vectors.detach().cpu())
            attention_info['write_vectors'].append(write_vectors.detach().cpu())
        
        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, controller_size)
        
        return outputs, attention_info
    
    def reset_state(self):
        """Reset controller state"""
        self.prev_read_weights = None
        self.prev_write_weights = None


class ExternalMemoryNetwork(nn.Module):
    """Complete external memory network with attention mechanisms"""
    
    def __init__(self, input_size: int, output_size: int, memory_size: int = 128, 
                 memory_dim: int = 32, controller_size: int = 64, num_read_heads: int = 4, 
                 num_write_heads: int = 1, device: str = 'cpu'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.controller_size = controller_size
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        self.device = device
        
        # Memory bank
        self.memory_bank = MemoryBank(memory_size, memory_dim, device)
        
        # Memory controller
        self.controller = MemoryController(
            input_size, memory_dim, controller_size, num_read_heads, num_write_heads
        )
        
        # Output projection
        self.output_projection = nn.Linear(controller_size, output_size)
        
        # Optional: Input embedding
        self.input_embedding = nn.Linear(input_size, input_size)
        
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through external memory network
        
        Args:
            inputs: (batch_size, seq_len, input_size) - Input sequence
        
        Returns:
            outputs: (batch_size, seq_len, output_size) - Network outputs
            memory_info: Dict with memory state and attention information
        """
        batch_size, seq_len, _ = inputs.shape
        
        # Input embedding
        embedded_inputs = self.input_embedding(inputs)
        
        # Process through memory controller
        controller_outputs, attention_info = self.controller(embedded_inputs, self.memory_bank)
        
        # Generate outputs
        outputs = self.output_projection(controller_outputs)
        
        # Get memory state
        memory_state = self.memory_bank.get_memory_state()
        
        # Prepare return information
        memory_info = {
            'memory_state': memory_state,
            'attention_info': attention_info,
            'controller_outputs': controller_outputs
        }
        
        return outputs, memory_info
    
    def reset_memory(self):
        """Reset memory bank and controller state"""
        self.memory_bank.reset_memory()
        self.controller.reset_state()
    
    def get_memory_visualization(self) -> Dict:
        """Get memory visualization data"""
        memory_state = self.memory_bank.get_memory_state()
        
        return {
            'memory_matrix': memory_state['memory'].detach().cpu().numpy(),
            'usage_vector': memory_state['usage'].detach().cpu().numpy(),
            'read_history': [w.numpy() for w in self.memory_bank.read_history[-10:]],  # Last 10 reads
            'write_history': [w.numpy() for w in self.memory_bank.write_history[-10:]]  # Last 10 writes
        }


class HierarchicalMemoryNetwork(nn.Module):
    """Hierarchical memory network with multiple memory levels"""
    
    def __init__(self, input_size: int, output_size: int, num_levels: int = 3,
                 memory_sizes: List[int] = None, memory_dims: List[int] = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_levels = num_levels
        
        if memory_sizes is None:
            memory_sizes = [64, 32, 16]  # Decreasing memory sizes
        if memory_dims is None:
            memory_dims = [16, 32, 64]  # Increasing memory dimensions
        
        # Create hierarchical memory banks
        self.memory_banks = nn.ModuleList([
            MemoryBank(memory_sizes[i], memory_dims[i]) 
            for i in range(num_levels)
        ])
        
        # Create controllers for each level
        self.controllers = nn.ModuleList([
            MemoryController(input_size if i == 0 else memory_dims[i-1], 
                           memory_dims[i], 32, 2, 1)
            for i in range(num_levels)
        ])
        
        # Cross-level attention
        self.cross_level_attention = nn.ModuleList([
            nn.MultiheadAttention(memory_dims[i], num_heads=4, batch_first=True)
            for i in range(num_levels)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(sum(memory_dims), output_size)
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass through hierarchical memory network"""
        batch_size, seq_len, _ = inputs.shape
        
        level_outputs = []
        memory_info = {}
        
        current_input = inputs
        
        for level in range(self.num_levels):
            # Process through current level
            level_output, level_info = self.controllers[level](current_input, self.memory_banks[level])
            
            # Apply cross-level attention if not the first level
            if level > 0:
                level_output, _ = self.cross_level_attention[level](
                    level_output, level_output, level_output
                )
            
            level_outputs.append(level_output)
            memory_info[f'level_{level}'] = level_info
            
            # Use current level output as input for next level
            current_input = level_output
        
        # Combine outputs from all levels
        combined_output = torch.cat(level_outputs, dim=2)
        
        # Generate final output
        final_output = self.output_projection(combined_output)
        
        return final_output, memory_info


if __name__ == "__main__":
    # Example usage
    print("External Memory Network Demo")
    print("=" * 50)
    
    # Create external memory network
    memory_network = ExternalMemoryNetwork(
        input_size=32,
        output_size=10,
        memory_size=64,
        memory_dim=16,
        controller_size=32,
        num_read_heads=2,
        num_write_heads=1
    )
    
    # Create dummy input
    batch_size, seq_len = 4, 8
    inputs = torch.randn(batch_size, seq_len, 32)
    
    # Forward pass
    outputs, memory_info = memory_network(inputs)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Memory state keys: {memory_info['memory_state'].keys()}")
    print(f"Attention info keys: {memory_info['attention_info'].keys()}")
    
    # Test hierarchical memory network
    print("\nTesting Hierarchical Memory Network:")
    hierarchical_network = HierarchicalMemoryNetwork(
        input_size=32,
        output_size=10,
        num_levels=2
    )
    
    h_outputs, h_memory_info = hierarchical_network(inputs)
    print(f"Hierarchical output shape: {h_outputs.shape}")
    print(f"Hierarchical memory info keys: {h_memory_info.keys()}")
