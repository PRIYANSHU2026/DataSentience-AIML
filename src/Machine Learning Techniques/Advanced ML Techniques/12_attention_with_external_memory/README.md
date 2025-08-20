# Attention with External Memory

## Overview
Attention with External Memory represents a revolutionary advancement in neural architectures by combining the power of attention mechanisms with persistent external memory banks. This approach enables neural networks to maintain, access, and manipulate information across extended time horizons, effectively bridging the gap between traditional neural networks and computational systems with explicit memory. By providing persistent storage that can be read from and written to using learned attention mechanisms, these models can handle tasks requiring long-term memory, complex reasoning, and information retrieval that would be impossible for standard recurrent or transformer architectures.

## Core Concepts

### External Memory Architecture
The core innovation lies in the separation of computation and memory:

```
Memory-Augmented Network = Controller + External Memory + Attention Mechanisms
```

**Key Components:**
- **Controller**: Neural network that processes inputs and generates control signals
- **Memory Bank**: External matrix M ∈ ℝ^(N×D) where N is memory size and D is memory dimension
- **Read/Write Heads**: Attention mechanisms that access specific memory locations
- **Memory Operations**: Read, write, erase, and update operations

### Memory Addressing Mechanisms
External memory systems use sophisticated addressing strategies:

**Content-Based Addressing:**
```
w_t = softmax(β_t K(M_t, k_t))
```

Where:
- `w_t` is the read/write weight vector
- `β_t` is the key strength parameter
- `K(·, ·)` is a similarity function (cosine, dot product)
- `k_t` is the query key from the controller

**Location-Based Addressing:**
```
w_t = g_t w_{t-1} + (1 - g_t) w_t^c
```

Where:
- `g_t` is the interpolation gate
- `w_{t-1}` is the previous weight vector
- `w_t^c` is the content-based weight vector

### Memory Operations

**Read Operation:**
```
r_t = ∑ᵢ w_t[i] M_t[i]
```

**Write Operation:**
```
M_t[i] = M_{t-1}[i] ⊙ (1 - w_t[i] e_t) + w_t[i] a_t
```

Where:
- `e_t` is the erase vector
- `a_t` is the add vector
- `⊙` denotes element-wise multiplication

### Attention Mechanisms for Memory

**Multi-Head Memory Attention:**
```python
def memory_attention(query, memory, num_heads=8):
    # Project query and memory to multiple heads
    Q = linear(query, num_heads * head_dim)
    K = linear(memory, num_heads * head_dim)
    V = linear(memory, num_heads * head_dim)
    
    # Reshape for multi-head attention
    Q = Q.view(batch_size, num_heads, head_dim)
    K = K.view(batch_size, memory_size, num_heads, head_dim)
    V = V.view(batch_size, memory_size, num_heads, head_dim)
    
    # Compute attention scores
    scores = torch.matmul(Q.unsqueeze(1), K.transpose(-2, -1)) / sqrt(head_dim)
    attention_weights = softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
```

## Bizarre and Advanced Aspects

### 1. Persistent Memory Across Time
Unlike traditional neural networks where information flows through activations, external memory maintains persistent state that can be accessed and modified across arbitrary time intervals.

### 2. Neural Memory Manipulation
The network can learn to perform sophisticated memory operations like allocation, deallocation, and garbage collection, mimicking human memory management.

### 3. Multi-Scale Memory Access
Advanced variants can access memory at multiple scales simultaneously, from fine-grained local access to coarse-grained global patterns.

### 4. Memory Compression and Decompression
The system can learn to compress information when writing to memory and decompress when reading, enabling efficient storage of complex patterns.

### 5. Memory Evolution and Adaptation
Memory contents can evolve over time, with the system learning to update and reorganize stored information based on new experiences.

### 6. Cross-Task Memory Transfer
Memory can be shared and transferred across different tasks, enabling meta-learning and transfer learning capabilities.

## Technical Architecture

### External Memory Network
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ExternalMemoryNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, memory_dim, 
                 num_read_heads, num_write_heads, controller_type='lstm'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        
        # Controller
        if controller_type == 'lstm':
            self.controller = nn.LSTM(input_size + memory_dim * num_read_heads, 
                                    hidden_size, batch_first=True)
        else:
            self.controller = nn.Linear(input_size + memory_dim * num_read_heads, 
                                      hidden_size)
        
        # Memory bank
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim))
        
        # Read heads
        self.read_heads = nn.ModuleList([
            MemoryReadHead(hidden_size, memory_dim) 
            for _ in range(num_read_heads)
        ])
        
        # Write heads
        self.write_heads = nn.ModuleList([
            MemoryWriteHead(hidden_size, memory_dim) 
            for _ in range(num_write_heads)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, input_size)
    
    def forward(self, x, memory_state=None):
        batch_size, seq_len, _ = x.shape
        
        # Initialize memory state
        if memory_state is None:
            memory_state = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        
        outputs = []
        read_vectors = []
        
        # Process sequence
        for t in range(seq_len):
            # Read from memory
            read_vectors_t = []
            for read_head in self.read_heads:
                read_vector = read_head(memory_state)
                read_vectors_t.append(read_vector)
            
            read_vectors_t = torch.cat(read_vectors_t, dim=-1)
            read_vectors.append(read_vectors_t)
            
            # Controller input
            controller_input = torch.cat([x[:, t, :], read_vectors_t], dim=-1)
            
            # Controller step
            if isinstance(self.controller, nn.LSTM):
                controller_output, _ = self.controller(controller_input.unsqueeze(1))
                controller_output = controller_output.squeeze(1)
            else:
                controller_output = self.controller(controller_input)
            
            # Write to memory
            for write_head in self.write_heads:
                memory_state = write_head(memory_state, controller_output)
            
            # Generate output
            output = self.output_projection(controller_output)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        read_vectors = torch.stack(read_vectors, dim=1)
        
        return outputs, memory_state, read_vectors

class MemoryReadHead(nn.Module):
    def __init__(self, hidden_size, memory_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_dim = memory_dim
        
        # Key generation
        self.key_projection = nn.Linear(hidden_size, memory_dim)
        
        # Key strength
        self.key_strength = nn.Linear(hidden_size, 1)
        
        # Interpolation gate
        self.interpolation_gate = nn.Linear(hidden_size, 1)
        
        # Shift parameters
        self.shift_parameters = nn.Linear(hidden_size, 3)  # Left, center, right
        
        # Sharpening parameter
        self.sharpening_parameter = nn.Linear(hidden_size, 1)
    
    def forward(self, memory, controller_output=None):
        batch_size, memory_size, memory_dim = memory.shape
        
        if controller_output is None:
            # Use zero controller output for initialization
            controller_output = torch.zeros(batch_size, self.hidden_size, 
                                          device=memory.device)
        
        # Generate key
        key = self.key_projection(controller_output)  # (batch_size, memory_dim)
        key = key.unsqueeze(1).expand(-1, memory_size, -1)  # (batch_size, memory_size, memory_dim)
        
        # Compute similarity
        similarity = F.cosine_similarity(key, memory, dim=-1)  # (batch_size, memory_size)
        
        # Apply key strength
        key_strength = torch.sigmoid(self.key_strength(controller_output))  # (batch_size, 1)
        similarity = similarity * key_strength
        
        # Content-based addressing
        content_weights = F.softmax(similarity, dim=-1)  # (batch_size, memory_size)
        
        # Location-based addressing (simplified)
        interpolation_gate = torch.sigmoid(self.interpolation_gate(controller_output))
        
        # For simplicity, we'll use content-based addressing only
        # In a full implementation, you'd maintain previous weights and apply shifts
        
        # Read from memory
        read_vector = torch.sum(content_weights.unsqueeze(-1) * memory, dim=1)
        
        return read_vector

class MemoryWriteHead(nn.Module):
    def __init__(self, hidden_size, memory_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_dim = memory_dim
        
        # Addressing (similar to read head)
        self.key_projection = nn.Linear(hidden_size, memory_dim)
        self.key_strength = nn.Linear(hidden_size, 1)
        self.interpolation_gate = nn.Linear(hidden_size, 1)
        self.shift_parameters = nn.Linear(hidden_size, 3)
        self.sharpening_parameter = nn.Linear(hidden_size, 1)
        
        # Write parameters
        self.erase_vector = nn.Linear(hidden_size, memory_dim)
        self.add_vector = nn.Linear(hidden_size, memory_dim)
    
    def forward(self, memory, controller_output):
        batch_size, memory_size, memory_dim = memory.shape
        
        # Generate addressing weights (similar to read head)
        key = self.key_projection(controller_output)
        key = key.unsqueeze(1).expand(-1, memory_size, -1)
        
        similarity = F.cosine_similarity(key, memory, dim=-1)
        key_strength = torch.sigmoid(self.key_strength(controller_output))
        similarity = similarity * key_strength
        
        write_weights = F.softmax(similarity, dim=-1)
        
        # Generate erase and add vectors
        erase_vector = torch.sigmoid(self.erase_vector(controller_output))  # (batch_size, memory_dim)
        add_vector = torch.tanh(self.add_vector(controller_output))  # (batch_size, memory_dim)
        
        # Expand for broadcasting
        erase_vector = erase_vector.unsqueeze(1).expand(-1, memory_size, -1)
        add_vector = add_vector.unsqueeze(1).expand(-1, memory_size, -1)
        write_weights = write_weights.unsqueeze(-1).expand(-1, -1, memory_dim)
        
        # Write operation
        new_memory = memory * (1 - write_weights * erase_vector) + write_weights * add_vector
        
        return new_memory
```

### Differentiable Neural Computer (DNC)
```python
class DifferentiableNeuralComputer(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, memory_dim, 
                 num_read_heads, num_write_heads):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        
        # Controller
        self.controller = nn.LSTM(input_size + memory_dim * num_read_heads, 
                                hidden_size, batch_first=True)
        
        # Memory bank
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim))
        
        # Usage vector
        self.usage = None
        
        # Precedence weighting
        self.precedence = None
        
        # Link matrix
        self.link = None
        
        # Read heads
        self.read_heads = nn.ModuleList([
            DNCReadHead(hidden_size, memory_size, memory_dim) 
            for _ in range(num_read_heads)
        ])
        
        # Write head
        self.write_head = DNCWriteHead(hidden_size, memory_size, memory_dim)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, input_size)
    
    def forward(self, x, memory_state=None):
        batch_size, seq_len, _ = x.shape
        
        # Initialize memory state
        if memory_state is None:
            memory_state = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
            self.usage = torch.zeros(batch_size, self.memory_size, device=x.device)
            self.precedence = torch.zeros(batch_size, self.memory_size, device=x.device)
            self.link = torch.zeros(batch_size, self.memory_size, self.memory_size, device=x.device)
        
        outputs = []
        read_vectors = []
        
        for t in range(seq_len):
            # Read from memory
            read_vectors_t = []
            for read_head in self.read_heads:
                read_vector = read_head(memory_state, self.usage, self.link)
                read_vectors_t.append(read_vector)
            
            read_vectors_t = torch.cat(read_vectors_t, dim=-1)
            read_vectors.append(read_vectors_t)
            
            # Controller input
            controller_input = torch.cat([x[:, t, :], read_vectors_t], dim=-1)
            
            # Controller step
            controller_output, _ = self.controller(controller_input.unsqueeze(1))
            controller_output = controller_output.squeeze(1)
            
            # Write to memory
            memory_state, self.usage, self.precedence, self.link = \
                self.write_head(memory_state, controller_output, self.usage, 
                              self.precedence, self.link)
            
            # Generate output
            output = self.output_projection(controller_output)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        read_vectors = torch.stack(read_vectors, dim=1)
        
        return outputs, memory_state, read_vectors

class DNCReadHead(nn.Module):
    def __init__(self, hidden_size, memory_size, memory_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        # Content-based addressing
        self.key = nn.Linear(hidden_size, memory_dim)
        self.key_strength = nn.Linear(hidden_size, 1)
        
        # Location-based addressing
        self.interpolation_gate = nn.Linear(hidden_size, 1)
        self.shift_parameters = nn.Linear(hidden_size, 3)
        self.sharpening_parameter = nn.Linear(hidden_size, 1)
        
        # Temporal memory linkage
        self.link_strength = nn.Linear(hidden_size, 1)
    
    def forward(self, memory, usage, link):
        batch_size = memory.shape[0]
        
        # Content-based addressing
        key = self.key(controller_output)
        key_strength = torch.sigmoid(self.key_strength(controller_output))
        
        # Compute similarity
        key_expanded = key.unsqueeze(1).expand(-1, self.memory_size, -1)
        similarity = F.cosine_similarity(key_expanded, memory, dim=-1)
        similarity = similarity * key_strength
        
        content_weights = F.softmax(similarity, dim=-1)
        
        # Location-based addressing (simplified)
        # In full DNC, this would include shifts and sharpening
        
        # Temporal memory linkage
        link_strength = torch.sigmoid(self.link_strength(controller_output))
        
        # Forward and backward reading
        forward_weights = torch.sum(link * content_weights.unsqueeze(-1), dim=1)
        backward_weights = torch.sum(link.transpose(-2, -1) * content_weights.unsqueeze(-1), dim=1)
        
        # Combine reading modes
        read_weights = (1 - link_strength) * content_weights + \
                      link_strength * (forward_weights + backward_weights) / 2
        
        # Read from memory
        read_vector = torch.sum(read_weights.unsqueeze(-1) * memory, dim=1)
        
        return read_vector

class DNCWriteHead(nn.Module):
    def __init__(self, hidden_size, memory_size, memory_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        # Write parameters
        self.erase_vector = nn.Linear(hidden_size, memory_dim)
        self.add_vector = nn.Linear(hidden_size, memory_dim)
        
        # Allocation gate
        self.allocation_gate = nn.Linear(hidden_size, 1)
        
        # Write gate
        self.write_gate = nn.Linear(hidden_size, 1)
        
        # Free gates
        self.free_gates = nn.Linear(hidden_size, self.num_read_heads)
    
    def forward(self, memory, controller_output, usage, precedence, link):
        batch_size = memory.shape[0]
        
        # Generate write parameters
        erase_vector = torch.sigmoid(self.erase_vector(controller_output))
        add_vector = torch.tanh(self.add_vector(controller_output))
        
        # Compute allocation weighting
        free_gates = torch.sigmoid(self.free_gates(controller_output))
        
        # Update usage
        for i, free_gate in enumerate(free_gates):
            usage = usage * (1 - free_gate * read_weights[i])
        
        # Allocation weighting
        allocation_weights = self._compute_allocation_weights(usage)
        
        # Write weighting
        write_gate = torch.sigmoid(self.write_gate(controller_output))
        allocation_gate = torch.sigmoid(self.allocation_gate(controller_output))
        
        write_weights = write_gate * (allocation_gate * allocation_weights + 
                                    (1 - allocation_gate) * content_weights)
        
        # Write operation
        erase_vector_expanded = erase_vector.unsqueeze(1).expand(-1, self.memory_size, -1)
        add_vector_expanded = add_vector.unsqueeze(1).expand(-1, self.memory_size, -1)
        write_weights_expanded = write_weights.unsqueeze(-1).expand(-1, -1, self.memory_dim)
        
        new_memory = memory * (1 - write_weights_expanded * erase_vector_expanded) + \
                    write_weights_expanded * add_vector_expanded
        
        # Update precedence
        precedence = (1 - torch.sum(write_weights, dim=-1, keepdim=True)) * precedence + \
                    write_weights
        
        # Update link matrix
        link = self._update_link_matrix(link, write_weights, precedence)
        
        return new_memory, usage, precedence, link
    
    def _compute_allocation_weights(self, usage):
        """Compute allocation weights based on usage"""
        # Sort usage in ascending order
        sorted_usage, indices = torch.sort(usage, dim=-1)
        
        # Compute allocation weights
        allocation_weights = torch.zeros_like(usage)
        for b in range(usage.shape[0]):
            for i in range(self.memory_size):
                allocation_weights[b, indices[b, i]] = (1 - sorted_usage[b, i]) * \
                    torch.prod(sorted_usage[b, :i])
        
        return allocation_weights
    
    def _update_link_matrix(self, link, write_weights, precedence):
        """Update temporal link matrix"""
        # Simplified link matrix update
        # In full DNC, this would be more complex
        return link
```

## Implementation Details

### Memory Networks
```python
class MemoryNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, memory_size, num_hops=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        self.num_hops = num_hops
        
        # Embeddings
        self.embedding_A = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_B = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_C = nn.Embedding(vocab_size, embedding_dim)
        
        # Memory
        self.memory = nn.Parameter(torch.randn(memory_size, embedding_dim))
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, story, query):
        batch_size = story.shape[0]
        
        # Embed story and query
        story_embedded = self.embedding_A(story)  # (batch_size, story_len, embedding_dim)
        query_embedded = self.embedding_B(query)  # (batch_size, query_len, embedding_dim)
        
        # Initialize memory state
        memory_state = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Multiple hops
        for hop in range(self.num_hops):
            # Compute attention over memory
            attention_weights = self._compute_attention(query_embedded, memory_state)
            
            # Read from memory
            memory_read = torch.sum(attention_weights.unsqueeze(-1) * memory_state, dim=1)
            
            # Update query
            query_embedded = query_embedded + memory_read.unsqueeze(1)
            
            # Update memory (simplified)
            if hop < self.num_hops - 1:
                memory_state = self._update_memory(memory_state, story_embedded)
        
        # Final answer
        answer_logits = self.output_projection(query_embedded)
        
        return answer_logits
    
    def _compute_attention(self, query, memory):
        """Compute attention weights between query and memory"""
        # Compute similarity
        query_sum = torch.sum(query, dim=1)  # (batch_size, embedding_dim)
        similarity = torch.matmul(query_sum, memory.transpose(-2, -1))  # (batch_size, memory_size)
        
        # Apply softmax
        attention_weights = F.softmax(similarity, dim=-1)
        
        return attention_weights
    
    def _update_memory(self, memory, story):
        """Update memory with story information"""
        # Simplified memory update
        # In practice, this would be more sophisticated
        return memory
```

### Stack-Augmented Neural Networks
```python
class StackAugmentedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, stack_size=100, stack_dim=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.stack_size = stack_size
        self.stack_dim = stack_dim
        
        # RNN controller
        self.rnn = nn.LSTM(input_size + stack_dim, hidden_size, batch_first=True)
        
        # Stack operations
        self.push_gate = nn.Linear(hidden_size, 1)
        self.pop_gate = nn.Linear(hidden_size, 1)
        self.no_op_gate = nn.Linear(hidden_size, 1)
        
        # Stack value projection
        self.stack_value = nn.Linear(hidden_size, stack_dim)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Initialize stack
        stack = torch.zeros(batch_size, self.stack_size, self.stack_dim, device=x.device)
        stack_pointer = torch.zeros(batch_size, 1, device=x.device)
        
        outputs = []
        
        for t in range(seq_len):
            # Read from stack
            stack_read = self._read_stack(stack, stack_pointer)
            
            # Controller input
            controller_input = torch.cat([x[:, t, :], stack_read], dim=-1)
            
            # RNN step
            controller_output, _ = self.rnn(controller_input.unsqueeze(1))
            controller_output = controller_output.squeeze(1)
            
            # Stack operations
            push_gate = torch.sigmoid(self.push_gate(controller_output))
            pop_gate = torch.sigmoid(self.pop_gate(controller_output))
            no_op_gate = torch.sigmoid(self.no_op_gate(controller_output))
            
            # Normalize gates
            gate_sum = push_gate + pop_gate + no_op_gate
            push_gate = push_gate / gate_sum
            pop_gate = pop_gate / gate_sum
            no_op_gate = no_op_gate / gate_sum
            
            # Update stack
            stack_value = torch.tanh(self.stack_value(controller_output))
            stack, stack_pointer = self._update_stack(stack, stack_pointer, 
                                                    stack_value, push_gate, pop_gate)
            
            # Generate output
            output = self.output_projection(controller_output)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs
    
    def _read_stack(self, stack, stack_pointer):
        """Read from the top of the stack"""
        batch_size = stack.shape[0]
        
        # Get stack pointer indices
        indices = stack_pointer.long().squeeze(-1)
        
        # Read from stack
        stack_read = torch.zeros(batch_size, self.stack_dim, device=stack.device)
        for b in range(batch_size):
            if indices[b] > 0:
                stack_read[b] = stack[b, indices[b] - 1]
        
        return stack_read
    
    def _update_stack(self, stack, stack_pointer, value, push_gate, pop_gate):
        """Update stack with push/pop operations"""
        batch_size = stack.shape[0]
        
        # Update stack pointer
        new_stack_pointer = stack_pointer + push_gate - pop_gate
        new_stack_pointer = torch.clamp(new_stack_pointer, 0, self.stack_size - 1)
        
        # Update stack contents
        new_stack = stack.clone()
        indices = new_stack_pointer.long().squeeze(-1)
        
        for b in range(batch_size):
            if push_gate[b] > 0.5 and indices[b] < self.stack_size:
                new_stack[b, indices[b]] = value[b]
        
        return new_stack, new_stack_pointer
```

## Advanced Variants

### 1. Neural Programmer-Interpreter
```python
class NeuralProgrammerInterpreter(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, memory_dim):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        # Program memory
        self.program_memory = nn.Parameter(torch.randn(memory_size, memory_dim))
        
        # Data memory
        self.data_memory = nn.Parameter(torch.randn(memory_size, memory_dim))
        
        # Controller
        self.controller = nn.LSTM(input_size + memory_dim * 2, hidden_size, batch_first=True)
        
        # Program counter
        self.program_counter = None
        
        # Instruction decoder
        self.instruction_decoder = nn.Linear(hidden_size, memory_dim)
    
    def forward(self, x, max_steps=10):
        batch_size = x.shape[0]
        
        # Initialize memories and program counter
        program_memory = self.program_memory.unsqueeze(0).expand(batch_size, -1, -1)
        data_memory = self.data_memory.unsqueeze(0).expand(batch_size, -1, -1)
        program_counter = torch.zeros(batch_size, 1, device=x.device)
        
        outputs = []
        
        for step in range(max_steps):
            # Read current instruction
            pc_indices = program_counter.long().squeeze(-1)
            instruction = torch.zeros(batch_size, self.memory_dim, device=x.device)
            for b in range(batch_size):
                if pc_indices[b] < self.memory_size:
                    instruction[b] = program_memory[b, pc_indices[b]]
            
            # Controller input
            controller_input = torch.cat([x, instruction], dim=-1)
            
            # Controller step
            controller_output, _ = self.controller(controller_input.unsqueeze(1))
            controller_output = controller_output.squeeze(1)
            
            # Execute instruction
            data_memory, program_counter = self._execute_instruction(
                controller_output, data_memory, program_counter
            )
            
            # Generate output
            output = self.instruction_decoder(controller_output)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs
    
    def _execute_instruction(self, controller_output, data_memory, program_counter):
        """Execute the current instruction"""
        # Simplified instruction execution
        # In practice, this would decode and execute various instruction types
        
        # Increment program counter
        new_program_counter = program_counter + 1
        
        return data_memory, new_program_counter
```

### 2. Hierarchical Memory Networks
```python
class HierarchicalMemoryNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, memory_levels=3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_levels = memory_levels
        
        # Hierarchical memory banks
        self.memory_banks = nn.ModuleList([
            nn.Parameter(torch.randn(2**i, hidden_size)) 
            for i in range(memory_levels)
        ])
        
        # Attention mechanisms for each level
        self.attention_mechanisms = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads=8) 
            for _ in range(memory_levels)
        ])
        
        # Controller
        self.controller = nn.LSTM(input_size + hidden_size * memory_levels, 
                                hidden_size, batch_first=True)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Initialize memory states
        memory_states = []
        for memory_bank in self.memory_banks:
            memory_state = memory_bank.unsqueeze(0).expand(batch_size, -1, -1)
            memory_states.append(memory_state)
        
        outputs = []
        
        for t in range(seq_len):
            # Read from all memory levels
            memory_reads = []
            for level in range(self.memory_levels):
                memory_read = self._read_memory_level(
                    x[:, t, :], memory_states[level], self.attention_mechanisms[level]
                )
                memory_reads.append(memory_read)
            
            # Concatenate memory reads
            all_memory_reads = torch.cat(memory_reads, dim=-1)
            
            # Controller input
            controller_input = torch.cat([x[:, t, :], all_memory_reads], dim=-1)
            
            # Controller step
            controller_output, _ = self.controller(controller_input.unsqueeze(1))
            controller_output = controller_output.squeeze(1)
            
            # Update memory states
            for level in range(self.memory_levels):
                memory_states[level] = self._update_memory_level(
                    memory_states[level], controller_output, level
                )
            
            # Generate output
            output = self.output_projection(controller_output)
            outputs.append(output)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs
    
    def _read_memory_level(self, query, memory, attention_mechanism):
        """Read from a specific memory level"""
        # Reshape for attention
        query = query.unsqueeze(1)  # (batch_size, 1, hidden_size)
        
        # Apply attention
        memory_read, _ = attention_mechanism(query, memory, memory)
        
        return memory_read.squeeze(1)
    
    def _update_memory_level(self, memory, controller_output, level):
        """Update a specific memory level"""
        # Simplified memory update
        # In practice, this would be more sophisticated
        return memory
```

## Performance Metrics

### 1. Memory Access Metrics
- **Memory utilization**: Percentage of memory slots used
- **Access patterns**: Distribution of memory access frequencies
- **Memory efficiency**: Ratio of useful to total memory accesses
- **Retrieval accuracy**: Accuracy of memory retrieval operations

### 2. Computational Metrics
- **Training time**: Time required for training convergence
- **Inference speed**: Time per forward pass
- **Memory overhead**: Additional memory requirements
- **Scalability**: Performance with increasing memory size

### 3. Task-Specific Metrics
- **Question answering accuracy**: For QA tasks
- **Program execution accuracy**: For program execution tasks
- **Memory retention**: Ability to retain information over time
- **Transfer learning**: Performance on unseen tasks

## Applications

### 1. Question Answering
- **Complex reasoning**: Multi-step reasoning with external memory
- **Knowledge retrieval**: Accessing stored knowledge for answers
- **Context understanding**: Maintaining context across long passages
- **Multi-hop reasoning**: Chaining multiple memory accesses

### 2. Language Modeling
- **Long-term dependencies**: Capturing dependencies across long sequences
- **Context memory**: Maintaining context across different parts of text
- **Knowledge integration**: Integrating external knowledge into language models
- **Memory-augmented generation**: Using memory for better text generation

### 3. Program Execution
- **Neural program execution**: Executing programs with neural networks
- **Algorithm learning**: Learning algorithms from examples
- **Code generation**: Generating code with memory of previous patterns
- **Program synthesis**: Synthesizing programs from specifications

### 4. Reasoning Tasks
- **Logical reasoning**: Performing logical operations with memory
- **Mathematical reasoning**: Solving mathematical problems step-by-step
- **Spatial reasoning**: Maintaining spatial relationships in memory
- **Temporal reasoning**: Reasoning about temporal sequences

## Research Frontiers

### 1. Scalable Memory Systems
- **Large-scale memory**: Scaling to millions of memory slots
- **Distributed memory**: Distributed memory across multiple devices
- **Memory compression**: Efficient memory compression techniques
- **Memory hierarchies**: Multi-level memory hierarchies

### 2. Adaptive Memory Management
- **Dynamic memory allocation**: Automatic memory allocation and deallocation
- **Memory optimization**: Optimizing memory usage for specific tasks
- **Memory sharing**: Sharing memory across different tasks
- **Memory transfer**: Transferring memory between different models

### 3. Memory Interpretability
- **Memory visualization**: Visualizing memory contents and access patterns
- **Memory analysis**: Analyzing what information is stored in memory
- **Memory debugging**: Debugging memory-related issues
- **Memory explanation**: Explaining memory-based decisions

### 4. Hybrid Memory Systems
- **Memory + Attention**: Combining external memory with attention mechanisms
- **Memory + Reinforcement Learning**: Using memory in RL agents
- **Memory + Graph Neural Networks**: Graph-structured memory
- **Memory + Transformers**: Memory-augmented transformer architectures

## Usage Examples

### Basic External Memory Network
```python
import torch
import torch.nn as nn

# Initialize external memory network
memory_network = ExternalMemoryNetwork(
    input_size=64,
    hidden_size=128,
    memory_size=256,
    memory_dim=32,
    num_read_heads=4,
    num_write_heads=1
)

# Create sample data
batch_size, seq_len = 16, 20
x = torch.randn(batch_size, seq_len, 64)

# Forward pass
outputs, memory_state, read_vectors = memory_network(x)

print(f"Output shape: {outputs.shape}")
print(f"Memory state shape: {memory_state.shape}")
print(f"Read vectors shape: {read_vectors.shape}")
```

### Differentiable Neural Computer
```python
# Initialize DNC
dnc = DifferentiableNeuralComputer(
    input_size=64,
    hidden_size=128,
    memory_size=256,
    memory_dim=32,
    num_read_heads=4,
    num_write_heads=1
)

# Forward pass
outputs, memory_state, read_vectors = dnc(x)

# Access memory contents
print(f"Memory contents shape: {memory_state.shape}")
print(f"Memory usage: {dnc.usage.shape}")
```

### Memory Network for Question Answering
```python
# Initialize memory network
vocab_size = 10000
memory_network = MemoryNetwork(
    vocab_size=vocab_size,
    embedding_dim=128,
    memory_size=50,
    num_hops=3
)

# Create sample data
story = torch.randint(0, vocab_size, (16, 100))  # 16 stories, 100 tokens each
query = torch.randint(0, vocab_size, (16, 10))   # 16 queries, 10 tokens each

# Forward pass
answer_logits = memory_network(story, query)

print(f"Answer logits shape: {answer_logits.shape}")
```

### Stack-Augmented RNN
```python
# Initialize stack-augmented RNN
stack_rnn = StackAugmentedRNN(
    input_size=64,
    hidden_size=128,
    stack_size=100,
    stack_dim=16
)

# Forward pass
outputs = stack_rnn(x)

print(f"Output shape: {outputs.shape}")
```

### Hierarchical Memory Network
```python
# Initialize hierarchical memory network
hierarchical_memory = HierarchicalMemoryNetwork(
    input_size=64,
    hidden_size=128,
    memory_levels=3
)

# Forward pass
outputs = hierarchical_memory(x)

print(f"Output shape: {outputs.shape}")
```

### Memory Visualization
```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_memory_access(memory_state, read_vectors, write_weights):
    """Visualize memory access patterns"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Memory contents
    sns.heatmap(memory_state[0].detach().numpy(), ax=axes[0], cmap='viridis')
    axes[0].set_title('Memory Contents')
    axes[0].set_xlabel('Memory Dimension')
    axes[0].set_ylabel('Memory Location')
    
    # Read patterns
    read_patterns = read_vectors[0].detach().numpy()
    sns.heatmap(read_patterns.T, ax=axes[1], cmap='plasma')
    axes[1].set_title('Read Patterns Over Time')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Read Head')
    
    # Write patterns
    write_patterns = write_weights[0].detach().numpy()
    sns.heatmap(write_patterns.T, ax=axes[2], cmap='inferno')
    axes[2].set_title('Write Patterns Over Time')
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Memory Location')
    
    plt.tight_layout()
    plt.show()

# Visualize memory access
visualize_memory_access(memory_state, read_vectors, write_weights)
```

## Files in this Directory
- `external_memory.py`: Core external memory implementation
- `memory_operations.py`: Memory read/write operations
- `attention_variants.py`: Different attention mechanisms
- `example_usage.py`: Working examples
