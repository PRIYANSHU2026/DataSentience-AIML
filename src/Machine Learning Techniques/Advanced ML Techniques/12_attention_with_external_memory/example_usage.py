"""
Example Usage: Attention with External Memory

This file demonstrates practical usage of external memory networks including:
- Basic memory-augmented sequence processing
- Question-answering with external memory
- Memory visualization and analysis
- Advanced memory operations
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from external_memory import (
    ExternalMemoryNetwork, HierarchicalMemoryNetwork,
    MemoryBank, MemoryController, ContentBasedAttention,
    LocationBasedAttention
)


def example_1_basic_memory_operations():
    """Example 1: Basic memory operations and visualization"""
    print("=" * 60)
    print("Example 1: Basic Memory Operations")
    print("=" * 60)
    
    # Create memory bank
    memory_bank = MemoryBank(memory_size=32, memory_dim=16)
    
    # Create some test data
    batch_size, num_heads = 2, 3
    read_weights = torch.randn(batch_size, num_heads, 32)
    read_weights = torch.softmax(read_weights, dim=2)  # Normalize
    
    # Read from memory
    read_vectors = memory_bank.read(read_weights)
    print(f"Read vectors shape: {read_vectors.shape}")
    
    # Write to memory
    write_weights = torch.randn(batch_size, 1, 32)
    write_weights = torch.softmax(write_weights, dim=2)
    write_vectors = torch.randn(batch_size, 1, 16)
    erase_vectors = torch.sigmoid(torch.randn(batch_size, 1, 16))
    
    memory_bank.write(write_weights, write_vectors, erase_vectors)
    
    # Get memory state
    memory_state = memory_bank.get_memory_state()
    print(f"Memory state keys: {memory_state.keys()}")
    print(f"Memory matrix shape: {memory_state['memory'].shape}")
    print(f"Usage vector shape: {memory_state['usage'].shape}")
    
    return memory_bank


def example_2_sequence_processing_with_memory():
    """Example 2: Sequence processing with external memory"""
    print("\n" + "=" * 60)
    print("Example 2: Sequence Processing with External Memory")
    print("=" * 60)
    
    # Create external memory network
    memory_network = ExternalMemoryNetwork(
        input_size=64,
        output_size=10,
        memory_size=128,
        memory_dim=32,
        controller_size=64,
        num_read_heads=4,
        num_write_heads=1
    )
    
    # Create sequence data
    batch_size, seq_len = 4, 12
    sequence = torch.randn(batch_size, seq_len, 64)
    
    # Process sequence
    outputs, memory_info = memory_network(sequence)
    
    print(f"Input sequence shape: {sequence.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Memory info keys: {memory_info.keys()}")
    
    # Analyze attention patterns
    read_weights = memory_info['attention_info']['read_weights']
    write_weights = memory_info['attention_info']['write_weights']
    
    print(f"Number of read operations: {len(read_weights)}")
    print(f"Number of write operations: {len(write_weights)}")
    print(f"Read weights shape: {read_weights[0].shape}")
    print(f"Write weights shape: {write_weights[0].shape}")
    
    return memory_network, memory_info


def example_3_question_answering_with_memory():
    """Example 3: Question answering with external memory"""
    print("\n" + "=" * 60)
    print("Example 3: Question Answering with External Memory")
    print("=" * 60)
    
    class MemoryQA(nn.Module):
        """Question answering system with external memory"""
        
        def __init__(self, vocab_size: int, embedding_dim: int = 64, memory_size: int = 128):
            super().__init__()
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.memory_size = memory_size
            
            # Embeddings
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            
            # External memory network
            self.memory_network = ExternalMemoryNetwork(
                input_size=embedding_dim,
                output_size=embedding_dim,
                memory_size=memory_size,
                memory_dim=32,
                controller_size=64,
                num_read_heads=4,
                num_write_heads=1
            )
            
            # Output projection
            self.output_projection = nn.Linear(embedding_dim, vocab_size)
            
        def forward(self, story: torch.Tensor, question: torch.Tensor) -> torch.Tensor:
            """
            Forward pass for question answering
            
            Args:
                story: (batch_size, story_len) - Story tokens
                question: (batch_size, question_len) - Question tokens
            
            Returns:
                answer_logits: (batch_size, vocab_size) - Answer logits
            """
            batch_size = story.shape[0]
            
            # Embed story and question
            story_embedded = self.embedding(story)  # (batch_size, story_len, embedding_dim)
            question_embedded = self.embedding(question)  # (batch_size, question_len, embedding_dim)
            
            # Process story through memory network
            story_outputs, memory_info = self.memory_network(story_embedded)
            
            # Process question (use last output as question representation)
            question_outputs, _ = self.memory_network(question_embedded)
            question_repr = question_outputs[:, -1, :]  # (batch_size, embedding_dim)
            
            # Generate answer
            answer_logits = self.output_projection(question_repr)
            
            return answer_logits, memory_info
    
    # Create QA model
    vocab_size = 1000
    qa_model = MemoryQA(vocab_size=vocab_size, embedding_dim=64, memory_size=128)
    
    # Create dummy data
    batch_size, story_len, question_len = 2, 20, 5
    story = torch.randint(0, vocab_size, (batch_size, story_len))
    question = torch.randint(0, vocab_size, (batch_size, question_len))
    
    # Forward pass
    answer_logits, memory_info = qa_model(story, question)
    
    print(f"Story shape: {story.shape}")
    print(f"Question shape: {question.shape}")
    print(f"Answer logits shape: {answer_logits.shape}")
    
    # Get predicted answer
    predicted_answers = torch.argmax(answer_logits, dim=1)
    print(f"Predicted answers: {predicted_answers}")
    
    return qa_model, memory_info


def example_4_memory_visualization():
    """Example 4: Memory visualization and analysis"""
    print("\n" + "=" * 60)
    print("Example 4: Memory Visualization")
    print("=" * 60)
    
    # Create memory network and process some data
    memory_network = ExternalMemoryNetwork(
        input_size=32,
        output_size=10,
        memory_size=64,
        memory_dim=16,
        controller_size=32,
        num_read_heads=2,
        num_write_heads=1
    )
    
    # Process sequence
    sequence = torch.randn(2, 10, 32)
    outputs, memory_info = memory_network(sequence)
    
    # Get visualization data
    viz_data = memory_network.get_memory_visualization()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Memory matrix heatmap
    memory_matrix = viz_data['memory_matrix']
    sns.heatmap(memory_matrix, ax=axes[0, 0], cmap='viridis')
    axes[0, 0].set_title('Memory Matrix')
    axes[0, 0].set_xlabel('Memory Dimension')
    axes[0, 0].set_ylabel('Memory Location')
    
    # Plot 2: Usage vector
    usage_vector = viz_data['usage_vector']
    axes[0, 1].bar(range(len(usage_vector)), usage_vector)
    axes[0, 1].set_title('Memory Usage Vector')
    axes[0, 1].set_xlabel('Memory Location')
    axes[0, 1].set_ylabel('Usage')
    
    # Plot 3: Read attention over time
    if viz_data['read_history']:
        read_history = np.array(viz_data['read_history'])
        read_history_avg = np.mean(read_history, axis=(0, 2))  # Average over batch and heads
        axes[1, 0].plot(read_history_avg.T)
        axes[1, 0].set_title('Read Attention Over Time')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Attention Weight')
        axes[1, 0].legend([f'Location {i}' for i in range(min(5, len(read_history_avg[0])))])
    
    # Plot 4: Write attention over time
    if viz_data['write_history']:
        write_history = np.array(viz_data['write_history'])
        write_history_avg = np.mean(write_history, axis=(0, 2))  # Average over batch and heads
        axes[1, 1].plot(write_history_avg.T)
        axes[1, 1].set_title('Write Attention Over Time')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Attention Weight')
        axes[1, 1].legend([f'Location {i}' for i in range(min(5, len(write_history_avg[0])))])
    
    plt.tight_layout()
    plt.savefig('memory_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Memory visualization saved to 'memory_visualization.png'")
    
    return viz_data


def example_5_hierarchical_memory_network():
    """Example 5: Hierarchical memory network"""
    print("\n" + "=" * 60)
    print("Example 5: Hierarchical Memory Network")
    print("=" * 60)
    
    # Create hierarchical memory network
    hierarchical_network = HierarchicalMemoryNetwork(
        input_size=64,
        output_size=10,
        num_levels=3,
        memory_sizes=[64, 32, 16],
        memory_dims=[16, 32, 64]
    )
    
    # Create input sequence
    batch_size, seq_len = 4, 8
    sequence = torch.randn(batch_size, seq_len, 64)
    
    # Process through hierarchical network
    outputs, memory_info = hierarchical_network(sequence)
    
    print(f"Input shape: {sequence.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Memory info keys: {memory_info.keys()}")
    
    # Analyze each level
    for level in range(3):
        level_key = f'level_{level}'
        if level_key in memory_info:
            level_info = memory_info[level_key]
            print(f"Level {level} - Read weights: {len(level_info['read_weights'])}")
            print(f"Level {level} - Write weights: {len(level_info['write_weights'])}")
    
    return hierarchical_network, memory_info


def example_6_memory_compression():
    """Example 6: Memory compression and decompression"""
    print("\n" + "=" * 60)
    print("Example 6: Memory Compression")
    print("=" * 60)
    
    class MemoryCompression(nn.Module):
        """Memory compression and decompression module"""
        
        def __init__(self, memory_dim: int, compression_ratio: float = 0.5):
            super().__init__()
            self.memory_dim = memory_dim
            self.compressed_dim = int(memory_dim * compression_ratio)
            
            # Compression and decompression networks
            self.compressor = nn.Sequential(
                nn.Linear(memory_dim, self.compressed_dim),
                nn.ReLU(),
                nn.Linear(self.compressed_dim, self.compressed_dim)
            )
            
            self.decompressor = nn.Sequential(
                nn.Linear(self.compressed_dim, memory_dim),
                nn.ReLU(),
                nn.Linear(memory_dim, memory_dim)
            )
        
        def compress(self, memory: torch.Tensor) -> torch.Tensor:
            """Compress memory"""
            batch_size, memory_size, memory_dim = memory.shape
            memory_flat = memory.view(-1, memory_dim)
            compressed_flat = self.compressor(memory_flat)
            compressed = compressed_flat.view(batch_size, memory_size, self.compressed_dim)
            return compressed
        
        def decompress(self, compressed: torch.Tensor) -> torch.Tensor:
            """Decompress memory"""
            batch_size, memory_size, compressed_dim = compressed.shape
            compressed_flat = compressed.view(-1, compressed_dim)
            decompressed_flat = self.decompressor(compressed_flat)
            decompressed = decompressed_flat.view(batch_size, memory_size, self.memory_dim)
            return decompressed
    
    # Create compression module
    compression_module = MemoryCompression(memory_dim=32, compression_ratio=0.5)
    
    # Test compression
    memory = torch.randn(2, 64, 32)  # (batch_size, memory_size, memory_dim)
    compressed = compression_module.compress(memory)
    decompressed = compression_module.decompress(compressed)
    
    print(f"Original memory shape: {memory.shape}")
    print(f"Compressed memory shape: {compressed.shape}")
    print(f"Decompressed memory shape: {decompressed.shape}")
    
    # Calculate compression ratio
    original_size = memory.numel()
    compressed_size = compressed.numel()
    compression_ratio = compressed_size / original_size
    print(f"Compression ratio: {compression_ratio:.2f}")
    
    # Calculate reconstruction error
    reconstruction_error = torch.mean((memory - decompressed) ** 2)
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    
    return compression_module


def example_7_memory_evolution():
    """Example 7: Memory evolution over time"""
    print("\n" + "=" * 60)
    print("Example 7: Memory Evolution")
    print("=" * 60)
    
    # Create memory network
    memory_network = ExternalMemoryNetwork(
        input_size=32,
        output_size=10,
        memory_size=64,
        memory_dim=16,
        controller_size=32,
        num_read_heads=2,
        num_write_heads=1
    )
    
    # Process multiple sequences and track memory evolution
    memory_evolution = []
    num_sequences = 5
    
    for i in range(num_sequences):
        # Create sequence
        sequence = torch.randn(2, 8, 32)
        
        # Process sequence
        outputs, memory_info = memory_network(sequence)
        
        # Get memory state
        memory_state = memory_info['memory_state']
        memory_evolution.append({
            'sequence': i,
            'memory_matrix': memory_state['memory'].detach().cpu().numpy(),
            'usage_vector': memory_state['usage'].detach().cpu().numpy()
        })
    
    # Visualize memory evolution
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for i in range(min(6, len(memory_evolution))):
        row, col = i // 3, i % 3
        
        # Plot memory matrix
        memory_matrix = memory_evolution[i]['memory_matrix']
        sns.heatmap(memory_matrix, ax=axes[row, col], cmap='viridis')
        axes[row, col].set_title(f'Memory Matrix - Sequence {i}')
        axes[row, col].set_xlabel('Memory Dimension')
        axes[row, col].set_ylabel('Memory Location')
    
    plt.tight_layout()
    plt.savefig('memory_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Memory evolution visualization saved to 'memory_evolution.png'")
    
    # Analyze memory changes
    print("\nMemory Evolution Analysis:")
    for i in range(1, len(memory_evolution)):
        prev_memory = memory_evolution[i-1]['memory_matrix']
        curr_memory = memory_evolution[i]['memory_matrix']
        
        change = np.mean(np.abs(curr_memory - prev_memory))
        print(f"Sequence {i-1} to {i}: Average change = {change:.6f}")
    
    return memory_evolution


def example_8_attention_analysis():
    """Example 8: Detailed attention analysis"""
    print("\n" + "=" * 60)
    print("Example 8: Attention Analysis")
    print("=" * 60)
    
    # Create memory network
    memory_network = ExternalMemoryNetwork(
        input_size=32,
        output_size=10,
        memory_size=32,
        memory_dim=16,
        controller_size=32,
        num_read_heads=2,
        num_write_heads=1
    )
    
    # Process sequence
    sequence = torch.randn(2, 10, 32)
    outputs, memory_info = memory_network(sequence)
    
    # Extract attention weights
    read_weights = memory_info['attention_info']['read_weights']
    write_weights = memory_info['attention_info']['write_weights']
    
    # Convert to numpy for analysis
    read_weights_np = torch.stack(read_weights).numpy()
    write_weights_np = torch.stack(write_weights).numpy()
    
    # Analyze attention patterns
    print("Attention Analysis:")
    print(f"Read weights shape: {read_weights_np.shape}")
    print(f"Write weights shape: {write_weights_np.shape}")
    
    # Calculate attention statistics
    read_entropy = -np.sum(read_weights_np * np.log(read_weights_np + 1e-8), axis=-1)
    write_entropy = -np.sum(write_weights_np * np.log(write_weights_np + 1e-8), axis=-1)
    
    print(f"Average read attention entropy: {np.mean(read_entropy):.4f}")
    print(f"Average write attention entropy: {np.mean(write_entropy):.4f}")
    
    # Find most attended memory locations
    avg_read_attention = np.mean(read_weights_np, axis=(0, 1, 2))  # Average over time, batch, heads
    avg_write_attention = np.mean(write_weights_np, axis=(0, 1, 2))
    
    most_read_locations = np.argsort(avg_read_attention)[-5:]
    most_written_locations = np.argsort(avg_write_attention)[-5:]
    
    print(f"Most read memory locations: {most_read_locations}")
    print(f"Most written memory locations: {most_written_locations}")
    
    # Create attention visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Read attention over time
    read_attention_avg = np.mean(read_weights_np, axis=(1, 2))  # Average over batch and heads
    axes[0, 0].imshow(read_attention_avg.T, aspect='auto', cmap='viridis')
    axes[0, 0].set_title('Read Attention Over Time')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Memory Location')
    
    # Plot 2: Write attention over time
    write_attention_avg = np.mean(write_weights_np, axis=(1, 2))  # Average over batch and heads
    axes[0, 1].imshow(write_attention_avg.T, aspect='auto', cmap='viridis')
    axes[0, 1].set_title('Write Attention Over Time')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Memory Location')
    
    # Plot 3: Attention entropy over time
    read_entropy_avg = np.mean(read_entropy, axis=(1, 2))
    write_entropy_avg = np.mean(write_entropy, axis=(1, 2))
    axes[1, 0].plot(read_entropy_avg, label='Read Entropy')
    axes[1, 0].plot(write_entropy_avg, label='Write Entropy')
    axes[1, 0].set_title('Attention Entropy Over Time')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Entropy')
    axes[1, 0].legend()
    
    # Plot 4: Memory location usage
    axes[1, 1].bar(range(len(avg_read_attention)), avg_read_attention, alpha=0.7, label='Read')
    axes[1, 1].bar(range(len(avg_write_attention)), avg_write_attention, alpha=0.7, label='Write')
    axes[1, 1].set_title('Memory Location Usage')
    axes[1, 1].set_xlabel('Memory Location')
    axes[1, 1].set_ylabel('Average Attention')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('attention_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Attention analysis visualization saved to 'attention_analysis.png'")
    
    return read_weights_np, write_weights_np


def main():
    """Run all external memory examples"""
    print("External Memory Network Examples")
    print("=" * 80)
    
    # Run all examples
    example_1_basic_memory_operations()
    example_2_sequence_processing_with_memory()
    example_3_question_answering_with_memory()
    example_4_memory_visualization()
    example_5_hierarchical_memory_network()
    example_6_memory_compression()
    example_7_memory_evolution()
    example_8_attention_analysis()
    
    print("\n" + "=" * 80)
    print("All external memory examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
