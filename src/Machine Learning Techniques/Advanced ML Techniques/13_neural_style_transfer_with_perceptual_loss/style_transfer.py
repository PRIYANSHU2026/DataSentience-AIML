"""
Style Transfer Networks

This module implements various style transfer networks including:
- Basic style transfer with perceptual loss
- Fast style transfer networks
- Multi-style transfer
- Training loops and optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from perceptual_loss import PerceptualLoss, MultiScalePerceptualLoss, StyleAnalyzer


class StyleTransferNetwork:
    """Basic style transfer network using perceptual loss"""
    
    def __init__(self, content_weight: float = 1.0, style_weight: float = 1e6,
                 content_layers: List[str] = None, style_layers: List[str] = None,
                 model_name: str = 'vgg19', device: str = 'cpu'):
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.device = device
        
        # Initialize perceptual loss
        self.perceptual_loss = PerceptualLoss(
            model_name=model_name,
            content_layers=content_layers,
            style_layers=style_layers,
            content_weight=content_weight,
            style_weight=style_weight
        ).to(device)
        
        # Style analyzer
        from perceptual_loss import FeatureExtractor
        feature_extractor = FeatureExtractor(model_name)
        self.style_analyzer = StyleAnalyzer(feature_extractor)
        
        # Training history
        self.training_history = []
    
    def transfer_style(self, content_image: torch.Tensor, style_image: torch.Tensor,
                      num_iterations: int = 300, learning_rate: float = 0.01,
                      save_intermediate: bool = False) -> Tuple[torch.Tensor, List]:
        """
        Perform style transfer using iterative optimization
        
        Args:
            content_image: Content image tensor
            style_image: Style image tensor
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for optimization
            save_intermediate: Whether to save intermediate results
        
        Returns:
            Stylized image and training history
        """
        # Move images to device
        content_image = content_image.to(self.device)
        style_image = style_image.to(self.device)
        
        # Initialize output image (start with content image)
        output_image = content_image.clone().requires_grad_(True)
        
        # Set targets for perceptual loss
        self.perceptual_loss.set_targets(content_image, style_image)
        
        # Optimizer
        optimizer = optim.LBFGS([output_image], lr=learning_rate)
        
        # Training history
        history = []
        intermediate_results = []
        
        def closure():
            optimizer.zero_grad()
            
            # Compute loss
            loss, loss_breakdown = self.perceptual_loss(output_image)
            
            # Backward pass
            loss.backward()
            
            # Record history
            history.append(loss_breakdown)
            
            return loss
        
        print(f"Starting style transfer with {num_iterations} iterations...")
        
        for i in range(num_iterations):
            # Optimization step
            loss = optimizer.step(closure)
            
            # Clamp values to valid range
            output_image.data.clamp_(0, 1)
            
            # Save intermediate results
            if save_intermediate and i % 50 == 0:
                intermediate_results.append(output_image.clone().detach())
            
            # Print progress
            if i % 50 == 0:
                print(f"Iteration {i}: Loss = {loss.item():.4f}")
        
        self.training_history = history
        return output_image.detach(), intermediate_results
    
    def analyze_style(self, style_image: torch.Tensor) -> Dict:
        """Analyze the style characteristics of an image"""
        return self.style_analyzer.analyze_style(style_image.to(self.device))
    
    def visualize_training_progress(self, save_path: str = 'training_progress.png'):
        """Visualize training progress"""
        if not self.training_history:
            print("No training history available")
            return
        
        # Extract loss values
        iterations = range(len(self.training_history))
        total_losses = [h['total_loss'] for h in self.training_history]
        content_losses = [h['content_loss'] for h in self.training_history]
        style_losses = [h['style_loss'] for h in self.training_history]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(iterations, total_losses, 'b-', label='Total Loss')
        plt.title('Total Loss Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(iterations, content_losses, 'g-', label='Content Loss')
        plt.title('Content Loss Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(iterations, style_losses, 'r-', label='Style Loss')
        plt.title('Style Loss Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(iterations, np.log10(style_losses), 'r-', label='Style Loss (log)')
        plt.title('Style Loss Over Time (Log Scale)')
        plt.xlabel('Iteration')
        plt.ylabel('Log Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training progress visualization saved to {save_path}")


class FastStyleTransfer(nn.Module):
    """Fast style transfer network using feedforward architecture"""
    
    def __init__(self, input_channels: int = 3, output_channels: int = 3):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=9, padding=4),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(128) for _ in range(5)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, output_channels, kernel_size=9, padding=4),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Encoder
        x = self.encoder(x)
        
        # Residual blocks
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        # Decoder
        x = self.decoder(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for fast style transfer"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class FastStyleTransferTrainer:
    """Trainer for fast style transfer networks"""
    
    def __init__(self, model: FastStyleTransfer, content_weight: float = 1.0,
                 style_weight: float = 1e6, device: str = 'cpu'):
        self.model = model.to(device)
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.device = device
        
        # Perceptual loss
        self.perceptual_loss = PerceptualLoss(
            content_weight=content_weight,
            style_weight=style_weight
        ).to(device)
        
        # Training history
        self.training_history = []
    
    def train(self, train_loader, content_image: torch.Tensor, style_image: torch.Tensor,
              num_epochs: int = 2, learning_rate: float = 0.001) -> List[Dict]:
        """
        Train the fast style transfer network
        
        Args:
            train_loader: Data loader for training images
            content_image: Reference content image
            style_image: Reference style image
            num_epochs: Number of training epochs
            learning_rate: Learning rate
        
        Returns:
            Training history
        """
        # Set targets for perceptual loss
        self.perceptual_loss.set_targets(content_image.to(self.device), style_image.to(self.device))
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch_idx, batch_images in enumerate(train_loader):
                batch_images = batch_images.to(self.device)
                
                # Forward pass
                stylized_images = self.model(batch_images)
                
                # Compute loss
                loss, loss_breakdown = self.perceptual_loss(stylized_images)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Record loss
                epoch_losses.append(loss_breakdown)
                
                # Print progress
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}: Loss = {loss.item():.4f}")
            
            # Record epoch statistics
            avg_loss = np.mean([l['total_loss'] for l in epoch_losses])
            self.training_history.append({
                'epoch': epoch,
                'avg_loss': avg_loss,
                'losses': epoch_losses
            })
        
        return self.training_history
    
    def transfer_style(self, content_image: torch.Tensor) -> torch.Tensor:
        """Transfer style using trained network"""
        self.model.eval()
        with torch.no_grad():
            stylized_image = self.model(content_image.to(self.device))
        return stylized_image


class MultiStyleTransfer(nn.Module):
    """Multi-style transfer network"""
    
    def __init__(self, num_styles: int, input_channels: int = 3, output_channels: int = 3):
        super().__init__()
        self.num_styles = num_styles
        
        # Style-specific networks
        self.style_networks = nn.ModuleList([
            FastStyleTransfer(input_channels, output_channels) for _ in range(num_styles)
        ])
        
        # Style classifier
        self.style_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(output_channels, num_styles),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor, style_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with multiple styles
        
        Args:
            x: Input image
            style_weights: Style mixing weights (batch_size, num_styles)
        
        Returns:
            Stylized image
        """
        batch_size = x.shape[0]
        
        if style_weights is None:
            # Use equal weights for all styles
            style_weights = torch.ones(batch_size, self.num_styles).to(x.device) / self.num_styles
        
        # Generate stylized images for each style
        stylized_images = []
        for i, style_network in enumerate(self.style_networks):
            stylized = style_network(x)
            stylized_images.append(stylized)
        
        # Combine styles using weights
        stylized_images = torch.stack(stylized_images, dim=1)  # (batch_size, num_styles, C, H, W)
        style_weights = style_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (batch_size, num_styles, 1, 1, 1)
        
        combined_image = torch.sum(stylized_images * style_weights, dim=1)
        
        return combined_image
    
    def get_style_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get style weights using the style classifier"""
        # Use the first style network to extract features
        features = self.style_networks[0](x)
        style_weights = self.style_classifier(features)
        return style_weights


class AdaptiveStyleTransfer(nn.Module):
    """Adaptive style transfer with dynamic style selection"""
    
    def __init__(self, num_styles: int, input_channels: int = 3, output_channels: int = 3):
        super().__init__()
        self.num_styles = num_styles
        
        # Multi-style network
        self.multi_style_network = MultiStyleTransfer(num_styles, input_channels, output_channels)
        
        # Content analyzer
        self.content_analyzer = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_styles),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor, manual_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with adaptive style selection
        
        Args:
            x: Input image
            manual_weights: Manual style weights (optional)
        
        Returns:
            Stylized image
        """
        if manual_weights is None:
            # Automatically determine style weights based on content
            style_weights = self.content_analyzer(x)
        else:
            style_weights = manual_weights
        
        return self.multi_style_network(x, style_weights)


class StyleTransferVisualizer:
    """Visualization tools for style transfer"""
    
    @staticmethod
    def create_comparison_grid(content_image: torch.Tensor, style_image: torch.Tensor,
                              output_image: torch.Tensor, save_path: str = 'style_transfer_comparison.png'):
        """Create a comparison grid of content, style, and output images"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Content image
        content_np = content_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        content_np = np.clip(content_np, 0, 1)
        axes[0].imshow(content_np)
        axes[0].set_title('Content Image')
        axes[0].axis('off')
        
        # Style image
        style_np = style_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        style_np = np.clip(style_np, 0, 1)
        axes[1].imshow(style_np)
        axes[1].set_title('Style Image')
        axes[1].axis('off')
        
        # Output image
        output_np = output_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_np = np.clip(output_np, 0, 1)
        axes[2].imshow(output_np)
        axes[2].set_title('Stylized Output')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Style transfer comparison saved to {save_path}")
    
    @staticmethod
    def create_intermediate_progress_grid(intermediate_results: List[torch.Tensor],
                                        save_path: str = 'intermediate_progress.png'):
        """Create a grid showing intermediate results during training"""
        if not intermediate_results:
            print("No intermediate results to visualize")
            return
        
        num_results = len(intermediate_results)
        cols = min(5, num_results)
        rows = (num_results + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, result in enumerate(intermediate_results):
            row = i // cols
            col = i % cols
            
            result_np = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
            result_np = np.clip(result_np, 0, 1)
            
            axes[row, col].imshow(result_np)
            axes[row, col].set_title(f'Iteration {i*50}')
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(num_results, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Intermediate progress visualization saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    print("Style Transfer Networks Demo")
    print("=" * 50)
    
    # Create dummy images
    batch_size = 1
    content_image = torch.randn(batch_size, 3, 256, 256)
    style_image = torch.randn(batch_size, 3, 256, 256)
    
    # Test basic style transfer
    print("Testing Basic Style Transfer...")
    style_transfer = StyleTransferNetwork(
        content_weight=1.0,
        style_weight=1e6,
        device='cpu'
    )
    
    # Perform style transfer (fewer iterations for demo)
    output_image, intermediate_results = style_transfer.transfer_style(
        content_image, style_image, num_iterations=50, save_intermediate=True
    )
    
    print(f"Output image shape: {output_image.shape}")
    print(f"Number of intermediate results: {len(intermediate_results)}")
    
    # Test fast style transfer
    print("\nTesting Fast Style Transfer...")
    fast_style_transfer = FastStyleTransfer()
    trainer = FastStyleTransferTrainer(fast_style_transfer, device='cpu')
    
    # Test forward pass
    stylized = trainer.transfer_style(content_image)
    print(f"Fast style transfer output shape: {stylized.shape}")
    
    # Test multi-style transfer
    print("\nTesting Multi-Style Transfer...")
    multi_style_transfer = MultiStyleTransfer(num_styles=3)
    multi_output = multi_style_transfer(content_image)
    print(f"Multi-style transfer output shape: {multi_output.shape}")
    
    # Test adaptive style transfer
    print("\nTesting Adaptive Style Transfer...")
    adaptive_transfer = AdaptiveStyleTransfer(num_styles=3)
    adaptive_output = adaptive_transfer(content_image)
    print(f"Adaptive style transfer output shape: {adaptive_output.shape}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualizer = StyleTransferVisualizer()
    visualizer.create_comparison_grid(content_image, style_image, output_image)
    visualizer.create_intermediate_progress_grid(intermediate_results)
    
    print("Style transfer demo completed!")
