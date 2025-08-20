"""
Neural Style Transfer with Perceptual Loss

This module implements perceptual loss functions for neural style transfer,
including content loss, style loss, and feature extraction from pre-trained networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt


class FeatureExtractor(nn.Module):
    """VGG-based feature extractor for perceptual loss computation"""
    
    def __init__(self, model_name: str = 'vgg19', pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        
        # Load pre-trained VGG model
        if model_name == 'vgg19':
            vgg = models.vgg19(pretrained=pretrained)
            self.features = vgg.features
        elif model_name == 'vgg16':
            vgg = models.vgg16(pretrained=pretrained)
            self.features = vgg.features
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Define layer names for feature extraction
        self.layer_names = {
            'conv1_1': 0, 'relu1_1': 1,
            'conv1_2': 2, 'relu1_2': 3,
            'pool1': 4,
            'conv2_1': 5, 'relu2_1': 6,
            'conv2_2': 7, 'relu2_2': 8,
            'pool2': 9,
            'conv3_1': 10, 'relu3_1': 11,
            'conv3_2': 12, 'relu3_2': 13,
            'conv3_3': 14, 'relu3_3': 15,
            'conv3_4': 16, 'relu3_4': 17,
            'pool3': 18,
            'conv4_1': 19, 'relu4_1': 20,
            'conv4_2': 21, 'relu4_2': 22,
            'conv4_3': 23, 'relu4_3': 24,
            'conv4_4': 25, 'relu4_4': 26,
            'pool4': 27,
            'conv5_1': 28, 'relu5_1': 29,
            'conv5_2': 30, 'relu5_2': 31,
            'conv5_3': 32, 'relu5_3': 33,
            'conv5_4': 34, 'relu5_4': 35,
            'pool5': 36
        }
        
        # Freeze the model parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor, target_layers: List[str]) -> Dict[str, torch.Tensor]:
        """
        Extract features from specified layers
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            target_layers: List of layer names to extract features from
        
        Returns:
            Dictionary mapping layer names to feature tensors
        """
        features = {}
        current_layer = 0
        
        for name, layer in self.features.named_children():
            x = layer(x)
            
            # Check if current layer is in target layers
            for target_layer in target_layers:
                if target_layer in self.layer_names and self.layer_names[target_layer] == current_layer:
                    features[target_layer] = x
            
            current_layer += 1
        
        return features
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input for VGG network"""
        # VGG normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        return (x - mean) / std


class GramMatrix(nn.Module):
    """Compute Gram matrix for style representation"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Gram matrix
        
        Args:
            x: Feature tensor (batch_size, channels, height, width)
        
        Returns:
            Gram matrix (batch_size, channels, channels)
        """
        batch_size, channels, height, width = x.size()
        
        # Reshape to (batch_size, channels, height*width)
        features = x.view(batch_size, channels, height * width)
        
        # Compute Gram matrix: G = F * F^T
        gram = torch.bmm(features, features.transpose(1, 2))
        
        # Normalize by the number of elements
        gram = gram / (channels * height * width)
        
        return gram


class ContentLoss(nn.Module):
    """Content loss for preserving image content"""
    
    def __init__(self, target_features: torch.Tensor, weight: float = 1.0):
        super().__init__()
        self.target_features = target_features.detach()
        self.weight = weight
        self.loss = nn.MSELoss()
    
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Compute content loss
        
        Args:
            input_features: Features from input image
        
        Returns:
            Content loss value
        """
        return self.weight * self.loss(input_features, self.target_features)


class StyleLoss(nn.Module):
    """Style loss using Gram matrices"""
    
    def __init__(self, target_gram: torch.Tensor, weight: float = 1.0):
        super().__init__()
        self.target_gram = target_gram.detach()
        self.weight = weight
        self.loss = nn.MSELoss()
        self.gram_matrix = GramMatrix()
    
    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Compute style loss
        
        Args:
            input_features: Features from input image
        
        Returns:
            Style loss value
        """
        input_gram = self.gram_matrix(input_features)
        return self.weight * self.loss(input_gram, self.target_gram)


class PerceptualLoss(nn.Module):
    """Main perceptual loss function combining content and style losses"""
    
    def __init__(self, model_name: str = 'vgg19', content_layers: List[str] = None, 
                 style_layers: List[str] = None, content_weight: float = 1.0, 
                 style_weight: float = 1e6):
        super().__init__()
        self.model_name = model_name
        self.content_weight = content_weight
        self.style_weight = style_weight
        
        # Default layers for content and style
        if content_layers is None:
            content_layers = ['relu4_2']
        if style_layers is None:
            style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        
        self.content_layers = content_layers
        self.style_layers = style_layers
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(model_name)
        self.gram_matrix = GramMatrix()
        
        # Loss modules
        self.content_losses = nn.ModuleList()
        self.style_losses = nn.ModuleList()
        
        # Target features (will be set during forward pass)
        self.target_content_features = {}
        self.target_style_grams = {}
    
    def set_targets(self, content_image: torch.Tensor, style_image: torch.Tensor):
        """Set target content and style features"""
        # Normalize images
        content_normalized = self.feature_extractor.normalize_input(content_image)
        style_normalized = self.feature_extractor.normalize_input(style_image)
        
        # Extract content features
        content_features = self.feature_extractor(content_normalized, self.content_layers)
        for layer_name, features in content_features.items():
            self.target_content_features[layer_name] = features
        
        # Extract style features and compute Gram matrices
        style_features = self.feature_extractor(style_normalized, self.style_layers)
        for layer_name, features in style_features.items():
            gram = self.gram_matrix(features)
            self.target_style_grams[layer_name] = gram
    
    def forward(self, input_image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute perceptual loss
        
        Args:
            input_image: Input image tensor
        
        Returns:
            Total loss and loss breakdown
        """
        # Normalize input
        input_normalized = self.feature_extractor.normalize_input(input_image)
        
        # Extract features from all target layers
        all_layers = list(set(self.content_layers + self.style_layers))
        features = self.feature_extractor(input_normalized, all_layers)
        
        # Compute content loss
        content_loss = 0.0
        for layer_name in self.content_layers:
            if layer_name in features and layer_name in self.target_content_features:
                content_loss += F.mse_loss(features[layer_name], self.target_content_features[layer_name])
        
        # Compute style loss
        style_loss = 0.0
        for layer_name in self.style_layers:
            if layer_name in features and layer_name in self.target_style_grams:
                input_gram = self.gram_matrix(features[layer_name])
                style_loss += F.mse_loss(input_gram, self.target_style_grams[layer_name])
        
        # Combine losses
        total_loss = self.content_weight * content_loss + self.style_weight * style_loss
        
        # Loss breakdown
        loss_breakdown = {
            'total_loss': total_loss.item(),
            'content_loss': content_loss.item(),
            'style_loss': style_loss.item(),
            'content_weight': self.content_weight,
            'style_weight': self.style_weight
        }
        
        return total_loss, loss_breakdown


class AdaptiveInstanceNormalization(nn.Module):
    """Adaptive Instance Normalization for style transfer"""
    
    def __init__(self, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, content_features: torch.Tensor, style_features: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive instance normalization
        
        Args:
            content_features: Content feature tensor
            style_features: Style feature tensor
        
        Returns:
            Normalized feature tensor
        """
        # Compute mean and variance for content features
        content_mean = torch.mean(content_features, dim=[2, 3], keepdim=True)
        content_var = torch.var(content_features, dim=[2, 3], keepdim=True) + self.epsilon
        content_std = torch.sqrt(content_var)
        
        # Compute mean and variance for style features
        style_mean = torch.mean(style_features, dim=[2, 3], keepdim=True)
        style_var = torch.var(style_features, dim=[2, 3], keepdim=True) + self.epsilon
        style_std = torch.sqrt(style_var)
        
        # Normalize content features
        normalized_content = (content_features - content_mean) / content_std
        
        # Apply style statistics
        stylized_features = normalized_content * style_std + style_mean
        
        return stylized_features


class MultiScalePerceptualLoss(nn.Module):
    """Multi-scale perceptual loss for better style transfer"""
    
    def __init__(self, scales: List[float] = None, **kwargs):
        super().__init__()
        if scales is None:
            scales = [1.0, 0.5, 0.25]
        
        self.scales = scales
        self.perceptual_losses = nn.ModuleList([
            PerceptualLoss(**kwargs) for _ in scales
        ])
    
    def set_targets(self, content_image: torch.Tensor, style_image: torch.Tensor):
        """Set targets for all scales"""
        for i, scale in enumerate(self.scales):
            # Resize images
            if scale != 1.0:
                content_scaled = F.interpolate(content_image, scale_factor=scale, mode='bilinear')
                style_scaled = F.interpolate(style_image, scale_factor=scale, mode='bilinear')
            else:
                content_scaled = content_image
                style_scaled = style_image
            
            self.perceptual_losses[i].set_targets(content_scaled, style_scaled)
    
    def forward(self, input_image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute multi-scale perceptual loss"""
        total_loss = 0.0
        loss_breakdown = {}
        
        for i, scale in enumerate(self.scales):
            # Resize input if needed
            if scale != 1.0:
                input_scaled = F.interpolate(input_image, scale_factor=scale, mode='bilinear')
            else:
                input_scaled = input_image
            
            # Compute loss for this scale
            loss, breakdown = self.perceptual_losses[i](input_scaled)
            total_loss += loss
            
            # Store breakdown
            for key, value in breakdown.items():
                if key not in loss_breakdown:
                    loss_breakdown[key] = []
                loss_breakdown[key].append(value)
        
        # Average the breakdown values
        for key in loss_breakdown:
            loss_breakdown[key] = np.mean(loss_breakdown[key])
        
        loss_breakdown['total_loss'] = total_loss.item()
        
        return total_loss, loss_breakdown


class StyleAnalyzer:
    """Analyze and visualize style characteristics"""
    
    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor
        self.gram_matrix = GramMatrix()
    
    def analyze_style(self, style_image: torch.Tensor, layer_names: List[str] = None) -> Dict:
        """Analyze style characteristics of an image"""
        if layer_names is None:
            layer_names = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        
        # Normalize image
        style_normalized = self.feature_extractor.normalize_input(style_image)
        
        # Extract features
        features = self.feature_extractor(style_normalized, layer_names)
        
        # Compute Gram matrices
        gram_matrices = {}
        for layer_name, feature in features.items():
            gram = self.gram_matrix(feature)
            gram_matrices[layer_name] = gram
        
        # Analyze characteristics
        analysis = {}
        for layer_name, gram in gram_matrices.items():
            # Compute eigenvalues for style complexity
            eigenvalues = torch.linalg.eigvals(gram.squeeze(0))
            analysis[layer_name] = {
                'gram_matrix': gram,
                'eigenvalues': eigenvalues,
                'style_complexity': torch.sum(torch.abs(eigenvalues)).item(),
                'feature_dimensions': features[layer_name].shape[1],
                'spatial_dimensions': features[layer_name].shape[2] * features[layer_name].shape[3]
            }
        
        return analysis
    
    def visualize_style_analysis(self, analysis: Dict, save_path: str = 'style_analysis.png'):
        """Visualize style analysis results"""
        num_layers = len(analysis)
        fig, axes = plt.subplots(2, num_layers, figsize=(4*num_layers, 8))
        
        for i, (layer_name, layer_analysis) in enumerate(analysis.items()):
            # Plot Gram matrix
            gram_matrix = layer_analysis['gram_matrix'].squeeze(0).detach().cpu().numpy()
            axes[0, i].imshow(gram_matrix, cmap='viridis')
            axes[0, i].set_title(f'{layer_name} - Gram Matrix')
            axes[0, i].axis('off')
            
            # Plot eigenvalues
            eigenvalues = layer_analysis['eigenvalues'].detach().cpu().numpy()
            axes[1, i].plot(np.real(eigenvalues), label='Real')
            axes[1, i].plot(np.imag(eigenvalues), label='Imaginary')
            axes[1, i].set_title(f'{layer_name} - Eigenvalues')
            axes[1, i].legend()
            axes[1, i].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Style analysis visualization saved to {save_path}")


if __name__ == "__main__":
    # Example usage
    print("Perceptual Loss Demo")
    print("=" * 50)
    
    # Create feature extractor
    feature_extractor = FeatureExtractor('vgg19')
    
    # Create dummy images
    batch_size = 1
    content_image = torch.randn(batch_size, 3, 256, 256)
    style_image = torch.randn(batch_size, 3, 256, 256)
    input_image = torch.randn(batch_size, 3, 256, 256)
    
    # Test feature extraction
    content_features = feature_extractor(feature_extractor.normalize_input(content_image), ['relu4_2'])
    print(f"Content features shape: {content_features['relu4_2'].shape}")
    
    # Test Gram matrix
    gram_matrix = GramMatrix()
    style_features = feature_extractor(feature_extractor.normalize_input(style_image), ['relu1_1'])
    gram = gram_matrix(style_features['relu1_1'])
    print(f"Gram matrix shape: {gram.shape}")
    
    # Test perceptual loss
    perceptual_loss = PerceptualLoss(
        content_weight=1.0,
        style_weight=1e6,
        content_layers=['relu4_2'],
        style_layers=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    )
    
    perceptual_loss.set_targets(content_image, style_image)
    total_loss, loss_breakdown = perceptual_loss(input_image)
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Content loss: {loss_breakdown['content_loss']:.4f}")
    print(f"Style loss: {loss_breakdown['style_loss']:.4f}")
    
    # Test style analyzer
    style_analyzer = StyleAnalyzer(feature_extractor)
    analysis = style_analyzer.analyze_style(style_image)
    print(f"Style analysis keys: {analysis.keys()}")
    
    for layer_name, layer_analysis in analysis.items():
        print(f"{layer_name} - Style complexity: {layer_analysis['style_complexity']:.4f}")
