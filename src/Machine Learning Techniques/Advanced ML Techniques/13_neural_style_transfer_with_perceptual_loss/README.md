# Neural Style Transfer with Perceptual Loss

## Overview
Neural Style Transfer with Perceptual Loss represents a paradigm shift in computer vision by enabling the transfer of artistic styles between images using deep neural networks. Unlike traditional image processing techniques that operate at the pixel level, this approach leverages high-level feature representations from pre-trained convolutional networks to understand and transfer artistic styles while preserving the semantic content of images. The use of perceptual losses based on deep features enables more sophisticated, visually appealing, and semantically meaningful style transfers that capture the essence of artistic styles at multiple levels of abstraction.

## Core Concepts

### Perceptual Loss Framework
The core innovation lies in using high-level feature representations instead of pixel-level comparisons:

```
Perceptual Loss = Content Loss + Style Loss + Total Variation Loss
```

**Key Components:**
- **Content Loss**: Preserves semantic content using feature representations
- **Style Loss**: Transfers artistic style using Gram matrices
- **Total Variation Loss**: Ensures spatial coherence and smoothness
- **Feature Extractor**: Pre-trained network (typically VGG) for feature extraction

### Content Loss
Content loss measures the difference between high-level features of content and generated images:

```
L_content = ||F_l(I_c) - F_l(I_g)||²
```

Where:
- `F_l(·)` represents features from layer l of the feature extractor
- `I_c` is the content image
- `I_g` is the generated image
- `||·||²` is the L2 norm

### Style Loss with Gram Matrices
Style loss captures the artistic style using Gram matrices of feature representations:

```
G_l = F_l(I)ᵀF_l(I)
L_style = ||G_l(I_s) - G_l(I_g)||²
```

Where:
- `G_l(·)` is the Gram matrix for layer l
- `I_s` is the style image
- The Gram matrix captures correlations between feature channels

### Multi-Scale Style Transfer
Style transfer operates at multiple scales to capture different aspects of style:

```
L_style_total = Σᵢ w_i L_style_i
```

Where:
- `w_i` are weights for different layers
- `L_style_i` is the style loss for layer i
- Different layers capture different aspects of style (texture, color, structure)

### Feature Extraction
```python
import torch
import torch.nn as nn
import torchvision.models as models

class FeatureExtractor(nn.Module):
    def __init__(self, model_name='vgg19', layers=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']):
        super().__init__()
        self.layers = layers
        
        # Load pre-trained VGG model
        if model_name == 'vgg19':
            vgg = models.vgg19(pretrained=True).features
        elif model_name == 'vgg16':
            vgg = models.vgg16(pretrained=True).features
        
        # Create layer mappings
        self.layer_mappings = {
            'relu1_1': 1, 'relu1_2': 3,
            'relu2_1': 6, 'relu2_2': 8,
            'relu3_1': 11, 'relu3_2': 13, 'relu3_3': 15, 'relu3_4': 17,
            'relu4_1': 20, 'relu4_2': 22, 'relu4_3': 24, 'relu4_4': 26,
            'relu5_1': 29, 'relu5_2': 31, 'relu5_3': 33, 'relu5_4': 35
        }
        
        # Build feature extractor
        self.features = nn.ModuleList()
        current_layer = 0
        
        for layer in vgg:
            if isinstance(layer, nn.ReLU):
                layer = nn.ReLU(inplace=False)
            
            self.features.append(layer)
            current_layer += 1
    
    def forward(self, x):
        features = {}
        current_layer = 0
        
        for layer in self.features:
            x = layer(x)
            current_layer += 1
            
            # Store features for specified layers
            for layer_name, layer_idx in self.layer_mappings.items():
                if current_layer == layer_idx:
                    features[layer_name] = x
        
        return features
```

## Bizarre and Advanced Aspects

### 1. Artistic Consciousness
Neural networks develop an understanding of artistic styles by learning to recognize and transfer complex visual patterns, textures, and color relationships that define artistic movements.

### 2. Perceptual Understanding
The network operates at multiple levels of abstraction, from low-level textures to high-level semantic structures, enabling sophisticated style analysis and transfer.

### 3. Style Decomposition
Artistic styles are broken down into mathematical representations (Gram matrices) that capture the statistical relationships between visual features.

### 4. Content-Style Disentanglement
The system learns to separate content (what is in the image) from style (how it is presented), enabling independent manipulation of these aspects.

### 5. Multi-Scale Style Transfer
Style transfer operates simultaneously at multiple scales, capturing both fine-grained textures and coarse-grained structural patterns.

### 6. Adaptive Style Transfer
Advanced variants can adapt the style transfer process based on the content and style characteristics, leading to more natural and contextually appropriate results.

## Technical Architecture

### Perceptual Loss Implementation
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor, content_layers=['relu4_2'], 
                 style_layers=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
                 content_weight=1.0, style_weight=1e6, tv_weight=1e-3):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        
        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, generated_image, content_image, style_image):
        # Extract features
        gen_features = self.feature_extractor(generated_image)
        content_features = self.feature_extractor(content_image)
        style_features = self.feature_extractor(style_image)
        
        # Compute content loss
        content_loss = 0
        for layer in self.content_layers:
            content_loss += F.mse_loss(gen_features[layer], content_features[layer])
        
        # Compute style loss
        style_loss = 0
        for layer in self.style_layers:
            gen_gram = self._compute_gram_matrix(gen_features[layer])
            style_gram = self._compute_gram_matrix(style_features[layer])
            style_loss += F.mse_loss(gen_gram, style_gram)
        
        # Compute total variation loss
        tv_loss = self._compute_total_variation_loss(generated_image)
        
        # Total loss
        total_loss = (self.content_weight * content_loss + 
                     self.style_weight * style_loss + 
                     self.tv_weight * tv_loss)
        
        return total_loss, {
            'content_loss': content_loss.item(),
            'style_loss': style_loss.item(),
            'tv_loss': tv_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def _compute_gram_matrix(self, features):
        """Compute Gram matrix for style representation"""
        batch_size, channels, height, width = features.size()
        
        # Reshape features to (batch_size, channels, height*width)
        features = features.view(batch_size, channels, height * width)
        
        # Compute Gram matrix: G = F * F^T
        gram_matrix = torch.bmm(features, features.transpose(1, 2))
        
        # Normalize by the number of elements
        gram_matrix = gram_matrix / (channels * height * width)
        
        return gram_matrix
    
    def _compute_total_variation_loss(self, image):
        """Compute total variation loss for spatial coherence"""
        batch_size, channels, height, width = image.size()
        
        # Compute horizontal and vertical differences
        horizontal_diff = image[:, :, :, 1:] - image[:, :, :, :-1]
        vertical_diff = image[:, :, 1:, :] - image[:, :, :-1, :]
        
        # Compute total variation
        tv_loss = torch.sum(torch.abs(horizontal_diff)) + torch.sum(torch.abs(vertical_diff))
        
        return tv_loss

class ContentLoss(nn.Module):
    def __init__(self, feature_extractor, target_features):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.target_features = target_features
        
        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, input_image):
        features = self.feature_extractor(input_image)
        loss = 0
        
        for layer, target_feature in self.target_features.items():
            loss += F.mse_loss(features[layer], target_feature)
        
        return loss

class StyleLoss(nn.Module):
    def __init__(self, feature_extractor, target_grams):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.target_grams = target_grams
        
        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def forward(self, input_image):
        features = self.feature_extractor(input_image)
        loss = 0
        
        for layer, target_gram in self.target_grams.items():
            input_gram = self._compute_gram_matrix(features[layer])
            loss += F.mse_loss(input_gram, target_gram)
        
        return loss
    
    def _compute_gram_matrix(self, features):
        batch_size, channels, height, width = features.size()
        features = features.view(batch_size, channels, height * width)
        gram_matrix = torch.bmm(features, features.transpose(1, 2))
        gram_matrix = gram_matrix / (channels * height * width)
        return gram_matrix
```

### Style Transfer Network
```python
class StyleTransferNetwork(nn.Module):
    def __init__(self, content_weight=1.0, style_weight=1e6, tv_weight=1e-3):
        super().__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Loss functions
        self.perceptual_loss = PerceptualLoss(
            self.feature_extractor,
            content_weight=content_weight,
            style_weight=style_weight,
            tv_weight=tv_weight
        )
    
    def transfer_style(self, content_image, style_image, num_iterations=300, 
                      learning_rate=0.01, device='cuda'):
        """Perform style transfer using iterative optimization"""
        # Move images to device
        content_image = content_image.to(device)
        style_image = style_image.to(device)
        
        # Initialize generated image with content image
        generated_image = content_image.clone().requires_grad_(True)
        
        # Optimizer
        optimizer = torch.optim.LBFGS([generated_image], lr=learning_rate)
        
        # Training loop
        for iteration in range(num_iterations):
            def closure():
                optimizer.zero_grad()
                
                # Compute loss
                loss, loss_dict = self.perceptual_loss(generated_image, content_image, style_image)
                
                # Backward pass
                loss.backward()
                
                return loss
            
            # Update generated image
            optimizer.step(closure)
            
            # Clamp values to valid range
            generated_image.data.clamp_(0, 1)
            
            # Print progress
            if iteration % 50 == 0:
                loss, loss_dict = self.perceptual_loss(generated_image, content_image, style_image)
                print(f'Iteration {iteration}: Loss = {loss_dict["total_loss"]:.4f}')
        
        return generated_image

class FastStyleTransfer(nn.Module):
    def __init__(self, style_image, content_weight=1.0, style_weight=1e6):
        super().__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Pre-compute style features and Gram matrices
        with torch.no_grad():
            style_features = self.feature_extractor(style_image)
            self.style_grams = {}
            for layer in style_features:
                self.style_grams[layer] = self._compute_gram_matrix(style_features[layer])
        
        # Transformer network
        self.transformer = TransformerNetwork()
    
    def forward(self, content_image):
        # Transform content image
        stylized_image = self.transformer(content_image)
        
        # Compute losses
        content_loss = self._compute_content_loss(stylized_image, content_image)
        style_loss = self._compute_style_loss(stylized_image)
        
        total_loss = self.content_weight * content_loss + self.style_weight * style_loss
        
        return stylized_image, total_loss
    
    def _compute_gram_matrix(self, features):
        batch_size, channels, height, width = features.size()
        features = features.view(batch_size, channels, height * width)
        gram_matrix = torch.bmm(features, features.transpose(1, 2))
        gram_matrix = gram_matrix / (channels * height * width)
        return gram_matrix
    
    def _compute_content_loss(self, stylized_image, content_image):
        stylized_features = self.feature_extractor(stylized_image)
        content_features = self.feature_extractor(content_image)
        
        loss = F.mse_loss(stylized_features['relu4_2'], content_features['relu4_2'])
        return loss
    
    def _compute_style_loss(self, stylized_image):
        stylized_features = self.feature_extractor(stylized_image)
        loss = 0
        
        for layer, target_gram in self.style_grams.items():
            stylized_gram = self._compute_gram_matrix(stylized_features[layer])
            loss += F.mse_loss(stylized_gram, target_gram)
        
        return loss

class TransformerNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
        self.in1 = nn.InstanceNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Downsampling
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        
        # Upsampling
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in4 = nn.InstanceNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.in5 = nn.InstanceNorm2d(32)
        self.relu5 = nn.ReLU(inplace=True)
        
        # Output convolution
        self.conv4 = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        # Initial convolution
        y = self.relu1(self.in1(self.conv1(x)))
        
        # Downsampling
        y = self.relu2(self.in2(self.conv2(y)))
        y = self.relu3(self.in3(self.conv3(y)))
        
        # Residual blocks
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        
        # Upsampling
        y = self.relu4(self.in4(self.deconv1(y)))
        y = self.relu5(self.in5(self.deconv2(y)))
        
        # Output
        y = self.tanh(self.conv4(y))
        
        return y

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.relu1(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out
```

## Implementation Details

### Adaptive Instance Normalization (AdaIN)
```python
class AdaIN(nn.Module):
    def __init__(self, content_channels, style_channels):
        super().__init__()
        self.content_channels = content_channels
        self.style_channels = style_channels
        
        # Style projection
        self.style_projection = nn.Linear(style_channels, content_channels * 2)
    
    def forward(self, content, style):
        # Compute mean and variance of content
        content_mean = torch.mean(content, dim=[2, 3], keepdim=True)
        content_var = torch.var(content, dim=[2, 3], keepdim=True, unbiased=False)
        content_std = torch.sqrt(content_var + 1e-8)
        
        # Compute mean and variance of style
        style_mean = torch.mean(style, dim=[2, 3], keepdim=True)
        style_var = torch.var(style, dim=[2, 3], keepdim=True, unbiased=False)
        style_std = torch.sqrt(style_var + 1e-8)
        
        # Normalize content
        content_norm = (content - content_mean) / content_std
        
        # Apply style
        output = content_norm * style_std + style_mean
        
        return output

class AdaINStyleTransfer(nn.Module):
    def __init__(self, content_weight=1.0, style_weight=1e6):
        super().__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # AdaIN layers
        self.adain1 = AdaIN(64, 64)
        self.adain2 = AdaIN(128, 128)
        self.adain3 = AdaIN(256, 256)
        self.adain4 = AdaIN(512, 512)
        
        # Decoder
        self.decoder = Decoder()
    
    def forward(self, content_image, style_image):
        # Extract features
        content_features = self.feature_extractor(content_image)
        style_features = self.feature_extractor(style_image)
        
        # Apply AdaIN
        stylized_features = {}
        stylized_features['relu4_1'] = self.adain4(content_features['relu4_1'], style_features['relu4_1'])
        stylized_features['relu3_1'] = self.adain3(content_features['relu3_1'], style_features['relu3_1'])
        stylized_features['relu2_1'] = self.adain2(content_features['relu2_1'], style_features['relu2_1'])
        stylized_features['relu1_1'] = self.adain1(content_features['relu1_1'], style_features['relu1_1'])
        
        # Decode stylized features
        output = self.decoder(stylized_features)
        
        return output

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Upsampling layers
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        
        # Instance normalization
        self.in1 = nn.InstanceNorm2d(256)
        self.in2 = nn.InstanceNorm2d(128)
        self.in3 = nn.InstanceNorm2d(64)
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
    
    def forward(self, features):
        x = self.relu(self.in1(self.up1(features['relu4_1'])))
        x = self.relu(self.in2(self.up2(x)))
        x = self.relu(self.in3(self.up3(x)))
        x = self.tanh(self.up4(x))
        
        return x
```

### Multi-Style Transfer
```python
class MultiStyleTransfer(nn.Module):
    def __init__(self, style_images, style_weights=None):
        super().__init__()
        self.style_images = style_images
        self.num_styles = len(style_images)
        
        if style_weights is None:
            self.style_weights = torch.ones(self.num_styles) / self.num_styles
        else:
            self.style_weights = torch.tensor(style_weights)
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Pre-compute style features and Gram matrices
        self.style_grams = []
        with torch.no_grad():
            for style_image in style_images:
                style_features = self.feature_extractor(style_image)
                style_gram = {}
                for layer in style_features:
                    style_gram[layer] = self._compute_gram_matrix(style_features[layer])
                self.style_grams.append(style_gram)
    
    def forward(self, content_image):
        # Extract content features
        content_features = self.feature_extractor(content_image)
        
        # Compute multi-style loss
        style_loss = 0
        for i, style_gram in enumerate(self.style_grams):
            weight = self.style_weights[i]
            for layer in style_gram:
                content_gram = self._compute_gram_matrix(content_features[layer])
                style_loss += weight * F.mse_loss(content_gram, style_gram[layer])
        
        return style_loss
    
    def _compute_gram_matrix(self, features):
        batch_size, channels, height, width = features.size()
        features = features.view(batch_size, channels, height * width)
        gram_matrix = torch.bmm(features, features.transpose(1, 2))
        gram_matrix = gram_matrix / (channels * height * width)
        return gram_matrix
```

## Advanced Variants

### 1. StyleGAN-based Style Transfer
```python
class StyleGANStyleTransfer(nn.Module):
    def __init__(self, stylegan_model):
        super().__init__()
        self.stylegan = stylegan_model
        
        # Style mapping network
        self.style_mapping = StyleMappingNetwork()
        
        # Content encoder
        self.content_encoder = ContentEncoder()
    
    def forward(self, content_image, style_image):
        # Encode content
        content_latent = self.content_encoder(content_image)
        
        # Encode style
        style_latent = self.style_mapping(style_image)
        
        # Generate stylized image
        stylized_image = self.stylegan(content_latent, style_latent)
        
        return stylized_image
```

### 2. Neural Doodles
```python
class NeuralDoodles(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.style_transfer = StyleTransferNetwork()
    
    def forward(self, content_image, style_image, user_guidance):
        # Apply user guidance
        guided_content = self._apply_guidance(content_image, user_guidance)
        
        # Perform style transfer
        stylized_image = self.style_transfer.transfer_style(guided_content, style_image)
        
        return stylized_image
    
    def _apply_guidance(self, content_image, user_guidance):
        # Apply user-provided guidance (e.g., color preferences, region-specific styling)
        # This is a simplified implementation
        guided_image = content_image * user_guidance
        return guided_image
```

### 3. Video Style Transfer
```python
class VideoStyleTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.style_transfer = StyleTransferNetwork()
        self.temporal_consistency = TemporalConsistencyLoss()
    
    def forward(self, content_video, style_image):
        batch_size, frames, channels, height, width = content_video.shape
        stylized_frames = []
        
        for frame_idx in range(frames):
            # Style transfer for current frame
            stylized_frame = self.style_transfer.transfer_style(
                content_video[:, frame_idx], style_image
            )
            
            # Apply temporal consistency
            if frame_idx > 0:
                stylized_frame = self._apply_temporal_consistency(
                    stylized_frame, stylized_frames[-1]
                )
            
            stylized_frames.append(stylized_frame)
        
        stylized_video = torch.stack(stylized_frames, dim=1)
        return stylized_video
    
    def _apply_temporal_consistency(self, current_frame, previous_frame):
        # Apply temporal consistency to maintain smooth transitions
        # This is a simplified implementation
        alpha = 0.8
        consistent_frame = alpha * current_frame + (1 - alpha) * previous_frame
        return consistent_frame
```

## Performance Metrics

### 1. Perceptual Quality Metrics
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **FID**: Fréchet Inception Distance
- **SSIM**: Structural Similarity Index
- **PSNR**: Peak Signal-to-Noise Ratio

### 2. Style Transfer Metrics
- **Style accuracy**: How well the target style is transferred
- **Content preservation**: How well the original content is preserved
- **Artistic quality**: Subjective assessment of artistic merit
- **Style consistency**: Consistency of style across the image

### 3. Computational Metrics
- **Training time**: Time required for training convergence
- **Inference speed**: Time per style transfer operation
- **Memory usage**: GPU/CPU memory consumption
- **Model size**: Size of the trained model

## Applications

### 1. Artistic Image Generation
- **Photography stylization**: Converting photographs to artistic styles
- **Art reproduction**: Reproducing famous artistic styles
- **Creative design**: Generating unique artistic interpretations
- **Art education**: Teaching art history and style analysis

### 2. Design and Advertising
- **Brand styling**: Creating brand-specific visual styles
- **Marketing materials**: Generating stylized marketing content
- **Product visualization**: Stylizing product images
- **Creative campaigns**: Creating unique visual campaigns

### 3. Film and Animation
- **Film stylization**: Applying artistic styles to film footage
- **Animation enhancement**: Stylizing animated content
- **Visual effects**: Creating stylized visual effects
- **Post-production**: Enhancing post-production workflows

### 4. Fashion and Photography
- **Fashion photography**: Creating stylized fashion images
- **Portrait stylization**: Stylizing portrait photographs
- **Fashion design**: Inspiring fashion design with artistic styles
- **Editorial content**: Creating stylized editorial content

### 5. Educational Tools
- **Art history**: Teaching art history through style transfer
- **Style analysis**: Analyzing and understanding artistic styles
- **Creative learning**: Encouraging creative expression
- **Art appreciation**: Enhancing art appreciation skills

## Research Frontiers

### 1. Advanced Style Transfer
- **3D style transfer**: Applying styles to 3D content
- **Audio style transfer**: Transferring styles to audio content
- **Text style transfer**: Transferring writing styles
- **Cross-modal style transfer**: Transferring styles between different modalities

### 2. Interactive Style Transfer
- **Real-time interaction**: Real-time style transfer with user interaction
- **User-guided transfer**: User-guided style transfer
- **Interactive editing**: Interactive style editing tools
- **Collaborative creation**: Collaborative style transfer platforms

### 3. Style Understanding
- **Style decomposition**: Breaking down styles into components
- **Style analysis**: Deep analysis of artistic styles
- **Style evolution**: Understanding style evolution over time
- **Style synthesis**: Synthesizing new artistic styles

### 4. Ethical and Social Impact
- **Cultural sensitivity**: Respecting cultural styles and traditions
- **Copyright issues**: Addressing copyright and intellectual property
- **Artistic integrity**: Maintaining artistic integrity
- **Social responsibility**: Responsible use of style transfer technology

## Usage Examples

### Basic Style Transfer
```python
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load images
content_image = Image.open('content.jpg')
style_image = Image.open('style.jpg')

# Preprocess images
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

content_tensor = transform(content_image).unsqueeze(0)
style_tensor = transform(style_image).unsqueeze(0)

# Initialize style transfer network
style_transfer = StyleTransferNetwork(
    content_weight=1.0,
    style_weight=1e6,
    tv_weight=1e-3
)

# Perform style transfer
stylized_image = style_transfer.transfer_style(
    content_tensor, 
    style_tensor, 
    num_iterations=300
)

# Save result
stylized_image = stylized_image.squeeze(0)
stylized_image = transforms.ToPILImage()(stylized_image)
stylized_image.save('stylized_output.jpg')
```

### Fast Style Transfer Training
```python
# Initialize fast style transfer
style_image = torch.randn(1, 3, 256, 256)
fast_transfer = FastStyleTransfer(style_image)

# Training loop
optimizer = torch.optim.Adam(fast_transfer.transformer.parameters(), lr=1e-3)
content_dataset = load_content_dataset()

for epoch in range(100):
    for content_image in content_dataset:
        optimizer.zero_grad()
        
        # Forward pass
        stylized_image, loss = fast_transfer(content_image)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
```

### Multi-Style Transfer
```python
# Load multiple style images
style_images = [
    torch.randn(1, 3, 256, 256),  # Style 1
    torch.randn(1, 3, 256, 256),  # Style 2
    torch.randn(1, 3, 256, 256)   # Style 3
]

# Initialize multi-style transfer
style_weights = [0.5, 0.3, 0.2]  # Weight for each style
multi_transfer = MultiStyleTransfer(style_images, style_weights)

# Perform multi-style transfer
content_image = torch.randn(1, 3, 256, 256)
stylized_image = multi_transfer(content_image)
```

### AdaIN Style Transfer
```python
# Initialize AdaIN style transfer
adain_transfer = AdaINStyleTransfer()

# Perform AdaIN style transfer
content_image = torch.randn(1, 3, 256, 256)
style_image = torch.randn(1, 3, 256, 256)

stylized_image = adain_transfer(content_image, style_image)
```

### Video Style Transfer
```python
# Initialize video style transfer
video_transfer = VideoStyleTransfer()

# Load video (batch_size, frames, channels, height, width)
content_video = torch.randn(1, 30, 3, 256, 256)  # 30 frames
style_image = torch.randn(1, 3, 256, 256)

# Perform video style transfer
stylized_video = video_transfer(content_video, style_image)

print(f"Stylized video shape: {stylized_video.shape}")
```

### Style Analysis
```python
def analyze_style(style_image):
    """Analyze the characteristics of a style image"""
    feature_extractor = FeatureExtractor()
    features = feature_extractor(style_image)
    
    style_analysis = {}
    
    for layer, feature in features.items():
        # Compute Gram matrix
        gram_matrix = compute_gram_matrix(feature)
        
        # Analyze style characteristics
        style_analysis[layer] = {
            'feature_mean': torch.mean(feature).item(),
            'feature_std': torch.std(feature).item(),
            'gram_matrix_norm': torch.norm(gram_matrix).item(),
            'feature_channels': feature.shape[1]
        }
    
    return style_analysis

# Analyze style
style_image = torch.randn(1, 3, 256, 256)
analysis = analyze_style(style_image)

for layer, stats in analysis.items():
    print(f"{layer}:")
    print(f"  Feature mean: {stats['feature_mean']:.4f}")
    print(f"  Feature std: {stats['feature_std']:.4f}")
    print(f"  Gram matrix norm: {stats['gram_matrix_norm']:.4f}")
```

## Files in this Directory
- `perceptual_loss.py`: Core perceptual loss implementation
- `style_transfer.py`: Style transfer network and training
- `gram_matrices.py`: Gram matrix computations and analysis
- `example_usage.py`: Working examples
