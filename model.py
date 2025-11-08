# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        return torch.cat([x, self.layers(x)], 1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super().__init__()
        self.block = nn.Sequential(
            *[DenseLayer(in_channels + i * growth_rate, growth_rate) 
              for i in range(num_layers)]
        )
        
    def forward(self, x):
        return self.block(x)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(2)
        )
        
    def forward(self, x):
        return self.layers(x)

class DenseNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, growth_rate=12):
        super().__init__()
        
        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Dense blocks
        num_channels = 64
        
        # Dense Block 1
        self.denseblock1 = DenseBlock(num_channels, num_layers=6, growth_rate=growth_rate)
        num_channels += 6 * growth_rate
        self.transition1 = TransitionLayer(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        # Dense Block 2
        self.denseblock2 = DenseBlock(num_channels, num_layers=8, growth_rate=growth_rate)
        num_channels += 8 * growth_rate
        self.transition2 = TransitionLayer(num_channels, num_channels // 2)
        num_channels = num_channels // 2
        
        # Dense Block 3
        self.denseblock3 = DenseBlock(num_channels, num_layers=8, growth_rate=growth_rate)
        num_channels += 8 * growth_rate
        
        # Final layers
        self.final = nn.Sequential(
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(num_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1 if num_classes == 2 else num_classes)
        )
        
        # Weight initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.denseblock1(x)
        x = self.transition1(x)
        x = self.denseblock2(x)
        x = self.transition2(x)
        x = self.denseblock3(x)
        x = self.final(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1
    
    print("--- Testing DenseNet Model ---")
    model = DenseNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    print("Model Architecture:")
    print(model)
    
    # Test with dummy input
    dummy_input = torch.randn(64, IN_CHANNELS, 28, 28)
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("DenseNet model test successful!")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")