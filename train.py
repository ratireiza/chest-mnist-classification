# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import DenseNet
import matplotlib.pyplot as plt
from utils import plot_training_history, visualize_random_val_predictions

class ChestXrayTrainer:
    def __init__(self):
        # Hyperparameters
        self.epochs = 15
        self.batch_size = 32
        self.base_lr = 0.001
        self.max_lr = 0.01
        self.weight_decay = 1e-5
        self.momentum = 0.9
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'lr': []
        }
        
        # Early stopping parameters
        self.patience = 5
        self.best_val_loss = float('inf')
        
    def setup(self):
        # Load data
        self.train_loader, self.val_loader, self.num_classes, self.in_channels = \
            get_data_loaders(self.batch_size)
        
        # Initialize model
        self.model = DenseNet(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            growth_rate=12
        ).to(self.device)
        
        # Loss function with class weights
        pos_weight = torch.tensor([2.0]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.base_lr,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.max_lr,
            epochs=self.epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4
        )
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.float().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Statistics
            running_loss += loss.item()
            predicted = (outputs > 0).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        return {
            'loss': running_loss / len(self.train_loader),
            'acc': 100 * correct / total
        }
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.float().to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                predicted = (outputs > 0).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return {
            'loss': running_loss / len(self.val_loader),
            'acc': 100 * correct / total
        }
    
    def train(self):
        self.setup()
        print(f"Training on device: {self.device}")
        print(self.model)
        
        try:
            for epoch in range(self.epochs):
                # Training phase
                train_metrics = self.train_epoch()
                
                # Validation phase
                val_metrics = self.validate()
                
                # Update history
                self.history['train_loss'].append(train_metrics['loss'])
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['train_acc'].append(train_metrics['acc'])
                self.history['val_acc'].append(val_metrics['acc'])
                self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
                
                # Print metrics
                print(
                    f"Epoch [{epoch+1}/{self.epochs}] | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['acc']:.2f}% | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['acc']:.2f}% | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
                
                # Early stopping check
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_metrics['loss'],
                        'history': self.history
                    }, 'best_model.pth')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"\nEarly stopping triggered after epoch {epoch+1}")
                        break
                        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        print("Training completed!")
        
        # Plot training history
        plot_training_history(
            self.history['train_loss'],
            self.history['val_loss'],
            self.history['train_acc'],
            self.history['val_acc']
        )
        
        # Visualize predictions
        visualize_random_val_predictions(
            self.model,
            self.val_loader,
            self.num_classes,
            count=10
        )

if __name__ == '__main__':
    trainer = ChestXrayTrainer()
    trainer.train()
