#!/usr/bin/env python3
"""
ML Training Tracker - Usage Examples

This file demonstrates how to use the ML Training Tracker with different
machine learning frameworks and scenarios.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Add the scripts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from ml_training_tracker import (
    TrainingTracker, TrainingConfig, TrainingMetrics,
    create_training_tracker, PyTorchTracker, TensorFlowTracker
)

def create_sample_data(n_samples=1000, n_features=20, n_classes=2):
    """Create sample classification data"""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_redundant=0,
        n_informative=n_features,
        random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)

class SimplePyTorchModel(nn.Module):
    """Simple PyTorch model for demonstration"""
    def __init__(self, input_size, hidden_size=64, num_classes=2):
        super(SimplePyTorchModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

def pytorch_training_example():
    """Example of using the tracker with PyTorch"""
    print("=" * 60)
    print("PyTorch Training Example")
    print("=" * 60)
    
    # Create training tracker
    tracker = create_training_tracker(
        experiment_name="pytorch_classification",
        model_name="simple_fnn",
        framework="pytorch",
        enable_tensorboard=True,
        enable_wandb=False,
        early_stopping_patience=5
    )
    
    # Create sample data
    X_train, X_val, y_train, y_val = create_sample_data()
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = SimplePyTorchModel(input_size=X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Start training
    tracker.start_training()
    
    num_epochs = 10
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Log metrics every 10 batches
            if batch_idx % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                metrics = TrainingMetrics(
                    epoch=epoch + 1,
                    step=epoch * len(train_loader) + batch_idx,
                    loss=loss.item(),
                    accuracy=train_correct / train_total if train_total > 0 else 0.0,
                    learning_rate=current_lr
                )
                tracker.log_metrics(metrics)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0
        
        # Create epoch metrics
        epoch_metrics = TrainingMetrics(
            epoch=epoch + 1,
            step=(epoch + 1) * len(train_loader),
            loss=avg_train_loss,
            accuracy=train_accuracy,
            val_loss=avg_val_loss,
            val_accuracy=val_accuracy,
            learning_rate=optimizer.param_groups[0]['lr']
        )
        
        # Call epoch end
        tracker.on_epoch_end(epoch + 1, epoch_metrics)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= tracker.config.early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # End training
    tracker.end_training()
    
    # Save report
    report_path = tracker.save_training_report()
    
    # Print summary
    summary = tracker.get_training_summary()
    print(f"\nPyTorch Training Summary:")
    print(f"Experiment: {summary['experiment_name']}")
    print(f"Total Epochs: {summary['total_epochs']}")
    print(f"Best Validation Loss: {summary['best_metrics'].get('best_loss', 'N/A'):.4f}")
    print(f"Report saved to: {report_path}")
    
    return tracker

def tensorflow_training_example():
    """Example of using the tracker with TensorFlow/Keras"""
    print("\n" + "=" * 60)
    print("TensorFlow Training Example")
    print("=" * 60)
    
    # Create training tracker
    tracker = create_training_tracker(
        experiment_name="tensorflow_classification",
        model_name="simple_dense",
        framework="tensorflow",
        enable_tensorboard=True,
        enable_wandb=False
    )
    
    # Create TensorFlow tracker
    tf_tracker = TensorFlowTracker(tracker)
    keras_callback = tf_tracker.create_keras_callback()
    
    # Create sample data
    X_train, X_val, y_train, y_val = create_sample_data()
    
    # Create and compile model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Start training
    tracker.start_training()
    
    # Train model with our custom callback
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        callbacks=[keras_callback],
        verbose=1
    )
    
    # End training
    tracker.end_training()
    
    # Save report
    report_path = tracker.save_training_report()
    
    # Print summary
    summary = tracker.get_training_summary()
    print(f"\nTensorFlow Training Summary:")
    print(f"Experiment: {summary['experiment_name']}")
    print(f"Total Epochs: {summary['total_epochs']}")
    print(f"Report saved to: {report_path}")
    
    return tracker

def custom_metrics_example():
    """Example showing custom metrics and advanced features"""
    print("\n" + "=" * 60)
    print("Custom Metrics Example")
    print("=" * 60)
    
    # Create training tracker with custom configuration
    config = TrainingConfig(
        experiment_name="custom_metrics_experiment",
        model_name="advanced_model",
        framework="pytorch",
        log_dir="./custom_logs",
        checkpoint_dir="./custom_checkpoints",
        save_frequency=50,
        max_checkpoints=3,
        early_stopping_patience=8,
        enable_tensorboard=True,
        enable_wandb=False
    )
    
    tracker = TrainingTracker(config)
    
    # Start training
    tracker.start_training()
    
    # Simulate training with custom metrics
    num_epochs = 15
    steps_per_epoch = 20
    
    for epoch in range(num_epochs):
        for step in range(steps_per_epoch):
            current_step = epoch * steps_per_epoch + step
            
            # Simulate various metrics
            loss = np.random.exponential(0.1) * (1 - epoch * 0.05)
            accuracy = min(0.95, 0.3 + epoch * 0.05 + np.random.normal(0, 0.02))
            f1_score = accuracy * 0.95 + np.random.normal(0, 0.01)
            precision = accuracy * 0.98 + np.random.normal(0, 0.01)
            recall = accuracy * 0.92 + np.random.normal(0, 0.01)
            
            # Create metrics with additional custom metrics
            metrics = TrainingMetrics(
                epoch=epoch + 1,
                step=current_step,
                loss=loss,
                accuracy=accuracy,
                learning_rate=0.001 * (0.95 ** epoch)
            )
            
            # Add custom metrics
            metrics.additional_metrics = {
                'f1_score': f1_score,
                'precision': precision,
                'recall': recall,
                'gradient_norm': np.random.exponential(0.5),
                'parameter_norm': 100 + epoch * 5 + np.random.normal(0, 2)
            }
            
            tracker.log_metrics(metrics)
            
            # Simulate some training time
            time.sleep(0.05)
        
        # End of epoch processing
        epoch_metrics = TrainingMetrics(
            epoch=epoch + 1,
            step=(epoch + 1) * steps_per_epoch,
            loss=loss,
            accuracy=accuracy,
            learning_rate=0.001 * (0.95 ** epoch)
        )
        epoch_metrics.additional_metrics = {
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall,
            'epoch_time': np.random.exponential(30)  # Simulated epoch time
        }
        
        tracker.on_epoch_end(epoch + 1, epoch_metrics)
        
        print(f"Epoch {epoch + 1}/{num_epochs} completed")
        print(f"  Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1_score:.4f}")
    
    # End training
    tracker.end_training()
    
    # Save comprehensive report
    report_path = tracker.save_training_report()
    
    # Print summary
    summary = tracker.get_training_summary()
    print(f"\nCustom Metrics Training Summary:")
    print(f"Experiment: {summary['experiment_name']}")
    print(f"Total Epochs: {summary['total_epochs']}")
    print(f"Best Loss: {summary['best_metrics'].get('best_loss', 'N/A'):.4f}")
    print(f"Best Accuracy: {summary['best_metrics'].get('best_accuracy', 'N/A'):.4f}")
    print(f"Report saved to: {report_path}")
    
    return tracker

def multi_experiment_comparison():
    """Example comparing multiple experiments"""
    print("\n" + "=" * 60)
    print("Multi-Experiment Comparison")
    print("=" * 60)
    
    # Create multiple trackers with different configurations
    trackers = []
    
    # Experiment 1: High learning rate
    tracker1 = create_training_tracker(
        experiment_name="high_lr_experiment",
        model_name="test_model",
        framework="pytorch"
    )
    trackers.append(tracker1)
    
    # Experiment 2: Low learning rate
    tracker2 = create_training_tracker(
        experiment_name="low_lr_experiment",
        model_name="test_model",
        framework="pytorch"
    )
    trackers.append(tracker2)
    
    # Simulate different training scenarios
    for i, tracker in enumerate(trackers):
        tracker.start_training()
        
        # Different learning rates for each experiment
        base_lr = 0.01 if i == 0 else 0.001
        
        for epoch in range(8):
            for step in range(10):
                current_step = epoch * 10 + step
                
                # Different loss curves based on learning rate
                if i == 0:  # High LR
                    loss = np.random.exponential(0.15) * (1 - epoch * 0.08)
                    accuracy = min(0.90, 0.4 + epoch * 0.06 + np.random.normal(0, 0.03))
                else:  # Low LR
                    loss = np.random.exponential(0.08) * (1 - epoch * 0.03)
                    accuracy = min(0.95, 0.3 + epoch * 0.08 + np.random.normal(0, 0.02))
                
                metrics = TrainingMetrics(
                    epoch=epoch + 1,
                    step=current_step,
                    loss=loss,
                    accuracy=accuracy,
                    learning_rate=base_lr * (0.9 ** epoch)
                )
                
                tracker.log_metrics(metrics)
            
            # End of epoch
            epoch_metrics = TrainingMetrics(
                epoch=epoch + 1,
                step=(epoch + 1) * 10,
                loss=loss,
                accuracy=accuracy,
                learning_rate=base_lr * (0.9 ** epoch)
            )
            tracker.on_epoch_end(epoch + 1, epoch_metrics)
        
        tracker.end_training()
        report_path = tracker.save_training_report()
        print(f"Experiment {tracker.config.experiment_name} completed. Report: {report_path}")
    
    # Compare the experiments
    from ml_training_tracker import plot_training_comparison
    
    comparison_path = "./logs/experiment_comparison.png"
    plot_training_comparison(trackers[0], trackers[1], save_path=comparison_path)
    print(f"Comparison plot saved to: {comparison_path}")
    
    return trackers

def main():
    """Run all examples"""
    print("ML Training Tracker - Comprehensive Examples")
    print("=" * 60)
    print("This script demonstrates various ways to use the ML Training Tracker")
    print("=" * 60)
    
    # Run examples based on user choice
    examples = {
        '1': ('PyTorch Training', pytorch_training_example),
        '2': ('TensorFlow Training', tensorflow_training_example),
        '3': ('Custom Metrics', custom_metrics_example),
        '4': ('Multi-Experiment Comparison', multi_experiment_comparison),
        '5': ('All Examples', lambda: [pytorch_training_example(), 
                                       tensorflow_training_example(), 
                                       custom_metrics_example(), 
                                       multi_experiment_comparison()])
    }
    
    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"{key}. {name}")
    
    choice = input("\nSelect an example to run (1-5) [default: 1]: ").strip() or '1'
    
    if choice in examples:
        print(f"\nRunning: {examples[choice][0]}")
        examples[choice][1]()
    else:
        print("Invalid choice. Running PyTorch example by default.")
        pytorch_training_example()
    
    print("\nExamples completed. Check the logs directory for detailed reports and plots.")

if __name__ == "__main__":
    main()