#!/usr/bin/env python3
"""
Machine Learning Training Progress Tracker

A comprehensive tool for tracking and visualizing machine learning model training progress.
Supports multiple ML frameworks, real-time monitoring, and detailed analytics.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import threading
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn
    import torch.optim
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. PyTorch-specific features disabled.")

TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow not available. TensorFlow-specific features disabled.")

@dataclass
class TrainingMetrics:
    """Data class to store training metrics"""
    epoch: int
    step: int
    loss: float
    accuracy: Optional[float] = None
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    timestamp: datetime = None
    additional_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.additional_metrics is None:
            self.additional_metrics = {}

@dataclass
class TrainingConfig:
    """Configuration for training tracking"""
    experiment_name: str
    model_name: str = "unknown"
    framework: str = "pytorch"
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    save_frequency: int = 100
    max_checkpoints: int = 5
    early_stopping_patience: int = 10
    metric_history_size: int = 1000
    enable_tensorboard: bool = True
    enable_wandb: bool = False
    wandb_project: str = ""
    wandb_entity: str = ""

class MetricsStorage:
    """Thread-safe storage for training metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.best_metrics: Dict[str, float] = {}
        self.lock = threading.Lock()
        
    def add_metric(self, metric: TrainingMetrics):
        """Add a new metric to the history"""
        with self.lock:
            self.metrics_history.append(metric)
            self._update_best_metrics(metric)
    
    def _update_best_metrics(self, metric: TrainingMetrics):
        """Update best metrics seen so far"""
        if metric.loss is not None:
            current_best = self.best_metrics.get('best_loss', float('inf'))
            if metric.loss < current_best:
                self.best_metrics['best_loss'] = metric.loss
                self.best_metrics['best_loss_epoch'] = metric.epoch
                self.best_metrics['best_loss_step'] = metric.step
        
        if metric.accuracy is not None:
            current_best = self.best_metrics.get('best_accuracy', 0.0)
            if metric.accuracy > current_best:
                self.best_metrics['best_accuracy'] = metric.accuracy
                self.best_metrics['best_accuracy_epoch'] = metric.epoch
                self.best_metrics['best_accuracy_step'] = metric.step
    
    def get_recent_metrics(self, n: int = 10) -> List[TrainingMetrics]:
        """Get the n most recent metrics"""
        with self.lock:
            return list(self.metrics_history)[-n:]
    
    def get_metrics_df(self) -> pd.DataFrame:
        """Get all metrics as a pandas DataFrame"""
        with self.lock:
            if not self.metrics_history:
                return pd.DataFrame()
            
            data = []
            for metric in self.metrics_history:
                row = {
                    'epoch': metric.epoch,
                    'step': metric.step,
                    'loss': metric.loss,
                    'accuracy': metric.accuracy,
                    'val_loss': metric.val_loss,
                    'val_accuracy': metric.val_accuracy,
                    'learning_rate': metric.learning_rate,
                    'timestamp': metric.timestamp
                }
                row.update(metric.additional_metrics)
                data.append(row)
            
            return pd.DataFrame(data)

class BaseCallback(ABC):
    """Base class for training callbacks"""
    
    @abstractmethod
    def on_epoch_end(self, epoch: int, metrics: TrainingMetrics):
        """Called at the end of each epoch"""
        pass
    
    @abstractmethod
    def on_training_end(self, final_metrics: TrainingMetrics):
        """Called when training ends"""
        pass

class EarlyStoppingCallback(BaseCallback):
    """Early stopping callback"""
    
    def __init__(self, patience: int = 10, monitor: str = 'val_loss', mode: str = 'min'):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.should_stop = False
    
    def on_epoch_end(self, epoch: int, metrics: TrainingMetrics):
        current_value = getattr(metrics, self.monitor, None)
        if current_value is None:
            return
        
        if self.mode == 'min':
            improved = current_value < self.best_value
        else:
            improved = current_value > self.best_value
        
        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {epoch} epochs")
    
    def on_training_end(self, final_metrics: TrainingMetrics):
        pass

class ModelCheckpointCallback(BaseCallback):
    """Model checkpoint callback"""
    
    def __init__(self, checkpoint_dir: str, save_frequency: int = 100, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_frequency = save_frequency
        self.max_checkpoints = max_checkpoints
        self.saved_checkpoints = []
    
    def on_epoch_end(self, epoch: int, metrics: TrainingMetrics):
        if epoch % self.save_frequency == 0:
            self._save_checkpoint(epoch, metrics)
    
    def _save_checkpoint(self, epoch: int, metrics: TrainingMetrics):
        """Save model checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch_{epoch}_{timestamp}.pth"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Save checkpoint metadata
        checkpoint_data = {
            'epoch': epoch,
            'metrics': asdict(metrics),
            'timestamp': timestamp
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        self.saved_checkpoints.append(checkpoint_path)
        
        # Remove old checkpoints
        if len(self.saved_checkpoints) > self.max_checkpoints:
            old_checkpoint = self.saved_checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def on_training_end(self, final_metrics: TrainingMetrics):
        pass

class TrainingTracker:
    """Main training tracker class"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.metrics_storage = MetricsStorage(config.metric_history_size)
        self.callbacks: List[BaseCallback] = []
        self.training_start_time: Optional[datetime] = None
        self.training_end_time: Optional[datetime] = None
        
        # Setup directories
        self.log_dir = Path(config.log_dir)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize callbacks
        self._initialize_callbacks()
        
        # Setup external integrations
        self._setup_external_integrations()
        
        logger.info(f"TrainingTracker initialized for experiment: {config.experiment_name}")
    
    def _setup_logging(self):
        """Setup file logging"""
        log_file = self.log_dir / f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def _initialize_callbacks(self):
        """Initialize training callbacks"""
        if self.config.early_stopping_patience > 0:
            self.add_callback(EarlyStoppingCallback(self.config.early_stopping_patience))
        
        self.add_callback(ModelCheckpointCallback(
            self.config.checkpoint_dir,
            self.config.save_frequency,
            self.config.max_checkpoints
        ))
    
    def _setup_external_integrations(self):
        """Setup external integrations like TensorBoard and W&B"""
        if self.config.enable_tensorboard and TORCH_AVAILABLE:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tensorboard_writer = SummaryWriter(
                    log_dir=self.log_dir / 'tensorboard'
                )
                logger.info("TensorBoard integration enabled")
            except ImportError:
                logger.warning("TensorBoard not available")
                self.tensorboard_writer = None
        else:
            self.tensorboard_writer = None
        
        if self.config.enable_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    name=self.config.experiment_name
                )
                self.wandb_available = True
                logger.info("Weights & Biases integration enabled")
            except ImportError:
                logger.warning("wandb not available")
                self.wandb_available = False
        else:
            self.wandb_available = False
    
    def add_callback(self, callback: BaseCallback):
        """Add a training callback"""
        self.callbacks.append(callback)
    
    def start_training(self):
        """Mark the start of training"""
        self.training_start_time = datetime.now()
        logger.info(f"Training started: {self.training_start_time}")
        
        # Save initial configuration
        config_path = self.log_dir / f"{self.config.experiment_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
    
    def log_metrics(self, metrics: TrainingMetrics):
        """Log training metrics"""
        self.metrics_storage.add_metric(metrics)
        
        # Log to console/file
        accuracy_str = f"{metrics.accuracy:.4f}" if metrics.accuracy is not None else "N/A"
        logger.info(f"Epoch {metrics.epoch}, Step {metrics.step}: "
                   f"loss={metrics.loss:.4f}, "
                   f"accuracy={accuracy_str}")
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('Loss/train', metrics.loss, metrics.step)
            if metrics.accuracy:
                self.tensorboard_writer.add_scalar('Accuracy/train', metrics.accuracy, metrics.step)
            if metrics.val_loss:
                self.tensorboard_writer.add_scalar('Loss/val', metrics.val_loss, metrics.step)
            if metrics.val_accuracy:
                self.tensorboard_writer.add_scalar('Accuracy/val', metrics.val_accuracy, metrics.step)
            if metrics.learning_rate:
                self.tensorboard_writer.add_scalar('Learning_Rate', metrics.learning_rate, metrics.step)
        
        # Log to W&B
        if self.wandb_available:
            import wandb
            log_dict = {
                'train/loss': metrics.loss,
                'epoch': metrics.epoch,
                'step': metrics.step
            }
            if metrics.accuracy:
                log_dict['train/accuracy'] = metrics.accuracy
            if metrics.val_loss:
                log_dict['val/loss'] = metrics.val_loss
            if metrics.val_accuracy:
                log_dict['val/accuracy'] = metrics.val_accuracy
            if metrics.learning_rate:
                log_dict['learning_rate'] = metrics.learning_rate
            log_dict.update({f'metrics/{k}': v for k, v in metrics.additional_metrics.items()})
            wandb.log(log_dict)
    
    def on_epoch_end(self, epoch: int, metrics: TrainingMetrics):
        """Called at the end of each epoch"""
        self.log_metrics(metrics)
        
        # Call callbacks
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics)
    
    def end_training(self):
        """Mark the end of training"""
        self.training_end_time = datetime.now()
        
        if self.training_start_time:
            duration = self.training_end_time - self.training_start_time
            logger.info(f"Training completed in {duration}")
        
        # Get final metrics
        recent_metrics = self.metrics_storage.get_recent_metrics(1)
        final_metrics = recent_metrics[0] if recent_metrics else None
        
        # Call callbacks
        for callback in self.callbacks:
            callback.on_training_end(final_metrics)
        
        # Close external integrations
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        if self.wandb_available:
            import wandb
            wandb.finish()
        
        logger.info("Training ended")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of the training process"""
        metrics_df = self.metrics_storage.get_metrics_df()
        best_metrics = self.metrics_storage.best_metrics
        
        summary = {
            'experiment_name': self.config.experiment_name,
            'model_name': self.config.model_name,
            'framework': self.config.framework,
            'total_epochs': metrics_df['epoch'].max() if not metrics_df.empty else 0,
            'total_steps': len(metrics_df),
            'best_metrics': best_metrics,
            'training_duration': None
        }
        
        if self.training_start_time and self.training_end_time:
            duration = self.training_end_time - self.training_start_time
            summary['training_duration'] = str(duration)
        
        return summary
    
    def save_training_report(self, filepath: Optional[str] = None):
        """Save a comprehensive training report"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.log_dir / f"{self.config.experiment_name}_report_{timestamp}.json"
        
        summary = self.get_training_summary()
        metrics_df = self.metrics_storage.get_metrics_df()
        
        # Generate plots
        plots_dir = self.log_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        self._generate_plots(plots_dir)
        
        report = {
            'summary': summary,
            'metrics_data': metrics_df.to_dict('records') if not metrics_df.empty else [],
            'config': asdict(self.config),
            'plots_directory': str(plots_dir)
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Training report saved to {filepath}")
        return filepath
    
    def _generate_plots(self, plots_dir: Path):
        """Generate training plots"""
        metrics_df = self.metrics_storage.get_metrics_df()
        if metrics_df.empty:
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Loss plot
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(metrics_df['step'], metrics_df['loss'], label='Training Loss', alpha=0.8)
        if 'val_loss' in metrics_df.columns and not metrics_df['val_loss'].isna().all():
            plt.plot(metrics_df['step'], metrics_df['val_loss'], label='Validation Loss', alpha=0.8)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy plot
        plt.subplot(2, 2, 2)
        if 'accuracy' in metrics_df.columns and not metrics_df['accuracy'].isna().all():
            plt.plot(metrics_df['step'], metrics_df['accuracy'], label='Training Accuracy', alpha=0.8)
        if 'val_accuracy' in metrics_df.columns and not metrics_df['val_accuracy'].isna().all():
            plt.plot(metrics_df['step'], metrics_df['val_accuracy'], label='Validation Accuracy', alpha=0.8)
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate plot
        plt.subplot(2, 2, 3)
        if 'learning_rate' in metrics_df.columns and not metrics_df['learning_rate'].isna().all():
            plt.plot(metrics_df['step'], metrics_df['learning_rate'], label='Learning Rate', alpha=0.8)
            plt.xlabel('Step')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Loss distribution
        plt.subplot(2, 2, 4)
        plt.hist(metrics_df['loss'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Loss')
        plt.ylabel('Frequency')
        plt.title('Loss Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'training_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Training vs validation correlation
        if ('accuracy' in metrics_df.columns and 'val_accuracy' in metrics_df.columns and 
            not metrics_df[['accuracy', 'val_accuracy']].isna().any(axis=1).all()):
            
            plt.figure(figsize=(10, 6))
            valid_data = metrics_df[['accuracy', 'val_accuracy']].dropna()
            plt.scatter(valid_data['accuracy'], valid_data['val_accuracy'], alpha=0.6)
            plt.xlabel('Training Accuracy')
            plt.ylabel('Validation Accuracy')
            plt.title('Training vs Validation Accuracy Correlation')
            plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Correlation')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(plots_dir / 'accuracy_correlation.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Training plots saved to {plots_dir}")

# PyTorch-specific integration
if TORCH_AVAILABLE:
    class PyTorchTracker:
        """PyTorch-specific training tracker integration"""
        
        def __init__(self, tracker: TrainingTracker):
            self.tracker = tracker
            self.step_count = 0
        
        def create_tracker_hook(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
            """Create a hook for PyTorch training loop"""
            def hook(module, input, output):
                self.step_count += 1
                
                # Get current learning rate
                lr = optimizer.param_groups[0]['lr']
                
                # Create metrics (this would be updated with actual loss/accuracy)
                metrics = TrainingMetrics(
                    epoch=0,  # Should be updated by user
                    step=self.step_count,
                    loss=0.0,  # Should be updated by user
                    learning_rate=lr
                )
                
                self.tracker.log_metrics(metrics)
            
            return hook
else:
    class PyTorchTracker:
        def __init__(self, tracker):
            raise ImportError("PyTorch not available. Please install PyTorch to use PyTorchTracker.")

# TensorFlow-specific integration
if TENSORFLOW_AVAILABLE:
    class TensorFlowTracker:
        """TensorFlow-specific training tracker integration"""
        
        def __init__(self, tracker: TrainingTracker):
            self.tracker = tracker
        
        def create_keras_callback(self):
            """Create a Keras callback for training tracking"""
            class TrackingCallback(tf.keras.callbacks.Callback):
                def __init__(self, tracker: TrainingTracker):
                    super().__init__()
                    self.tracker = tracker
                    self.epoch = 0
                
                def on_epoch_end(self, epoch, logs=None):
                    self.epoch = epoch
                    logs = logs or {}
                    
                    metrics = TrainingMetrics(
                        epoch=epoch,
                        step=epoch,  # Simplified for Keras
                        loss=logs.get('loss', 0.0),
                        accuracy=logs.get('accuracy'),
                        val_loss=logs.get('val_loss'),
                        val_accuracy=logs.get('val_accuracy')
                    )
                    
                    self.tracker.on_epoch_end(epoch, metrics)
            
            return TrackingCallback(self.tracker)
else:
    class TensorFlowTracker:
        def __init__(self, tracker):
            raise ImportError("TensorFlow not available. Please install TensorFlow to use TensorFlowTracker.")

# Utility functions
def create_training_tracker(
    experiment_name: str,
    model_name: str = "unknown",
    framework: str = "pytorch",
    **kwargs
) -> TrainingTracker:
    """Create a training tracker with default configuration"""
    config = TrainingConfig(
        experiment_name=experiment_name,
        model_name=model_name,
        framework=framework,
        **kwargs
    )
    return TrainingTracker(config)

def plot_training_comparison(
    tracker1: TrainingTracker,
    tracker2: TrainingTracker,
    save_path: str = None
):
    """Compare training progress between two trackers"""
    metrics1 = tracker1.metrics_storage.get_metrics_df()
    metrics2 = tracker2.metrics_storage.get_metrics_df()
    
    plt.figure(figsize=(15, 10))
    
    # Loss comparison
    plt.subplot(2, 3, 1)
    plt.plot(metrics1['step'], metrics1['loss'], label=f"{tracker1.config.experiment_name}", alpha=0.8)
    plt.plot(metrics2['step'], metrics2['loss'], label=f"{tracker2.config.experiment_name}", alpha=0.8)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy comparison
    if 'accuracy' in metrics1.columns and 'accuracy' in metrics2.columns:
        plt.subplot(2, 3, 2)
        plt.plot(metrics1['step'], metrics1['accuracy'], label=f"{tracker1.config.experiment_name}", alpha=0.8)
        plt.plot(metrics2['step'], metrics2['accuracy'], label=f"{tracker2.config.experiment_name}", alpha=0.8)
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Training time comparison
    plt.subplot(2, 3, 3)
    summary1 = tracker1.get_training_summary()
    summary2 = tracker2.get_training_summary()
    
    experiments = [summary1['experiment_name'], summary2['experiment_name']]
    durations = [
        summary1.get('training_duration', '0:00:00'),
        summary2.get('training_duration', '0:00:00')
    ]
    
    plt.bar(experiments, [str(d).count(':') for d in durations])  # Simplified visualization
    plt.title('Training Duration Comparison')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt

if __name__ == "__main__":
    # Example usage
    print("ML Training Tracker - Example Usage")
    print("=" * 50)
    
    # Create a tracker
    tracker = create_training_tracker(
        experiment_name="example_experiment",
        model_name="simple_cnn",
        framework="pytorch",
        enable_tensorboard=True,
        enable_wandb=False
    )
    
    # Start training
    tracker.start_training()
    
    # Simulate training loop
    print("Simulating training process...")
    for epoch in range(5):
        for step in range(10):
            # Simulate metrics
            loss = np.random.exponential(0.1) + 0.05 * (5 - epoch)
            accuracy = min(0.95, 0.5 + 0.1 * epoch + np.random.normal(0, 0.05))
            
            metrics = TrainingMetrics(
                epoch=epoch,
                step=epoch * 10 + step,
                loss=loss,
                accuracy=accuracy,
                learning_rate=0.001 * (0.9 ** epoch)
            )
            
            tracker.log_metrics(metrics)
            time.sleep(0.1)  # Simulate training time
        
        # End of epoch
        tracker.on_epoch_end(epoch, metrics)
    
    # End training
    tracker.end_training()
    
    # Save report
    report_path = tracker.save_training_report()
    
    # Print summary
    summary = tracker.get_training_summary()
    print(f"\nTraining Summary:")
    print(f"Experiment: {summary['experiment_name']}")
    print(f"Total Epochs: {summary['total_epochs']}")
    print(f"Total Steps: {summary['total_steps']}")
    print(f"Best Loss: {summary['best_metrics'].get('best_loss', 'N/A')}")
    print(f"Best Accuracy: {summary['best_metrics'].get('best_accuracy', 'N/A')}")
    print(f"Report saved to: {report_path}")
    
    print("\nTraining tracker example completed successfully!")