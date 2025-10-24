# ML Training Tracker

A comprehensive Python library for tracking and visualizing machine learning model training progress. This tool provides real-time monitoring, detailed analytics, and integration with popular ML frameworks.

## Features

### Core Functionality
- **Real-time Training Monitoring**: Track loss, accuracy, and custom metrics during training
- **Multi-Framework Support**: Works with PyTorch, TensorFlow/Keras, and custom training loops
- **Advanced Analytics**: Comprehensive training analysis and visualization
- **Checkpoint Management**: Automatic model checkpointing with configurable retention
- **Early Stopping**: Built-in early stopping with customizable patience

### Logging and Integration
- **TensorBoard Integration**: Automatic logging to TensorBoard
- **Weights & Biases**: Optional W&B integration for experiment tracking
- **File Logging**: Comprehensive logging with rotation and management
- **JSON Reports**: Detailed training reports in JSON format

### Visualization
- **Training Plots**: Automatic generation of training curves
- **Metric Correlation**: Analysis of metric relationships
- **Comparison Tools**: Compare multiple training runs
- **Custom Plots**: Support for custom visualization requirements

## Installation

### Basic Installation
```bash
pip install numpy pandas matplotlib seaborn
```

### Full Installation with ML Frameworks
```bash
# Install with PyTorch support
pip install torch torchvision

# Install with TensorFlow support
pip install tensorflow

# Install all optional dependencies
pip install tensorboard wandb scikit-learn plotly
```

### From Requirements File
```bash
pip install -r requirements_ml_tracker.txt
```

## Quick Start

### Basic Usage

```python
from ml_training_tracker import create_training_tracker, TrainingMetrics

# Create a tracker
tracker = create_training_tracker(
    experiment_name="my_experiment",
    model_name="simple_cnn",
    framework="pytorch"
)

# Start training
tracker.start_training()

# Training loop
for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        # Your training code here
        loss = train_step(batch)
        accuracy = calculate_accuracy(batch)
        
        # Log metrics
        metrics = TrainingMetrics(
            epoch=epoch,
            step=step,
            loss=loss,
            accuracy=accuracy
        )
        tracker.log_metrics(metrics)
    
    # End of epoch
    tracker.on_epoch_end(epoch, metrics)

# End training
tracker.end_training()

# Save report
report_path = tracker.save_training_report()
```

### PyTorch Integration

```python
import torch
from ml_training_tracker import create_training_tracker, TrainingMetrics

# Create tracker
tracker = create_training_tracker(
    experiment_name="pytorch_classification",
    model_name="resnet18",
    framework="pytorch",
    enable_tensorboard=True
)

# Your PyTorch training loop
tracker.start_training()

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Log metrics
        if batch_idx % log_interval == 0:
            metrics = TrainingMetrics(
                epoch=epoch,
                step=batch_idx,
                loss=loss.item(),
                learning_rate=optimizer.param_groups[0]['lr']
            )
            tracker.log_metrics(metrics)
    
    # Validation and epoch end
    val_loss, val_accuracy = validate(model, val_loader)
    epoch_metrics = TrainingMetrics(
        epoch=epoch,
        step=len(train_loader) * (epoch + 1),
        loss=loss.item(),
        val_loss=val_loss,
        val_accuracy=val_accuracy
    )
    tracker.on_epoch_end(epoch, epoch_metrics)

tracker.end_training()
```

### TensorFlow/Keras Integration

```python
import tensorflow as tf
from ml_training_tracker import create_training_tracker, TensorFlowTracker

# Create tracker
tracker = create_training_tracker(
    experiment_name="tensorflow_classification",
    model_name="dense_model",
    framework="tensorflow"
)

# Create TensorFlow tracker
tf_tracker = TensorFlowTracker(tracker)
keras_callback = tf_tracker.create_keras_callback()

# Compile and train model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    callbacks=[keras_callback]
)
```

## Advanced Features

### Custom Metrics

```python
from ml_training_tracker import TrainingMetrics

# Create metrics with custom values
metrics = TrainingMetrics(
    epoch=epoch,
    step=step,
    loss=loss_value,
    accuracy=accuracy_value
)

# Add custom metrics
metrics.additional_metrics = {
    'f1_score': f1_value,
    'precision': precision_value,
    'recall': recall_value,
    'gradient_norm': grad_norm,
    'parameter_norm': param_norm
}

tracker.log_metrics(metrics)
```

### Custom Callbacks

```python
from ml_training_tracker import BaseCallback

class CustomCallback(BaseCallback):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.best_accuracy = 0.0
    
    def on_epoch_end(self, epoch, metrics):
        if metrics.accuracy and metrics.accuracy > self.best_accuracy:
            self.best_accuracy = metrics.accuracy
            print(f"New best accuracy: {self.best_accuracy:.4f}")
        
        if metrics.accuracy and metrics.accuracy >= self.threshold:
            print(f"Accuracy threshold reached: {self.threshold}")
    
    def on_training_end(self, final_metrics):
        print(f"Training completed. Best accuracy: {self.best_accuracy:.4f}")

# Add custom callback
tracker.add_callback(CustomCallback(threshold=0.90))
```

### Configuration

```python
from ml_training_tracker import TrainingConfig, TrainingTracker

config = TrainingConfig(
    experiment_name="my_advanced_experiment",
    model_name="transformer_model",
    framework="pytorch",
    log_dir="./custom_logs",
    checkpoint_dir="./custom_checkpoints",
    save_frequency=100,
    max_checkpoints=5,
    early_stopping_patience=10,
    enable_tensorboard=True,
    enable_wandb=True,
    wandb_project="my_project",
    wandb_entity="my_team"
)

tracker = TrainingTracker(config)
```

### Comparing Experiments

```python
from ml_training_tracker import plot_training_comparison

# Run multiple experiments
tracker1 = run_experiment_with_config(config1)
tracker2 = run_experiment_with_config(config2)

# Compare results
plot_training_comparison(tracker1, tracker2, save_path="comparison.png")
```

## API Reference

### TrainingTracker Class

Main class for tracking training progress.

#### Methods

- `start_training()`: Mark the start of training
- `log_metrics(metrics: TrainingMetrics)`: Log training metrics
- `on_epoch_end(epoch: int, metrics: TrainingMetrics)`: Called at epoch end
- `end_training()`: Mark the end of training
- `save_training_report(filepath: str = None)`: Save comprehensive report
- `get_training_summary()`: Get training summary statistics

### TrainingMetrics Class

Data class for storing training metrics.

#### Attributes

- `epoch`: Current epoch number
- `step`: Current step number
- `loss`: Training loss value
- `accuracy`: Training accuracy (optional)
- `val_loss`: Validation loss (optional)
- `val_accuracy`: Validation accuracy (optional)
- `learning_rate`: Current learning rate (optional)
- `timestamp`: Timestamp of the metric
- `additional_metrics`: Dictionary of custom metrics

### TrainingConfig Class

Configuration class for training tracker.

#### Attributes

- `experiment_name`: Name of the experiment
- `model_name`: Name of the model
- `framework`: ML framework ('pytorch', 'tensorflow')
- `log_dir`: Directory for logs
- `checkpoint_dir`: Directory for checkpoints
- `save_frequency`: Checkpoint save frequency
- `max_checkpoints`: Maximum number of checkpoints to keep
- `early_stopping_patience`: Early stopping patience
- `enable_tensorboard`: Enable TensorBoard logging
- `enable_wandb`: Enable Weights & Biases
- `wandb_project`: W&B project name
- `wandb_entity`: W&B entity name

## Examples

Run the example script to see various usage patterns:

```bash
python ml_tracker_examples.py
```

This will show you:
1. PyTorch training with the tracker
2. TensorFlow/Keras training with the tracker
3. Custom metrics and advanced features
4. Multi-experiment comparison

## Output Files

The tracker generates several types of output files:

### Logs Directory
- `experiment_name_timestamp.log`: Detailed training log
- `experiment_name_config.json`: Configuration used
- `tensorboard/`: TensorBoard logs
- `plots/`: Generated plots and visualizations

### Checkpoints Directory
- `checkpoint_epoch_X_timestamp.json`: Model checkpoints

### Reports
- `experiment_name_report_timestamp.json`: Comprehensive training report

## Visualization

The tracker automatically generates several types of plots:

1. **Training Curves**: Loss and accuracy over time
2. **Validation Curves**: Validation metrics comparison
3. **Learning Rate Schedule**: Learning rate changes
4. **Metric Correlations**: Relationships between metrics
5. **Loss Distribution**: Distribution of loss values
6. **Custom Plots**: User-defined visualizations

## Best Practices

### 1. Experiment Naming
- Use descriptive names for experiments
- Include model type, dataset, and key parameters
- Example: `resnet50_cifar10_adam_lr0.001`

### 2. Metric Logging
- Log metrics at appropriate intervals
- Include both training and validation metrics
- Add custom metrics relevant to your task

### 3. Checkpoint Management
- Set appropriate save frequency
- Configure maximum checkpoints to manage disk space
- Use meaningful checkpoint names

### 4. Configuration Management
- Keep configuration in version control
- Document all parameters
- Use consistent naming conventions

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **TensorBoard Not Working**: Install tensorboard package
3. **W&B Integration**: Check API key and project settings
4. **Memory Issues**: Reduce metric history size or checkpoint frequency

### Performance Tips

1. **Reduce Logging Frequency**: Log metrics every N steps instead of every step
2. **Limit History Size**: Configure `metric_history_size` appropriately
3. **Manage Checkpoints**: Set reasonable `max_checkpoints`
4. **Disable Unused Features**: Turn off TensorBoard/W&B if not needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Changelog

### Version 1.0.0
- Initial release
- PyTorch and TensorFlow support
- TensorBoard and W&B integration
- Comprehensive visualization
- Advanced analytics and reporting