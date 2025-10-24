#!/usr/bin/env python3
"""
ML Training Tracker - Quick Demo

This script demonstrates the basic functionality of the ML Training Tracker
without requiring PyTorch or TensorFlow.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from ml_training_tracker import create_training_tracker, TrainingMetrics
import numpy as np

def basic_demo():
    """Basic demonstration of the training tracker"""
    print("=" * 60)
    print("ML Training Tracker - Basic Demo")
    print("=" * 60)
    
    # Create a training tracker
    tracker = create_training_tracker(
        experiment_name="basic_demo",
        model_name="simple_model",
        framework="custom",
        enable_tensorboard=False,
        enable_wandb=False
    )
    
    # Start training
    tracker.start_training()
    print("Training started...")
    
    # Simulate a training loop
    num_epochs = 5
    steps_per_epoch = 20
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        for step in range(steps_per_epoch):
            current_step = epoch * steps_per_epoch + step
            
            # Simulate training metrics
            # Loss decreases over time (with some noise)
            base_loss = 1.0 * (0.9 ** epoch)
            loss = base_loss + np.random.normal(0, base_loss * 0.1)
            loss = max(0.01, loss)  # Ensure positive loss
            
            # Accuracy increases over time (with some noise)
            base_accuracy = 0.5 + (epoch * 0.08) + (step * 0.001)
            accuracy = min(0.99, base_accuracy + np.random.normal(0, 0.02))
            
            # Learning rate decay
            learning_rate = 0.001 * (0.95 ** epoch)
            
            # Create metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                step=current_step,
                loss=loss,
                accuracy=accuracy,
                learning_rate=learning_rate
            )
            
            # Log metrics (only every 5 steps to reduce output)
            if step % 5 == 0:
                tracker.log_metrics(metrics)
        
        # End of epoch processing
        epoch_metrics = TrainingMetrics(
            epoch=epoch,
            step=(epoch + 1) * steps_per_epoch,
            loss=loss,
            accuracy=accuracy,
            learning_rate=learning_rate
        )
        
        tracker.on_epoch_end(epoch, epoch_metrics)
        
        print(f"Epoch {epoch + 1} completed - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # End training
    tracker.end_training()
    print("\nTraining completed!")
    
    # Get training summary
    summary = tracker.get_training_summary()
    print(f"\nTraining Summary:")
    print(f"  Experiment: {summary['experiment_name']}")
    print(f"  Total Epochs: {summary['total_epochs']}")
    print(f"  Total Steps: {summary['total_steps']}")
    print(f"  Best Loss: {summary['best_metrics'].get('best_loss', 'N/A'):.4f}")
    print(f"  Best Accuracy: {summary['best_metrics'].get('best_accuracy', 'N/A'):.4f}")
    if summary.get('training_duration'):
        print(f"  Training Duration: {summary['training_duration']}")
    
    # Save report
    report_path = tracker.save_training_report()
    print(f"\nDetailed report saved to: {report_path}")
    
    # Check if plots were generated
    import os
    plots_dir = os.path.join(tracker.config.log_dir, 'plots')
    if os.path.exists(plots_dir):
        plot_files = os.listdir(plots_dir)
        if plot_files:
            print(f"Training plots saved to: {plots_dir}")
            for file in plot_files:
                print(f"  - {file}")
    
    return tracker

def advanced_demo():
    """Advanced demonstration with custom metrics"""
    print("\n" + "=" * 60)
    print("ML Training Tracker - Advanced Demo")
    print("=" * 60)
    
    # Create a training tracker with custom configuration
    from ml_training_tracker import TrainingConfig
    
    config = TrainingConfig(
        experiment_name="advanced_demo",
        model_name="advanced_model",
        framework="custom",
        log_dir="./demo_logs",
        checkpoint_dir="./demo_checkpoints",
        save_frequency=50,
        max_checkpoints=3,
        early_stopping_patience=0,  # Disable for demo
        enable_tensorboard=False,
        enable_wandb=False
    )
    
    from ml_training_tracker import TrainingTracker
    tracker = TrainingTracker(config)
    
    # Start training
    tracker.start_training()
    print("Advanced training started...")
    
    # Simulate training with custom metrics
    num_epochs = 3
    steps_per_epoch = 30
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        for step in range(steps_per_epoch):
            current_step = epoch * steps_per_epoch + step
            
            # Simulate various metrics
            loss = 0.5 * (0.85 ** epoch) + np.random.exponential(0.05)
            accuracy = min(0.98, 0.4 + epoch * 0.15 + step * 0.002)
            
            # Custom metrics
            f1_score = accuracy * 0.95 + np.random.normal(0, 0.01)
            precision = accuracy * 0.97 + np.random.normal(0, 0.01)
            recall = accuracy * 0.93 + np.random.normal(0, 0.01)
            gradient_norm = np.random.exponential(0.1)
            parameter_norm = 100 + epoch * 2 + np.random.normal(0, 1)
            
            # Create metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                step=current_step,
                loss=loss,
                accuracy=accuracy,
                learning_rate=0.001 * (0.9 ** epoch)
            )
            
            # Add custom metrics
            metrics.additional_metrics = {
                'f1_score': f1_score,
                'precision': precision,
                'recall': recall,
                'gradient_norm': gradient_norm,
                'parameter_norm': parameter_norm
            }
            
            # Log metrics (only every 10 steps)
            if step % 10 == 0:
                tracker.log_metrics(metrics)
        
        # End of epoch
        epoch_metrics = TrainingMetrics(
            epoch=epoch,
            step=(epoch + 1) * steps_per_epoch,
            loss=loss,
            accuracy=accuracy,
            learning_rate=0.001 * (0.9 ** epoch)
        )
        epoch_metrics.additional_metrics = {
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall,
            'gradient_norm': gradient_norm,
            'parameter_norm': parameter_norm
        }
        
        tracker.on_epoch_end(epoch, epoch_metrics)
        
        print(f"Epoch {epoch + 1} completed - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1_score:.4f}")
    
    # End training
    tracker.end_training()
    print("\nAdvanced training completed!")
    
    # Get training summary
    summary = tracker.get_training_summary()
    print(f"\nAdvanced Training Summary:")
    print(f"  Experiment: {summary['experiment_name']}")
    print(f"  Total Epochs: {summary['total_epochs']}")
    print(f"  Total Steps: {summary['total_steps']}")
    print(f"  Best Loss: {summary['best_metrics'].get('best_loss', 'N/A'):.4f}")
    print(f"  Best Accuracy: {summary['best_metrics'].get('best_accuracy', 'N/A'):.4f}")
    
    # Save report
    report_path = tracker.save_training_report()
    print(f"\nAdvanced report saved to: {report_path}")
    
    return tracker

def main():
    """Run the demo"""
    print("ML Training Tracker - Quick Demo")
    print("This demo shows the basic functionality of the ML Training Tracker")
    print("without requiring PyTorch or TensorFlow installation.")
    
    try:
        # Run basic demo
        basic_tracker = basic_demo()
        
        # Run advanced demo
        advanced_tracker = advanced_demo()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Install PyTorch or TensorFlow for framework-specific examples")
        print("2. Run the full examples script: python ml_tracker_examples.py")
        print("3. Check the generated reports and plots in the logs directory")
        print("4. Integrate the tracker into your own ML training pipelines")
        
    except Exception as e:
        print(f"Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())