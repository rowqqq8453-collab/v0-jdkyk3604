#!/usr/bin/env python3
"""
Test script for ML Training Tracker
Basic functionality tests to ensure the tracker works correctly.
"""

import os
import sys
import tempfile
import shutil
import numpy as np

# Add the scripts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from ml_training_tracker import (
    TrainingTracker, TrainingConfig, TrainingMetrics, MetricsStorage
)

def test_metrics_storage():
    """Test the MetricsStorage class"""
    print("Testing MetricsStorage...")
    
    storage = MetricsStorage(max_history=100)
    
    # Add some metrics
    for i in range(50):
        metric = TrainingMetrics(
            epoch=i // 10,
            step=i,
            loss=np.random.exponential(0.1),
            accuracy=min(0.95, 0.5 + i * 0.01 + np.random.normal(0, 0.02))
        )
        storage.add_metric(metric)
    
    # Test retrieval
    recent_metrics = storage.get_recent_metrics(10)
    assert len(recent_metrics) == 10, "Should retrieve 10 recent metrics"
    
    # Test DataFrame conversion
    df = storage.get_metrics_df()
    assert len(df) == 50, "DataFrame should contain all metrics"
    assert 'loss' in df.columns, "DataFrame should contain loss column"
    assert 'accuracy' in df.columns, "DataFrame should contain accuracy column"
    
    # Test best metrics
    best_metrics = storage.best_metrics
    assert 'best_loss' in best_metrics, "Should track best loss"
    assert 'best_accuracy' in best_metrics, "Should track best accuracy"
    
    print("✓ MetricsStorage test passed")

def test_training_config():
    """Test the TrainingConfig class"""
    print("Testing TrainingConfig...")
    
    config = TrainingConfig(
        experiment_name="test_experiment",
        model_name="test_model",
        framework="pytorch",
        log_dir="./test_logs",
        checkpoint_dir="./test_checkpoints",
        enable_tensorboard=False,
        enable_wandb=False
    )
    
    assert config.experiment_name == "test_experiment"
    assert config.model_name == "test_model"
    assert config.framework == "pytorch"
    assert config.log_dir == "./test_logs"
    assert config.checkpoint_dir == "./test_checkpoints"
    assert config.enable_tensorboard == False
    assert config.enable_wandb == False
    
    print("✓ TrainingConfig test passed")

def test_training_tracker():
    """Test the TrainingTracker class"""
    print("Testing TrainingTracker...")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        config = TrainingConfig(
            experiment_name="test_tracker",
            model_name="test_model",
            framework="pytorch",
            log_dir=os.path.join(temp_dir, "logs"),
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            enable_tensorboard=False,
            enable_wandb=False,
            early_stopping_patience=0  # Disable early stopping for testing
        )
        
        tracker = TrainingTracker(config)
        
        # Test training lifecycle
        tracker.start_training()
        
        # Simulate training
        for epoch in range(3):
            for step in range(10):
                metrics = TrainingMetrics(
                    epoch=epoch,
                    step=epoch * 10 + step,
                    loss=np.random.exponential(0.1),
                    accuracy=min(0.95, 0.5 + epoch * 0.1 + step * 0.01)
                )
                tracker.log_metrics(metrics)
            
            # End of epoch
            epoch_metrics = TrainingMetrics(
                epoch=epoch,
                step=(epoch + 1) * 10,
                loss=np.random.exponential(0.1),
                accuracy=min(0.95, 0.5 + epoch * 0.1)
            )
            tracker.on_epoch_end(epoch, epoch_metrics)
        
        tracker.end_training()
        
        # Test summary
        summary = tracker.get_training_summary()
        print(f"Summary: {summary}")  # Debug info
        assert summary['experiment_name'] == "test_tracker"
        assert summary['total_epochs'] == 2  # Epochs are 0-indexed, so max epoch is 2 for 3 epochs
        # The total steps count all the individual log_metrics calls
        assert summary['total_steps'] >= 30  # At least the minimum expected
        assert 'best_metrics' in summary
        
        # Test report generation
        report_path = tracker.save_training_report()
        assert os.path.exists(report_path), "Report file should be created"
        
        # Check if plots were generated
        plots_dir = os.path.join(config.log_dir, 'plots')
        if os.path.exists(plots_dir):
            plot_files = os.listdir(plots_dir)
            assert len(plot_files) > 0, "Plot files should be generated"
    
    print("✓ TrainingTracker test passed")

def test_custom_metrics():
    """Test custom metrics functionality"""
    print("Testing Custom Metrics...")
    
    config = TrainingConfig(
        experiment_name="test_custom",
        model_name="test_model",
        framework="pytorch",
        enable_tensorboard=False,
        enable_wandb=False
    )
    
    tracker = TrainingTracker(config)
    tracker.start_training()
    
    # Test metrics with custom values
    metrics = TrainingMetrics(
        epoch=0,
        step=0,
        loss=0.5,
        accuracy=0.85,
        val_loss=0.4,
        val_accuracy=0.90,
        learning_rate=0.001
    )
    
    # Add custom metrics
    metrics.additional_metrics = {
        'f1_score': 0.88,
        'precision': 0.89,
        'recall': 0.87,
        'gradient_norm': 0.05,
        'parameter_norm': 125.5
    }
    
    tracker.log_metrics(metrics)
    
    # Verify custom metrics are stored
    df = tracker.metrics_storage.get_metrics_df()
    for key in metrics.additional_metrics.keys():
        assert key in df.columns, f"Custom metric {key} should be in DataFrame"
    
    tracker.end_training()
    print("✓ Custom Metrics test passed")

def test_thread_safety():
    """Test thread safety of MetricsStorage"""
    print("Testing Thread Safety...")
    
    import threading
    import time
    
    storage = MetricsStorage(max_history=1000)
    threads = []
    errors = []
    
    def add_metrics(thread_id):
        try:
            for i in range(100):
                metric = TrainingMetrics(
                    epoch=thread_id,
                    step=i,
                    loss=np.random.exponential(0.1),
                    accuracy=np.random.random()
                )
                storage.add_metric(metric)
                time.sleep(0.001)  # Small delay to increase contention
        except Exception as e:
            errors.append(str(e))
    
    # Create multiple threads
    for i in range(5):
        thread = threading.Thread(target=add_metrics, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Check for errors
    assert len(errors) == 0, f"Thread safety errors: {errors}"
    
    # Verify all metrics were added
    df = storage.get_metrics_df()
    assert len(df) == 500, f"Expected 500 metrics, got {len(df)}"
    
    print("✓ Thread Safety test passed")

def run_all_tests():
    """Run all tests"""
    print("Running ML Training Tracker Tests")
    print("=" * 50)
    
    try:
        test_metrics_storage()
        test_training_config()
        test_training_tracker()
        test_custom_metrics()
        test_thread_safety()
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("The ML Training Tracker is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)