"""Test suite for the Predictive Maintenance System."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.generator import SensorDataGenerator, create_feature_engineering_pipeline, prepare_training_data
from src.models.predictive_models import (
    LogisticRegressionModel, XGBoostModel, RandomForestModel,
    IsolationForestAnomalyModel, get_model_by_name
)
from src.eval.metrics import PredictiveMaintenanceEvaluator
from src.utils.helpers import set_random_seeds, validate_data_quality


class TestDataGenerator:
    """Test cases for data generation."""
    
    def test_sensor_data_generator_init(self):
        """Test SensorDataGenerator initialization."""
        generator = SensorDataGenerator(seed=42)
        assert generator.seed == 42
        
    def test_generate_equipment_data(self):
        """Test equipment data generation."""
        generator = SensorDataGenerator(seed=42)
        df = generator.generate_equipment_data(n_equipment=10, n_days=30)
        
        assert len(df) == 300  # 10 equipment * 30 days
        assert 'equipment_id' in df.columns
        assert 'temperature' in df.columns
        assert 'vibration' in df.columns
        assert 'pressure' in df.columns
        assert 'rpm' in df.columns
        assert 'failure' in df.columns
        
    def test_feature_engineering(self):
        """Test feature engineering pipeline."""
        generator = SensorDataGenerator(seed=42)
        df = generator.generate_equipment_data(n_equipment=5, n_days=30)
        
        df_engineered = create_feature_engineering_pipeline(df)
        
        # Check that new features were created
        assert len(df_engineered.columns) > len(df.columns)
        
        # Check for rolling statistics
        rolling_features = [col for col in df_engineered.columns if '_mean_7d' in col or '_std_7d' in col]
        assert len(rolling_features) > 0
        
    def test_prepare_training_data(self):
        """Test training data preparation."""
        generator = SensorDataGenerator(seed=42)
        df = generator.generate_equipment_data(n_equipment=5, n_days=30)
        df_engineered = create_feature_engineering_pipeline(df)
        
        X, y = prepare_training_data(df_engineered, target_col='failure_in_7_days')
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert X.shape[1] > 0


class TestModels:
    """Test cases for machine learning models."""
    
    def test_logistic_regression_model(self):
        """Test LogisticRegressionModel."""
        model = LogisticRegressionModel(random_state=42)
        assert model.model_name == "LogisticRegression"
        assert model.random_state == 42
        
    def test_xgboost_model(self):
        """Test XGBoostModel."""
        model = XGBoostModel(random_state=42)
        assert model.model_name == "XGBoost"
        assert model.random_state == 42
        
    def test_random_forest_model(self):
        """Test RandomForestModel."""
        model = RandomForestModel(random_state=42)
        assert model.model_name == "RandomForest"
        assert model.random_state == 42
        
    def test_isolation_forest_model(self):
        """Test IsolationForestAnomalyModel."""
        model = IsolationForestAnomalyModel(random_state=42)
        assert model.model_name == "IsolationForest"
        assert model.random_state == 42
        
    def test_model_factory(self):
        """Test model factory function."""
        model = get_model_by_name('logistic_regression', random_state=42)
        assert isinstance(model, LogisticRegressionModel)
        
        with pytest.raises(ValueError):
            get_model_by_name('unknown_model')


class TestEvaluation:
    """Test cases for evaluation metrics."""
    
    def test_evaluator_init(self):
        """Test PredictiveMaintenanceEvaluator initialization."""
        evaluator = PredictiveMaintenanceEvaluator(
            false_alarm_cost=100,
            missed_failure_cost=10000,
            maintenance_cost=500
        )
        
        assert evaluator.false_alarm_cost == 100
        assert evaluator.missed_failure_cost == 10000
        assert evaluator.maintenance_cost == 500
        
    def test_basic_metrics_calculation(self):
        """Test basic metrics calculation."""
        evaluator = PredictiveMaintenanceEvaluator()
        
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0])
        
        metrics = evaluator._calculate_basic_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'specificity' in metrics
        
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
        
    def test_business_metrics_calculation(self):
        """Test business metrics calculation."""
        evaluator = PredictiveMaintenanceEvaluator(
            false_alarm_cost=100,
            missed_failure_cost=10000,
            maintenance_cost=500
        )
        
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0])
        
        metrics = evaluator._calculate_business_metrics(y_true, y_pred)
        
        assert 'total_cost' in metrics
        assert 'cost_per_prediction' in metrics
        assert 'precision_at_10' in metrics
        
        assert metrics['total_cost'] >= 0
        assert metrics['cost_per_prediction'] >= 0


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_random_seeds(self):
        """Test random seed setting."""
        set_random_seeds(42)
        # This is hard to test directly, but we can ensure it doesn't raise an error
        assert True
        
    def test_validate_data_quality(self):
        """Test data quality validation."""
        # Create test data
        df = pd.DataFrame({
            'col1': [1, 2, 3, np.nan, 5],
            'col2': [1.0, 2.0, 3.0, 4.0, 5.0],
            'col3': ['a', 'b', 'c', 'd', 'e']
        })
        
        quality_report = validate_data_quality(df)
        
        assert 'total_rows' in quality_report
        assert 'total_columns' in quality_report
        assert 'missing_values' in quality_report
        assert 'duplicate_rows' in quality_report
        
        assert quality_report['total_rows'] == 5
        assert quality_report['total_columns'] == 3
        assert quality_report['missing_values']['col1'] == 1


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training pipeline."""
        # Generate data
        generator = SensorDataGenerator(seed=42)
        df = generator.generate_equipment_data(n_equipment=10, n_days=30)
        
        # Feature engineering
        df_engineered = create_feature_engineering_pipeline(df)
        
        # Prepare training data
        X, y = prepare_training_data(df_engineered, target_col='failure_in_7_days')
        
        # Train model
        model = LogisticRegressionModel(random_state=42)
        model.fit(X, y)
        
        # Make predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        # Evaluate
        evaluator = PredictiveMaintenanceEvaluator()
        results = evaluator.evaluate_model(
            y_true=y.values,
            y_pred=y_pred,
            y_proba=y_proba,
            model_name="TestModel"
        )
        
        assert 'basic_metrics' in results
        assert 'business_metrics' in results
        assert 'probability_metrics' in results
        
    def test_model_save_load(self):
        """Test model saving and loading."""
        # Generate small dataset
        generator = SensorDataGenerator(seed=42)
        df = generator.generate_equipment_data(n_equipment=5, n_days=10)
        df_engineered = create_feature_engineering_pipeline(df)
        X, y = prepare_training_data(df_engineered, target_col='failure_in_7_days')
        
        # Train model
        model = LogisticRegressionModel(random_state=42)
        model.fit(X, y)
        
        # Save model
        model.save("test_model.pkl")
        
        # Load model
        loaded_model = LogisticRegressionModel(random_state=42)
        loaded_model.load("test_model.pkl")
        
        # Test predictions are the same
        original_pred = model.predict(X)
        loaded_pred = loaded_model.predict(X)
        
        np.testing.assert_array_equal(original_pred, loaded_pred)
        
        # Clean up
        Path("test_model.pkl").unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])
