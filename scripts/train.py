"""Main training script for the Predictive Maintenance System."""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.generator import SensorDataGenerator, create_feature_engineering_pipeline, prepare_training_data
from src.models.predictive_models import (
    LogisticRegressionModel, XGBoostModel, LightGBMModel, RandomForestModel,
    IsolationForestAnomalyModel, ModelEnsemble, get_model_by_name
)
from src.eval.metrics import PredictiveMaintenanceEvaluator
from src.utils.helpers import (
    setup_logging, load_config, set_random_seeds, create_directory_structure,
    prepare_time_series_split, create_model_comparison_plot
)

logger = logging.getLogger(__name__)


def main():
    """Main training pipeline for predictive maintenance system."""
    
    # Setup
    setup_logging("INFO")
    logger.info("Starting Predictive Maintenance System Training")
    
    # Load configuration
    config = load_config("configs/config.yaml")
    set_random_seeds(config['seed'])
    
    # Create directory structure
    create_directory_structure(".")
    
    # Generate synthetic data
    logger.info("Generating synthetic sensor data")
    generator = SensorDataGenerator(seed=config['seed'])
    
    df = generator.generate_equipment_data(
        n_equipment=config['data']['n_equipment'],
        n_days=config['data']['n_days'],
        failure_rate=config['data']['failure_rate'],
        sensor_noise=config['data']['sensor_noise']
    )
    
    # Save raw data
    df.to_csv("data/raw/sensor_data.csv", index=False)
    logger.info(f"Raw data saved: {len(df)} records")
    
    # Feature engineering
    logger.info("Performing feature engineering")
    df_engineered = create_feature_engineering_pipeline(df)
    df_engineered.to_csv("data/processed/engineered_data.csv", index=False)
    
    # Prepare training data
    X, y = prepare_training_data(df_engineered, target_col='failure_in_7_days')
    
    # Time-aware train/test split
    train_df, test_df = prepare_time_series_split(df_engineered, test_size=config['evaluation']['test_size'])
    X_train, y_train = prepare_training_data(train_df, target_col='failure_in_7_days')
    X_test, y_test = prepare_training_data(test_df, target_col='failure_in_7_days')
    
    logger.info(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    logger.info(f"Failure rate in training: {y_train.mean():.2%}")
    logger.info(f"Failure rate in test: {y_test.mean():.2%}")
    
    # Initialize evaluator
    evaluator = PredictiveMaintenanceEvaluator(
        false_alarm_cost=config['business']['false_alarm_cost'],
        missed_failure_cost=config['business']['missed_failure_cost'],
        maintenance_cost=config['business']['maintenance_cost']
    )
    
    # Define models to train
    models_to_train = [
        ('logistic_regression', LogisticRegressionModel(**config['models']['logistic_regression'])),
        ('xgboost', XGBoostModel(**config['models']['xgboost'])),
        ('lightgbm', LightGBMModel(**config['models']['xgboost'])),  # Reuse xgboost config
        ('random_forest', RandomForestModel(random_state=config['seed'])),
    ]
    
    # Train and evaluate models
    results = []
    trained_models = {}
    
    for model_name, model in models_to_train:
        logger.info(f"Training {model_name}")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Evaluate model
            evaluation_result = evaluator.evaluate_model(
                y_true=y_test.values,
                y_pred=y_pred,
                y_proba=y_proba,
                model_name=model_name
            )
            
            results.append(evaluation_result)
            trained_models[model_name] = model
            
            # Save model
            model.save(f"models/{model_name}_model.pkl")
            
            logger.info(f"{model_name} training completed successfully")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            continue
    
    # Create evaluation report
    if results:
        logger.info("Creating evaluation report")
        report_df = evaluator.create_evaluation_report(results, save_path="assets/evaluation_report.csv")
        
        # Print results
        print("\n" + "="*80)
        print("PREDICTIVE MAINTENANCE MODEL EVALUATION RESULTS")
        print("="*80)
        print(report_df.to_string(index=False, float_format='%.3f'))
        
        # Create comparison plots
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC_AUC', 'Total_Cost']
        available_metrics = [m for m in metrics_to_plot if m in report_df.columns]
        
        if available_metrics:
            create_model_comparison_plot(
                report_df, 
                available_metrics, 
                save_path="assets/model_comparison.png"
            )
        
        # Find best model
        best_model_idx = report_df['ROC_AUC'].idxmax() if 'ROC_AUC' in report_df.columns else report_df['F1-Score'].idxmax()
        best_model_name = report_df.iloc[best_model_idx]['Model']
        
        print(f"\nBest performing model: {best_model_name}")
        print(f"ROC AUC: {report_df.iloc[best_model_idx]['ROC_AUC']:.3f}")
        print(f"F1-Score: {report_df.iloc[best_model_idx]['F1-Score']:.3f}")
        
        # Generate sample predictions
        if best_model_name in trained_models:
            logger.info("Generating sample predictions")
            sample_predictions = trained_models[best_model_name].predict_proba(X_test.head(10))
            
            sample_df = pd.DataFrame({
                'equipment_id': test_df['equipment_id'].head(10),
                'actual_failure': y_test.head(10),
                'predicted_probability': sample_predictions[:, 1],
                'predicted_class': (sample_predictions[:, 1] > 0.5).astype(int)
            })
            
            print("\nSample Predictions:")
            print(sample_df.to_string(index=False, float_format='%.3f'))
            
            sample_df.to_csv("assets/sample_predictions.csv", index=False)
    
    # Train anomaly detection models
    logger.info("Training anomaly detection models")
    
    # Use only normal data for anomaly detection training
    normal_data = train_df[train_df['failure_in_7_days'] == 0]
    X_normal, _ = prepare_training_data(normal_data, target_col='failure_in_7_days')
    
    anomaly_models = [
        ('isolation_forest', IsolationForestAnomalyModel(**config['models']['isolation_forest'])),
    ]
    
    for model_name, model in anomaly_models:
        try:
            logger.info(f"Training {model_name}")
            model.fit(X_normal)
            model.save(f"models/{model_name}_model.pkl")
            logger.info(f"{model_name} training completed")
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
    
    logger.info("Training pipeline completed successfully")
    print("\nTraining completed! Check the 'models/', 'assets/', and 'data/' directories for outputs.")


if __name__ == "__main__":
    main()
