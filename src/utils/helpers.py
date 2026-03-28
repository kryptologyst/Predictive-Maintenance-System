"""Utility functions and helpers for the predictive maintenance system."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import yaml
from pathlib import Path
import warnings
from sklearn.model_selection import train_test_split

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    
    # Set additional seeds if packages are available
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
        
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def create_directory_structure(base_path: str) -> None:
    """Create the necessary directory structure for the project.
    
    Args:
        base_path: Base path for the project
    """
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'logs',
        'assets',
        'configs',
        'scripts',
        'notebooks',
        'tests',
        'demo'
    ]
    
    for directory in directories:
        Path(base_path, directory).mkdir(parents=True, exist_ok=True)
        
    logger.info(f"Created directory structure at {base_path}")


def prepare_time_series_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    time_col: str = 'date',
    equipment_col: str = 'equipment_id'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare time-aware train/test split for time series data.
    
    Args:
        df: DataFrame with time series data
        test_size: Proportion of data to use for testing
        time_col: Name of the time column
        equipment_col: Name of the equipment ID column
        
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info("Preparing time-aware train/test split")
    
    # Sort by equipment and time
    df_sorted = df.sort_values([equipment_col, time_col])
    
    # Calculate split point for each equipment
    equipment_splits = {}
    for equipment_id in df_sorted[equipment_col].unique():
        equipment_data = df_sorted[df_sorted[equipment_col] == equipment_id]
        split_idx = int(len(equipment_data) * (1 - test_size))
        equipment_splits[equipment_id] = split_idx
    
    # Split the data
    train_data = []
    test_data = []
    
    for equipment_id in df_sorted[equipment_col].unique():
        equipment_data = df_sorted[df_sorted[equipment_col] == equipment_id]
        split_idx = equipment_splits[equipment_id]
        
        train_data.append(equipment_data.iloc[:split_idx])
        test_data.append(equipment_data.iloc[split_idx:])
    
    train_df = pd.concat(train_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    logger.info(f"Train set: {len(train_df)} samples")
    logger.info(f"Test set: {len(test_df)} samples")
    
    return train_df, test_df


def calculate_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """Calculate and return feature importance.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to return
        
    Returns:
        DataFrame with feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    else:
        logger.warning("Model does not have feature_importances_ attribute")
        return pd.DataFrame()


def create_model_comparison_plot(
    results_df: pd.DataFrame,
    metrics: List[str],
    save_path: Optional[str] = None
) -> None:
    """Create a comparison plot for multiple models.
    
    Args:
        results_df: DataFrame with model results
        metrics: List of metrics to plot
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics[:4]):
        if metric in results_df.columns:
            sns.barplot(data=results_df, x='Model', y=metric, ax=axes[i])
            axes[i].set_title(f'{metric} Comparison')
            axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")
    
    plt.show()


def generate_synthetic_alert_data(
    n_alerts: int = 1000,
    equipment_types: List[str] = ['Motor', 'Pump', 'Compressor', 'Generator']
) -> pd.DataFrame:
    """Generate synthetic alert data for demonstration.
    
    Args:
        n_alerts: Number of alerts to generate
        equipment_types: List of equipment types
        
    Returns:
        DataFrame with synthetic alert data
    """
    np.random.seed(42)
    
    alerts = []
    alert_types = ['Temperature High', 'Vibration High', 'Pressure Low', 'RPM Anomaly']
    severity_levels = ['Low', 'Medium', 'High', 'Critical']
    
    for i in range(n_alerts):
        alert = {
            'alert_id': f'ALT_{i:06d}',
            'equipment_id': np.random.randint(0, 100),
            'equipment_type': np.random.choice(equipment_types),
            'alert_type': np.random.choice(alert_types),
            'severity': np.random.choice(severity_levels),
            'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 30)),
            'resolved': np.random.choice([True, False], p=[0.7, 0.3]),
            'maintenance_required': np.random.choice([True, False], p=[0.4, 0.6]),
        }
        alerts.append(alert)
    
    return pd.DataFrame(alerts)


def calculate_maintenance_savings(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    maintenance_cost: float = 500,
    failure_cost: float = 10000
) -> Dict[str, float]:
    """Calculate maintenance cost savings.
    
    Args:
        y_true: True failure labels
        y_pred: Predicted failure labels
        maintenance_cost: Cost of preventive maintenance
        failure_cost: Cost of equipment failure
        
    Returns:
        Dictionary with cost calculations
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Costs with predictions
    predicted_maintenance_cost = (tp + fp) * maintenance_cost
    predicted_failure_cost = fn * failure_cost
    total_predicted_cost = predicted_maintenance_cost + predicted_failure_cost
    
    # Costs without predictions (baseline)
    baseline_maintenance_cost = len(y_true) * maintenance_cost  # Maintain everything
    baseline_failure_cost = 0  # No failures if everything maintained
    total_baseline_cost = baseline_maintenance_cost + baseline_failure_cost
    
    # Savings
    cost_savings = total_baseline_cost - total_predicted_cost
    savings_percentage = (cost_savings / total_baseline_cost) * 100
    
    return {
        'predicted_maintenance_cost': predicted_maintenance_cost,
        'predicted_failure_cost': predicted_failure_cost,
        'total_predicted_cost': total_predicted_cost,
        'baseline_maintenance_cost': baseline_maintenance_cost,
        'baseline_failure_cost': baseline_failure_cost,
        'total_baseline_cost': total_baseline_cost,
        'cost_savings': cost_savings,
        'savings_percentage': savings_percentage,
    }


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data quality and return summary statistics.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with data quality metrics
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
    }
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    infinite_values = {}
    for col in numeric_cols:
        infinite_count = np.isinf(df[col]).sum()
        if infinite_count > 0:
            infinite_values[col] = infinite_count
    
    quality_report['infinite_values'] = infinite_values
    
    return quality_report


def create_equipment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics for each equipment.
    
    Args:
        df: DataFrame with equipment data
        
    Returns:
        DataFrame with equipment summary statistics
    """
    numeric_cols = ['temperature', 'vibration', 'pressure', 'rpm']
    
    summary = df.groupby('equipment_id').agg({
        'equipment_type': 'first',
        'age_years': 'first',
        'temperature': ['mean', 'std', 'min', 'max'],
        'vibration': ['mean', 'std', 'min', 'max'],
        'pressure': ['mean', 'std', 'min', 'max'],
        'rpm': ['mean', 'std', 'min', 'max'],
        'failure': 'sum',
        'failure_in_7_days': 'sum',
        'failure_in_30_days': 'sum',
    }).round(3)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    
    return summary.reset_index()
