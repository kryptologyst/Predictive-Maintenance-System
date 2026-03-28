"""Data generation and processing for predictive maintenance system."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SensorDataGenerator:
    """Generate realistic sensor data for predictive maintenance simulation."""
    
    def __init__(self, seed: int = 42) -> None:
        """Initialize the data generator with random seed.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        
    def generate_equipment_data(
        self,
        n_equipment: int = 100,
        n_days: int = 365,
        failure_rate: float = 0.15,
        sensor_noise: float = 0.1,
    ) -> pd.DataFrame:
        """Generate comprehensive sensor data for multiple equipment.
        
        Args:
            n_equipment: Number of equipment units to simulate
            n_days: Number of days to simulate
            failure_rate: Probability of equipment failure
            sensor_noise: Noise level for sensor readings
            
        Returns:
            DataFrame with sensor readings and failure labels
        """
        logger.info(f"Generating data for {n_equipment} equipment over {n_days} days")
        
        data = []
        start_date = datetime.now() - timedelta(days=n_days)
        
        for equipment_id in range(n_equipment):
            # Generate equipment characteristics
            equipment_type = np.random.choice(['Motor', 'Pump', 'Compressor', 'Generator'])
            age_years = np.random.uniform(0.5, 15)
            maintenance_frequency = np.random.uniform(30, 180)  # days
            
            # Determine if this equipment will fail
            will_fail = np.random.random() < failure_rate
            
            # If equipment will fail, determine failure day (distributed across time period)
            failure_day = None
            if will_fail:
                # Distribute failures more evenly across the time period
                failure_day = np.random.randint(30, n_days - 7)  # Leave room for prediction window
            
            for day in range(n_days):
                current_date = start_date + timedelta(days=day)
                
                # Base sensor readings based on equipment type and age
                base_temp = self._get_base_temperature(equipment_type, age_years)
                base_vibration = self._get_base_vibration(equipment_type, age_years)
                base_pressure = self._get_base_pressure(equipment_type, age_years)
                base_rpm = self._get_base_rpm(equipment_type)
                
                # Add degradation over time if equipment will fail
                degradation_factor = 1.0
                if will_fail and failure_day is not None:
                    # Start degradation 30 days before failure
                    if day >= failure_day - 30:
                        degradation_factor = 1 + (day - (failure_day - 30)) / 30 * 2
                
                # Add seasonal effects
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * day / 365)
                
                # Add random noise
                noise_temp = np.random.normal(0, sensor_noise)
                noise_vibration = np.random.normal(0, sensor_noise)
                noise_pressure = np.random.normal(0, sensor_noise)
                noise_rpm = np.random.normal(0, sensor_noise * 0.5)
                
                # Calculate final sensor readings
                temperature = base_temp * degradation_factor * seasonal_factor + noise_temp
                vibration = base_vibration * degradation_factor + noise_vibration
                pressure = base_pressure * degradation_factor + noise_pressure
                rpm = base_rpm * degradation_factor + noise_rpm
                
                # Determine failure status
                failure = 0
                if will_fail and day >= failure_day:
                    failure = 1
                
                # Also mark failures in the 7-day prediction window
                failure_in_7_days = 0
                if will_fail and failure_day - day <= 7 and day < failure_day:
                    failure_in_7_days = 1
                
                # Add maintenance events
                days_since_maintenance = day % maintenance_frequency
                maintenance_due = days_since_maintenance > maintenance_frequency - 7
                
                data.append({
                    'equipment_id': equipment_id,
                    'equipment_type': equipment_type,
                    'date': current_date,
                    'age_years': age_years,
                    'temperature': max(0, temperature),
                    'vibration': max(0, vibration),
                    'pressure': max(0, pressure),
                    'rpm': max(0, rpm),
                    'maintenance_due': maintenance_due,
                    'days_since_maintenance': days_since_maintenance,
                    'failure': failure,
                    'failure_in_7_days': failure_in_7_days,
                    'failure_in_30_days': 1 if will_fail and failure_day is not None and failure_day - day <= 30 and day < failure_day else 0,
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} sensor readings")
        logger.info(f"Failure rate: {df['failure'].mean():.2%}")
        
        return df
    
    def _get_base_temperature(self, equipment_type: str, age_years: float) -> float:
        """Get base temperature for equipment type and age."""
        base_temps = {
            'Motor': 45 + age_years * 2,
            'Pump': 35 + age_years * 1.5,
            'Compressor': 55 + age_years * 3,
            'Generator': 40 + age_years * 2.5,
        }
        return base_temps.get(equipment_type, 40)
    
    def _get_base_vibration(self, equipment_type: str, age_years: float) -> float:
        """Get base vibration for equipment type and age."""
        base_vibrations = {
            'Motor': 0.3 + age_years * 0.05,
            'Pump': 0.2 + age_years * 0.03,
            'Compressor': 0.4 + age_years * 0.08,
            'Generator': 0.25 + age_years * 0.04,
        }
        return base_vibrations.get(equipment_type, 0.3)
    
    def _get_base_pressure(self, equipment_type: str, age_years: float) -> float:
        """Get base pressure for equipment type and age."""
        base_pressures = {
            'Motor': 1.0 + age_years * 0.05,
            'Pump': 2.5 + age_years * 0.1,
            'Compressor': 8.0 + age_years * 0.2,
            'Generator': 1.2 + age_years * 0.08,
        }
        return base_pressures.get(equipment_type, 1.5)
    
    def _get_base_rpm(self, equipment_type: str) -> float:
        """Get base RPM for equipment type."""
        base_rpms = {
            'Motor': 1800,
            'Pump': 1500,
            'Compressor': 3000,
            'Generator': 3600,
        }
        return base_rpms.get(equipment_type, 1800)


def create_feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features for predictive maintenance.
    
    Args:
        df: Raw sensor data DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    logger.info("Creating feature engineering pipeline")
    
    df_eng = df.copy()
    
    # Time-based features
    df_eng['day_of_year'] = df_eng['date'].dt.dayofyear
    df_eng['month'] = df_eng['date'].dt.month
    df_eng['day_of_week'] = df_eng['date'].dt.dayofweek
    
    # Rolling statistics for each equipment
    sensor_cols = ['temperature', 'vibration', 'pressure', 'rpm']
    
    for col in sensor_cols:
        # Rolling means
        df_eng[f'{col}_mean_7d'] = df_eng.groupby('equipment_id')[col].rolling(7).mean().values
        df_eng[f'{col}_mean_30d'] = df_eng.groupby('equipment_id')[col].rolling(30).mean().values
        
        # Rolling standard deviations
        df_eng[f'{col}_std_7d'] = df_eng.groupby('equipment_id')[col].rolling(7).std().values
        df_eng[f'{col}_std_30d'] = df_eng.groupby('equipment_id')[col].rolling(30).std().values
        
        # Trend features
        df_eng[f'{col}_trend_7d'] = df_eng.groupby('equipment_id')[col].rolling(7).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 7 else 0
        ).values
        
        # Change from baseline
        baseline = df_eng.groupby('equipment_id')[col].transform(lambda x: x.quantile(0.1))
        df_eng[f'{col}_change_from_baseline'] = df_eng[col] - baseline
    
    # Equipment-specific features
    df_eng['temperature_pressure_ratio'] = df_eng['temperature'] / (df_eng['pressure'] + 1e-6)
    df_eng['vibration_rpm_ratio'] = df_eng['vibration'] / (df_eng['rpm'] + 1e-6)
    
    # Maintenance-related features
    df_eng['maintenance_overdue'] = df_eng['days_since_maintenance'] > df_eng.groupby('equipment_id')['days_since_maintenance'].transform('max') * 0.8
    
    logger.info(f"Created {len(df_eng.columns) - len(df.columns)} new features")
    
    return df_eng


def prepare_training_data(df: pd.DataFrame, target_col: str = 'failure_in_7_days') -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare data for model training.
    
    Args:
        df: Feature-engineered DataFrame
        target_col: Target column name
        
    Returns:
        Tuple of features DataFrame and target Series
    """
    logger.info(f"Preparing training data with target: {target_col}")
    
    # Select feature columns (exclude metadata and target columns)
    exclude_cols = [
        'equipment_id', 'date', 'failure', 'failure_in_7_days', 'failure_in_30_days',
        'equipment_type', 'age_years', 'maintenance_due', 'days_since_maintenance'
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].fillna(0)  # Fill NaN values with 0
    y = df[target_col]
    
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y
