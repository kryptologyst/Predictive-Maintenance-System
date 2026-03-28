# Predictive Maintenance System

A comprehensive predictive maintenance system for equipment failure prediction using machine learning. This system anticipates equipment failures before they occur using historical sensor data, helping businesses reduce downtime and maintenance costs.

## ⚠️ IMPORTANT DISCLAIMER

**This is a research and educational demonstration system.**

- This system is **NOT intended for automated decision-making** without human review
- All predictions should be validated by qualified maintenance professionals
- This system uses synthetic data for demonstration purposes
- **Do not use this system for actual equipment maintenance decisions**

## Features

### Core Capabilities
- **Multi-Model Approach**: Logistic Regression, XGBoost, LightGBM, Random Forest, and Anomaly Detection
- **Time Series Aware**: Proper handling of temporal dependencies and equipment-specific patterns
- **Feature Engineering**: Advanced feature creation including rolling statistics, trends, and equipment-specific ratios
- **Business Metrics**: Cost-sensitive evaluation with false alarm and missed failure costs
- **Anomaly Detection**: Isolation Forest and One-Class SVM for unsupervised failure detection

### Advanced Features
- **Ensemble Methods**: Model combination for improved predictions
- **SHAP Explanations**: Model interpretability and feature importance
- **Uncertainty Quantification**: Prediction confidence intervals
- **Interactive Demo**: Streamlit-based web application
- **Comprehensive Evaluation**: Precision@K, ROC-AUC, business cost analysis

## Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda package manager

### Setup
```bash
# Clone the repository
git clone https://github.com/kryptologyst/Predictive-Maintenance-System.git
cd Predictive-Maintenance-System

# Install dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Verify Installation
```bash
python -c "import src; print('Installation successful')"
```

## Quick Start

### 1. Generate Data and Train Models
```bash
python scripts/train.py
```

This will:
- Generate synthetic sensor data for 100 equipment units over 365 days
- Perform feature engineering
- Train multiple models (Logistic Regression, XGBoost, Random Forest)
- Evaluate models with business metrics
- Save trained models and results

### 2. Launch Interactive Demo
```bash
streamlit run demo/app.py
```

The demo provides:
- Data visualization and exploration
- Interactive model training
- Real-time predictions
- Business impact analysis
- Equipment health monitoring

### 3. View Results
Check the following directories for outputs:
- `models/`: Trained model files
- `assets/`: Evaluation reports and visualizations
- `data/`: Generated and processed datasets

## Project Structure

```
predictive-maintenance-system/
├── src/                          # Source code
│   ├── data/                     # Data processing
│   │   ├── __init__.py
│   │   └── generator.py          # Data generation and feature engineering
│   ├── models/                   # Machine learning models
│   │   ├── __init__.py
│   │   └── predictive_models.py  # Model implementations
│   ├── eval/                     # Evaluation metrics
│   │   ├── __init__.py
│   │   └── metrics.py           # Business and ML metrics
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       └── helpers.py           # Helper functions
├── configs/                      # Configuration files
│   └── config.yaml             # Main configuration
├── scripts/                      # Training scripts
│   └── train.py                # Main training pipeline
├── demo/                         # Demo application
│   └── app.py                  # Streamlit demo
├── tests/                        # Unit tests
├── data/                         # Data storage
│   ├── raw/                     # Raw data
│   └── processed/               # Processed data
├── models/                       # Trained models
├── assets/                       # Reports and visualizations
├── logs/                         # Log files
├── pyproject.toml               # Project configuration
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Configuration

The system is configured via `configs/config.yaml`:

```yaml
# Random seed for reproducibility
seed: 42

# Data configuration
data:
  n_equipment: 100
  n_days: 365
  failure_rate: 0.15
  sensor_noise: 0.1

# Model configuration
models:
  logistic_regression:
    random_state: 42
    max_iter: 1000
  xgboost:
    random_state: 42
    max_depth: 6
    n_estimators: 100
    learning_rate: 0.1

# Business metrics
business:
  false_alarm_cost: 100
  missed_failure_cost: 10000
  maintenance_cost: 500
```

## Data Schema

### Sensor Data
The system generates synthetic sensor data with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| equipment_id | int | Unique equipment identifier |
| equipment_type | str | Type of equipment (Motor, Pump, Compressor, Generator) |
| date | datetime | Timestamp of sensor reading |
| age_years | float | Equipment age in years |
| temperature | float | Temperature sensor reading (°C) |
| vibration | float | Vibration sensor reading |
| pressure | float | Pressure sensor reading (bar) |
| rpm | float | RPM sensor reading |
| failure | int | Binary failure indicator (0/1) |
| failure_in_7_days | int | Failure prediction target (7-day horizon) |
| failure_in_30_days | int | Failure prediction target (30-day horizon) |

### Generated Features
The system automatically creates additional features:
- Rolling statistics (7-day and 30-day means, std, trends)
- Equipment-specific ratios (temperature/pressure, vibration/RPM)
- Time-based features (day of year, month, day of week)
- Maintenance-related features (days since maintenance, overdue flags)

## Models

### Supervised Learning Models
1. **Logistic Regression**: Baseline model with good interpretability
2. **XGBoost**: Gradient boosting with high performance
3. **LightGBM**: Fast gradient boosting alternative
4. **Random Forest**: Ensemble method with feature importance

### Unsupervised Learning Models
1. **Isolation Forest**: Anomaly detection for failure patterns
2. **One-Class SVM**: Support vector-based anomaly detection

### Model Selection
Models are evaluated using:
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Business Metrics**: Total cost, cost per prediction, precision@K
- **Cost-Sensitive Evaluation**: False alarm vs missed failure costs

## Evaluation Metrics

### Machine Learning Metrics
- **Accuracy**: Overall prediction correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Average Precision**: Area under the precision-recall curve

### Business Metrics
- **Precision@K**: Precision for top-K predictions (K=10, 50, 100)
- **Total Cost**: Sum of false alarm, missed failure, and maintenance costs
- **Cost per Prediction**: Average cost per prediction
- **ROI**: Return on investment compared to baseline scenario

### Cost Structure
- **False Alarm Cost**: Cost of unnecessary maintenance ($100 default)
- **Missed Failure Cost**: Cost of equipment failure ($10,000 default)
- **Maintenance Cost**: Cost of preventive maintenance ($500 default)

## Usage Examples

### Basic Training
```python
from src.data.generator import SensorDataGenerator, create_feature_engineering_pipeline
from src.models.predictive_models import XGBoostModel
from src.eval.metrics import PredictiveMaintenanceEvaluator

# Generate data
generator = SensorDataGenerator(seed=42)
df = generator.generate_equipment_data(n_equipment=100, n_days=365)

# Feature engineering
df_engineered = create_feature_engineering_pipeline(df)

# Prepare training data
X, y = prepare_training_data(df_engineered, target_col='failure_in_7_days')

# Train model
model = XGBoostModel(random_state=42)
model.fit(X, y)

# Evaluate
evaluator = PredictiveMaintenanceEvaluator()
results = evaluator.evaluate_model(y_true=y_test, y_pred=y_pred, y_proba=y_proba)
```

### Model Prediction
```python
# Make predictions
y_pred = model.predict(X_new)
y_proba = model.predict_proba(X_new)

# Get failure probability
failure_probability = y_proba[:, 1]
```

### Anomaly Detection
```python
from src.models.predictive_models import IsolationForestAnomalyModel

# Train on normal data only
normal_data = df[df['failure'] == 0]
X_normal, _ = prepare_training_data(normal_data)

# Train anomaly detector
anomaly_model = IsolationForestAnomalyModel(contamination=0.1)
anomaly_model.fit(X_normal)

# Detect anomalies
anomaly_scores = anomaly_model.predict_proba(X_new)
```

## Demo Application

The Streamlit demo provides an interactive interface for:

### Data Overview Tab
- Equipment distribution and sensor readings
- Time series visualization
- Failure event analysis

### Model Training Tab
- Interactive model selection and training
- Real-time performance metrics
- Model comparison

### Predictions Tab
- Equipment-specific failure predictions
- Risk assessment and recommendations
- Prediction confidence scores

### Business Impact Tab
- Cost analysis and ROI calculations
- Performance vs cost trade-offs
- Savings projections

### Equipment Analysis Tab
- Equipment health scoring
- Priority ranking for maintenance
- Comprehensive equipment summaries

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ scripts/ demo/
ruff check src/ scripts/ demo/
```

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

## Limitations and Considerations

### Data Limitations
- Uses synthetic data for demonstration
- Real-world data may have different patterns and noise characteristics
- Sensor calibration and data quality issues not modeled

### Model Limitations
- Models trained on limited synthetic data
- May not generalize to different equipment types or environments
- Requires retraining with real data for production use

### Business Considerations
- Cost parameters should be calibrated to specific business context
- Maintenance schedules and constraints not fully modeled
- Human expertise and domain knowledge essential for validation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{predictive_maintenance_system,
  title={Predictive Maintenance System},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Predictive-Maintenance-System}
}
```

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Review the documentation and examples

---

**Remember: This system is for research and educational purposes only. Always consult qualified professionals for actual maintenance decisions.**
# Predictive-Maintenance-System
