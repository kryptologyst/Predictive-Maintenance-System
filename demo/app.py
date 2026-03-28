"""Streamlit demo application for Predictive Maintenance System."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import joblib
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.generator import SensorDataGenerator, create_feature_engineering_pipeline
from src.utils.helpers import set_random_seeds, load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This is a research and educational demonstration system.</strong></p>
    <ul>
        <li>This system is NOT intended for automated decision-making without human review</li>
        <li>All predictions should be validated by qualified maintenance professionals</li>
        <li>This system uses synthetic data for demonstration purposes</li>
        <li>Do not use this system for actual equipment maintenance decisions</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🔧 Predictive Maintenance System</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")
st.sidebar.markdown("### System Parameters")

# Load configuration
try:
    config = load_config("configs/config.yaml")
except FileNotFoundError:
    st.error("Configuration file not found. Please ensure configs/config.yaml exists.")
    st.stop()

# Sidebar controls
n_equipment = st.sidebar.slider("Number of Equipment", 10, 200, config['data']['n_equipment'])
n_days = st.sidebar.slider("Simulation Days", 30, 365, config['data']['n_days'])
failure_rate = st.sidebar.slider("Failure Rate", 0.05, 0.30, config['data']['failure_rate'])
sensor_noise = st.sidebar.slider("Sensor Noise Level", 0.01, 0.20, config['data']['sensor_noise'])

st.sidebar.markdown("### Business Parameters")
false_alarm_cost = st.sidebar.number_input("False Alarm Cost ($)", 50, 500, config['business']['false_alarm_cost'])
missed_failure_cost = st.sidebar.number_input("Missed Failure Cost ($)", 5000, 50000, config['business']['missed_failure_cost'])
maintenance_cost = st.sidebar.number_input("Maintenance Cost ($)", 200, 1000, config['business']['maintenance_cost'])

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Overview", "🤖 Model Training", "📈 Predictions", "💰 Business Impact", "🔍 Equipment Analysis"])

with tab1:
    st.header("Data Overview")
    
    if st.button("Generate New Data", key="generate_data"):
        with st.spinner("Generating sensor data..."):
            set_random_seeds(42)
            generator = SensorDataGenerator(seed=42)
            
            df = generator.generate_equipment_data(
                n_equipment=n_equipment,
                n_days=n_days,
                failure_rate=failure_rate,
                sensor_noise=sensor_noise
            )
            
            # Feature engineering
            df_engineered = create_feature_engineering_pipeline(df)
            
            # Store in session state
            st.session_state['data'] = df_engineered
            st.session_state['raw_data'] = df
            
        st.success("Data generated successfully!")
    
    if 'data' in st.session_state:
        df = st.session_state['data']
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Equipment Units", f"{df['equipment_id'].nunique():,}")
        with col3:
            st.metric("Failure Rate", f"{df['failure'].mean():.1%}")
        with col4:
            st.metric("Equipment Types", f"{df['equipment_type'].nunique()}")
        
        # Equipment type distribution
        st.subheader("Equipment Type Distribution")
        equipment_counts = df['equipment_type'].value_counts()
        fig_pie = px.pie(values=equipment_counts.values, names=equipment_counts.index, title="Equipment Types")
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Sensor readings over time
        st.subheader("Sensor Readings Over Time")
        
        # Sample equipment for visualization
        sample_equipment = st.selectbox("Select Equipment to View", df['equipment_id'].unique()[:10])
        
        equipment_data = df[df['equipment_id'] == sample_equipment].sort_values('date')
        
        fig_sensors = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperature', 'Vibration', 'Pressure', 'RPM'),
            vertical_spacing=0.1
        )
        
        fig_sensors.add_trace(
            go.Scatter(x=equipment_data['date'], y=equipment_data['temperature'], name='Temperature'),
            row=1, col=1
        )
        fig_sensors.add_trace(
            go.Scatter(x=equipment_data['date'], y=equipment_data['vibration'], name='Vibration'),
            row=1, col=2
        )
        fig_sensors.add_trace(
            go.Scatter(x=equipment_data['date'], y=equipment_data['pressure'], name='Pressure'),
            row=2, col=1
        )
        fig_sensors.add_trace(
            go.Scatter(x=equipment_data['date'], y=equipment_data['rpm'], name='RPM'),
            row=2, col=2
        )
        
        fig_sensors.update_layout(height=600, showlegend=False, title=f"Sensor Readings - Equipment {sample_equipment}")
        st.plotly_chart(fig_sensors, use_container_width=True)
        
        # Failure events
        failure_events = df[df['failure'] == 1]
        if not failure_events.empty:
            st.subheader("Failure Events")
            st.dataframe(failure_events[['equipment_id', 'equipment_type', 'date', 'temperature', 'vibration', 'pressure', 'rpm']].head(10))

with tab2:
    st.header("Model Training")
    
    if 'data' not in st.session_state:
        st.warning("Please generate data first in the Data Overview tab.")
    else:
        df = st.session_state['data']
        
        # Prepare training data
        from src.data.generator import prepare_training_data
        from src.models.predictive_models import LogisticRegressionModel, XGBoostModel, RandomForestModel
        from src.eval.metrics import PredictiveMaintenanceEvaluator
        from src.utils.helpers import prepare_time_series_split
        
        X, y = prepare_training_data(df, target_col='failure_in_7_days')
        
        # Train/test split
        train_df, test_df = prepare_time_series_split(df, test_size=0.2)
        X_train, y_train = prepare_training_data(train_df, target_col='failure_in_7_days')
        X_test, y_test = prepare_training_data(test_df, target_col='failure_in_7_days')
        
        st.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        
        # Model selection
        models_to_train = st.multiselect(
            "Select Models to Train",
            ["Logistic Regression", "XGBoost", "Random Forest"],
            default=["Logistic Regression", "XGBoost"]
        )
        
        if st.button("Train Models", key="train_models"):
            results = []
            
            with st.spinner("Training models..."):
                evaluator = PredictiveMaintenanceEvaluator(
                    false_alarm_cost=false_alarm_cost,
                    missed_failure_cost=missed_failure_cost,
                    maintenance_cost=maintenance_cost
                )
                
                for model_name in models_to_train:
                    try:
                        if model_name == "Logistic Regression":
                            model = LogisticRegressionModel(random_state=42)
                        elif model_name == "XGBoost":
                            model = XGBoostModel(random_state=42)
                        elif model_name == "Random Forest":
                            model = RandomForestModel(random_state=42)
                        
                        # Train model
                        model.fit(X_train, y_train)
                        
                        # Make predictions
                        y_pred = model.predict(X_test)
                        y_proba = model.predict_proba(X_test)
                        
                        # Evaluate
                        result = evaluator.evaluate_model(
                            y_true=y_test.values,
                            y_pred=y_pred,
                            y_proba=y_proba,
                            model_name=model_name
                        )
                        
                        results.append(result)
                        
                    except Exception as e:
                        st.error(f"Error training {model_name}: {str(e)}")
            
            if results:
                # Display results
                st.subheader("Model Performance")
                
                # Create results DataFrame
                results_data = []
                for result in results:
                    row = {
                        'Model': result['model_name'],
                        'Accuracy': result['basic_metrics']['accuracy'],
                        'Precision': result['basic_metrics']['precision'],
                        'Recall': result['basic_metrics']['recall'],
                        'F1-Score': result['basic_metrics']['f1_score'],
                        'ROC AUC': result['probability_metrics']['roc_auc'],
                        'Total Cost': result['business_metrics']['total_cost'],
                    }
                    results_data.append(row)
                
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Store results in session state
                st.session_state['model_results'] = results_df
                st.session_state['trained_models'] = {result['model_name']: model for result in results}
                
                st.success("Models trained successfully!")

with tab3:
    st.header("Predictions")
    
    if 'model_results' not in st.session_state:
        st.warning("Please train models first in the Model Training tab.")
    else:
        st.subheader("Equipment Failure Predictions")
        
        # Select equipment for prediction
        if 'data' in st.session_state:
            df = st.session_state['data']
            equipment_options = df['equipment_id'].unique()
            
            selected_equipment = st.selectbox("Select Equipment", equipment_options)
            
            # Get latest data for selected equipment
            equipment_data = df[df['equipment_id'] == selected_equipment].sort_values('date').tail(1)
            
            if not equipment_data.empty:
                # Prepare features
                from src.data.generator import prepare_training_data
                X_equipment, _ = prepare_training_data(equipment_data, target_col='failure_in_7_days')
                
                # Make predictions with all trained models
                predictions = {}
                
                for model_name, model in st.session_state['trained_models'].items():
                    try:
                        proba = model.predict_proba(X_equipment)
                        predictions[model_name] = proba[0][1]  # Probability of failure
                    except Exception as e:
                        st.error(f"Error making prediction with {model_name}: {str(e)}")
                
                if predictions:
                    # Display predictions
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Failure Probability")
                        for model_name, prob in predictions.items():
                            st.metric(f"{model_name}", f"{prob:.1%}")
                    
                    with col2:
                        st.subheader("Current Sensor Readings")
                        latest_data = equipment_data.iloc[0]
                        st.metric("Temperature", f"{latest_data['temperature']:.1f}°C")
                        st.metric("Vibration", f"{latest_data['vibration']:.3f}")
                        st.metric("Pressure", f"{latest_data['pressure']:.1f} bar")
                        st.metric("RPM", f"{latest_data['rpm']:.0f}")
                    
                    # Risk assessment
                    avg_prob = np.mean(list(predictions.values()))
                    
                    if avg_prob > 0.7:
                        st.error("🔴 HIGH RISK: Immediate maintenance recommended")
                    elif avg_prob > 0.4:
                        st.warning("🟡 MEDIUM RISK: Schedule maintenance within 7 days")
                    else:
                        st.success("🟢 LOW RISK: Equipment operating normally")
                    
                    # Prediction confidence
                    st.subheader("Prediction Confidence")
                    confidence = 1 - np.std(list(predictions.values()))
                    st.metric("Confidence", f"{confidence:.1%}")
                    
                    if confidence < 0.7:
                        st.warning("Low prediction confidence. Consider manual inspection.")

with tab4:
    st.header("Business Impact Analysis")
    
    if 'model_results' not in st.session_state:
        st.warning("Please train models first in the Model Training tab.")
    else:
        results_df = st.session_state['model_results']
        
        # Cost analysis
        st.subheader("Cost Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost comparison chart
            fig_cost = px.bar(
                results_df, 
                x='Model', 
                y='Total Cost',
                title="Total Cost by Model",
                color='Total Cost',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig_cost, use_container_width=True)
        
        with col2:
            # Performance vs Cost scatter
            fig_scatter = px.scatter(
                results_df,
                x='ROC AUC',
                y='Total Cost',
                size='F1-Score',
                hover_data=['Model', 'Precision', 'Recall'],
                title="Performance vs Cost Trade-off"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # ROI calculation
        st.subheader("Return on Investment")
        
        # Baseline scenario (no predictions)
        total_equipment = len(st.session_state['data']['equipment_id'].unique())
        baseline_cost = total_equipment * maintenance_cost
        
        # Calculate savings for each model
        savings_data = []
        for _, row in results_df.iterrows():
            savings = baseline_cost - row['Total Cost']
            roi = (savings / baseline_cost) * 100
            savings_data.append({
                'Model': row['Model'],
                'Savings': savings,
                'ROI': roi
            })
        
        savings_df = pd.DataFrame(savings_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Baseline Cost (No Predictions)", f"${baseline_cost:,.0f}")
            best_model = savings_df.loc[savings_df['Savings'].idxmax()]
            st.metric("Best Model Savings", f"${best_model['Savings']:,.0f}")
        
        with col2:
            st.metric("Best Model ROI", f"{best_model['ROI']:.1f}%")
            st.metric("Payback Period", f"{baseline_cost / best_model['Savings']:.1f} months")

with tab5:
    st.header("Equipment Analysis")
    
    if 'data' not in st.session_state:
        st.warning("Please generate data first in the Data Overview tab.")
    else:
        df = st.session_state['data']
        
        # Equipment summary
        st.subheader("Equipment Summary")
        
        equipment_summary = df.groupby('equipment_id').agg({
            'equipment_type': 'first',
            'age_years': 'first',
            'temperature': ['mean', 'std', 'max'],
            'vibration': ['mean', 'std', 'max'],
            'pressure': ['mean', 'std', 'max'],
            'rpm': ['mean', 'std', 'max'],
            'failure': 'sum',
            'failure_in_7_days': 'sum',
            'failure_in_30_days': 'sum'
        }).round(3)
        
        # Flatten column names
        equipment_summary.columns = ['_'.join(col).strip() for col in equipment_summary.columns]
        equipment_summary = equipment_summary.reset_index()
        
        # Display summary
        st.dataframe(equipment_summary.head(20), use_container_width=True)
        
        # Equipment health score
        st.subheader("Equipment Health Score")
        
        # Calculate health score based on sensor readings
        health_scores = []
        for _, row in equipment_summary.iterrows():
            # Normalize sensor readings (lower is better for most metrics)
            temp_score = max(0, 100 - (row['temperature_max'] - 40) * 2)
            vib_score = max(0, 100 - row['vibration_max'] * 100)
            pressure_score = max(0, 100 - abs(row['pressure_mean'] - 2.0) * 10)
            
            # Combine scores
            health_score = (temp_score + vib_score + pressure_score) / 3
            health_scores.append(health_score)
        
        equipment_summary['health_score'] = health_scores
        
        # Health score distribution
        fig_health = px.histogram(
            equipment_summary,
            x='health_score',
            nbins=20,
            title="Equipment Health Score Distribution"
        )
        st.plotly_chart(fig_health, use_container_width=True)
        
        # Equipment ranking
        st.subheader("Equipment Priority Ranking")
        
        # Sort by health score and failure history
        equipment_summary['priority_score'] = (
            equipment_summary['health_score'] * 0.7 + 
            (100 - equipment_summary['failure_in_30_days_sum'] * 10) * 0.3
        )
        
        priority_df = equipment_summary.sort_values('priority_score').head(10)
        
        st.dataframe(
            priority_df[['equipment_id', 'equipment_type', 'health_score', 'failure_in_30_days_sum', 'priority_score']],
            use_container_width=True
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Predictive Maintenance System - Research & Educational Demo</p>
    <p><strong>⚠️ This system is for demonstration purposes only. Do not use for actual maintenance decisions.</strong></p>
</div>
""", unsafe_allow_html=True)
