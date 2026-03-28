"""Evaluation metrics and model assessment for predictive maintenance."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, precision_recall_curve, roc_curve
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class PredictiveMaintenanceEvaluator:
    """Comprehensive evaluator for predictive maintenance models."""
    
    def __init__(
        self,
        false_alarm_cost: float = 100,
        missed_failure_cost: float = 10000,
        maintenance_cost: float = 500,
    ) -> None:
        """Initialize the evaluator with business costs.
        
        Args:
            false_alarm_cost: Cost of unnecessary maintenance
            missed_failure_cost: Cost of equipment failure
            maintenance_cost: Cost of preventive maintenance
        """
        self.false_alarm_cost = false_alarm_cost
        self.missed_failure_cost = missed_failure_cost
        self.maintenance_cost = maintenance_cost
        
    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            model_name: Name of the model for reporting
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info(f"Evaluating {model_name}")
        
        results = {
            'model_name': model_name,
            'basic_metrics': self._calculate_basic_metrics(y_true, y_pred),
            'business_metrics': self._calculate_business_metrics(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }
        
        if y_proba is not None:
            results['probability_metrics'] = self._calculate_probability_metrics(y_true, y_proba)
            
        return results
        
    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of basic metrics
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'specificity': self._calculate_specificity(y_true, y_pred),
        }
        
    def _calculate_probability_metrics(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate probability-based metrics.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of probability metrics
        """
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]  # Get probability of positive class
            
        return {
            'roc_auc': roc_auc_score(y_true, y_proba),
            'average_precision': average_precision_score(y_true, y_proba),
            'brier_score': self._calculate_brier_score(y_true, y_proba),
        }
        
    def _calculate_business_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate business-relevant metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of business metrics
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate costs
        false_alarm_cost_total = fp * self.false_alarm_cost
        missed_failure_cost_total = fn * self.missed_failure_cost
        maintenance_cost_total = (tp + fp) * self.maintenance_cost
        total_cost = false_alarm_cost_total + missed_failure_cost_total + maintenance_cost_total
        
        # Calculate precision@K metrics
        precision_at_k = self._calculate_precision_at_k(y_true, y_pred, k_values=[10, 50, 100])
        
        return {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'false_alarm_cost': false_alarm_cost_total,
            'missed_failure_cost': missed_failure_cost_total,
            'maintenance_cost': maintenance_cost_total,
            'total_cost': total_cost,
            'cost_per_prediction': total_cost / len(y_true),
            'precision_at_10': precision_at_k.get(10, 0),
            'precision_at_50': precision_at_k.get(50, 0),
            'precision_at_100': precision_at_k.get(100, 0),
        }
        
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Specificity score
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
        
    def _calculate_brier_score(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calculate Brier score for probability calibration.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Brier score
        """
        return np.mean((y_proba - y_true) ** 2)
        
    def _calculate_precision_at_k(self, y_true: np.ndarray, y_pred: np.ndarray, k_values: List[int]) -> Dict[int, float]:
        """Calculate precision at different K values.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            k_values: List of K values to calculate precision for
            
        Returns:
            Dictionary mapping K values to precision scores
        """
        precision_at_k = {}
        
        for k in k_values:
            if k <= len(y_pred):
                # Sort by prediction confidence (assuming higher values = more confident)
                sorted_indices = np.argsort(y_pred)[::-1]
                top_k_indices = sorted_indices[:k]
                top_k_true = y_true[top_k_indices]
                precision_at_k[k] = np.mean(top_k_true) if len(top_k_true) > 0 else 0
                
        return precision_at_k
        
    def cross_validate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = 'roc_auc'
    ) -> Dict[str, Any]:
        """Perform cross-validation on a model.
        
        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target vector
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {cv}-fold cross-validation")
        
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_min': cv_scores.min(),
            'cv_max': cv_scores.max(),
        }
        
    def create_evaluation_report(
        self,
        results: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Create a comprehensive evaluation report.
        
        Args:
            results: List of evaluation results from multiple models
            save_path: Optional path to save the report
            
        Returns:
            DataFrame with evaluation results
        """
        logger.info("Creating evaluation report")
        
        report_data = []
        
        for result in results:
            model_name = result['model_name']
            basic_metrics = result['basic_metrics']
            business_metrics = result['business_metrics']
            
            row = {
                'Model': model_name,
                'Accuracy': basic_metrics['accuracy'],
                'Precision': basic_metrics['precision'],
                'Recall': basic_metrics['recall'],
                'F1-Score': basic_metrics['f1_score'],
                'Specificity': basic_metrics['specificity'],
                'Total_Cost': business_metrics['total_cost'],
                'Cost_per_Prediction': business_metrics['cost_per_prediction'],
                'Precision@10': business_metrics['precision_at_10'],
                'Precision@50': business_metrics['precision_at_50'],
                'Precision@100': business_metrics['precision_at_100'],
            }
            
            if 'probability_metrics' in result:
                prob_metrics = result['probability_metrics']
                row.update({
                    'ROC_AUC': prob_metrics['roc_auc'],
                    'Average_Precision': prob_metrics['average_precision'],
                    'Brier_Score': prob_metrics['brier_score'],
                })
                
            report_data.append(row)
            
        report_df = pd.DataFrame(report_data)
        
        if save_path:
            report_df.to_csv(save_path, index=False)
            logger.info(f"Evaluation report saved to {save_path}")
            
        return report_df
        
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Optional path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Failure'],
                   yticklabels=['Normal', 'Failure'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")
            
        plt.show()
        
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> None:
        """Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Optional path to save the plot
        """
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
            
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve plot saved to {save_path}")
            
        plt.show()
        
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> None:
        """Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Optional path to save the plot
        """
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]
            
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-recall curve plot saved to {save_path}")
            
        plt.show()
